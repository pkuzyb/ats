import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

import argparse
import librosa
import numpy as np
import os
import random
import soundfile as sf

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--expt', type=str, default=None)
parser.add_argument('-i', type=str, default=None)
parser.add_argument('-o', type=str, default=None)
parser.add_argument('--mode', type=str, default='default')
args = parser.parse_args()

if args.expt is not None or (args.i is None or args.o is None):
    plt_dir = 'downloads/%s/plots' % args.expt
    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)

    with open('data/train_%s/wav.scp' % args.expt, 'r') as inf:
        lines = inf.readlines()
    wav_scp_lines = [l.strip() for l in lines]
    with open('data/train_%s/feats.scp' % args.expt, 'r') as inf:
        lines = inf.readlines()
    feats_scp_lines = [l.strip() for l in lines]
    assert len(wav_scp_lines) == len(feats_scp_lines)
    idxs = np.arange(len(feats_scp_lines))
    np.random.seed(0)
    np.random.shuffle(idxs)
    idxs = idxs[:10]
    wav_scp_lines = [wav_scp_lines[i] for i in idxs]
    feats_scp_lines = [feats_scp_lines[i] for i in idxs]
    wav_ps = [l.split()[-1] for l in wav_scp_lines]
    actions_ps = [l.split()[-1] for l in feats_scp_lines]

    cmap = matplotlib.cm.cool
    norm = matplotlib.colors.Normalize() # vmin=5, vmax=10)
    for wav_p, actions_p in zip(wav_ps, actions_ps):
        a, sr = sf.read(wav_p)
        spec = librosa.feature.melspectrogram(a, sr=sr, n_fft=768, hop_length=512, win_length=768) # (n_mels, t)
        spec = librosa.power_to_db(spec, ref=np.max)
        art_mat = np.load(actions_p)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
        axes[0].imshow(spec, cmap=cmap, aspect="auto", interpolation='none')
        axes[1].imshow(art_mat.transpose(), cmap=cmap, aspect="auto", interpolation='none')
        axes[0].title.set_text('spectrogram')
        axes[1].title.set_text('articulatory')
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='horizontal', fraction=.1)
        f = os.path.basename(wav_p)
        plt.savefig(os.path.join(plt_dir, f.replace('.wav', '.png')))
        plt.close()
else:
    plt_dir = args.o
    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)

    fs = os.listdir(args.i)
    wav_fs = [f for f in fs if f.endswith('_gen.wav')]
    action_fs = [f for f in fs if '_sc_' in f]# if f[:-4]+'.wav' in wav_fs]
    wav_fs = sorted(wav_fs)
    action_fs = []
    for f in wav_fs:
        l_list = f[:-8].split('_')
        af = '_'.join(l_list[:-1])+'_sc_'+l_list[-1]+'.npy'
        action_fs.append(af)
    wav_ps = [os.path.join(args.i, f) for f in wav_fs]
    actions_ps = [os.path.join(args.i, f) for f in action_fs]
    '''
    if len(wav_ps) > 10:
        idxs = np.arange(len(actions_ps))
        np.random.seed(0)
        np.random.shuffle(idxs)
        idxs = idxs[:10]
        wav_ps = [wav_ps[i] for i in idxs]
        actions_ps = [actions_ps[i] for i in idxs]
    '''

    cmap = matplotlib.cm.cool
    norm = matplotlib.colors.Normalize() # vmin=5, vmax=10)
    for wav_p, actions_p in tqdm(zip(wav_ps, actions_ps), total=len(wav_ps)):
        art_mat = np.load(actions_p)
        if args.mode == 'default':
            a, sr = sf.read(wav_p)
            spec = librosa.feature.melspectrogram(a, sr=sr, n_fft=768, hop_length=512, win_length=768) # (n_mels, t)
            spec = librosa.power_to_db(spec, ref=np.max)
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
            axes[0].imshow(spec, cmap=cmap, aspect="auto", interpolation='none')
            axes[1].imshow(art_mat.transpose(), cmap=cmap, aspect="auto", interpolation='none')
            axes[0].title.set_text('spectrogram')
            axes[1].title.set_text('articulatory')
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='horizontal', fraction=.1)
        else:
            plt.imshow(art_mat.transpose())
        f = os.path.basename(wav_p)
        plt.savefig(os.path.join(plt_dir, f.replace('.wav', '.png')))
        plt.close()
