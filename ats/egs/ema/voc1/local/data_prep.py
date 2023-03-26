import argparse
import logging
import os

import librosa
import numpy as np
import scipy.io.wavfile
import soundfile as sf
import yaml

from tqdm import tqdm


def mk_wavs_and_actions(expt):
    expt_dir = os.path.join('/home/peter/peter/ssim/pt/src/outputs', expt)
    preds_path = os.path.join(expt_dir, 'preds.npy')
    preds = np.load(preds_path)
    downloads_subdir = os.path.join('downloads', expt)
    if not os.path.exists(downloads_subdir):
        os.makedirs(downloads_subdir)
    wav_dir = os.path.join(downloads_subdir, 'wav')
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)
    pred_path = '/home/peter/peter/ssim/pt/src/pt_pred.wav'
    pred, sr_p = sf.read(pred_path)
    wav_paths = []
    for i, pred in tqdm(enumerate(preds), total=len(preds)):
        fid = '%d' % i
        output_path = os.path.join(wav_dir, '%s.wav' % fid)
        new_p = output_path.replace('.wav', '_16.wav')
        if not os.path.exists(new_p):
            scipy.io.wavfile.write(output_path, sr_p, pred)
            os.system('sox %s -r %d -c 1 -b 16 %s'  % (output_path, sr_p, new_p))
            os.remove(output_path)
        wav_paths.append(new_p)
    actions_dir = os.path.join(downloads_subdir, 'actions')
    if not os.path.exists(actions_dir):
        os.makedirs(actions_dir)
    actions_path = os.path.join(expt_dir, 'action_mat.npy')
    actions = np.load(actions_path)
    action_paths = []
    for i, action in enumerate(actions):
        fid = '%d' % i
        output_path = os.path.join(actions_dir, '%s.npy' % fid)
        if not os.path.exists(output_path):
            np.save(output_path, action)  # (seq_len, num_feats) = (5, 6)
        action_paths.append(output_path)
    assert len(preds) == len(actions)
    return wav_paths, action_paths

expts = ['base_l24_0', 'base_l24_1', 'base_l24_2', 'base_l24_3']
suffix = '_base_l24'
fids = []
wav_paths = []
action_paths = []
for expt in expts:
    cwav_paths, caction_paths = mk_wavs_and_actions(expt)
    fids += ['all_%s-%d' % (expt, i) for i in range(len(cwav_paths))]
    wav_paths += cwav_paths
    action_paths += caction_paths

if not os.path.exists('data/train%s' % suffix):
    os.makedirs('data/train%s' % suffix)
if not os.path.exists('data/dev%s' % suffix):
    os.makedirs('data/dev%s' % suffix)
if not os.path.exists('data/eval%s' % suffix):
    os.makedirs('data/eval%s' % suffix)

idxs = np.arange(len(fids))
np.random.seed(0)
np.random.shuffle(idxs)
num_dev = 1000
num_eval = 1000
num_train = len(idxs)-num_dev-num_eval
train_idxs = idxs[:num_train]
dev_idxs = idxs[num_train:num_train+num_dev]
eval_idxs = idxs[num_train+num_dev:]

# train_frac = 0.9
# dev_frac = 0.05
# num_train = int(len(fids)*train_frac)
# num_dev = int(len(fids)*dev_frac)

train_fids = [fids[i] for i in train_idxs]
train_wav_paths = [wav_paths[i] for i in train_idxs]
train_feat_paths = [action_paths[i] for i in train_idxs]
dev_fids = [fids[i] for i in dev_idxs]
dev_wav_paths = [wav_paths[i] for i in dev_idxs]
dev_feat_paths = [action_paths[i] for i in dev_idxs]
eval_fids = [fids[i] for i in eval_idxs]
eval_wav_paths = [wav_paths[i] for i in eval_idxs]
eval_feat_paths = [action_paths[i] for i in eval_idxs]

with open('data/train%s/wav.scp' % suffix, 'w+') as ouf:
    for fid, p in zip(train_fids, train_wav_paths):
        ouf.write('%s %s\n' % (fid, p))
with open('data/dev%s/wav.scp' % suffix, 'w+') as ouf:
    for fid, p in zip(dev_fids, dev_wav_paths):
        ouf.write('%s %s\n' % (fid, p))
with open('data/eval%s/wav.scp' % suffix, 'w+') as ouf:
    for fid, p in zip(eval_fids, eval_wav_paths):
        ouf.write('%s %s\n' % (fid, p))

with open('data/train%s/feats.scp' % suffix, 'w+') as ouf:
    for fid, p in zip(train_fids, train_feat_paths):
        ouf.write('%s %s\n' % (fid, p))
with open('data/dev%s/feats.scp' % suffix, 'w+') as ouf:
    for fid, p in zip(dev_fids, dev_feat_paths):
        ouf.write('%s %s\n' % (fid, p))
with open('data/eval%s/feats.scp' % suffix, 'w+') as ouf:
    for fid, p in zip(eval_fids, eval_feat_paths):
        ouf.write('%s %s\n' % (fid, p))
