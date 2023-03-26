# Copyright 2023 Sean Metzger (Copyright added by Peter Wu)

# Load packages
import numpy as np
import pandas as pd
import argparse
from os.path import join
import torchaudio
import torch
from torchaudio.models import decoder
from torchaudio.models.decoder import download_pretrained_files
print('torch version', torch.__version__)
print('torch audio version', torchaudio.__version__)
curdir = '/userdata/smetzger/repos/bravo_lj'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.utils.data import DataLoader, TensorDataset
import copy
import wandb
# import wandb # Wandb is currently used for data logging. 
# Path to where the neural and label files are stored.
data_dir = '/userdata/smetzger/b3_first_day/sean_alex_postop_b3/data'
device = 'cuda' # Set to cpu if you dont have a gpu avail. 

# Set up the experiment, can change the hyperparameters as you see fit or edit for your own models
parser = argparse.ArgumentParser()
parser.add_argument('--decimation', 
                   default=6, 
                   type=int, 
                   help='How much to downsample neural data')
parser.add_argument('--hidden_dim',
                    type=int,
                   default=256,
                   help="how many hid.units in model")
parser.add_argument('--lr', 
                    type=float,
                   default=1e-3,
                   help='learning rate')
parser.add_argument('--ks', 
                    type=int,
                   default=2,
                   help='ks of input conv')
parser.add_argument('--num_layers',
                   type=int, 
                   default=4,
                   help='number of layers')
parser.add_argument('--dropout', 
                   type=float, 
                   default=0.5, 
                   help='dropout amount')
parser.add_argument('--feat_stream', 
                   type=str, 
                   default='both',
                   help='which stream. both, hga, or raw')
parser.add_argument('--bs',
                   type=int, 
                   default=64, 
                   help='batch size')
parser.add_argument('--smooth',
                   type=int,
                   default=0)
parser.add_argument('--no_normalize', 
                   action='store_false',
                   help='If you add')
parser.add_argument('--LM_WEIGHT', 
                   help='how much the LM is weighted during beam search', 
                   type=float, 
                   default=3.23)
parser.add_argument('--WORD_SCORE', 
                   help='word insertion score for beam',
                    type=float,
                    default=-.26
                   )
parser.add_argument('--beam_width', 
                   help='beam size to use',
                   type=int,
                   default=100)
parser.add_argument('--checkpoint_dir',
                   help='where 2 load model',
                   type=str,
                   default=None)
parser.add_argument('--feedforward', 
                   help='no bidirectional',
                   action='store_true')
parser.add_argument('--pretrained',
                   help='pretrained model',
                   type=str,
                   default=None)
parser.add_argument('--train_amt',
                   help='amt of train data',
                   type=float, 
                    default=1.0)
parser.add_argument('--samples_to_trim',
                   help='num samps back to go',
                   default=0, 
                   type=int)
parser.add_argument('--ndense',
                   help='transfer dense',
                   default=40,
                   type=int)
parser.add_argument('--transfer_audio', 
                   help='true if transfer audio. then switch conv', 
                   action='store_true')

parser.add_argument('--num_50', 
                   help='how many of the 50 phrase datapoints to use, so to not over fit', 
                   type=int,
                   default=500)
parser.add_argument('--num_500', 
                   help='how much 500 phrase data to use',
                   type=int,
                   default=None)
parser.add_argument('--eval_set', 
                   help='which eval set to use', 
                   type=int,
                   default=None)

parser.add_argument('--model_type', 
                    help='which model to use',
                   type=str,
                   default='cnnrnn')

#### NEW ARGS FOR CRDNN
parser.add_argument('--KS2', 
                   help='kernel size for second conv',
                    type=int,
                   default=2)
parser.add_argument('--stride1',
                   help='stride, conv1',
                   type=int,
                   default=1)
parser.add_argument('--stride2', 
                   help='stride, conv2',
                   type=int,
                   default=2)

parser.add_argument('--jitter_amt', 
                    help='how much 2 jit',
                    type=float, 
                    default=0.5)

parser.add_argument('--chan_noise', 
                    help='how much noise', 
                    type=float,
                    default=0.04)

parser.add_argument('--blackout_prob', 
                    help='odds of blackout', 
                    type=float, 
                    default=0.05)
parser.add_argument('--weight_decay', 
                    help='wd', 
                    type=float,
                    default=None)

parser.add_argument('--goat_test_set',
                   help='use the test set that got us 55 wer with 1600 samples',
                   action='store_true')

parser.add_argument('--printall', 
                   help='print every pred',
                   action='store_true')

parser.add_argument('--word_ct_weight',
                   help='how much to weight the word count loss',
                   type=float, 
                   default=0.1)

parser.add_argument('--clipamt',
                   help='how much to clip grad',
                   type=float,
                   default=1.0)

parser.add_argument('--winstart', 
                   help='how much prior time',
                   type=float, 
                   default=0.0)
parser.add_argument('--winend',
                   help='how much time after',
                   type=float,
                   default=7.5)

parser.add_argument('--normalization_strategy', 
                    help='how to normalize the data',
                    type=str, 
                    default='typical')

args = vars(parser.parse_args())

wandb.init(project='b3_ctc_nu_phrase', 
          config=args)

#### Experiment is set up, now lets load in the neural data and corresponding labels. If you want you can probably stop here. 

# neural = np.load(join(data_dir, 'neural_phrase_ctc.npy'))

# labels = pd.read_hdf(join(data_dir, 'nu_phrase_ctc_labels_1k_pause_large.h5'))
# labels2 = pd.read_hdf(join(data_dir, '50_phrase_ctc_labels.h5'))
labels = pd.read_hdf(join(data_dir, 'nu_phrase_ctc_labels_1k.h5'))
# labels4 = pd.read_hdf(join(data))
# if not args['num_50'] is None: 
#     labels2 = labels2[-args['num_50']:]
# if not args['num_500'] is None:
#     labels = labels[-args['num_500']:]

# ##### The neural data is one large array. It will have shape N_trials, Timesteps, Channels
# ###### It is at 200 HZ, first 253 channels are HGA, second 253 are raw. There is 35s of activity per sentence
# ###### Each trial should be cropped using the length in the labels dataframe during dataloading. 
# # Downsampling neural activity is highly recommended (see below) 

# len_real = len(labels)
# len_50 = len(labels2)
# len_new = len(labels3)

# labels = pd.concat((labels, labels2, labels3))
##### Labels has several columns
# ph_label - list, each item is the phonemes from that utterance, in order <br> 
# txt_label - the ground truth text label for that utterance <br>
# length - the length of the utterance, in samples, at 200 Hz <br>
# block - which block the labels come from <br>
# word times - What time (time 0 = start of neural data) each word was presented to B3. 

labels.head()
all_labs = set(labels['txt_label'].values)

all_words = []
for a in all_labs: 
    all_words.extend(a.split(' '))
print(len(set(all_words)))

# The i-th row in the labels dataframe corresponds to the i-th neural activity

### From here, you can write your own decoders etc. Below is an example of how I process the data if it's useful. 

### NEURAL DATA PREPROCESSING. This can be a very big deal for accuracy, so I suggest tinkering with parameters.

# from scipy.signal import decimate

def normalize(x, axis=-1, order=2):
    """Normalizes a Numpy array.
    Args:
        x: Numpy array to normalize.
        axis: axis along which to normalize.
        
        order: Normalization order (e.g. `order=2` for L2 norm).
    Returns:
        A normalized copy of the array.
    """
    l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
    l2[l2 == 0] = 1
    return x / np.expand_dims(l2, axis)

# # Decimation of 3 worked well so far. Other papers have used a factor of 6. 
# # This can take a second. 
# # TODO. make this adjustable. 
# X = decimate(neural, args['decimation'], axis=1, ftype='fir')
# del neural # Clear memory. 

import os
# assert args['decimation'] == 6
if args['decimation'] ==6:
#     X = np.load(os.path.join(data_dir, f"neural_nu_phrase_ctc_dec_1k_pause_ds_{args['decimation']}_large.npy"))
#     if not args['num_500'] is None:
#         X = X[-args['num_500']:]
#     X2 = np.load(os.path.join(data_dir, 'neural_50_phrase_ctc_dec.npy'))
#     if not args['num_50'] is None: 
#         X2 = X2[-args['num_50']:]
#     X = np.concatenate((X, X2), axis=0)
#     print(X.shape)
#     X3 = np.load(os.path.join(data_dir, f"neural_nu_phrase_ctc_dec_1k_ds_{args['decimation']}.npy"))
#     X = np.concatenate((X, X3), axis=0)
    
else: 
    X = np.load(os.path.join(data_dir, f"neural_nu_phrase_ctc_dec_1k_pause_ds_{args['decimation']}_large.npy"))

from os.path import join
# from scipy.ndimage import gaussian_filter1d
# if args['smooth'] > 0: 
#     X = gaussian_filter1d(X, args['smooth'], axis=1) # Smoothing was helpful early on. 
# print(X.shape)

print(X.shape)

np.min(np.min(X, axis=0), axis=0).shape

def minmax_scaling(X): 
    chanmins = np.min(np.min(X, axis=0), axis=0)
    chanmax = np.max(np.max(X, axis=0), axis=0)
    X = X-chanmins
    X = X/(chanmax-chanmins)
    print('zero 1', chanmins, chanmax)
    return X

def pertrial_minmax(X): 
    chanmins = np.min(X, axis=1, keepdims=True)
    chanmax = np.max(X, axis=1, keepdims=True)
    X = X - chanmins
    X = X/ (chanmax -chanmins)
    return X

def rezscore(X): 
    chanmeans = np.mean(np.mean(X, axis=0), axis=0)
    chanstd = np.mean(np.std(X, axis=1), axis=0)
    print('cm, cst', chanmeans, chanstd)
    X = X - chanmeans
    X = X- chanstd
    return X

if args['normalization_strategy'] == 'typical':
    X[:, :, :X.shape[-1]//2] = normalize(X[:, :, :X.shape[-1]//2])
    X[:, :, X.shape[-1]//2:] = normalize(X[:, :, X.shape[-1]//2:])
elif args['normalization_strategy'] == 'norm_all_at_once':
    X= normalize(X)
elif args['normalization_strategy'] == 'norm_times':
    X = normalize(X, axis=1)
elif args['normalization_strategy'] == 'zero_to_one':
    X = minmax_scaling(X)
elif args['normalization_strategy'] == 'rezscore':
    X = rezscore(X)
elif args['normalization_strategy'] == 'none': 
    print('no normalization.')
elif args['normalization_strategy'] == 'pertrial_minmax':
    X = pertrial_minmax(X)
    


# select feature stream

if args['feat_stream'] == 'hga':
    X = X[:, :, :X.shape[-1]//2]
elif args['feat_stream'] == 'raw':
    X = X[:, :, X.shape[-1]//2:]
print('final X shape', X.shape)

if args['samples_to_trim']>0:
    X = X[:, :-args['samples_to_trim'], :]
print('trimmed X', X.shape)

# The following code just preprocesses the phoneme labels. 
### It should take minor changes to get this to work with letters.

from data_loading_utilities.clean_labels import clean_labels
labels, all_ph = clean_labels(labels)
phone_enc  = {v:k for k,v in enumerate(sorted([a for a in list(set(all_ph)) if not a == '|']))}

#### Setup the files needed for the torchaudio CTC decoder. You can see more on this here, including how to adapt it to letters. 
# https://pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html

files  = download_pretrained_files("librispeech-4-gram") # Download a 4-gram librispeech LM, may take a sec.

# Here we're building a lexicon. Basically we just want to say h
lex = {}
for k,v in zip(labels['txt_label'], labels['ph_label']):
    if not '|' in v:
        lex[k] = ' '.join(v) + ' |'
    else: 
        v  = '_'.join(v)
        v = v.split('|')
        for kk, vv in zip(k.split(' '), v):
            vv = vv.replace('_', ' ').strip() + ' |'
            if not kk == '':
                lex[kk] = vv
strings = []
for k, v in lex.items():
    string = k + ' ' + v
    strings.append(string)
    
strings =  [s for s in strings if len(s) > 3]
f = open(join(curdir, "for_ctc/lexicon_phrases_1k.txt"), "w")
f.writelines([s+ '\n' for s in strings])
f.close()

print('example lexicon items')
for s in strings[:5]:
    print(s)
print('vocabulary size:', len(strings))

tokens = ['-', '|'] + list(phone_enc.keys())
with open(join(curdir, 'for_ctc/tokens_phrases_1k.txt'), 'w') as f:
    f.writelines([t + '\n' for t in tokens])

# Final encoder for labels to go from tokens to labels.
enc_final = {v:k for k,v in enumerate(tokens)}

from torchaudio.models.decoder import ctc_decoder
from torchaudio.functional import edit_distance

beam_search_decoder= ctc_decoder(
    lexicon = join(curdir, 'for_ctc/lexicon_phrases_1k.txt'),
    tokens = join(curdir, 'for_ctc/tokens_phrases_1k.txt'),
    lm = join(curdir, 'custom_lms/full_corpus_lm_3_abs_slm.binary'),
    nbest=3,
    beam_size=args['beam_width'],
    lm_weight=args['LM_WEIGHT'],
    word_score=args['WORD_SCORE'],
    sil_token = '|', 
    unk_word = '<unk>',
)

# Get a greedy decoder ready as well,can be useful to see how much LM is helping.
from train.ctc_decoding import GreedyCTCDecoder
greedy_decoder = GreedyCTCDecoder(tokens)
greedy = GreedyCTCDecoder(labels=list(enc_final.keys()))

# Prepare neural and target data for CTC loss

y_final = []
for t, targ in zip(labels['txt_label'], labels['ph_label']):
    cur_y = []
    cur_y.append(enc_final['|'])
    for ph in targ:
        cur_y.append(enc_final[ph])
    cur_y.append(enc_final['|'])
    y_final.append(cur_y)

y_final_ = -1*np.ones((len(y_final), np.max([len(y) for y in y_final])))
targ_lengths =[]
for k, y in enumerate(y_final):
    y_final_[k, :len(y)] = np.array(y)
    targ_lengths.append(len(y))
targ_lengths = np.array(targ_lengths)
Y = y_final_

lens = [(l//args['decimation']) for l in labels['length']] # Adjust lengths based on decimation. 
# Finalize the lengths. 
outlens = targ_lengths
lens = np.array(lens)
lens = lens - args['samples_to_trim']
lens = [min(l, X.shape[1]) for l in lens]
lens = np.array(lens)# Some lengths may be a sample over. 

trainsets = []
inds = np.arange(len(X))
print(len(inds), 'nsamp')
test_inds_eligible= inds
if args['goat_test_set']: 
    test_inds_eligible = np.arange(1650) #np.arange(len_real+len_50, len_real+len_50+len_new) # Lets only test on the new blocks. 
test_inds_eligible = np.arange(9300)
np.random.seed(1337)
np.random.shuffle(test_inds_eligible)
final_inds = [3532, 3959, 2128, 3668, 3860, 4687, 4957, 3656]
for k in range(10): 
    te_inds = test_inds_eligible[k*(len(inds)//40): (k+1)*(len(inds)//40)]
    te_inds = [i for i in te_inds if not i in final_inds]
    tr_inds = [i for i in inds if not i in te_inds]    
    tr_inds = [i for i in tr_inds if not i in final_inds]
    trainsets.append((tr_inds, te_inds))

gt_text = labels['txt_label'].values



tokens

if not args['eval_set'] is None: 
    eval_set = args['eval_set']
    trainsets = trainsets[eval_set:eval_set+1]
    

from data_loading_utilities.torch_ctc_loaders import CTCDataset, Jitter, Blackout, AdditiveNoise, LevelChannelNoise, ScaleAugment

from torchvision import transforms

b1args = {'winstart': 0, 'winend': 7.5,  'additive_noise_level': 0.0027354917297051813, 
        'scale_augment_low': 0.9551356218945801, 'scale_augment_high': 1.0713824626558794, 
        'blackout_len': 0.30682868940865543, 'blackout_prob': 0.04787032280216536,
        'random_channel_noise_sigma': 0.028305685438945617
        }

train_jitter = Jitter((-1, 8), (args['winstart'], args['winend']), jitter_amt=args['jitter_amt'], decimation=args['decimation'])
test_jitter = Jitter((-1,8), (args['winstart'], args['winend']), jitter_amt=0.0, decimation=args['decimation'])
train_jitter.winsize, test_jitter.winsize

lens[:] = train_jitter.winsize

blackout = Blackout(b1args['blackout_len'], args['blackout_prob'])
noise = AdditiveNoise(b1args['additive_noise_level'])
chan_noise = LevelChannelNoise(args['chan_noise'])
scale = ScaleAugment(b1args['scale_augment_low'], b1args['scale_augment_high'])

composed = transforms.Compose([
    train_jitter,  blackout , noise, chan_noise, scale
])

test_augs = transforms.Compose([
    test_jitter
])

print(lens[:10])

# test_augs

from train.ctc_aux_loss_trainer import train_loop
from models.cnn_rnn_w_aux import AUXCnnRnnClassifier
from data_loading_utilities.torch_ctc_loaders import CTCDataset_Wordct

word_targets = []
for l in labels['txt_label'].values:
    word_targets.append(len(l.split(' ')))
    
    

word_targets= np.array(word_targets)

word_targets[:3]

labels.iloc[2863]

import matplotlib.pyplot as plt
plt.hist(word_targets)

np.argmin(word_targets)

np.max(word_targets)

n_wordtarg = np.max(word_targets)

n_wordtarg

for train, test in trainsets:
    # Train test split, plus load into dataset
        # Train test split, plus load into dataset
    train_amt = int(len(train)*args['train_amt'])
    print('num samples', train_amt)
    wandb.log({'num_samples':train_amt})
    train = train[:train_amt]
    print(len(train), train_amt)
    lens = np.array(list(lens))
    X_tr, X_te = X[train], X[test]
    Y_tr, Y_te = Y[train], Y[test]
    print('Xtr', X_tr.shape, 'Y_tr', Y_tr.shape)
    lens_tr, lens_te = lens[train], lens[test]
    inds_tr, inds_te = np.array(train), np.array(test) # for loading text labels.
    outlens_tr, outlens_te = outlens[train], outlens[test]
    word_targs_tr, word_targs_te = word_targets[train], word_targets[test]
    
    train_dset = CTCDataset_Wordct(X_tr, Y_tr, lens_tr, outlens_tr, inds_tr, word_targs_tr, transform=composed)
    test_dset = CTCDataset_Wordct(X_te, Y_te, lens_te, outlens_te, inds_te, word_targs_te, transform=test_augs)
    
    # TODO: Add transforms from torchaudio.transforms
    train_loader = DataLoader(train_dset, batch_size=args['bs'], shuffle=True) 
    test_loader = DataLoader(test_dset, batch_size=args['bs'], shuffle=False)
    break
    
# Initialize the model. 
device='cuda'
if not args['feedforward']:
    if not args['pretrained'] is None: 
        n_targ = args['ndense']
    else: 
        n_targ=len((enc_final))


    if args['model_type'] == 'cnnrnn':
        model = AUXCnnRnnClassifier(rnn_dim=args['hidden_dim'], KS=args['ks'], 
                                         num_layers=args['num_layers'],
                                         dropout=args['dropout'], n_targ=n_targ,
                                  bidirectional=True, in_channels=X_tr.shape[-1], nword_targ=n_wordtarg+1)

wandb.log({
    'n_targ':len((enc_final)),
    'in_channels':X_tr.shape[-1]
})

if args['weight_decay'] is None:
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
else: 
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
model = model.to(device)
model =train_loop(model, train_loader,
                test_loader, 
                  optimizer,
                device, gt_text, greedy, beam_search_decoder, tokens, start_eval=0, 
                 wandb_log=True, checkpoint_dir=args['checkpoint_dir'], printall=args['printall'],
                 wordloss_weight=args['word_ct_weight'], clipamt =args['clipamt'])

# Currently we only use one fold for model dev. 