import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

import argparse
import math
import numpy as np
import os
import soundfile as sf

from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm


def knapsack(W, wt, val, n):
    dp = [[0 for i in range(W + 1)] for j in range(n + 1)]
    dp = np.array(dp).astype(float)

    for i in tqdm(range(1, n + 1)):
        for w in range(1, W + 1):
            if wt[i - 1] <= w:
                dp[i][w] = max(val[i - 1] + dp[i - 1][w - wt[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][W], dp


def knapsack_with_example_solution(W: int, wt: list, val: list):
    """
    Solves the integer weights knapsack problem returns one of
    the several possible optimal subsets.
    Parameters
    ---------
    W: int, the total maximum weight for the given knapsack problem.
    wt: list, the vector of weights for all items where wt[i] is the weight
    of the i-th item.
    val: list, the vector of values for all items where val[i] is the value
    of the i-th item
    Returns
    -------
    optimal_val: float, the optimal value for the given knapsack problem
    example_optional_set: set, the indices of one of the optimal subsets
    which gave rise to the optimal value.
    Examples
    -------
    >>> knapsack_with_example_solution(10, [1, 3, 5, 2], [10, 20, 100, 22])
    (142, {2, 3, 4})
    >>> knapsack_with_example_solution(6, [4, 3, 2, 3], [3, 2, 4, 4])
    (8, {3, 4})
    >>> knapsack_with_example_solution(6, [4, 3, 2, 3], [3, 2, 4])
    Traceback (most recent call last):
        ...
    ValueError: The number of weights must be the same as the number of values.
    But got 4 weights and 3 values
    """
    if not (isinstance(wt, (list, tuple)) and isinstance(val, (list, tuple))):
        raise ValueError(
            "Both the weights and values vectors must be either lists or tuples"
        )

    num_items = len(wt)
    if num_items != len(val):
        raise ValueError(
            "The number of weights must be the "
            "same as the number of values.\nBut "
            f"got {num_items} weights and {len(val)} values"
        )
    for i in range(num_items):
        if not isinstance(wt[i], int):
            raise TypeError(
                "All weights must be integers but "
                f"got weight of type {type(wt[i])} at index {i}"
            )

    optimal_val, dp_table = knapsack(W, wt, val, num_items)
    example_optional_set: set = set()
    _construct_solution(dp_table, wt, num_items, W, example_optional_set)

    return optimal_val, example_optional_set


def _construct_solution(dp: list, wt: list, i: int, j: int, optimal_set: set):
    """
    Recursively reconstructs one of the optimal subsets given
    a filled DP table and the vector of weights
    Parameters
    ---------
    dp: list of list, the table of a solved integer weight dynamic programming problem
    wt: list or tuple, the vector of weights of the items
    i: int, the index of the  item under consideration
    j: int, the current possible maximum weight
    optimal_set: set, the optimal subset so far. This gets modified by the function.
    Returns
    -------
    None
    """
    # for the current item i at a maximum weight j to be part of an optimal subset,
    # the optimal value at (i, j) must be greater than the optimal value at (i-1, j).
    # where i - 1 means considering only the previous items at the given maximum weight
    if i > 0 and j > 0:
        if dp[i - 1][j] == dp[i][j]:
            _construct_solution(dp, wt, i - 1, j, optimal_set)
        else:
            optimal_set.add(i)
            _construct_solution(dp, wt, i - 1, j - wt[i - 1], optimal_set)


parser = argparse.ArgumentParser()
parser.add_argument('ind')
parser.add_argument('oud')
parser.add_argument('--mode', default='concat')
parser.add_argument('--minmax', default=None)
parser.add_argument('--fmin', default=None, type=float)
parser.add_argument('--fmax', default=None, type=float)
args = parser.parse_args()

if not os.path.exists(args.oud):
    os.makedirs(args.oud)

if args.minmax is not None:
    with open(args.minmax, 'r') as inf:
        lines = inf.readlines()
    lines = [l.strip() for l in lines]
    l_list = lines[0].split()
    f0_min = float(l_list[0])
    f0_max = float(l_list[1])
else:
    f0_min = args.fmin
    f0_max = args.fmax
fs = os.listdir(args.ind)
fs = [f for f in fs if f.endswith('_gen.wav')]
fids = set()
for f in fs:
    fid = '_'.join(f[:-8].split('_')[:-1])
    fids.add(fid)
fids = sorted(list(fids))
for fid in fids:
    cfs = []
    for f in fs:
        if f.startswith(fid) and len(f[len(fid)+1:].split('_')) == 2:
            cfs.append(f)
    cfs = sorted(cfs, key=lambda x: int(x[:-8].split('_')[-1]))
    cps = [os.path.join(args.ind, f) for f in cfs]
    oup = os.path.join(args.oud, fid+'.wav')
    if args.mode == 'concat':
        os.system('sox %s %s' % (' '.join(cps), oup))
    elif args.mode == 'add_mid':
        audios = []
        for p in cps:
            a, sr = sf.read(p)
            audios.append(a)
        window_len = len(audios[0])
        assert window_len % 4 == 0
        final = []
        final.append(audios[0][:int(window_len*3/4)])
        for a in audios[1:-2]:
            final.append(a[int(window_len/4):int(window_len*3/4)])
        if len(audios[-2]) == window_len: # audio ends exactly at end of this window
            final.append(audios[-2][int(window_len/4):int(window_len*3/4)])
            final.append(audios[-1][int(window_len/4):])
        elif len(audios[-2]) > int(window_len*3/4):
            final.append(audios[-2][int(window_len/4):int(window_len*3/4)])
            final.append(audios[-1][int(window_len/4):])
        else:
            final.append(audios[-2][int(window_len/4):])
        final = np.concatenate(final)
        wavfile.write(oup, sr, final)
    elif args.mode == 'wsola':
        audios = []
        sr = None
        for p in cps:
            a, sr = sf.read(p)
            audios.append(a)
        assert sr is not None
        window_len = len(audios[0])
        assert window_len % 4 == 0
        best_deltas = []
        best_nonneg_deltas = []
        best_ccs = []
        best_nonneg_ccs = []
        for i in range(len(audios)-1):
            assert len(audios[i]) >= len(audios[i+1])
        for i in range(len(audios)-1):
            audio0 = audios[i]
            audio1 = audios[i+1]
            file0 = cfs[i]
            file1 = cfs[i+1]
            flist0 = file0.split('_')
            flist1 = file1.split('_')
            npy0 = os.path.join(args.ind, '_'.join(flist0[:-2])+'_sc_'+flist0[-2]+'.npy')
            npy1 = os.path.join(args.ind, '_'.join(flist1[:-2])+'_sc_'+flist1[-2]+'.npy')
            if not os.path.exists(npy0):
                npy0 = os.path.join(args.ind, '_'.join(flist0[:-2])+'_'+flist0[-2]+'.npy')
            if not os.path.exists(npy1):
                npy1 = os.path.join(args.ind, '_'.join(flist1[:-2])+'_'+flist1[-2]+'.npy')
            sc0 = np.load(npy0)
            sc1 = np.load(npy1)
            f0_0 = sc0[-1][0]*(f0_max-f0_min)+f0_min
            f0_1 = sc0[0][0]*(f0_max-f0_min)+f0_min
            period_0 = sr/f0_0
            period_1 = sr/f0_1
            period = math.ceil(max(period_0, period_1))
            # print(period)
            best_cc = 0.0
            best_delta = 0
            best_nonneg_cc = 0.0
            best_nonneg_delta = 0
            if len(audio1) > int(window_len/2): # for last window
                for delta in range(-period, period): # NOTE was range(0, period):
                    pt0 = audio0[int(window_len/2)+delta:]
                    pt1 = audio1[:int(window_len/2)-delta]
                    if len(pt0) == len(pt1):
                        # assert len(pt0) == len(pt1)
                        cc = np.correlate(pt0, pt1)# /delta
                        cc = cc[0]
                        if cc > best_cc:
                            best_cc = cc
                            best_delta = delta
                        if cc > best_nonneg_cc and delta >= 0:
                            best_nonneg_cc = cc
                            best_nonneg_delta = delta
            best_deltas.append(best_delta)
            best_ccs.append(best_cc)
            best_nonneg_deltas.append(best_nonneg_delta)
            best_nonneg_ccs.append(best_nonneg_cc)
        
        best_deltas = best_nonneg_deltas
        '''
        cc_diffs = [best_ccs[i]-best_nonneg_ccs[i] for i in range(len(best_ccs))] # positive, smaller is better
        max_cc_diff = max(cc_diffs)
        scaled_cc_diffs = [-d+max_cc_diff for d in cc_diffs]
        delta_diffs = [best_nonneg_deltas[i]-best_deltas[i] for i in range(len(best_nonneg_deltas))]
        sol = knapsack_with_example_solution(-sum(best_deltas), delta_diffs, scaled_cc_diffs) # pick the idxs with min diffs
        val, idxs = sol
        idxs = np.array(list(idxs))-1
        changed_idxs = [i for i in idxs if best_nonneg_deltas[i] != best_deltas[i]]
        for i in changed_idxs:
            best_deltas[i] = best_nonneg_deltas[i]
        '''

        if all([d >= 0 for d in best_deltas]): # nonneg deltas
            final = []
            final.append(audios[0][:int(window_len/2)+best_deltas[0]])
            right = audios[0][int(window_len/2)+best_deltas[0]:] # right pt of audio0
            for i in range(1, len(audios)-1):
                left = audios[i][:int(window_len/2)-best_deltas[i-1]] # left pt of audio1
                hann = np.hanning(2*len(left)) # len(left) == len(right)
                right_h = right*hann[len(left):]
                left_h = left*hann[:len(left)]
                final.append(right_h+left_h)
                if best_deltas[i-1] != 0 or best_deltas[i] != 0:
                    final.append(audios[i][int(window_len/2)-best_deltas[i-1]:int(window_len/2)+best_deltas[i]])
                right = audios[i][int(window_len/2)+best_deltas[i]:]
            final.append(right)
            final = np.concatenate(final)
        else:
            final = []
            final.append(audios[0][:int(window_len/2)+best_deltas[0]])
            delta0 = best_deltas[i-1]
            for i in range(1, len(audios)-1):
                len0 = int(window_len/2)-delta0
                len1 = int(window_len/2)+best_deltas[i]
                if len0 <= len1:  # no overlap, same as nonneg deltas case
                    pt0 = audios[i-1][int(window_len/2)+delta0:]
                    pt1 = audios[i][:len0]
                    hann = np.hanning(2*len(pt0))
                    pt0_hann = pt0*hann[len(pt0):]
                    pt1_hann = pt1*hann[:len(pt0)]
                    pt01 = pt0_hann+pt1_hann
                    final.append(pt01)
                    if len0 < len1:
                        final.append(audios[i][len0:len1])
                    delta0 = best_deltas[i]
                else:  # len0 > len1
                    pt0 = audios[i-1][int(window_len/2)+delta0:int(window_len/2)+delta0+len1]
                    pt1 = audios[i][:len1]
                    hann = np.hanning(2*len(pt0))
                    pt0_hann = pt0*hann[len(pt0):]
                    pt1_hann = pt1*hann[:len(pt0)]
                    pt01 = pt0_hann+pt1_hann
                    final.append(pt01)
                    delta0 = best_deltas[i]
            final.append(audios[len(audios)-2][int(window_len/2)+best_deltas[len(audios)-2]:])
            final = np.concatenate(final)
        wavfile.write(oup, sr, final)
