import argparse
import numpy as np
import os

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--ids', nargs='+', required=True)
parser.add_argument('-o', required=True)
args = parser.parse_args()


for action_type in ['pitch', 'periodicity']:
    fmax = -1e6
    fmin = 1e6
    for data_id in args.ids:
        downloads_subdir = os.path.join('downloads', data_id)
        mins_maxs_path = os.path.join(downloads_subdir, 'mins_maxs.txt')
        actions_dir = os.path.join(downloads_subdir, action_type)
        action_files = os.listdir(actions_dir)
        action_files = [f for f in action_files if f.endswith('.npy')]
        action_files = sorted(action_files)
        wavs_dir = os.path.join(downloads_subdir, 'wav')
        # if not os.path.exists(mins_maxs_path):
        fs = os.listdir(actions_dir)
        fs = [f for f in fs if f.endswith('.npy')]
        fs = sorted(fs)
        for f in tqdm(fs):
            p = os.path.join(actions_dir, f)
            arr = np.load(p) # (seq_len,)
            fmax = max(fmax, arr.max())
            fmin = min(fmin, arr.min())    
    with open('downloads/pcd_min_max/'+args.o+'_%s.txt' % action_type, 'w+') as ouf:
        ouf.write('%f %f\n' % (fmin, fmax))
'''
        else:
            with open(mins_maxs_path, 'r') as inf:
                lines = inf.readlines()
            lines = [l.strip() for l in lines]
            fmins = [float(l.split()[0]) for l in lines]
            fmaxs = [float(l.split()[1]) for l in lines]
            fmins = np.array(fmins)
            fmaxs = np.array(fmaxs)
'''