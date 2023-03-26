import argparse
import numpy as np
import os

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('ind')
parser.add_argument('oud')
parser.add_argument('window', type=int)
parser.add_argument('step', type=int)
args = parser.parse_args()

if not os.path.exists(args.oud):
    os.makedirs(args.oud)

fs = os.listdir(args.ind)
fs = [f for f in fs if f.endswith('_sc.npy')]
if len(fs) == 0:
    feats_scp_p = os.path.join(args.ind, 'feats.scp')
    with open(feats_scp_p, 'r') as inf:
        lines = inf.readlines()
    lines = [l.strip() for l in lines]
    l_lists = [l.split() for l in lines]
    fids = [l_list[0] for l_list in l_lists]
    ps = [l_list[1] for l_list in l_lists]
else:
    fids = [f[:-4] for f in fs]
    ps = [os.path.join(args.ind, f) for f in fs]

seg_fids = []
seg_ps = []
for fid, p in tqdm(zip(fids, ps), total=len(fids)):
    arr = np.load(p) # (seq_len, 30)
    # print(len(arr))
    sis = [si for si in range(0, len(arr), args.step)]
    for i, si in enumerate(sis):
        carr = arr[si:si+args.window]
        cf = fid + ('_%d.npy' % i)
        cp = os.path.join(args.oud, cf)
        np.save(cp, carr)
        seg_fids.append(cf[:-4])
        seg_ps.append(cp)
new_feats_scp_p = os.path.join(args.oud, 'feats.scp')
with open(new_feats_scp_p, 'w+') as ouf:
    for fid, p in zip(seg_fids, seg_ps):
        ouf.write('%s %s\n' % (fid, p))
