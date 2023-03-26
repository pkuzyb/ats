import argparse
import numpy as np
import os

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('dir')
args = parser.parse_args()

sc_pattern = '_sc_'
fs = os.listdir(args.dir)
fs = [f for f in fs if sc_pattern in f]
if len(fs) == 0:
    sc_pattern = '_sc'
    fs = os.listdir(args.dir)
    fs = [f for f in fs if sc_pattern in f]
ps = []
fids = []
for f in fs:
    p = os.path.join(args.dir, f)
    ps.append(p)
    fid = f[:-4].replace('_sc_', '_').replace('_sc', '')
    fids.append(fid)
with open(os.path.join(args.dir, 'feats.scp'), 'w+') as ouf:
    for fid, p in zip(fids, ps):
        ouf.write('%s %s\n' % (fid, p))
with open(os.path.join(args.dir, 'feats2.scp'), 'w+') as ouf:
    if sc_pattern == '_sc':
        for fid, p in zip(fids, ps):
            ouf.write('%s %s\n' % (fid, p.replace('_sc.npy', '_gen.npy')))
    else:
        for fid, p in zip(fids, ps):
            ouf.write('%s %s\n' % (fid, p.replace('_sc_', '_').replace('.npy', '_gen.npy')))
