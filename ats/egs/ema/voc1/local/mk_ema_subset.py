import os
import random
import sys

in_tag = sys.argv[1]
out_tag = sys.argv[2]
num_train = int(sys.argv[3])

train_data_d = 'data/mngu0_train_%s' % out_tag
dev_data_d = 'data/mngu0_val_%s' % out_tag
eval_data_d = 'data/mngu0_test_%s' % out_tag
if not os.path.exists(train_data_d):
    os.system('cp -r data/mngu0_train_%s %s'  % (in_tag, train_data_d))
if not os.path.exists(dev_data_d):
    os.system('cp -r data/mngu0_val_%s %s' % (in_tag, dev_data_d))
if not os.path.exists(eval_data_d):
    os.system('cp -r data/mngu0_test_%s %s' % (in_tag, eval_data_d))

fids_d = {}
fid2wav = {}
fid2feat = {}
for (data_subd, phase) in [(train_data_d, 'train'), (dev_data_d, 'dev'), (eval_data_d, 'eval')]:
    wav_scp_p = os.path.join(data_subd, 'wav.scp')
    feats_scp_p = os.path.join(data_subd, 'feats.scp')
    with open(wav_scp_p, 'r') as inf:
        lines = inf.readlines()
    wav_lines = [l.strip() for l in lines]
    for l in wav_lines:
        l_list = l.split()
        fid = l_list[0]
        wav = l_list[1]
        assert fid not in fid2wav
        fid2wav[fid] = wav
    with open(feats_scp_p, 'r') as inf:
        lines = inf.readlines()
    lines = [l.strip() for l in lines]
    for l in lines:
        l_list = l.split()
        fid = l_list[0]
        feat = l_list[1]
        assert fid not in fid2feat
        fid2feat[fid] = feat
    fids = [l.split()[0] for l in lines]
    fids_d[phase] = fids

num_data = {'train': num_train, 'dev': 60, 'eval': 60} # 50k, about 18 hours

for i, (data_subd, phase) in enumerate([(train_data_d, 'train'), (dev_data_d, 'dev'), (eval_data_d, 'eval')]):
    orig_fids = fids_d[phase]
    c_num_data = num_data[phase]
    if len(orig_fids) == c_num_data:
        fids = orig_fids
    else:
        orig_fids = sorted(orig_fids)
        random.Random(i).shuffle(orig_fids)
        fids = orig_fids[:c_num_data]
    assert len(fids) == c_num_data
    wav_scp_p = os.path.join(data_subd, 'wav.scp')
    feats_scp_p = os.path.join(data_subd, 'feats.scp')
    with open(wav_scp_p, 'w+') as ouf:
        for fid in fids:
            ouf.write('%s %s\n' % (fid, fid2wav[fid]))
    with open(feats_scp_p, 'w+') as ouf:
        for fid in fids:
            ouf.write('%s %s\n' % (fid, fid2feat[fid]))
