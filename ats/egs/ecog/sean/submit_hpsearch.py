# Copyright 2023 Sean Metzger (Copyright added by Peter Wu)

print('Sko Buffs')
import subprocess
import shlex
import pandas as pd

exp_str = ""
total_runs = 0
for strat in ['norm_times']: #['norm_all_at_once', 'norm_times', 'none']:
    for dropout in [.4]:
        for hid in [500]:
            for jitter_amt in [1]:
                for wd in [.00001]:
                    for layers in [3]:
                        for winstart in [-.5]:
                            for ks in [4]:
                                exp_name = "frq"
                                filename = f'/userdata/smetzger/repos/bravo_lj/runs/{exp_name}_{strat}.txt'
                                exp_str = f"submit_job -q  mind-gpu -g 2 -m 56 -o {filename} -n {strat} -x /userdata/smetzger/torchaud/bin/python "
                                exp_str += f"/userdata/smetzger/repos/bravo_lj/phrases_newnorms_large.py --jitter_amt {jitter_amt} --chan_noise .0 --weight_decay {wd} --LM_WEIGHT 4 --train_amt 1 --hidden_dim {hid} --ks {ks} --dropout {dropout} --num_layers {layers} --eval_set 1 --decimation 6 --word_ct_weight 0.0 --clipamt .0001 --winstart {winstart} --checkpoint_dir /userdata/smetzger/repos/bravo_lj/checkpoints"
                                exp_str += f" --normalization_strategy {strat}"

                                cmd = shlex.split(exp_str)
                                subprocess.run(cmd, stderr=subprocess.STDOUT)
                                # assert False
                                total_runs += 1
print(total_runs)