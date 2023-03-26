#!/bin/bash

parallel-wavegan-decode \
    --checkpoint $2 \
    --feats-scp ema_samples/feats.scp \
    --outdir ema_samples \
    --config $3 \
    --verbose 1

parallel-wavegan-decode \
    --checkpoint exp/k_mngu0_train_fnema_m2s_mngu/checkpoint-240000steps.pkl \
    --feats-scp ema_samples/feats2.scp \
    --outdir ema_samples \
    --config conf/scratch_80.yaml \
    --verbose 1

# python3 local/plot_arrs.py -i samples -o samples
