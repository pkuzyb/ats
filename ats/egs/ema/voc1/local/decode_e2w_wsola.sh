#!/bin/bash

CUDA_VISIBLE_DEVICES=1 parallel-wavegan-decode \
    --checkpoint $2 \
    --feats-scp ema_samples/feats.scp \
    --outdir ema_samples_seg \
    --config $3 \
    --verbose 1

python3 local/combine_segs.py ema_samples_seg ema_samples_combined --mode wsola --fmin 49.183701 --fmax 141.904282
