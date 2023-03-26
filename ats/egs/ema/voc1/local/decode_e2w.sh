#!/bin/bash

CUDA_VISIBLE_DEVICES=1 parallel-wavegan-decode \
    --checkpoint $2 \
    --feats-scp ema_samples/feats.scp \
    --outdir ema_samples_gen \
    --config $3 \
    --verbose 1
