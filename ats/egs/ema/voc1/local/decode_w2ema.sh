#!/bin/bash

# Does articulatory inversion on the .wav files in samples/ema_wavs/wav.scp
# and outputs the estimated EMA trajectories in samples/ema_samples_npy

parallel-wavegan-decode \
    --checkpoint $1 \
    --feats-scp samples/ema_wavs/wav.scp \
    --outdir samples/ema_samples_npy \
    --config $2 \
    --verbose 1
