# AtS

- Articulatory Synthesis
- Anything-to-Speech

## Installing

```bash
git clone https://github.com/articulatory/ats.git
cd ats
pip3 install -e .
```

## Codebase

- Models are in `ats/models`.
- `egs/ema/voc1/run.sh` is used for training and testing.

## EMA-to-Speech

### Preprocess Data

```bash
cd egs/ema/voc1
mkdir downloads
# download MNGU0 dataset and move emadata/ folder to downloads/
python3 local/mk_ema_feats.py
python3 local/pitch.py downloads/emadata/cin_us_mngu0 --hop 80
python3 local/combine_feats.py downloads/emadata/cin_us_mngu0 --feats pitch actions -o fnema
```

### Train Model

```bash
./run.sh --conf conf/e2w_hifigan.yaml --stage 1 --tag e2w_hifi --train_set mngu0_train_fnema --dev_set mngu0_val_fnema --eval_set mngu0_test_fnema
```

- Stage 1 in `./run.sh` is preprocessing and thus only needs to be run once per train-dev.-eval. triple. Stage 2 is training, so subsequent training experiments with the same data can use `./run.sh --stage 2`.
- Replace `conf/e2w_hifigan.yaml` with `conf/e2w_hifigan_car.yaml` to use our autoregressive model (HiFi-GAN CAR)
- To train with batches containing entire utterances as in [Gaddy & Klein, 2021](https://arxiv.org/abs/2106.01933), add the following to the config:
```
batch_sampler_type: SizeAwareSampler
batch_sampler_params:
    max_len: 256000  # = batch_size*batch_max_steps
```

## Speech-to-EMA

```bash
# Download https://drive.google.com/drive/folders/1SeDIyZMWvAl6Aorm4PTVUnnT80Gm_GJc?usp=sharing.
# Create samples/ema_wavs/wav.scp and add the files to decode.
#   Each line is comprised of "[identifier] [path to .wav file]".
#   The resulting estimated EMA features will be in samples/ema_samples_gen/[identifier]_gen.wav
local/decode_w2ema.sh mocha_train_fnema_w2ema_hifi/checkpoint-120000steps.pkl conf/w2e_hifi_inv.yaml
```

## Creating Your Own Speech Synthesizer

```bash
cd egs
mkdir <your_id>
cp -r TEMPLATE/voc1 <your_id>
```

- To use your own model, add the model code to a new file in `ats/models` and an extra line referencing that file in `ats/models/__init__.py`. Then, change `generator_type` or `discriminator_type` in the `.yaml` config to the name of the new model class.
- To customize the loss function, similarly modify the code in `ats/losses`. Then, call the loss function in `ats/bin/train.py`. Existing loss functions can be toggled on/off and modified through the `.yaml` config, e.g., in the "STFT LOSS SETTING" and "ADVERSARIAL LOSS SETTING" sections.

## Acknowledgements

Based on https://github.com/kan-bayashi/ParallelWaveGAN.
