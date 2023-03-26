#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Peter Wu
#  Apache 2.0 License (https://www.apache.org/licenses/LICENSE-2.0)

"""Training script."""

import argparse
import functools
import logging
import os
import sys

from collections import defaultdict

import matplotlib
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import ats
import ats.models
import ats.optimizers
import ats.samplers
import ats.transforms

from ats.datasets import SpeechDataset, get_task_features
from ats.datasets import combine_fixed_length, mk_ar_tensor, parse_batch
from ats.layers import PQMF
from ats.losses import DiscriminatorAdversarialLoss
from ats.losses import FeatureMatchLoss
from ats.losses import GeneratorAdversarialLoss
from ats.losses import MelSpectrogramLoss
from ats.losses import MultiResolutionSTFTLoss
# from ats.losses import InterLoss
from ats.utils import read_hdf5

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


class Trainer(object):
    """Customized trainer module for training ats models."""

    def __init__(self, steps, epochs, data_loader, sampler, model, criterion, optimizer, scheduler, config, device=torch.device("cpu"),
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (dict): Dict of criterions. It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contrain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
        self.best_mel_loss = 1.0e6

    def run(self):
        """Run training."""
        self.tqdm = tqdm(initial=self.steps, total=self.config["train_max_steps"], desc="[train]")
        while True:
            self._train_epoch()  # train one epoch
            if self.finish_train:
                break
        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        state_dict = {
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
                "discriminator": self.optimizer["discriminator"].state_dict(),
            },
            "scheduler": {
                "generator": self.scheduler["generator"].state_dict(),
                "discriminator": self.scheduler["discriminator"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if self.config["distributed"]:
            state_dict["model"] = {
                "generator": self.model["generator"].module.state_dict(),
                "discriminator": self.model["discriminator"].module.state_dict(),
            }
        else:
            state_dict["model"] = {
                "generator": self.model["generator"].state_dict(),
                "discriminator": self.model["discriminator"].state_dict(),
            }
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config["distributed"]:
            self.model["generator"].module.load_state_dict(state_dict["model"]["generator"])
            self.model["discriminator"].module.load_state_dict(state_dict["model"]["discriminator"])
        else:
            self.model["generator"].load_state_dict(state_dict["model"]["generator"])
            self.model["discriminator"].load_state_dict(state_dict["model"]["discriminator"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["generator"].load_state_dict(state_dict["optimizer"]["generator"])
            self.optimizer["discriminator"].load_state_dict(state_dict["optimizer"]["discriminator"])
            self.scheduler["generator"].load_state_dict(state_dict["scheduler"]["generator"])
            self.scheduler["discriminator"].load_state_dict(state_dict["scheduler"]["discriminator"])

    def _train_step(self, batch):
        """Train model one step."""
        x, y, _, ar, _, spk_id, ph, _, _ = parse_batch(batch, self.device)

        #######################
        #      Generator      #
        #######################
        if self.steps > self.config.get("generator_train_start_steps", 0):
            y_ = self.model["generator"](x, spk_id=spk_id, ar=ar)
            if self.config["use_ph_loss"]:
                y_, ph_ = y_

            # reconstruct the signal from multi-band signal
            if self.config["generator_params"]["out_channels"] > 1 and self.config.get("pqmf", False):
                y_mb_ = y_
                y_ = self.criterion["pqmf"].synthesis(y_mb_)

            # initialize
            gen_loss = 0.0

            # multi-resolution sfft loss
            if self.config["use_stft_loss"]:
                sc_loss, mag_loss = self.criterion["stft"](y_, y)
                gen_loss += sc_loss + mag_loss
                self.total_train_loss[
                    "train/spectral_convergence_loss"
                ] += sc_loss.item()
                self.total_train_loss[
                    "train/log_stft_magnitude_loss"
                ] += mag_loss.item()

            # subband multi-resolution stft loss
            if self.config["use_subband_stft_loss"]:
                gen_loss *= 0.5  # for balancing with subband stft loss
                y_mb = self.criterion["pqmf"].analysis(y)
                sub_sc_loss, sub_mag_loss = self.criterion["sub_stft"](y_mb_, y_mb)
                gen_loss += 0.5 * (sub_sc_loss + sub_mag_loss)
                self.total_train_loss[
                    "train/sub_spectral_convergence_loss"
                ] += sub_sc_loss.item()
                self.total_train_loss[
                    "train/sub_log_stft_magnitude_loss"
                ] += sub_mag_loss.item()

            # mel spectrogram loss
            if self.config["use_mel_loss"]:
                mel_loss = self.criterion["mel"](y_, y)
                gen_loss += mel_loss
                self.total_train_loss["train/mel_loss"] += mel_loss.item()

            # weighting aux loss
            gen_loss *= self.config.get("lambda_aux", 1.0)

            # phoneme loss
            if self.config["use_ph_loss"]:
                ph_loss = self.criterion["ph"](ph_, ph)
                gen_loss += self.config["lambda_ph"] * ph_loss
                self.total_train_loss["train/ph_loss"] += ph_loss.item()

            # adversarial loss
            if ar is not None:
                disc_y = torch.cat([ar, y], dim=2)
                disc_y_ = torch.cat([ar, y_], dim=2)
            else:
                disc_y = y
                disc_y_ = y_
            if self.steps > self.config["discriminator_train_start_steps"]:
                p_ = self.model["discriminator"](disc_y_)
                adv_loss = self.criterion["gen_adv"](p_)
                self.total_train_loss["train/adversarial_loss"] += adv_loss.item()

                # feature matching loss
                if self.config["use_feat_match_loss"]:
                    # no need to track gradients
                    with torch.no_grad():
                        p = self.model["discriminator"](disc_y)
                    fm_loss = self.criterion["feat_match"](p_, p)
                    self.total_train_loss[
                        "train/feature_matching_loss"
                    ] += fm_loss.item()
                    adv_loss += self.config["lambda_feat_match"] * fm_loss

                # add adversarial loss to generator loss
                gen_loss += self.config["lambda_adv"] * adv_loss

            self.total_train_loss["train/generator_loss"] += gen_loss.item()

            # update generator
            self.optimizer["generator"].zero_grad()
            gen_loss.backward()
            if self.config["generator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["generator"].parameters(),
                    self.config["generator_grad_norm"],
                )
            self.optimizer["generator"].step()
            if self.config["generator_scheduler_type"] == "ReduceLROnPlateau":
                self.scheduler["generator"].step(gen_loss)
            else: 
                self.scheduler["generator"].step()

        #######################
        #    Discriminator    #
        #######################
        if self.steps > self.config["discriminator_train_start_steps"]:
            # re-compute y_ which leads better quality
            with torch.no_grad():
                y_ = self.model["generator"](x, spk_id=spk_id, ar=ar, ph=ph)
                if self.config["use_ph_loss"]:
                    y_, ph_ = y_
            if self.config["generator_params"]["out_channels"] > 1 and self.config.get("pqmf", False):
                y_ = self.criterion["pqmf"].synthesis(y_)

            # discriminator loss
            if ar is not None:
                disc_y = torch.cat([ar, y], dim=2)
                disc_y_ = torch.cat([ar, y_], dim=2)
            else:
                disc_y = y
                disc_y_ = y_
            p = self.model["discriminator"](disc_y)
            p_ = self.model["discriminator"](disc_y_.detach())
            real_loss, fake_loss = self.criterion["dis_adv"](p_, p)
            dis_loss = real_loss + fake_loss
            self.total_train_loss["train/real_loss"] += real_loss.item()
            self.total_train_loss["train/fake_loss"] += fake_loss.item()
            self.total_train_loss["train/discriminator_loss"] += dis_loss.item()

            # update discriminator
            self.optimizer["discriminator"].zero_grad()
            dis_loss.backward()
            if self.config["discriminator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.config["discriminator_grad_norm"],
                )
            self.optimizer["discriminator"].step()
            if self.config["discriminator_scheduler_type"] == "ReduceLROnPlateau":
                self.scheduler["discriminator"].step(dis_loss)
            else: 
                self.scheduler["discriminator"].step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.config["rank"] == 0:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({self.train_steps_per_epoch} steps per epoch)."
        )

        # needed for shuffle in distributed training
        if self.config["distributed"]:
            self.sampler["train"].set_epoch(self.epochs)

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        x, y, _, ar, _, spk_id, ph, _, _ = parse_batch(batch, self.device)

        #######################
        #      Generator      #
        #######################
        y_ = self.model["generator"](x, spk_id=spk_id, ar=ar, ph=ph)
        if self.config["use_ph_loss"]:
            y_, ph_ = y_
        if self.config["generator_params"]["out_channels"] > 1 and self.config.get("pqmf", False):
            y_mb_ = y_
            y_ = self.criterion["pqmf"].synthesis(y_mb_)

        # initialize
        aux_loss = 0.0  # called gen_loss during training

        # multi-resolution stft loss
        if self.config["use_stft_loss"]:
            sc_loss, mag_loss = self.criterion["stft"](y_, y)
            aux_loss += sc_loss + mag_loss
            self.total_eval_loss["eval/spectral_convergence_loss"] += sc_loss.item()
            self.total_eval_loss["eval/log_stft_magnitude_loss"] += mag_loss.item()

        # subband multi-resolution stft loss
        if self.config.get("use_subband_stft_loss", False):
            aux_loss *= 0.5  # for balancing with subband stft loss
            y_mb = self.criterion["pqmf"].analysis(y)
            sub_sc_loss, sub_mag_loss = self.criterion["sub_stft"](y_mb_, y_mb)
            self.total_eval_loss[
                "eval/sub_spectral_convergence_loss"
            ] += sub_sc_loss.item()
            self.total_eval_loss[
                "eval/sub_log_stft_magnitude_loss"
            ] += sub_mag_loss.item()
            aux_loss += 0.5 * (sub_sc_loss + sub_mag_loss)

        # mel spectrogram loss
        if self.config["use_mel_loss"]:
            mel_loss = self.criterion["mel"](y_, y)
            aux_loss += mel_loss
            self.total_eval_loss["eval/mel_loss"] += mel_loss.item()

        # weighting stft loss
        aux_loss *= self.config.get("lambda_aux", 1.0)

        # phoneme loss
        if self.config["use_ph_loss"]:
            ph_loss = self.criterion["ph"](ph_, ph)
            aux_loss += self.config["lambda_ph"] * ph_loss
            self.total_eval_loss["eval/ph_loss"] += ph_loss.item()

        # adversarial loss
        if ar is not None:
            disc_y = torch.cat([ar, y], dim=2)
            disc_y_ = torch.cat([ar, y_], dim=2)
        else:
            disc_y = y
            disc_y_ = y_
        p_ = self.model["discriminator"](disc_y_)
        adv_loss = self.criterion["gen_adv"](p_)
        gen_loss = aux_loss + self.config["lambda_adv"] * adv_loss

        # feature matching loss
        if self.config["use_feat_match_loss"]:
            p = self.model["discriminator"](disc_y)
            fm_loss = self.criterion["feat_match"](p_, p)
            self.total_eval_loss["eval/feature_matching_loss"] += fm_loss.item()
            gen_loss += (
                self.config["lambda_adv"] * self.config["lambda_feat_match"] * fm_loss
            )

        #######################
        #    Discriminator    #
        #######################
        p = self.model["discriminator"](disc_y)
        p_ = self.model["discriminator"](disc_y_)

        # discriminator loss
        real_loss, fake_loss = self.criterion["dis_adv"](p_, p)
        dis_loss = real_loss + fake_loss

        # add to total eval loss
        self.total_eval_loss["eval/adversarial_loss"] += adv_loss.item()
        self.total_eval_loss["eval/generator_loss"] += gen_loss.item()
        self.total_eval_loss["eval/real_loss"] += real_loss.item()
        self.total_eval_loss["eval/fake_loss"] += fake_loss.item()
        self.total_eval_loss["eval/discriminator_loss"] += dis_loss.item()

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.data_loader["dev"], desc="[eval]"), 1
        ):
            # eval one step
            self._eval_step(batch)

            # save intermediate result
            if eval_steps_per_epoch == 1:
                self._generate_and_save_intermediate_result(batch)

        logging.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(
                f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}."
            )

        if self.total_eval_loss["eval/mel_loss"] < self.best_mel_loss:
            best_mel_p = os.path.join(self.config["outdir"], "best_mel_step.txt")
            with open(best_mel_p, "w+") as ouf:
                ouf.write("%d\n" % self.steps)
            self.save_checkpoint(os.path.join(self.config["outdir"], "best_mel_ckpt.pkl"))
            self.best_mel_loss = self.total_eval_loss["eval/mel_loss"]

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    @torch.no_grad()
    def _generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # delayed import to avoid error https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/parallel_wavegan/bin/train.py#L448
        import matplotlib.pyplot as plt

        x_temp, y_temp, _, ar_temp, _, spk_id_temp, ph_temp, _, _ = parse_batch(batch, self.device)

        # generate
        y_temp_ = self.model["generator"](x_temp, spk_id=spk_id_temp, ar=ar_temp, ph=ph_temp)
        if self.config["use_ph_loss"]:
            y_temp_, ph_temp_ = y_temp_
        if self.config["generator_params"]["out_channels"] > 1 and self.config.get("pqmf", False):
            y_temp_ = self.criterion["pqmf"].synthesis(y_temp_)

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (y, y_) in enumerate(zip(y_temp, y_temp_), 1):
            if y.shape[0] == 1:
                y = y[0]
            if y_.shape[0] == 1:
                y_ = y_[0]
            if len(y.shape) == 1:
                y, y_ = y.view(-1).cpu().numpy(), y_.view(-1).cpu().numpy()
            else:
                y, y_ = y.cpu().numpy(), y_.cpu().numpy()
            figname = os.path.join(dirname, f"{idx}.png")
            plt.subplot(2, 1, 1)
            if len(y.shape) == 1:
                plt.plot(y)
            else:  # (C, T')
                plt.imshow(y)
            plt.title("groundtruth speech")
            plt.subplot(2, 1, 2)
            if len(y_.shape) == 1:
                plt.plot(y_)
            else:  # (C, T')
                plt.imshow(y_)
            plt.title(f"generated speech @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()
            if len(y.shape) == 1:
                y = np.clip(y, -1, 1)
                y_ = np.clip(y_, -1, 1)
                sf.write(figname.replace(".png", "_ref.wav"), y, self.config["sampling_rate"], "PCM_16")
                sf.write(figname.replace(".png", "_gen.wav"), y_, self.config["sampling_rate"], "PCM_16")
            if idx >= self.config["num_save_intermediate_results"]:
                break

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl"))
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}.")
            self._write_to_tensorboard(self.total_train_loss)
            self.total_train_loss = defaultdict(float)  # reset

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True


class SpeechCollater(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(self, batch_max_steps=20480, hop_size=256, aux_context_window=0, \
        use_noise_input=False, dataset_mode='a2w', use_spk_id=False, use_ph=False, config=None,
    ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            batch_max_steps (int): The maximum length of input signal in batch.
            hop_size (int): Hop size of auxiliary features.
            aux_context_window (int): Context window size for auxiliary feature conv.
            use_noise_input (bool): Whether to use noise input.

        assumes all mel lengths are > self.batch_max_frames + 2 * aux_context_window

        """
        assert batch_max_steps % hop_size == 0
        self.batch_max_steps = batch_max_steps
        self.batch_max_frames = batch_max_steps // hop_size  # for mel and art
        self.hop_size = hop_size
        self.aux_context_window = aux_context_window
        self.use_noise_input = use_noise_input
        self.dataset_mode = dataset_mode
        self.ar_len = None
        if config["generator_params"].get("use_ar", False):
            self.ar_len = int(config["generator_params"].get("ar_input", 512)/config["generator_params"]["out_channels"])
        self.package_mode = config.get("package_mode", "random_window")

        # set useful values in random cutting
        self.start_offset = aux_context_window  # 0, only used for selecting start idx
        self.end_offset = -(self.batch_max_frames + aux_context_window)
            # -self.batch_max_frames; only used for selecting start idx

        self.config = config
        if self.config is not None:
            self.audio_seq_len = self.config["batch_max_steps"]
            self.feature_seq_len = int(self.audio_seq_len/self.config["hop_size"])

        self.x_key, self.y_key, self.feature_key, self.feature_set = get_task_features(self.dataset_mode)

        if use_ph:
            self.feature_set.add('ph')
        if use_spk_id:
            self.feature_set.add('spk_id')

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features.

        Returns:
            Tensor: Gaussian noise batch (B, 1, T).
            Tensor: Auxiliary feature batch (B, C, T'), where
                T = (T' - 2 * aux_context_window) * hop_size.
            Tensor: Target signal batch (B, 1, T).

        """
        batch_d = {k: [] for k in self.feature_set}
        for d in batch:
            min_feature_len = 1e12
            for k in self.feature_set:
                if k == 'audio':
                    min_feature_len = min(min_feature_len, int(len(d[k])/self.hop_size))
                elif k != 'spk_id':
                    min_feature_len = min(min_feature_len, len(d[k]))
            if min_feature_len + self.end_offset > self.start_offset:
                for k in self.feature_set:
                    if k == 'audio':
                        d[k] = d[k][:min_feature_len*self.hop_size]
                    elif k != 'spk_id':
                        d[k] = d[k][:min_feature_len]  # (T, C)
                    batch_d[k].append(d[k])
        tensor_batch = {}
        if 'spk_id' in self.feature_set:
            tensor_batch['spk_id'] = torch.tensor(batch_d['spk_id'], dtype=torch.long)
        if self.package_mode == 'window':
            if self.ar_len is not None:
                logging.error('autoregression unimplemented for package_mode %s' % self.package_mode)
                exit()
            for k in self.feature_set:
                if k == 'ph':
                    ctensors = [torch.from_numpy(t).long() for t in batch_d[k]]
                else:
                    ctensors = [torch.from_numpy(t).float() for t in batch_d[k]]
                if k == 'audio':
                    tensor_batch[k] = combine_fixed_length([t.to(self.device, non_blocking=True).unsqueeze(1) for t in ctensors], self.audio_seq_len)
                else:
                    tensor_batch[k] = combine_fixed_length([t.to(self.device, non_blocking=True) for t in ctensors], self.feature_seq_len)
        elif self.package_mode == 'random_window':  # make batch with random cut
            c_lengths = [c.shape[0] for c in batch_d[self.feature_key]]
            # NOTE assumes that all c_lengths >= self.batch_max_frames
            start_frames = np.array([np.random.randint(self.start_offset, cl+self.end_offset) for cl in c_lengths])
            wav_starts = start_frames * self.hop_size
            wav_ends = wav_starts + self.batch_max_steps
            feature_starts = start_frames - self.aux_context_window
            feature_ends = start_frames + self.batch_max_frames + self.aux_context_window
            for k in self.feature_set:
                if k == 'audio':
                    cbatch = [cf[start:end] for cf, start, end in zip(batch_d[k], wav_starts, wav_ends)]
                    cbatch = np.stack(cbatch, axis=0)
                    tensor_batch[k] = torch.tensor(cbatch, dtype=torch.float).unsqueeze(1)  # (B, 1, T)
                else:
                    cbatch = [cf[start:end] for cf, start, end in zip(batch_d[k], feature_starts, feature_ends)]
                    cbatch = np.stack(cbatch, axis=0)
                    if k == 'ph':
                        tensor_batch[k] = torch.tensor(cbatch, dtype=torch.long)
                    else:
                        tensor_batch[k] = torch.tensor(cbatch, dtype=torch.float).transpose(2, 1)  # (B, C, T')
            if self.ar_len is not None:
                out_feats = batch_d[self.y_key]
                if self.y_key == 'audio':
                    out_starts = wav_starts
                else:
                    out_starts = feature_starts
                tensor_batch['ar'] = mk_ar_tensor(out_feats, out_starts, self.ar_len)
        elif self.package_mode == 'pad':
            if self.ar_len is not None:
                logging.error('autoregression unimplemented for package_mode %s' % self.package_mode)
                exit()
            max_feature_len = max([len(t) for t in batch_d[self.feature_key]])
            for k in self.feature_set:
                if k == 'audio':
                    audio_tensors = [torch.from_numpy(t).float() for t in batch_d[k]]
                    max_audio_len = max_feature_len*self.hop_size
                    new_audios = []
                    for t in audio_tensors:
                        pad_length = max_audio_len-len(t)
                        cpad = torch.zeros(pad_length, *t.size()[1:], dtype=torch.float, device=t.device)
                        new_t = torch.cat([t, cpad], 0)
                        new_audios.append(new_t)
                    tensor_batch[k] = torch.stack(new_audios).unsqueeze(1)  # (B, 1, T)
                elif k == 'ph':
                    ph_batch = [torch.from_numpy(feat).long() for feat in batch_d[k]]
                    new_phs = []
                    for t in ph_batch:
                        pad_length = max_feature_len-len(t)
                        cpad = torch.zeros(pad_length, *t.size()[1:], dtype=torch.long, device=t.device)
                        new_t = torch.cat([t, cpad], 0)
                        new_phs.append(new_t)
                    tensor_batch[k] = torch.stack(new_phs)
                else:
                    ctensors = [torch.from_numpy(t).float() for t in batch_d[k]]
                    new_feats = []
                    for t in ctensors:
                        pad_length = max_feature_len-len(t)
                        cpad = torch.zeros(pad_length, *t.size()[1:], dtype=torch.float, device=t.device)
                        new_t = torch.cat([t, cpad], 0)
                        new_feats.append(new_t)
                    tensor_batch['art'] = torch.stack(new_feats).transpose(2, 1)  # (B, C, T')
        tensor_batch['x'] = tensor_batch[self.x_key]
        tensor_batch['y'] = tensor_batch[self.y_key]
        return tensor_batch


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train articulatory model (See detail in ats/bin/train.py)."
    )
    parser.add_argument(
        "--train-wav-scp",
        default=None,
        type=str,
        help="kaldi-style wav.scp file for training. "
        "you need to specify either train-*-scp or train-dumpdir.",
    )
    parser.add_argument(
        "--train-feats-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file for training. "
        "you need to specify either train-*-scp or train-dumpdir.",
    )
    parser.add_argument(
        "--train-segments",
        default=None,
        type=str,
        help="kaldi-style segments file for training.",
    )
    parser.add_argument(
        "--train-dumpdir",
        default=None,
        type=str,
        help="directory including training data. "
        "you need to specify either train-*-scp or train-dumpdir.",
    )
    parser.add_argument(
        "--train-dumpdirs",
        default=None,
        type=str,
        help="directory including training data. "
        "you need to specify either train-*-scp or train-dumpdir.",
    )
    parser.add_argument(
        "--dev-wav-scp",
        default=None,
        type=str,
        help="kaldi-style wav.scp file for validation. "
        "you need to specify either dev-*-scp or dev-dumpdir.",
    )
    parser.add_argument(
        "--dev-feats-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file for vaidation. "
        "you need to specify either dev-*-scp or dev-dumpdir.",
    )
    parser.add_argument(
        "--dev-segments",
        default=None,
        type=str,
        help="kaldi-style segments file for validation.",
    )
    parser.add_argument(
        "--dev-dumpdir",
        default=None,
        type=str,
        help="directory including development data. "
        "you need to specify either dev-*-scp or dev-dumpdir.",
    )
    parser.add_argument(
        "--dev-dumpdirs",
        default=None,
        type=str,
        help="directory including development data. "
        "you need to specify either dev-*-scp or dev-dumpdir.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--pretrain",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to load pretrained params. (default="")',
    )
    parser.add_argument(
        "--pretrain2",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to load pretrained params. (default="")',
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--rank",
        "--local_rank",
        default=0,
        type=int,
        help="rank for distributed training. no need to explictly specify.",
    )
    args = parser.parse_args()

    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.rank)
        # setup for distributed training
        # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.distributed = args.world_size > 1
        if args.distributed:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # suppress logging for distributed training
    if args.rank != 0:
        sys.stdout = open(os.devnull, "w")

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # check arguments
    if (args.train_feats_scp is not None and args.train_dumpdir is not None) or (
        args.train_feats_scp is None and args.train_dumpdir is None
    ):
        raise ValueError("Please specify either --train-dumpdir or --train-*-scp.")
    if (args.dev_feats_scp is not None and args.dev_dumpdir is not None) or (
        args.dev_feats_scp is None and args.dev_dumpdir is None
    ):
        raise ValueError("Please specify either --dev-dumpdir or --dev-*-scp.")

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    config["version"] = ats.__version__  # add version info
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # get dataset
    if config["remove_short_samples"]:
        mel_length_threshold = config["batch_max_steps"] // config[
            "hop_size"
        ] + 2 * config["generator_params"].get("aux_context_window", 0)
    else:
        mel_length_threshold = None
    if args.train_wav_scp is None or args.dev_wav_scp is None:
        if config["format"] == "hdf5":
            audio_query, mel_query = "*.h5", "*.h5"
            audio_load_fn = lambda x: read_hdf5(x, "wave")  # NOQA
            mel_load_fn = lambda x: read_hdf5(x, "feats")  # NOQA
        elif config["format"] == "npy":
            audio_query, mel_query = "*-wave.npy", "*-feats.npy"
            audio_load_fn = np.load
            mel_load_fn = np.load
        else:
            raise ValueError("support only hdf5 or npy format.")
    if "dataset_mode" not in config:
        dataset_mode = 'default'
    else:
        dataset_mode = config["dataset_mode"]
    if "transform" not in config:
        transform = None
    else:
        transform = config["transform"]
    input_transform = config.get("input_transform", transform)
    if input_transform is not None:
        input_transform = getattr(ats.transforms, input_transform)
    output_transform = config.get("output_transform", transform)
    if output_transform is not None:
        output_transform = getattr(ats.transforms, output_transform)

    assert args.train_dumpdir is not None and args.dev_dumpdir is not None
    use_spk_id = config["generator_params"].get("use_spk_id", False)
    use_ph = config["generator_params"].get("use_ph", False) or config["generator_params"].get("use_ph_loss", False) \
                or dataset_mode == 'ph2a' or dataset_mode == 'ph2m'
    train_dataset = SpeechDataset(
        root_dir=args.train_dumpdir, audio_query=audio_query, audio_load_fn=audio_load_fn, mel_query=mel_query, mel_load_fn=mel_load_fn,
        allow_cache=config.get("allow_cache", False), transform=transform,
        input_transform=input_transform, output_transform=output_transform,
        use_spk_id=use_spk_id, use_ph=use_ph, dataset_mode=dataset_mode,
    )
    if use_spk_id:
        assert len(train_dataset.spks) == config["generator_params"]["num_spk"]
    dev_dataset = SpeechDataset(
        root_dir=args.dev_dumpdir, audio_query=audio_query, audio_load_fn=audio_load_fn, mel_query=mel_query, mel_load_fn=mel_load_fn,
        allow_cache=config.get("allow_cache", False), transform=transform,
        input_transform=input_transform, output_transform=output_transform,
        use_spk_id=use_spk_id, use_ph=use_ph, spks=train_dataset.spks, dataset_mode=dataset_mode,
    )
    train_collater = SpeechCollater(
        batch_max_steps=config["batch_max_steps"], hop_size=config["hop_size"],
        aux_context_window=config["generator_params"].get("aux_context_window", 0), # so 0
        use_noise_input=config.get("generator_type", "ParallelWaveGANGenerator") in ["ParallelWaveGANGenerator"],
        dataset_mode=dataset_mode, use_spk_id=use_spk_id, use_ph=use_ph, config=config,
    )
    dev_collater = SpeechCollater(
        batch_max_steps=config["batch_max_steps"], hop_size=config["hop_size"],
        aux_context_window=config["generator_params"].get("aux_context_window", 0), # so 0
        use_noise_input=config.get("generator_type", "ParallelWaveGANGenerator") in ["ParallelWaveGANGenerator"],
        dataset_mode=dataset_mode, use_spk_id=use_spk_id, use_ph=use_ph, config=config,
    )  # NOTE package_mode originally was always random_window for dev

    logging.info(f"The number of training files = {len(train_dataset)}.")
    logging.info(f"The number of development files = {len(dev_dataset)}.")

    dataset = {"train": train_dataset, "dev": dev_dataset}

    sampler = {"train": None, "dev": None}
    if args.distributed:
        # setup sampler for distributed training
        from torch.utils.data.distributed import DistributedSampler

        sampler["train"] = DistributedSampler(dataset=dataset["train"], num_replicas=args.world_size, rank=args.rank, shuffle=True)
        sampler["dev"] = DistributedSampler(dataset=dataset["dev"], num_replicas=args.world_size, rank=args.rank, shuffle=False)

    data_loader = {
        "dev": DataLoader(
            dataset=dataset["dev"], shuffle=False if args.distributed else True, collate_fn=dev_collater,
            batch_size=config["batch_size"], num_workers=config["num_workers"], sampler=sampler["dev"], pin_memory=config["pin_memory"],
        ),
    }

    batch_sampler_type = config.get("batch_sampler_type", "None")
    batch_sampler = {"train": None, "dev": None}
    if batch_sampler_type != "None":
        train_audio_lens_path = os.path.join(args.train_dumpdir, 'train_audio_lens.npy')
        if os.path.exists(train_audio_lens_path):
            train_audio_lens = np.load(train_audio_lens_path)
        else:
            train_audio_lens = []
            for audio, art in train_dataset:
                train_audio_lens.append(len(audio))
            train_audio_lens = np.array(train_audio_lens)
            np.save(train_audio_lens_path, train_audio_lens)
        batch_sampler_class = getattr(ats.samplers, batch_sampler_type)
        batch_sampler["train"] = batch_sampler_class(train_audio_lens, **config["batch_sampler_params"])
        data_loader["train"] = DataLoader(
            dataset=dataset["train"], collate_fn=train_collater,
            num_workers=config["num_workers"], batch_sampler=batch_sampler["train"], pin_memory=config["pin_memory"],
        )
    else:
        data_loader["train"] = DataLoader(
            dataset=dataset["train"], shuffle=False if args.distributed else True, collate_fn=train_collater,
            batch_size=config["batch_size"], num_workers=config["num_workers"], sampler=sampler["train"], pin_memory=config["pin_memory"],
        )

    # define models
    generator_class = getattr(
        ats.models,
        # keep compatibility
        config.get("generator_type", "ParallelWaveGANGenerator"),
    )
    discriminator_class = getattr(
        ats.models,
        # keep compatibility
        config.get("discriminator_type", "ParallelWaveGANDiscriminator"),
    )
    model = {
        "generator": generator_class(**config["generator_params"]).to(device),
        "discriminator": discriminator_class(**config["discriminator_params"]).to(device),
    }

    def count_parameters(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    logging.info("generator params = %s." % count_parameters(model["generator"]))

    # define criterions
    criterion = {
        "gen_adv": GeneratorAdversarialLoss(
            # keep compatibility
            **config.get("generator_adv_loss_params", {})
        ).to(device),
        "dis_adv": DiscriminatorAdversarialLoss(
            # keep compatibility
            **config.get("discriminator_adv_loss_params", {})
        ).to(device),
    }
    if config.get("use_stft_loss", True):  # keep compatibility
        config["use_stft_loss"] = True
        criterion["stft"] = MultiResolutionSTFTLoss(
            **config["stft_loss_params"],
        ).to(device)
    if config.get("use_subband_stft_loss", False):  # keep compatibility
        assert config["generator_params"]["out_channels"] > 1
        criterion["sub_stft"] = MultiResolutionSTFTLoss(
            **config["subband_stft_loss_params"],
        ).to(device)
    else:
        config["use_subband_stft_loss"] = False
    if config.get("use_feat_match_loss", False):  # keep compatibility
        criterion["feat_match"] = FeatureMatchLoss(
            # keep compatibility
            **config.get("feat_match_loss_params", {}),
        ).to(device)
    else:
        config["use_feat_match_loss"] = False
    if config.get("use_mel_loss", False):  # keep compatibility
        if "dataset_mode" not in config or config["dataset_mode"] == 'default' or config["dataset_mode"].endswith('2w'):
            if config.get("mel_loss_params", None) is None:
                criterion["mel"] = MelSpectrogramLoss(
                    fs=config["sampling_rate"],
                    fft_size=config["fft_size"],
                    hop_size=config["hop_size"],
                    win_length=config["win_length"],
                    window=config["window"],
                    num_mels=config["num_mels"],
                    fmin=config["fmin"],
                    fmax=config["fmax"],
                ).to(device)
            else:
                criterion["mel"] = MelSpectrogramLoss(
                    **config["mel_loss_params"],
                ).to(device)
        elif config["dataset_mode"] == 'a2m' or config["dataset_mode"] == 'w2a' or config["dataset_mode"] == 'm2a' \
                or config["dataset_mode"] == 'ph2a' or dataset_mode == 'ph2m' or dataset_mode == 'h2a' \
                or dataset_mode == 'mfcc2a' or dataset_mode.endswith('2tvpm') or dataset_mode.endswith('2tv') \
                or dataset_mode.endswith('2tvpff') or dataset_mode.endswith('2tvpffema'):
            criterion["mel"] = F.l1_loss
        else:
            raise ValueError("dataset_mode %s not supported" % config["dataset_mode"])
    else:
        config["use_mel_loss"] = False
    if config["generator_params"].get("use_ph_loss", False):  # keep compatibility
        criterion["ph"] = F.cross_entropy
        config["use_ph_loss"] = True
    else:
        config["use_ph_loss"] = False

    # define special module for subband processing
    if config["generator_params"]["out_channels"] > 1 and config.get("pqmf", False):
        criterion["pqmf"] = PQMF(
            subbands=config["generator_params"]["out_channels"],
            # keep compatibility
            **config.get("pqmf_params", {}),
        ).to(device)

    # define optimizers and schedulers
    generator_optimizer_class = getattr(
        ats.optimizers,
        # keep compatibility
        config.get("generator_optimizer_type", "RAdam"),
    )
    discriminator_optimizer_class = getattr(
        ats.optimizers,
        # keep compatibility
        config.get("discriminator_optimizer_type", "RAdam"),
    )
    optimizer = {
        "generator": generator_optimizer_class(
            model["generator"].parameters(),
            **config["generator_optimizer_params"],
        ),
        "discriminator": discriminator_optimizer_class(
            model["discriminator"].parameters(),
            **config["discriminator_optimizer_params"],
        ),
    }
    generator_scheduler_class = getattr(
        torch.optim.lr_scheduler,
        # keep compatibility
        config.get("generator_scheduler_type", "StepLR"),
    )
    discriminator_scheduler_class = getattr(
        torch.optim.lr_scheduler,
        # keep compatibility
        config.get("discriminator_scheduler_type", "StepLR"),
    )
    scheduler = {
        "generator": generator_scheduler_class(
            optimizer=optimizer["generator"],
            **config["generator_scheduler_params"],
        ),
        "discriminator": discriminator_scheduler_class(
            optimizer=optimizer["discriminator"],
            **config["discriminator_scheduler_params"],
        ),
    }
    if args.distributed:
        # wrap model for distributed training
        try:
            logging.error('need to uncomment apex.parallel and DistributedDataParallel lines')
            exit()
            # from apex.parallel import DistributedDataParallel
        except ImportError:
            raise ImportError(
                "apex is not installed. please check https://github.com/NVIDIA/apex."
            )
        # model["generator"] = DistributedDataParallel(model["generator"])
        # model["discriminator"] = DistributedDataParallel(model["discriminator"])

    # show settings
    logging.info(model["generator"])
    logging.info(model["discriminator"])
    logging.info(optimizer["generator"])
    logging.info(optimizer["discriminator"])
    logging.info(scheduler["generator"])
    logging.info(scheduler["discriminator"])
    for criterion_ in criterion.values():
        logging.info(criterion_)

    # define trainer
    trainer = Trainer(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # load pretrained parameters from checkpoint
    if len(args.pretrain) != 0:
        checkpoint2_path = args.pretrain2 if len(args.pretrain2) != 0 else None
        trainer.load_checkpoint(args.pretrain, load_only_params=True, checkpoint2_path=checkpoint2_path)
        logging.info(f"Successfully loaded parameters from {args.pretrain}.")
        if len(args.pretrain2) != 0:
            logging.info(f"Successfully loaded parameters from {args.pretrain2}.")

    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")

    # run training loop
    try:
        trainer.run()
    finally:
        trainer.save_checkpoint(
            os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl")
        )
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
