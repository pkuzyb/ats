# -*- coding: utf-8 -*-

# Copyright 2022 Peter Wu
#  Apache 2.0 License (https://www.apache.org/licenses/LICENSE-2.0)

"""Data utils."""

import numpy as np
import torch


def combine_fixed_length(tensor_list, length):
    '''
    Args:
        tensor_list: list

    Return:
        tensor with shape (batch_size, num_feats, T)
    '''
    total_length = sum(t.size(0) for t in tensor_list)
    if total_length % length != 0:
        pad_length = length - (total_length % length)
        tensor_list = list(tensor_list)  # copy
        tensor_list.append(torch.zeros(pad_length,*tensor_list[0].size()[1:], dtype=tensor_list[0].dtype, device=tensor_list[0].device))
        total_length += pad_length
    tensor = torch.cat(tensor_list, 0)
    n = total_length // length
    combined = tensor.view(n, length, *tensor.size()[1:])
    combined = combined.permute(0, 2, 1)
    return combined


def mk_ar_tensor(cfeatures, feature_starts, ar_len):
    """Creates the autoregressive features preceding the given features at the given starting indices.
    
    Args:
        cfeatures: features to extract autoregressive features from
            list of 1D (e.g., waveform) or 2D (e.g., spectrum) features
        feature_starts: indices before which the autoregressive features are located
        ar_len: the length of the autoregressive features
    """
    is_1d = len(cfeatures[0].shape) == 1
    ar_batch = []
    for ar_feat, start in zip(cfeatures, feature_starts):
        if start >= ar_len:
            ar = ar_feat[start-ar_len:start]
        else:
            ar = ar_feat[:start]  # (T, channels)
            if is_1d:
                ar = np.pad(ar, (ar_len-len(ar), 0), 'constant', constant_values=0)
            else:
                ar = np.pad(ar, ((ar_len-len(ar),0), (0,0)), mode='constant', constant_values=0)
        ar_batch.append(ar)
    ar_batch = np.stack(ar_batch, axis=0)
    ar_batch = torch.tensor(ar_batch, dtype=torch.float)
    if is_1d:
        ar_batch = ar_batch.unsqueeze(1)  # (B, 1, T_ar)
    else:
        ar_batch = ar_batch.transpose(2, 1)  # (B, channels, T_ar)
    return ar_batch


def parse_batch(cbatch, device):
    cx = cbatch['x']
    cy = cbatch['y'].to(device)
    cy2 = None if 'y2' not in cbatch else cbatch['y2'].to(device)
    car = None if 'ar' not in cbatch else cbatch['ar'].to(device)
    car2 = None if 'ar2' not in cbatch else cbatch['ar2'].to(device)
    cspk_id = None if 'spk_id' not in cbatch else cbatch['spk_id'].to(device)
    cph = None if 'ph' not in cbatch else cbatch['ph'].to(device)
    cpitch = None if 'pitch' not in cbatch else cbatch['pitch'].to(device)
    cperiodicity = None if 'periodicity' not in cbatch else cbatch['periodicity'].to(device)
    if isinstance(cx, list):  # eg in multimodal case
        cx = [t.to(device) if t is not None else None for t in cx]
    else:
        cx = cx.to(device)
    return cx, cy, cy2, car, car2, cspk_id, cph, cpitch, cperiodicity
