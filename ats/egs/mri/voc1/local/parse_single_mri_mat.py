#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Peter Wu
#  Apache 2.0 License (https://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import scipy
import traceback

from scipy.interpolate import interp1d


def parse_mri_mat(inp, oup):
    """Parses MRI datum.

    Args:
        inp: input .mat file
        oup: output .npy file
    """
    m = scipy.io.loadmat(inp)
    try:
        feats = []
        for raw_frame in m['trackdata'][0]:
            feat0 = raw_frame[0][0][0][0][0][0][0][0][0][0][0]  # (70, 2)
            feat1 = raw_frame[0][0][0][0][0][0][0][1][0][0][0]  # (60, 2) or (40, 2)
            ints = raw_frame[0][0][0][0][0][0][0][1][0][0][1][:, 0]
            # interpolating feat1 so that all have the same length
            new_feat1 = [[], [], [], []]
            for feat1_idx in range(len(ints)):
                cint = ints[feat1_idx]
                cfeat1_v = feat1[cint]
                new_feat1[cint-1].append(cfeat1_v)
            assert len(new_feat1[3]) == 5
            assert len(new_feat1[0]) >= 10
            assert len(new_feat1[1]) >= 10
            new_feat1 = [np.stack(new_feat1[0]), np.stack(new_feat1[1]), np.stack(new_feat1[3])]
            if len (new_feat1[0]) > 10:
                x = np.linspace(0, 9, num=len(new_feat1[0]), endpoint=True)
                y = new_feat1[0]
                f = interp1d(x, y, axis=0)
                xnew = np.linspace(0, 9, num=10, endpoint=True)
                new_feat1[0] = f(xnew)
            if len (new_feat1[1]) > 10:
                x = np.linspace(0, 9, num=len(new_feat1[1]), endpoint=True)
                y = new_feat1[1]
                f = interp1d(x, y, axis=0)
                xnew = np.linspace(0, 9, num=10, endpoint=True)
                new_feat1[1] = f(xnew)
            new_feat1 = np.concatenate(new_feat1)
            assert new_feat1.shape[0] == 25 and new_feat1.shape[1] == 2
            feat2 = raw_frame[0][0][0][0][0][0][0][2][0][0][0]  # (60, 2)
            feat_list = [feat0[:, 0], feat0[:, 1], new_feat1[:, 0], new_feat1[:, 1], feat2[:, 0], feat2[:, 1]]
            feat = np.concatenate(feat_list, axis=0)
            feats.append(feat)
        feat = np.stack(feats)                    
        np.save(oup, feat)
    except Exception as e:
        print(e)
        traceback.print_exc()
