#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Peter Wu
#  MIT License (https://opensource.org/licenses/MIT)

"""Segment waveform using phoneme boundaries."""

import os
import soundfile as sf

from praatio import textgrid


phoneme_vocab = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', 'spn']
vowels_no_suf = ["AA", "AE", "AO", "AH", "EH", "AW", "AY", "ER", "OW", "EY", "OY", "UH", "IH", "UW", "IY"]
new_vowels = []
for v in vowels_no_suf:
    new_vowels += [v+'0', v+'1', v+'2']
vowels = new_vowels
vowel_set = set(vowels)


def split_wav_by_ph(tg_p, wav_p):
    """Segment waveform using phoneme boundaries.

    spn is silence.

    Args:
        tg_p: path to .TextGrid file
        wav_p: path to .wav file

    Return:
        ph2wavs: {phoneme: waveform}
        vowel_wavs: vowel waveforms
        const_wavs: consonant waveforms
    """
    ph2wavs = {}
    vowel_wavs = set()
    const_wavs = set()
    cwav, sr = sf.read(wav_p)
    assert sr == 16000
    if os.path.exists(tg_p):
        tg = textgrid.openTextgrid(tg_p, includeEmptyIntervals=False)
        wordTier = tg.tierDict['phones']
        intervals = wordTier.entryList
        for (start, end, phoneme) in intervals:
            si = int(start*sr)
            ei = int(end*sr)
            wav_seg = cwav[si:ei]
            if phoneme in ph2wavs:
                ph2wavs[phoneme].append(wav_seg)
            else:
                ph2wavs[phoneme] = [wav_seg]
            if phoneme != 'spn':
                if phoneme in vowel_set:
                    vowel_wavs.add(wav_seg)
                else:
                    const_wavs.add(wav_seg)
    return ph2wavs, vowel_wavs, const_wavs
