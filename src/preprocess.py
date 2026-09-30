"""Signal pre-processing (CH03-01 resampling, CH04-03 FIR pre-emphasis).

Stored audio (data/processed) = mono + 16 kHz + edge silence trimmed.
Pre-emphasis is NOT baked into the stored files: it is applied only before
spectral features, so pitch and the MFCC baseline can use the clean signal.
"""
from math import gcd

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter, resample_poly

from src import config as cfg


def to_mono(x):
    """(n,) or (n, ch) -> (n,). The stereo files have identical channels."""
    x = np.asarray(x, dtype=float)
    return x if x.ndim == 1 else x.mean(axis=1)


def resample(x, sr_in, sr_out):
    """Rational rate change I/D (44.1k -> 16k: I/D = 160/441).
    resample_poly = upsample by I, Kaiser-window FIR low-pass, downsample by D."""
    if sr_in == sr_out:
        return np.asarray(x, dtype=float)
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g)


def trim_silence(x, sr, top_db=40.0, frame_ms=25.0, min_len=cfg.FRAME_LEN):
    """Drop leading/trailing frames whose RMS is more than `top_db` below the
    loudest frame. Pauses inside the utterance are kept.
    Returns the original signal if trimming would leave < min_len samples."""
    L = max(1, int(sr * frame_ms / 1000))
    if len(x) < L:
        return x
    frames = sliding_window_view(x, L)[::L]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    if rms.max() == 0:
        return x
    db = 20 * np.log10(np.maximum(rms, 1e-12) / rms.max())
    loud = np.flatnonzero(db > -top_db)
    start, stop = loud[0] * L, min(len(x), (loud[-1] + 1) * L)
    return x[start:stop] if stop - start >= min_len else x


def pre_emphasis(x, a=cfg.PRE_EMPHASIS):
    """First-order FIR high-pass: y[n] = x[n] - a x[n-1]."""
    return lfilter([1.0, -a], [1.0], x)


def frame_signal(x, frame_len=cfg.FRAME_LEN, hop=cfg.HOP):
    """(n,) -> (n_frames, frame_len). Zero-pads the tail so no sample is lost."""
    x = np.asarray(x, dtype=float)
    n_frames = 1 + max(0, int(np.ceil((len(x) - frame_len) / hop)))
    padded = np.zeros((n_frames - 1) * hop + frame_len)
    padded[: len(x)] = x
    return sliding_window_view(padded, frame_len)[::hop]


def load_for_storage(audio, sr):
    """Raw decoded audio -> signal written to data/processed."""
    y = resample(to_mono(audio), sr, cfg.SR)
    return trim_silence(y, cfg.SR)
