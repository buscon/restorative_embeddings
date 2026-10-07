"""Audio conditioning for training data: one place for what the ARAUS export does.

44.1 kHz, stereo, 30 Hz high-pass (removes infrasound), -23 LUFS integrated loudness.
generation/02_export.py has its own copy of these steps for ARAUS; the ISD scripts use
this one. Keep them equivalent if you change either.
"""
from math import gcd

import numpy as np


def condition_for_training(y: np.ndarray, sr: int, target_sr: int = 44100,
                           highpass_hz: float = 30.0, target_lufs: float = -23.0) -> np.ndarray:
    """y: (samples, channels) float. Returns (samples, 2) float32 at target_sr."""
    import pyloudnorm
    from scipy.signal import butter, resample_poly, sosfiltfilt

    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    elif y.shape[1] > 2:
        y = y[:, :2]
    if not np.isfinite(y).all():
        raise ValueError("non-finite samples")

    if sr != target_sr:
        g = gcd(sr, target_sr)
        y = resample_poly(y, target_sr // g, sr // g, axis=0)
    y = sosfiltfilt(butter(4, highpass_hz, btype="highpass", fs=target_sr, output="sos"), y, axis=0)
    y = y.astype(np.float32)

    loudness = pyloudnorm.Meter(target_sr).integrated_loudness(y)
    if not np.isfinite(loudness):
        raise ValueError("loudness not measurable (silent file?)")
    y = pyloudnorm.normalize.loudness(y, loudness, target_lufs)
    if np.abs(y).max() > 1.0:
        y = np.tanh(y)
    return y.astype(np.float32)
