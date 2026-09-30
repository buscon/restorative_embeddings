"""Audio loading and preparation shared by ARAUS and ISD.

Both datasets are turned into the same representation before any feature is
computed:

* binaural -> mono (mean of the two channels; CLAP is a mono model),
* resampled to 48 kHz (CLAP's training rate),
* at most the first 30 s kept (ARAUS stimuli are 30 s; most ISD recordings
  are 30-35 s, but ~70 are shorter, down to a few seconds),
* RMS-normalised to a fixed digital level.

The level normalisation is deliberate. ISD WAVs are calibrated in pascal,
while ARAUS stimuli are digital mixes whose gains were set for headphone
playback, so their raw amplitudes are not comparable. Absolute level is
therefore removed from the audio and supplied separately as a calibrated
feature (LA50) taken from each dataset's own measurements.
"""

from __future__ import annotations

from collections import OrderedDict
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 48_000
CLIP_SECONDS = 30
WINDOW_SECONDS = 10
TARGET_RMS_DBFS = -26.0


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def resample(y: np.ndarray, sr: int, target: int = TARGET_SR) -> np.ndarray:
    if sr == target:
        return y
    g = gcd(int(sr), int(target))
    return resample_poly(y, target // g, sr // g)


def normalise_rms(y: np.ndarray, dbfs: float = TARGET_RMS_DBFS) -> np.ndarray:
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    if rms < 1e-12:
        return y
    return y * (10 ** (dbfs / 20) / rms)


def prepare(x: np.ndarray, sr: int) -> np.ndarray:
    """Multichannel or mono array -> normalised mono float32 at 48 kHz, at most
    30 s long. Shorter recordings are NOT padded: silence would distort both the
    embedding and the indices (it made temporal entropy NaN, for example)."""
    y = resample(to_mono(np.asarray(x, dtype=np.float64)), sr)
    return normalise_rms(y[: CLIP_SECONDS * TARGET_SR]).astype(np.float32)


def windows(y: np.ndarray, n: int = CLIP_SECONDS // WINDOW_SECONDS) -> np.ndarray:
    """Signal -> (3, 10 s) windows, always exactly 10 s so CLAP never crops
    randomly (its feature extractor uses random truncation for longer input).

    * 30 s: three adjacent windows (0-10, 10-20, 20-30 s).
    * 10-30 s: three evenly spaced windows that overlap and cover the whole
      recording, e.g. 20 s -> 0-10, 5-15, 10-20 s.
    * under 10 s: the recording is repeated to fill 10 s (as CLAP itself does
      for short input), and that one window is used three times.
    A fixed number of windows keeps batching simple; with 30 s input the result
    is identical to plain non-overlapping splitting.
    """
    w = WINDOW_SECONDS * TARGET_SR
    if len(y) < w:
        y = np.tile(y, int(np.ceil(w / len(y))))[:w]
        return np.repeat(y[None, :], n, axis=0)
    starts = np.linspace(0, len(y) - w, n).round().astype(int)
    return np.stack([y[s:s + w] for s in starts])


def read(path: str | Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x, sr


class ArausMixer:
    """Rebuilds ARAUS stimuli in memory (soundscape + masker at a given SMR).

    Replicates `make_augmented_soundscapes` from
    github.com/ntudsp/araus-dataset-baseline-models (araus_utils.py), so the
    ~132 GB of augmented WAVs never have to be written to disk.
    """

    def __init__(self, soundscapes, maskers, soundscape_dir, masker_dir, cache_size: int = 32):
        self.s = soundscapes.set_index("soundscape")
        self.m = maskers.set_index("masker")
        self.sdir = Path(soundscape_dir)
        self.mdir = Path(masker_dir)
        self.cache: OrderedDict[str, tuple[np.ndarray, int]] = OrderedDict()
        self.cache_size = cache_size

    def _load(self, path: Path):
        key = str(path)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        val = read(path)
        self.cache[key] = val
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return val

    def mix(self, soundscape: str, masker: str, smr: float) -> tuple[np.ndarray, int]:
        gain_s = float(self.s.at[soundscape, "gain_s"])
        leq_s = float(self.s.at[soundscape, "insitu_leq"])
        leq_m = leq_s - smr
        r = int(np.round(leq_m))
        gain_c = float(self.m.at[masker, f"gain_{r}dB"])
        leq_c = float(self.m.at[masker, f"leq_at_gain_{r}dB"])
        gain_m = gain_c * 10 ** ((leq_m - leq_c) / 20)

        x_s, sr_s = self._load(self.sdir / soundscape)
        x_m, sr_m = self._load(self.mdir / masker)
        if sr_s != sr_m:
            raise ValueError(f"sample-rate mismatch {soundscape} ({sr_s}) vs {masker} ({sr_m})")
        x_m = np.repeat(x_m[:, :1], x_s.shape[1], axis=1)  # mono masker -> both ears
        n = min(len(x_s), len(x_m))
        return gain_s * x_s[:n] + gain_m * x_m[:n], sr_s
