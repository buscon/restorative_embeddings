"""Ecoacoustic and descriptive indices (scikit-maad + librosa).

Computed on the prepared (mono, 48 kHz, 30 s, RMS-normalised) signal so the
values are comparable across ARAUS and ISD. Indices with absolute dB
thresholds (e.g. ADI) would otherwise depend on each dataset's calibration.

Frequency limits follow common scikit-maad defaults for urban/terrestrial
work: anthropophony 0-1 kHz, biophony 1-10 kHz (NDSI), 2-15 kHz (BI).
"""

from __future__ import annotations

import numpy as np
import librosa
from maad import features, sound

INDEX_COLUMNS = [
    "NDSI", "BI", "ACI", "ADI", "H", "spectral_centroid", "spectral_flatness", "onset_rate",
]


def compute_indices(y: np.ndarray, sr: int) -> dict[str, float]:
    psd, _, fn, _ = sound.spectrogram(y, sr, nperseg=1024, noverlap=512, mode="psd")
    amp, _, _, _ = sound.spectrogram(y, sr, nperseg=1024, noverlap=512, mode="amplitude")

    ndsi = features.soundscape_index(psd, fn, flim_bioPh=(1000, 10000), flim_antroPh=(0, 1000))[0]
    bi = features.bioacoustics_index(amp, fn, flim=(2000, 15000))
    aci = features.acoustic_complexity_index(amp)[2]
    adi = features.acoustic_diversity_index(amp, fn, fmin=0, fmax=20000, bin_step=1000,
                                            dB_threshold=-50, index="shannon")
    h = features.temporal_entropy(y) * features.frequency_entropy(psd)[0]

    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    return {
        "NDSI": float(ndsi),
        "BI": float(bi),
        "ACI": float(aci),
        "ADI": float(adi),
        "H": float(h),
        "spectral_centroid": float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()),
        "spectral_flatness": float(librosa.feature.spectral_flatness(y=y).mean()),
        "onset_rate": len(onsets) / (len(y) / sr),  # onsets per second
    }
