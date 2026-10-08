"""SoundAQnet input features without the authors' Windows tools.

The SoundSCaper pipeline turns every clip into
  * a log-mel spectrogram: mono, 16 kHz, 512-point Hann window, hop 160, 64 mel bins, 50-8000 Hz
    (Feature_log_mel/log_mel_spectrogram.py), shape (3001, 64) for 30 s, and
  * a time-varying ISO 532-1 loudness curve in sone, 2 ms steps (ISO_532-1.exe, free field),
    shape (15000, 1) for 30 s.
Both were computed from a mono 44.1 kHz wav (channel mean), and the loudness was calibrated with a
1 kHz sine of 60 dB SPL. Here the loudness is computed with MoSQITo (loudness_zwtv), which
reproduces the shipped demo curve to 0.0002 sone on average (correlation 1.0), and the log-mel with
torchlibrosa as in the original script.

LEVEL. Both features depend on absolute level. `pa_per_unit` says how many pascals one unit of the
input samples is:
  * ARAUS-style wavs (digital, as used to train SoundAQnet): ARAUS_PA_PER_DIGITAL;
  * ISD recordings (float, "calibrated in Pa"): 1.0;
  * uncalibrated audio: pick an assumed level and say so (see assume_level()).
The log-mel is computed on the equivalent ARAUS digital signal (pa / ARAUS_PA_PER_DIGITAL), because
that is the scale the network saw in training.
"""
import numpy as np
from scipy.signal import resample_poly
from math import gcd

# Pa per digital unit of the ARAUS wavs: the calibration file (1 kHz sine, 60 dB SPL = 0.02 Pa rms)
# has rms 0.007052689 in digital units, so 0.02 / 0.007052689.
ARAUS_PA_PER_DIGITAL = 2.835797716391444
SR_WAV = 44100
SR_MEL = 16000
CLIP_SECONDS = 30


def to_mono_44k(x, sr):
    """(n,) or (n, channels) -> mono float64 at 44.1 kHz (channel mean, as librosa mono=True)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != SR_WAV:
        g = gcd(int(sr), SR_WAV)
        x = resample_poly(x, SR_WAV // g, int(sr) // g)
    return x


def assume_level(y, target_dba_rms_pa=None, rms_db_spl=65.0):
    """Scale an uncalibrated signal so that its rms corresponds to `rms_db_spl` dB SPL (unweighted).
    Returns the factor to multiply the samples with to get pascals. This is an assumption, not a
    measurement: every clip then has the same overall level, and the loudness feature carries no
    level information beyond spectral shape and dynamics."""
    rms = float(np.sqrt(np.mean(np.asarray(y, np.float64) ** 2)))
    return (20e-6 * 10 ** (rms_db_spl / 20)) / max(rms, 1e-12)


def log_mel(y_digital_44k):
    """Mono 44.1 kHz signal on the ARAUS digital scale -> (frames, 64) log-mel (float32)."""
    import librosa
    import torch
    from torchlibrosa.stft import LogmelFilterBank, Spectrogram

    y = librosa.resample(np.asarray(y_digital_44k, np.float32), orig_sr=SR_WAV, target_sr=SR_MEL, res_type="soxr_hq")
    spec = Spectrogram(n_fft=512, hop_length=160, win_length=512, window="hann", center=True,
                       pad_mode="reflect", freeze_parameters=True)
    mel = LogmelFilterBank(sr=SR_MEL, n_fft=512, n_mels=64, fmin=50, fmax=SR_MEL // 2, ref=1.0, amin=1e-10,
                           top_db=None, freeze_parameters=True)
    with torch.no_grad():
        return mel(spec(torch.tensor(y[None, :], dtype=torch.float32)))[0, 0].numpy()


def iso_loudness(y_pa_44k):
    """Mono 44.1 kHz signal in pascal -> (frames, 1) time-varying loudness in sone, 2 ms steps."""
    from mosqito.sq_metrics import loudness_zwtv

    n, _, _, _ = loudness_zwtv(np.asarray(y_pa_44k, np.float64), SR_WAV, field_type="free")
    return np.asarray(n, np.float32)[:, None]


def extract(x, sr, pa_per_unit, max_seconds=CLIP_SECONDS, gain_db=0.0):
    """Audio in file units -> (log_mel, loudness). Clips longer than max_seconds are cut to the
    first max_seconds; shorter clips are NOT padded (zeros would distort both features). gain_db is
    applied to the pressure signal, for level-sensitivity tests."""
    y = to_mono_44k(x, sr)[: int(max_seconds * SR_WAV)]
    y_pa = y * pa_per_unit * 10 ** (gain_db / 20)
    return log_mel(y_pa / ARAUS_PA_PER_DIGITAL), iso_loudness(y_pa)
