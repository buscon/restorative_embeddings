# SoundAQnet: validation on ISD (plan B, step 0)

Goal: before using SoundAQnet (Hou et al., 2024, the rater inside SoundSCaper) to label Clotho or
other uncalibrated clips with ISOPleasant, check on audio it was not trained on whether it agrees
with what people rated, and whether its output survives shorter clips and level changes. ISD
(823 rated field recordings, calibrated in pascal) is the test set. The decision rule is in
`02_evaluate_isd.py` and was fixed before any ISD result was seen.

## What is here

| File | Purpose |
|---|---|
| `dgl_shim.py` | The few DGL calls SoundAQnet makes, re-implemented with plain tensors (DGL has no wheels for current PyTorch/CUDA, e.g. RTX 50 series). |
| `model.py` | `SoundAQnetRunner`: imports the model code from the SoundSCaper checkout, loads weights and normalisation (numpy-only unpickler for the third-party pickles), `predict(mel, loudness)`. |
| `features.py` | Log-mel (torchlibrosa, as the authors) and ISO 532-1 time-varying loudness with MoSQITo instead of the Windows `ISO_532-1.exe`. Level handling is explained in the docstring. |
| `00_demo_check.py` | Reproduces the demo that ships with SoundSCaper. Run first on every machine. |
| `01_rate_isd.py` | Runs SoundAQnet on the rated ISD recordings under 6 conditions (full, gain -20/-10/+10 dB, first 10 s, first 5 s). Cached and resumable. |
| `02_evaluate_isd.py` | Correlation with the human ratings (cluster bootstrap over locations), bias, spread, sensitivity, decision rule. |

## Setup

```bash
mkdir -p third_party && git clone https://github.com/Yuanbo2020/SoundSCaper third_party/SoundSCaper
git -C third_party/SoundSCaper checkout 84e7ef0   # the commit these scripts were tested with (26 Mar 2026)
# environment B (stable-audio-tools venv, torch already there):
pip install mosqito torchlibrosa librosa soundfile scipy pandas tqdm
python soundaqnet/00_demo_check.py            # about 1 minute; add --skip-features for the fast part
```

The SoundSCaper repo states no licence, so it is cloned into `third_party/` (git-ignored) and not
copied into this repo. The clone is about 700 MB (many model checkpoints and demo audio); a sparse
checkout of `Inferring_soundscape_clips_for_LLM/`, `Feature_log_mel/` and `Feature_loudness_ISO532_1/`
is enough.

Then, with ISD downloaded and `scripts/01_prepare.py` run:

```bash
python soundaqnet/01_rate_isd.py --workers 8 --device cuda     # hours on few cores, see below
python soundaqnet/02_evaluate_isd.py
```

## Verified so far (7 Oct 2026, CPU, Python 3.13, torch 2.x, mosqito 1.2.1, librosa 1.0)

- **Model port:** with the shipped features, outputs equal the shipped demo output (ISOPleasant
  0.423366 vs 0.42336637, eight PAQ values to 5 digits, event probabilities within 2e-7).
- **Loudness:** MoSQITo `loudness_zwtv`, free field, calibrated like the authors (1 kHz sine at
  60 dB SPL = 0.02 Pa rms, giving 2.8358 Pa per digital unit), matches the shipped ISO 532-1 curve
  (correlation 1.0, mean abs difference 0.0002 sone).
- **Log-mel:** close but not identical (mean abs difference 0.09 dB, a few low-energy bins up to
  5 dB), from the resampler (the authors used an older librosa). End to end from the wav, ISOPleasant
  is 0.413 vs 0.423 shipped, PAQ values within 0.03.
- **One clip, one anecdote, not evidence:** on the demo clip ISOPleasant moved from 0.42 to 0.56
  (gain -20 dB), 0.51 (-10 dB), 0.33 (+10 dB); first 10 s gave 0.42, first 5 s 0.27. So level and
  length do change the output; ISD will show how much.
- **Scripts 01 and 02** ran end to end on 6 synthetic noise files with random "ratings". This tests
  the mechanics only. They have **not** been run on ISD.

## Things to know

- **Speed:** the loudness step takes about 60 s per 30 s clip on 2 cores (MoSQITo is slow). 823
  clips for `full` plus 5 conditions on 150 clips is roughly 1500 jobs, so use `--workers` and
  expect hours on a laptop. Features are cached in `soundaqnet/cache_isd/`.
- **Calibration assumption:** ISD samples are taken to be pascal (`pa_per_unit = 1.0`, as stated in
  the milestone-1 notes). `01_rate_isd.py` prints the rms level of 40 recordings; if the median is
  not about 50 to 85 dB SPL, stop and check the ISD documentation.
- **Mono:** the authors computed both features from a mono (channel mean) 44.1 kHz file; ISD is
  binaural, so the same is done here. Binaural loudness would differ.
- **Log-mel and level:** the log-mel is also level dependent, so the gain conditions change both
  inputs. ISD signals are converted to the ARAUS digital scale (divide by 2.8358) before the mel.
- **Outputs are coarse:** 3 scene classes (public square, park, street traffic) and 15 audio-event
  classes. Training data: ARAUS (30 s clips, headphones, urban scenes).
- **Clips shorter than 30 s** are not padded; the network accepts about 2.8 s and longer.
- **Uncalibrated audio (Clotho, FSD50K):** `features.assume_level()` sets an assumed rms level. If
  the gain conditions on ISD show strong level dependence, this assumption drives the ratings.
