# Plan B Implementation Guide

This document provides a step-by-step guide to implementing Plan B: fine-tuning Stable Audio Open with FSD50K/AudioSet and predicted pleasantness labels.

## Overview

Plan B differs from Plan A (which uses ARAUS with human pleasantness ratings) by:

1. **Data source:** Large general sound datasets (FSD50K ~51k clips, AudioSet ~2M)
2. **Pleasantness labels:** Predicted by SoundAQnet, not human-rated
3. **Model:** Stable Audio Open (SAO Small or 1.0), same as plan A
4. **Evaluation:** Same automatic metrics + human listening test

## Timeline and Effort

- **Step 0 (validation):** 1–2 days (depends on SoundAQnet setup)
- **Steps 1–2 (selection & rating):** 2–3 days (downloading FSD50K, running inference)
- **Steps 3–4 (captions & export):** 1 day
- **Step 5 (fine-tuning):** 3–7 days (GPU time, depends on batch size and data)
- **Steps 6–7 (evaluation & listening test):** 2–3 days

**Total:** ~2–3 weeks

## Prerequisites

### Python environment
```bash
# Create a fresh environment for plan B
conda create -n plan_b python=3.11 -y
conda activate plan_b

# Install dependencies (in the project root)
pip install -r requirements.txt  # Project requirements
pip install librosa soundfile pyloudnorm  # Audio processing
pip install scipy statsmodels  # Statistics
```

### External repositories
```bash
# SoundSCaper (for SoundAQnet)
cd third_party
git clone https://github.com/Yuanbo2020/SoundSCaper.git
cd SoundSCaper
pip install -r requirements.txt
cd ../..

# Stable Audio Tools (for SAO fine-tuning)
pip install git+https://github.com/stability-ai/stable-audio-tools.git
```

### Data
- **FSD50K:** Download from Zenodo (10.5281/zenodo.4060432)
- **ARAUS + ISD:** Already in data/processed/ (from milestone 1)

### GPU (strongly recommended)
- VRAM needed: 16+ GB for SAO Small, 24+ GB for SAO 1.0

## Step 0: Validation Gate

**Goal:** Confirm that the chosen rater (SoundAQnet or CLAP ridge) is reliable.

```bash
python generation_b/00_validate_rater.py \
    --isd data/processed/isd.csv \
    --out generation_b
```

**What to implement:**
- Load SoundAQnet model from SoundSCaper/
- Load audio from ISD recordings
- Run predictions at full length, 5s, 10s cuts, and ±10 dB levels
- Compute Pearson r with human ISOPleasant ratings
- Decision: use SoundAQnet if r >= 0.37 and stable; otherwise use CLAP ridge

## Steps 1–4: Core Pipeline

- **01_select.py:** Filter FSD50K to soundscape-like clips (8–12k clips)
- **02_rate.py:** Apply rater to get ISOPleasant predictions
- **03_captions.py:** Generate "{scene} with {sources}. {pleasantness} soundscape." captions
- **04_export.py:** Normalize audio to −23 LUFS, resample to 44.1 kHz, split train/test

Expected time: 4–5 days (mostly downloading FSD50K and running inference)

## Steps 5–7: Fine-tuning & Evaluation

- **05_generate.py:** Generate audio at 5 pleasantness levels (planned)
- **06_evaluate.py:** Automatic metrics (FAD, CLAP score, Spearman within-prompt)
- **07_listening_set.py:** Human listening test (essential for plan B)

## Key Differences from Plan A

| Aspect | Plan A | Plan B |
|--------|--------|--------|
| Data | ARAUS + ISD (240 + 823 clips) | FSD50K (8–12k clips) |
| Labels | Human pleasantness ratings | SoundAQnet predictions |
| Acoustic variety | Limited (urban, lab) | High (Freesound dataset) |
| Level calibration | Yes (calibrated recordings) | No (digital level arbitrary) |
| Rater known shortcut | Balanced (psychoacoustic features) | Source recognition bias |
| Implementation effort | Moderate | High (data processing) |
| Evaluation time | 1–2 weeks | 2–3 weeks |

## Common Issues & Solutions

### Issue: Most ISOPleasant ratings near zero (|x| < 0.2 for >60%)
**Solution:** Use quantile bins (quintiles) instead of fixed thresholds in step 3.

### Issue: SoundAQnet r on ISD < 0.30
**Solution:** Switch to CLAP ridge model (r = 0.37), or stop plan B if CLAP also < 0.30.

### Issue: FSD50K audio loading fails
**Solution:** Check audio files exist and are readable with ffmpeg. Re-download if corrupted.

### Issue: Out of memory during fine-tuning
**Solution:** Reduce batch_size (4 → 2), increase gradient_accumulation_steps (2 → 4) in config.

### Issue: Generated audio doesn't vary by pleasantness level
**Solution:** Check captions are correctly formatted with pleasantness words. Run control experiment (same data, no pleasantness word) to isolate the effect.

## Quality Checklist

- [ ] Step 0: Rater r >= 0.30 on held-out ISD
- [ ] Step 1: >= 5,000 soundscape-like clips selected
- [ ] Step 2: >= 98% of clips rated successfully
- [ ] Step 3: 20 random captions are readable and cover all 5 pleasantness levels
- [ ] Step 4: Audio normalized to −23 ± 1 LUFS, resampled to 44.1 kHz
- [ ] Step 5: Training loss smoothly decreasing, no NaN
- [ ] Step 6: CLAP score comparable to or better than baseline
- [ ] Step 7: Human listening test shows pleasantness effect (p < 0.05)

## Expected Outputs

```
generation_b/
├── validation_results.json       # Rater performance, decision rule verdict
├── fsd50k_selected.csv           # 8–12k clips selected
├── ratings.csv                   # ISOPleasant predictions for each clip
├── captions.csv                  # Pleasantness-conditioned captions
├── train/ and test/              # Normalized audio files
├── splits.json and metadata.json # Train/test split metadata
├── checkpoints/                  # Fine-tuned SAO model
├── samples/                      # Generated audio (5 levels × N prompts)
├── results/                      # Evaluation metrics
└── listening_test.json           # Human listening test results
```

## References

- **Plan:** docs/plan_generation_fsd50k.md
- **Plan A:** generation/ directory, PHASE_1_SUMMARY.md
- **SoundAQnet:** github.com/Yuanbo2020/SoundSCaper (Hou et al., 2024)
- **FSD50K:** zenodo.org/record/4060432 (Fonseca et al., 2022)
- **SAO:** github.com/stability-ai/stable-audio-tools (Evans et al., 2024)
