# Plan B: Fine-tuning Stable Audio Open with FSD50K/AudioSet and SoundAQnet

This directory implements Plan B from the research proposal: fine-tuning Stable Audio Open (SAO) on large general sound datasets (FSD50K and/or AudioSet) with human content labels and **predicted** pleasantness ratings from SoundAQnet.

See `docs/plan_generation_fsd50k.md` for the full plan and rationale.

## Quick Start

```bash
# Step 0: Validate the rater (SoundAQnet or CLAP ridge)
python generation_b/00_validate_rater.py --isd data/processed/isd.csv --out generation_b

# Step 1: Select soundscape-like clips from FSD50K
python generation_b/01_select.py --fsd50k /path/to/FSD50K --out generation_b

# Step 2: Rate selected clips
python generation_b/02_rate.py --selected generation_b/fsd50k_selected.csv --out generation_b

# Step 3: Generate captions
python generation_b/03_captions.py --ratings generation_b/ratings.csv --selected generation_b/fsd50k_selected.csv --out generation_b

# Step 4: Export audio for fine-tuning
python generation_b/04_export.py --captions generation_b/captions.csv --audio-root /path/to/FSD50K/audio --out generation_b

# Step 5–7: Fine-tuning, generation, evaluation (see train/ directory)
```

## Steps

### 00_validate_rater.py
**Validation gate** for the chosen rater before rating any FSD50K clips.

Tests:
- Pearson r on held-out ISD recordings (calibrated, ~7–10 s)
- Length sensitivity: 5 s, 10 s cuts
- Level sensitivity: ±20 dB, ±10 dB rescaling
- Comparison with milestone-1 CLAP ridge model

**Decision rule (fixed before running):**
- Use SoundAQnet if: r_full >= 0.37 (CLAP baseline) AND stable under 10 s cut and ±10 dB changes
- Otherwise use CLAP ridge model
- Stop plan B if neither reaches r ≈ 0.3 at 10 s (labels too noisy)

**Output:** `validation_results.json`

### 01_select.py
Filter FSD50K (primary) and/or AudioSet to soundscape-relevant content.

Filters:
- **License:** CC0, CC-BY only (drop CC-BY-NC unless `--keep-nc`)
- **Length:** >= 5 s (pilot SAO Small generates up to ~11 s)
- **Content:** AudioSet ontology classes for soundscapes
  - Keep: natural sounds, animals/birds, traffic, crowds, footsteps, construction, domestic ambiences
  - Drop: music, instruments, speech-dominant, isolated foley

Also creates `selection_criteria.json` listing all classes.

**Output:** `fsd50k_selected.csv`, `audioset_selected.csv` (optional), `selection_criteria.json`

### 02_rate.py
Apply the validated rater (SoundAQnet or CLAP ridge) to all selected clips.

Generates:
- ISOPleasant, ISOEventful (SoundAQnet)
- Or ISOPleasant only (CLAP ridge, no level input)
- 8 PAQ items, loudness, metadata

Checks distribution: if most ratings compress near neutral (|x| < 0.2), plan B should use quantile bins instead of fixed thresholds.

**Output:** `ratings.csv`, `rating_failures.json`

### 03_captions.py
Create captions from FSD50K labels, scene (placeholder), and pleasantness ratings.

Template: `"{scene} with {sources}. {pleasantness} soundscape."`

Choices:
- Fixed bins (five thresholds from plan A) or quantile bins (quintiles on this dataset)
- Scene from SoundAQnet (limited to 3 classes) or derived from sources
- Pleasantness label mapped from ISOPleasant score

Reviews 20 random captions (interactive check) before proceeding.

**Output:** `captions.csv`, `distribution.json`

### 04_export.py
Prepare audio for SAO fine-tuning:
- Normalize to −23 LUFS (target loudness)
- Resample to 44.1 kHz
- Convert to stereo (mono → duplicate)
- Split into train/test by uploader (avoids near-duplicates)
- Optional: precompute SAO latent embeddings

**Output:** `train/`, `test/` directories with audio; `splits.json`, `metadata.json`

### 05_generate.py (planned)
Generate audio samples from fine-tuned SAO at five pleasantness levels using held-out prompts.

### 06_evaluate.py (planned)
Automatic evaluation (within-prompt Spearman, FAD, prompt adherence, source shortcut check).

### 07_listening_set.py (planned)
Create blinded listening test combining base SAO, plan A, and plan B samples.

## Data and Prerequisites

### FSD50K
Download from Zenodo (4060432):
```
FSD50K/
  ├── fsd50k.all.csv (metadata)
  ├── fsd50k.eval_audio.csv
  ├── dev_audio/
  ├── eval_audio/
  └── test_audio/ (optional)
```

~43k clips with CC0/CC-BY licenses; ~51k total. Typical length 7–10 s.

### AudioSet
From GitHub (gstax/AudioSet):
- Weak labels (clip-level, often incomplete)
- 10 s YouTube clips (download required; many unavailable)
- **Legal note:** Confirm with your university before using YouTube content for training

### SoundAQnet
From github.com/Yuanbo2020/SoundSCaper:
```bash
git clone https://github.com/Yuanbo2020/SoundSCaper.git
pip install -r SoundSCaper/requirements.txt
```

Requires:
- Pretrained models (included in repo)
- Input: log-mel spectrogram, ISO 532-1 loudness
- Output: 8 PAQ values, ISOPleasant, ISOEventful, 3 scene classes

### Stable Audio Open
From github.com/stability-ai/stable-audio-tools:
```bash
git clone https://github.com/stability-ai/stable-audio-tools.git
pip install stable-audio-tools
```

Models: SAO Small (11 s max), SAO 1.0 (30 s max)

## Implementation Notes

### Level Calibration
FSD50K and AudioSet are **uncalibrated** (digital level is arbitrary). SoundAQnet needs ISO 532-1 loudness as input.

Options:
1. **Assume fixed playback level** (e.g., −23 LUFS for all): mentioned in captions, or
2. **Normalize all clips** to one level before rating, or
3. **Use CLAP ridge** instead (no level input, but reads mostly sources)

This is decided in step 0 validation.

### The "Source Shortcut"
Both raters (SoundAQnet and CLAP ridge) are largely source-recognizers. If pleasantness labels come from a model that reads sources, fine-tuned SAO learns "pleasant = these sources," and within-prompt variation (same scene at different levels) is harder to achieve.

Mitigations:
- Human listening test (essential for plan B)
- Check plan A vs. plan B on the same prompts (direct comparison)
- Hybrid approach: FSD50K for variety + ARAUS/ISD for pleasantness words

### Clip Length
SAO Small generates up to ~11 s. FSD50K clips average 7–10 s, AudioSet 10 s. Longer clips fit better.

## Decisions to Make

1. **Rater:** SoundAQnet or CLAP ridge (step 0 decides)
2. **Datasets:** FSD50K only or also AudioSet?
3. **Licenses:** Keep CC-BY-NC clips for non-commercial research?
4. **Minimum length:** 5 s (SAO Small) or 10 s (SAO 1.0)?
5. **Pleasantness bins:** Fixed thresholds or quantile-based (per-dataset)?
6. **Run plan B alone or alongside plan A with shared evaluation?**

## Caveats

- Pleasantness labels are **model predictions**, not human judgements, and inherit the rater's training data biases (e.g., SoundAQnet trained on ARAUS: lab, headphones, urban scenes).
- Audio is **uncalibrated**; level, a main driver of pleasantness, may be missing.
- **Source shortcut by construction:** the rater reads mostly sources, so "pleasant" may mean "more natural sounds."
- Probable **overlap between FSD50K and SAO's pretraining** (both use Freesound).
- **Only a human listening test** can show whether generated pleasantness levels are perceived as intended.

## References

- Hou et al. (2024). Soundscape captioning using SoundAQnet and LLM. arXiv:2406.05914. Code: github.com/Yuanbo2020/SoundSCaper
- Fonseca et al. (2022). FSD50K: an open dataset of human-labeled sound events. IEEE/ACM TASLP. arXiv:2010.00475. Data: Zenodo 4060432
- Gemmeke et al. (2017). Audio Set: an ontology and human-labeled dataset. ICASSP
- Evans et al. (2024). Stable Audio Open. arXiv:2407.14358
- Ooi et al. (2023). ARAUS. arXiv:2207.01078
- Mitchell et al. International Soundscape Database v1.0. Zenodo 10672568

## Status

- ✓ Step 0 (template)
- ✓ Step 1 (template)
- ✓ Step 2 (template)
- ✓ Step 3 (template)
- ✓ Step 4 (template)
- ☐ Step 5–7 (to implement)

Each script provides a working template with placeholders for external libraries (librosa, SoundAQnet, SAO). Fill in the audio processing and model integration as needed.
