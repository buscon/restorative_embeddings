# Phase 1 Complete: Caption Generation & Audio Export Pipeline

**Status: READY FOR REAL-DATA TESTING**

## What Was Built

### Core Modules
1. **`rsd/sources.py`** (97 lines)
   - Extracted CLAP text embedding utilities from scripts/06_source_recognition.py
   - Reusable functions: `text_embeddings()`, `scores()`
   - Defines CATEGORIES (traffic, other_noise, human, natural) and ALT_PHRASES
   - **Used by**: caption generation pipeline

2. **`generation/01_captions.py`** (176 lines)
   - Generates pleasantness-conditioned captions from ARAUS/ISD responses
   - Five pleasantness bins: very_unpleasant (≤−0.50), unpleasant, neutral, pleasant, very_pleasant (≥+0.50)
   - Caption template: `"{scene}, {sources}, {loudness}. {pleasantness} soundscape."`
   - Outputs: `captions.csv` with [id, dataset, caption, ISOPleasant, bin, n_ratings, LA50]
   - **Tested**: ✓ Smoke test generated 4 captions (2 ARAUS + 2 ISD)

3. **`generation/02_export.py`** (216 lines)
   - Exports stratified 6k ARAUS subset with captions and loudness normalization
   - Stratification by pleasantness bin × masker type for balanced diversity
   - Audio processing:
     * Rebuilds ARAUS stimuli in-memory using ArausMixer (no disk writes of 132 GB)
     * Resamples to 44.1 kHz
     * Normalizes to −23 LUFS using pyloudnorm
     * Stereo, 16-bit WAV output
   - Outputs: `audio/araus_XXXXXXX.wav` (6k files) + `export_metadata.csv`
   - **Tested**: ✓ Smoke test exported 2 audio files with correct format

4. **`generation/03_scene_mapping.py`** (84 lines)
   - Joins ARAUS soundscapes with USotW metadata for scene descriptions
   - Hand-annotates ISD locations to scene categories (urban, park, water, etc.)
   - Outputs: `araus_scenes.csv`, `isd_scenes.csv`
   - **Tested**: ✓ Smoke test created mappings for 2 soundscapes + 1 location

5. **`generation/04_review_captions.py`** (79 lines)
   - Samples 50 stratified captions across pleasantness bins
   - Outputs TSV for easy review in Excel/Google Sheets
   - Includes review checklist (templates, bin alignment, diversity, duplicates)

### Test Suite
**`tests/smoke_generation.py`** (215 lines)
- End-to-end validation on synthetic ARAUS + ISD data
- Runs all 4 generation steps in sequence
- Validates output formats and structure
- **Result**: ✓ PASSED (4 minutes on CPU)

## Key Design Decisions

1. **Pleasantness Binning**: Fixed thresholds (−0.50, −0.15, +0.15, +0.50) instead of quantile-based
   - Reason: Allows year-round comparison if labels are pre-registered
   - Tested on smoke data: all 4 generated captions fall in "pleasant" bin (as expected for synthetic data with high ISOPleasant)

2. **Stratified Sampling**: 6k from ~22k ARAUS stimuli (27% of data)
   - Reason: Balances training duration (~42 hours audio), GPU memory, and acoustic diversity
   - Stratification ensures proportional bin representation and masker type variety

3. **Loudness Normalization**: −23 LUFS (broadcast standard)
   - Reason: Industry standard for speech/audio training; removes absolute level bias between ARAUS (playback) and ISD (field)
   - Implemented via pyloudnorm with silent-clip handling

4. **Caption Template**: Consistent 4-part structure
   - Reason: Ensures semantic coherence; easy to extend with additional fields later
   - Example: "Urban park, birds singing and people talking, moderately loud. Pleasant soundscape."

## Next Steps: Running on Real Data

To proceed to the caption review checkpoint:

```bash
# 1. Ensure ARAUS and ISD are prepared:
python scripts/01_prepare.py --araus data/raw/araus --isd data/raw/isd --out data/processed

# 2. Generate scene mappings (joins with USotW metadata if available):
python generation/03_scene_mapping.py --araus data/raw/araus --processed data/processed --out data/generation

# 3. Generate all captions (takes ~1 minute):
python generation/01_captions.py --processed data/processed --out data/generation

# 4. Sample 50 for review (interactive):
python generation/04_review_captions.py --captions data/generation/captions.csv --out data/generation

# 5. Open data/generation/captions_review_sample.tsv in Excel/Google Sheets and review against the checklist
```

## Review Checklist (Before Proceeding)

When you run `04_review_captions.py` and open the TSV:

1. **Templates**: All captions follow format? Grammar correct?
2. **Bin Alignment**: Does pleasantness label match ISOPleasant value? (e.g., −0.30 should be "unpleasant", not "pleasant")
3. **Source Diversity**: Different masker types (bird, water, traffic, construction) produce distinct source descriptions?
4. **Loudness Mapping**: Quiet (<50 LA50) → "quiet"? Loud (>70) → "loud"?
5. **Uniqueness**: Any duplicate captions or near-duplicates within the same bin?
6. **Coherence**: Does each caption read naturally?

**Decision gate**: If >90% pass the checklist, proceed to training. If <90%, document issues and rebuild captions accordingly.

## File Structure

```
data/generation/
├── captions.csv                    (all ARAUS + ISD captions)
├── captions_review_sample.tsv      (50 sampled for review)
├── export_metadata.csv             (6k ARAUS export with WAV paths)
├── araus_scenes.csv                (soundscape → scene description)
├── isd_scenes.csv                  (location → scene category)
└── audio/
    ├── araus_0000000.wav           (exported stereo, 44.1 kHz, −23 LUFS)
    ├── araus_0000001.wav
    └── ... (6k total)
```

## Dependencies Added

- `pyloudnorm>=0.2.0` for loudness normalization
- Existing: scipy, pandas, numpy, soundfile, soundscapy

## Known Limitations

1. **ISD scene mappings are hand-annotated**: Only 26 locations; future versions can use NLP/clustering
2. **Caption template is simple**: No fine-grained source attributes (e.g., "distant traffic" vs "near traffic")
3. **Loudness normalization**: Skipped for silent clips; they pass through unchanged (rare in real data)
4. **Stratification**: Assumes pleasantness bins are meaningful for SAO training; should be validated post-training

## What Comes Next (Phase 2)

Once captions are approved:

- **`training/01_prepare_training_data.py`**: Package 6k stimuli + captions as WebDataset for SAO fine-tuning
- **`training/02_finetune_sao.py`**: Conditional training loop (pleasantness-guided generation)
- **`training/03_evaluate.py`**: Inference on held-out ISD test set + listening study
- **`training/04_hybrid_isd.py`**: Combine SAO generation with source recognition

---

**All Phase 1 code is tested, documented, and ready for your review on real data.**
