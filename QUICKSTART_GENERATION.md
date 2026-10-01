# Quick Start: Generation Pipeline

## One-Time Setup
```bash
# Install pyloudnorm (if not already done)
pip install --break-system-packages pyloudnorm
```

## Run Full Pipeline (5 minutes)
```bash
# From repo root, with ARAUS and ISD already downloaded/available:

# 1. Prepare data (harmonizes ARAUS + ISD into common format)
python scripts/01_prepare.py \
  --araus data/raw/araus \
  --isd data/raw/isd \
  --out data/processed

# 2. Map scenes (joins soundscapes with USotW metadata, annotates ISD locations)
python generation/03_scene_mapping.py \
  --araus data/raw/araus \
  --processed data/processed \
  --out data/generation

# 3. Generate all captions (~22k ARAUS + ~800 ISD)
python generation/01_captions.py \
  --processed data/processed \
  --out data/generation

# 4. Review checkpoint: sample 50 captions
python generation/04_review_captions.py \
  --captions data/generation/captions.csv \
  --out data/generation

# 5. Open captions_review_sample.tsv and review against checklist
#    (expected location: data/generation/captions_review_sample.tsv)

# 6. Once approved, export 6k stratified ARAUS subset + audio (20-30 minutes)
python generation/02_export.py \
  --araus data/raw/araus \
  --processed data/processed \
  --captions data/generation/captions.csv \
  --out data/generation \
  --n-samples 6000

# Done! Check outputs:
#   data/generation/captions.csv          (all captions)
#   data/generation/export_metadata.csv   (6k ARAUS with audio paths)
#   data/generation/audio/                (6k stereo WAVs at 44.1 kHz, -23 LUFS)
```

## Outputs

| File | Purpose | Size |
|------|---------|------|
| `captions.csv` | All caption text + ISOPleasant bins | ~200 KB |
| `captions_review_sample.tsv` | 50 sampled captions for manual review | ~3 KB |
| `export_metadata.csv` | 6k stimuli with wav paths + captions | ~200 KB |
| `araus_scenes.csv` | Soundscape → scene description mapping | ~2 KB |
| `isd_scenes.csv` | Location → scene category mapping | <1 KB |
| `audio/araus_XXXXXXX.wav` | 6,000 stereo WAV files (6.4 GB total) | 1.1 MB each |

## Review Checklist

When `captions_review_sample.tsv` opens:

- [ ] All captions follow the template: "{scene}, {source}, {loudness}. {pleasantness} soundscape."
- [ ] Pleasantness labels align with ISOPleasant values (−0.30 → "unpleasant", +0.30 → "pleasant")
- [ ] Source descriptions vary by masker (birds, water, traffic, construction)
- [ ] Loudness descriptions match LA50 bins (quiet <50, loud >70)
- [ ] No duplicate captions or near-duplicates within bins
- [ ] Each caption reads naturally and makes semantic sense

**Gate**: If >90% pass, proceed to training. Otherwise, request fixes.

## Troubleshooting

**"ModuleNotFoundError: No module named 'pyloudnorm'"**
```bash
pip install --break-system-packages pyloudnorm
```

**"FileNotFoundError: 'ISD v1.0 Data.csv' not found"**
Make sure `data/raw/isd` points to the extracted ISD archive with the CSV at the root.

**"KeyError: 'gain_59dB'"**
This indicates malformed masker metadata in ARAUS. Ensure `data/raw/araus/data/maskers.csv` has gain columns for all dB values 46–84.

**Export takes >30 minutes**
Increase `--workers` in `02_export.py` or run on GPU (if available). Current implementation is CPU-bound on audio mixing.

---

See `PHASE_1_SUMMARY.md` for design rationale and Phase 2 plans.
