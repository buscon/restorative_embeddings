# Restorative soundscape generation (ARAUS -> Stable Audio Open)

Fine-tune Stable Audio Open 1.0 on ARAUS soundscapes whose captions carry the rated
pleasantness, then test whether generation responds to that number.

Caption format (defined in `rsd/captions.py`, used for training and generation):

    park soundscape with flowing water in the foreground [ISOPleasant: 0.50]

The scene text comes from the masker type (birdsong, construction noise, traffic noise,
flowing water, wind) and the sign of the SMR (negative = masker louder = "foreground").

## Overview of the steps

| # | What | Script | Output | Env |
|---|------|--------|--------|-----|
| 1 | Download ARAUS (and ISD) | `scripts/download_araus.sh`, `download_isd.sh` | `data/raw/` | A |
| 2 | Build tables | `scripts/01_prepare.py` | `data/processed/*.csv` | A |
| 3 | Scene text per soundscape | `generation/03_scene_mapping.py` | `data/generation/araus_scenes.csv` | A |
| 4 | Captions | `generation/01_captions.py` | `data/generation/captions.csv` | A |
| 5 | Rebuild + export audio | `generation/02_export.py` | `data/generation/audio/*.wav`, `export_metadata.csv` | A |
| 6 | Pick a balanced subset | `generation/train/01_select_balanced_dataset.py` | `generation/train/selected_360.csv` | A |
| 7 | Cut 10 s chunks + training captions | `generation/train/02_make_training_chunks.py` | `generation/train/araus_for_sa3_360/` (wavs, `metadata.csv`, `dataset_config.json`) | A |
| 8 | Fine-tune | `generation/train/03_train.sh` | checkpoints (`.ckpt`) | B |
| 9 | Generate one file | `generation/generate.py` | wav | B |
| 10 | Generate the grid | `generation/generate_grid.py` | `out_grid_*/` (wavs, `grid.csv`) | B |
| 11 | Evaluate the grid | `generation/evaluate_grid.py` | `metrics.csv` + printed summary | B (or A) |

Env A = this repo's `.venv` (Python 3.10, `requirements.txt`). Env B = a separate
stable-audio-tools venv with a CUDA PyTorch.

What each data step does:

- **2** reads the ARAUS and ISD files and writes the stimulus and response tables.
  It loads ISD unconditionally, so ISD must be downloaded even though only ARAUS is
  used for training.
- **3, 4** produce `captions.csv` (scene text plus loudness/pleasantness wording). Step 5
  needs this file. These captions are **not** the training captions; those are rebuilt in
  step 7 in the short `[ISOPleasant: x]` format.
- **5** rebuilds each ARAUS stimulus from soundscape + masker at the stored SMR
  (`rsd/audio.py`, same formula as the ARAUS authors), keeps stereo, resamples once to
  44.1 kHz, applies a 30 Hz high-pass (removes infrasound), normalises to -23 LUFS and
  writes 16-bit wavs. The default is 6000 stimuli stratified by pleasantness bin and
  masker type (roughly 30 GB).
- **6** samples 360 of them proportionally to the pleasantness bins (rounding gives about
  358 files).
- **7** cuts each file into complete 10 s chunks (2 or 3 per file), writes
  `metadata.csv` (`file,caption`) and a `dataset_config.json` with absolute paths for the
  machine it runs on.
- **8** full fine-tune from the base weights with stable-audio-tools. Earlier runs used
  batch size 1, 8 accumulation batches, learning rate 5e-5, a checkpoint every 500 steps,
  T5-base text conditioning.
- **10, 11** generate every combination of seed x masker x pleasantness level (default 4 x 5 x 3
  = 60 files) with one loaded model, then compute spectral/loudness metrics per file, the
  mean per masker and level, and per (masker, seed) Spearman correlations with the
  pleasantness number.

## Running everything on a fresh server

```bash
git clone <repo-url> ~/Documents/restorative_embeddings
cd ~/Documents/restorative_embeddings
```

### A. Data (repo venv)

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash scripts/download_araus.sh                      # ~3 GB into data/raw/araus
SKIP_LOCKDOWN=1 bash scripts/download_isd.sh        # data/raw/isd, needed by step 2

python scripts/01_prepare.py --araus data/raw/araus --isd data/raw/isd --out data/processed
python generation/03_scene_mapping.py --araus data/raw/araus --processed data/processed --out data/generation
python generation/01_captions.py --processed data/processed --out data/generation
python generation/02_export.py --araus data/raw/araus --processed data/processed \
    --captions data/generation/captions.csv --out data/generation --workers 8

python generation/train/01_select_balanced_dataset.py \
    --metadata data/generation/export_metadata.csv \
    --output-csv generation/train/selected_360.csv --num-samples 360 --seed 42
python generation/train/02_make_training_chunks.py \
    --selected generation/train/selected_360.csv \
    --audio-dir data/generation/audio \
    --out generation/train/araus_for_sa3_360
```

Checks: `ls data/generation/audio | wc -l` should be close to 6000 (failures are listed in
`data/generation/export_errors.csv`); `wc -l generation/train/araus_for_sa3_360/metadata.csv`
should be roughly 700 to 1100; `head generation/train/araus_for_sa3_360/metadata.csv` should
show captions like the example above.

### B. Model, training, generation (stable-audio-tools venv)

One-time setup. The base model is gated on Hugging Face: accept the licence for
`stabilityai/stable-audio-open-1.0`, then log in.

```bash
git clone https://github.com/Stability-AI/stable-audio-tools ~/Documents/stable-audio-tools
python3 -m venv ~/stableaudio/.venv && source ~/stableaudio/.venv/bin/activate
# RTX 5090 (sm_120) needed the cu128 wheels; adjust for your GPU
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install ~/Documents/stable-audio-tools
pip install "huggingface_hub[cli]" wandb
huggingface-cli login
mkdir -p ~/stableaudio/models/stabilityai__stable-audio-open-1.0
huggingface-cli download stabilityai/stable-audio-open-1.0 model.safetensors \
    --local-dir ~/stableaudio/models/stabilityai__stable-audio-open-1.0
```

**Model config.** `configs/model_config.json` must be the full SAO 1.0 config (architecture,
T5-base conditioning, and a `training` section with `learning_rate`) that was used for the
earlier runs. It is not generated by any script here. Copy the working one into the repo
once and commit it, so a fresh clone is self-contained:

```bash
cp <path to your working model_config.json> ~/Documents/restorative_embeddings/configs/model_config.json
```

Train (from the stable-audio-tools checkout; log in to W&B or set `WANDB_MODE=offline`):

```bash
cd ~/Documents/stable-audio-tools
bash ~/Documents/restorative_embeddings/generation/train/03_train.sh
# resume: RESUME=<last .ckpt> bash .../03_train.sh
```

Generate and evaluate (from the repo; `--ckpt` takes the base `model.safetensors` or a
Lightning `.ckpt`):

```bash
cd ~/Documents/restorative_embeddings
CKPT=<path to .ckpt>
python generation/generate.py --ckpt $CKPT --config configs/model_config.json \
    --prompt "park soundscape with flowing water in the background [ISOPleasant: 0.50]" --output test.wav
python generation/generate_grid.py --ckpt $CKPT --config configs/model_config.json --out out_grid_1500
python generation/evaluate_grid.py out_grid_1500 --compare out_grid_1000   # compare is optional
```

`generate.py` takes its seed from the `SEED` constant inside the script. `generate_grid.py`
takes `--seeds`, `--levels`, `--maskers`, `--position foreground|background`, and skips
files that already exist. For the base-model baseline use the same grid command with
`--ckpt model.safetensors`.


## Variant: fine-tune on ISD instead of ARAUS

ISD (in-situ recordings from London, Venice, Granada and Groningen, rated by the people
who were there) is processed in its own folder, `isd/`. It does not need the ARAUS data, the
steps 2 to 7 above, or `01_prepare.py`.

| # | What | Script | Output |
|---|------|--------|--------|
| 1 | Download ISD (without the unrated lockdown archives) | `isd/01_download.sh` | `data/raw/isd/` |
| 2 | Balanced selection | `isd/02_select_balanced.py` | `isd/selected_isd.csv` |
| 3 | Condition + cut into 10 s chunks + captions | `isd/03_make_training_chunks.py` | `isd/train/isd_for_sao/` (wavs, `metadata.csv`, `dataset_config.json`) |
| 4 | Fine-tune | `isd/04_train.sh` | checkpoints in `~/stableaudio/runs_isd` |

```bash
cd ~/Documents/restorative_embeddings && source .venv/bin/activate
bash isd/01_download.sh
python isd/02_select_balanced.py --isd data/raw/isd --output-csv isd/selected_isd.csv --num-samples 360
python isd/03_make_training_chunks.py --selected isd/selected_isd.csv --out isd/train/isd_for_sao
# then, in the stable-audio-tools venv, from the stable-audio-tools checkout:
bash ~/Documents/restorative_embeddings/isd/04_train.sh
```

- **Selection:** as equal as availability allows over the five ISOPleasant bins (very
  unpleasant to very pleasant); within each bin as equal as possible over the four cities; and
  within a city taking recordings in turn from its locations. When a bin or city has too few
  recordings, the others make up the difference, and the script prints what was available and
  what was taken. ISD is skewed towards pleasant recordings (only about 19 very unpleasant ones,
  all from London), so the unpleasant end will be thin whatever the settings. `--primary city`
  balances cities first instead. `--min-ratings 2` keeps only recordings rated by at least two
  people. The ISOPleasant of a recording is the mean over its raters.
- **Audio:** the same conditioning as the ARAUS export (`rsd/conditioning.py`): stereo,
  44.1 kHz, 30 Hz high-pass, -23 LUFS.
- **Captions:** built from what the raters reported hearing. ISD asks how much of four sound
  types people heard (items `ssi01` to `ssi04`: traffic, other noise, human sounds, natural
  sounds; 1 = not at all, 5 = dominates completely). The caption names the sources whose mean
  rating is at least 3.5 (at most two), plus the city, for example
  `soundscape with traffic noise and human sounds in London [ISOPleasant: -0.21]`, or
  `soundscape with no dominant sound source in ...` when none reaches 3.5. `--caption
  content_place` adds the place name, `--caption place` gives place and city only. The
  mapping of the four items was inferred from the data (for example traffic is highest at
  Camden Town and Euston Tap, natural sounds at Regent's Park Japan) and not from the ISD
  metadata workbook, so confirm it there if you rely on it. Prompts for generation must use the
  same form. With `--caption content` the place is not in the caption, so the model cannot
  take the pleasantness from the place name.
- **Grid generation and evaluation** work as before. For an ISD model give the places and a
  template, for example
  `python generation/generate_grid.py --ckpt X.ckpt --config configs/model_config.json --out out_grid_isd --maskers "traffic noise and human sounds" "natural sounds" "other noise" --prompt-template "soundscape with {item} in London [ISOPleasant: {p}]"`
  (one run per city, using source phrases that occur in the training captions).

## Variant: fine-tune on Clotho (planned, selection and caption check only)

Idea: a proof of concept with human-written captions instead of the templated ones. Take
Clotho (Freesound clips of 15 to 30 s, five crowd-written captions each), rate the clips with
SoundAQnet, and add the predicted ISOPleasant to the caption. See `clotho/README.md` for the
numbers and caveats. So far there are the clip selection, audio extraction and a CLAP check
of the captions against the audio; nothing has been rated or trained.

```bash
cd ~/Documents/restorative_embeddings && source .venv/bin/activate
bash clotho/01_download_metadata.sh                 # CSVs only, 3 MB, into data/raw/clotho
python clotho/02_select_candidates.py --show 50     # counts, writes clotho/selected_clotho.csv
bash clotho/03_download_audio.sh                    # about 7 GB, resumable
pip install py7zr && python clotho/04_extract_selected_audio.py
# stable-audio-tools venv (torch + transformers):
python clotho/05_clap_caption_check.py              # best caption per clip -> clotho/checked/
```

## Layout

- `rsd/`: audio mixing (`audio.py`), ARAUS/ISD loading (`data.py`), caption format (`captions.py`)
- `generation/`: scene mapping, captions, export, generation, grid, evaluation
- `generation/train/`: subset selection, chunking, training script, dataset config
- `isd/`: the ISD variant (download, selection, chunking, training)
- `clotho/`: the Clotho variant (metadata, clip selection, audio extraction, CLAP caption check)
- `configs/`: put `model_config.json` here
- `tests/smoke_generation.py`: synthetic test of captions and export
- `archive/`: earlier experiments (CLAP embedding/regression milestone, LoRA and SA3
  attempts, debug scripts); not part of the pipeline

## Status and caveats

- Steps 1 to 7 and 9 to 11 were run in pieces during development; the whole chain on a fresh
  clone has not been run end to end, so expect to fix small path issues on the first pass.
  `03_train.sh` is reconstructed from the settings of the earlier runs, not copied from one.
- `dataset_config_360.json` in `generation/train/` has paths hard-coded to one server
  account; the generated `araus_for_sa3_360/dataset_config.json` is the portable one.
- The grid metrics are descriptive spectral and loudness measures, not perceptual ratings.
  With three levels per group, a trend with the pleasantness number must be larger than the
  seed-to-seed spread and be confirmed by listening before it is read as an effect.
