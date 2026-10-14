#!/usr/bin/env bash
# Word-label range grids for the ISD model trained with isd/08_make_words_set.py, plus the base-model control.
# Same settings as isd/06_range_det_checkpoints.sh (7 prompts x seeds 1-3, sampler dpmpp-2m), but the five levels
# -0.7 -0.35 0 0.35 0.7 are written as "very unpleasant ... very pleasant". Output folders contain the wavs,
# grid.csv, prompts.csv, evaluation.txt; compare them with generation/compare_grids.py.
#
#   source ~/Documents/stable-audio-tools/.venv/bin/activate
#   cd ~/Documents/restorative_embeddings && git pull
#   bash isd/09_words_grid.sh 3000 5000          # checkpoints of the words run
#   bash isd/09_words_grid.sh base               # base model with the word tags (control, no training)
#   SECS=30 is the default here (whole-file models)
set -euo pipefail
REPO="${REPO:-$HOME/Documents/restorative_embeddings}"
CKPT_DIR="${CKPT_DIR:-$(ls -d $HOME/stableaudio/runs_isd_words/isd_words_whole/*/checkpoints 2>/dev/null | head -1)}"
BASE="${BASE:-$HOME/stableaudio/models/stabilityai__stable-audio-open-1.0/model.safetensors}"
CONFIG="${CONFIG:-$REPO/configs/model_config.json}"
OUTROOT="${OUTROOT:-$HOME/grids_isd_words}"
SECS="${SECS:-30}"
cd "$REPO"; mkdir -p "$OUTROOT"
for S in "$@"; do
  if [ "$S" = "base" ]; then CK="$BASE"; OUT="$OUTROOT/base_with_words"
  else
    CK=$(ls "$CKPT_DIR"/epoch=*-step=${S}.ckpt 2>/dev/null | head -1 || true)
    OUT="$OUTROOT/isd_words_${S}"
  fi
  if [ -z "$CK" ] || [ ! -e "$CK" ]; then echo "no checkpoint for '$S' (looked in $CKPT_DIR)"; continue; fi
  echo "=== $S: $CK -> $OUT"
  python generation/generate_prompt_range.py --labels \
      --ckpt "$CK" --config "$CONFIG" --out "$OUT" \
      --sampler dpmpp-2m --seconds "$SECS" --evaluate 2>&1 | tee -a "$OUTROOT/log_${S}.txt"
done
