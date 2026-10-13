#!/usr/bin/env bash
# Deterministic range grids (7 prompts x seeds 1-3 x levels -0.7..0.7) for further ISD checkpoints,
# same settings as ~/grids/isd_all_6000_range_det (sampler dpmpp-2m), so the folders are comparable.
# Output: ~/grids/isd_all_<step>_range_det with the wavs, grid.csv, prompts.csv, metrics.csv, evaluation.txt.
#
#   source ~/Documents/stable-audio-tools/.venv/bin/activate
#   cd ~/Documents/restorative_embeddings && git pull
#   bash isd/06_range_det_checkpoints.sh            # steps 4000 and 8000
#   bash isd/06_range_det_checkpoints.sh 3000 5000  # or other steps
set -euo pipefail
REPO="${REPO:-$HOME/Documents/restorative_embeddings}"
CKPT_DIR="${CKPT_DIR:-$HOME/stableaudio/runs_isd/isd_all_balanced/9185hkux/checkpoints}"
CONFIG="${CONFIG:-$REPO/configs/model_config.json}"
OUTROOT="${OUTROOT:-$HOME/grids}"   # use a new folder for a new run, existing files are skipped
LEVELS="${LEVELS:--0.7 -0.4 -0.1 0.1 0.4 0.7}"   # pleasantness numbers, e.g. LEVELS="-0.7 0.0 0.7"
SECS="${SECS:-10}"   # length of the generated clips; use 30 for the whole-file models
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(4000 8000)

cd "$REPO"; mkdir -p "$OUTROOT"
for S in "${STEPS[@]}"; do
  CK=$(ls "$CKPT_DIR"/epoch=*-step=${S}.ckpt 2>/dev/null | head -1 || true)
  if [ -z "$CK" ]; then echo "no checkpoint for step $S in $CKPT_DIR"; continue; fi
  OUT="$OUTROOT/isd_all_${S}_range_det"
  echo "=== step $S: $CK -> $OUT"
  python generation/generate_prompt_range.py \
      --ckpt "$CK" --config "$CONFIG" --out "$OUT" \
      --levels $LEVELS --sampler dpmpp-2m --seconds "$SECS" --evaluate 2>&1 | tee -a "$OUTROOT/isd_all_range_det_${S}.log"
done
