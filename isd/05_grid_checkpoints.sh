#!/usr/bin/env bash
# Grids for several checkpoints of the all-ISD model (captions without a city).
# For each step: full range (-0.7 0 0.7, 6 seeds), positive only (0.1 0.4 0.7) and
# negative only (-0.7 -0.4 -0.1), 4 seeds each; 3 phrases. 162 files per checkpoint.
# Existing files are skipped, so an interrupted run can be restarted.
#
#   source ~/Documents/stable-audio-tools/.venv/bin/activate
#   bash isd/05_grid_checkpoints.sh 6000 12000 19000
#   STEPS_DIR=... RUN=... GRIDS=... are overridable
set -uo pipefail
REPO="${REPO:-$HOME/Documents/restorative_embeddings}"
CK_DIR="${CK_DIR:-$(ls -d $HOME/stableaudio/runs_isd/isd_all_balanced/*/checkpoints | head -1)}"
GRIDS="${GRIDS:-$HOME/grids}"
CONFIG="${CONFIG:-$REPO/configs/model_config.json}"
STEPS=("$@"); [ ${#STEPS[@]} -eq 0 ] && STEPS=(6000 12000 19000)
ITEMS=("soundscape with human sounds" "soundscape with natural sounds" "soundscape with traffic noise")
TEMPLATE="{item} [ISOPleasant: {p}]"

for s in "${STEPS[@]}"; do
  ck=$(ls "$CK_DIR"/epoch=*-step=${s}.ckpt 2>/dev/null | head -1)
  [ -z "$ck" ] && { echo "no checkpoint for step $s in $CK_DIR"; continue; }
  echo "=== step $s: $ck"
  run() {  # name, seeds, levels...
    local name=$1 seeds=$2; shift 2
    python "$REPO/generation/generate_grid.py" --ckpt "$ck" --config "$CONFIG" \
      --out "$GRIDS/isd_all_${s}_$name" --seeds $seeds --levels "$@" \
      --maskers "${ITEMS[@]}" --prompt-template "$TEMPLATE" || return 1
  }
  run full "1 2 3 4 5 6" -0.7 0.0 0.7 || break
  run pos  "1 2 3 4"     0.1 0.4 0.7  || break
  run neg  "1 2 3 4"     -0.7 -0.4 -0.1 || break
done

for s in "${STEPS[@]}"; do
  for name in full pos neg; do
    d="$GRIDS/isd_all_${s}_$name"
    [ -d "$d" ] && { echo; echo "##### step $s $name"; python "$REPO/generation/evaluate_grid.py" "$d"; }
  done
done 2>&1 | tee "$GRIDS/isd_all_eval.txt"
