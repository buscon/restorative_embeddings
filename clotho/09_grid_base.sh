#!/usr/bin/env bash
# Base-model reference grids for the Clotho test: the same prompts, levels and seeds as the fine-tuned grids.
#   with_tag  prompts "<caption> [ISOPleasant: x]"  (reference for the rated run)
#   no_tag    prompts "<caption>"                   (reference for the control run; the levels only repeat the same prompt)
# Run inside the stable-audio-tools venv, from the repo folder:
#   source ~/stableaudio/.venv/bin/activate && cd ~/Documents/restorative_embeddings
#   bash clotho/09_grid_base.sh
# Override by env var: BASE (base weights), CONFIG, OUT, SEEDS, LEVELS. Existing files are skipped, so a rerun resumes.
set -euo pipefail
BASE="${BASE:-$HOME/stableaudio/models/stabilityai__stable-audio-open-1.0/model.safetensors}"
CONFIG="${CONFIG:-configs/model_config.json}"
OUT="${OUT:-$HOME/grids}"
SEEDS="${SEEDS:-1 2 3 4 5 6}"
LEVELS="${LEVELS:--0.4 -0.2 0.0 0.2 0.4}"

CAPS=("The wind is blowing and the waves are flowing"
      "It is a busy day in the city and public transportation is everywhere"
      "Cicadas are chirping in the woodland area in a consistent fashion while the wind blows")

# shellcheck disable=SC2086
python generation/generate_grid.py --ckpt "$BASE" --config "$CONFIG" --out "$OUT/clotho_base_with_tag" \
  --maskers "${CAPS[@]}" --prompt-template "{item} [ISOPleasant: {p}]" --levels $LEVELS --seeds $SEEDS
python generation/evaluate_grid.py "$OUT/clotho_base_with_tag"

# The no-tag prompt does not depend on the level: one level is enough (the same prompt and seed give the same file).
# shellcheck disable=SC2086
python generation/generate_grid.py --ckpt "$BASE" --config "$CONFIG" --out "$OUT/clotho_base_no_tag" \
  --maskers "${CAPS[@]}" --prompt-template "{item}" --levels 0.0 --seeds $SEEDS
echo "done: $OUT/clotho_base_with_tag and $OUT/clotho_base_no_tag"
