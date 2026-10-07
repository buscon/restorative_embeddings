#!/usr/bin/env bash
# Full fine-tune of Stable Audio Open 1.0 with stable-audio-tools.
# Run from inside the stable-audio-tools checkout, with its venv active:
#   source ~/stableaudio/.venv/bin/activate && cd ~/Documents/stable-audio-tools
#   bash ~/Documents/restorative_embeddings/generation/train/03_train.sh
#
# Needs: configs/model_config.json (full SAO model config with a "training" section),
#        the base weights model.safetensors, and the dataset config written by
#        02_make_training_chunks.py. Logging uses Weights & Biases (run `wandb login`,
#        or export WANDB_MODE=offline).
#
# Settings below are the ones of the earlier runs (batch 1, accumulate 8, checkpoint
# every 500 steps; the learning rate lives in the model config). Override by env var.
#
# Resume an interrupted run (restores optimizer + step counter):
#   RESUME=/path/to/last.ckpt bash 03_train.sh
# Do not use --pretrained-ckpt-path for resuming; it loads weights only.
set -euo pipefail

REPO="${REPO:-$HOME/Documents/restorative_embeddings}"
BASE="${BASE_WEIGHTS:-$HOME/stableaudio/models/stabilityai__stable-audio-open-1.0/model.safetensors}"
DATASET_CONFIG="${DATASET_CONFIG:-$REPO/generation/train/araus_for_sa3_360/dataset_config.json}"

if [ -n "${RESUME:-}" ]; then
  INIT=(--ckpt-path "$RESUME")
else
  INIT=(--pretrained-ckpt-path "$BASE")
fi

python3 train.py \
  --dataset-config "$DATASET_CONFIG" \
  --model-config   "$REPO/configs/model_config.json" \
  --name "${RUN_NAME:-araus_sao_360}" \
  --save-dir "${SAVE_DIR:-$HOME/stableaudio/runs}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --accum-batches "${ACCUM_BATCHES:-8}" \
  --checkpoint-every "${CKPT_EVERY:-500}" \
  "${INIT[@]}"
