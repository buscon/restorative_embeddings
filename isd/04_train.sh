#!/usr/bin/env bash
# Fine-tune Stable Audio Open on the ISD chunks. Same settings as the ARAUS run; only the
# dataset config and the run name differ. Run from the stable-audio-tools checkout with its
# venv active (see generation/train/03_train.sh for the full notes, RESUME, BATCH_SIZE, ...).
#
#   source ~/stableaudio/.venv/bin/activate && cd ~/Documents/stable-audio-tools
#   bash ~/Documents/restorative_embeddings/isd/04_train.sh
set -euo pipefail
REPO="${REPO:-$HOME/Documents/restorative_embeddings}"
export DATASET_CONFIG="${DATASET_CONFIG:-$REPO/isd/train/isd_for_sao/dataset_config.json}"
export RUN_NAME="${RUN_NAME:-isd_sao_360}"
export SAVE_DIR="${SAVE_DIR:-$HOME/stableaudio/runs_isd}"
exec bash "$REPO/generation/train/03_train.sh"
