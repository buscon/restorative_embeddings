#!/usr/bin/env bash
# Blind listening playlists for the common prompt set (isd/14_common_prompts.sh): same prompts, seeds and levels on all models.
#   pairs   : -0.7 against +0.7 for each model and the base model, mixed and shuffled (tests whether the rating changes anything)
#   models  : for each prompt and level, the five candidates (ARAUS, ISD, joint, Clotho, base) in random order to rank by
#             pleasantness (tests which model sounds more pleasant on the same text); levels $MODEL_LEVELS
# Output: $LISTEN/common/pairs and $LISTEN/common/models (playlist.m3u, key.csv = answers, responses.csv, audio/).
#   bash isd/15_common_playlists.sh       SEEDS="1 2" bash isd/15_common_playlists.sh      LUFS=-23 ...  to equalise loudness
set -euo pipefail
REPO="${REPO:-$HOME/Documents/restorative_embeddings}"
OUTROOT="${OUTROOT:-$HOME/grids_3msde}"
LISTEN="${LISTEN:-$HOME/listen_g12}"
SEEDS="${SEEDS:-1}"
MODEL_LEVELS="${MODEL_LEVELS:-0.7}"
PY="${PY:-python3}"
cd "$REPO"
LUFS_ARG=(); [ -n "${LUFS:-}" ] && LUFS_ARG=(--lufs "$LUFS")
G=(); for n in common_araus_6000_g12 common_numbers_5000_g12 common_joint_5000_g12 common_clotho_5000_g12 common_base_g12; do
  [ -e "$OUTROOT/$n/grid.csv" ] || { echo "!! $OUTROOT/$n/grid.csv missing"; exit 1; }; G+=("$OUTROOT/$n"); done
$PY generation/make_listening_playlists.py pairs  --grids "${G[@]}" --out "$LISTEN/common/pairs" \
    --seeds $SEEDS --lo -0.7 --hi 0.7 --rng 1 "${LUFS_ARG[@]}"
$PY generation/make_listening_playlists.py models --grids "${G[@]}" --out "$LISTEN/common/models" \
    --seeds $SEEDS --levels $MODEL_LEVELS --rng 1 "${LUFS_ARG[@]}"
