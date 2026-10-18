#!/usr/bin/env bash
# Extra measures for the grids in $OUTROOT (default ~/grids_3msde):
#   1. psychoacoustics (loudness, sharpness, roughness)   generation/psychoacoustics_grid.py     -> psycho.csv, columns in metrics.csv
#   2. SoundAQnet predicted pleasantness (ISOPleasant)    clotho/10_score_grid_soundaqnet.py     -> soundaqnet.csv, column sq_isop
# Both are resumable (finished files are skipped), so an interrupted run can be repeated. Each grid needs metrics.csv
# (written by generate_prompt_range.py --evaluate / evaluate_grid.py).
#
#   source ~/Documents/stable-audio-tools/.venv/bin/activate
#   cd ~/Documents/restorative_embeddings && git pull
#   DRY=1 bash isd/12_measure_grids.sh                       # list the grids and the commands
#   bash isd/12_measure_grids.sh                             # all grids in $OUTROOT, both measures
#   bash isd/12_measure_grids.sh score_5000 numbers_5000     # only these grids (names of folders in $OUTROOT)
#   MEASURES=psycho bash isd/12_measure_grids.sh             # only psychoacoustics   (MEASURES=sq: only SoundAQnet)
#
# Environment: OUTROOT, WORKERS (default 8), DEVICE (cuda), SOUNDSCAPER (third_party/SoundSCaper), SPL (65),
# SQ_LEVELS (assumed levels for SoundAQnet; default "65", the earlier Clotho rating used "60 65 70").
# Disk: SoundAQnet caches its features in <grid>/cache_soundaqnet (about 1 MB per file and level).
set -uo pipefail
REPO="${REPO:-$HOME/Documents/restorative_embeddings}"
OUTROOT="${OUTROOT:-$HOME/grids_3msde}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
SOUNDSCAPER="${SOUNDSCAPER:-third_party/SoundSCaper}"
SPL="${SPL:-65}"
SQ_LEVELS="${SQ_LEVELS:-65}"
MEASURES="${MEASURES:-psycho sq}"
cd "$REPO"

if [ $# -gt 0 ]; then GRIDS=("$@"); else GRIDS=($(for d in "$OUTROOT"/*/; do [ -e "$d/metrics.csv" ] && basename "$d"; done)); fi
[ ${#GRIDS[@]} -eq 0 ] && { echo "no grids with metrics.csv in $OUTROOT"; exit 1; }
if [[ " $MEASURES " == *" sq "* ]] && [ ! -d "$SOUNDSCAPER" ]; then
  echo "!! $SOUNDSCAPER not found (SoundAQnet needs the SoundSCaper checkout, see soundaqnet/README.md); skipping SoundAQnet"
  MEASURES="${MEASURES/sq/}"
fi

run() { echo "+ $*"; [ -n "${DRY:-}" ] || "$@"; }

for G in "${GRIDS[@]}"; do
  D="$OUTROOT/$G"
  [ -e "$D/metrics.csv" ] || { echo "!! $D has no metrics.csv, skipped"; continue; }
  echo "=== $G"
  if [[ " $MEASURES " == *" psycho "* ]]; then
    run python generation/psychoacoustics_grid.py "$D" --workers "$WORKERS" --spl "$SPL" 2>&1 | tee -a "$OUTROOT/log_measure_${G}.txt"
  fi
  if [[ " $MEASURES " == *" sq "* ]]; then
    run python clotho/10_score_grid_soundaqnet.py "$D" --soundscaper "$SOUNDSCAPER" --workers "$WORKERS" \
        --device "$DEVICE" --levels $SQ_LEVELS 2>&1 | tee -a "$OUTROOT/log_measure_${G}.txt"
    run python generation/add_soundaqnet_to_metrics.py "$D"
  fi
done

cat <<MSG

Compare, for example (ISD; use --levels -0.4 0 0.4 for the Clotho grids):
python generation/compare_grids.py \\
    base_numbers=$OUTROOT/base_with_numbers numbers5000=$OUTROOT/numbers_5000 words5000=$OUTROOT/words_5000 \\
    score3000=$OUTROOT/score_3000 score5000=$OUTROOT/score_5000 score7000=$OUTROOT/score_7000 \\
    --levels -0.7 0 0.7 --metrics sq_isop loud_N50 loud_N5 sharp rough --detail sq_isop --out ~/compare_extra_3msde.csv
MSG
