#!/usr/bin/env bash
# Base model (Stable Audio Open 1.0, no fine-tuning) with the rating written as a WORD instead of a number.
# The base model cannot read the numbers as ratings; it can only react to pleasantness words it already knows.
# The rating of a level is written as the nearest of five words (generate_prompt_range.py --labels):
#     -0.7 very unpleasant | -0.35 unpleasant | 0 neutral | 0.35 pleasant | 0.7 very pleasant
# so the levels of the main test (-0.7, 0, +0.7) become "very unpleasant", "neutral", "very pleasant" and the extra
# levels of the common set (+-0.4) become "unpleasant" and "pleasant". The prompt keeps the tag format of all other runs,
# for example  "soundscape with traffic noise and human sounds [ISOPleasant: very unpleasant]".
# Everything else is as in the other g12 grids: sampler dpmpp-3m-sde, guidance 12, 30 s, seeds 1-3.
#
# Grids written to $OUTROOT (default ~/grids_3msde), 3 seeds each:
#   base_words_g12          the 7 ISD prompts (the default prompts of generate_prompt_range.py), levels -0.7 0 0.7   (63 files)
#   base_araus_words_g12    the 5 ARAUS prompts (generation/prompts_araus.txt), levels -0.7 0 0.7                     (45 files)
#   common_base_words_g12   the 3 common prompts (generation/prompts_common.txt), levels -0.7 -0.4 0 0.4 0.7         (45 files)
# (base_with_words, guidance 7, already exists for the ISD prompts.)
#
#   source ~/Documents/stable-audio-tools/.venv/bin/activate
#   cd ~/Documents/restorative_embeddings && git pull
#   DRY=1 bash isd/16_base_words.sh gen           # list what would run
#   bash isd/16_base_words.sh gen                 # 1. generate (use tmux or nohup)
#   bash isd/16_base_words.sh measure             # 2. psychoacoustics + SoundAQnet (isd/12_measure_grids.sh)
#   bash isd/16_base_words.sh plot                # 3. plots in $PLOTS (default ~/plots_base_words)
#   bash isd/16_base_words.sh gen measure plot    # all three
# Optional: TEMPLATE='{item}, {p}' bash isd/16_base_words.sh gen   writes the word as plain text without the tag.
set -euo pipefail
REPO="${REPO:-$HOME/Documents/restorative_embeddings}"
RUNS="${RUNS:-$HOME/stableaudio}"
BASE="${BASE:-$RUNS/models/stabilityai__stable-audio-open-1.0/model.safetensors}"
CONFIG="${CONFIG:-$REPO/configs/model_config.json}"
OUTROOT="${OUTROOT:-$HOME/grids_3msde}"
PLOTS="${PLOTS:-$HOME/plots_base_words}"
SAMPLER="${SAMPLER:-dpmpp-3m-sde}"
GUIDANCE="${GUIDANCE:-12}"
SECS="${SECS:-30}"
SEEDS="${SEEDS:-1 2 3}"
TEMPLATE="${TEMPLATE:-}"
[ -z "$TEMPLATE" ] && TEMPLATE='{item} [ISOPleasant: {p}]'
GRIDS=(base_words_g12 base_araus_words_g12 common_base_words_g12)

cd "$REPO"; mkdir -p "$OUTROOT"
[ $# -eq 0 ] && { sed -n 2,24p "$0"; exit 0; }

# gen NAME LEVELS [extra args]
gen() {
  local name="$1" levels="$2"; shift 2
  [ -e "$BASE" ] || { echo "!! base model not found: $BASE"; return 0; }
  echo "=== $name: base model with words -> $OUTROOT/$name (levels $levels, sampler $SAMPLER, guidance $GUIDANCE)"
  [ -n "${DRY:-}" ] && return 0
  python generation/generate_prompt_range.py --ckpt "$BASE" --config "$CONFIG" --out "$OUTROOT/$name" \
      --labels --levels $levels --seeds $SEEDS --sampler "$SAMPLER" --guidance "$GUIDANCE" --seconds "$SECS" \
      --template "$TEMPLATE" "$@" --evaluate 2>&1 | tee -a "$OUTROOT/log_${name}.txt"
}

for STEP in "$@"; do
  case "$STEP" in
    gen)
      gen base_words_g12         "-0.7 0.0 0.7"
      gen base_araus_words_g12   "-0.7 0.0 0.7"            --prompts-file generation/prompts_araus.txt
      gen common_base_words_g12  "-0.7 -0.4 0.0 0.4 0.7"   --prompts-file generation/prompts_common.txt
      ;;
    measure)
      if [ -n "${DRY:-}" ]; then echo "would run: OUTROOT=$OUTROOT bash isd/12_measure_grids.sh ${GRIDS[*]}"
      else OUTROOT="$OUTROOT" bash isd/12_measure_grids.sh "${GRIDS[@]}"; fi
      ;;
    plot)
      O="$OUTROOT"; mkdir -p "$PLOTS"
      echo "+ plot_grid_features.py ... --out $PLOTS/features_base_words"
      [ -n "${DRY:-}" ] || python generation/plot_grid_features.py "ARAUS prompts=$O/base_araus_words_g12" \
          "ISD prompts=$O/base_words_g12" --metrics sharp A_gt4k loud_N50 rough --out "$PLOTS/features_base_words" --slide
      echo "+ plot_isopleasant_strip.py ... --out $PLOTS/isopleasant_base_words"
      [ -n "${DRY:-}" ] || python generation/plot_isopleasant_strip.py "ARAUS prompts=$O/base_araus_words_g12" \
          "ISD prompts=$O/base_words_g12" --out "$PLOTS/isopleasant_base_words" --slide
      echo "+ compare_grids.py (common prompts, base with numbers against base with words)"
      [ -n "${DRY:-}" ] || python generation/compare_grids.py base_numbers=$O/common_base_g12 base_words=$O/common_base_words_g12 \
          --levels -0.7 0 0.7 --metrics sq_isop sharp A_gt4k loud_N50 rough --detail sq_isop --out "$PLOTS/compare_base_words.csv"
      ;;
    *) echo "unknown step '$STEP' (use gen, measure, plot)";;
  esac
done
