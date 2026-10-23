#!/usr/bin/env bash
# The same three prompts (generation/prompts_common.txt: natural, urban, general) on every model, guidance 12,
# so that models can be compared on identical text, seeds and levels.
#
# Grids written to $OUTROOT (default ~/grids_3msde):
#   common_araus_6000_g12   common_numbers_5000_g12   common_joint_5000_g12   common_clotho_5000_g12   common_base_g12
# 3 prompts x 3 seeds x 5 levels (-0.7 -0.4 0 0.4 0.7) = 45 files of 30 s per grid. The levels +-0.4 are inside the range of
# every model (Clotho was trained on +-0.4, the others on about +-1); +-0.7 is the main contrast of the other four.
#
#   source ~/Documents/stable-audio-tools/.venv/bin/activate
#   cd ~/Documents/restorative_embeddings && git pull
#   DRY=1 bash isd/14_common_prompts.sh            # list what would run
#   bash isd/14_common_prompts.sh gen              # 1. generate (use tmux or nohup)
#   bash isd/14_common_prompts.sh measure          # 2. psychoacoustics + SoundAQnet (isd/12_measure_grids.sh)
#   bash isd/14_common_prompts.sh plot             # 3. features and ISOPleasant plots in $PLOTS (default ~/plots_common)
#   bash isd/14_common_prompts.sh gen measure plot # all three
# The plot step runs plot_isopleasant.py with $SSPY_PY (default ~/venv_sspy/bin/python, the soundscapy environment).
set -euo pipefail
REPO="${REPO:-$HOME/Documents/restorative_embeddings}"
RUNS="${RUNS:-$HOME/stableaudio}"
BASE="${BASE:-$RUNS/models/stabilityai__stable-audio-open-1.0/model.safetensors}"
CONFIG="${CONFIG:-$REPO/configs/model_config.json}"
OUTROOT="${OUTROOT:-$HOME/grids_3msde}"
PLOTS="${PLOTS:-$HOME/plots_common}"
SSPY_PY="${SSPY_PY:-$HOME/venv_sspy/bin/python}"
SAMPLER="${SAMPLER:-dpmpp-3m-sde}"
GUIDANCE="${GUIDANCE:-12}"
SECS="${SECS:-30}"
SEEDS="${SEEDS:-1 2 3}"
LEVELS="${LEVELS:--0.7 -0.4 0.0 0.4 0.7}"
PROMPTS_FILE="${PROMPTS_FILE:-generation/prompts_common.txt}"

NUM_DIR="$RUNS/runs_isd/isd_all_whole/57jbv1r6/checkpoints"
ARAUS_DIR="$RUNS/runs_araus/araus_whole/l2xivkew/checkpoints"
JOINT_DIR="$RUNS/runs_joint/joint_whole_clean04/nxf8qc76/checkpoints"
CLOTHO_DIR="$RUNS/runs_clotho/clotho_whole_rated/487z5lmc/checkpoints"
GRIDS=(common_araus_6000_g12 common_numbers_5000_g12 common_joint_5000_g12 common_clotho_5000_g12 common_base_g12)

cd "$REPO"; mkdir -p "$OUTROOT"
[ $# -eq 0 ] && { sed -n 2,19p "$0"; exit 0; }

# gen NAME CKPT_OR_DIR STEP
gen() {
  local name="$1" src="$2" step="$3" ck="$2"
  if [ "$step" != "-" ]; then ck=$(ls "$src"/epoch=*-step=${step}.ckpt 2>/dev/null | head -1 || true); fi
  if [ -z "$ck" ] || [ ! -e "$ck" ]; then echo "!! no checkpoint for $name (looked in $src, step $step)"; return 0; fi
  echo "=== $name: $ck -> $OUTROOT/$name (sampler $SAMPLER, guidance $GUIDANCE)"
  [ -n "${DRY:-}" ] && return 0
  python generation/generate_prompt_range.py --ckpt "$ck" --config "$CONFIG" --out "$OUTROOT/$name" \
      --prompts-file "$PROMPTS_FILE" --seeds $SEEDS --levels $LEVELS --sampler "$SAMPLER" --guidance "$GUIDANCE" \
      --seconds "$SECS" --evaluate 2>&1 | tee -a "$OUTROOT/log_${name}.txt"
}

for STEP in "$@"; do
  case "$STEP" in
    gen)
      gen common_araus_6000_g12   "$ARAUS_DIR"  6000
      gen common_numbers_5000_g12 "$NUM_DIR"    5000
      gen common_joint_5000_g12   "$JOINT_DIR"  5000
      gen common_clotho_5000_g12  "$CLOTHO_DIR" 5000
      gen common_base_g12         "$BASE"       -
      ;;
    measure)
      if [ -n "${DRY:-}" ]; then echo "would run: OUTROOT=$OUTROOT bash isd/12_measure_grids.sh ${GRIDS[*]}"
      else OUTROOT="$OUTROOT" bash isd/12_measure_grids.sh "${GRIDS[@]}"; fi
      ;;
    plot)
      mkdir -p "$PLOTS"; O="$OUTROOT"; B="$O/common_base_g12"
      RUNS_ARGS=("ARAUS=$O/common_araus_6000_g12@$B" "ISD=$O/common_numbers_5000_g12@$B"
                 "ISD+ARAUS=$O/common_joint_5000_g12@$B" "Clotho=$O/common_clotho_5000_g12@$B")
      echo "+ plot_grid_features.py ... --out $PLOTS/features"
      [ -n "${DRY:-}" ] || python generation/plot_grid_features.py "${RUNS_ARGS[@]}" \
          --metrics sharp A_gt4k loud_N50 rough --out "$PLOTS/features" --slide
      echo "+ plot_isopleasant.py ... --out $PLOTS/isopleasant"
      [ -n "${DRY:-}" ] || "$SSPY_PY" generation/plot_isopleasant.py "ARAUS=$O/common_araus_6000_g12" \
          "ISD=$O/common_numbers_5000_g12" "ISD+ARAUS=$O/common_joint_5000_g12" \
          "Clotho=$O/common_clotho_5000_g12" --controls "ARAUS=$B" "ISD=$B" "ISD+ARAUS=$B" "Clotho=$B" \
          --out "$PLOTS/isopleasant" --slide
      echo "+ compare_grids.py (all models and the base in one table; the +-0.4 contrast is common to all)"
      for L in "-0.7 0 0.7" "-0.4 0 0.4"; do
        [ -n "${DRY:-}" ] || python generation/compare_grids.py base=$B araus=$O/common_araus_6000_g12 \
            numbers=$O/common_numbers_5000_g12 joint=$O/common_joint_5000_g12 clotho=$O/common_clotho_5000_g12 \
            --levels $L --metrics sq_isop sharp A_gt4k loud_N50 rough --detail sq_isop \
            --out "$PLOTS/compare_${L// /_}.csv"
      done
      ;;
    *) echo "unknown step '$STEP' (use gen, measure or plot)";;
  esac
done
