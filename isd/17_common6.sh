#!/usr/bin/env bash
# Six common prompts on every model, to ask whether fine-tuning changes the predicted pleasantness more than prompting the
# base model with words. Guidance 12, sampler dpmpp-3m-sde, 30 s, seeds 1-3, levels -0.7 0 +0.7 (the contrast used in the talk).
#
# The three prompts of the existing common set (generation/prompts_common.txt) are NOT regenerated: their files stay in
# common_*_g12. What is new:
#   common6new_<model>_g12   the three NEW prompts (generation/prompts_common6_new.txt) for ARAUS 6000, ISD 5000, ISD+ARAUS 5000
#                            and the base model with numbers (tag format as in all other runs)        27 files each
#   common6_base_words_g12   the base model with the rating as a WORD, all six prompts (generation/prompts_common6.txt):
#                            "<prompt>, very unpleasant" | "<prompt>, neutral" | "<prompt>, very pleasant"    54 files
# No tag and no number in the base-model prompt: the base model only gets words it can understand.
#
#   source ~/Documents/stable-audio-tools/.venv/bin/activate
#   cd ~/Documents/restorative_embeddings && git pull
#   DRY=1 bash isd/17_common6.sh gen           # list what would run
#   bash isd/17_common6.sh gen                 # 1. generate (use tmux or nohup)
#   bash isd/17_common6.sh measure             # 2. psychoacoustics + SoundAQnet (isd/12_measure_grids.sh)
#   bash isd/17_common6.sh analyse             # 3. table and figures in $PLOTS (default ~/plots_common6)
#   bash isd/17_common6.sh gen measure analyse # all three
# Optional: WORDS_TEMPLATE='{item}, {p}' is the default; WORD_LEVELS="-0.7 0.0 0.7".
set -euo pipefail
REPO="${REPO:-$HOME/Documents/restorative_embeddings}"
RUNS="${RUNS:-$HOME/stableaudio}"
BASE="${BASE:-$RUNS/models/stabilityai__stable-audio-open-1.0/model.safetensors}"
CONFIG="${CONFIG:-$REPO/configs/model_config.json}"
OUTROOT="${OUTROOT:-$HOME/grids_3msde}"
PLOTS="${PLOTS:-$HOME/plots_common6}"
SAMPLER="${SAMPLER:-dpmpp-3m-sde}"
GUIDANCE="${GUIDANCE:-12}"
SECS="${SECS:-30}"
SEEDS="${SEEDS:-1 2 3}"
LEVELS="${LEVELS:--0.7 0.0 0.7}"
WORDS_TEMPLATE="${WORDS_TEMPLATE:-}"
[ -z "$WORDS_TEMPLATE" ] && WORDS_TEMPLATE='{item}, {p}'

NUM_DIR="$RUNS/runs_isd/isd_all_whole/57jbv1r6/checkpoints"
ARAUS_DIR="$RUNS/runs_araus/araus_whole/l2xivkew/checkpoints"
JOINT_DIR="$RUNS/runs_joint/joint_whole_clean04/nxf8qc76/checkpoints"
NEW_GRIDS=(common6new_araus_6000_g12 common6new_numbers_5000_g12 common6new_joint_5000_g12 common6new_base_g12 common6_base_words_g12)

cd "$REPO"; mkdir -p "$OUTROOT"
[ $# -eq 0 ] && { sed -n 2,22p "$0"; exit 0; }

# gen NAME CKPT_OR_DIR STEP PROMPTS_FILE LEVELS [extra args]
gen() {
  local name="$1" src="$2" step="$3" pf="$4" lv="$5"; shift 5
  local ck="$src"
  if [ "$step" != "-" ]; then ck=$(ls "$src"/epoch=*-step=${step}.ckpt 2>/dev/null | head -1 || true); fi
  if [ -z "$ck" ] || [ ! -e "$ck" ]; then echo "!! no checkpoint for $name (looked in $src, step $step)"; return 0; fi
  echo "=== $name: $ck -> $OUTROOT/$name (prompts $pf, levels $lv, sampler $SAMPLER, guidance $GUIDANCE)"
  [ -n "${DRY:-}" ] && return 0
  python generation/generate_prompt_range.py --ckpt "$ck" --config "$CONFIG" --out "$OUTROOT/$name" \
      --prompts-file "$pf" --seeds $SEEDS --levels $lv --sampler "$SAMPLER" --guidance "$GUIDANCE" \
      --seconds "$SECS" "$@" --evaluate 2>&1 | tee -a "$OUTROOT/log_${name}.txt"
}

for STEP in "$@"; do
  case "$STEP" in
    gen)
      NEW=generation/prompts_common6_new.txt
      gen common6new_araus_6000_g12   "$ARAUS_DIR" 6000 $NEW "$LEVELS"
      gen common6new_numbers_5000_g12 "$NUM_DIR"   5000 $NEW "$LEVELS"
      gen common6new_joint_5000_g12   "$JOINT_DIR" 5000 $NEW "$LEVELS"
      gen common6new_base_g12         "$BASE"      -    $NEW "$LEVELS"
      gen common6_base_words_g12      "$BASE"      -    generation/prompts_common6.txt "${WORD_LEVELS:-$LEVELS}" \
          --labels --template "$WORDS_TEMPLATE"
      ;;
    measure)
      if [ -n "${DRY:-}" ]; then echo "would run: OUTROOT=$OUTROOT bash isd/12_measure_grids.sh ${NEW_GRIDS[*]}"
      else OUTROOT="$OUTROOT" bash isd/12_measure_grids.sh "${NEW_GRIDS[@]}"; fi
      ;;
    analyse)
      O="$OUTROOT"; mkdir -p "$PLOTS"
      echo "+ compare_common6.py ... --out $PLOTS/common6"
      [ -n "${DRY:-}" ] || python generation/compare_common6.py \
          "base (words)=$O/common6_base_words_g12" \
          "base (numbers)=$O/common_base_g12,$O/common6new_base_g12" \
          "ARAUS=$O/common_araus_6000_g12,$O/common6new_araus_6000_g12" \
          "ISD=$O/common_numbers_5000_g12,$O/common6new_numbers_5000_g12" \
          "ISD+ARAUS=$O/common_joint_5000_g12,$O/common6new_joint_5000_g12" \
          --ref "base (words)" --out "$PLOTS/common6"
      [ -n "${DRY:-}" ] || python generation/compare_common6.py \
          "base (words)=$O/common6_base_words_g12" \
          "base (numbers)=$O/common_base_g12,$O/common6new_base_g12" \
          "ARAUS=$O/common_araus_6000_g12,$O/common6new_araus_6000_g12" \
          "ISD=$O/common_numbers_5000_g12,$O/common6new_numbers_5000_g12" \
          "ISD+ARAUS=$O/common_joint_5000_g12,$O/common6new_joint_5000_g12" \
          --ref "base (words)" --out "$PLOTS/common6_slide" --slide >/dev/null
      ;;
    *) echo "unknown step '$STEP' (use gen, measure, analyse)";;
  esac
done
