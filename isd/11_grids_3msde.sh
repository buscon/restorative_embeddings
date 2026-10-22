#!/usr/bin/env bash
# Full ISD grids with the sampler dpmpp-3m-sde (SAMPLER) for all fine-tuned ISD models and the base-model controls.
# Same settings as the earlier dpmpp-2m grids: 7 prompts x seeds 1-3 x levels -0.7 0 0.7, 30 s, 100 steps, guidance 7
# (63 files per grid). Only the sampler differs. Existing files are skipped, so an interrupted run can be repeated.
#
#   source ~/Documents/stable-audio-tools/.venv/bin/activate
#   cd ~/Documents/restorative_embeddings && git pull
#   bash isd/11_grids_3msde.sh A          # priority A: score 5000, words 5000, numbers 5000, base with words, base with numbers
#   bash isd/11_grids_3msde.sh B          # score 3000 and 7000, words 6000
#   bash isd/11_grids_3msde.sh C          # guidance 12 and 15 for numbers 5000
#   bash isd/11_grids_3msde.sh D          # ARAUS 6000 (guidance 7) and ARAUS prompts on the base model (generate_grid.py)
#   bash isd/11_grids_3msde.sh E          # Clotho 5000, 4000, 2000 and the Clotho base model with the tag (generate_grid.py)
#   GUIDANCE_DEFAULT=10 OUTROOT=$HOME/grids_g10 bash isd/11_grids_3msde.sh G   # main five grids with guidance 10 (H: base controls)
#   bash isd/11_grids_3msde.sh I          # guidance 12 for the grids still missing (joint, Clotho, base-model controls)
#   bash isd/11_grids_3msde.sh A B C D E  # all (use tmux or nohup: a run takes hours)
#   DRY=1 bash isd/11_grids_3msde.sh A    # only list what would be generated
#
# Output: $OUTROOT/<name>/ (wavs, grid.csv, prompts.csv, metrics.csv, evaluation.txt) and $OUTROOT/log_<name>.txt.
# Compare afterwards with generation/compare_grids.py (command printed at the end).
set -euo pipefail
REPO="${REPO:-$HOME/Documents/restorative_embeddings}"
RUNS="${RUNS:-$HOME/stableaudio}"
BASE="${BASE:-$RUNS/models/stabilityai__stable-audio-open-1.0/model.safetensors}"
CONFIG="${CONFIG:-$REPO/configs/model_config.json}"
CONFIG_SCORE="${CONFIG_SCORE:-$REPO/configs/model_config_score.json}"
OUTROOT="${OUTROOT:-$HOME/grids_3msde}"
SAMPLER="${SAMPLER:-dpmpp-3m-sde}"
SECS="${SECS:-30}"
SEEDS="${SEEDS:-1 2 3}"
LEVELS="${LEVELS:--0.7 0.0 0.7}"
GUIDANCE_DEFAULT="${GUIDANCE_DEFAULT:-7}"   # e.g. GUIDANCE_DEFAULT=10 OUTROOT=$HOME/grids_g10 bash isd/11_grids_3msde.sh G H

NUM_DIR="$RUNS/runs_isd/isd_all_whole/57jbv1r6/checkpoints"
WORDS_DIR="$RUNS/runs_isd_words/isd_words_whole/dap9tpf4/checkpoints"
SCORE_DIR="$RUNS/runs_isd_score/isd_score_whole/75en8ym7/checkpoints"

cd "$REPO"; mkdir -p "$OUTROOT"

# gen NAME CKPT_OR_DIR STEP CONFIG GUIDANCE [extra args of generate_prompt_range.py]
gen() {
  local name="$1" src="$2" step="$3" cfg="$4" g="$5"; shift 5
  local ck="$src"
  if [ "$step" != "-" ]; then ck=$(ls "$src"/epoch=*-step=${step}.ckpt 2>/dev/null | head -1 || true); fi
  if [ -z "$ck" ] || [ ! -e "$ck" ]; then echo "!! no checkpoint for $name (looked in $src, step $step)"; return 0; fi
  if [ ! -e "$cfg" ]; then echo "!! config $cfg not found (needed for $name)"; return 0; fi
  echo "=== $name: $ck -> $OUTROOT/$name (sampler $SAMPLER, guidance $g)"
  if [ -n "${DRY:-}" ]; then return 0; fi
  python generation/generate_prompt_range.py --ckpt "$ck" --config "$cfg" --out "$OUTROOT/$name" \
      --seeds $SEEDS --levels $LEVELS --sampler "$SAMPLER" --guidance "$g" --seconds "$SECS" "$@" --evaluate \
      2>&1 | tee -a "$OUTROOT/log_${name}.txt"
}

ARAUS_DIR="$RUNS/runs_araus/araus_whole/l2xivkew/checkpoints"
JOINT_DIR="$RUNS/runs_joint/joint_whole_clean04/nxf8qc76/checkpoints"
CLOTHO_DIR="$RUNS/runs_clotho/clotho_whole_rated/487z5lmc/checkpoints"
CLOTHO_CAPS=("The wind is blowing and the waves are flowing"
             "It is a busy day in the city and public transportation is everywhere"
             "Cicadas are chirping in the woodland area in a consistent fashion while the wind blows")

# gridold NAME CKPT_OR_DIR STEP CONFIG GUIDANCE [--clotho]   (generate_grid.py: ARAUS maskers, or the three Clotho captions)
gridold() {
  local name="$1" src="$2" step="$3" cfg="$4" g="$5" kind="${6:-}"
  local ck="$src"
  if [ "$step" != "-" ]; then ck=$(ls "$src"/epoch=*-step=${step}.ckpt 2>/dev/null | head -1 || true); fi
  if [ -z "$ck" ] || [ ! -e "$ck" ]; then echo "!! no checkpoint for $name (looked in $src, step $step)"; return 0; fi
  echo "=== $name: $ck -> $OUTROOT/$name (sampler $SAMPLER, guidance $g)"
  if [ -n "${DRY:-}" ]; then return 0; fi
  if [ "$kind" = "--clotho" ]; then
    python generation/generate_grid.py --ckpt "$ck" --config "$cfg" --out "$OUTROOT/$name" \
        --maskers "${CLOTHO_CAPS[@]}" --prompt-template "{item} [ISOPleasant: {p}]" \
        --levels $CLOTHO_LEVELS --seeds $SEEDS --seconds "$SECS" --sampler "$SAMPLER" --guidance "$g" \
        2>&1 | tee -a "$OUTROOT/log_${name}.txt"
  else
    python generation/generate_grid.py --ckpt "$ck" --config "$cfg" --out "$OUTROOT/$name" \
        --levels $LEVELS --seeds $SEEDS --seconds "$SECS" --sampler "$SAMPLER" --guidance "$g" \
        2>&1 | tee -a "$OUTROOT/log_${name}.txt"
  fi
  python generation/evaluate_grid.py "$OUTROOT/$name" 2>&1 | tee "$OUTROOT/$name/evaluation.txt"
}

for P in "$@"; do
  case "$P" in
    A)
      gen score_5000         "$SCORE_DIR" 5000 "$CONFIG_SCORE" $GUIDANCE_DEFAULT --score-cond
      gen words_5000         "$WORDS_DIR" 5000 "$CONFIG"       $GUIDANCE_DEFAULT --labels
      gen numbers_5000       "$NUM_DIR"   5000 "$CONFIG"       $GUIDANCE_DEFAULT
      gen base_with_words    "$BASE"      -    "$CONFIG"       $GUIDANCE_DEFAULT --labels
      gen base_with_numbers  "$BASE"      -    "$CONFIG"       $GUIDANCE_DEFAULT
      ;;
    B)
      gen score_7000         "$SCORE_DIR" 7000 "$CONFIG_SCORE" $GUIDANCE_DEFAULT --score-cond
      gen score_3000         "$SCORE_DIR" 3000 "$CONFIG_SCORE" $GUIDANCE_DEFAULT --score-cond
      gen words_6000         "$WORDS_DIR" 6000 "$CONFIG"       $GUIDANCE_DEFAULT --labels
      ;;
    C)
      gen numbers_5000_g12   "$NUM_DIR"   5000 "$CONFIG"       12
      gen numbers_5000_g15   "$NUM_DIR"   5000 "$CONFIG"       15
      ;;
    D)
      # ARAUS: generate_grid.py with its default masker prompts, as in the earlier grids (those used dpmpp-2m)
      gridold araus_6000 "$ARAUS_DIR" 6000 "$CONFIG" $GUIDANCE_DEFAULT
      gridold base_araus_prompts "$BASE" - "$CONFIG" $GUIDANCE_DEFAULT
      gridold araus_6000_g12 "$ARAUS_DIR" 6000 "$CONFIG" 12
      gridold araus_6000_g15 "$ARAUS_DIR" 6000 "$CONFIG" 15
      ;;
    E)
      CLOTHO_LEVELS="-0.4 0.0 0.4"
      gridold clotho_5000 "$CLOTHO_DIR" 5000 "$CONFIG" $GUIDANCE_DEFAULT --clotho
      gridold clotho_base_with_tag "$BASE" - "$CONFIG" $GUIDANCE_DEFAULT --clotho
      gridold clotho_4000 "$CLOTHO_DIR" 4000 "$CONFIG" $GUIDANCE_DEFAULT --clotho
      gridold clotho_2000 "$CLOTHO_DIR" 2000 "$CONFIG" $GUIDANCE_DEFAULT --clotho
      ;;
    G)
      # main set: the five grids of the talk (ISD numbers/words/score, joint, ARAUS); use with GUIDANCE_DEFAULT=10
      gen numbers_5000       "$NUM_DIR"   5000 "$CONFIG"       $GUIDANCE_DEFAULT
      gen words_5000         "$WORDS_DIR" 5000 "$CONFIG"       $GUIDANCE_DEFAULT --labels
      gen score_5000         "$SCORE_DIR" 5000 "$CONFIG_SCORE" $GUIDANCE_DEFAULT --score-cond
      gen joint_5000         "$JOINT_DIR" 5000 "$CONFIG"       $GUIDANCE_DEFAULT
      gridold araus_6000 "$ARAUS_DIR" 6000 "$CONFIG" $GUIDANCE_DEFAULT
      ;;
    H)
      # base-model controls for the main set
      gen base_with_numbers  "$BASE"      -    "$CONFIG"       $GUIDANCE_DEFAULT
      gen base_with_words    "$BASE"      -    "$CONFIG"       $GUIDANCE_DEFAULT --labels
      gridold base_araus_prompts "$BASE" - "$CONFIG" $GUIDANCE_DEFAULT
      ;;
    I)
      # guidance 12 for the grids that do not exist yet (numbers_5000_g12 and araus_6000_g12 are made by group C and D);
      # outputs go to $OUTROOT with the suffix _g12. Run with the default OUTROOT (~/grids_3msde).
      CLOTHO_LEVELS="-0.4 0.0 0.4"
      gen joint_5000_g12            "$JOINT_DIR" 5000 "$CONFIG" 12
      gen base_with_numbers_g12     "$BASE"      -    "$CONFIG" 12
      gridold base_araus_prompts_g12 "$BASE" - "$CONFIG" 12
      gridold clotho_5000_g12 "$CLOTHO_DIR" 5000 "$CONFIG" 12 --clotho
      gridold clotho_base_with_tag_g12 "$BASE" - "$CONFIG" 12 --clotho
      ;;
    *) echo "unknown group '$P' (use A, B, C, D, E, G, H or I)";;
  esac
done

cat <<MSG

Compare (after evaluation has run for every folder):
python generation/compare_grids.py \\
    base_words=$OUTROOT/base_with_words base_numbers=$OUTROOT/base_with_numbers \\
    numbers5000=$OUTROOT/numbers_5000 words5000=$OUTROOT/words_5000 words6000=$OUTROOT/words_6000 \\
    score3000=$OUTROOT/score_3000 score5000=$OUTROOT/score_5000 score7000=$OUTROOT/score_7000 \\
    --levels -0.7 0 0.7 --detail flat_db
MSG
