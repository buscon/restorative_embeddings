#!/usr/bin/env bash
# Re-generate and re-analyse every file whose prompt has the word "birdsong" (the files contain music). New wording: "birds chirping".
#   ARAUS prompt   park soundscape with birdsong in the background      -> park soundscape with birds chirping in the background
#   ISD / common   soundscape with birdsong and wind in the trees       -> soundscape with birds chirping and wind in the trees
#   common6new     soundscape with flowing water and birdsong           -> soundscape with flowing water and birds chirping
# Nothing is overwritten. For every affected grid X:
#   X_birds   only the re-generated prompt (new wording), with metrics, psychoacoustics and SoundAQnet
#   X_chirp   the complete grid: old files without the word + the new files (symbolic links), tables merged
#             (generation/merge_grid_birds.py). All analysis below reads the X_chirp folders.
# Same checkpoints, seeds 1-3, levels, sampler (dpmpp-3m-sde), 100 steps, 30 s and guidance 12 as the old grids. 126 files.
#
#   source ~/Documents/stable-audio-tools/.venv/bin/activate
#   cd ~/Documents/restorative_embeddings && git pull
#   DRY=1 bash isd/18_birds_chirping.sh gen        # list what would run
#   bash isd/18_birds_chirping.sh gen              # 1. generate (use tmux or nohup; existing files are skipped)
#   bash isd/18_birds_chirping.sh measure          # 2. psychoacoustics + SoundAQnet on the new files (isd/12_measure_grids.sh)
#   bash isd/18_birds_chirping.sh merge            # 3. build the X_chirp folders
#   bash isd/18_birds_chirping.sh analyse          # 4. figures and tables in $PLOTS (default ~/plots_birds)
#   bash isd/18_birds_chirping.sh gen measure merge analyse   # all four
# Not covered: the guidance-7 grids (the g7 numbers on the slides) and the old joint ISD+ARAUS and Clotho grids.
set -euo pipefail
REPO="${REPO:-$HOME/Documents/restorative_embeddings}"
RUNS="${RUNS:-$HOME/stableaudio}"
BASE="${BASE:-$RUNS/models/stabilityai__stable-audio-open-1.0/model.safetensors}"
CONFIG="${CONFIG:-$REPO/configs/model_config.json}"
OUTROOT="${OUTROOT:-$HOME/grids_3msde}"
PLOTS="${PLOTS:-$HOME/plots_birds}"
TRAIN_DIR="${TRAIN_DIR:-$HOME/soundaqnet_training}"
SAMPLER="${SAMPLER:-dpmpp-3m-sde}"
GUIDANCE="${GUIDANCE:-12}"
SECS="${SECS:-30}"
SEEDS="${SEEDS:-1 2 3}"
L3="-0.7 0.0 0.7"
L5="-0.7 -0.4 0.0 0.4 0.7"
WORD="${WORD:-birdsong}"

NUM_DIR="$RUNS/runs_isd/isd_all_whole/57jbv1r6/checkpoints"
ARAUS_DIR="$RUNS/runs_araus/araus_whole/l2xivkew/checkpoints"
# grid | what: araus = generate_grid.py with the masker phrase; others = generate_prompt_range.py with a prompts file
GRIDS=(araus_6000_g12 base_araus_prompts_g12 numbers_5000_g12 base_with_numbers_g12
       common_araus_6000_g12 common_numbers_5000_g12 common_base_g12
       common6new_araus_6000_g12 common6new_numbers_5000_g12 common6new_base_g12 common6_base_words_g12)

cd "$REPO"; mkdir -p "$OUTROOT"
[ $# -eq 0 ] && { sed -n 2,22p "$0"; exit 0; }

ckpt() {  # DIR STEP -> checkpoint path (or the file itself when STEP is -)
  if [ "$2" = "-" ]; then echo "$1"; else ls "$1"/epoch=*-step=$2.ckpt 2>/dev/null | head -1 || true; fi
}

# gen_prompts NAME CKPT_OR_DIR STEP PROMPTS_FILE LEVELS [extra args]
gen_prompts() {
  local name="$1" src="$2" step="$3" pf="$4" lv="$5"; shift 5
  local ck; ck=$(ckpt "$src" "$step")
  if [ -z "$ck" ] || [ ! -e "$ck" ]; then echo "!! no checkpoint for $name (looked in $src, step $step)"; return 0; fi
  echo "=== ${name}_birds: $ck (prompts $pf, levels $lv)"
  [ -n "${DRY:-}" ] && return 0
  python generation/generate_prompt_range.py --ckpt "$ck" --config "$CONFIG" --out "$OUTROOT/${name}_birds" \
      --prompts-file "$pf" --seeds $SEEDS --levels $lv --sampler "$SAMPLER" --guidance "$GUIDANCE" \
      --seconds "$SECS" "$@" --evaluate 2>&1 | tee -a "$OUTROOT/log_${name}_birds.txt"
}

# gen_araus NAME CKPT_OR_DIR STEP   (the ARAUS masker prompt, as in isd/11_grids_3msde.sh gridold)
gen_araus() {
  local name="$1" src="$2" step="$3"
  local ck; ck=$(ckpt "$src" "$step")
  if [ -z "$ck" ] || [ ! -e "$ck" ]; then echo "!! no checkpoint for $name (looked in $src, step $step)"; return 0; fi
  echo "=== ${name}_birds: $ck (ARAUS masker 'birds chirping', levels $L3)"
  [ -n "${DRY:-}" ] && return 0
  python generation/generate_grid.py --ckpt "$ck" --config "$CONFIG" --out "$OUTROOT/${name}_birds" \
      --maskers "birds chirping" --levels $L3 --seeds $SEEDS --seconds "$SECS" --sampler "$SAMPLER" \
      --guidance "$GUIDANCE" 2>&1 | tee -a "$OUTROOT/log_${name}_birds.txt"
  python generation/evaluate_grid.py "$OUTROOT/${name}_birds" 2>&1 | tee "$OUTROOT/${name}_birds/evaluation.txt"
}

for STEP in "$@"; do
  case "$STEP" in
    gen)
      gen_araus   araus_6000_g12            "$ARAUS_DIR" 6000
      gen_araus   base_araus_prompts_g12    "$BASE"      -
      gen_prompts numbers_5000_g12          "$NUM_DIR"   5000 generation/prompts_birds_isd.txt      "$L3"
      gen_prompts base_with_numbers_g12     "$BASE"      -    generation/prompts_birds_isd.txt      "$L3"
      gen_prompts common_araus_6000_g12     "$ARAUS_DIR" 6000 generation/prompts_birds_natural.txt  "$L5"
      gen_prompts common_numbers_5000_g12   "$NUM_DIR"   5000 generation/prompts_birds_natural.txt  "$L5"
      gen_prompts common_base_g12           "$BASE"      -    generation/prompts_birds_natural.txt  "$L5"
      gen_prompts common6new_araus_6000_g12   "$ARAUS_DIR" 6000 generation/prompts_birds_natural2.txt "$L3"
      gen_prompts common6new_numbers_5000_g12 "$NUM_DIR"   5000 generation/prompts_birds_natural2.txt "$L3"
      gen_prompts common6new_base_g12         "$BASE"      -    generation/prompts_birds_natural2.txt "$L3"
      # the base model with the rating as a WORD ("<prompt>, very pleasant"), as in isd/17_common6.sh
      gen_prompts common6_base_words_g12      "$BASE"      -    generation/prompts_birds_both.txt "$L3" \
          --labels --template '{item}, {p}'
      ;;
    measure)
      NEW=(); for g in "${GRIDS[@]}"; do NEW+=("${g}_birds"); done
      if [ -n "${DRY:-}" ]; then echo "would run: OUTROOT=$OUTROOT bash isd/12_measure_grids.sh ${NEW[*]}"
      else OUTROOT="$OUTROOT" bash isd/12_measure_grids.sh "${NEW[@]}"; fi
      ;;
    merge)
      for g in "${GRIDS[@]}"; do
        echo "=== $g -> ${g}_chirp"
        [ -n "${DRY:-}" ] || python generation/merge_grid_birds.py --old "$OUTROOT/$g" --new "$OUTROOT/${g}_birds" \
            --out "$OUTROOT/${g}_chirp" --word "$WORD"
      done
      ;;
    analyse)
      O="$OUTROOT"; mkdir -p "$PLOTS"; M="--metrics sharp A_gt4k loud_N50 rough --xlim 20 --slide"
      echo "+ plot_grid_features.py (Results 1) -> $PLOTS/features"
      [ -n "${DRY:-}" ] || python generation/plot_grid_features.py \
          "ARAUS=$O/araus_6000_g12_chirp@$O/base_araus_prompts_g12_chirp" \
          "ISD=$O/numbers_5000_g12_chirp@$O/base_with_numbers_g12_chirp" $M --out "$PLOTS/features"
      echo "+ plot_grid_features.py (Base model 1) -> $PLOTS/features_base"
      [ -n "${DRY:-}" ] || python generation/plot_grid_features.py \
          "ARAUS prompts=$O/base_araus_prompts_g12_chirp" "ISD prompts=$O/base_with_numbers_g12_chirp" \
          $M --out "$PLOTS/features_base"
      echo "+ plot_levels_lines.py (Results 2, Base model 2) -> $PLOTS/levels_lines_slide"
      [ -n "${DRY:-}" ] || python generation/plot_levels_lines.py \
          --train "ARAUS=$TRAIN_DIR/araus_whole.csv" "ISD=$TRAIN_DIR/isd_all_whole.csv" \
          --gen "ARAUS=$O/common_araus_6000_g12_chirp,$O/common6new_araus_6000_g12_chirp" \
                "ISD=$O/common_numbers_5000_g12_chirp,$O/common6new_numbers_5000_g12_chirp" \
          --base "$O/common6_base_words_g12_chirp" --out "$PLOTS/levels_lines_slide" --slide --no-delta
      echo "+ compare_common6.py -> $PLOTS/common6 (table, and _slide for the figure)"
      for S in "" "--slide"; do
        OUTN="$PLOTS/common6"; [ -n "$S" ] && OUTN="$PLOTS/common6_slide"
        [ -n "${DRY:-}" ] || python generation/compare_common6.py \
            "base (words)=$O/common6_base_words_g12_chirp" \
            "base (numbers)=$O/common_base_g12_chirp,$O/common6new_base_g12_chirp" \
            "ARAUS=$O/common_araus_6000_g12_chirp,$O/common6new_araus_6000_g12_chirp" \
            "ISD=$O/common_numbers_5000_g12_chirp,$O/common6new_numbers_5000_g12_chirp" \
            --ref "base (words)" --out "$OUTN" $S
      done
      echo "Done. Copy to the deck folder figures/ (same names as before):"
      echo "  features_effects*, features_lines*       <- $PLOTS/features_*"
      echo "  features_base_effects*, features_base_lines* <- $PLOTS/features_base_*"
      echo "  levels_lines_slide.pdf/.png/_table.csv    <- $PLOTS/levels_lines_slide*"
      echo "  common6_delta_slide.pdf, common6_table.csv <- $PLOTS/common6_slide_delta.pdf, $PLOTS/common6_table.csv"
      ;;
    *) echo "unknown step '$STEP' (use gen, measure, merge, analyse)";;
  esac
done
