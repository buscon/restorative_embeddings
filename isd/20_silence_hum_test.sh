#!/usr/bin/env bash
# Why does "silence in a quiet room at night" give a switching ~50 Hz hum? Four variants, 5 seeds each, base model, 30 s, dpmpp-3m-sde:
#   A_no_night      "silence in a quiet room"                       guidance 12   (is "at night" the cause?)
#   B_no_silence    "a quiet room at night"                         guidance 12   (is "silence" the cause?)
#   C_guidance7     "silence in a quiet room at night"              guidance 7    (does strong guidance amplify it?)
#   D_negative      "silence in a quiet room at night"              guidance 12, negative prompt "crickets, ticking, hum"
# The original (guidance 12, no negative prompt) is already in $OUTROOT/base_ae_g12 (prompt ae_silence_2) and is the reference.
# hum_check.py then measures the hum (strongest line 30-130 Hz above the mid spectrum, and how often it switches).
#
#   source ~/Documents/stable-audio-tools/.venv/bin/activate
#   cd ~/Documents/restorative_embeddings && git pull
#   DRY=1 bash isd/20_silence_hum_test.sh gen          # list
#   bash isd/20_silence_hum_test.sh gen analyse        # about 20 clips x 4 variants = 100 min of audio, a few minutes of GPU time
# Listen to the files too: the numbers say whether a hum is there, not whether it is the sound you heard.
set -euo pipefail
REPO="${REPO:-$HOME/Documents/restorative_embeddings}"
RUNS="${RUNS:-$HOME/stableaudio}"
BASE="${BASE:-$RUNS/models/stabilityai__stable-audio-open-1.0/model.safetensors}"
CONFIG="${CONFIG:-$REPO/configs/model_config.json}"
OUTROOT="${OUTROOT:-$HOME/grids_3msde}"
PLOTS="${PLOTS:-$HOME/plots_silence}"
SEEDS="${SEEDS:-1 2 3 4 5}"
SECS="${SECS:-30}"
SAMPLER="${SAMPLER:-dpmpp-3m-sde}"
T="$OUTROOT/silence_test"
cd "$REPO"
[ $# -eq 0 ] && { sed -n 2,14p "$0"; exit 0; }
run() { echo "+ $*"; [ -n "${DRY:-}" ] || "$@"; }
# gen NAME PROMPT GUIDANCE [extra args]
gen() {
  local name="$1" prompt="$2" g="$3"; shift 3
  mkdir -p "$T"; local pf="$T/prompt_$name.txt"
  printf '%s|silence_test|%s\n' "$name" "$prompt" > "$pf"
  run python generation/generate_prompt_range.py --ckpt "$BASE" --config "$CONFIG" --out "$T/$name" --prompts-file "$pf" \
      --template "{item}" --levels 0.0 --seeds $SEEDS --sampler "$SAMPLER" --guidance "$g" --seconds "$SECS" "$@"
}
for STEP in "$@"; do
  case "$STEP" in
    gen)
      gen A_no_night   "silence in a quiet room"          12
      gen B_no_silence "a quiet room at night"            12
      gen C_guidance7  "silence in a quiet room at night"  7
      gen D_negative   "silence in a quiet room at night" 12 --negative-prompt "crickets, ticking, hum" ;;
    analyse)
      mkdir -p "$PLOTS"
      run python generation/hum_check.py "original (g12)=$OUTROOT/base_ae_g12" "A no night=$T/A_no_night" \
          "B no silence=$T/B_no_silence" "C guidance 7=$T/C_guidance7" "D negative prompt=$T/D_negative" \
          --prefix ae_silence_2 --out "$PLOTS/hum" ;;
    *) echo "unknown step '$STEP' (use gen, analyse)";;
  esac
done
