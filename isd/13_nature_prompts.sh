#!/usr/bin/env bash
# Nature-only prompts on the fine-tuned ISD models (+ base control), same settings as isd/11_grids_3msde.sh.
# 7 prompts x seeds 1-3 x levels -0.7 0 0.7, 30 s, 100 steps, guidance 7, dpmpp-3m-sde.
#
#   bash isd/13_nature_prompts.sh N        # numbers 5000, words 5000, base with words
#   bash isd/13_nature_prompts.sh M        # same, with --negative-prompt "music" (tests the shared music bed)
#   bash isd/13_nature_prompts.sh G        # numbers 5000 at guidance 3 and 4.5 (tests guidance)
#   DRY=1 bash isd/13_nature_prompts.sh N  # only list what would be generated
set -euo pipefail
REPO="${REPO:-$HOME/Documents/restorative_embeddings}"
RUNS="${RUNS:-$HOME/stableaudio}"
BASE="${BASE:-$RUNS/models/stabilityai__stable-audio-open-1.0/model.safetensors}"
CONFIG="${CONFIG:-$REPO/configs/model_config.json}"
OUTROOT="${OUTROOT:-$HOME/grids_nature}"
SAMPLER="${SAMPLER:-dpmpp-3m-sde}"
SECS="${SECS:-30}"; SEEDS="${SEEDS:-1 2 3}"; LEVELS="${LEVELS:--0.7 0.0 0.7}"
PROMPTS="${PROMPTS:-$REPO/isd/prompts_nature.txt}"
NUM_DIR="$RUNS/runs_isd/isd_all_whole/57jbv1r6/checkpoints"
WORDS_DIR="$RUNS/runs_isd_words/isd_words_whole/dap9tpf4/checkpoints"
cd "$REPO"; mkdir -p "$OUTROOT"

gen() {  # NAME CKPT_OR_DIR STEP GUIDANCE [extra args]
  local name="$1" src="$2" step="$3" g="$4"; shift 4
  local ck="$src"
  if [ "$step" != "-" ]; then ck=$(ls "$src"/epoch=*-step=${step}.ckpt 2>/dev/null | head -1 || true); fi
  if [ -z "$ck" ] || [ ! -e "$ck" ]; then echo "!! no checkpoint for $name"; return 0; fi
  echo "=== $name: $ck -> $OUTROOT/$name (sampler $SAMPLER, guidance $g)"
  if [ -n "${DRY:-}" ]; then return 0; fi
  python generation/generate_prompt_range.py --ckpt "$ck" --config "$CONFIG" --out "$OUTROOT/$name" \
      --prompts-file "$PROMPTS" --seeds $SEEDS --levels $LEVELS --sampler "$SAMPLER" --guidance "$g" \
      --seconds "$SECS" "$@" --evaluate 2>&1 | tee -a "$OUTROOT/log_${name}.txt"
}

for P in "$@"; do
  case "$P" in
    N) gen nat_numbers_5000 "$NUM_DIR" 5000 7
       gen nat_words_5000   "$WORDS_DIR" 5000 7 --labels
       gen nat_base_words   "$BASE" - 7 --labels ;;
    M) gen nat_numbers_5000_nomusic "$NUM_DIR" 5000 7 --negative-prompt "music"
       gen nat_words_5000_nomusic   "$WORDS_DIR" 5000 7 --labels --negative-prompt "music" ;;
    G) gen nat_numbers_5000_g3   "$NUM_DIR" 5000 3
       gen nat_numbers_5000_g4p5 "$NUM_DIR" 5000 4.5 ;;
    *) echo "unknown group $P"; exit 1 ;;
  esac
done
echo "Listen: python generation/make_listening_playlists.py quality --grids $OUTROOT/nat_numbers_5000 ... --out ~/listen/nature"
