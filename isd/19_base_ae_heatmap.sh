#!/usr/bin/env bash
# Base Stable Audio Open 1.0, plain prompts, SoundAQnet as rater: events x affective qualities heatmap
# (SoundSCaper paper, Fig. 6 style). 15 audio-event classes x 4 prompts x 5 seeds = 300 clips of 30 s, guidance 12, dpmpp-3m-sde.
# No pleasantness word, no tag: the base model alone. File names: s<seed>_<prompt>_p0.00.wav (the "_p0.00" is only the naming
# convention of generate_prompt_range.py, there is no number in the prompt).
#
#   source ~/Documents/stable-audio-tools/.venv/bin/activate
#   cd ~/Documents/restorative_embeddings && git pull
#   DRY=1 bash isd/19_base_ae_heatmap.sh prompts gen     # list what would run
#   bash isd/19_base_ae_heatmap.sh prompts               # 0. writes generation/prompts_base_ae.txt and the label maps
#   bash isd/19_base_ae_heatmap.sh gen                   # 1. generate (tmux or nohup; about an hour, existing files are skipped)
#   bash isd/19_base_ae_heatmap.sh rate                  # 2. SoundAQnet, all outputs (hours on few cores; cached)
#   bash isd/19_base_ae_heatmap.sh ref                   # 3. ARAUS baseline: N real stimuli rebuilt from data/raw/araus, rated at the true
#                                                        #    level and at the assumed 65 dB (soundaqnet/05_rate_araus_reference.py)
#   bash isd/19_base_ae_heatmap.sh plot                  # 4. heatmaps (events x 8 attributes + ISOPleasant/ISOEventful) with the baseline,
#                                                        #    and scene views (radar, scatter/density, joint) in $PLOTS
# Baseline: ARAUS_N stimuli (default 600) of the folds ARAUS_FOLDS (default all, 0-5), equal numbers per masker type.
# Fold 0 alone gave only 42 stimuli without silence maskers. Some or all folds may be SoundAQnet training data.
# Two baselines: "65" is rated exactly like the generated clips (like for like), "true" at the calibrated ARAUS level.
# Environment: OUTROOT (~/grids_3msde), PLOTS (~/plots_base_ae), SEEDS (1 2 3 4 5), SECS (30), GUIDANCE (12), LABELS (primary|broad),
# WORKERS (8), DEVICE (cuda), SOUNDSCAPER (third_party/SoundSCaper), UNIT (clip|prompt), SQ_LEVELS (65).
set -euo pipefail
REPO="${REPO:-$HOME/Documents/restorative_embeddings}"
RUNS="${RUNS:-$HOME/stableaudio}"
BASE="${BASE:-$RUNS/models/stabilityai__stable-audio-open-1.0/model.safetensors}"
CONFIG="${CONFIG:-$REPO/configs/model_config.json}"
OUTROOT="${OUTROOT:-$HOME/grids_3msde}"
PLOTS="${PLOTS:-$HOME/plots_base_ae}"
NAME="${NAME:-base_ae_g12}"
SEEDS="${SEEDS:-1 2 3 4 5}"
SECS="${SECS:-30}"
GUIDANCE="${GUIDANCE:-12}"
SAMPLER="${SAMPLER:-dpmpp-3m-sde}"
LABELS="${LABELS:-broad}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
SOUNDSCAPER="${SOUNDSCAPER:-third_party/SoundSCaper}"
UNIT="${UNIT:-clip}"
ARAUS_ROOT="${ARAUS_ROOT:-$REPO/data/raw/araus}"
ARAUS_N="${ARAUS_N:-600}"
ARAUS_FOLDS="${ARAUS_FOLDS:-0 1 2 3 4 5}"
EVENT_ORDER="${EVENT_ORDER:-Bird;Animal;Wind;Water;Natural sounds;Vehicle;Traffic;Sounds of things;Environment and background;Outside, rural or natural;Speech;Human sounds;Music;Noise;Silence}"   # row order of the SoundSCaper README figure 6 (semicolon-separated); empty = model order
AQ_ORDER="${AQ_ORDER:-}"         # comma-separated, all 8 attributes
FIG8_EVENTS="${FIG8_EVENTS:-Bird;Traffic}"   # events of the figure 8 style plot (README fig. 8 shows two events; set the same ones)
REF_OUT="${REF_OUT:-$OUTROOT/araus_ref}"
SQ_LEVELS="${SQ_LEVELS:-65}"
cd "$REPO"
[ $# -eq 0 ] && { sed -n 2,24p "$0"; exit 0; }
run() { echo "+ $*"; [ -n "${DRY:-}" ] || "$@"; }

for STEP in "$@"; do
  case "$STEP" in
    prompts)
      run python generation/make_ae_prompts.py ;;
    gen)
      run python generation/generate_prompt_range.py --ckpt "$BASE" --config "$CONFIG" --out "$OUTROOT/$NAME" \
          --prompts-file generation/prompts_base_ae.txt --template "{item}" --levels 0.0 --seeds $SEEDS \
          --sampler "$SAMPLER" --guidance "$GUIDANCE" --seconds "$SECS" ;;
    rate)
      run python soundaqnet/04_rate_folder_full.py "$OUTROOT/$NAME" --soundscaper "$SOUNDSCAPER" --workers "$WORKERS" \
          --device "$DEVICE" --levels $SQ_LEVELS ;;
    ref)
      run python soundaqnet/05_rate_araus_reference.py --araus "$ARAUS_ROOT" --soundscaper "$SOUNDSCAPER" --n "$ARAUS_N" \
          --folds $ARAUS_FOLDS --workers "$WORKERS" --device "$DEVICE" --out "$REF_OUT" ;;
    plot)
      mkdir -p "$PLOTS"
      for COND in 65 true; do
        REFCSV="$REF_OUT/soundaqnet_full_$COND.csv"
        if [ -z "${DRY:-}" ] && [ ! -e "$REFCSV" ]; then echo "!! $REFCSV missing (run the ref step); plotting without baseline"; REFARG=(); else
          REFARG=(--ref "$REFCSV" --ref-name "ARAUS ($COND)"); fi
        ORD=()
        [ -n "$EVENT_ORDER" ] && ORD+=(--event-order "$EVENT_ORDER")
        [ -n "$AQ_ORDER" ] && ORD+=(--aq-order "$AQ_ORDER")
        run python generation/plot_ae_aq_heatmap.py --grid "$OUTROOT/$NAME" --labels "generation/ae_label_${LABELS}.csv" \
            "${REFARG[@]}" --extra-iso --fig7 --fig8 "$FIG8_EVENTS" "${ORD[@]}" --unit "$UNIT" --out "$PLOTS/base_ae_ref$COND"
        [ ${#REFARG[@]} -gt 0 ] && run python generation/compare_class_means.py --grid "$OUTROOT/$NAME" --ref "$REFCSV" \
            --out "$PLOTS/class_means_ref$COND"
        run python generation/plot_scene_views.py --grid "$OUTROOT/$NAME" "${REFARG[@]}" --out "$PLOTS/scenes_ref$COND"
      done ;;
    *) echo "unknown step '$STEP' (use prompts, gen, rate, ref, plot)";;
  esac
done
