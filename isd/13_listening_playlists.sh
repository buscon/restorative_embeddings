#!/usr/bin/env bash
# Listening playlists for the guidance 12 grids, one folder per training (as in the plots), three prompts per run:
# a mostly natural, an urban and a general one. Built with generation/make_listening_playlists.py.
#
#   cd ~/Documents/restorative_embeddings && git pull
#   DRY=1 bash isd/13_listening_playlists.sh        # check that all grids and prompts are found
#   bash isd/13_listening_playlists.sh              # build everything in $LISTEN (default ~/listen_g12)
#   SEEDS="1 2 3" bash isd/13_listening_playlists.sh   # more seeds (default: seed 1 only)
#   LUFS=-23 bash isd/13_listening_playlists.sh        # equalise loudness (default: original levels)
#
# Per training (araus, isd_numbers, joint, clotho) two outputs:
#   <LISTEN>/<training>/blind/    playlist.m3u, key.csv (do NOT open before rating), responses.csv, audio/
#       blind pairs, -0.7 vs +0.7 (Clotho -0.4 vs +0.4), same prompt and seed; the fine-tuned model and its
#       base-model control are mixed in random order, so the listener does not know which is which
#   <LISTEN>/<training>/open/     <grid>.m3u per grid, ordered prompt > seed > level (-0.7, 0, +0.7), not blind
#       for your own check of artifacts and of the middle level
set -uo pipefail
REPO="${REPO:-$HOME/Documents/restorative_embeddings}"
OUTROOT="${OUTROOT:-$HOME/grids_3msde}"   # on the Mac: ~/Downloads/Downloads/grids_3msde
LISTEN="${LISTEN:-$HOME/listen_g12}"
SEEDS="${SEEDS:-1}"
cd "$REPO"
PY="${PY:-python3}"
PL="$PY generation/make_listening_playlists.py"
LUFS_ARG=(); [ -n "${LUFS:-}" ] && LUFS_ARG=(--lufs "$LUFS")
run() { echo "+ $*"; [ -n "${DRY:-}" ] || "$@"; }

# name | fine-tuned grid | base-model grid | prompt substrings (natural, urban, general) | lo hi
build() {
  local name="$1" ft="$2" base="$3" lo="$4" hi="$5"; shift 5
  local maskers=("$@")
  for g in "$ft" "$base"; do [ -e "$OUTROOT/$g/grid.csv" ] || { echo "!! $OUTROOT/$g/grid.csv missing, $name skipped"; return; }; done
  echo "=== $name  prompts: ${maskers[*]}"
  run $PL pairs --grids "$OUTROOT/$ft" "$OUTROOT/$base" --out "$LISTEN/$name/blind" \
      --maskers "${maskers[@]}" --seeds $SEEDS --lo "$lo" --hi "$hi" --rng 1 "${LUFS_ARG[@]}"
  run $PL quality --grids "$OUTROOT/$ft" "$OUTROOT/$base" --out "$LISTEN/$name/open" \
      --maskers "${maskers[@]}" --seeds $SEEDS --lo "$lo" --hi "$hi"
}

# ARAUS has five maskers and no general one; wind is the most neutral of them.
build araus        araus_6000_g12   base_araus_prompts_g12   -0.7 0.7 birdsong "traffic noise" wind
build isd_numbers  numbers_5000_g12 base_with_numbers_g12    -0.7 0.7 "birdsong and wind" "traffic noise and human" "no dominant"
build joint        joint_5000_g12   base_with_numbers_g12    -0.7 0.7 "birdsong and wind" "traffic noise and human" "no dominant"
# Clotho has three captions: waves and wind (natural), busy city (urban), cicadas in woodland (natural, the third one).
build clotho       clotho_5000_g12  clotho_base_with_tag_g12 -0.4 0.4 blowing busy cicadas
echo; echo "Done: $LISTEN"; [ -n "${DRY:-}" ] || ls "$LISTEN"/*/blind/playlist.m3u
