#!/usr/bin/env bash
# Download the Clotho v2.1 audio archives (7z) into data/raw/clotho/archives.
# Sizes: development 4.5 GB, validation 1.3 GB, evaluation 1.2 GB (about 7 GB in total).
# Resumable (curl -C -); safe to re-run. Pass split names to fetch only some of them.
# Use 04_extract_selected_audio.py afterwards to unpack only the selected clips.
#
#   bash clotho/03_download_audio.sh                      # all three splits
#   bash clotho/03_download_audio.sh evaluation           # one split
#   OUT=/big/disk/clotho_archives bash clotho/03_download_audio.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${OUT:-data/raw/clotho/archives}"
BASE="https://zenodo.org/api/records/4783391/files"
SPLITS=("$@")
[ ${#SPLITS[@]} -eq 0 ] && SPLITS=(development validation evaluation)
mkdir -p "$OUT"
for s in "${SPLITS[@]}"; do
    f="clotho_audio_${s}.7z"
    echo "get  $f"
    curl -L --retry 5 --retry-delay 3 -C - -o "$OUT/$f" "$BASE/$f/content"
done
ls -lh "$OUT"
