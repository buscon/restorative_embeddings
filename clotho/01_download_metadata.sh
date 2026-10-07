#!/usr/bin/env bash
# Download the Clotho v2.1 caption and metadata CSVs (about 3.3 MB) into data/raw/clotho.
# Audio is NOT downloaded here (7z archives: development 4.5 GB, validation 1.3 GB,
# evaluation 1.2 GB); fetch it later for the selected clips only, from
# https://zenodo.org/records/4783391. Resumable; safe to re-run.
#
#   bash clotho/01_download_metadata.sh                 # into data/raw/clotho
#   bash clotho/01_download_metadata.sh /big/disk/clotho
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-data/raw/clotho}"
BASE="https://zenodo.org/api/records/4783391/files"
mkdir -p "$OUT"
for f in clotho_captions_development.csv clotho_captions_validation.csv clotho_captions_evaluation.csv \
         clotho_metadata_development.csv clotho_metadata_validation.csv clotho_metadata_evaluation.csv LICENSE; do
    if [ -s "$OUT/$f" ]; then echo "have $f"; continue; fi
    echo "get  $f"
    # Zenodo drops the connection now and then; retry a few times.
    curl -sS -L --retry 5 --retry-delay 3 -C - -o "$OUT/$f" "$BASE/$f/content"
done
ls -l "$OUT"
