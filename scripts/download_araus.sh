#!/usr/bin/env bash
# Download ARAUS v1 (responses + participant data + source soundscapes and maskers)
# with the dataset authors' own script. ~3 GB. Run this on your own machine:
# the response CSVs are hosted on DR-NTU (researchdata.ntu.edu.sg).
#
#   bash scripts/download_araus.sh               # into data/raw/araus
#
# The augmented stimuli (~132 GB as WAV) are NOT generated: scripts/02_embed.py
# rebuilds each stimulus in memory from soundscape + masker + SMR.
set -euo pipefail

OUT="${1:-data/raw/araus}"
if [ ! -d "$OUT/code" ]; then
  git clone --depth 1 https://github.com/ntudsp/araus-dataset-baseline-models.git "$OUT"
fi
cd "$OUT/code"
python -c "import wget, six" 2>/dev/null || pip install wget six
python download.py manifest.csv

cd ..
for f in data/responses.csv data/participants.csv data/soundscapes.csv data/maskers.csv; do
  test -f "$f" || { echo "missing $f - download incomplete"; exit 1; }
done
echo "ok: $(ls soundscapes | wc -l) soundscapes, $(ls maskers | wc -l) maskers"
