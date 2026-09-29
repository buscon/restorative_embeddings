#!/usr/bin/env bash
# Download the International Soundscape Database (ISD v1.0, Zenodo 10672568, CC BY 4.0)
#
#   bash scripts/download_isd.sh                 # into data/raw/isd
#   bash scripts/download_isd.sh /big/disk/isd   # somewhere else
#   SKIP_LOCKDOWN=1 bash scripts/download_isd.sh # skip the 2020 lockdown archives
#
# The lockdown archives hold ~630 recordings WITHOUT survey answers. They are
# not needed for training or testing, only for the unsupervised exploration.
# wget -c resumes interrupted downloads; re-running the script is safe.
set -euo pipefail

OUT="${1:-data/raw/isd}"
BASE="https://zenodo.org/records/10672568/files"
mkdir -p "$OUT/zips"

wget -c -q --show-progress -O "$OUT/ISD v1.0 Data.csv"      "$BASE/ISD%20v1.0%20Data.csv?download=1"
wget -c -q --show-progress -O "$OUT/ISD v1.0 Metadata.xlsx" "$BASE/ISD%20v1.0%20Metadata.xlsx?download=1"

ZIPS=(WAV_Granada_1 WAV_Groningen_1 WAV_London_1 WAV_London_2 WAV_London_3 WAV_London_4 WAV_Venice_1)
if [ -z "${SKIP_LOCKDOWN:-}" ]; then ZIPS+=(WAV_Lockdown_London WAV_Lockdown_Venice); fi

for z in "${ZIPS[@]}"; do
  echo "== $z"
  wget -c -q --show-progress -O "$OUT/zips/$z.zip" "$BASE/$z.zip?download=1"
  # skip macOS resource forks (__MACOSX/._*.wav are not audio) and .hadx side files
  unzip -q -n "$OUT/zips/$z.zip" -d "$OUT/audio" -x "__MACOSX/*" "*.DS_Store" "*.hadx"
done

echo "done. WAV files: $(find "$OUT/audio" -name '*.wav' | wc -l)"
echo "you can delete $OUT/zips to free space"
