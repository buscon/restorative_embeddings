#!/usr/bin/env bash
# Download the International Soundscape Database (ISD v1.0) into data/raw/isd.
# Skips the 2020 lockdown archives: they have no survey ratings, so they cannot be used
# for pleasantness-conditioned training. Resumable (wget -c); safe to re-run.
#
#   bash isd/01_download.sh                # into data/raw/isd
#   bash isd/01_download.sh /big/disk/isd  # somewhere else
set -euo pipefail
cd "$(dirname "$0")/.."
SKIP_LOCKDOWN=1 bash scripts/download_isd.sh "${1:-data/raw/isd}"
