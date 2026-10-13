#!/usr/bin/env bash
# Does a higher guidance scale make the pleasantness number easier to hear?
# For each checkpoint and each guidance value: 7 prompts x seeds 1-3 x levels -0.7, 0.0, 0.7 (63 files),
# deterministic sampler (dpmpp-2m) so the same seed gives the same starting noise at every level.
# Output: ~/grids/guid/isd_all_<step>_g<guidance>_det (wavs, grid.csv, metrics.csv, evaluation.txt)
# and playlists in ~/grids/guid/playlists (seed 1, ordered guidance by guidance, p = -0.7, 0.0, 0.7).
#
#   source ~/Documents/stable-audio-tools/.venv/bin/activate
#   cd ~/Documents/restorative_embeddings && git pull
#   bash isd/07_guidance_grid.sh                       # steps 6000 12000 19000, guidance 7 12 15
#   STEPS="6000" GUIDANCE="7 12" bash isd/07_guidance_grid.sh
set -euo pipefail
REPO="${REPO:-$HOME/Documents/restorative_embeddings}"
CKPT_DIR="${CKPT_DIR:-$HOME/stableaudio/runs_isd/isd_all_balanced/9185hkux/checkpoints}"
CONFIG="${CONFIG:-$REPO/configs/model_config.json}"
SECS="${SECS:-10}"   # length of the generated clips; use 30 for the whole-file models
STEPS="${STEPS:-6000 12000 19000}"
GUIDANCE="${GUIDANCE:-7 12 15}"
ROOT="${ROOT:-$HOME/grids/guid}"   # use a new folder for a new run, existing files are skipped
mkdir -p "$ROOT"
cd "$REPO"

for S in $STEPS; do
  CK=$(ls "$CKPT_DIR"/epoch=*-step=${S}.ckpt 2>/dev/null | head -1 || true)
  if [ -z "$CK" ]; then echo "no checkpoint for step $S in $CKPT_DIR"; continue; fi
  for G in $GUIDANCE; do
    OUT="$ROOT/isd_all_${S}_g${G}_det"
    echo "=== step $S, guidance $G -> $OUT"
    python generation/generate_prompt_range.py \
        --ckpt "$CK" --config "$CONFIG" --out "$OUT" \
        --levels -0.7 0.0 0.7 --guidance "$G" --sampler dpmpp-2m --seconds "$SECS" --evaluate \
        2>&1 | tee -a "$ROOT/log_${S}_g${G}.txt"
  done
done

# playlists (relative paths, so copy the whole ~/grids/guid folder and open them from playlists/)
python3 - "$ROOT" "$GUIDANCE" "$STEPS" <<'PY'
import re, sys
from pathlib import Path
root, guid, steps = Path(sys.argv[1]), sys.argv[2].split(), sys.argv[3].split()
pl = root / "playlists"; pl.mkdir(exist_ok=True)
pat = re.compile(r"^s1_(.+)_p(-?\d+\.\d+)\.wav$")
for s in steps:
    per = {}
    for g in guid:
        d = root / f"isd_all_{s}_g{g}_det"
        for f in sorted(d.glob("s1_*.wav")) if d.exists() else []:
            m = pat.match(f.name)
            if m: per.setdefault(m.group(1), {}).setdefault(g, []).append((float(m.group(2)), f))
    for prompt, byg in per.items():
        with open(pl / f"step{s}_{prompt}_seed1.m3u", "w") as out:
            out.write("#EXTM3U\n")
            for g in guid:
                for p, f in sorted(byg.get(g, [])):
                    out.write(f"#EXTINF:10,step {s} guidance {g} p={p:+.1f}\n../{f.parent.name}/{f.name}\n")
print("playlists:", len(list(pl.glob('*.m3u'))), "in", pl)
PY
