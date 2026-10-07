"""Step 5 - stratified, blinded listening sample.

Draws the same number of recordings from every k-means cluster (clusters
from step 4), shuffles them and copies them under neutral names, so you can
listen without knowing the cluster, the location or the ratings.

    python scripts/05_listening_sample.py --per-cluster 6
    python scripts/05_listening_sample.py --per-cluster 6 --araus-per-cluster 2

Output (results/listening/):
    audio/L001.wav ...        ISD recordings (first 30 s, peak-normalised)
    audio/A001.wav ...        optional rebuilt ARAUS stimuli
    listening_sheet.csv       file name + empty columns for your notes
    key.csv                   file name -> dataset, id, cluster, location
                              (open only after listening)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsd.audio import ArausMixer, read  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--processed", default="data/processed")
ap.add_argument("--clusters", default="results/clusters")
ap.add_argument("--araus", default="data/raw/araus")
ap.add_argument("--out", default="results/listening")
ap.add_argument("--per-cluster", type=int, default=6)
ap.add_argument("--araus-per-cluster", type=int, default=0)
ap.add_argument("--rated-only", action="store_true", help="draw only ISD recordings that have ratings")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
out = Path(a.out)
(out / "audio").mkdir(parents=True, exist_ok=True)

recs = pd.read_csv(Path(a.processed) / "isd_recordings.csv").set_index("GroupID")
cl = pd.read_csv(Path(a.clusters) / "isd_clusters.csv", index_col=0)
recs = recs.join(cl[["cluster"]], how="inner")
if a.rated_only:
    recs = recs[recs.n_ratings > 0]


def write_clip(x, sr, path):
    x = x[: 30 * sr]
    x = x / (np.abs(x).max() + 1e-12) * 0.5  # listening level only; not for analysis
    sf.write(str(path), x, sr, subtype="PCM_24")


picked = []
for c, g in recs.groupby("cluster"):
    take = g.sample(min(a.per_cluster, len(g)), random_state=int(rng.integers(1e9)))
    picked += [{"dataset": "isd", "id": gid, "cluster": c, "location": r.LocationID, "source": r.wav}
               for gid, r in take.iterrows()]

if a.araus_per_cluster:
    stim = pd.read_csv(Path(a.processed) / "araus_stimuli.csv").set_index("stimulus_id")
    stim = stim.join(pd.read_csv(Path(a.clusters) / "araus_clusters.csv", index_col=0), how="inner")
    for c, g in stim.groupby("cluster"):
        take = g.sample(min(a.araus_per_cluster, len(g)), random_state=int(rng.integers(1e9)))
        picked += [{"dataset": "araus", "id": sid, "cluster": c, "location": r.soundscape, "source": None}
                   for sid, r in take.iterrows()]

key = pd.DataFrame(picked).sample(frac=1, random_state=a.seed).reset_index(drop=True)
mixer = None
names = []
for i, r in key.iterrows():
    prefix = "L" if r.dataset == "isd" else "A"
    name = f"{prefix}{i + 1:03d}.wav"
    if r.dataset == "isd":
        x, sr = read(r.source)
    else:
        if mixer is None:
            root = Path(a.araus)
            mixer = ArausMixer(pd.read_csv(root / "data/soundscapes.csv"), pd.read_csv(root / "data/maskers.csv"),
                               root / "soundscapes", root / "maskers")
        s, m, smr = r.id.split("|")
        x, sr = mixer.mix(s, m, float(smr))
    write_clip(x, sr, out / "audio" / name)
    names.append(name)
key.insert(0, "file", names)
key.drop(columns="source").to_csv(out / "key.csv", index=False)
sheet = key[["file"]].assign(restorative_1to7="", pleasant_1to5="", dominant_sources="", notes="")
sheet.to_csv(out / "listening_sheet.csv", index=False)
print(f"{len(key)} clips in {out/'audio'}; fill in {out/'listening_sheet.csv'}, then open key.csv")
