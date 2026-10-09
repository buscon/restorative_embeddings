#!/usr/bin/env python3
"""Rate the files of a generate_grid.py folder with SoundAQnet and test whether the rating follows the number in the prompt.

For every file <folder>/s<seed>_<caption>_p<level>.wav the generated audio is rated like the Clotho clips were rated
(06_rate_soundaqnet.py): mono, each assumed level in --levels dB SPL rms, every released checkpoint, mean ISOPleasant.
Then, per caption and per seed, the Spearman correlation between the number in the prompt and the rating of the
generated audio, and the rating at each level. Written to <folder>/soundaqnet.csv (one row per file) and printed.

    python clotho/10_score_grid_soundaqnet.py ~/grids/clotho_rated_1000 --soundscaper third_party/SoundSCaper \
        --workers 8 --device cuda
    (also for ~/grids/clotho_base, and for the control run's grid)

What a positive result is: a rho clearly above 0 in most (caption, seed) groups and a rating that rises with the
number. What it is not: evidence that people hear the difference; the rater is the same model that produced the
training labels. A grid of the control run (trained without the number) should show no trend.
"""
import argparse
import os
import re
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soundaqnet import features as F  # noqa: E402

NAME = re.compile(r"^s(?P<seed>\d+)_(?P<cap>.+)_p(?P<p>-?\d+\.\d+)\.wav$")


def job(args):
    wav, level, cache = args
    import torch
    torch.set_num_threads(1)
    path = Path(cache) / f"{Path(wav).stem}__{level:g}.npz"
    if path.exists():
        return str(path)
    x, sr = sf.read(wav, dtype="float64", always_2d=True)
    y = F.to_mono_44k(x, sr)[: F.CLIP_SECONDS * F.SR_WAV]
    y_pa = y * F.assume_level(y, rms_db_spl=level)
    np.savez(path, mel=F.log_mel(y_pa / F.ARAUS_PA_PER_DIGITAL), loud=F.iso_loudness(y_pa))
    return str(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--soundscaper", default="third_party/SoundSCaper")
    ap.add_argument("--levels", type=float, nargs="+", default=[60.0, 65.0, 70.0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    folder = Path(a.folder).expanduser()
    files = sorted(p for p in folder.glob("*.wav") if NAME.match(p.name))
    if a.limit:
        files = files[: a.limit]
    print(f"{len(files)} files in {folder}")
    cache = folder / "cache_soundaqnet"
    cache.mkdir(exist_ok=True)
    jobs = [(str(p), lv, str(cache)) for p in files for lv in a.levels]
    with ProcessPoolExecutor(a.workers) as ex:
        paths = list(tqdm(ex.map(job, jobs, chunksize=1), total=len(jobs), desc="features"))
    paths = np.array(paths).reshape(len(files), len(a.levels))

    from soundaqnet.model import SoundAQnetRunner
    root = Path(a.soundscaper) / "Inferring_soundscape_clips_for_LLM/application/system/model"
    runners = [SoundAQnetRunner(a.soundscaper, ckpt=c.name, device=a.device) for c in sorted(root.glob("*.pth"))]
    rows = []
    for i, p in enumerate(tqdm(files, desc="model")):
        m = NAME.match(p.name)
        pred = []
        for j in range(len(a.levels)):
            z = np.load(paths[i, j])
            for run in runners:
                pred.append(float(run.predict(z["mel"][None], z["loud"][None])["ISOPleasant"][0]))
        rows.append({"file": p.name, "seed": int(m["seed"]), "caption": m["cap"], "p": float(m["p"]),
                     "rating": np.mean(pred), "rating_sd": np.std(pred, ddof=1)})
    d = pd.DataFrame(rows)
    d.to_csv(folder / "soundaqnet.csv", index=False)

    print("\nmean rating of the generated audio by caption and prompt number:")
    print(d.pivot_table(index="caption", columns="p", values="rating", aggfunc="mean").round(3).to_string())
    print(f"\nmean within-file instability (sd over levels x checkpoints): {d.rating_sd.mean():.3f}; "
          f"sd of the rating between files: {d.rating.std():.3f}")
    rhos = []
    for (cap, seed), g in d.groupby(["caption", "seed"]):
        if g.p.nunique() > 2:
            rhos.append({"caption": cap, "seed": seed, "rho": spearmanr(g.p, g.rating)[0]})
    r = pd.DataFrame(rhos)
    print(f"\nSpearman rho (prompt number vs rating) per (caption, seed): positive {int((r.rho > 0).sum())}, "
          f"negative {int((r.rho < 0).sum())} of {len(r)}; mean rho {r.rho.mean():+.2f}")
    print(r.groupby("caption").rho.agg(["mean", "count"]).round(2).to_string())
    lo, hi = d.p.min(), d.p.max()
    eff = d[d.p == hi].groupby("caption").rating.mean() - d[d.p == lo].groupby("caption").rating.mean()
    sdseed = d.groupby(["caption", "p"]).rating.std().groupby("caption").mean()
    print(f"\nrating at p={hi:+.1f} minus p={lo:+.1f}, with the seed-to-seed sd:")
    print(pd.DataFrame({"effect": eff, "seed_sd": sdseed}).round(3).to_string())


if __name__ == "__main__":
    main()
