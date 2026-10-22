#!/usr/bin/env python3
"""Rate the files of a training set with SoundAQnet and compare with the rating in the caption.

Purpose: a reference for the generated grids. SoundAQnet is an imperfect rater; knowing how its output behaves on the
training audio (which has a known rating in the caption) shows how large a difference it can see between very
unpleasant and very pleasant files. The same rater and the same settings as in clotho/10_score_grid_soundaqnet.py
are used (mono, assumed level 65 dB SPL rms, mean over every released checkpoint), so the numbers are comparable
with the ISOPleasant effects of the grids.

Input: a training folder with metadata.csv (file,caption). The rating is read from "[ISOPleasant: x.xx]" at the end of
the caption; captions without a number (words format) are skipped. Repeats made by the dataset builders (names ending
in _r1.wav, _r2.wav, ...) are links to the same audio and are rated once; --weighted counts them as in training.

    python soundaqnet/03_rate_training_sets.py ~/path/isd_all_whole --soundscaper third_party/SoundSCaper \
        --workers 8 --device cuda [--limit 20]
    python soundaqnet/03_rate_training_sets.py --summary-only ~/path/isd_all_whole      # reprint from the csv
    python soundaqnet/03_rate_training_sets.py ~/path/joint_whole_clean04 --prefix araus_ ...   # only the ARAUS files

Output: <out> (default ~/soundaqnet_training/<folder name>.csv, one row per audio file) and a printed summary:
per pleasantness group (ISD thresholds) the number of files, the mean and sd of the caption rating and of the
SoundAQnet rating, the correlation between them, and the difference very pleasant minus very unpleasant.

Reading it: for ISD the caption rating comes from listeners and SoundAQnet did not train on ISD, so the correlation is a
check of the rater. For ARAUS, SoundAQnet was trained on ARAUS ratings, so agreement is expected and only shows the
range of the rater. For Clotho the caption rating is a SoundAQnet prediction itself (circular).
"""
import argparse
import os
import re
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

BINS = ["very_unpleasant", "unpleasant", "neutral", "pleasant", "very_pleasant"]
TAG = re.compile(r"\[ISOPleasant:\s*(-?\d+(?:\.\d+)?)\]\s*$")
REPEAT = re.compile(r"_r\d+\.wav$")


def pleasantness_bin(p):
    return ("very_unpleasant" if p <= -0.5 else "unpleasant" if p <= -0.15 else
            "neutral" if p <= 0.15 else "pleasant" if p <= 0.5 else "very_pleasant")


def collect(folder, prefix=""):
    """Return (table of unique audio files with caption rating and repeat count, number of captions without a number)."""
    folder = Path(folder).expanduser()
    meta = pd.read_csv(folder / "metadata.csv")
    rows, skipped, counts = [], 0, {}
    for f, cap in zip(meta["file"], meta["caption"]):
        base = REPEAT.sub(".wav", f)
        counts[base] = counts.get(base, 0) + 1
    for f, cap in zip(meta["file"], meta["caption"]):
        if REPEAT.search(f) or not f.startswith(prefix):
            continue
        m = TAG.search(str(cap))
        if not m:
            skipped += 1
            continue
        p = float(m.group(1))
        rows.append({"file": f, "train_rating": p, "bin": pleasantness_bin(p), "n_listed": counts[f],
                     "caption": cap})
    return pd.DataFrame(rows), skipped


def summarise(d, weighted=False):
    """Print per-group statistics of the SoundAQnet rating against the caption rating."""
    from scipy.stats import pearsonr, spearmanr
    w = d["n_listed"] if weighted else pd.Series(1, index=d.index)
    print(f"\n{len(d)} audio files" + (f" ({int(w.sum())} entries with repeats, weighted)" if weighted else ""))
    out = []
    for b in BINS:
        g = d[d.bin == b]
        if not len(g):
            continue
        ww = w[g.index]
        out.append({"group": b, "files": len(g),
                    "train_mean": np.average(g.train_rating, weights=ww),
                    "sq_mean": np.average(g.sq_rating, weights=ww),
                    "sq_sd": g.sq_rating.std(),
                    "within_file_sd": g.sq_sd.mean()})
    t = pd.DataFrame(out)
    print(t.round(3).to_string(index=False))
    if len(d) > 2:
        rho = spearmanr(d.train_rating, d.sq_rating)[0]
        r = pearsonr(d.train_rating, d.sq_rating)[0]
        print(f"\ncaption rating vs SoundAQnet: Spearman rho {rho:+.2f}, Pearson r {r:+.2f}, "
              f"mean bias (SoundAQnet minus caption) {np.average(d.sq_rating - d.train_rating, weights=w):+.3f}")
    if {"very_unpleasant", "very_pleasant"} <= set(t.group):
        lo = t.set_index("group").loc["very_unpleasant", "sq_mean"]
        hi = t.set_index("group").loc["very_pleasant", "sq_mean"]
        print(f"SoundAQnet, very pleasant minus very unpleasant files: {hi - lo:+.3f} "
              f"(caption rating difference {t.set_index('group').loc['very_pleasant', 'train_mean'] - t.set_index('group').loc['very_unpleasant', 'train_mean']:+.3f})")


def job(args):
    wav, level, cache = args
    import soundfile as sf
    import torch
    from soundaqnet import features as F
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
    ap.add_argument("--levels", type=float, nargs="+", default=[65.0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None, help="only the first N files (test run)")
    ap.add_argument("--prefix", default="", help="only files whose name starts with this (e.g. araus_ in the joint set)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None, help="feature cache (default ~/soundaqnet_cache/<folder name>, outside the dataset)")
    ap.add_argument("--weighted", action="store_true", help="count repeated files as often as they are listed in training")
    ap.add_argument("--summary-only", action="store_true", help="reprint the summary from an existing csv")
    a = ap.parse_args()

    folder = Path(a.folder).expanduser()
    tag = f"{folder.name}_{a.prefix.strip('_')}" if a.prefix else folder.name
    out = Path(a.out).expanduser() if a.out else Path.home() / "soundaqnet_training" / f"{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    if a.summary_only:
        d = pd.read_csv(out)
        summarise(d, a.weighted)
        return

    d, skipped = collect(folder, a.prefix)
    if skipped:
        print(f"{skipped} captions without a number skipped")
    if a.limit:
        d = d.sample(min(a.limit, len(d)), random_state=0).reset_index(drop=True)
    print(f"{len(d)} audio files in {folder}")
    if not len(d):
        sys.exit("nothing to rate")

    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    cache = Path(a.cache).expanduser() if a.cache else Path.home() / "soundaqnet_cache" / tag
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(str(folder / f), lv, str(cache)) for f in d.file for lv in a.levels]
    with ProcessPoolExecutor(a.workers) as ex:
        paths = list(tqdm(ex.map(job, jobs, chunksize=1), total=len(jobs), desc="features"))
    paths = np.array(paths).reshape(len(d), len(a.levels))

    from soundaqnet.model import SoundAQnetRunner
    root = Path(a.soundscaper) / "Inferring_soundscape_clips_for_LLM/application/system/model"
    runners = [SoundAQnetRunner(a.soundscaper, ckpt=c.name, device=a.device) for c in sorted(root.glob("*.pth"))]
    mean, sd = [], []
    for i in tqdm(range(len(d)), desc="model"):
        pred = []
        for j in range(len(a.levels)):
            z = np.load(paths[i, j])
            for run in runners:
                pred.append(float(run.predict(z["mel"][None], z["loud"][None])["ISOPleasant"][0]))
        mean.append(np.mean(pred))
        sd.append(np.std(pred, ddof=1))
    d["sq_rating"], d["sq_sd"] = mean, sd
    d.to_csv(out, index=False)
    print(f"written {out}")
    summarise(d, a.weighted)


if __name__ == "__main__":
    main()
