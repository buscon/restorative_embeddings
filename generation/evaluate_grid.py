#!/usr/bin/env python3
"""Evaluate a generate_grid.py output folder.

Per file: share of power below 20 Hz (sub20_pct), A-weighted power share above 1 kHz and
4 kHz (A_gt1k, A_gt4k, %), stereo correlation, loudness variation (cv = std/mean of
0.5 s RMS), overall RMS (dB), flat_db and peak_db (tonality, 100 Hz - 8 kHz;
more negative flat_db and higher peak_db mean more tonal, music-like content). Written to <dir>/metrics.csv.

Then, per metric and masker: mean by pleasantness level, and for each (masker, seed) the
Spearman rho between the pleasantness value and the metric, plus a sign count over all
groups. With only 3 levels per group rho is coarse; read the sign count as direction only.

    python generation/evaluate_grid.py out_grid_1500 [--compare out_grid_1000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch
from scipy.stats import spearmanr

METRICS = ["sub20_pct", "A_gt1k", "A_gt4k", "stereo_corr", "cv", "rms_db", "flat_db", "peak_db"]
TREND_METRICS = ["A_gt1k", "A_gt4k", "cv", "rms_db", "flat_db", "peak_db"]


def a_weight_db(f):
    f = np.maximum(f, 1e-3)
    f2 = f ** 2
    ra = (12194.0 ** 2 * f2 ** 2) / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194.0 ** 2))
    return 20 * np.log10(ra) + 2.0


def file_metrics(path):
    y, sr = sf.read(path, always_2d=True)
    y = y.astype(np.float64)
    mono = y.mean(axis=1)
    f, pxx = welch(mono, fs=sr, nperseg=min(8192, len(mono)))
    total = pxx.sum() + 1e-20
    pa = pxx * 10 ** (a_weight_db(f) / 10)
    ta = pa.sum() + 1e-20
    out = {
        "sub20_pct": 100 * pxx[f < 20].sum() / total,
        "A_gt1k": 100 * pa[f >= 1000].sum() / ta,
        "A_gt4k": 100 * pa[f >= 4000].sum() / ta,
        "stereo_corr": float(np.corrcoef(y[:, 0], y[:, 1])[0, 1]) if y.shape[1] > 1 else 1.0,
    }
    # tonality over 100 Hz - 8 kHz, frame by frame (2048 samples, hop 1024):
    #   flat_db = mean spectral flatness in dB (0 = noise-like, more negative = more tonal)
    #   peak_db = mean (strongest bin minus median bin) in dB (higher = clearer tonal peaks)
    from scipy.signal import stft
    ff, _, Z = stft(mono, fs=sr, nperseg=2048, noverlap=1024)
    band = (ff >= 100) & (ff <= 8000)
    P = np.abs(Z[band]) ** 2 + 1e-20
    flat = np.exp(np.log(P).mean(axis=0)) / P.mean(axis=0)
    out["flat_db"] = float(np.mean(10 * np.log10(flat)))
    out["peak_db"] = float(np.mean(10 * np.log10(P.max(axis=0) / np.median(P, axis=0))))
    n = int(0.5 * sr)
    frames = mono[: len(mono) // n * n].reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(axis=1)) + 1e-12
    out["cv"] = float(rms.std() / rms.mean())
    out["rms_db"] = 20 * np.log10(np.sqrt((mono ** 2).mean()) + 1e-12)
    return out


def compute(folder: Path) -> pd.DataFrame:
    manifest = folder / "grid.csv"
    if not manifest.exists():
        sys.exit(f"{manifest} not found: evaluate_grid.py only reads folders written by generate_grid.py")
    grid = pd.read_csv(manifest)
    need = {"seed", "masker", "p", "file"}
    if not need <= set(grid.columns):
        sys.exit(f"{manifest} has columns {list(grid.columns)}, expected {sorted(need)}; "
                 "this folder was not written by generate_grid.py. Re-generate it with that script.")
    rows = []
    for r in grid.itertuples():
        p = folder / r.file
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            continue
        rows.append({"seed": r.seed, "masker": r.masker.replace(" ", "_"), "p": r.p, **file_metrics(p)})
    df = pd.DataFrame(rows)
    df.to_csv(folder / "metrics.csv", index=False)
    return df


def summarise(df: pd.DataFrame, label: str):
    print(f"\n===== {label}: {len(df)} files =====")
    print("\nMean by masker and pleasantness level:")
    print(df.groupby(["masker", "p"])[METRICS].mean().round(2).to_string())
    print("\nSeed-to-seed SD (mean over groups) vs. level effect (mean p=+0.8 minus p=-0.8):")
    for m in TREND_METRICS:
        sd = df.groupby(["masker", "p"])[m].std().mean()
        lo = df[df.p == df.p.min()].groupby("masker")[m].mean()
        hi = df[df.p == df.p.max()].groupby("masker")[m].mean()
        print(f"  {m:8s} seed SD {sd:7.2f} | effect by masker: " +
              ", ".join(f"{k}={hi[k] - lo[k]:+.2f}" for k in lo.index))
    print("\nSpearman rho (p vs metric) per (masker, seed) group; negative = metric falls with higher p:")
    for m in TREND_METRICS:
        rhos = []
        for _, g in df.groupby(["masker", "seed"]):
            if g.p.nunique() > 1 and g[m].nunique() > 1:
                rhos.append(spearmanr(g.p, g[m])[0])
        rhos = np.array(rhos)
        neg, pos = int((rhos < 0).sum()), int((rhos > 0).sum())
        print(f"  {m:8s} negative {neg}, positive {pos} of {len(rhos)} groups, mean rho {rhos.mean():+.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--compare", help="second grid folder (e.g. an earlier checkpoint)")
    args = ap.parse_args()
    summarise(compute(Path(args.folder)), args.folder)
    if args.compare:
        summarise(compute(Path(args.compare)), args.compare)


if __name__ == "__main__":
    main()
