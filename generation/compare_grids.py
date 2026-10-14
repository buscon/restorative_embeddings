#!/usr/bin/env python3
"""Put several evaluated grids (metrics.csv written by evaluate_grid.py) into one comparison.

For every grid, masker (or prompt) and metric:
  effect   mean at the highest pleasantness level minus mean at the lowest level
  seed SD  seed-to-seed standard deviation of one grid and metric (mean over all masker x level groups,
           the same definition as in evaluate_grid.py)
  ratio    effect / seed SD
  clear    |ratio| >= --k (default 2)
  ordered  the means change in one direction across all levels (lowest -> ... -> highest);
           False when the middle level lies outside the two extremes (e.g. p = 0.0 odd one out)

Prints a summary per grid and metric, and (with --detail METRIC) one row per grid and masker.
The full table is written to --out (default compare_grids.csv).

    python generation/compare_grids.py \\
        isd_g7=~/grids_isd_whole/isd_all_5000_range_det isd_g12=~/grids_isd_whole/guid/isd_all_5000_g12_det \\
        araus_g7=~/grids_araus_whole/step6000 araus_g12=~/grids_araus_whole/step6000_g12 \\
        --detail A_gt4k

Run evaluate_grid.py on a folder first if it has no metrics.csv. Grids with different level sets can be
compared, but the effect then refers to each grid's own lowest and highest level (printed in the summary).
With 3 seeds per group the seed SD itself is uncertain; read ratios near the threshold with care.
These metrics describe the signal; they do not measure what a listener hears.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["A_gt1k", "A_gt4k", "stereo_corr", "cv", "rms_db", "flat_db", "peak_db"]


def load(label: str, folder: str) -> pd.DataFrame:
    f = Path(folder).expanduser() / "metrics.csv"
    if not f.exists():
        sys.exit(f"{f} not found: run  python generation/evaluate_grid.py {folder}  first")
    df = pd.read_csv(f)
    df["grid"] = label
    return df


def effects(df: pd.DataFrame, metrics: list, k: float) -> pd.DataFrame:
    rows = []
    for grid, gdf in df.groupby("grid", sort=False):
        # only levels that every masker has (a folder re-run with another level set can hold a mix)
        per_masker = [set(np.round(g.p.unique(), 4)) for _, g in gdf.groupby("masker")]
        levels = sorted(set.intersection(*per_masker))
        if len(levels) < 2:
            print(f"warning: {grid}: fewer than 2 levels shared by all maskers, skipped")
            continue
        gdf = gdf.assign(p=np.round(gdf.p, 4))
        lo, hi = levels[0], levels[-1]
        for m in metrics:
            sd = gdf.groupby(["masker", "p"])[m].std().mean()
            for masker, g in gdf.groupby("masker"):
                means = g.groupby("p")[m].mean()
                if lo not in means.index or hi not in means.index:
                    continue
                seq = means.loc[levels].values
                eff = seq[-1] - seq[0]
                steps = np.sign(np.diff(seq))
                rows.append({
                    "grid": grid, "masker": masker, "metric": m, "levels": f"{lo:g}..{hi:g} ({len(levels)})",
                    "n_seeds": int(g.seed.nunique()), "low": seq[0], "high": seq[-1], "effect": eff,
                    "seed_sd": sd, "ratio": eff / sd if sd > 0 else np.nan,
                    "clear": bool(sd > 0 and abs(eff) >= k * sd),
                    "ordered": bool(np.all(steps == np.sign(eff)) and np.sign(eff) != 0),
                })
    return pd.DataFrame(rows)


def summary(eff: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (grid, metric), g in eff.groupby(["grid", "metric"], sort=False):
        out.append({
            "grid": grid, "metric": metric, "levels": g.levels.iloc[0], "maskers": len(g),
            "seed_sd": round(g.seed_sd.iloc[0], 2),
            "clear": int(g.clear.sum()),
            "clear_ordered": int((g.clear & g.ordered).sum()),
            "clear_not_ordered": int((g.clear & ~g.ordered).sum()),
            "median_abs_ratio": round(float(np.nanmedian(np.abs(g.ratio))), 1),
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grids", nargs="+", help="label=folder (folder with metrics.csv)")
    ap.add_argument("--metrics", nargs="+", default=METRICS)
    ap.add_argument("--k", type=float, default=2.0, help="effect counts as clear if |effect| >= k x seed SD")
    ap.add_argument("--levels", type=float, nargs="+", default=None,
                    help="use only these pleasantness levels, e.g. --levels -0.7 0 0.7")
    ap.add_argument("--detail", nargs="*", default=[], help="metrics to print per masker")
    ap.add_argument("--out", default="compare_grids.csv")
    a = ap.parse_args()

    frames = []
    for item in a.grids:
        if "=" not in item:
            sys.exit(f"'{item}': use label=folder")
        label, folder = item.split("=", 1)
        frames.append(load(label, folder))
    allf = pd.concat(frames, ignore_index=True)
    if a.levels:
        allf = allf[np.isclose(allf.p.values[:, None], np.array(a.levels)[None, :]).any(axis=1)]
    eff = effects(allf, a.metrics, a.k)
    eff.to_csv(a.out, index=False)

    pd.set_option("display.width", 200)
    print(f"\nSummary (clear = |effect| >= {a.k:g} x seed SD; ordered = monotone over all levels)\n")
    print(summary(eff).to_string(index=False))
    for m in a.detail:
        print(f"\nDetail: {m}\n")
        d = eff[eff.metric == m][["grid", "masker", "low", "high", "effect", "seed_sd", "ratio", "clear", "ordered"]]
        print(d.round(2).to_string(index=False))
    print(f"\nfull table: {a.out}")


if __name__ == "__main__":
    main()
