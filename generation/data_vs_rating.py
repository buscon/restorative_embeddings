#!/usr/bin/env python3
"""Which way does the TRAINING AUDIO change with the rating? Compare with what a model generates.

For a training folder (wav files + metadata.csv with captions "<description> [ISOPleasant: x.xx]") the same
descriptive metrics as in evaluate_grid.py are computed per file. For every metric it prints
  rho        Spearman correlation between the rating and the metric over all files
  slope      linear regression slope per rating unit; 'expected change' = slope x (hi - lo), the change a model
             would show between rating lo and hi if it reproduced the trend of the data
  within     median Spearman rho inside groups of files with the same description text (groups with >= --min-group
             files): separates the effect of the rating from the effect of the description (e.g. natural vs traffic)
Repeats (_r1, _r2 ... of the same file) are counted once. Compare the sign and the size of 'expected change' with the
effect in the generated grids (compare_grids.py): a generated effect in the opposite direction of the data, or much
larger than the data trend, is not simply the learned relationship.

    python generation/data_vs_rating.py --data-dir clotho/train_whole --lo -0.4 --hi 0.4
    python generation/data_vs_rating.py --data-dir isd/train/isd_all_whole --lo -0.7 --hi 0.7
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate_grid import file_metrics  # noqa: E402

TAG = re.compile(r"^(?P<text>.*?)\s*\[ISOPleasant:\s*(?P<val>[-+]?\d+(?:\.\d+)?)\]\s*$")
METRICS = ["A_gt1k", "A_gt4k", "stereo_corr", "cv", "rms_db", "flat_db", "peak_db"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--lo", type=float, default=-0.4)
    ap.add_argument("--hi", type=float, default=0.4)
    ap.add_argument("--max-files", type=int, default=0, help="0 = all")
    ap.add_argument("--min-group", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    data = Path(a.data_dir).expanduser()
    meta = pd.read_csv(data / "metadata.csv")
    rows = []
    for r in meta.itertuples():
        if re.search(r"_r\d+\.wav$", r.file):
            continue
        m = TAG.match(str(r.caption))
        if m and (data / r.file).exists():
            rows.append({"file": r.file, "text": m["text"], "rating": float(m["val"])})
    if a.max_files and len(rows) > a.max_files:
        rows = [rows[i] for i in np.linspace(0, len(rows) - 1, a.max_files).astype(int)]
    print(f"{len(rows)} files; computing metrics ...", flush=True)
    for i, r in enumerate(rows):
        r.update(file_metrics(str(data / r["file"])))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out) if a.out else data / "data_vs_rating.csv"
    df.to_csv(out, index=False)
    x = df.rating.values
    print(f"\nrating: min {x.min():+.2f}, median {np.median(x):+.2f}, max {x.max():+.2f}, SD {x.std():.2f}")
    sizes = df.groupby("text").size()
    big = sizes[sizes >= a.min_group].index
    print(f"{len(sizes)} different description texts; {len(big)} with >= {a.min_group} files "
          f"({int(sizes[big].sum())} files)\n")
    hdr = f"expected change {a.lo:+g} -> {a.hi:+g}"
    print(f"{'metric':12s}{'rho':>7s}{'p':>12s}{'slope':>10s}  {hdr:>30s}  {'within-group rho (median)':>28s}")
    for m in METRICS:
        y = df[m].values
        if np.nanstd(y) < 1e-9:
            print(f"{m:12s}  constant in the data (skipped)")
            continue
        rho, p = spearmanr(x, y)
        slope = np.polyfit(x, y, 1)[0]
        wr = []
        for t in big:
            g = df[df.text == t]
            if g[m].std() > 0 and g.rating.std() > 0:
                wr.append(spearmanr(g.rating, g[m]).correlation)
        w = f"{np.median(wr):+.2f} ({len(wr)} groups)" if wr else "n/a"
        print(f"{m:12s}{rho:+7.2f}{p:12.3g}{slope:+10.2f}  {slope * (a.hi - a.lo):+30.2f}  {w:>28s}")
    print(f"\nper-file table: {out}")


if __name__ == "__main__":
    main()
