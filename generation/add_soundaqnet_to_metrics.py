#!/usr/bin/env python3
"""Merge the SoundAQnet rating of a grid folder (soundaqnet.csv, from clotho/10_score_grid_soundaqnet.py) into its
metrics.csv as the column sq_isop, so that compare_grids.py can use it like any other metric:

    python generation/add_soundaqnet_to_metrics.py ~/grids_3msde/score_5000 [more folders ...]
    python generation/compare_grids.py a=<folder> b=<folder> --levels -0.7 0 0.7 --metrics sq_isop --detail sq_isop

The rows are matched through grid.csv (file name), not through the prompt text. Re-running evaluate_grid.py on a folder
rewrites metrics.csv and drops the column; run this script again afterwards.
"""
import sys
from pathlib import Path

import pandas as pd

COL = "sq_isop"


def merge(folder: Path) -> bool:
    sq, grid, met = folder / "soundaqnet.csv", folder / "grid.csv", folder / "metrics.csv"
    for f in (sq, grid, met):
        if not f.exists():
            print(f"{folder.name}: {f.name} missing, skipped")
            return False
    g = pd.read_csv(grid)
    g["masker"] = g.masker.astype(str).str.replace(" ", "_")
    s = pd.read_csv(sq)[["file", "rating"]].rename(columns={"rating": COL})
    key = g[["file", "seed", "masker", "p"]].merge(s, on="file")
    m = pd.read_csv(met)
    m = m.drop(columns=[COL], errors="ignore").merge(key[["seed", "masker", "p", COL]], on=["seed", "masker", "p"], how="left")
    m.to_csv(met, index=False)
    print(f"{folder.name}: {COL} added to {int(m[COL].notna().sum())} of {len(m)} rows")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ok = [merge(Path(a).expanduser()) for a in sys.argv[1:]]
    sys.exit(0 if all(ok) else 1)
