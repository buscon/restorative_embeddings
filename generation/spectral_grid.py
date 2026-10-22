#!/usr/bin/env python3
"""Spectral (brightness) measures for the files of a grid folder.

For every file listed in <folder>/grid.csv, on the mono mix (mean of the channels), from the long-term power
spectrum (Welch, 4096 samples):

  centroid_hz     spectral centroid, power-weighted mean frequency, 20 Hz - 20 kHz (Hz). Higher = brighter.
  rolloff85_hz    frequency below which 85% of the power lies, 20 Hz - 20 kHz (Hz). Higher = brighter.
  tilt_db_oct     spectral tilt: slope of the 1/3-octave band levels (dB) against frequency (octaves),
                  100 Hz - 10 kHz. Less negative = brighter (flatter spectrum, more high-frequency energy).

None of them depends on the overall level, so no assumed level is needed (unlike loudness and roughness in
psychoacoustics_grid.py). Together with sharpness (DIN 45692) and A_gt1k / A_gt4k from evaluate_grid.py they describe
how bright a file is; they are signal measures and do not say what a listener hears.

Written to <folder>/spectral.csv and merged into <folder>/metrics.csv (by seed, masker, p) so that compare_grids.py can use
the new columns. Fast (a few seconds per grid):

    python generation/spectral_grid.py ~/grids_3msde/joint_5000 ~/grids_3msde/numbers_5000
    python generation/compare_grids.py a=~/grids_3msde/joint_5000 --metrics centroid_hz rolloff85_hz tilt_db_oct
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch

COLS = ["centroid_hz", "rolloff85_hz", "tilt_db_oct"]


def spectral_measures(y, sr):
    """y: mono signal. Returns the three measures."""
    f, p = welch(y, fs=sr, nperseg=min(4096, len(y)))
    band = (f >= 20) & (f <= 20000)
    fb, pb = f[band], p[band] + 1e-30
    centroid = float((fb * pb).sum() / pb.sum())
    cum = np.cumsum(pb) / pb.sum()
    rolloff = float(fb[np.searchsorted(cum, 0.85)])
    # 1/3-octave band levels, 100 Hz - 10 kHz (centres 100 * 2^(k/3)), regress level (dB) on octave number
    k = np.arange(0, int(np.floor(3 * np.log2(10000 / 100))) + 1)
    centres = 100 * 2.0 ** (k / 3)
    lo, hi = centres * 2 ** (-1 / 6), centres * 2 ** (1 / 6)
    levels, octs = [], []
    for c, a, b in zip(centres, lo, hi):
        m = (f >= a) & (f < b)
        if m.any():
            levels.append(10 * np.log10(p[m].sum() + 1e-30))
            octs.append(np.log2(c / 100))
    tilt = float(np.polyfit(octs, levels, 1)[0]) if len(levels) > 3 else float("nan")
    return {"centroid_hz": centroid, "rolloff85_hz": rolloff, "tilt_db_oct": tilt}


def process(folder: Path):
    grid = pd.read_csv(folder / "grid.csv")
    rows = []
    for fn in grid.file:
        path = folder / fn
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        x, sr = sf.read(path, dtype="float64", always_2d=True)
        rows.append({"file": fn, **spectral_measures(x.mean(axis=1), sr)})
    res = pd.DataFrame(rows)
    res.to_csv(folder / "spectral.csv", index=False)
    mpath = folder / "metrics.csv"
    if mpath.exists():
        g = grid.copy()
        g["masker"] = g.masker.astype(str).str.replace(" ", "_")
        g = g[["file", "seed", "masker", "p"]].merge(res, on="file")
        m = pd.read_csv(mpath)
        m = m.drop(columns=[c for c in COLS if c in m.columns])
        m = m.merge(g[["seed", "masker", "p"] + COLS], on=["seed", "masker", "p"], how="left")
        m.to_csv(mpath, index=False)
        print(f"{folder.name}: {len(res)} files, merged {COLS} into metrics.csv "
              f"({int(m[COLS].notna().all(axis=1).sum())} of {len(m)} rows complete)")
    else:
        print(f"{folder.name}: {len(res)} files; no metrics.csv, run generation/evaluate_grid.py to merge")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for a in sys.argv[1:]:
        process(Path(a).expanduser())
