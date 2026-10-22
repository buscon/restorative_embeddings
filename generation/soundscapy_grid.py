#!/usr/bin/env python3
"""Acoustic indices of the files of a grid folder with Soundscapy (https://github.com/MitchellAcoustics/Soundscapy).

Soundscapy (>= 0.8, extra "audio") is a standard toolbox for soundscape analysis; it wraps acoustic-toolbox (levels),
MoSQITo (psychoacoustics) and scikit-maad (ecoacoustic indices). Every file is read as a binaural recording and
calibrated to one assumed level (--spl, default 65 dB SPL rms, as in psychoacoustics_grid.py and the SoundAQnet
rating), because generated files have an arbitrary digital level. The indices of the two channels are averaged.

Default set (fast, about 5 s per 30 s file):
  LAeq, LZeq, LCeq (+ percentiles)      acoustic-toolbox, level
  S_perseg_*                            MoSQITo sharpness, DIN 45692, from the third-octave spectrum per segment
  maad spectral indices (44)            e.g. ACI, ADI, AEI, BI, H, NDSI, EAS, ECU, EPS, LFC/MFC/HFC, ...
  maad temporal indices (16)            e.g. ZCR, Ht, ACTtFraction, ...
Slow (hours for a full set of grids, use --loudness / --roughness and several workers):
  --loudness   ISO 532-1 loudness N_* and sharpness S_L_* (DIN 45692 from loudness), per channel
  --roughness  roughness R_* (Daniel and Weber)
The same MoSQITo functions are used by psychoacoustics_grid.py (columns loud_N50, sharp, rough), which are the quicker
way to get them (one channel, shorter roughness excerpt). Soundscapy has no spectral centroid; spectral_grid.py adds it.

Written to <folder>/soundscapy.csv (resumable: finished files are skipped) and merged into <folder>/metrics.csv
(by seed, masker, p) with the prefix "sspy_" for every column (e.g. sspy_LAeq, sspy_S_perseg_avg, sspy_ACI).

    pip install "soundscapy[audio]"
    python generation/soundscapy_grid.py ~/grids_3msde/joint_5000 --workers 8
    python generation/soundscapy_grid.py ~/grids_3msde/joint_5000 --workers 8 --loudness --roughness
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PREFIX = "sspy_"


def settings(loudness: bool, roughness: bool):
    from soundscapy.audio import AnalysisSettings
    s = AnalysisSettings.default()
    on = {("AcousticToolbox", "LAeq"), ("AcousticToolbox", "LZeq"), ("AcousticToolbox", "LCeq"),
          ("MoSQITo", "sharpness_din_perseg"),
          ("scikit_maad", "all_temporal_alpha_indices"), ("scikit_maad", "all_spectral_alpha_indices")}
    if loudness:
        on |= {("MoSQITo", "loudness_zwtv"), ("MoSQITo", "sharpness_din_from_loudness")}
    if roughness:
        on |= {("MoSQITo", "roughness_dw")}
    for lib in ("AcousticToolbox", "MoSQITo", "scikit_maad"):
        for metric in getattr(s, lib).model_dump().keys():
            s.update_setting(lib, metric, run=(lib, metric) in on)
    return s


def measure(args):
    path, spl, loudness, roughness = args
    from soundscapy.audio import Binaural, process_all_metrics
    b = Binaural.from_wav(path, calibrate_to=spl)
    df = process_all_metrics(b, settings(loudness, roughness), parallel=False)
    num = df.select_dtypes("number")
    return {"file": Path(path).name, **{PREFIX + str(c): float(v) for c, v in num.mean(axis=0).items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--spl", type=float, default=65.0, help="assumed level of every file, dB SPL rms")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--loudness", action="store_true", help="also ISO 532-1 loudness and DIN sharpness from loudness (slow)")
    ap.add_argument("--roughness", action="store_true", help="also roughness (slow)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    folder = Path(a.folder).expanduser()
    grid = pd.read_csv(folder / "grid.csv")
    out_csv = folder / "soundscapy.csv"
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["file"])
    need = [PREFIX + "N_5"] * a.loudness + [PREFIX + "R_avg"] * a.roughness
    todo = []
    for f in grid.file:
        if not (folder / f).exists():
            continue
        row = done[done.file == f]
        if row.empty or any(c not in done.columns or pd.isna(row[c].iloc[0]) for c in need):
            todo.append(f)
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(grid)} files in grid.csv, {len(todo)} to do (assumed level {a.spl:g} dB SPL"
          f"{', loudness' if a.loudness else ''}{', roughness' if a.roughness else ''})", flush=True)
    jobs = [(str(folder / f), a.spl, a.loudness, a.roughness) for f in todo]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(measure, jobs, chunksize=1), 1):
            rows.append(r)
            if i % 5 == 0 or i == len(jobs):
                new = pd.DataFrame(rows)
                cur = pd.concat([done[~done.file.isin(new.file)], new], ignore_index=True)
                cur.to_csv(out_csv, index=False)
                print(f"  {i}/{len(jobs)}", flush=True)
    res = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["file"])
    cols = [c for c in res.columns if c != "file"]

    mpath = folder / "metrics.csv"
    if mpath.exists() and len(res):
        g = grid.copy()
        g["masker"] = g.masker.astype(str).str.replace(" ", "_")
        g = g[["file", "seed", "masker", "p"]].merge(res, on="file")
        m = pd.read_csv(mpath)
        m = m.drop(columns=[c for c in cols if c in m.columns])
        m = m.merge(g[["seed", "masker", "p"] + cols], on=["seed", "masker", "p"], how="left")
        m.to_csv(mpath, index=False)
        print(f"merged {len(cols)} columns into {mpath} ({int(m[cols].notna().all(axis=1).sum())} of {len(m)} rows complete)")
    keep = [c for c in cols if c.startswith(PREFIX) and any(k in c for k in ("LAeq", "S_perseg_avg", "ACI", "MEANf", "ZCR"))]
    if keep:
        print(res[keep].describe().round(3).to_string())


if __name__ == "__main__":
    main()
