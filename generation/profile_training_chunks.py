#!/usr/bin/env python3
"""Acoustic profile of the training chunks by pleasantness bin, next to generated grids.

Question it answers: do the metrics of the fine-tuned model's outputs (stationarity cv, RMS,
stereo correlation, brightness, tonality, infrasound) match the training audio in the same
number range, or does the model drift (over-smoothing, near-mono noise, ...)?

For each unique training chunk (symlinked repeats "_r<k>.wav" are skipped) the same metrics as
evaluate_grid.py are computed, after peak-normalising the chunk like the generated files
(generate_*.py peak-normalise), so rms_db and cv are comparable. Bin = pleasantness_bin() of
the number in the caption (as in isd/02_select_balanced.py). Generated grids (folders with a
metrics.csv from evaluate_grid.py) are mapped to the same bins by their level p.

    python generation/profile_training_chunks.py isd/train/isd_all_balanced \
        --grid ft=~/grids/isd_all_6000_range_det --grid base=~/grids/base_range_det \
        --out ~/grids/training_profile

Writes <out>/training_profile.csv (one row per chunk) and <out>/profile_summary.txt.
"""
import argparse
import os
import re
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate_grid import METRICS, file_metrics  # noqa: E402

BINS = ["very_unpleasant", "unpleasant", "neutral", "pleasant", "very_pleasant"]
SHOW = ["cv", "rms_db", "stereo_corr", "A_gt1k", "A_gt4k", "flat_db", "peak_db", "sub20_pct"]
NUM = re.compile(r"\[ISOPleasant:\s*(-?\d+(?:\.\d+)?)\]")


def pleasantness_bin(x: float) -> str:
    if x <= -0.5:
        return "very_unpleasant"
    if x <= -0.15:
        return "unpleasant"
    if x <= 0.15:
        return "neutral"
    if x <= 0.5:
        return "pleasant"
    return "very_pleasant"


def source_group(item: str) -> str:
    s = item.lower()
    nat, trf, hum, oth = ("natural" in s), ("traffic" in s), ("human" in s), ("other noise" in s)
    if "no dominant" in s:
        return "none"
    n = nat + trf + hum + oth
    if n == 1:
        return "natural" if nat else "traffic" if trf else "human" if hum else "other_noise"
    return "mixed"


def one(path):
    y, sr = sf.read(path, always_2d=True)
    y = y / max(float(np.abs(y).max()), 1e-8)
    fd, tmp = tempfile.mkstemp(suffix=".wav")   # system temp dir, never the training folder
    os.close(fd)
    try:
        sf.write(tmp, y, sr, subtype="FLOAT")   # file_metrics reads from disk
        return file_metrics(tmp)
    finally:
        Path(tmp).unlink(missing_ok=True)


def table(df, by, title):
    g = df.groupby(by, observed=True)[SHOW]
    out = g.mean().round(2)
    out.insert(0, "n", g.size())
    print(f"\n{title}\n{out.to_string()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="training folder with metadata.csv (file,caption) and the wavs")
    ap.add_argument("--grid", action="append", default=[], metavar="NAME=FOLDER",
                    help="generated grid folder (has metrics.csv from evaluate_grid.py); repeatable")
    ap.add_argument("--out", default=None, help="where to write the csv and summary (default: the folder)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only the first N chunks (quick test)")
    a = ap.parse_args()

    folder = Path(a.folder).expanduser()
    meta = pd.read_csv(folder / "metadata.csv")
    meta = meta[~meta.file.str.contains(r"_r\d+\.wav$")].copy()   # skip repeat symlinks
    if a.limit:
        meta = meta.head(a.limit)
    meta["p"] = meta.caption.map(lambda c: float(NUM.search(c).group(1)))
    meta["bin"] = meta.p.map(pleasantness_bin)
    meta["item"] = meta.caption.map(lambda c: NUM.sub("", c).strip())
    meta["group"] = meta["item"].map(source_group)
    print(f"{len(meta)} unique chunks in {folder}; computing metrics ({a.workers} workers) ...", flush=True)
    with Pool(a.workers) as pool:
        mets = pool.map(one, [str(folder / f) for f in meta.file], chunksize=16)
    df = pd.concat([meta.reset_index(drop=True), pd.DataFrame(mets)], axis=1)

    out = Path(a.out).expanduser() if a.out else folder
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "training_profile.csv", index=False)

    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        df["bin"] = pd.Categorical(df["bin"], BINS, ordered=True)
        table(df, "bin", "TRAINING chunks by pleasantness bin (mean):")
        table(df, ["bin", "group"], "TRAINING chunks by bin and source group (mean):")
        print("\nTRAINING chunks, SD within bin:")
        print(df.groupby("bin", observed=True)[SHOW].std().round(2).to_string())

        rows = [df.assign(source="train")[["source", "bin"] + SHOW]]
        for spec in a.grid:
            name, _, path = spec.partition("=")
            g = pd.read_csv(Path(path).expanduser() / "metrics.csv")
            g["bin"] = g.p.map(pleasantness_bin)
            rows.append(g.assign(source=name)[["source", "bin"] + SHOW])
        if a.grid:
            allr = pd.concat(rows)
            allr["bin"] = pd.Categorical(allr["bin"], BINS, ordered=True)
            print("\nTRAINING vs GENERATED, same bins (generated levels: -0.7 vu, -0.4 u, -0.1/0.1 neutral, "
                  "0.4 pleasant, 0.7 vp):")
            for m in SHOW:
                t = allr.pivot_table(index="bin", columns="source", values=m, aggfunc="mean", observed=False)
                print(f"\n{m}\n{t.round(2).to_string()}")
    text = buf.getvalue()
    (out / "profile_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"written: {out / 'training_profile.csv'}, {out / 'profile_summary.txt'}")


if __name__ == "__main__":
    main()
