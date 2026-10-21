#!/usr/bin/env python3
"""Psychoacoustic measures for the files of a grid folder: loudness (ISO 532-1), sharpness (DIN 45692) and roughness.

For every file listed in <folder>/grid.csv the script computes, with MoSQITo (the package SoundAQnet's loudness input
already uses):

  loud_N50   median of the time-varying loudness N(t), sone (ISO 532-1, Zwicker, free field)
  loud_N5    loudness exceeded 5% of the time (peaks), sone
  sharp      mean sharpness, acum (DIN 45692, computed from the same loudness)
  rough      mean roughness, asper (Daniel and Weber), on an excerpt of --rough-seconds (slow: ~2.5 min per 30 s)

Tonality, with the tone-to-noise ratio (TNR) and the prominence ratio (PR) of ECMA-418-1 (MoSQITo), on segments of
0.5 s; only prominent tones count, a segment without one has TNR = PR = 0:

  tnr_mean   mean over segments of the total TNR, dB (0 for segments without a prominent tone)
  tnr_share  share of segments with at least one prominent tone (0 to 1)
  pr_mean    same for the prominence ratio, dB
  pr_share   same for the prominence ratio

Written to <folder>/psycho.csv (resumable: finished files are skipped; files that lack the tonality columns only get those, not the slow loudness
and roughness again; --no-tonality switches the tonality off) and merged into <folder>/metrics.csv
(by seed, masker, p) so that compare_grids.py can use the new columns:

    python generation/psychoacoustics_grid.py ~/grids_3msde/score_5000 --workers 8
    python generation/compare_grids.py a=~/grids_3msde/score_5000 b=~/grids_3msde/numbers_5000 \
        --levels -0.7 0 0.7 --metrics loud_N50 loud_N5 sharp rough --detail sharp

Level: generated files have an arbitrary digital level, and loudness and roughness depend on the absolute level. Each
file is therefore set to the same assumed level (--spl, default 65 dB SPL rms, unweighted) before the measures are
computed, as clotho/10_score_grid_soundaqnet.py does. The differences between files then come from spectrum and
temporal structure, not from how loud the model happened to render them (that is rms_db in metrics.csv). The absolute
sone, acum and asper values are not those of a calibrated recording; only the comparison between levels is meaningful.
Fluctuation strength is not in MoSQITo (1.2.1) and is not computed.

These are descriptions of the signal. They are standard in soundscape research, but they do not by themselves say
whether a listener finds the sound more pleasant.
"""
import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
FS = 44100
PSY = ["loud_N50", "loud_N5", "sharp", "rough"]
TON = ["tnr_mean", "tnr_share", "pr_mean", "pr_share"]
COLS = PSY + TON


def _to_mono_44k(x, sr):
    from scipy.signal import resample_poly
    y = x.mean(axis=1) if x.ndim > 1 else x
    if sr != FS:
        from math import gcd
        g = gcd(int(sr), FS)
        y = resample_poly(y, FS // g, int(sr) // g)
    return np.asarray(y, np.float64)


def tonality(y_pa):
    from mosqito.sq_metrics import pr_ecma_perseg, tnr_ecma_perseg
    t = np.asarray(tnr_ecma_perseg(y_pa, FS, prominence=True)[0], float)
    p = np.asarray(pr_ecma_perseg(y_pa, FS, prominence=True)[0], float)
    return {"tnr_mean": float(np.mean(t)), "tnr_share": float(np.mean(t > 0)),
            "pr_mean": float(np.mean(p)), "pr_share": float(np.mean(p > 0))}


def measure(args):
    path, spl, rough_seconds, do_rough, do_psy, do_ton = args
    from mosqito.sq_metrics import loudness_zwtv, roughness_dw, sharpness_din_from_loudness
    try:  # numba version of the slow decay step of the loudness (same result), from this repository
        from soundaqnet import fast_loudness
        fast_loudness.install()
    except Exception:
        pass
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    y = _to_mono_44k(x, sr)
    rms = float(np.sqrt(np.mean(y ** 2)))
    y_pa = y * (20e-6 * 10 ** (spl / 20)) / max(rms, 1e-12)
    out = {"file": Path(path).name}
    if do_psy:
        n, ns, _, _ = loudness_zwtv(y_pa, FS, field_type="free")
        s = sharpness_din_from_loudness(n, ns, weighting="din")
        out.update({"loud_N50": float(np.median(n)), "loud_N5": float(np.percentile(n, 95)),
                    "sharp": float(np.mean(s)), "rough": np.nan})
        if do_rough:
            m = int(rough_seconds * FS)
            a = max(0, (len(y_pa) - m) // 2)
            r = roughness_dw(y_pa[a:a + m], FS)[0]
            out["rough"] = float(np.mean(r))
    if do_ton:
        out.update(tonality(y_pa))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--spl", type=float, default=65.0, help="assumed overall level of every file, dB SPL rms")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rough-seconds", type=float, default=10.0, help="length of the excerpt (from the middle) for roughness")
    ap.add_argument("--no-roughness", action="store_true")
    ap.add_argument("--no-tonality", action="store_true", help="skip TNR and PR (ECMA-418-1)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    folder = Path(a.folder).expanduser()
    grid = pd.read_csv(folder / "grid.csv")
    out_csv = folder / "psycho.csv"
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["file"] + COLS)
    for c in COLS:
        if c not in done.columns:
            done[c] = np.nan
    done[COLS] = done[COLS].apply(pd.to_numeric, errors="coerce").astype(float)
    done = done.drop_duplicates("file", keep="last").set_index("file")
    jobs, n_ton_only = [], 0
    for f in grid.file:
        if not (folder / f).exists():
            continue
        row = done.loc[f] if f in done.index else None
        do_psy = row is None or pd.isna(row["loud_N50"])
        do_ton = (not a.no_tonality) and (row is None or pd.isna(row["tnr_mean"]))
        if do_psy or do_ton:
            jobs.append((str(folder / f), a.spl, a.rough_seconds, not a.no_roughness, do_psy, do_ton))
            n_ton_only += (not do_psy)
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{len(grid)} files in grid.csv, {len(jobs)} to do ({n_ton_only} of them tonality only; "
          f"assumed level {a.spl:g} dB SPL)")
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(measure, jobs, chunksize=1), 1):
            name = r.pop("file")
            for k, v in r.items():
                done.loc[name, k] = v
            if i % 5 == 0 or i == len(jobs):
                done.reset_index().to_csv(out_csv, index=False)
                print(f"  {i}/{len(jobs)}", flush=True)
    res = done.reset_index()
    res.to_csv(out_csv, index=False)
    cols = [c for c in COLS if res[c].notna().any()]

    mpath = folder / "metrics.csv"
    if mpath.exists():
        g = grid.copy()
        g["masker"] = g.masker.astype(str).str.replace(" ", "_")
        g = g[["file", "seed", "masker", "p"]].merge(res, on="file")
        m = pd.read_csv(mpath).drop(columns=[c for c in COLS if c in pd.read_csv(mpath, nrows=0).columns])
        m = m.merge(g[["seed", "masker", "p"] + cols], on=["seed", "masker", "p"], how="left")
        m.to_csv(mpath, index=False)
        print(f"merged {cols} into {mpath} ({m[cols].notna().all(axis=1).sum()} of {len(m)} rows complete)")
    print(res[cols].describe().round(2).to_string())


if __name__ == "__main__":
    main()
