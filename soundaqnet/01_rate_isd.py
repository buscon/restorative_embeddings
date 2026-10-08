#!/usr/bin/env python3
"""Run SoundAQnet on the human-rated ISD recordings, plain and under perturbations.

This is step 0 of plan B (validation gate): can SoundAQnet reproduce what people rated on audio it
was not trained on, and does it survive the conditions of uncalibrated, shorter clips?

Conditions (all start from the first 30 s of the recording, ISD samples are in pascal):
    full        as is
    gain-20 gain-10 gain+10    the pressure signal scaled by that many dB (level sensitivity)
    cut10 cut5  only the first 10 s or 5 s (length sensitivity)
`full` runs on every rated recording. The other conditions run on a random subset of --sens-n
recordings, because the ISO 532-1 loudness takes 20 to 60 s per 30 s clip and is the bottleneck.
Features are cached per recording and condition in --cache, so the script can be stopped and rerun.

    python soundaqnet/01_rate_isd.py --soundscaper third_party/SoundSCaper \
        --recordings data/processed/isd_recordings.csv --out soundaqnet/isd_soundaqnet.csv \
        --workers 8 --device cuda [--limit 20] [--sens-n 150]

--recordings is the table written by scripts/01_prepare.py (columns GroupID, wav, n_ratings).
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soundaqnet import features as F  # noqa: E402

CONDITIONS = {  # name -> (gain_db, max_seconds)
    "full": (0.0, 30), "gain-20": (-20.0, 30), "gain-10": (-10.0, 30), "gain+10": (10.0, 30),
    "cut10": (0.0, 10), "cut5": (0.0, 5),
}
ISD_PA_PER_UNIT = 1.0  # ISD wavs are 32-bit float in pascal (see archive/docs/project_report_milestone1.md)


def job(args):
    gid, wav, cond, cache = args
    path = Path(cache) / f"{gid}__{cond}.npz"
    if path.exists():
        return str(path)
    gain, secs = CONDITIONS[cond]
    x, sr = sf.read(wav, dtype="float64", always_2d=True, frames=int(31 * 48000))
    mel, loud = F.extract(x, sr, ISD_PA_PER_UNIT, max_seconds=secs, gain_db=gain)
    np.savez(path, mel=mel, loud=loud)
    return str(path)


def level_report(table, n=40):
    """Sanity check of the calibration assumption: rms level of some recordings in dB SPL."""
    lv = []
    for wav in table.wav.head(n):
        x, sr = sf.read(wav, dtype="float64", always_2d=True, frames=int(10 * 48000))
        lv.append(20 * np.log10(np.sqrt(np.mean(x.mean(axis=1) ** 2)) / 20e-6))
    lv = np.array(lv)
    print(f"unweighted rms level of {len(lv)} recordings (if samples are pascal): "
          f"min {lv.min():.0f}, median {np.median(lv):.0f}, max {lv.max():.0f} dB SPL")
    if np.median(lv) < 35 or np.median(lv) > 95:
        print("WARNING: these levels are implausible for city soundscapes (expect about 50 to 85 dB SPL). "
              "The wavs may not be in pascal; check the ISD documentation before using the results.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--soundscaper", default="third_party/SoundSCaper")
    ap.add_argument("--recordings", default="data/processed/isd_recordings.csv")
    ap.add_argument("--out", default="soundaqnet/isd_soundaqnet.csv")
    ap.add_argument("--cache", default="soundaqnet/cache_isd")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--sens-n", type=int, default=150, help="recordings used for the non-full conditions")
    ap.add_argument("--limit", type=int, default=None, help="only the first N rated recordings (for tests)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rec = pd.read_csv(a.recordings)
    rec = rec[rec.n_ratings > 0].reset_index(drop=True)
    if a.limit:
        rec = rec.head(a.limit)
    print(f"{len(rec)} rated ISD recordings")
    level_report(rec)
    sens = set(rec.GroupID.sample(min(a.sens_n, len(rec)), random_state=a.seed))
    Path(a.cache).mkdir(parents=True, exist_ok=True)

    jobs = [(r.GroupID, r.wav, c, a.cache) for r in rec.itertuples() for c in a.conditions
            if c == "full" or r.GroupID in sens]
    print(f"{len(jobs)} feature jobs (workers {a.workers}); cached ones are skipped")
    with ProcessPoolExecutor(a.workers) as ex:
        paths = list(tqdm(ex.map(job, jobs, chunksize=1), total=len(jobs), desc="features"))

    from soundaqnet.model import EVENT_LABELS, PAQ_NAMES, SCENE_LABELS, SoundAQnetRunner
    run = SoundAQnetRunner(a.soundscaper, device=a.device)
    rows = []
    for (gid, _, cond, _), p in tqdm(list(zip(jobs, paths)), desc="model"):
        z = np.load(p)
        o = run.predict(z["mel"][None], z["loud"][None])
        row = {"GroupID": gid, "condition": cond, "scene": SCENE_LABELS[o["scene"][0].argmax()],
               "ISOPleasant": o["ISOPleasant"][0], "ISOEventful": o["ISOEventful"][0]}
        row.update({f"p_{s}": o["scene"][0][i] for i, s in enumerate(SCENE_LABELS)})
        row.update({f"paq_{k}": o[k][0] for k in PAQ_NAMES})
        row.update({f"ev_{e}": o["event"][0][i] for i, e in enumerate(EVENT_LABELS)})
        rows.append(row)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
