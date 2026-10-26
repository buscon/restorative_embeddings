#!/usr/bin/env python3
"""Does a generated file contain a low steady tone (hum), and does it switch on and off?

For every wav (mono mix):
  hum_peak_hz   frequency of the strongest line between 30 and 130 Hz (mean spectrum)
  hum_prom_db   level of that line above the median spectrum between 200 and 2000 Hz (dB, power per Hz)
  low_vs_mid_db power in 0-200 Hz minus power in 200-6000 Hz (dB, total power in the band)
  on_frac       share of 0.25 s frames in which the 30-130 Hz band is within 6 dB of its loudest frame
  switches      number of on/off changes of that band over the file (threshold halfway between its 10th and 90th percentile)
A file with a hum has a large hum_prom_db (the 50 Hz-type hum of the first test file: about 42 dB) and, if the
hum is intermittent, a low on_frac and several switches. Written as <out>_files.csv and <out>_summary.csv (mean and sd per label).

    python generation/hum_check.py "original=~/grids_3msde/base_ae_g12" "no night=~/grids_3msde/silence_test/A_no_night" ... --out ~/plots_silence/hum
    (folders: all wav files are used; with a grid.csv, only files whose prompt_id starts with --prefix, e.g. ae_silence_2)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import medfilt, spectrogram


def analyse(path):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    m = x.mean(1)
    f, t, S = spectrogram(m, sr, nperseg=int(sr * 0.5), noverlap=int(sr * 0.25), mode="psd")   # 0.5 s windows, 0.25 s hop, 2 Hz bins
    mean = S.mean(1)
    lo = (f >= 30) & (f <= 130)
    mid = (f >= 200) & (f <= 2000)
    k = np.argmax(np.where(lo, mean, 0))
    prom = 10 * np.log10(mean[k] + 1e-20) - 10 * np.log10(np.median(mean[mid]) + 1e-20)
    band = lambda a, b: 10 * np.log10(mean[(f >= a) & (f < b)].sum() + 1e-20)  # noqa: E731
    env = 10 * np.log10(S[lo].sum(0) + 1e-20)
    on = env > env.max() - 6
    hi, lw = np.percentile(env, 90), np.percentile(env, 10)
    state = medfilt((env > (hi + lw) / 2).astype(float), 3) > 0.5   # 3 frames: ignore single-frame flicker
    return {"hum_peak_hz": f[k], "hum_prom_db": prom, "low_vs_mid_db": band(0, 200) - band(200, 6000),
            "on_frac": float(on.mean()), "switches": int((state[1:] != state[:-1]).sum()) if hi - lw > 6 else 0,
            "peak": float(np.abs(x).max()), "rms_dbfs": 20 * np.log10(np.sqrt((m ** 2).mean()) + 1e-12)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", nargs="+", help='"label=folder" (several folders can be joined with a comma)')
    ap.add_argument("--prefix", default=None, help="with a grid.csv: only prompt_id starting with this (e.g. ae_silence_2)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for s in a.sets:
        label, folders = s.split("=", 1)
        for fo in folders.split(","):
            fo = Path(fo).expanduser()
            files = sorted(fo.glob("*.wav"))
            if a.prefix and (fo / "grid.csv").exists():
                g = pd.read_csv(fo / "grid.csv")
                if g.prompt_id.astype(str).str.startswith("ae_").any():   # only the base_ae grid has many prompts
                    keep = set(g[g.prompt_id.astype(str).str.startswith(a.prefix)].file)
                    files = [p for p in files if p.name in keep]
            for p in files:
                rows.append({"label": label, "file": p.name, **analyse(p)})
    d = pd.DataFrame(rows)
    Path(a.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(f"{a.out}_files.csv", index=False)
    order = list(dict.fromkeys(d.label))
    cols = ["hum_peak_hz", "hum_prom_db", "low_vs_mid_db", "on_frac", "switches"]
    summ = d.groupby("label")[cols].agg(["mean", "std"]).reindex(order)
    summ.insert(0, "n", d.groupby("label").size().reindex(order))
    summ.to_csv(f"{a.out}_summary.csv")
    print(summ.round(2).to_string())
    print(f"\nwrote {a.out}_files.csv and {a.out}_summary.csv")


if __name__ == "__main__":
    main()
