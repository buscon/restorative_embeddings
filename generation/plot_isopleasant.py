#!/usr/bin/env python3
"""ISOPleasant / ISOEventful (circumplex) plot of generated grids, made with Soundscapy.

Input: grid folders with metrics.csv that contain sq_isop and sq_isev, the SoundAQnet predictions written by
clotho/10_score_grid_soundaqnet.py and merged by generation/add_soundaqnet_to_metrics.py (the ISOEventful column exists
only in files written after that change: rerun  MEASURES=sq bash isd/12_measure_grids.sh <grid>  once; the features are
cached). Each run is label=folder:

    python generation/plot_isopleasant.py \\
        "ARAUS=~/grids_3msde/araus_6000" "ISD (numbers)=~/grids_3msde/numbers_5000" \\
        "ISD+ARAUS=~/grids_3msde/joint_5000" "Clotho=~/grids_3msde/clotho_5000" --out ~/plots/isopleasant

One panel per run (Soundscapy scatter plot on the circumplex axes). Every dot is a generated file, coloured by the
pleasantness value in its prompt (light = lowest, dark = highest); the large dots with a black edge are the means of
the levels, joined by a line from the lowest to the highest level. Use --controls label=folder to draw the control
(base model) in the same panel as grey dots. --zoom sets the half-width of the axes (SoundAQnet predictions are
compressed: about +-0.3 on ARAUS, so the default 0.5 shows the structure; use 1 for the full circumplex).

Reading it: if the rating moves the sound along the pleasant axis, the dark dots lie to the right of the light dots.
Predictions are from SoundAQnet, which is a weak rater here (see the validation); the plot shows what the rater sees,
not what listeners hear.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAMP = ["#c6dbef", "#6baed6", "#2171b5", "#08306b", "#041a3d"]  # one hue, light -> dark


def load(path):
    f = Path(path).expanduser() / "metrics.csv"
    if not f.exists():
        sys.exit(f"{f} not found")
    df = pd.read_csv(f)
    for c in ("sq_isop", "sq_isev", "p"):
        if c not in df.columns:
            sys.exit(f"{f} has no column {c}: run isd/12_measure_grids.sh (MEASURES=sq) and generation/add_soundaqnet_to_metrics.py")
    df["p"] = df.p.round(4)
    return df


def level_label(p):
    return "0" if abs(p) < 1e-9 else f"{p:+g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=folder")
    ap.add_argument("--controls", nargs="*", default=[], help="label=folder of the control for the run with that label")
    ap.add_argument("--zoom", type=float, default=0.5)
    ap.add_argument("--out", default="isopleasant")
    a = ap.parse_args()

    import soundscapy.plotting as sp
    plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white"})
    runs = [(it.split("=", 1)[0], load(it.split("=", 1)[1])) for it in a.runs]
    ctrl = {it.split("=", 1)[0]: load(it.split("=", 1)[1]) for it in a.controls}

    n = len(runs)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 5.2 * nrows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    summary = []
    for ax, (label, df) in zip(axes.ravel(), runs):
        levels = sorted(df.p.unique())
        colors = {level_label(p): RAMP[int(round(i * (len(RAMP) - 1) / max(len(levels) - 1, 1)))] for i, p in enumerate(levels)}
        d = df.rename(columns={"sq_isop": "ISOPleasant", "sq_isev": "ISOEventful"}).copy()
        d["level"] = d.p.map(level_label)
        d["level"] = pd.Categorical(d.level, [level_label(p) for p in levels])
        if label in ctrl:
            c = ctrl[label].rename(columns={"sq_isop": "ISOPleasant", "sq_isev": "ISOEventful"})
            ax.scatter(c.ISOPleasant, c.ISOEventful, s=14, color="#b9b8b2", alpha=0.6, zorder=1, label="base model")
        sp.scatter(d, ax=ax, x="ISOPleasant", y="ISOEventful", hue="level", palette=colors, title=label, legend="brief")
        means = d.groupby("p")[["ISOPleasant", "ISOEventful"]].mean().loc[levels]
        ax.plot(means.ISOPleasant, means.ISOEventful, color="#0b0b0b", lw=1.4, zorder=6)
        for (p, r), lv in zip(means.iterrows(), levels):
            ax.scatter([r.ISOPleasant], [r.ISOEventful], s=90, color=colors[level_label(lv)], edgecolor="#0b0b0b", lw=1.4, zorder=7)
        ax.set_xlim(-a.zoom, a.zoom)
        ax.set_ylim(-a.zoom, a.zoom)
        summary.append({"run": label, "n_files": len(d), "levels": f"{levels[0]:g}..{levels[-1]:g}",
                        "pleasant_low": means.ISOPleasant.iloc[0], "pleasant_high": means.ISOPleasant.iloc[-1],
                        "eventful_low": means.ISOEventful.iloc[0], "eventful_high": means.ISOEventful.iloc[-1]})
    fig.tight_layout()
    out = str(Path(a.out).expanduser())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    s = pd.DataFrame(summary)
    s["d_pleasant"] = s.pleasant_high - s.pleasant_low
    s["d_eventful"] = s.eventful_high - s.eventful_low
    s.to_csv(f"{out}_means.csv", index=False)
    print(s.round(3).to_string(index=False))
    print(f"written {out}.png/.pdf and {out}_means.csv")


if __name__ == "__main__":
    main()
