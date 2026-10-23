#!/usr/bin/env python3
"""ISOPleasant only: predicted pleasantness of generated grids against the rating in the prompt.

Input: grid folders with metrics.csv that contain sq_isop (SoundAQnet, see add_soundaqnet_to_metrics.py) and p.
One panel per run, in one row. Horizontal axis: predicted ISOPleasant. One row of dots per rating level
(lowest at the bottom). Every small dot is one generated file; the large dot with a black edge is the mean of the
level, joined by a line; the text gives the mean difference (highest minus lowest level).
If the rating moves the sound towards pleasant, the dark dots lie to the right of the light ones.

    python generation/plot_isopleasant_strip.py \\
        "ARAUS=~/grids_3msde/araus_6000_g12" "ISD=~/grids_3msde/numbers_5000_g12" \\
        "ISD+ARAUS=~/grids_3msde/joint_5000_g12" "Clotho=~/grids_3msde/clotho_5000_g12" \\
        --out ~/plots/isopleasant --slide

Writes <out>[_slide].png/.pdf and <out>[_slide]_means.csv (same columns as plot_isopleasant.py without the
eventful values). Use --controls label=folder to add the base model of the same prompts as grey dots.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAMP = ["#9ecae1", "#5b9bd5", "#2171b5", "#08306b", "#041a3d"]  # one hue, light -> dark


def load(path):
    f = Path(path).expanduser() / "metrics.csv"
    if not f.exists():
        sys.exit(f"{f} not found")
    df = pd.read_csv(f)
    for c in ("sq_isop", "p"):
        if c not in df.columns:
            sys.exit(f"{f} has no column {c}")
    df["p"] = df.p.round(4)
    return df


def level_label(p):
    return "0" if abs(p) < 1e-9 else f"{p:+g}".replace("-", "−")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=folder")
    ap.add_argument("--controls", nargs="*", default=[], help="label=folder of the base model for the run with that label")
    ap.add_argument("--zoom", type=float, default=0.5, help="half-width of the horizontal axis")
    ap.add_argument("--out", default="isopleasant")
    ap.add_argument("--slide", action="store_true", help="larger type, sized for a slide (<out>_slide.pdf/.png)")
    a = ap.parse_args()

    plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white"})
    if a.slide:
        plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "axes.titlesize": 15,
                             "xtick.labelsize": 12, "ytick.labelsize": 12})
    runs = [(it.split("=", 1)[0], load(it.split("=", 1)[1])) for it in a.runs]
    ctrl = {it.split("=", 1)[0]: load(it.split("=", 1)[1]) for it in a.controls}
    n = len(runs)
    wd, ht = (3.5, 3.0) if a.slide else (3.4, 2.8)
    fig, axes = plt.subplots(1, n, figsize=(wd * n, ht), squeeze=False)
    rng = np.random.default_rng(0)
    summary, first_colors = [], None
    for k, (ax, (label, df)) in enumerate(zip(axes.ravel(), runs)):
        levels = sorted(df.p.unique())
        cols = [RAMP[int(round(i * (len(RAMP) - 1) / max(len(levels) - 1, 1)))] for i in range(len(levels))]
        first_colors = first_colors or cols
        ax.axvline(0, color="0.7", lw=0.8, zorder=0)
        if label in ctrl:
            c = ctrl[label]
            for i, lv in enumerate(levels):
                v = c[np.isclose(c.p, lv)].sq_isop.values
                ax.scatter(v, i + rng.uniform(-0.28, 0.28, len(v)), s=16, color="#b9b8b2", alpha=0.7, zorder=1, linewidths=0)
        means = []
        for i, (lv, col) in enumerate(zip(levels, cols)):
            v = df[np.isclose(df.p, lv)].sq_isop.values
            ax.scatter(v, i + rng.uniform(-0.28, 0.28, len(v)), s=20, color=col, alpha=0.8, zorder=2, linewidths=0)
            means.append(v.mean())
        ax.plot(means, range(len(levels)), color="#0b0b0b", lw=1.3, zorder=3)
        for i, (m, col) in enumerate(zip(means, cols)):
            ax.scatter([m], [i], s=95, color=col, edgecolor="#0b0b0b", lw=1.4, zorder=4)
        d = means[-1] - means[0]
        ax.text(0.5, 1.0, "\u0394 = " + f"{d:+.2f}".replace("-", "\u2212"), transform=ax.transAxes, ha="center", va="bottom",
                fontsize=plt.rcParams["font.size"] - 1, color="0.25")
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels([level_label(p) for p in levels])
        ax.set_ylim(-0.6, len(levels) - 0.4)
        ax.set_xlim(-a.zoom, a.zoom)
        ax.set_title(label, pad=20)
        ax.set_xlabel("predicted ISOPleasant")
        if k == 0:
            ax.set_ylabel("rating in the prompt")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        summary.append({"run": label, "n_files": len(df), "levels": f"{levels[0]:g}..{levels[-1]:g}",
                        "pleasant_low": means[0], "pleasant_high": means[-1], "d_pleasant": d})
    if a.slide:
        handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=8) for c in first_colors]
        names = ["lowest rating"] + ["middle"] * (len(first_colors) - 2) + ["highest rating"]
        if ctrl:
            handles.append(plt.Line2D([], [], marker="o", ls="", color="#b9b8b2", ms=8)); names.append("base model")
        fig.legend(handles, names, loc="lower center", ncol=len(names), frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.09, 1, 1))
    else:
        fig.tight_layout()
    out = str(Path(a.out).expanduser()) + ("_slide" if a.slide else "")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    s = pd.DataFrame(summary)
    s.to_csv(f"{out}_means.csv", index=False)
    print(s.round(3).to_string(index=False))
    print(f"written {out}.png/.pdf and {out}_means.csv")


if __name__ == "__main__":
    main()
