#!/usr/bin/env python3
"""Did the fine-tuning change the predicted pleasantness more than prompting the base model with words?

Input: one or more models, each = label=folder[,folder...]; the folders (grids with metrics.csv containing sq_isop) are
merged, so the three old common prompts and the three new ones can be read as one model. Only the levels -0.7 and +0.7
are used. For every (prompt, seed) pair the effect is  d = SoundAQnet ISOPleasant at +0.7 minus at -0.7.
The same seed gives the same starting noise at both levels, so d is a paired difference.

  mean d per model, with a 95% bootstrap interval over the (prompt, seed) pairs;
  difference to the reference model (--ref), with an interval from resampling the pairs the two models share.

Writes <out>_table.csv, <out>_delta.pdf/.png (effect per model with interval) and <out>_dots.pdf/.png (every file,
one panel per model, rows = rating in the prompt).

    python generation/compare_common6.py \\
        "base (words)=~/grids_3msde/common6_base_words_g12" \\
        "base (numbers)=~/grids_3msde/common_base_g12,~/grids_3msde/common6new_base_g12" \\
        "ARAUS=~/grids_3msde/common_araus_6000_g12,~/grids_3msde/common6new_araus_6000_g12" \\
        "ISD=~/grids_3msde/common_numbers_5000_g12,~/grids_3msde/common6new_numbers_5000_g12" \\
        "ISD+ARAUS=~/grids_3msde/common_joint_5000_g12,~/grids_3msde/common6new_joint_5000_g12" \\
        --ref "base (words)" --out ~/plots/common6
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAMP = ["#9ecae1", "#2171b5", "#041a3d"]  # lowest, middle, highest rating


def load_model(spec):
    frames = []
    for folder in spec.split(","):
        f = Path(folder).expanduser() / "metrics.csv"
        if not f.exists():
            sys.exit(f"{f} not found")
        d = pd.read_csv(f)
        if "sq_isop" not in d.columns:
            sys.exit(f"{f} has no sq_isop (run isd/12_measure_grids.sh)")
        frames.append(d[["seed", "masker", "p", "sq_isop"]])
    d = pd.concat(frames, ignore_index=True)
    d["p"] = d.p.round(2)
    return d[d.p.isin([-0.7, 0.0, 0.7])]


def pair_effects(d):
    hi = d[d.p == 0.7].set_index(["masker", "seed"]).sq_isop
    lo = d[d.p == -0.7].set_index(["masker", "seed"]).sq_isop
    return (hi - lo).dropna()


def boot_ci(x, rng, n=10000):
    x = np.asarray(x, float)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="label=folder[,folder...]")
    ap.add_argument("--ref", default=None, help="label of the reference model (the base model with words)")
    ap.add_argument("--out", default="common6")
    ap.add_argument("--slide", action="store_true", help="larger type")
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 13 if a.slide else 10, "figure.facecolor": "white", "savefig.facecolor": "white"})
    rng = np.random.default_rng(0)
    data = {}
    for it in a.models:
        label, spec = it.split("=", 1)
        data[label] = load_model(spec)
    eff = {k: pair_effects(v) for k, v in data.items()}
    ref = eff.get(a.ref) if a.ref else None
    rows = []
    for k, e in eff.items():
        lo, hi = boot_ci(e.values, rng)
        r = {"model": k, "n_pairs": len(e), "n_prompts": e.index.get_level_values(0).nunique(),
             "delta": e.mean(), "ci_lo": lo, "ci_hi": hi}
        if ref is not None and k != a.ref:
            common = e.index.intersection(ref.index)
            diff = (e.loc[common] - ref.loc[common]).values
            dlo, dhi = boot_ci(diff, rng)
            r.update(diff_vs_ref=diff.mean(), diff_lo=dlo, diff_hi=dhi, n_shared=len(common))
        rows.append(r)
    tab = pd.DataFrame(rows)
    out = str(Path(a.out).expanduser())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out + "_table.csv", index=False)
    print(tab.round(3).to_string(index=False))

    # figure 1: effect per model with interval
    labels = list(eff)
    fig, ax = plt.subplots(figsize=(6.4 if a.slide else 5.6, 0.62 * len(labels) + 1.2))
    for i, k in enumerate(labels[::-1]):
        r = tab[tab.model == k].iloc[0]
        ax.errorbar(r.delta, i, xerr=[[r.delta - r.ci_lo], [r.ci_hi - r.delta]], fmt="o", color="#08306b",
                    ecolor="#08306b", capsize=4, ms=8)
        e = eff[k]
        ax.scatter(e.values, np.full(len(e), i) + np.linspace(-0.18, 0.18, len(e)), s=10, color="#9ecae1", zorder=0, linewidths=0)
    ax.axvline(0, color="0.5", lw=0.8)
    if ref is not None:
        ax.axvline(ref.mean(), color="#C8102E", lw=1.0, ls="--")
        ax.text(ref.mean(), len(labels) - 0.45, " " + a.ref, color="#C8102E", fontsize=plt.rcParams["font.size"] - 2, va="bottom")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel("predicted ISOPleasant, +0.7 minus −0.7\n(mean and 95% interval)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}_delta.{ext}", dpi=200)

    # figure 2: every file, one panel per model
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=((3.0 if a.slide else 2.8) * n, 3.0), sharex=True, squeeze=False)
    for k, ax in zip(labels, axes.ravel()):
        d = data[k]
        lv = [-0.7, 0.0, 0.7]
        for i, (p, c) in enumerate(zip(lv, RAMP)):
            v = d[d.p == p].sq_isop.values
            ax.scatter(v, i + rng.uniform(-0.28, 0.28, len(v)), s=16, color=c, alpha=0.8, linewidths=0)
        m = [d[d.p == p].sq_isop.mean() for p in lv]
        ax.plot(m, range(3), color="#0b0b0b", lw=1.2)
        for i, (x, c) in enumerate(zip(m, RAMP)):
            ax.scatter([x], [i], s=80, color=c, edgecolor="#0b0b0b", lw=1.3, zorder=4)
        ax.axvline(0, color="0.75", lw=0.8, zorder=0)
        ax.set_yticks(range(3))
        ax.set_yticklabels(["−0.7", "0", "+0.7"] if ax is axes.ravel()[0] else ["", "", ""])
        ax.set_title(k, fontsize=plt.rcParams["font.size"] - 1)
        ax.set_xlabel("predicted ISOPleasant")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes.ravel()[0].set_ylabel("rating in the prompt")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}_dots.{ext}", dpi=200)
    print("written", out + "_table.csv", out + "_delta.pdf", out + "_dots.pdf")


if __name__ == "__main__":
    main()
