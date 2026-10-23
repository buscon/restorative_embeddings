#!/usr/bin/env python3
"""Training data against the fine-tuned model: does the model reproduce the relationship that the rater sees in the training audio?

For each model two sets of SoundAQnet ISOPleasant values are put on the same axes:
  training files   x = rating in the caption, y = SoundAQnet prediction (soundaqnet/03_rate_training_sets.py: columns
                   train_rating, sq_rating), with a straight-line fit and binned means;
  generated files  x = rating in the prompt (-0.7, 0, +0.7), y = SoundAQnet prediction (metrics.csv of a grid, column sq_isop).
The rater is the same and its settings are the same, so the two are comparable. If the model had learned the rating as well
as the rater can see it, the generated means would lie on the line of the training data.

  expected effect  = fit(+0.7) - fit(-0.7)  on the training data (interval: bootstrap over training files)
  generated effect = mean over (prompt, seed) pairs of SoundAQnet(+0.7) - SoundAQnet(-0.7)  (interval: bootstrap over pairs)
  ratio            = generated / expected

Reading it: SoundAQnet compresses the rating scale and, on ISD, hardly follows the listeners' ratings (a flat line), so a
flat training line means the rater cannot see the rating in that audio and a generated effect cannot be called "following the
training data" either way.

    python generation/compare_training_vs_generated.py \\
        --train "ARAUS=~/soundaqnet_training/araus_whole.csv" "ISD=~/soundaqnet_training/isd_all_whole.csv" \\
                "ISD+ARAUS=~/soundaqnet_training/joint_whole_clean04.csv" \\
        --gen   "ARAUS=~/grids_3msde/araus_6000_g12" "ISD=~/grids_3msde/numbers_5000_g12" \\
                "ISD+ARAUS=~/grids_3msde/joint_5000_g12" --out ~/plots/training_vs_generated [--slide]
Several csv files or grid folders for one model can be joined with commas.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEVEL_COLORS = {-0.7: "#9ecae1", 0.0: "#2171b5", 0.7: "#041a3d"}


def read_train(spec):
    d = pd.concat([pd.read_csv(Path(p).expanduser()) for p in spec.split(",")], ignore_index=True)
    for c in ("train_rating", "sq_rating"):
        if c not in d.columns:
            sys.exit(f"column {c} missing in {spec}")
    return d[["train_rating", "sq_rating"]].dropna()


def read_gen(spec):
    fr = []
    for p in spec.split(","):
        f = Path(p).expanduser() / "metrics.csv"
        d = pd.read_csv(f)
        if "sq_isop" not in d.columns:
            sys.exit(f"{f} has no sq_isop")
        fr.append(d[["seed", "masker", "p", "sq_isop"]])
    d = pd.concat(fr, ignore_index=True)
    d["p"] = d.p.round(2)
    return d[d.p.isin([-0.7, 0.0, 0.7])]


def fit(x, y):
    b, a0 = np.polyfit(x, y, 1)
    return b, a0


def boot(fn, n_items, rng, n=5000):
    return np.percentile([fn(rng.integers(0, n_items, n_items)) for _ in range(n)], [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True, help="label=csv[,csv...]")
    ap.add_argument("--gen", nargs="+", required=True, help="label=folder[,folder...]")
    ap.add_argument("--out", default="training_vs_generated")
    ap.add_argument("--slide", action="store_true")
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 13 if a.slide else 10, "figure.facecolor": "white", "savefig.facecolor": "white"})
    rng = np.random.default_rng(0)
    train = {s.split("=", 1)[0]: read_train(s.split("=", 1)[1]) for s in a.train}
    gen = {s.split("=", 1)[0]: read_gen(s.split("=", 1)[1]) for s in a.gen}
    labels = [k for k in train if k in gen]

    rows = []
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=((4.0 if a.slide else 3.8) * n, 3.6), sharey=True, squeeze=False)
    for k, ax in zip(labels, axes.ravel()):
        t, g = train[k], gen[k]
        x, y = t.train_rating.values, t.sq_rating.values
        b, a0 = fit(x, y)
        exp = b * 1.4
        lo, hi = boot(lambda i: fit(x[i], y[i])[0] * 1.4, len(x), rng)
        hi_g = g[g.p == 0.7].set_index(["masker", "seed"]).sq_isop
        lo_g = g[g.p == -0.7].set_index(["masker", "seed"]).sq_isop
        d = (hi_g - lo_g).dropna().values
        glo, ghi = boot(lambda i: d[i].mean(), len(d), rng)
        r = np.corrcoef(x, y)[0, 1]
        # empirical difference between the extreme training groups (|rating| from 0.5 to 0.9)
        emp = (y[(x >= 0.5) & (x <= 0.9)].mean() - y[(x <= -0.5) & (x >= -0.9)].mean()) if ((x >= 0.5) & (x <= 0.9)).any() and ((x <= -0.5) & (x >= -0.9)).any() else np.nan
        rows.append({"model": k, "n_train": len(x), "r_train": r, "slope": b, "expected_delta": exp,
                     "exp_lo": lo, "exp_hi": hi, "extreme_groups_delta": emp, "generated_delta": d.mean(),
                     "gen_lo": glo, "gen_hi": ghi, "n_pairs": len(d),
                     "ratio_generated_to_expected": d.mean() / exp if abs(exp) > 1e-9 else np.nan,
                     "bias_at_-0.7": g[g.p == -0.7].sq_isop.mean() - (a0 - 0.7 * b),
                     "bias_at_0": g[g.p == 0.0].sq_isop.mean() - a0,
                     "bias_at_+0.7": g[g.p == 0.7].sq_isop.mean() - (a0 + 0.7 * b)})
        ax.scatter(x, y, s=6, color="0.78", linewidths=0, zorder=1)
        edges = np.linspace(-1, 1, 11)
        c = (edges[:-1] + edges[1:]) / 2
        m = [y[(x >= e0) & (x < e1)].mean() if ((x >= e0) & (x < e1)).sum() >= 5 else np.nan for e0, e1 in zip(edges[:-1], edges[1:])]
        ax.plot(c, m, color="0.35", lw=1.2, zorder=2)
        xs = np.array([-1, 1])
        ax.plot(xs, a0 + b * xs, color="0.2", lw=1.0, ls="--", zorder=2)
        for p, col in LEVEL_COLORS.items():
            v = g[g.p == p].sq_isop.values
            ax.scatter(np.full(len(v), p) + rng.uniform(-0.04, 0.04, len(v)), v, s=14, color=col, alpha=0.6, linewidths=0, zorder=3)
            ax.scatter([p], [v.mean()], s=110, color=col, edgecolor="#0b0b0b", lw=1.4, zorder=5)
        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.set_title(f"{k} model")
        ax.set_xlabel("rating (caption or prompt)")
        ax.set_xlim(-1.05, 1.05)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes.ravel()[0].set_ylabel("SoundAQnet ISOPleasant")
    handles = [plt.Line2D([], [], marker="o", ls="", color="0.78", ms=6), plt.Line2D([], [], color="0.2", ls="--"),
               plt.Line2D([], [], marker="o", ls="", color="#2171b5", mec="#0b0b0b", ms=9)]
    fig.legend(handles, ["training files", "line fitted to training files", "generated files (large dot = mean)"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = str(Path(a.out).expanduser())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    # second figure: box plots, training files against generated files in the same three rating groups
    groups = [("very unpleasant\n(\u22120.7)", -0.7, lambda x: x <= -0.5),
              ("neutral\n(0)", 0.0, lambda x: (x >= -0.15) & (x <= 0.15)),
              ("very pleasant\n(+0.7)", 0.7, lambda x: x > 0.5)]
    fig, axes = plt.subplots(1, n, figsize=((4.4 if a.slide else 4.2) * n, 3.8), sharey=True, squeeze=False)
    for k, ax in zip(labels, axes.ravel()):
        t, g = train[k], gen[k]
        for gi, (name, p, sel) in enumerate(groups):
            tv = t.sq_rating.values[sel(t.train_rating.values)]
            gv = g[g.p == p].sq_isop.values
            for off, v, col, lab in ((-0.19, tv, "0.72", "training"), (0.19, gv, LEVEL_COLORS[p], "generated")):
                bp = ax.boxplot([v], positions=[gi + off], widths=0.32, patch_artist=True, showfliers=False,
                                medianprops=dict(color="#0b0b0b", lw=1.4), whiskerprops=dict(color="0.4"),
                                capprops=dict(color="0.4"), boxprops=dict(color="0.4"))
                bp["boxes"][0].set_facecolor(col)
                ax.scatter([gi + off], [v.mean()], marker="D", s=26, color="white", edgecolor="#0b0b0b", zorder=5, lw=0.9)
                ax.text(gi + off, ax.get_ylim()[0] if False else -0.78, f"n={len(v)}", ha="center", va="bottom",
                        fontsize=plt.rcParams["font.size"] - 4, color="0.35")
            ax.scatter(np.full(len(gv), gi + 0.19) + rng.uniform(-0.07, 0.07, len(gv)), gv, s=9, color="#0b0b0b", alpha=0.35, linewidths=0, zorder=4)
        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.set_xticks(range(3))
        ax.set_xticklabels([x[0] for x in groups])
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(-0.82, 0.62)
        ax.set_title(f"{k}: training and generated")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes.ravel()[0].set_ylabel("SoundAQnet ISOPleasant")
    hb = [plt.Rectangle((0, 0), 1, 1, fc="0.72", ec="0.4"), plt.Rectangle((0, 0), 1, 1, fc="#2171b5", ec="0.4"),
          plt.Line2D([], [], marker="D", ls="", color="white", mec="#0b0b0b", ms=6)]
    fig.legend(hb, ["training files (rating in the caption)", "generated files (rating in the prompt)", "mean"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}_boxes.{ext}", dpi=200)

    tab = pd.DataFrame(rows)
    tab.to_csv(out + "_table.csv", index=False)
    print(tab.round(3).to_string(index=False))
    print("written", out + ".pdf", out + "_boxes.pdf", out + "_table.csv")


if __name__ == "__main__":
    main()
