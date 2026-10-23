#!/usr/bin/env python3
"""Gap between generated and training files, per rating group.

For each model and each requested rating (-0.7, 0, +0.7):
  gap = mean SoundAQnet ISOPleasant of the generated files - mean of the training files in the same rating group
  (training groups: caption rating <= -0.5, within +-0.15, > 0.5). Interval: 95% bootstrap of both means.
Zero = the generated files score like the training files of that rating. Only meaningful where the training scores rise with
the rating (ARAUS); for ISD the training line is flat.

    python generation/plot_gap_to_training.py \\
        --train "ARAUS=~/soundaqnet_training/araus_whole.csv" "ISD=~/soundaqnet_training/isd_all_whole.csv" \\
                "ISD+ARAUS (ISD training files)=~/soundaqnet_training/isd_all_whole.csv" \\
        --gen "ARAUS=~/grids_3msde/common_araus_6000_g12,~/grids_3msde/common6new_araus_6000_g12" ... \\
        --out ~/plots/gap_to_training [--slide]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEVELS = [-0.7, 0.0, 0.7]
SEL = [lambda x: x <= -0.5, lambda x: (x >= -0.15) & (x <= 0.15), lambda x: x > 0.5]
RNG = np.random.default_rng(0)


def read_gen(spec):
    d = pd.concat([pd.read_csv(Path(p).expanduser() / "metrics.csv")[["seed", "masker", "p", "sq_isop"]] for p in spec.split(",")])
    d["p"] = d.p.round(2)
    return d


def gap(gv, tv, n=5000):
    b = [RNG.choice(gv, len(gv)).mean() - RNG.choice(tv, len(tv)).mean() for _ in range(n)]
    return gv.mean() - tv.mean(), *np.percentile(b, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--out", default="gap_to_training")
    ap.add_argument("--slide", action="store_true")
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 13 if a.slide else 10, "figure.facecolor": "white", "savefig.facecolor": "white"})
    train = {s.split("=", 1)[0]: pd.read_csv(Path(s.split("=", 1)[1]).expanduser()) for s in a.train}
    gen = {s.split("=", 1)[0]: read_gen(s.split("=", 1)[1]) for s in a.gen}
    labels = [k for k in gen if k in train]
    fig, axes = plt.subplots(1, len(labels), figsize=((4.3 if a.slide else 4.0) * len(labels), 3.6), sharey=True, squeeze=False)
    rows = []
    for k, ax in zip(labels, axes.ravel()):
        t = train[k]; x, y = t.train_rating.values, t.sq_rating.values; g = gen[k]
        res = [gap(g[g.p == p].sq_isop.values, y[s(x)]) for p, s in zip(LEVELS, SEL)]
        m = np.array([r[0] for r in res]); lo = np.array([r[1] for r in res]); hi = np.array([r[2] for r in res])
        ax.axhline(0, color="0.3", lw=1.0)
        ax.errorbar(range(3), m, yerr=[m - lo, hi - m], fmt="o", color="#08306b", ecolor="#08306b", capsize=4, ms=8)
        for i, v in enumerate(m):
            ax.annotate(f"{v:+.2f}", (i, v), textcoords="offset points", xytext=(12, 4), fontsize=plt.rcParams["font.size"] - 2)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["very\nunpleasant\n(−0.7)", "neutral\n(0)", "very\npleasant\n(+0.7)"])
        ax.set_xlim(-0.5, 2.6)
        ax.set_title(k.replace(" (ISD training files)", ""))
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for p, r in zip(LEVELS, res):
            rows.append({"panel": k, "level": p, "gap": r[0], "ci_lo": r[1], "ci_hi": r[2]})
    axes.ravel()[0].set_ylabel("generated minus training files\n(predicted ISOPleasant)")
    fig.tight_layout()
    out = str(Path(a.out).expanduser()); Path(out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    d = pd.DataFrame(rows); d.to_csv(f"{out}_table.csv", index=False); print(d.round(3).to_string())


if __name__ == "__main__":
    main()
