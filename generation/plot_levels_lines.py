#!/usr/bin/env python3
"""Predicted pleasantness against the requested rating: training files, base model, fine-tuned model in one panel.

One panel per fine-tuned model. x = rating group (-0.7, 0, +0.7), y = SoundAQnet ISOPleasant (mean and 95% bootstrap interval).
  grey   training files in the same rating groups (caption rating <= -0.5, within +-0.15, > 0.5): what the rater sees in the
         training audio, the target a model that followed the rating would reach
  red    base model prompted with words (same prompts as the fine-tuned model): what the prompt text alone gives
  blue   fine-tuned model
The steeper the blue line is than the red one, the larger the effect of fine-tuning; the closer to the grey one, the closer to
the training data. Delta in each panel = value at +0.7 minus value at -0.7.

    python generation/plot_levels_lines.py \\
        --train "ARAUS=~/soundaqnet_training/araus_whole.csv" "ISD=~/soundaqnet_training/isd_all_whole.csv" \\
                "ISD+ARAUS (ISD training files)=~/soundaqnet_training/isd_all_whole.csv" \\
        --gen "ARAUS=~/grids_3msde/common_araus_6000_g12,~/grids_3msde/common6new_araus_6000_g12" \\
              "ISD=~/grids_3msde/common_numbers_5000_g12,~/grids_3msde/common6new_numbers_5000_g12" \\
              "ISD+ARAUS (ISD training files)=~/grids_3msde/common_joint_5000_g12,~/grids_3msde/common6new_joint_5000_g12" \\
        --base ~/grids_3msde/common6_base_words_g12 --out ~/plots/levels_lines [--slide]
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
    return d[d.p.isin(LEVELS)]


def mean_ci(v, n=5000):
    v = np.asarray(v)
    b = [v[RNG.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return v.mean(), *np.percentile(b, [2.5, 97.5])


def line_gen(g):
    return [mean_ci(g[g.p == p].sq_isop.values) for p in LEVELS]


def line_train(t):
    x, y = t.train_rating.values, t.sq_rating.values
    return [mean_ci(y[s(x)]) for s in SEL]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--base", required=True, help="folder(s) of the base model prompted with words")
    ap.add_argument("--out", default="levels_lines")
    ap.add_argument("--slide", action="store_true")
    ap.add_argument("--no-delta", action="store_true", help="leave the rise (+0.7 minus -0.7) out of the legend")
    ap.add_argument("--truth", action="store_true", help="add the listeners' rating of the training groups (ground truth)")
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 13 if a.slide else 10, "figure.facecolor": "white", "savefig.facecolor": "white"})
    train = {s.split("=", 1)[0]: pd.read_csv(Path(s.split("=", 1)[1].split(",")[0]).expanduser()) for s in a.train}
    gen = {s.split("=", 1)[0]: read_gen(s.split("=", 1)[1]) for s in a.gen}
    base = line_gen(read_gen(a.base))
    labels = [k for k in gen if k in train]
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=((4.3 if a.slide else 4.0) * n, 3.9), sharey=True, squeeze=False)
    cols = {"training files": "0.45", "base model (words)": "#C8102E", "fine-tuned": "#08306b"}
    rows = []
    for k, ax in zip(labels, axes.ravel()):
        curves = {"training files": line_train(train[k]), "base model (words)": base, "fine-tuned": line_gen(gen[k])}
        off = {"training files": -0.04, "base model (words)": 0.0, "fine-tuned": 0.04}
        for name, c in curves.items():
            m = np.array([t[0] for t in c]); lo = np.array([t[1] for t in c]); hi = np.array([t[2] for t in c])
            xs = np.array(LEVELS) + off[name]
            ax.errorbar(xs, m, yerr=[m - lo, hi - m], color=cols[name], marker="o", ms=6, capsize=3, lw=1.6,
                        ls="--" if name == "training files" else "-", label=name if a.no_delta else f"{name}  ({m[2]-m[0]:+.2f})")
            rows.append({"panel": k, "curve": name, **{f"mean_{p}": v for p, v in zip(LEVELS, m)}, "delta": m[2] - m[0]})
        if a.truth:
            t = train[k]; x = t.train_rating.values
            tr = [x[sl(x)].mean() for sl in SEL]
            ax.plot(np.array(LEVELS) - 0.04, tr, color="k", marker="s", mfc="white", ms=6, lw=1.2, ls=":",
                    label=f"listeners' rating  ({tr[2]-tr[0]:+.2f})")
            rows.append({"panel": k, "curve": "listeners' rating", **{f"mean_{p}": v for p, v in zip(LEVELS, tr)}, "delta": tr[2] - tr[0]})
        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.set_xticks(LEVELS)
        ax.set_xticklabels(["very unpleasant\n(−0.7)", "neutral\n(0)", "very pleasant\n(+0.7)"])
        ax.set_title(k.replace(" (ISD training files)", ""))
        ax.legend(frameon=False, fontsize=plt.rcParams["font.size"] - 3, loc="upper left", title=None if a.no_delta else "(+0.7 minus \u22120.7)",
                  title_fontsize=plt.rcParams["font.size"] - 3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes.ravel()[0].set_ylabel("predicted ISOPleasant\n(mean, 95% interval)")
    fig.tight_layout()
    out = str(Path(a.out).expanduser()); Path(out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    pd.DataFrame(rows).to_csv(f"{out}_table.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
