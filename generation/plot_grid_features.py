#!/usr/bin/env python3
"""Plot acoustic features of generated grids against the pleasantness value, one column per run.

Input: grid folders with metrics.csv, with the columns added by psychoacoustics_grid.py (loud_N50, sharp, rough, ...),
spectral_grid.py (centroid_hz, rolloff85_hz, tilt_db_oct) and evaluate_grid.py (A_gt4k, ...). Each run is
label=folder, optionally with the base-model control of the same prompts after an @:

    python generation/plot_grid_features.py \\
        "ARAUS=~/grids_3msde/araus_6000@~/grids_3msde/base_araus_prompts" \\
        "ISD=~/grids_3msde/numbers_5000@~/grids_3msde/base_with_numbers" \\
        "ISD+ARAUS=~/grids_3msde/joint_5000@~/grids_3msde/base_with_numbers" \\
        "Clotho=~/grids_3msde/clotho_5000@~/grids_3msde/clotho_base_with_tag" \\
        --metrics sharp centroid_hz A_gt4k loud_N50 rough --out ~/plots/features

Writes <out>_lines.png/.pdf and <out>_effects.png/.pdf and <out>_effects.csv (with --slide also <out>_effects_slide.png/.pdf: large type, short labels).

Figure 1 (lines): mean over seeds for every prompt (thin grey), mean over prompts (blue), and the control (orange, dashed)
at each pleasantness value. The y axis is shared along a row, so the size of the changes can be compared between runs.
Each run uses its own pleasantness values (Clotho: -0.4, 0, 0.4).

Figure 2 (effects): for every prompt the change from the lowest to the highest pleasantness value, in units of the
seed-to-seed standard deviation of the run (the same definition as compare_grids.py). The grey band is |effect| < 2 seed SD;
dots outside it count as "clear". With 3 seeds the seed SD itself is uncertain; this shows direction and size, not significance.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_grids import effects  # noqa: E402

LABELS = {
    "sharp": "Sharpness (acum)",
    "centroid_hz": "Spectral centroid (Hz)",
    "rolloff85_hz": "Spectral roll-off 85% (Hz)",
    "tilt_db_oct": "Spectral tilt (dB/octave)",
    "A_gt1k": "A-weighted power > 1 kHz (%)",
    "A_gt4k": "A-weighted power > 4 kHz (%)",
    "loud_N50": "Loudness N50 (sone)",
    "loud_N5": "Loudness N5 (sone)",
    "rough": "Roughness (asper)",
    "flat_db": "Spectral flatness (dB)",
    "tnr_share": "Share with prominent tone",
    "rms_db": "RMS (dB)",
    "sq_isop": "SoundAQnet ISOPleasant",
    "sspy_LAeq": "LAeq (dB, Soundscapy)",
    "sspy_S_perseg_avg": "Sharpness per segment (acum, Soundscapy)",
    "sspy_S_L_avg": "Sharpness from loudness (acum, Soundscapy)",
    "sspy_N_5": "Loudness N5 (sone, Soundscapy)",
    "sspy_R_avg": "Roughness (asper, Soundscapy)",
    "sspy_ZCR": "Zero-crossing rate (Hz, Soundscapy)",
    "sspy_ACI": "Acoustic complexity index (ACI)",
    "sspy_NDSI": "NDSI",
    "sspy_BI": "Bioacoustic index (BI)",
    "sspy_Ht": "Temporal entropy (Ht)",
    "sspy_Hf": "Spectral entropy (Hf)",
}
DEFAULT_METRICS = ["sharp", "centroid_hz", "A_gt4k", "loud_N50", "rough"]
SHORT = {"sharp": "Sharpness", "centroid_hz": "Centroid", "rolloff85_hz": "Roll-off", "tilt_db_oct": "Tilt",
         "A_gt1k": "Power > 1 kHz", "A_gt4k": "Power > 4 kHz", "loud_N50": "Loudness", "loud_N5": "Loudness N5",
         "rough": "Roughness", "flat_db": "Flatness", "sspy_ZCR": "Zero crossings", "sspy_S_perseg_avg": "Sharpness (sspy)",
         "sq_isop": "SoundAQnet", "sspy_LAeq": "LAeq", "sspy_ACI": "ACI", "sspy_NDSI": "NDSI", "sspy_BI": "BI"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9992"


def load(path):
    f = Path(path).expanduser() / "metrics.csv"
    if not f.exists():
        sys.exit(f"{f} not found")
    df = pd.read_csv(f)
    df["p"] = df.p.round(4)
    return df


def parse(items):
    runs = []
    for it in items:
        if "=" not in it:
            sys.exit(f"'{it}': use label=folder[@control_folder]")
        label, rest = it.split("=", 1)
        folder, _, base = rest.partition("@")
        runs.append((label, load(folder), load(base) if base else None))
    return runs


def style():
    plt.rcParams.update({
        "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def plot_lines(runs, metrics, out):
    nr, nc = len(metrics), len(runs)
    fig, axes = plt.subplots(nr, nc, figsize=(3.1 * nc + 1.4, 2.2 * nr + 0.9), sharey="row", squeeze=False)
    for j, (label, df, base) in enumerate(runs):
        levels = sorted(df.p.unique())
        for i, m in enumerate(metrics):
            ax = axes[i, j]
            if m not in df.columns or df[m].isna().all():
                ax.text(0.5, 0.5, f"no {m}", transform=ax.transAxes, ha="center", color=MUTED)
                continue
            per = df.groupby(["masker", "p"])[m].mean().unstack("p")[levels]
            for _, row in per.iterrows():
                ax.plot(levels, row.values, color=GREY, lw=1, alpha=0.7, zorder=2)
            ax.plot(levels, per.mean().values, color=BLUE, lw=2.2, marker="o", ms=5, zorder=4)
            if base is not None and m in base.columns:
                bl = [x for x in levels if x in set(base.p)]
                bm = base.groupby("p")[m].mean().reindex(bl)
                ax.plot(bl, bm.values, color=ORANGE, lw=2, ls="--", marker="s", ms=4, zorder=3)
            ax.set_xticks(levels)
            ax.set_xticklabels([f"{x:+g}" if x else "0" for x in levels])
            if i == 0:
                ax.set_title(f"{label}\n({df.masker.nunique()} prompts, {df.seed.nunique()} seeds)", fontsize=11, color=INK)
            if j == 0:
                ax.set_ylabel(LABELS.get(m, m), fontsize=10)
            if i == nr - 1:
                ax.set_xlabel("pleasantness value in the prompt")
    handles = [plt.Line2D([], [], color=GREY, lw=1), plt.Line2D([], [], color=BLUE, lw=2.2, marker="o"),
               plt.Line2D([], [], color=ORANGE, lw=2, ls="--", marker="s")]
    fig.legend(handles, ["one prompt (mean of seeds)", "mean of prompts", "base model, same prompts"],
               loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}_lines.{ext}", dpi=200)
    plt.close(fig)


def plot_effects(runs, metrics, out, k=2.0, slide=False, xlim=None):
    frames = [df.assign(grid=label) for label, df, _ in runs]
    metrics = [m for m in metrics if all(m in f.columns for f in frames)]
    eff = effects(pd.concat(frames, ignore_index=True), metrics, k)
    eff.to_csv(f"{out}_effects.csv", index=False)
    nc = len(runs)
    if slide:   # large type for a beamer slide of about 11 cm width
        plt.rcParams.update({"font.size": 14, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 13})
        fig, axes = plt.subplots(1, nc, figsize=(8.6, 0.6 * len(metrics) + 1.5), sharex=True, sharey=True, squeeze=False)
    else:
        fig, axes = plt.subplots(1, nc, figsize=(3.0 * nc + 1.6, 0.55 * len(metrics) + 1.8), sharex=True, sharey=True, squeeze=False)
    lim = float(np.clip(np.nanpercentile(np.abs(eff.ratio), 95) * 1.15, 4.0, 20.0))
    if xlim:
        lim = float(xlim)  # fixed axis, so that figures of different runs share one scale  # points beyond are drawn at the edge as arrows
    for j, (label, _, _) in enumerate(runs):
        ax = axes[0, j]
        e = eff[eff.grid == label]
        ax.axvspan(-k, k, color=GRID, zorder=0)
        ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
        for i, m in enumerate(metrics):
            g = e[e.metric == m]
            yy = np.full(len(g), i) + np.linspace(-0.12, 0.12, len(g)) if len(g) > 1 else np.full(len(g), float(i))
            r = g.ratio.values
            beyond = np.abs(r) > lim
            rc = np.clip(r, -lim, lim)
            clear = g.clear.values
            for sel, kw in ((~clear, dict(facecolor="white", edgecolor=GREY, lw=1.3, zorder=3)), (clear, dict(color=BLUE, zorder=4))):
                for flag, arrow in ((False, False), (True, True)):
                    m_ = sel & (beyond == flag)
                    if not m_.any():
                        continue
                    if arrow:
                        for sgn, mark in ((1, ">"), (-1, "<")):
                            mm = m_ & (np.sign(r) == sgn)
                            if mm.any():
                                ax.scatter(rc[mm], yy[mm], s=44, marker=mark, **kw)
                    else:
                        ax.scatter(rc[m_], yy[m_], s=36, marker="o", **kw)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels([(SHORT if slide else LABELS).get(m, m) for m in metrics], fontsize=13 if slide else 10)
        ax.set_ylim(len(metrics) - 0.5, -0.5)
        ax.set_xlim(-lim, lim)
        ax.set_title(label, fontsize=14 if slide else 11, color=INK)
        ax.set_xlabel("change in seed SD" if slide else "effect (high - low) in seed SD")
        ax.grid(axis="y", visible=False)
    if not slide:
        fig.text(0.5, 0.012, "grey band: |effect| < 2 seed SD; filled: clear; open: not clear; one dot per prompt; arrow: beyond the axis",
                 ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0 if slide else 0.07, 1, 1))
    name = f"{out}_effects_slide" if slide else f"{out}_effects"
    for ext in ("png", "pdf"):
        fig.savefig(f"{name}.{ext}", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=folder[@control_folder]")
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    ap.add_argument("--out", default="features")
    ap.add_argument("--xlim", type=float, default=None, help="fixed x axis limit (+-) of the effects figures, in seed SD")
    ap.add_argument("--slide", action="store_true", help="also write <out>_effects_slide.pdf/.png: larger type, short labels, for a slide")
    a = ap.parse_args()
    style()
    out = str(Path(a.out).expanduser())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    runs = parse(a.runs)
    plot_lines(runs, a.metrics, out)
    plot_effects(runs, a.metrics, out, xlim=a.xlim)
    if a.slide:
        plot_effects(runs, a.metrics, out, slide=True, xlim=a.xlim)
    print(f"written {out}_lines.png/.pdf, {out}_effects.png/.pdf, {out}_effects.csv")


if __name__ == "__main__":
    main()
