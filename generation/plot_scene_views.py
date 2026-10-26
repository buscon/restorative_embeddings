#!/usr/bin/env python3
"""Scene views of SoundAQnet predictions for generated clips, with a real-ARAUS baseline (SoundSCaper README figures 9 to 11).

Scene = the most probable of the three predicted scene classes (public square, park, street traffic), for generated and reference clips alike
(the generation prompts have no scene label).
  <out>_radar.png     (fig. 9)  mean of the 8 attributes per scene on a radar plot, generated (solid) and reference (dashed)
  <out>_scatter.png   (fig. 10) ISOPleasant x ISOEventful per scene: generated points, density contours of generated (colour) and reference (grey)
  <out>_joint.png     (fig. 11) all scenes in one scatter with marginal densities of ISOPleasant (top) and ISOEventful (right)
  <out>_summary.csv   clips per scene and mean ISOPleasant/ISOEventful, and the Kolmogorov-Smirnov distance generated vs reference
Densities are drawn only for scenes with at least 10 clips. The attribute axes of the radar plot follow the circumplex order, not the paper's.

    python generation/plot_scene_views.py --grid ~/grids_3msde/base_ae_g12 --ref ~/grids_3msde/araus_ref/soundaqnet_full_65.csv \
        --ref-name "ARAUS (65 dB)" --out ~/plots_base_ae/scenes_ref65
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ks_2samp

SCENES = ["public_square", "park", "street_traffic"]
LABEL = {"public_square": "public square", "park": "park", "street_traffic": "street traffic"}
COLOUR = {"public_square": "#7b3294", "park": "#1b9e77", "street_traffic": "#d95f02"}
CIRCUMPLEX = ["pleasant", "vibrant", "eventful", "chaotic", "annoying", "monotonous", "uneventful", "calm"]


def with_scene(d):
    d = d.copy()
    d["scene"] = np.array(SCENES)[d[[f"p_{s}" for s in SCENES]].to_numpy().argmax(1)]
    return d


def density(ax, x, y, **kw):
    if len(x) < 10:
        return
    k = gaussian_kde(np.vstack([x, y]))
    gx, gy = np.meshgrid(np.linspace(*ax.get_xlim(), 80), np.linspace(*ax.get_ylim(), 80))
    z = k(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contour(gx, gy, z, levels=4, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True, help="folder with soundaqnet_full.csv (generated clips)")
    ap.add_argument("--ref", default=None, help="soundaqnet_full csv of the reference set")
    ap.add_argument("--ref-name", default="reference")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = with_scene(pd.read_csv(Path(a.grid).expanduser() / "soundaqnet_full.csv"))
    r = with_scene(pd.read_csv(Path(a.ref).expanduser())) if a.ref else None
    lim = max(0.1, np.abs(np.r_[g.ISOPleasant, g.ISOEventful, r.ISOPleasant if r is not None else [0],
                               r.ISOEventful if r is not None else [0]]).max() * 1.1)

    rows = []
    for name, d in (("generated", g), ("reference", r)):
        if d is None:
            continue
        for s in SCENES:
            x = d[d.scene == s]
            rows.append({"set": name, "scene": s, "n": len(x), "ISOPleasant": x.ISOPleasant.mean(), "ISOEventful": x.ISOEventful.mean()})
    summ = pd.DataFrame(rows)
    if r is not None:
        ks = []
        for s in SCENES:
            gg, rr = g[g.scene == s], r[r.scene == s]
            ks.append({"scene": s, **{f"KS_{c}": (ks_2samp(gg[c], rr[c])[0] if len(gg) > 4 and len(rr) > 4 else np.nan)
                                      for c in ("ISOPleasant", "ISOEventful")}})
        ks.append({"scene": "all", **{f"KS_{c}": ks_2samp(g[c], r[c])[0] for c in ("ISOPleasant", "ISOEventful")}})
        summ = summ.merge(pd.DataFrame(ks), on="scene", how="left")
    summ.to_csv(f"{a.out}_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))

    # fig. 9: radar
    ang = np.linspace(0, 2 * np.pi, len(CIRCUMPLEX), endpoint=False).tolist()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={"polar": True}, gridspec_kw={"wspace": 0.55})
    for ax, s in zip(axs, SCENES):
        for d, ls, nm in ((g, "-", "generated"), (r, "--", a.ref_name)):
            if d is None or not (d.scene == s).any():
                continue
            v = [d[d.scene == s][f"paq_{c}"].mean() for c in CIRCUMPLEX]
            ax.plot(ang + ang[:1], v + v[:1], ls, color=COLOUR[s] if d is g else "0.4", label=f"{nm} (n={int((d.scene == s).sum())})")
        ax.set_xticks(ang, CIRCUMPLEX, fontsize=8)
        ax.set_ylim(1, 5)
        ax.set_rticks([2, 3, 4])
        ax.set_rlabel_position(112)
        ax.set_title(LABEL[s], fontsize=10)
        ax.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.15, -0.12))
    fig.savefig(f"{a.out}_radar.png", dpi=200, bbox_inches="tight")

    # fig. 10: scatter and density per scene
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True, sharey=True)
    for ax, s in zip(axs, SCENES):
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        x = g[g.scene == s]
        ax.scatter(x.ISOPleasant, x.ISOEventful, s=10, color=COLOUR[s], alpha=0.6, label=f"generated (n={len(x)})")
        density(ax, x.ISOPleasant, x.ISOEventful, colors=COLOUR[s], linewidths=1)
        if r is not None:
            y = r[r.scene == s]
            density(ax, y.ISOPleasant, y.ISOEventful, colors="0.4", linewidths=1, linestyles="--")
            ax.plot([], [], "--", color="0.4", label=f"{a.ref_name} density (n={len(y)})")
        ax.set_title(LABEL[s], fontsize=10)
        ax.set_xlabel("ISOPleasant")
        ax.legend(fontsize=7, loc="lower left")
    axs[0].set_ylabel("ISOEventful")
    fig.savefig(f"{a.out}_scatter.png", dpi=200, bbox_inches="tight")

    # fig. 11: joint with marginals
    fig = plt.figure(figsize=(7.5, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    top = fig.add_subplot(gs[0, 0], sharex=ax)
    right = fig.add_subplot(gs[1, 1], sharey=ax)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    grid = np.linspace(-lim, lim, 200)
    for s in SCENES:
        x = g[g.scene == s]
        ax.scatter(x.ISOPleasant, x.ISOEventful, s=10, color=COLOUR[s], alpha=0.6, label=f"{LABEL[s]} (n={len(x)})")
        if len(x) >= 10:
            top.plot(grid, gaussian_kde(x.ISOPleasant)(grid), color=COLOUR[s])
            right.plot(gaussian_kde(x.ISOEventful)(grid), grid, color=COLOUR[s])
    if r is not None:
        top.plot(grid, gaussian_kde(r.ISOPleasant)(grid), "--", color="0.3")
        right.plot(gaussian_kde(r.ISOEventful)(grid), grid, "--", color="0.3")
        ax.plot([], [], "--", color="0.3", label=f"{a.ref_name}, all scenes (marginals, n={len(r)})")
    ax.axhline(0, color="0.85", lw=0.8)
    ax.axvline(0, color="0.85", lw=0.8)
    ax.set_xlabel("ISOPleasant")
    ax.set_ylabel("ISOEventful")
    ax.legend(fontsize=7, loc="lower left")
    top.axis("off")
    right.axis("off")
    fig.savefig(f"{a.out}_joint.png", dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}_radar.png, _scatter.png, _joint.png, _summary.csv")


if __name__ == "__main__":
    main()
