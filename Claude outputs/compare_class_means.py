#!/usr/bin/env python3
"""Mean predicted affect per source class: generated prompts against real ARAUS stimuli with the matching masker (no correlations).

ARAUS stimuli are a park soundscape plus one masker of a known type (bird, water, traffic, wind, construction), rated by listeners
and by SoundAQnet. For each masker type this script takes the generated prompts that name the same kind of source (table MATCH below,
edit it) and compares, per type: ISOPleasant, ISOEventful and the eight attributes of
  - ARAUS, SoundAQnet (the reference csv from soundaqnet/05_rate_araus_reference.py, 65 dB or true level)
  - ARAUS, listeners (human_ISOPleasant, human_ISOEventful; mean of the ratings of each stimulus)
  - generated, SoundAQnet
with 95% bootstrap intervals (over stimuli for ARAUS, over prompts for the generated clips; seeds are averaged within a prompt).
It is a different question from the heatmap: not "which events go with which affect" but "does a bird prompt land where a bird-masked
park soundscape lands". Differences are expected: the generated clips are not park soundscapes plus a masker, and ARAUS maskers vary in
level relative to the soundscape (SMR), which the prompts do not.

    python generation/compare_class_means.py --grid ~/grids_3msde/base_ae_g12 --ref ~/grids_3msde/araus_ref/soundaqnet_full_65.csv \
        --out ~/plots_base_ae/class_means_ref65
Output: <out>.csv (one row per masker type and source) and <out>.png (dots with intervals; ISOPleasant, ISOEventful).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ARAUS masker type -> prompt ids of the generated set (generation/make_ae_prompts.py)
MATCH = {
    "bird": ["ae_bird_1", "ae_bird_2", "ae_bird_3", "ae_bird_4"],
    "water": ["ae_water_1", "ae_water_2", "ae_water_3", "ae_water_4"],
    "traffic": ["ae_traffic_1", "ae_traffic_2", "ae_traffic_3", "ae_traffic_4"],
    "wind": ["ae_wind_1", "ae_wind_2", "ae_wind_3", "ae_wind_4"],
    "construction": ["ae_noise_2"],   # "a construction site with drilling and hammering"
}
PAQ = ['pleasant', 'eventful', 'chaotic', 'vibrant', 'uneventful', 'calm', 'annoying', 'monotonous']
VARS = ["ISOPleasant", "ISOEventful"] + [f"paq_{p}" for p in PAQ]


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = pd.read_csv(Path(a.grid).expanduser() / "grid.csv")[["file", "prompt_id"]].merge(
        pd.read_csv(Path(a.grid).expanduser() / "soundaqnet_full.csv"), on="file")
    r = pd.read_csv(Path(a.ref).expanduser())
    gp = g.groupby("prompt_id")[VARS].mean()           # seeds averaged within a prompt
    rows = []
    for mt, pids in MATCH.items():
        gg = gp.loc[[p for p in pids if p in gp.index]]
        rr = r[r.masker_type == mt]
        for v in VARS:
            for src, x in (("ARAUS SoundAQnet", rr[v]), ("generated SoundAQnet", gg[v])):
                lo, hi = boot_ci(x)
                rows.append({"masker_type": mt, "source": src, "variable": v, "n": len(x), "mean": float(np.mean(x)) if len(x) else np.nan,
                             "ci_low": lo, "ci_high": hi})
            if v in ("ISOPleasant", "ISOEventful") and f"human_{v}" in rr:
                lo, hi = boot_ci(rr[f"human_{v}"])
                rows.append({"masker_type": mt, "source": "ARAUS listeners", "variable": v, "n": len(rr),
                             "mean": float(rr[f"human_{v}"].mean()), "ci_low": lo, "ci_high": hi})
    d = pd.DataFrame(rows)
    d.to_csv(f"{a.out}.csv", index=False)
    print(d[d.variable.isin(["ISOPleasant", "ISOEventful"])].round(3).to_string(index=False))

    colour = {"ARAUS listeners": "#444444", "ARAUS SoundAQnet": "#2b83ba", "generated SoundAQnet": "#d95f02"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    types = list(MATCH)
    for ax, v in zip(axs, ("ISOPleasant", "ISOEventful")):
        for k, src in enumerate(colour):
            sub = d[(d.variable == v) & (d.source == src)].set_index("masker_type").reindex(types)
            y = np.arange(len(types)) + (k - 1) * 0.22
            ax.errorbar(sub["mean"], y, xerr=[sub["mean"] - sub.ci_low, sub.ci_high - sub["mean"]], fmt="o", ms=5, color=colour[src],
                        capsize=2, label=src)
        ax.axvline(0, color="0.85", lw=0.8)
        ax.set_yticks(range(len(types)), [f"{t} (n={int((r.masker_type == t).sum())} / {len(MATCH[t])} prompts)" for t in types], fontsize=8)
        ax.set_xlabel(v)
        ax.invert_yaxis()
    axs[0].legend(fontsize=8, loc="lower left")
    fig.savefig(f"{a.out}.png", dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}.csv and .png")


if __name__ == "__main__":
    main()
