"""Circumplex figure: rated soundscapes in the ISOPleasant / ISOEventful plane, with the eight attributes.

Input, one of:
  --csv FILE        table with columns ISOPleasant, ISOEventful (e.g. isd/selected_isd.csv)
  --araus-root DIR  clone of ntudsp/araus-dataset-baseline-models (uses rsd.data.load_araus, mean per stimulus)
Output: PDF (and PNG) of one panel, 16:9-friendly.

  python generation/plot_circumplex.py --csv isd/selected_isd.csv --name ISD --out plots/circumplex_isd
  python generation/plot_circumplex.py --araus-root ~/araus-dataset-baseline-models --name ARAUS --out plots/circumplex_araus
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ap = argparse.ArgumentParser()
ap.add_argument("--csv"); ap.add_argument("--araus-root")
ap.add_argument("--name", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--size", type=float, nargs=2, default=[4.6, 5.4])
ap.add_argument("--mode", choices=["auto", "scatter", "density"], default="auto",
                help="auto: density above 3000 points, otherwise one dot per point")
ap.add_argument("--no-marginal", action="store_true", help="omit the ISOPleasant histogram above the plot")
a = ap.parse_args()

def araus_stimuli(root):
    """Same selection as rsd.data.load_araus (folds 0-5, no attention stimuli), mean per stimulus.
    The ISO 12913-3 formula is written out here so that soundscapy (which sets a numpy print option
    that breaks scipy in some environments) is not imported."""
    d = Path(root) / "data"
    r = pd.read_csv(d / "responses.csv")
    r = r[r.fold_r.between(0, 5) & (r.is_attention == 0)].copy()
    P = ["pleasant", "vibrant", "eventful", "chaotic", "annoying", "monotonous", "uneventful", "calm"]
    r = r.dropna(subset=P)
    c, k = np.cos(np.pi / 4), 4 + np.sqrt(32)
    r["ISOPleasant"] = ((r.pleasant - r.annoying) + c * (r.calm - r.chaotic) + c * (r.vibrant - r.monotonous)) / k
    r["ISOEventful"] = ((r.eventful - r.uneventful) + c * (r.chaotic - r.calm) + c * (r.vibrant - r.monotonous)) / k
    r["stimulus_id"] = r.soundscape + "|" + r.masker + "|" + r.smr.astype(int).astype(str)
    return r.groupby("stimulus_id").agg(ISOPleasant=("ISOPleasant", "mean"), ISOEventful=("ISOEventful", "mean"),
                                        n_ratings=("ISOPleasant", "size")).reset_index()

if a.araus_root:
    df = araus_stimuli(a.araus_root)
else:
    df = pd.read_csv(a.csv)
df = df.dropna(subset=["ISOPleasant", "ISOEventful"])
x, y = df.ISOPleasant.values, df.ISOEventful.values

BLUE = "#003D7C"
fig, ax = plt.subplots(figsize=a.size)
if a.mode == "density" or (a.mode == "auto" and len(df) > 3000):
    # many points on a discrete grid (ARAUS: mostly one rating per stimulus): show density, not dots
    hb = ax.hexbin(x, y, gridsize=24, extent=(-1, 1, -1, 1), cmap="Blues", mincnt=1, linewidths=0.2,
                   edgecolors="white", zorder=2)
    cb = fig.colorbar(hb, ax=ax, shrink=0.6, pad=0.02); cb.ax.tick_params(labelsize=7)
    cb.set_label("number of stimuli", fontsize=8)
else:
    ax.scatter(x, y, s=9, color=BLUE, alpha=0.35, linewidths=0, zorder=3)
ax.axhline(0, color="0.45", lw=0.8, zorder=1); ax.axvline(0, color="0.45", lw=0.8, zorder=1)
t = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(t), np.sin(t), color="0.45", lw=0.8, zorder=4)
r = 1.08
lab = dict(fontsize=9, color="0.15", ha="center", va="center")
ax.text(r, 0, "pleasant", ha="left", va="center", fontsize=9, color="0.15")
ax.text(-r, 0, "annoying", ha="right", va="center", fontsize=9, color="0.15")
ax.text(0, r, "eventful", **{**lab, "va": "bottom"})
ax.text(0, -r, "uneventful", **{**lab, "va": "top"})
d = 1.2 * np.cos(np.pi / 4)
ax.text(d, d, "vibrant", **lab); ax.text(-d, d, "chaotic", **lab)
ax.text(-d, -d, "monotonous", **lab); ax.text(d, -d, "calm", **lab)
ax.set_xlim(-1.85, 1.85); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal")
ax.set_xlabel("ISOPleasant", fontsize=9); ax.set_ylabel("ISOEventful", fontsize=9)
ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.tick_params(labelsize=8)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ttl = f"{a.name}: {len(df)} " + ("stimuli" if a.araus_root else "recordings") if a.name else ""
if ttl and a.no_marginal: ax.set_title(ttl, fontsize=9)
if not a.no_marginal:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    top = make_axes_locatable(ax).append_axes("top", size="22%", pad=0.08, sharex=ax)
    top.hist(x, bins=np.linspace(-1, 1, 21), color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.4)
    top.axvline(0, color="0.45", lw=0.8)
    top.axvline(x.mean(), color="#C8102E", lw=1.2)
    top.set_yticks([]); top.tick_params(labelbottom=False, bottom=False)
    for sp in ("top", "right", "left"): top.spines[sp].set_visible(False)
    if ttl: top.set_title(ttl, fontsize=9)
fig.tight_layout()
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=200)
print(out.with_suffix(".pdf"), "n =", len(df), "mean P/E = %.2f/%.2f" % (x.mean(), y.mean()),
      "range P %.2f..%.2f E %.2f..%.2f" % (x.min(), x.max(), y.min(), y.max()))
