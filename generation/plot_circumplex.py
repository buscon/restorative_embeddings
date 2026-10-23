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
ap.add_argument("--size", type=float, nargs=2, default=[4.6, 4.6])
a = ap.parse_args()

if a.araus_root:
    from rsd.data import load_araus
    df = load_araus(a.araus_root)["stimuli"]
else:
    df = pd.read_csv(a.csv)
df = df.dropna(subset=["ISOPleasant", "ISOEventful"])
x, y = df.ISOPleasant.values, df.ISOEventful.values

BLUE = "#003D7C"
fig, ax = plt.subplots(figsize=a.size)
ax.scatter(x, y, s=9, color=BLUE, alpha=0.35, linewidths=0, zorder=3)
ax.axhline(0, color="0.45", lw=0.8, zorder=1); ax.axvline(0, color="0.45", lw=0.8, zorder=1)
t = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(t), np.sin(t), color="0.75", lw=0.8, zorder=1)
r = 1.08
lab = dict(fontsize=9, color="0.15", ha="center", va="center")
ax.text(r, 0, "pleasant", ha="left", va="center", fontsize=9, fontweight="bold", color="0.15")
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
if ttl: ax.set_title(ttl, fontsize=9)
fig.tight_layout()
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=200)
print(out.with_suffix(".pdf"), "n =", len(df), "mean P/E = %.2f/%.2f" % (x.mean(), y.mean()),
      "range P %.2f..%.2f E %.2f..%.2f" % (x.min(), x.max(), y.min(), y.max()))
