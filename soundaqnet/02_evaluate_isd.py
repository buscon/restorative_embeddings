#!/usr/bin/env python3
"""Compare SoundAQnet's predictions with the human ratings of the ISD recordings.

Input: the table of ISD recordings with the mean human ISOPleasant/ISOEventful per recording
(data/processed/isd_recordings.csv) and the predictions from 01_rate_isd.py.

Reported per condition:
  * recording-level Pearson and Spearman r with the human mean rating, with a 95% interval from a
    cluster bootstrap over locations (recordings from one location are not independent);
  * location-level r (mean over the recordings of each location);
  * bias (mean prediction minus mean rating) and the spread ratio sd(pred) / sd(human): a ratio far
    below 1 means the predictions are compressed towards the middle.
Sensitivity (non-full conditions, on the recordings that were run under both): correlation between
the full-length prediction and the perturbed one, mean shift, and the change in r with the humans.

Decision rule of plan B (set the thresholds BEFORE looking at results; defaults below):
  1. recording-level r on `full` >= --ref-r (0.37, the milestone-1 ARAUS CLAP ridge model);
  2. stable under cut10, gain-10 and gain+10: corr(full, cond) >= --min-stab (0.90), mean shift
     within --max-shift (0.10 ISOPleasant units) and r with the humans not lower than on `full`
     by more than --max-drop (0.05) on the same recordings.
If 1 holds but 2 fails, SoundAQnet needs an assumed fixed level for uncalibrated clips (state it),
or another rater. The milestone-1 CLAP ridge model has to be tested under the same conditions
(not done here) before deciding between the two.

    python soundaqnet/02_evaluate_isd.py --recordings data/processed/isd_recordings.csv \
        --pred soundaqnet/isd_soundaqnet.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def cluster_boot_r(df, x, y, group, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = {g: d for g, d in df.groupby(group)}
    keys = list(groups)
    rs = []
    for _ in range(n):
        s = pd.concat([groups[k] for k in rng.choice(keys, len(keys))])
        if s[x].std() > 0 and s[y].std() > 0:
            rs.append(np.corrcoef(s[x], s[y])[0, 1])
    return np.percentile(rs, [2.5, 97.5])


def summary(d, target, loc="LocationID"):
    obs, pred = d[target + "_human"], d[target + "_pred"]
    r, _ = pearsonr(obs, pred)
    rho, _ = spearmanr(obs, pred)
    lo, hi = cluster_boot_r(d, target + "_human", target + "_pred", loc)
    byloc = d.groupby(loc)[[target + "_human", target + "_pred"]].mean()
    r_loc = np.corrcoef(byloc.iloc[:, 0], byloc.iloc[:, 1])[0, 1] if len(byloc) > 2 else np.nan
    return {"n": len(d), "r": r, "ci_lo": lo, "ci_hi": hi, "spearman": rho, "r_location": r_loc, "n_loc": len(byloc),
            "bias": pred.mean() - obs.mean(), "spread_ratio": pred.std() / obs.std()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", default="data/processed/isd_recordings.csv")
    ap.add_argument("--pred", default="soundaqnet/isd_soundaqnet.csv")
    ap.add_argument("--out", default="soundaqnet/isd_evaluation.csv")
    ap.add_argument("--ref-r", type=float, default=0.37)
    ap.add_argument("--min-stab", type=float, default=0.90)
    ap.add_argument("--max-shift", type=float, default=0.10)
    ap.add_argument("--max-drop", type=float, default=0.05)
    a = ap.parse_args()

    rec = pd.read_csv(a.recordings)
    rec = rec[rec.n_ratings > 0][["GroupID", "LocationID", "ISOPleasant", "ISOEventful"]]
    pred = pd.read_csv(a.pred)
    both = pred.merge(rec, on="GroupID", suffixes=("_pred", "_human"))
    conds = [c for c in ["full", "gain-20", "gain-10", "gain+10", "cut10", "cut5"] if c in set(both.condition)]

    rows = []
    print("condition   target        n     r  [95% CI]        spearman  r_loc(n)     bias  spread")
    for c in conds:
        d = both[both.condition == c]
        for t in ("ISOPleasant", "ISOEventful"):
            s = summary(d, t)
            rows.append({"condition": c, "target": t, **s})
            print(f"{c:10s}  {t:11s} {s['n']:4d} {s['r']:+.3f} [{s['ci_lo']:+.2f},{s['ci_hi']:+.2f}]  "
                  f"{s['spearman']:+.3f}   {s['r_location']:+.3f}({s['n_loc']})  {s['bias']:+.3f}  {s['spread_ratio']:.2f}")
    pd.DataFrame(rows).to_csv(a.out, index=False)

    print("\nSensitivity vs full length / original level, ISOPleasant (same recordings):")
    full = both[both.condition == "full"].set_index("GroupID")
    stable = {}
    for c in [c for c in conds if c != "full"]:
        d = both[both.condition == c].set_index("GroupID")
        idx = d.index.intersection(full.index)
        f, p = full.loc[idx], d.loc[idx]
        corr = np.corrcoef(f.ISOPleasant_pred, p.ISOPleasant_pred)[0, 1]
        shift = (p.ISOPleasant_pred - f.ISOPleasant_pred).mean()
        r_full = np.corrcoef(f.ISOPleasant_human, f.ISOPleasant_pred)[0, 1]
        r_c = np.corrcoef(p.ISOPleasant_human, p.ISOPleasant_pred)[0, 1]
        ok = corr >= a.min_stab and abs(shift) <= a.max_shift and (r_c - r_full) >= -a.max_drop
        stable[c] = ok
        print(f"{c:8s} n={len(idx):4d}  corr(full,cond) {corr:.3f}  mean shift {shift:+.3f}  "
              f"r_human full {r_full:+.3f} -> cond {r_c:+.3f}  {'stable' if ok else 'NOT stable'}")

    print("\nDecision rule (plan B, step 0):")
    r_full = [r for r in rows if r["condition"] == "full" and r["target"] == "ISOPleasant"][0]
    print(f"  1. recording-level r on full = {r_full['r']:+.3f} (CI {r_full['ci_lo']:+.2f} to {r_full['ci_hi']:+.2f}) "
          f"vs reference {a.ref_r:.2f}: {'passes' if r_full['r'] >= a.ref_r else 'fails'}")
    need = [c for c in ("cut10", "gain-10", "gain+10") if c in stable]
    if need:
        print("  2. stability: " + ", ".join(f"{c} {'ok' if stable[c] else 'not stable'}" for c in need))
    else:
        print("  2. stability: the perturbation conditions were not run")
    print("  The same conditions must be run for the CLAP ridge model before choosing between the raters.")


if __name__ == "__main__":
    main()
