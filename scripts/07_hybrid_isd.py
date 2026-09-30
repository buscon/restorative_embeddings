"""Step 7 - does zero-shot source recognition improve field predictions?

Step 6 showed that CLAP's zero-shot source scores (natural sounds, other
noise) relate to ISOPleasant on ISD beyond the psychoacoustic prediction, but
only descriptively. This script tests it out of sample.

Every model is a small linear combination whose weights are fitted on ISD
with leave-one-location-out cross-validation (LOLO): each of the 18 locations
is predicted by weights fitted on the other 17. The inputs themselves are not
fitted to ISD:
    psy      prediction of the psychoacoustic ridge trained on ARAUS (step 3)
    clapsy   prediction of the CLAP+psycho ridge trained on ARAUS (step 3)
    sources  CLAP zero-shot relative scores (step 6); human sounds is the
             reference category, because the four relative scores sum to 0

Models (each with an intercept):
    M0  psy                                   baseline, recalibrated to ISD
    M0b clapsy                                ARAUS-trained CLAP model, recalibrated
    M1  psy + natural + other_noise           the two sources that mattered in step 6
    M2  psy + natural + other_noise + traffic
    M3  natural + other_noise + traffic       sources only, no acoustics model

The baselines get the same recalibration (intercept and slope) as the hybrids,
so any gain comes from the source scores, not from fixing the lab-to-field
offset. Out-of-fold predictions are scored at recording and location level;
differences to M0 use the paired location bootstrap (on the out-of-fold
predictions, without refitting). Correlations with overall quality (sss01)
and wish to revisit (sss05) are reported for ISOPleasant.

    python scripts/07_hybrid_isd.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="results/hybrid")
ap.add_argument("--n-boot", type=int, default=2000)
a = ap.parse_args()
res, out = Path(a.results), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

pred = pd.read_csv(res / "isd_predictions.csv").set_index("GroupID")
src = pd.read_csv(res / "sources" / "isd_source_scores.csv", index_col=0)
d = pred.join(src[["traffic_rel", "other_noise_rel", "natural_rel"]], how="inner")

MODELS = {
    "M0 psy": ["psy"],
    "M0b clapsy": ["clapsy"],
    "M1 psy+natural+other_noise": ["psy", "natural_rel", "other_noise_rel"],
    "M2 psy+natural+other_noise+traffic": ["psy", "natural_rel", "other_noise_rel", "traffic_rel"],
    "M3 sources only": ["natural_rel", "other_noise_rel", "traffic_rel"],
}
BASE = "M0 psy"


def lolo(df, X, t):
    oof = pd.Series(np.nan, index=df.index)
    for loc in df.LocationID.unique():
        test = df.LocationID == loc
        m = LinearRegression().fit(df.loc[~test, X], df.loc[~test, t])
        oof[test] = m.predict(df.loc[test, X])
    full = LinearRegression().fit(df[X], df[t])  # for reporting the weights only
    beta = full.coef_ * df[X].std().values / df[t].std()  # standardised coefficients
    return oof, dict(zip(X, beta))


def draws(groups):
    u, inv = np.unique(groups, return_inverse=True)
    mem = [np.flatnonzero(inv == k) for k in range(len(u))]
    for _ in range(a.n_boot):
        yield np.concatenate([mem[k] for k in rng.integers(0, len(u), len(u))])


def pearson(y, p):
    return np.corrcoef(y, p)[0, 1]


rows, comp, coefs = [], [], []
for t in ["ISOPleasant", "ISOEventful"]:
    df = d.assign(psy=d[f"pred_{t}_psycho_ridge"], clapsy=d[f"pred_{t}_clap+psycho_ridge"]).dropna(
        subset=[t, "psy", "clapsy", "natural_rel"])
    g = df.LocationID.values
    y = df[t].values
    oofs = {}
    for name, X in MODELS.items():
        oof, w = lolo(df, X, t)
        oofs[name] = oof.values
        loc = df.assign(p=oof).groupby("LocationID")[[t, "p"]].mean()
        bs = [pearson(y[s], oof.values[s]) for s in draws(g)]
        row = {"target": t, "model": name, "r": pearson(y, oof), "r_lo": np.nanpercentile(bs, 2.5),
               "r_hi": np.nanpercentile(bs, 97.5), "R2": r2_score(y, oof), "loc_r": pearson(loc[t], loc.p),
               "n": len(df), "n_locations": df.LocationID.nunique()}
        if t == "ISOPleasant":
            for proxy in ["sss01", "sss05"]:
                ok = df[proxy].notna().values
                row[f"r_{proxy}"] = pearson(df[proxy].values[ok], oof.values[ok])
        rows.append(row)
        coefs.append({"target": t, "model": name, **{k: round(v, 4) for k, v in w.items()}})
    for name in MODELS:
        if name == BASE:
            continue
        A, B = oofs[name], oofs[BASE]
        dr = np.array([pearson(y[s], A[s]) - pearson(y[s], B[s]) for s in draws(g)])
        dR = np.array([r2_score(y[s], A[s]) - r2_score(y[s], B[s]) for s in draws(g)])
        comp.append({"target": t, "model": name, "vs": BASE,
                     "dr": pearson(y, A) - pearson(y, B), "dr_lo": np.percentile(dr, 2.5),
                     "dr_hi": np.percentile(dr, 97.5), "dr_p": min(1, 2 * min((dr <= 0).mean(), (dr >= 0).mean())),
                     "dR2": r2_score(y, A) - r2_score(y, B), "dR2_lo": np.percentile(dR, 2.5),
                     "dR2_hi": np.percentile(dR, 97.5),
                     "dR2_p": min(1, 2 * min((dR <= 0).mean(), (dR >= 0).mean()))})

R, C, W = pd.DataFrame(rows), pd.DataFrame(comp), pd.DataFrame(coefs)
R.to_csv(out / "hybrid_scores.csv", index=False)
C.to_csv(out / "hybrid_vs_psy.csv", index=False)
W.to_csv(out / "hybrid_weights_full_fit.csv", index=False)

pd.set_option("display.width", 200)
print(f"Leave-one-location-out on ISD ({R.n.iloc[0]} recordings, {R.n_locations.iloc[0]} locations)")
print("\nScores of out-of-fold predictions (recording level; loc_r = over location means)")
print(R.drop(columns=["n", "n_locations"]).round(3).to_string(index=False))
print(f"\nDifference to '{BASE}' (paired location bootstrap on out-of-fold predictions)")
print(C.drop(columns="vs").round(3).to_string(index=False))
print("\nStandardised weights (beta) when fitted on all 18 locations - for interpretation only")
print(W.to_string(index=False))
print(f"\nwrote {out}/")
