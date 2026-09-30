"""Step 7 - does zero-shot source recognition improve field predictions?

Step 6 showed that CLAP's zero-shot source scores (natural sounds, other
noise) relate to ISOPleasant on ISD beyond the psychoacoustic prediction, but
only descriptively. This script tests it out of sample.

Every model is a small linear combination whose weights are fitted on ISD with
grouped cross-validation. The inputs themselves are not fitted to ISD:
    psy      prediction of the psychoacoustic ridge trained on ARAUS (step 3)
    clapsy   prediction of the CLAP+psycho ridge trained on ARAUS (step 3)
    sources  CLAP zero-shot relative scores (step 6); human sounds is the
             reference category, because the four relative scores sum to 0

Models (each with an intercept):
    M0  psy                                   baseline, recalibrated to ISD
    M0b clapsy                                ARAUS-trained CLAP model, recalibrated
    M1  psy + natural + other_noise           the two sources that mattered in step 6
    M2  psy + natural + other_noise + traffic primary: all sources, no selection
    M3  natural + other_noise + traffic       sources only, no acoustics model

The baselines get the same recalibration (intercept and slope) as the hybrids,
so any gain comes from the source scores, not from fixing the lab-to-field
offset. Differences to M0 use the paired location bootstrap on the
out-of-fold predictions (without refitting).

PRIMARY ANALYSIS: questionnaire phrases, leave-one-location-out (18 folds),
printed in full.

ROBUSTNESS (compact table at the end, results/hybrid/robustness.csv):
    * leave-one-CITY-out (4 folds: London 11 locations, Granada 4, Venice 2,
      Groningen 1). Holding out London means fitting on 7 locations only.
    * the two further phrase sets from step 6 ("alternative", "labels").

    python scripts/07_hybrid_isd.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

CITY = {
    "CampoPrincipe": "Granada", "CarloV": "Granada", "MiradorSanNicolas": "Granada", "PlazaBibRambla": "Granada",
    "SanMarco": "Venice", "MonumentoGaribaldi": "Venice", "Noorderplantsoen": "Groningen",
    "CamdenTown": "London", "EustonTap": "London", "MarchmontGarden": "London", "PancrasLock": "London",
    "RegentsParkFields": "London", "RegentsParkJapan": "London", "RussellSq": "London", "StPaulsCross": "London",
    "StPaulsRow": "London", "TateModern": "London", "TorringtonSq": "London",
}
MODELS = {
    "M0 psy": ["psy"],
    "M0b clapsy": ["clapsy"],
    "M1 psy+natural+other_noise": ["psy", "natural_rel", "other_noise_rel"],
    "M2 psy+natural+other_noise+traffic": ["psy", "natural_rel", "other_noise_rel", "traffic_rel"],
    "M3 sources only": ["natural_rel", "other_noise_rel", "traffic_rel"],
}
BASE, MAIN = "M0 psy", "M2 psy+natural+other_noise+traffic"
TARGETS = ["ISOPleasant", "ISOEventful"]

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="results/hybrid")
ap.add_argument("--n-boot", type=int, default=2000)
a = ap.parse_args()
res, out = Path(a.results), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
pred = pd.read_csv(res / "isd_predictions.csv").set_index("GroupID")
unknown = sorted(set(pred.LocationID) - set(CITY))
if unknown:
    print(f"note: no city known for {unknown}; each is treated as its own city")
pred["City"] = pred.LocationID.map(CITY).fillna(pred.LocationID)


def draws(groups, seed=0):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(groups, return_inverse=True)
    mem = [np.flatnonzero(inv == k) for k in range(len(u))]
    for _ in range(a.n_boot):
        yield np.concatenate([mem[k] for k in rng.integers(0, len(u), len(u))])


def pearson(y, p):
    return np.corrcoef(y, p)[0, 1]


def boot_ci(stat, y, p, groups):
    bs = np.array([stat(y[s], p[s]) for s in draws(groups)])
    return np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


def paired(stat, y, pa, pb, groups):
    d = np.array([stat(y[s], pa[s]) - stat(y[s], pb[s]) for s in draws(groups)])
    return (stat(y, pa) - stat(y, pb), np.percentile(d, 2.5), np.percentile(d, 97.5),
            min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def cross_fit(df, X, t, fold_col):
    oof = pd.Series(np.nan, index=df.index)
    for f in df[fold_col].unique():
        test = df[fold_col] == f
        oof[test] = LinearRegression().fit(df.loc[~test, X], df.loc[~test, t]).predict(df.loc[test, X])
    full = LinearRegression().fit(df[X], df[t])
    beta = full.coef_ * df[X].std().values / df[t].std()  # standardised, full-data fit
    return oof.values, dict(zip(X, beta))


def run(phrase_set, fold_col, detail):
    src = pd.read_csv(res / "sources" / f"isd_source_scores_{phrase_set}.csv", index_col=0)
    d = pred.join(src[["traffic_rel", "other_noise_rel", "natural_rel"]], how="inner")
    rows, comp, coefs = [], [], []
    for t in TARGETS:
        df = d.assign(psy=d[f"pred_{t}_psycho_ridge"], clapsy=d[f"pred_{t}_clap+psycho_ridge"]).dropna(
            subset=[t, "psy", "clapsy", "natural_rel"])
        y, g = df[t].values, df.LocationID.values
        oofs = {}
        for name, X in MODELS.items():
            oof, w = cross_fit(df, X, t, fold_col)
            oofs[name] = oof
            if detail:
                lo, hi = boot_ci(pearson, y, oof, g)
                loc = df.assign(p=oof).groupby("LocationID")[[t, "p"]].mean()
                row = {"target": t, "model": name, "r": pearson(y, oof), "r_lo": lo, "r_hi": hi,
                       "R2": r2_score(y, oof), "loc_r": pearson(loc[t], loc.p), "n": len(df),
                       "n_locations": df.LocationID.nunique()}
                if t == "ISOPleasant":
                    for proxy in ["sss01", "sss05"]:
                        ok = df[proxy].notna().values
                        row[f"r_{proxy}"] = pearson(df[proxy].values[ok], oof[ok])
                rows.append(row)
                coefs.append({"target": t, "model": name, **{k: round(v, 4) for k, v in w.items()}})
        for name in MODELS:
            if name == BASE:
                continue
            dr = paired(pearson, y, oofs[name], oofs[BASE], g)
            dR = paired(r2_score, y, oofs[name], oofs[BASE], g)
            comp.append({"phrase_set": phrase_set, "validation": fold_col, "target": t, "model": name,
                         "r_model": pearson(y, oofs[name]), "r_base": pearson(y, oofs[BASE]),
                         "dr": dr[0], "dr_lo": dr[1], "dr_hi": dr[2], "dr_p": dr[3],
                         "dR2": dR[0], "dR2_lo": dR[1], "dR2_hi": dR[2], "dR2_p": dR[3]})
    return pd.DataFrame(rows), pd.DataFrame(comp), pd.DataFrame(coefs)


pd.set_option("display.width", 220)

# ---- primary --------------------------------------------------------------------
R, C, W = run("questionnaire", "LocationID", detail=True)
R.to_csv(out / "hybrid_scores.csv", index=False)
C.to_csv(out / "hybrid_vs_psy.csv", index=False)
W.to_csv(out / "hybrid_weights_full_fit.csv", index=False)
print(f"PRIMARY: questionnaire phrases, leave-one-location-out ({R.n.iloc[0]} recordings, "
      f"{R.n_locations.iloc[0]} locations)")
print("\nScores of out-of-fold predictions (recording level; loc_r = over location means)")
print(R.drop(columns=["n", "n_locations"]).round(3).to_string(index=False))
print(f"\nDifference to '{BASE}' (paired location bootstrap on out-of-fold predictions)")
print(C.drop(columns=["phrase_set", "validation", "r_model", "r_base"]).round(3).to_string(index=False))
print("\nStandardised weights (beta) when fitted on all 18 locations - for interpretation only")
print(W.to_string(index=False))

# ---- robustness -------------------------------------------------------------------
allc = []
for ps in ["questionnaire", "alternative", "labels"]:
    if not (res / "sources" / f"isd_source_scores_{ps}.csv").exists():
        print(f"\n(skip phrase set '{ps}': run 06_source_recognition.py first)")
        continue
    for fold_col in ["LocationID", "City"]:
        allc.append(C if (ps, fold_col) == ("questionnaire", "LocationID") else run(ps, fold_col, detail=False)[1])
rob = pd.concat(allc, ignore_index=True)
rob["validation"] = rob.validation.map({"LocationID": "leave-one-location-out", "City": "leave-one-city-out"})
rob.to_csv(out / "robustness.csv", index=False)
show = rob[rob.model.isin([MAIN, "M3 sources only", "M0b clapsy"])]
print(f"\nROBUSTNESS: each model vs '{BASE}' under the same validation and phrase set")
for t in TARGETS:
    print(f"\n{t}")
    print(show[show.target == t][["phrase_set", "validation", "model", "r_base", "r_model", "dr", "dr_lo", "dr_hi",
                                  "dr_p", "dR2", "dR2_lo", "dR2_hi"]].round(3).to_string(index=False))
print(f"\nwrote {out}/")
