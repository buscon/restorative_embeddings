"""Step 3 - train on ARAUS, test on ARAUS fold 0 and on ISD.

Feature sets (all standardised):
    psycho         7 harmonised psychoacoustic predictors (rsd.data.PSYCHO)
    handcrafted    psycho + ecoacoustic indices
    clap           CLAP embedding (level-normalised audio)
    clap+level     CLAP + calibrated LA50 (puts absolute level back)
    clap+psycho    CLAP + the 7 psychoacoustic predictors

Models: ridge for every set; gradient boosting for the low-dimensional sets.
Hyper-parameters are chosen with the official ARAUS folds 1-5 (disjoint in
soundscapes, maskers and participants, so no leakage). The chosen model is
refitted on folds 1-5 and then applied, unchanged, to

    * ARAUS fold 0 (independent in-domain test set), and
    * ISD (external field test: other cities, in-situ ratings).

ISD is scored at three levels, because one rating per person makes the
individual level very noisy (location ICC ~0.30, see docs/session_handoff.md):
    individual ratings, recording means, location means (18 locations).
R^2 on ISD is sensitive to a lab-vs-field offset in the mean (ISD is rated
clearly more pleasant than ARAUS), so three further numbers are reported:
Pearson r, the mean bias, and R^2 after removing the mean offset
("centred": observed and predicted values each minus their own ISD mean).
Centred R^2 answers "is the order right?", plain R^2 "is the level right?".
Centred R^2 uses one statistic from the test data (the means), so it is a
descriptive diagnostic of the transfer, not a clean out-of-sample score.
95 % CIs for recording-level r come from a bootstrap over locations.

ARAUS test fold 0 has only 48 stimuli, each rated by the same 5 people. It is
scored per response and per stimulus (mean of the 5 ratings).

    python scripts/03_train_evaluate.py
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsd.data import PSYCHO, TARGETS  # noqa: E402
from rsd.indices import INDEX_COLUMNS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--processed", default="data/processed")
ap.add_argument("--features", default="data/features")
ap.add_argument("--out", default="results")
ap.add_argument("--no-hgb", action="store_true", help="skip gradient boosting (faster)")
ap.add_argument("--n-boot", type=int, default=2000)
a = ap.parse_args()
proc, feat, out = Path(a.processed), Path(a.features), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)


def load_emb(name):
    z = np.load(feat / f"{name}_clap.npz", allow_pickle=False)
    cols = [f"clap_{i}" for i in range(z["emb"].shape[1])]
    return pd.DataFrame(z["emb"], columns=cols).assign(id=z["ids"]), cols


def load_ind(name):
    p = feat / f"{name}_indices.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["id"] + INDEX_COLUMNS)


# ---- assemble ---------------------------------------------------------------
ar_emb, CLAP = load_emb("araus")
is_emb, _ = load_emb("isd")
araus = (pd.read_csv(proc / "araus_responses.csv")
         .merge(ar_emb, left_on="stimulus_id", right_on="id")
         .merge(load_ind("araus"), on="id", how="left"))
recs = (pd.read_csv(proc / "isd_recordings.csv")
        .merge(is_emb, left_on="GroupID", right_on="id")
        .merge(load_ind("isd"), on="id", how="left"))
ratings = pd.read_csv(proc / "isd_ratings.csv")
ratings = ratings[ratings.has_audio][["GroupID", "LocationID"] + TARGETS]
rated = recs[recs.n_ratings > 0].copy()

FEATURE_SETS = {
    "psycho": PSYCHO,
    "handcrafted": PSYCHO + INDEX_COLUMNS,
    "clap": CLAP,
    "clap+level": CLAP + ["LA50"],
    "clap+psycho": CLAP + PSYCHO,
}
RIDGE_GRID = {"ridge__alpha": np.logspace(-1, 5, 13)}
HGB_GRID = {"histgradientboostingregressor__max_leaf_nodes": [15, 31],
            "histgradientboostingregressor__min_samples_leaf": [50, 200],
            "histgradientboostingregressor__l2_regularization": [0.0, 1.0]}


def models(n_features):
    yield "ridge", make_pipeline(StandardScaler(), Ridge()), RIDGE_GRID
    if not a.no_hgb and n_features <= 50:
        yield "hgb", make_pipeline(StandardScaler(), HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=500, early_stopping=True, random_state=0)), HGB_GRID


def centred_r2(obs, pred):
    """R^2 after subtracting each series' own mean: removes a constant offset only."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    return r2_score(obs - obs.mean(), pred - pred.mean())


def r_and_ci(pred, obs, groups):
    r = pearsonr(pred, obs)[0]
    g = np.asarray(groups)
    uniq = np.unique(g)
    idx = {u: np.flatnonzero(g == u) for u in uniq}
    boots = []
    for _ in range(a.n_boot):
        take = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))])
        if np.std(pred[take]) > 0 and np.std(obs[take]) > 0:
            boots.append(pearsonr(pred[take], obs[take])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return r, lo, hi


rows, preds_out = [], rated[["GroupID", "LocationID", "n_ratings"] + TARGETS + ["sss01", "sss05"]].copy()
for fs_name, cols in FEATURE_SETS.items():
    tr = araus[araus.fold_r.between(1, 5)].dropna(subset=cols + TARGETS)
    te = araus[araus.fold_r == 0].dropna(subset=cols + TARGETS)
    isd = rated.dropna(subset=cols)
    if len(tr) == 0 or len(isd) == 0:
        print(f"skip {fs_name}: missing features")
        continue
    split = PredefinedSplit(tr.fold_r.values - 1)
    for m_name, est, grid in models(len(cols)):
        for t in TARGETS:
            gs = GridSearchCV(est, grid, cv=split, scoring="r2", n_jobs=-1).fit(tr[cols].values, tr[t].values)
            oof = cross_val_predict(gs.best_estimator_, tr[cols].values, tr[t].values, cv=split, n_jobs=-1)
            model = gs.best_estimator_  # already refitted on folds 1-5
            joblib.dump(model, out / f"model_{fs_name}_{m_name}_{t}.joblib")

            p_te = model.predict(te[cols].values)
            te_stim = te.assign(pred=p_te).groupby("stimulus_id")[[t, "pred"]].mean()
            p_isd = pd.Series(model.predict(isd[cols].values), index=isd.GroupID)
            ind = ratings.assign(pred=ratings.GroupID.map(p_isd)).dropna(subset=["pred"])
            rec = isd.assign(pred=p_isd.values)
            loc = rec.groupby("LocationID")[[t, "pred"]].mean()
            r_rec, lo, hi = r_and_ci(rec.pred.values, rec[t].values, rec.LocationID.values)
            preds_out.loc[preds_out.GroupID.isin(isd.GroupID), f"pred_{t}_{fs_name}_{m_name}"] = \
                preds_out.GroupID.map(p_isd)

            row = {
                "features": fs_name, "model": m_name, "target": t,
                "best_params": json.dumps({k.split("__")[1]: float(v) for k, v in gs.best_params_.items()}),
                "araus_cv_r2": r2_score(tr[t], oof),
                "araus_test_r2": r2_score(te[t], p_te),
                "araus_test_stim_r2": r2_score(te_stim[t], te_stim.pred),
                "n_test_stimuli": len(te_stim),
                "araus_target_var": tr[t].var(),
                "isd_indiv_r2": r2_score(ind[t], ind.pred), "isd_indiv_r": pearsonr(ind.pred, ind[t])[0],
                "isd_indiv_r2_centred": centred_r2(ind[t], ind.pred),
                "isd_rec_r2": r2_score(rec[t], rec.pred), "isd_rec_r2_centred": centred_r2(rec[t], rec.pred), "isd_rec_r": r_rec, "isd_rec_r_lo": lo, "isd_rec_r_hi": hi,
                "isd_loc_r": pearsonr(loc.pred, loc[t])[0], "isd_loc_rho": spearmanr(loc.pred, loc[t])[0],
                "isd_bias": rec.pred.mean() - rec[t].mean(),
                "n_train": len(tr), "n_isd_recordings": len(rec), "n_isd_ratings": len(ind),
                "n_isd_locations": rec.LocationID.nunique(),
            }
            if t == "ISOPleasant":  # restoration-adjacent proxies (recording level)
                for proxy in ["sss01", "sss05"]:
                    ok = rec[proxy].notna()
                    row[f"isd_r_{proxy}"] = pearsonr(rec.pred[ok], rec[proxy][ok])[0]
            rows.append(row)
            print(f"{fs_name:12s} {m_name:5s} {t:11s} CV R2 {row['araus_cv_r2']:.3f} | test {row['araus_test_stim_r2']:.3f} (stim) | "
                  f"ISD rec r {r_rec:.2f} [{lo:.2f},{hi:.2f}] R2 {row['isd_rec_r2']:.3f} "
                  f"centred {row['isd_rec_r2_centred']:.3f} bias {row['isd_bias']:+.2f} | loc r {row['isd_loc_r']:.2f}")

res = pd.DataFrame(rows)
res.to_csv(out / "metrics.csv", index=False)
preds_out.to_csv(out / "isd_predictions.csv", index=False)
show = ["features", "model", "target", "araus_cv_r2", "araus_test_r2", "araus_test_stim_r2",
        "isd_indiv_r2", "isd_rec_r2", "isd_rec_r2_centred", "isd_rec_r", "isd_rec_r_lo", "isd_rec_r_hi", "isd_loc_r", "isd_bias"]
(out / "metrics.md").write_text(res[show].round(3).to_markdown(index=False))
print(f"\nwrote {out/'metrics.csv'}, {out/'metrics.md'}, {out/'isd_predictions.csv'}")
