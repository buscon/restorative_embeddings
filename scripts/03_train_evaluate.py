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

Differences between feature sets are tested with a paired bootstrap: both
models are scored on the same resampled units, so shared noise cancels.
    ARAUS CV: R^2 of out-of-fold predictions, resampling participants.
    ISD:      recording-level Pearson r, resampling locations.
Every model is compared with the psychoacoustic ridge baseline, and the
extended CLAP sets with plain CLAP (results/paired_differences.csv).

ARAUS test fold 0 has only 48 stimuli, each rated by the same 5 people. It is
scored per response and per stimulus (mean of the 5 ratings).

    python scripts/03_train_evaluate.py
"""

import argparse
import json
import warnings
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
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

# scipy warns about an "ill-conditioned matrix" when a ridge penalty is small
# relative to the (strongly correlated) CLAP dimensions. It is harmless: the
# chosen penalty is set by cross-validation. scikit-learn passes this filter
# on to its worker processes.
warnings.filterwarnings("ignore", category=LinAlgWarning)

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
# alpha from 1 to 1e6. A chosen value at the top of the grid (or at the bottom
# for the CLAP sets) is printed, since the optimum may then lie outside it.
RIDGE_GRID = {"ridge__alpha": np.logspace(0, 6, 13)}
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


def cluster_draws(groups, n):
    """n bootstrap index arrays, each resampling whole groups with replacement."""
    g = np.asarray(groups)
    uniq, inv = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    for _ in range(n):
        yield np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])


def pearson(y, p):
    return np.corrcoef(y, p)[0, 1]


def r_and_ci(pred, obs, groups):
    boots = [pearson(obs[t], pred[t]) for t in cluster_draws(groups, a.n_boot)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return pearson(obs, pred), lo, hi


def paired_delta(y, pa, pb, groups, stat, n):
    """stat(A) - stat(B) with a cluster bootstrap CI and two-sided p."""
    d = [stat(y[t], pa[t]) - stat(y[t], pb[t]) for t in cluster_draws(groups, n)]
    d = np.asarray(d)[~np.isnan(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return stat(y, pa) - stat(y, pb), lo, hi, p


oof_store, isd_store = {}, {}
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
            if m_name == "ridge":
                alpha = gs.best_params_["ridge__alpha"]
                at_top = alpha == grid["ridge__alpha"][-1]
                at_bottom = alpha == grid["ridge__alpha"][0] and len(cols) > 50  # low-dim sets may need none
                if at_top or at_bottom:
                    print(f"  note: {fs_name} {t} chose alpha={alpha:g}, at the edge of the grid")
            oof = cross_val_predict(gs.best_estimator_, tr[cols].values, tr[t].values, cv=split, n_jobs=-1)
            model = gs.best_estimator_  # already refitted on folds 1-5
            oof_store[(fs_name, m_name, t)] = pd.Series(oof, index=tr.index)
            joblib.dump(model, out / f"model_{fs_name}_{m_name}_{t}.joblib")

            p_te = model.predict(te[cols].values)
            te_stim = te.assign(pred=p_te).groupby("stimulus_id")[[t, "pred"]].mean()
            p_isd = pd.Series(model.predict(isd[cols].values), index=isd.GroupID)
            isd_store[(fs_name, m_name, t)] = p_isd
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

# ---- paired comparisons --------------------------------------------------------
BASE, CLAP_BASE = ("psycho", "ridge"), ("clap", "ridge")
pairs = [(k, BASE) for k in {(f, m) for f, m, _ in oof_store} if k != BASE]
pairs += [((f, "ridge"), CLAP_BASE) for f in ["clap+level", "clap+psycho"]]
comp = []
for t in TARGETS:
    for A, B in pairs:
        ka, kb = (*A, t), (*B, t)
        if ka not in oof_store or kb not in oof_store:
            continue
        common = oof_store[ka].index.intersection(oof_store[kb].index)  # same ARAUS rows
        y = araus.loc[common, t].values
        cv = paired_delta(y, oof_store[ka][common].values, oof_store[kb][common].values,
                          araus.loc[common, "participant"].values, r2_score, max(200, a.n_boot // 2))
        g = isd_store[ka].index.intersection(isd_store[kb].index)            # same ISD recordings
        rr = rated.set_index("GroupID").loc[g]
        isd_d = paired_delta(rr[t].values, isd_store[ka][g].values, isd_store[kb][g].values,
                             rr.LocationID.values, pearson, a.n_boot)
        comp.append({"target": t, "model": "_".join(A), "vs": "_".join(B),
                     "araus_cv_dR2": cv[0], "araus_cv_dR2_lo": cv[1], "araus_cv_dR2_hi": cv[2], "araus_cv_p": cv[3],
                     "isd_rec_dr": isd_d[0], "isd_rec_dr_lo": isd_d[1], "isd_rec_dr_hi": isd_d[2], "isd_rec_p": isd_d[3],
                     "n_araus": len(common), "n_isd": len(g)})
comp = pd.DataFrame(comp).sort_values(["target", "vs", "model"])
comp.to_csv(out / "paired_differences.csv", index=False)
print("\nPaired differences (A - B); 95 % cluster-bootstrap CI; p two-sided")
for _, c in comp.iterrows():
    print(f"{c.target:11s} {c.model:17s} vs {c.vs:13s} ARAUS dR2 {c.araus_cv_dR2:+.3f} "
          f"[{c.araus_cv_dR2_lo:+.3f},{c.araus_cv_dR2_hi:+.3f}] p={c.araus_cv_p:.3f} | "
          f"ISD dr {c.isd_rec_dr:+.3f} [{c.isd_rec_dr_lo:+.3f},{c.isd_rec_dr_hi:+.3f}] p={c.isd_rec_p:.3f}")

res = pd.DataFrame(rows)
res.to_csv(out / "metrics.csv", index=False)
preds_out.to_csv(out / "isd_predictions.csv", index=False)
show = ["features", "model", "target", "araus_cv_r2", "araus_test_r2", "araus_test_stim_r2",
        "isd_indiv_r2", "isd_rec_r2", "isd_rec_r2_centred", "isd_rec_r", "isd_rec_r_lo", "isd_rec_r_hi", "isd_loc_r", "isd_bias"]
(out / "metrics.md").write_text(
    "## Scores\n\n" + res[show].round(3).to_markdown(index=False)
    + "\n\n## Paired differences (A - B)\n\n" + comp.round(3).to_markdown(index=False) + "\n")
print(f"\nwrote {out/'metrics.csv'}, {out/'metrics.md'}, {out/'paired_differences.csv'}, {out/'isd_predictions.csv'}")
