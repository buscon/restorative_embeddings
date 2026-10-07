"""Step 4 - unsupervised structure of the CLAP space.

k-means is fitted on the ARAUS stimuli only (the training data), then ISD
recordings are assigned to the nearest ARAUS cluster. Each cluster is then
described by:
    * mean ISOPleasant / ISOEventful in ARAUS (lab) and in ISD (field),
    * which masker types dominate it in ARAUS (bird, water, traffic, ...),
    * which ISD locations fall into it,
    * mean ecoacoustic indices and level.
If a cluster is pleasant in the lab *and* in the field, that is evidence that
the embedding captures something transferable, independent of any regressor.

HDBSCAN is run separately on the ISD recordings as an exploratory check of
the field data's own structure (it does not need k and can mark noise).

    python scripts/04_clusters.py --k 10
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsd.data import TARGETS  # noqa: E402
from rsd.indices import INDEX_COLUMNS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--processed", default="data/processed")
ap.add_argument("--features", default="data/features")
ap.add_argument("--out", default="results/clusters")
ap.add_argument("--k", type=int, default=None, help="number of k-means clusters (default: best silhouette in 4-16)")
ap.add_argument("--pca", type=int, default=50)
ap.add_argument("--min-cluster-size", type=int, default=15)
a = ap.parse_args()
proc, feat, out = Path(a.processed), Path(a.features), Path(a.out)
out.mkdir(parents=True, exist_ok=True)


def load(name):
    z = np.load(feat / f"{name}_clap.npz", allow_pickle=False)
    ind = feat / f"{name}_indices.csv"
    ind = pd.read_csv(ind).set_index("id") if ind.exists() else pd.DataFrame()
    return z["ids"], z["emb"], ind


a_ids, a_emb, a_ind = load("araus")
i_ids, i_emb, i_ind = load("isd")
stim = pd.read_csv(proc / "araus_stimuli.csv").set_index("stimulus_id").loc[a_ids]
recs = pd.read_csv(proc / "isd_recordings.csv").set_index("GroupID").loc[i_ids]

scaler = StandardScaler().fit(a_emb)
pca = PCA(min(a.pca, a_emb.shape[1]), random_state=0).fit(scaler.transform(a_emb))
Za, Zi = pca.transform(scaler.transform(a_emb)), pca.transform(scaler.transform(i_emb))
print(f"PCA {pca.n_components_} comps explain {pca.explained_variance_ratio_.sum():.1%} of ARAUS embedding variance")

# ---- choose k -----------------------------------------------------------------
rng = np.random.default_rng(0)
sample = rng.choice(len(Za), min(5000, len(Za)), replace=False)
sil = {}
for k in range(4, 17):
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(Za)
    sil[k] = silhouette_score(Za[sample], lab[sample])
pd.Series(sil, name="silhouette").rename_axis("k").to_csv(out / "kmeans_silhouette.csv")
k = a.k or max(sil, key=sil.get)
print("silhouette by k:", {kk: round(v, 3) for kk, v in sil.items()}, "-> k =", k)
print("(low silhouette values mean the space has no sharp clusters; treat clusters as regions, not types)")

km = KMeans(k, n_init=20, random_state=0).fit(Za)
stim["cluster"], recs["cluster"] = km.labels_, km.predict(Zi)

# ---- profiles -------------------------------------------------------------------
stim = stim.join(a_ind, how="left")
recs = recs.join(i_ind, how="left")
rated = recs[recs.n_ratings > 0]
prof = []
for c in range(k):
    s, r, rr = stim[stim.cluster == c], recs[recs.cluster == c], rated[rated.cluster == c]
    row = {"cluster": c, "n_araus_stimuli": len(s), "n_isd_recordings": len(r), "n_isd_rated": len(rr),
           "araus_ISOPleasant": s.ISOPleasant.mean(), "araus_ISOEventful": s.ISOEventful.mean(),
           "isd_ISOPleasant": rr.ISOPleasant.mean(), "isd_ISOEventful": rr.ISOEventful.mean(),
           "araus_maskers": ", ".join(f"{m} {v:.0%}" for m, v in s.masker_type.value_counts(normalize=True).head(3).items()),
           "isd_locations": ", ".join(f"{m} ({v})" for m, v in r.LocationID.value_counts().head(3).items()),
           "LA50_araus": s.LA50.mean(), "LA50_isd": rr.LA50.mean()}
    for col in INDEX_COLUMNS:
        if col in stim:
            row[f"{col}_araus"], row[f"{col}_isd"] = s[col].mean(), r[col].mean()
    prof.append(row)
prof = pd.DataFrame(prof)
prof.to_csv(out / "kmeans_profiles.csv", index=False)

both = prof.dropna(subset=["isd_ISOPleasant"])
both = both[both.n_isd_rated >= 5]
if len(both) >= 3:
    for t in TARGETS:
        r = np.corrcoef(both[f"araus_{t}"], both[f"isd_{t}"])[0, 1]
        print(f"cluster-level agreement lab vs field, {t}: r = {r:.2f} over {len(both)} clusters with >=5 rated ISD recordings")

stim[["cluster"]].to_csv(out / "araus_clusters.csv")
recs[["cluster", "LocationID", "n_ratings"] + TARGETS].to_csv(out / "isd_clusters.csv")

# ---- HDBSCAN on ISD alone ----------------------------------------------------------
h = HDBSCAN(min_cluster_size=a.min_cluster_size).fit(Zi)
recs["hdbscan"] = h.labels_
hs = (recs.groupby("hdbscan")
      .agg(n=("cluster", "size"), n_rated=("n_ratings", lambda x: (x > 0).sum()),
           ISOPleasant=("ISOPleasant", "mean"), ISOEventful=("ISOEventful", "mean"),
           top_locations=("LocationID", lambda x: ", ".join(x.value_counts().head(3).index))))
hs.to_csv(out / "isd_hdbscan_profiles.csv")
print(f"HDBSCAN on ISD: {len(hs) - (-1 in hs.index)} clusters, {(h.labels_ == -1).mean():.0%} noise")

# ---- picture ------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
ax[0].scatter(Za[:, 0], Za[:, 1], c=stim.ISOPleasant, s=2, cmap="RdYlGn", vmin=-0.8, vmax=0.8)
ax[0].set_title("ARAUS stimuli, coloured by ISOPleasant")
sc = ax[1].scatter(Zi[:, 0], Zi[:, 1], c=recs.ISOPleasant.fillna(0), s=8, cmap="RdYlGn", vmin=-0.8, vmax=0.8,
                   alpha=np.where(recs.n_ratings > 0, 1, 0.15))
ax[1].set_title("ISD recordings (faint = unrated)")
for x in ax:
    x.set_xlabel("PC1"); x.set_ylabel("PC2")
fig.colorbar(sc, ax=ax, label="ISOPleasant")
fig.savefig(out / "pca_isopleasant.png", dpi=150, bbox_inches="tight")
print(f"wrote profiles and figure to {out}/")
print(prof[["cluster", "n_araus_stimuli", "n_isd_rated", "araus_ISOPleasant", "isd_ISOPleasant",
            "araus_maskers", "isd_locations"]].round(2).to_string(index=False))
