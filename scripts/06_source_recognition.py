"""Step 6 - does CLAP recognise sound sources in the field?

Why: CLAP beats psychoacoustics clearly in ARAUS but not on ISD. One
explanation is that CLAP mainly recognises sound sources, and that the link
between sources and pleasantness differs between the lab (where it is set by
the added masker: birds, water, traffic, ...) and the field. This script
separates two questions:

  A. Recognition: do CLAP's zero-shot source scores match the source
     dominance people reported in situ? ISD asks "To what extent do you
     presently hear ..." (1 = not at all, 5 = dominates completely) for
       ssi01 traffic noise, ssi02 other noise, ssi03 human sounds,
       ssi04 natural sounds.
     As a lab check, the same scores are tested against the known masker
     class of each ARAUS stimulus (ROC AUC, per SMR).

  B. Source -> pleasantness: is the association between source scores and
     ISOPleasant the same in ARAUS and ISD? A source-only model trained on
     ARAUS is also applied to ISD.

If A holds but B differs, CLAP "hears" the field correctly and the problem is
how ARAUS ties sources to pleasantness. If A fails, the embeddings themselves
do not transfer well to field recordings.

Scores are cosine similarities between the stored audio embeddings and CLAP
text embeddings of several phrases per category (the example sounds in the
ISD questionnaire), averaged per category. "Relative" scores subtract the
mean over the four categories, which removes a general "busy / loud" factor.
Needs the real CLAP model (not RSD_FAKE_EMBED); runs in about a minute.

    python scripts/06_source_recognition.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import PredefinedSplit, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsd.embed import DEFAULT_MODEL  # noqa: E402

CATEGORIES = {  # ISD item -> phrases (from the questionnaire's own examples)
    "traffic": ("ssi01", ["cars driving past", "buses and trucks in street traffic", "a train passing",
                          "an airplane flying overhead", "road traffic noise"]),
    "other_noise": ("ssi02", ["a siren", "construction work", "industrial machinery",
                              "goods being loaded and unloaded", "a jackhammer"]),
    "human": ("ssi03", ["people having a conversation", "people laughing", "children playing",
                        "footsteps", "a crowd of people talking"]),
    "natural": ("ssi04", ["birds singing", "flowing water", "wind in the trees and leaves",
                          "a water fountain", "birdsong in a park"]),
}
ARAUS_CLASS = {"traffic": "traffic", "construction": "other_noise",
               "bird": "natural", "water": "natural", "wind": "natural"}

ap = argparse.ArgumentParser()
ap.add_argument("--processed", default="data/processed")
ap.add_argument("--features", default="data/features")
ap.add_argument("--out", default="results/sources")
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--n-boot", type=int, default=2000)
a = ap.parse_args()
proc, feat, out = Path(a.processed), Path(a.features), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)


# ---- text embeddings ------------------------------------------------------------
@torch.no_grad()
def text_embeddings():
    from transformers import ClapModel, ClapProcessor
    model = ClapModel.from_pretrained(a.model).eval()
    tok = ClapProcessor.from_pretrained(a.model).tokenizer
    res = {}
    for cat, (_, phrases) in CATEGORIES.items():
        inp = tok([f"the sound of {p}" for p in phrases], padding=True, return_tensors="pt")
        o = model.get_text_features(**inp)
        e = (o if torch.is_tensor(o) else o.pooler_output).numpy()
        res[cat] = e / np.linalg.norm(e, axis=1, keepdims=True)
    return res


def load_audio(name):
    z = np.load(feat / f"{name}_clap.npz", allow_pickle=False)
    e = z["emb"]
    if e.shape[1] != 512:
        sys.exit(f"{name}_clap.npz has {e.shape[1]} dims - was it made with RSD_FAKE_EMBED? Needs real CLAP.")
    return pd.Index(z["ids"]), e / np.linalg.norm(e, axis=1, keepdims=True)


def scores(emb, txt):
    raw = pd.DataFrame({c: (emb @ t.T).mean(axis=1) for c, t in txt.items()})
    rel = raw.sub(raw.mean(axis=1), axis=0).add_suffix("_rel")
    return pd.concat([raw, rel], axis=1)


def draws(groups):
    u, inv = np.unique(groups, return_inverse=True)
    mem = [np.flatnonzero(inv == k) for k in range(len(u))]
    for _ in range(a.n_boot):
        yield np.concatenate([mem[k] for k in rng.integers(0, len(u), len(u))])


def r_ci(x, y, groups):
    x, y = np.asarray(x, float), np.asarray(y, float)
    bs = [np.corrcoef(x[t], y[t])[0, 1] for t in draws(groups)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return np.corrcoef(x, y)[0, 1], lo, hi


txt = text_embeddings()
cats = list(CATEGORIES)

# ---- ISD --------------------------------------------------------------------------
ids, emb = load_audio("isd")
S_isd = scores(emb, txt).set_index(ids)
ratings = pd.read_csv(proc / "isd_ratings.csv", low_memory=False)
ssi = [CATEGORIES[c][0] for c in cats]
rec = (ratings[ratings.has_audio].groupby("GroupID")
       .agg(LocationID=("LocationID", "first"), ISOPleasant=("ISOPleasant", "mean"),
            **{s: (s, "mean") for s in ssi}))
isd = rec.join(S_isd, how="inner").dropna(subset=ssi)
S_isd.to_csv(out / "isd_source_scores.csv")

print(f"A. Recognition on ISD ({len(isd)} rated recordings, {isd.LocationID.nunique()} locations)")
print("   r between CLAP score and rated dominance; 95 % CI by location bootstrap")
rowsA = []
for c in cats:
    s = CATEGORIES[c][0]
    for kind in ["", "_rel"]:
        r, lo, hi = r_ci(isd[c + kind], isd[s], isd.LocationID.values)
        loc = isd.groupby("LocationID")[[c + kind, s]].mean()
        rowsA.append({"category": c, "item": s, "score": "relative" if kind else "raw",
                      "r_recording": r, "lo": lo, "hi": hi, "r_location": loc.corr().iloc[0, 1]})
A = pd.DataFrame(rowsA)
A.to_csv(out / "isd_recognition.csv", index=False)
print(A.round(3).to_string(index=False))

M = pd.DataFrame({c: [np.corrcoef(isd[c + "_rel"], isd[s])[0, 1] for s in ssi] for c in cats}, index=ssi)
M.to_csv(out / "isd_recognition_matrix.csv")
print("\n   Matrix: rows = rated item, columns = CLAP relative score (diagonal should be highest)")
print(M.round(2).to_string())

# ---- ARAUS lab check ----------------------------------------------------------------
a_ids, a_emb = load_audio("araus")
S_ar = scores(a_emb, txt).set_index(a_ids)
stim = pd.read_csv(proc / "araus_stimuli.csv").set_index("stimulus_id").join(S_ar, how="inner")
stim["cls"] = stim.masker_type.map(ARAUS_CLASS)  # silence -> NaN
rowsL = []
for c in ["traffic", "other_noise", "natural"]:
    for smr, g in stim.groupby("smr"):
        g = g[g.masker_type != "silence"]
        y = (g.cls == c).astype(int)
        if 0 < y.sum() < len(y):
            rowsL.append({"category": c, "smr": smr, "auc_relative": roc_auc_score(y, g[c + "_rel"]),
                          "auc_raw": roc_auc_score(y, g[c]), "n": len(g)})
L = pd.DataFrame(rowsL)
L.to_csv(out / "araus_masker_auc.csv", index=False)
print("\nA2. Lab check: ROC AUC for detecting the ARAUS masker class (0.5 = chance)")
print("    SMR = soundscape minus masker level; -6 dB = masker 6 dB louder")
print(L.pivot(index="category", columns="smr", values="auc_relative").round(2).to_string())

# ---- B: source -> pleasantness ---------------------------------------------------------
print("\nB. Correlation of CLAP relative source scores with ISOPleasant")
rowsB = []
for c in cats:
    rowsB.append({"category": c,
                  "araus_r": np.corrcoef(stim[c + "_rel"], stim.ISOPleasant)[0, 1],
                  "isd_r": np.corrcoef(isd[c + "_rel"], isd.ISOPleasant)[0, 1],
                  "isd_r_rated_item": np.corrcoef(isd[CATEGORIES[c][0]], isd.ISOPleasant)[0, 1]})
B = pd.DataFrame(rowsB)
B.to_csv(out / "source_pleasantness.csv", index=False)
print(B.round(3).to_string(index=False))
print("   (isd_r_rated_item uses the people's own source ratings - the field 'truth')")

X = [c + "_rel" for c in cats]
tr = stim[stim.fold.between(1, 5)]
oof = cross_val_predict(LinearRegression(), tr[X], tr.ISOPleasant, cv=PredefinedSplit(tr.fold.values - 1))
m = LinearRegression().fit(tr[X], tr.ISOPleasant)
r, lo, hi = r_ci(m.predict(isd[X]), isd.ISOPleasant, isd.LocationID.values)
print(f"\n   Source-only model (4 scores): ARAUS CV R2 {r2_score(tr.ISOPleasant, oof):.3f} (stimulus means) | "
      f"ISD recording r {r:.2f} [{lo:.2f}, {hi:.2f}]")
print(f"   coefficients: " + ", ".join(f"{c} {v:+.2f}" for c, v in zip(cats, m.coef_)))
print(f"\nwrote {out}/")
