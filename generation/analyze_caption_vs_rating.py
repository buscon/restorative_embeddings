#!/usr/bin/env python3
"""How much of the rating does the rest of the caption already explain?

If the scene text alone (caption without the rating) predicts ISOPleasant well, a text-to-audio
model has little reason to read the rating from the caption: it can fit the audio with the scene
words. The rating then only matters through the part the text does NOT explain, the within-text
spread, and that part has to be visible in the audio.

Reads a CSV with a caption column in the training format "<scene> [ISOPleasant: x]" (or any text
that contains "ISOPleasant: x"), e.g. the metadata written by 02_make_training_chunks.py or
isd/03_make_training_chunks.py, and reports

    * number of distinct scene texts, rating sd overall and within texts
    * eta^2: share of rating variance between distinct scene texts (adjusted for group count)
    * cross-validated R^2 of a ridge regression on word 1-2-grams of the scene text (5-fold)
    * the same R^2 with texts shuffled (should be about 0)
    * optionally (--t5) how the T5 tokenizer splits the rating part

    python generation/analyze_caption_vs_rating.py --csv <metadata.csv> [--caption-col caption]

Reading: R^2 near 1 means the text already gives the rating; the model can ignore the rating word.
Within-text sd small relative to overall sd means the same.
"""
import argparse
import re

import numpy as np
import pandas as pd

RATING = re.compile(r"\s*[\[(]?\s*ISOPleasant:\s*(-?\d+(?:\.\d+)?)\s*[\])]?")


def split_caption(c):
    m = RATING.search(c)
    if not m:
        return c.strip(), np.nan
    return RATING.sub("", c).strip(" .,"), float(m.group(1))


def eta_squared(text, y):
    df = pd.DataFrame({"t": text, "y": y})
    g = df.groupby("t").y
    n, k = len(df), g.ngroups
    ss_tot = ((df.y - df.y.mean()) ** 2).sum()
    ss_bet = (g.transform("mean").sub(df.y.mean()) ** 2).sum()
    ss_wit = ss_tot - ss_bet
    eta = ss_bet / ss_tot
    adj = 1 - (ss_wit / (n - k)) / (ss_tot / (n - 1)) if n > k else np.nan  # adjusted R^2 of the group model
    sd_within = np.sqrt(ss_wit / max(n - k, 1))
    return eta, adj, sd_within, k


def cv_r2(text, y, seed=0, shuffle=False):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    t = np.asarray(text, dtype=object)
    if shuffle:
        t = np.random.default_rng(seed).permutation(t)
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), lowercase=True), RidgeCV(alphas=np.logspace(-3, 2, 12)))
    pred = cross_val_predict(pipe, t, y, cv=KFold(5, shuffle=True, random_state=seed))
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--caption-col", default="caption")
    ap.add_argument("--t5", action="store_true", help="show how the T5 tokenizer splits rating strings")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    parts = df[a.caption_col].astype(str).map(split_caption)
    text = np.array([p[0] for p in parts], dtype=object)
    y = np.array([p[1] for p in parts], dtype=float)
    ok = ~np.isnan(y)
    text, y = text[ok], y[ok]
    print(f"{len(y)} clips with a rating in the caption (of {len(df)})")
    print(f"example scene text: {text[0]!r}")
    print(f"rating: mean {y.mean():.2f}, sd {y.std(ddof=1):.3f}, range {y.min():.2f} to {y.max():.2f}")
    eta, adj, sdw, k = eta_squared(text, y)
    print(f"{k} distinct scene texts; within-text rating sd {sdw:.3f} (overall {y.std(ddof=1):.3f})")
    print(f"eta^2 (variance between texts) {eta:.2f}, adjusted {adj:.2f}")
    r2 = cv_r2(text, y)
    r2s = np.mean([cv_r2(text, y, seed=s, shuffle=True) for s in range(5)])
    print(f"5-fold CV R^2 of rating from scene text alone: {r2:.2f}   (texts shuffled: {r2s:.2f})")
    if r2 > 0.6:
        print("-> the scene text already explains most of the rating; the rating word adds little the model needs.")
    elif r2 < 0.2:
        print("-> the scene text explains little; the rating word carries information the audio has to show.")
    else:
        print("-> partly explained by the scene text.")
    if a.t5:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("google-t5/t5-base")
        for v in ["-0.68", "-0.15", "0.00", "0.15", "0.28", "0.50", "0.80"]:
            print(f"  [ISOPleasant: {v}] ->", tok.tokenize(f"[ISOPleasant: {v}]"))


if __name__ == "__main__":
    main()
