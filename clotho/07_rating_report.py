#!/usr/bin/env python3
"""What the SoundAQnet ratings of the Clotho clips look like, and what a stability filter would keep.

Reads clotho/rated.csv (06_rate_soundaqnet.py) and prints
  * the distribution of the rating (mean over 4 checkpoints x 3 assumed levels)
  * how large the instability is compared with the spread between clips
    (isop_sd: sd over all predictions, isop_sd_ckpt: between checkpoints, isop_sd_lvl: between assumed levels)
  * for several filters: clips and hours kept, rating distribution after the filter, and how the
    content changes (share of clips with Music or Speech likely, scene classes), because a stability
    filter can quietly select a kind of clip.
No filter is applied to any file here. Thresholds are chosen in the training-set script.

    python clotho/07_rating_report.py [--rated clotho/rated.csv]
"""
import argparse

import numpy as np
import pandas as pd


def describe(d, label, total_h):
    r = d.isop_mean
    print(f"{label:<34} n={len(d):>5} {d.dur.sum() / 3600:5.1f} h  rating mean {r.mean():+.2f} sd {r.std():.2f}"
          f"  range {r.min():+.2f}..{r.max():+.2f}  <=-0.15: {100 * (r <= -.15).mean():3.0f}%"
          f"  middle: {100 * ((r > -.15) & (r < .15)).mean():3.0f}%  >=0.15: {100 * (r >= .15).mean():3.0f}%"
          f"  music>.5: {100 * (d.ev_Music > .5).mean():3.0f}%  speech>.5: {100 * (d.ev_Speech > .5).mean():3.0f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rated", default="clotho/rated.csv")
    a = ap.parse_args()
    d = pd.read_csv(a.rated)
    if "dur" not in d:
        d["dur"] = d["sec"]
    print(f"{len(d)} clips, {d.dur.sum() / 3600:.1f} h\n")
    r = d.isop_mean
    print("rating (ISOPleasant, -1..1) percentiles:", {p: round(float(np.percentile(r, p)), 2) for p in [0, 5, 25, 50, 75, 95, 100]})
    print(f"spread between clips (sd of the rating): {r.std():.3f}")
    for c in ["isop_sd", "isop_sd_ckpt", "isop_sd_lvl"]:
        print(f"  {c:<13} median {d[c].median():.3f}  mean {d[c].mean():.3f}   (ratio to between-clip sd: {d[c].median() / r.std():.2f})")
    print("  isop_sd_lvl is the change of the rating when the assumed level moves by 5 dB; it is not an error of the rater,\n"
          "  it is the price of not knowing the real level.\n")
    print(f"correlation of instability with the rating: sd vs |rating| r={np.corrcoef(d.isop_sd, r.abs())[0, 1]:+.2f}, "
          f"sd vs rating r={np.corrcoef(d.isop_sd, r)[0, 1]:+.2f}\n")
    print("filters (keep the most stable clips):")
    describe(d, "all", 0)
    for col, label in [("isop_sd", "total sd"), ("isop_sd_lvl", "level sd"), ("isop_sd_ckpt", "checkpoint sd")]:
        for q in [75, 50, 25]:
            t = np.percentile(d[col], q)
            describe(d[d[col] <= t], f"{label} <= {t:.3f} (best {q}%)", 0)
    t1, t2 = d.isop_sd_lvl.median(), d.isop_sd_ckpt.median()
    describe(d[(d.isop_sd_lvl <= t1) & (d.isop_sd_ckpt <= t2)], f"lvl<={t1:.2f} and ckpt<={t2:.2f}", 0)
    print("\nscene class (argmax) over all clips:", d.scene.value_counts(normalize=True).round(2).to_dict())


if __name__ == "__main__":
    main()
