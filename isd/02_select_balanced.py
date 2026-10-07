#!/usr/bin/env python3
"""Select a balanced subset of rated ISD recordings for fine-tuning.

Balance: equal numbers per city (London, Venice, Granada, Groningen), and within each city
as equal as availability allows across the five ISOPleasant bins. Within a (city, bin) cell
recordings are taken round-robin across the locations of that city, so no single square
dominates. If a city or bin cannot supply its share, the shortfall goes to the others.

ISOPleasant is the mean over the people who rated that recording (ISD gives each person one
rating per recording, so single ratings are noisy; use --min-ratings 2 to demand more).

    python isd/02_select_balanced.py --isd data/raw/isd --output-csv isd/selected_isd.csv \
        --num-samples 360 --seed 42 [--min-ratings 1]

Alternatively read the table written by scripts/01_prepare.py:
    --recordings-csv data/processed/isd_recordings.csv
"""
import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

BINS = ["very_unpleasant", "unpleasant", "neutral", "pleasant", "very_pleasant"]
CITIES = ["London", "Venice", "Granada", "Groningen"]


def pleasantness_bin(x: float) -> str:
    if x <= -0.5:
        return "very_unpleasant"
    if x <= -0.15:
        return "unpleasant"
    if x <= 0.15:
        return "neutral"
    if x <= 0.5:
        return "pleasant"
    return "very_pleasant"


def city_of(path: str) -> str:
    m = re.search("|".join(CITIES), str(path), flags=re.I)
    return {c.lower(): c for c in CITIES}[m.group(0).lower()] if m else "unknown"


def waterfill(quota: int, caps: dict) -> dict:
    """Split quota as evenly as possible over keys, never exceeding caps[key]."""
    out = {k: 0 for k in caps}
    active = [k for k in caps if caps[k] > 0]
    left = min(quota, sum(caps.values()))
    while left > 0 and active:
        share, extra = divmod(left, len(active))
        if share == 0:
            for k in sorted(active, key=lambda k: -(caps[k] - out[k]))[:extra]:
                out[k] += 1
            break
        done = []
        for k in active:
            give = min(share, caps[k] - out[k])
            out[k] += give
            left -= give
            if out[k] >= caps[k]:
                done.append(k)
        active = [k for k in active if k not in done]
    return out


def round_robin(rows: pd.DataFrame, k: int, rng: random.Random) -> list:
    by_loc = defaultdict(list)
    for r in rows.to_dict("records"):
        by_loc[r["LocationID"]].append(r)
    locs = sorted(by_loc)
    for loc in locs:
        rng.shuffle(by_loc[loc])
    picked = []
    while len(picked) < k and any(by_loc.values()):
        order = locs[:]
        rng.shuffle(order)
        for loc in order:
            if by_loc[loc] and len(picked) < k:
                picked.append(by_loc[loc].pop())
    return picked


def select(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    cities = sorted(df.city.unique())
    city_quota = waterfill(n, {c: int((df.city == c).sum()) for c in cities})
    picked = []
    for c in cities:
        sub = df[df.city == c]
        bin_quota = waterfill(city_quota[c], {b: int((sub.bin == b).sum()) for b in BINS})
        for b in BINS:
            if bin_quota[b]:
                picked += round_robin(sub[sub.bin == b], bin_quota[b], rng)
    return pd.DataFrame(picked)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isd", default="data/raw/isd")
    ap.add_argument("--recordings-csv", help="use isd_recordings.csv from 01_prepare.py instead of --isd")
    ap.add_argument("--output-csv", default="isd/selected_isd.csv")
    ap.add_argument("--num-samples", type=int, default=360)
    ap.add_argument("--min-ratings", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.recordings_csv:
        recs = pd.read_csv(a.recordings_csv)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from rsd.data import load_isd
        recs = load_isd(a.isd)["recordings"]

    recs = recs[(recs.n_ratings >= a.min_ratings) & recs.ISOPleasant.notna() & recs.wav.notna()].copy()
    recs["city"] = recs.wav.map(city_of)
    recs["bin"] = recs.ISOPleasant.map(pleasantness_bin)
    print(f"{len(recs)} rated recordings with audio (min ratings {a.min_ratings})")
    print("\navailable, city x bin:")
    print(pd.crosstab(recs.city, recs.bin).reindex(columns=BINS, fill_value=0).to_string())
    if (recs.city == "unknown").any():
        print(f"\nWARNING: {(recs.city == 'unknown').sum()} recordings have no recognisable city in their path")

    sel = select(recs, a.num_samples, a.seed)
    cols = ["GroupID", "wav", "city", "LocationID", "ISOPleasant", "ISOEventful", "n_ratings", "bin"]
    sel = sel[[c for c in cols if c in sel.columns]].sort_values(["city", "bin", "GroupID"])
    Path(a.output_csv).parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(a.output_csv, index=False)

    print(f"\nselected {len(sel)} -> {a.output_csv}")
    print(pd.crosstab(sel.city, sel.bin).reindex(columns=BINS, fill_value=0).to_string())
    print("\nrecordings per location:")
    print(sel.groupby(["city", "LocationID"]).size().to_string())
    short = [b for b in BINS if (sel.bin == b).sum() < a.num_samples / len(BINS) * 0.8]
    if short:
        print(f"\nNote: bins below 80% of an even share (availability limit): {', '.join(short)}")


if __name__ == "__main__":
    main()
