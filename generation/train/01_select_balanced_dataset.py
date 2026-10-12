#!/usr/bin/env python3
"""Select a subset of ARAUS stimuli for training.

Pleasantness bins (same thresholds as the ISD scripts, from the number ISOPleasant):
very_unpleasant <= -0.5 < unpleasant <= -0.15 < neutral <= 0.15 < pleasant <= 0.5 < very_pleasant.

--balance proportional (default, as before): each bin gets a share proportional to its size.
--balance equal: every bin gets the same number of stimuli, spread round-robin over the
    masker x (foreground/background) cells of that bin, so every masker type appears at every
    pleasantness level as far as the data allow. (Use this for the ISD-like training set.)

Filters (applied before selection), for stimuli you do not want to train on:
--exclude-bases FILE   text file, one base soundscape per line (first field of stimulus_id,
                       "<soundscape>|<masker file>|<smr>"); the base and all its mixes are dropped,
                       e.g. bases with music in the background (see joint/screen_araus_music.py).
--maskers a,b          keep only these masker types (bird, construction, traffic, water, wind).
--no-foreground a,b    drop these masker types when the masker is louder than the soundscape (SMR < 0).

    python generation/train/01_select_balanced_dataset.py \
        --metadata data/generation/export_metadata.csv --output-csv generation/train/selected_araus_800.csv \
        --num-samples 800 --balance equal
Output: CSV with the metadata columns (export_id first) plus a "bin" column computed here.
"""
import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

BINS = ["very_unpleasant", "unpleasant", "neutral", "pleasant", "very_pleasant"]


def pleasantness_bin(p):
    return ("very_unpleasant" if p <= -0.5 else "unpleasant" if p <= -0.15 else
            "neutral" if p <= 0.15 else "pleasant" if p <= 0.5 else "very_pleasant")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--num-samples", type=int, default=360)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--balance", choices=["proportional", "equal"], default="proportional")
    ap.add_argument("--exclude-bases", default="")
    ap.add_argument("--maskers", default="")
    ap.add_argument("--no-foreground", default="")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    excl = {l.strip() for l in open(a.exclude_bases)} - {""} if a.exclude_bases else set()
    keep_m = {m for m in a.maskers.split(",") if m}
    no_fg = {m for m in a.no_foreground.split(",") if m}

    by_bin = defaultdict(list)
    n_all = n_drop = 0
    with open(a.metadata, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                p = float(r["ISOPleasant"])
                parts = r["stimulus_id"].split("|")
                masker, smr = parts[1].split("_")[0], float(parts[2])
            except (KeyError, ValueError, IndexError):
                continue
            if p != p:
                continue
            n_all += 1
            if parts[0] in excl or (keep_m and masker not in keep_m) or (smr < 0 and masker in no_fg):
                n_drop += 1
                continue
            r = dict(r)
            if "id" in r and "export_id" not in r:
                r["export_id"] = r.pop("id")
            r["bin"] = pleasantness_bin(p)
            r["_cell"] = (masker, smr < 0)
            by_bin[r["bin"]].append(r)
    total = sum(len(v) for v in by_bin.values())
    print(f"{n_all} stimuli, {n_drop} removed by filters, {total} available")
    if not total:
        sys.exit("nothing left to select from")
    n = min(a.num_samples, total)

    selected = []
    for b in BINS:
        items = by_bin.get(b, [])
        if a.balance == "proportional":
            k = min(len(items), max(1, int(n * len(items) / total))) if items else 0
            picked = rng.sample(items, k)
        else:
            k = min(len(items), n // len(BINS))
            cells = defaultdict(list)
            for r in items:
                cells[r["_cell"]].append(r)
            for v in cells.values():
                rng.shuffle(v)
            picked = []
            while len(picked) < k and any(cells.values()):
                for key in sorted(cells):
                    if cells[key] and len(picked) < k:
                        picked.append(cells[key].pop())
        selected += picked
        print(f"  {b:16s} available {len(items):5d}  selected {len(picked):5d}")
    print(f"  {'total':16s} available {total:5d}  selected {len(selected):5d}")

    fields = [k for k in selected[0] if k != "_cell"]
    fields.remove("export_id")
    fields.insert(0, "export_id")
    out = Path(a.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(selected)
    print(f"saved {len(selected)} stimuli to {out}")


if __name__ == "__main__":
    main()
