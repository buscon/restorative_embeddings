#!/usr/bin/env python3
"""Replace the rows of one prompt word in a finished grid by a re-generated partial grid.

The old grid (e.g. araus_6000_g12) has files and tables made with the word "birdsong". The new partial grid
(e.g. araus_6000_g12_birds: only the re-generated prompt, with metrics.csv, psycho.csv, soundaqnet.csv from
isd/12_measure_grids.sh) has the same prompt with the new wording. This script writes a third folder that holds the
complete grid with the new rows instead of the old ones. The old and the new folder are not changed.

  <out>/*.wav            symbolic links: old files without the word, plus the new files
  <out>/grid.csv, prompts.csv, metrics.csv, psycho.csv, soundaqnet.csv
                         old rows without the word (any text cell containing it), then the new rows

Checks: the number of removed rows must equal the number of new rows in every table, and no old row with the word
may remain. evaluation.txt is not copied (it would describe the old files). Run evaluate_grid.py on <out> if you need it.

    python generation/merge_grid_birds.py --old ~/grids_3msde/araus_6000_g12 --new ~/grids_3msde/araus_6000_g12_birds \\
        --out ~/grids_3msde/araus_6000_g12_chirp [--word birdsong]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

TABLES = ["grid.csv", "prompts.csv", "metrics.csv", "psycho.csv", "soundaqnet.csv"]


def has_word(df, word):
    m = pd.Series(False, index=df.index)
    for c in df.columns:
        if df[c].dtype == object:
            m |= df[c].astype(str).str.contains(word, case=False, regex=False)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--word", default="birdsong")
    a = ap.parse_args()
    old, new, out = (Path(x).expanduser().resolve() for x in (a.old, a.new, a.out))
    if out in (old, new):
        sys.exit("--out must be a third folder")
    out.mkdir(parents=True, exist_ok=True)

    for name in TABLES:
        o, n = old / name, new / name
        if not o.exists() and not n.exists():
            continue
        if o.exists() != n.exists():
            sys.exit(f"!! {name} exists in only one of the folders ({'old' if o.exists() else 'new'}); "
                     f"run isd/12_measure_grids.sh on {new.name} first")
        od, nd = pd.read_csv(o), pd.read_csv(n)
        drop = has_word(od, a.word)
        if has_word(nd, a.word).any():
            sys.exit(f"!! the new {name} still contains '{a.word}'")
        if drop.sum() != len(nd):
            sys.exit(f"!! {name}: {drop.sum()} old rows with '{a.word}' but {len(nd)} new rows; the grids do not match "
                     f"(seeds, levels or prompts differ)")
        pd.concat([od[~drop], nd], ignore_index=True).to_csv(out / name, index=False)
        print(f"{name}: removed {drop.sum()}, added {len(nd)}, total {len(od) - drop.sum() + len(nd)}")

    n_link = 0
    for src_dir in (old, new):
        for w in sorted(src_dir.glob("*.wav")):
            if src_dir == old and a.word.lower() in w.name.lower():
                continue
            link = out / w.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(w)
            n_link += 1
    print(f"{n_link} wav links in {out}")


if __name__ == "__main__":
    main()
