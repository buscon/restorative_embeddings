#!/usr/bin/env python3
"""Copy a training set and replace the rating number in the caption by its pleasantness BIN as words.

    soundscape with natural sounds [ISOPleasant: 0.62]   ->   soundscape with natural sounds [ISOPleasant: very pleasant]

Why: the continuous number is split into irregular T5 tokens and each value is seen only a few times. With five labels
every label is seen hundreds of times, and with words the frozen text encoder already knows their meaning. Only the
value after "ISOPleasant:" changes; the rest of the caption, the audio and all training settings stay the same.
(--labels digits writes 1..5 instead, --labels words is the default.)

Bins are the ISD bins used everywhere in this repo (isd/02_select_balanced.py):
    <= -0.5 very unpleasant | <= -0.15 unpleasant | <= 0.15 neutral | <= 0.5 pleasant | above very pleasant

The audio files are linked (symlinks), not copied. The output folder must not exist yet or must be empty
(a folder that mixes files of two builds trains on both, which has happened before).

    python isd/08_make_words_set.py [--src isd/train/isd_all_whole] [--out isd/train/isd_all_whole_words]
Writes <out>/metadata.csv, <out>/dataset_config.json, <out>/bin_labels.csv (file, rating, bin, label) and the links.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BINS = ["very_unpleasant", "unpleasant", "neutral", "pleasant", "very_pleasant"]
WORDS = {"very_unpleasant": "very unpleasant", "unpleasant": "unpleasant", "neutral": "neutral",
         "pleasant": "pleasant", "very_pleasant": "very pleasant"}
DIGITS = {b: str(i + 1) for i, b in enumerate(BINS)}
TAG = re.compile(r"^(?P<text>.*?)\s*\[ISOPleasant:\s*(?P<val>[-+]?\d+(?:\.\d+)?)\]\s*$")


def pleasantness_bin(x):   # same thresholds as isd/02_select_balanced.py
    return ("very_unpleasant" if x <= -0.5 else "unpleasant" if x <= -0.15 else
            "neutral" if x <= 0.15 else "pleasant" if x <= 0.5 else "very_pleasant")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(REPO / "isd" / "train" / "isd_all_whole"))
    ap.add_argument("--out", default=str(REPO / "isd" / "train" / "isd_all_whole_words"))
    ap.add_argument("--labels", choices=["words", "digits"], default="words")
    a = ap.parse_args()
    src, out = Path(a.src).expanduser(), Path(a.out).expanduser()
    labels = WORDS if a.labels == "words" else DIGITS
    if not (src / "metadata.csv").exists():
        sys.exit(f"{src}/metadata.csv not found")
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty: use a new folder (or move the old one away)")
    out.mkdir(parents=True, exist_ok=True)

    rows, bad = [], 0
    for r in csv.DictReader(open(src / "metadata.csv", encoding="utf-8")):
        m = TAG.match(r["caption"])
        f = src / r["file"]
        if not m or not f.exists():
            bad += 1
            continue
        x = float(m["val"])
        b = pleasantness_bin(x)
        rows.append({"file": r["file"], "text": m["text"], "rating": x, "bin": b,
                     "caption": f"{m['text']} [ISOPleasant: {labels[b]}]", "target": f.resolve()})
    if not rows:
        sys.exit("no usable files")
    for r in rows:
        link = out / r["file"]
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(r["target"], link)
    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "caption"])
        w.writerows((r["file"], r["caption"]) for r in rows)
    with open(out / "bin_labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "rating", "bin", "label"])
        w.writerows((r["file"], r["rating"], r["bin"], labels[r["bin"]]) for r in rows)
    adapter = REPO / "generation" / "train" / "sa3_finetuning_data" / "metadata_adapter.py"
    (out / "dataset_config.json").write_text(json.dumps({
        "dataset_type": "audio_dir", "random_crop": False,
        "datasets": [{"id": "isd_words", "path": str(out.resolve()), "recursive": False,
                      "extensions": [".wav"], "custom_metadata_module": str(adapter)}]}, indent=2))

    uniq = defaultdict(set)
    cnt = Counter()
    for r in rows:
        cnt[r["bin"]] += 1
        uniq[r["bin"]].add(re.sub(r"_r\d+(?=\.wav$)", "", r["file"]))
    print(f"{len(rows)} files linked into {out} ({bad} skipped without tag or audio)")
    print(f"\n{'bin':16s}{'label':>18s}{'files':>8s}{'different':>11s}")
    for b in BINS:
        print(f"{b:16s}{labels[b]:>18s}{cnt[b]:8d}{len(uniq[b]):11d}")
    print("\nexamples:")
    for r in rows[:3] + rows[-2:]:
        print(f"  {r['file']}: {r['caption']}   (rating {r['rating']:+.2f})")
    print(f"\ndataset config: {out / 'dataset_config.json'}")


if __name__ == "__main__":
    main()
