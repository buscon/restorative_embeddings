#!/usr/bin/env python3
"""Copy a training set, remove the "[ISOPleasant: x]" tag from the caption and store x as a number instead.

    soundscape with natural sounds [ISOPleasant: 0.62]   ->   caption "soundscape with natural sounds", score 0.62

For a model with an extra number conditioner ("pleasantness", see generation/make_score_config.py), so the rating
reaches the model only through that number and not through the text. Audio files are symlinked. The output
folder must not exist yet or must be empty.
    python isd/10_make_score_set.py [--src isd/train/isd_all_whole] [--out isd/train/isd_all_whole_score]
Writes <out>/metadata.csv (file, caption, score), dataset_config.json (custom_metadata_module =
generation/train/sa3_finetuning_data/score_metadata_adapter.py) and the links. Files without a tag are skipped.
"""
import argparse, csv, json, os, re, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
TAG = re.compile(r"^(?P<text>.*?)\s*\[ISOPleasant:\s*(?P<val>[-+]?\d+(?:\.\d+)?)\]\s*$")
ADAPTER = REPO / "generation/train/sa3_finetuning_data/score_metadata_adapter.py"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(REPO / "isd/train/isd_all_whole"))
    ap.add_argument("--out", default=str(REPO / "isd/train/isd_all_whole_score"))
    a = ap.parse_args()
    src, out = Path(a.src).expanduser(), Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty; choose another --out")
    out.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    with open(src / "metadata.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = TAG.match(r["caption"])
            wav = src / r["file"]
            if not m or not wav.exists():
                skipped += 1
                continue
            rows.append((r["file"], m["text"].strip(), float(m["val"])))
    for name, _, _ in rows:
        os.symlink((src / name).resolve(), out / name)
    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "caption", "score"])
        w.writerows((n, c, f"{s:.4f}") for n, c, s in rows)
    cfg = {"dataset_type": "audio_dir", "random_crop": False,
           "datasets": [{"id": "isd_score", "path": str(out), "recursive": False, "extensions": [".wav"],
                         "custom_metadata_module": str(ADAPTER)}]}
    (out / "dataset_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    sc = sorted(s for _, _, s in rows)
    print(f"{len(rows)} files linked into {out} ({skipped} skipped without tag or audio)")
    if sc:
        print(f"score range {sc[0]:+.2f} .. {sc[-1]:+.2f}, median {sc[len(sc)//2]:+.2f}")
    print("examples:")
    for n, c, s in rows[:5]:
        print(f"  {n}: {c!r}  score {s:+.2f}")
    print("dataset config:", out / "dataset_config.json")


if __name__ == "__main__":
    main()
