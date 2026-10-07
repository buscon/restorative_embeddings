#!/usr/bin/env python3
"""Cut the selected ARAUS wavs into 10 s chunks and write captions for training.

Input : selected CSV from 01_select_balanced_dataset.py (needs export_id, stimulus_id,
        ISOPleasant; wav_path is used if present) and the exported audio from
        generation/02_export.py.
Output: <out>/<export_id>_c<i>.wav   (10 s, only complete chunks)
        <out>/metadata.csv           (file,caption)  -> read by metadata_adapter.py
        <out>/dataset_config.json    stable-audio-tools dataset config with absolute paths
                                     for THIS machine (use it with train.py --dataset-config)

Caption: "park soundscape with <masker> in the foreground|background [ISOPleasant: x.xx]"
(see rsd/captions.py). stimulus_id has the form  <soundscape>|<masker file>|<smr>.

    python generation/train/02_make_training_chunks.py \
        --selected generation/train/selected_360.csv \
        --audio-dir data/generation/audio \
        --out generation/train/araus_for_sa3_360
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rsd.captions import MASKER_WORDS, make_caption  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected", required=True)
    ap.add_argument("--audio-dir", default="data/generation/audio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-seconds", type=float, default=10.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_out, n_files, n_missing = [], 0, 0

    with open(args.selected, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        export_id = r.get("export_id") or r.get("id")
        src = Path(r["wav_path"]) if r.get("wav_path") else Path(args.audio_dir) / f"{export_id}.wav"
        if not src.exists():
            alt = Path(args.audio_dir) / f"{export_id}.wav"
            src = alt if alt.exists() else src
        if not src.exists():
            n_missing += 1
            print(f"missing: {src}")
            continue

        parts = r["stimulus_id"].split("|")
        masker_type = parts[1].split("_")[0]          # e.g. bird_00071.wav -> bird
        smr = float(parts[2])
        if masker_type not in MASKER_WORDS:
            print(f"unknown masker type {masker_type!r} in {r['stimulus_id']}, skipped")
            continue
        caption = make_caption(MASKER_WORDS[masker_type], smr < 0, float(r["ISOPleasant"]))

        data, sr = sf.read(src, always_2d=True)
        step = int(round(args.chunk_seconds * sr))
        for i in range(len(data) // step):
            name = f"{export_id}_c{i}.wav"
            sf.write(out / name, data[i * step:(i + 1) * step], sr, subtype="PCM_16")
            rows_out.append({"file": name, "caption": caption})
        n_files += 1

    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "caption"])
        w.writeheader()
        w.writerows(rows_out)

    adapter = Path(__file__).resolve().parent / "sa3_finetuning_data" / "metadata_adapter.py"
    config = {
        "dataset_type": "audio_dir",
        "random_crop": False,
        "datasets": [{
            "id": "restorative_soundscapes",
            "path": str(out.resolve()),
            "recursive": False,
            "extensions": [".wav"],
            "custom_metadata_module": str(adapter),
        }],
    }
    (out / "dataset_config.json").write_text(json.dumps(config, indent=2))

    print(f"{n_files} source files -> {len(rows_out)} chunks in {out} ({n_missing} missing)")


if __name__ == "__main__":
    main()
