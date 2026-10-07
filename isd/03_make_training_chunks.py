#!/usr/bin/env python3
"""Condition the selected ISD recordings and cut them into 10 s training chunks.

Conditioning (rsd/conditioning.py, same treatment as the ARAUS export): stereo, 44.1 kHz,
30 Hz high-pass, -23 LUFS. Note that the loudness normalisation removes the absolute
recording level, as it does for ARAUS.

Output: <out>/isd_<GroupID>_c<i>.wav, <out>/metadata.csv (file,caption) and
        <out>/dataset_config.json (absolute paths for this machine).
Caption: "soundscape at <Place Name> in <City> [ISOPleasant: x.xx]" (rsd/captions.py).
The text carries place and city only, not the sound sources, because ISD has no masker label.

A recording up to --tolerance seconds short of a whole number of chunks still gets that
many chunks (the last one zero-padded); a shorter remainder is dropped.

    python isd/03_make_training_chunks.py --selected isd/selected_isd.csv --out isd/train/isd_for_sao
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsd.captions import make_isd_caption  # noqa: E402
from rsd.conditioning import condition_for_training  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected", default="isd/selected_isd.csv")
    ap.add_argument("--out", default="isd/train/isd_for_sao")
    ap.add_argument("--chunk-seconds", type=float, default=10.0)
    ap.add_argument("--tolerance", type=float, default=0.25)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    SR = 44100
    step = int(round(a.chunk_seconds * SR))
    tol = int(round(a.tolerance * SR))
    rows_out, n_ok, n_fail = [], 0, 0

    with open(a.selected, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        try:
            y, sr = sf.read(r["wav"], always_2d=True)
            y = condition_for_training(y, sr, target_sr=SR)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"skipped {r['GroupID']}: {e}")
            continue
        caption = make_isd_caption(r["LocationID"], r["city"], float(r["ISOPleasant"]))
        for i in range((len(y) + tol) // step):
            chunk = y[i * step:(i + 1) * step]
            if len(chunk) < step:
                chunk = np.pad(chunk, ((0, step - len(chunk)), (0, 0)))
            name = f"isd_{r['GroupID']}_c{i}.wav"
            sf.write(out / name, chunk, SR, subtype="PCM_16")
            rows_out.append({"file": name, "caption": caption})
        n_ok += 1

    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "caption"])
        w.writeheader()
        w.writerows(rows_out)

    adapter = (Path(__file__).resolve().parents[1] / "generation" / "train"
               / "sa3_finetuning_data" / "metadata_adapter.py")
    config = {"dataset_type": "audio_dir", "random_crop": False,
              "datasets": [{"id": "isd_soundscapes", "path": str(out.resolve()), "recursive": False,
                            "extensions": [".wav"], "custom_metadata_module": str(adapter)}]}
    (out / "dataset_config.json").write_text(json.dumps(config, indent=2))
    print(f"{n_ok} recordings -> {len(rows_out)} chunks in {out} ({n_fail} failed)")


if __name__ == "__main__":
    main()
