#!/usr/bin/env python3
"""Turn the selected ARAUS stimuli into training files and captions (same layout as the ISD builder).

Input : selected CSV from 01_select_balanced_dataset.py (export_id, stimulus_id, ISOPleasant; wav_path
        optional) and the exported audio (data/generation/audio/araus_<id>.wav, 30 s each).
Output: <out>/araus_<export_id>_c<i>.wav, <out>/metadata.csv (file,caption),
        <out>/dataset_config.json (absolute paths for THIS machine), <out>/chunk_info.csv
        (file, bin, masker, foreground, ISOPleasant) for later analysis.

Length (like isd/03_make_training_chunks.py):
  --whole-max 47   keep every stimulus whole (a 30 s stimulus becomes one 30 s file; the rating belongs
                   to the whole stimulus); only stimuli longer than 47 s would be split into equal pieces.
                   stable-audio-tools pads each file to the model's sample size and passes the real length
                   as seconds_total.
  (default, --whole-max 0)  fixed chunks of --chunk-seconds (10 s), as in the earlier ARAUS runs.
--bin-repeat very_unpleasant=2,...   see each file of that bin this many times per epoch (repeats are
                   symlinks listed in metadata.csv), as in the ISD builder.

Captions (--caption):
  native      (default) "park soundscape with <masker> in the foreground|background [ISOPleasant: x.xx]"
  harmonised  ISD vocabulary, e.g. "soundscape with natural sounds and traffic noise [ISOPleasant: x.xx]"
              (assumption: the park counts as natural sounds; bird/water/wind add nothing;
              construction -> other noise; a masker louder than the soundscape is named first).
stimulus_id has the form  <soundscape>|<masker file>|<smr>.

    python generation/train/02_make_training_chunks.py --selected generation/train/selected_araus_800.csv \
        --audio-dir data/generation/audio --out generation/train/araus_whole --whole-max 47
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rsd.captions import MASKER_WORDS, format_pleasantness, make_caption  # noqa: E402

BINS = ["very_unpleasant", "unpleasant", "neutral", "pleasant", "very_pleasant"]
HARMONISE = {"bird": "natural sounds", "water": "natural sounds", "wind": "natural sounds",
             "traffic": "traffic noise", "construction": "other noise"}


def pleasantness_bin(p):
    return ("very_unpleasant" if p <= -0.5 else "unpleasant" if p <= -0.15 else
            "neutral" if p <= 0.15 else "pleasant" if p <= 0.5 else "very_pleasant")


def harmonised_caption(masker, foreground, p):
    word = HARMONISE[masker]
    if word == "natural sounds":
        what = word
    else:
        what = f"{word} and natural sounds" if foreground else f"natural sounds and {word}"
    return f"soundscape with {what} [ISOPleasant: {format_pleasantness(p)}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected", required=True)
    ap.add_argument("--audio-dir", default="data/generation/audio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-seconds", type=float, default=10.0)
    ap.add_argument("--whole-max", type=float, default=0.0)
    ap.add_argument("--min-seconds", type=float, default=10.0, help="whole mode: drop stimuli shorter than this")
    ap.add_argument("--caption", choices=["native", "harmonised"], default="native")
    ap.add_argument("--bin-repeat", default="")
    a = ap.parse_args()
    bin_repeat = {k.strip(): int(v) for k, v in (kv.split("=") for kv in a.bin_repeat.split(",") if kv.strip())}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_out, info_rows, n_files, n_missing = [], [], 0, 0
    eff = defaultdict(lambda: [0, 0, 0, 0.0, 0.0])   # bin -> stimuli, files, files incl. repeats, unique s, seen s

    with open(a.selected, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        export_id = r.get("export_id") or r.get("id")
        src = Path(r["wav_path"]) if r.get("wav_path") else Path(a.audio_dir) / f"araus_{int(export_id):07d}.wav"
        if not src.exists():
            alt = Path(a.audio_dir) / f"araus_{int(export_id):07d}.wav"
            src = alt if alt.exists() else src
        if not src.exists():
            n_missing += 1
            print(f"missing: {src}")
            continue

        parts = r["stimulus_id"].split("|")
        masker = parts[1].split("_")[0]          # e.g. bird_00071.wav -> bird
        fg = float(parts[2]) < 0
        p = float(r["ISOPleasant"])
        if masker not in MASKER_WORDS:
            print(f"unknown masker type {masker!r} in {r['stimulus_id']}, skipped")
            continue
        caption = (make_caption(MASKER_WORDS[masker], fg, p) if a.caption == "native"
                   else harmonised_caption(masker, fg, p))
        b = pleasantness_bin(p)
        reps = bin_repeat.get(b, 1)

        data, sr = sf.read(src, always_2d=True)
        if a.whole_max:
            if len(data) < a.min_seconds * sr:
                continue
            n = int(np.ceil(len(data) / (a.whole_max * sr)))
            bounds = [int(round(k * len(data) / n)) for k in range(n + 1)]
            pieces = [data[bounds[k]:bounds[k + 1]] for k in range(n)]
        else:
            step = int(round(a.chunk_seconds * sr))
            pieces = [data[i * step:(i + 1) * step] for i in range(len(data) // step)]
        for i, piece in enumerate(pieces):
            name = f"araus_{export_id}_c{i}.wav"
            sf.write(out / name, piece, sr, subtype="PCM_16")
            rows_out.append({"file": name, "caption": caption})
            info_rows.append({"file": name, "bin": b, "masker": masker, "foreground": int(fg), "ISOPleasant": p})
            for k in range(1, reps):
                link = out / f"araus_{export_id}_c{i}_r{k}.wav"
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(name)
                rows_out.append({"file": link.name, "caption": caption})
            e = eff[b]
            e[1] += 1
            e[2] += reps
            e[3] += len(piece) / sr
            e[4] += reps * len(piece) / sr
        eff[b][0] += 1
        n_files += 1

    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "caption"])
        w.writeheader()
        w.writerows(rows_out)
    with open(out / "chunk_info.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "bin", "masker", "foreground", "ISOPleasant"])
        w.writeheader()
        w.writerows(info_rows)

    adapter = Path(__file__).resolve().parent / "sa3_finetuning_data" / "metadata_adapter.py"
    config = {
        "dataset_type": "audio_dir",
        "random_crop": False,
        "datasets": [{"id": "restorative_soundscapes", "path": str(out.resolve()), "recursive": False,
                      "extensions": [".wav"], "custom_metadata_module": str(adapter)}],
    }
    (out / "dataset_config.json").write_text(json.dumps(config, indent=2))

    print(f"{n_files} stimuli -> {len(rows_out)} files in {out} ({n_missing} missing)")
    print(f"{'bin':16s}{'stimuli':>8s}{'files':>7s}{'incl. repeats':>14s}{'hours seen':>11s}{'unique h':>10s}")
    for b in BINS:
        s, nf, nt, us, ss = eff[b]
        print(f"{b:16s}{s:8d}{nf:7d}{nt:14d}{ss / 3600:11.2f}{us / 3600:10.2f}")
    print(f"unique audio: {sum(v[3] for v in eff.values()) / 3600:.2f} h in {sum(v[1] for v in eff.values())} "
          f"files; files incl. repeats: {sum(v[2] for v in eff.values())}")


if __name__ == "__main__":
    main()
