#!/usr/bin/env python3
"""Build one training set from an existing ISD chunk folder plus a stratified ARAUS selection.

Why: in the ISD-only data the pleasantness number is confounded with the sound source
(natural sounds are mostly pleasant, traffic mostly unpleasant). ARAUS contains the same
base park recordings with different maskers at different levels, rated by people, so the
number varies within a source. Adding ARAUS chunks supplies (some) counterexamples.

Inputs
  --isd-dir     folder written by isd/03_make_training_chunks.py (metadata.csv + wavs/symlinks).
                Everything in it is linked into the joint folder unchanged (repeats included).
  --metadata    ARAUS export_metadata.csv (needs id|export_id, stimulus_id, ISOPleasant)
  --audio-dir   ARAUS exported audio (<id>.wav)
Output (--out)
  araus_<id>_c<i>.wav, symlinks to the ISD files, metadata.csv (file,caption),
  dataset_config.json, chunk_sources.csv (file,source,bin,masker,foreground) for later analysis.

ARAUS selection: --araus-files files in total, split equally over the five pleasantness
bins (limited by availability), and inside each bin spread round-robin over the
masker x (foreground/background) cells, so that every masker type appears at every
pleasantness level as far as the data allow.

Captions (--araus-caption)
  harmonised (default)  same vocabulary as ISD, e.g. "soundscape with natural sounds and traffic
                        noise [ISOPleasant: 0.12]". The dataset cannot be identified from the
                        caption, so counterexamples apply to the ISD phrases too.
  native                the original ARAUS wording ("park soundscape with traffic noise in the
                        background [...]"). The word "park" then marks the dataset.
Harmonised mapping (ASSUMPTION, see HARMONISE below): the base recordings are park soundscapes,
so the park itself counts as "natural sounds"; a masker adds its own ISD source word; a loud
masker (SMR < 0, "foreground") is named first. The mapping bird/water/wind -> natural sounds and
construction -> other noise is my reading, not something verified against ISD ratings.

    python joint/make_joint_dataset.py --isd-dir isd/train/isd_all_balanced \
        --metadata data/generation/export_metadata.csv --audio-dir data/generation/audio \
        --out joint/train/joint_isd_araus --araus-files 400
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsd.captions import MASKER_WORDS, format_pleasantness, make_caption  # noqa: E402

BINS = ["very_unpleasant", "unpleasant", "neutral", "pleasant", "very_pleasant"]


def pleasantness_bin(p):   # same thresholds as isd/02_select_balanced.py
    return ("very_unpleasant" if p <= -0.5 else "unpleasant" if p <= -0.15 else
            "neutral" if p <= 0.15 else "pleasant" if p <= 0.5 else "very_pleasant")


# masker type -> ISD source word the masker adds to the (natural) park recording
HARMONISE = {"bird": "natural sounds", "water": "natural sounds", "wind": "natural sounds",
             "traffic": "traffic noise", "construction": "other noise"}


def harmonised_caption(masker, foreground, p):
    word = HARMONISE[masker]
    if word == "natural sounds":
        what = "natural sounds"
    else:
        what = f"{word} and natural sounds" if foreground else f"natural sounds and {word}"
    return f"soundscape with {what} [ISOPleasant: {format_pleasantness(p)}]"


def link(dst: Path, target: Path):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(target)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isd-dir", required=True)
    ap.add_argument("--metadata", default="data/generation/export_metadata.csv")
    ap.add_argument("--audio-dir", default="data/generation/audio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--araus-files", type=int, default=400)
    ap.add_argument("--araus-caption", choices=["harmonised", "native"], default="harmonised")
    ap.add_argument("--chunk-seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    # ---- ARAUS candidates
    cands = []
    with open(a.metadata, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                p = float(r["ISOPleasant"])
                masker = r["stimulus_id"].split("|")[1].split("_")[0]
                smr = float(r["stimulus_id"].split("|")[2])
            except (KeyError, ValueError, IndexError):
                continue
            if masker not in MASKER_WORDS or p != p:
                continue
            cands.append({"id": r.get("export_id") or r.get("id"), "wav_path": r.get("wav_path") or "",
                          "p": p, "bin": pleasantness_bin(p), "masker": masker, "fg": smr < 0})
    print(f"ARAUS candidates: {len(cands)}")
    if cands:
        print("example row:", {k: cands[0][k] for k in ("id", "wav_path")})

    by_bin = defaultdict(lambda: defaultdict(list))   # bin -> (masker, fg) -> rows
    for c in cands:
        by_bin[c["bin"]][(c["masker"], c["fg"])].append(c)
    per_bin = a.araus_files // len(BINS)
    chosen = []
    for b in BINS:
        cells = {k: rng.sample(v, len(v)) for k, v in by_bin[b].items()}
        picked = []
        while len(picked) < per_bin and any(cells.values()):
            for k in sorted(cells):
                if cells[k] and len(picked) < per_bin:
                    picked.append(cells[k].pop())
        chosen += picked
        print(f"  {b:16s} requested {per_bin}, got {len(picked)}")

    # ---- output folder
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_out, src_rows = [], []

    isd_dir = Path(a.isd_dir).resolve()
    with open(isd_dir / "metadata.csv", encoding="utf-8") as f:
        isd_rows = list(csv.DictReader(f))
    for r in isd_rows:
        src = isd_dir / r["file"]
        link(out / r["file"], src.resolve())
        p = float(r["caption"].rsplit(":", 1)[1].strip(" ]"))
        rows_out.append({"file": r["file"], "caption": r["caption"]})
        src_rows.append({"file": r["file"], "source": "isd", "bin": pleasantness_bin(p), "masker": "", "foreground": ""})
    print(f"ISD chunk files linked: {len(isd_rows)}")

    n_chunks, n_missing = 0, 0
    for c in chosen:
        tried = [Path(c["wav_path"])] if c["wav_path"] else []
        tried.append(Path(a.audio_dir) / f"{c['id']}.wav")
        digits = "".join(ch for ch in str(c["id"]) if ch.isdigit())
        if digits:   # exported files are named araus_<7 digits>.wav
            tried.append(Path(a.audio_dir) / f"araus_{int(digits):07d}.wav")
        src = next((t for t in tried if t.exists()), None)
        if src is None:
            n_missing += 1
            if n_missing <= 3:
                print("missing, tried:", [str(t) for t in tried])
            continue
        if a.araus_caption == "native":
            caption = make_caption(MASKER_WORDS[c["masker"]], c["fg"], c["p"])
        else:
            caption = harmonised_caption(c["masker"], c["fg"], c["p"])
        data, sr = sf.read(src, always_2d=True)
        if sr != 44100:
            print(f"warning: {src.name} has sample rate {sr}")
        step = int(round(a.chunk_seconds * sr))
        for i in range(len(data) // step):
            name = f"araus_{c['id']}_c{i}.wav"
            sf.write(out / name, data[i * step:(i + 1) * step], sr, subtype="PCM_16")
            rows_out.append({"file": name, "caption": caption})
            src_rows.append({"file": name, "source": "araus", "bin": c["bin"], "masker": c["masker"],
                             "foreground": int(c["fg"])})
            n_chunks += 1
    if chosen and n_missing == len(chosen):
        sys.exit("no ARAUS audio found: check --audio-dir and the file naming (see 'missing, tried' above)")
    print(f"ARAUS: {len(chosen) - n_missing} files -> {n_chunks} chunks ({n_missing} missing audio)")

    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "caption"])
        w.writeheader()
        w.writerows(rows_out)
    with open(out / "chunk_sources.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "source", "bin", "masker", "foreground"])
        w.writeheader()
        w.writerows(src_rows)
    adapter = (Path(__file__).resolve().parents[1] / "generation" / "train"
               / "sa3_finetuning_data" / "metadata_adapter.py")
    (out / "dataset_config.json").write_text(json.dumps({
        "dataset_type": "audio_dir", "random_crop": False,
        "datasets": [{"id": "isd_araus", "path": str(out.resolve()), "recursive": False,
                      "extensions": [".wav"], "custom_metadata_module": str(adapter)}]}, indent=2))

    # ---- summary: chunk files (incl. ISD repeats) per bin and source
    cnt = defaultdict(int)
    for s in src_rows:
        cnt[(s["bin"], s["source"])] += 1
    print(f"\n{'bin':16s}{'isd':>8s}{'araus':>8s}{'araus share':>13s}")
    for b in BINS:
        i, ar = cnt[(b, "isd")], cnt[(b, "araus")]
        print(f"{b:16s}{i:8d}{ar:8d}{ar / max(1, i + ar):13.0%}")
    print(f"\nARAUS chunks by bin x masker (unique):")
    tab = defaultdict(int)
    for s in src_rows:
        if s["source"] == "araus":
            tab[(s["bin"], s["masker"])] += 1
    maskers = sorted(MASKER_WORDS)
    print(f"{'bin':16s}" + "".join(f"{m:>13s}" for m in maskers))
    for b in BINS:
        print(f"{b:16s}" + "".join(f"{tab[(b, m)]:13d}" for m in maskers))
    print(f"\ntotal chunk files: {len(rows_out)} ({len(rows_out) * a.chunk_seconds / 3600:.2f} h seen per epoch)")


if __name__ == "__main__":
    main()
