#!/usr/bin/env python3
"""Build the Clotho training sets for Stable Audio Open: with the rating in the caption, and the control without it.

Input is clotho/rated.csv (06_rate_soundaqnet.py: CLAP-checked clips with the best caption, the SoundAQnet rating
and its stability). Clips are filtered by rating stability, conditioned like the ISD and ARAUS
training audio (stereo, 44.1 kHz, 30 Hz high-pass, -23 LUFS; rsd/conditioning.py) and cut into 10 s chunks
(a clip up to --tolerance s short of whole chunks still gets that many chunks, the last one zero-padded).

Caption:   "<best Clotho caption without final period> [ISOPleasant: x.xx]"   (rsd/captions.py format)
Outputs under --out:
    rated/      chunks, metadata.csv (file, caption with rating), dataset_config.json
    norating/   the same chunks as symlinks, metadata.csv (caption without the rating), dataset_config.json
    clips.csv   one row per kept clip: rating, stabilities, soundscape_like, chunks, split (train/holdout)
    holdout.csv clips kept out of training for later generation checks (--holdout)
The two training runs differ only in the caption text. Filters (all optional, printed with their effect):
    --max-sd-ckpt   sd of the rating between the four checkpoints
    --max-sd-lvl    sd of the rating between the assumed levels
    --max-sd        sd over all 12 predictions
    --no-music / --no-speech   drop clips whose Music / Speech event probability exceeds --event-thr (the rater is
                    trained on park scenes with added maskers; music and speech clips are far from that)
    --soundscape-only   keep only clips with a soundscape tag and >= 2 keyword captions

    python clotho/08_build_training_set.py --rated clotho/rated.csv --audio-dir data/raw/clotho/audio_selected \
        --out clotho/train [--max-sd-ckpt 0.113 --max-sd-lvl 0.142] [--dry-run]
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsd.conditioning import condition_for_training  # noqa: E402

SR = 44100


def fmt(p):
    s = f"{p:.2f}"
    return "0.00" if s == "-0.00" else s


def summary(d, label):
    r = d.isop_mean
    print(f"{label:<28} n={len(d):>5} {d.dur.sum() / 3600:5.1f} h  rating mean {r.mean():+.2f} sd {r.std():.2f} "
          f"range {r.min():+.2f}..{r.max():+.2f}  <=-0.15 {100 * (r <= -.15).mean():3.0f}%  "
          f"middle {100 * ((r > -.15) & (r < .15)).mean():3.0f}%  >=0.15 {100 * (r >= .15).mean():3.0f}%  "
          f"music>.5 {100 * (d.ev_Music > .5).mean():3.0f}%  speech>.5 {100 * (d.ev_Speech > .5).mean():3.0f}%")


def write_config(folder: Path, adapter: Path, dsid: str):
    cfg = {"dataset_type": "audio_dir", "random_crop": False,
           "datasets": [{"id": dsid, "path": str(folder.resolve()), "recursive": False, "extensions": [".wav"],
                         "custom_metadata_module": str(adapter)}]}
    (folder / "dataset_config.json").write_text(json.dumps(cfg, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rated", default="clotho/rated.csv")
    ap.add_argument("--audio-dir", default="data/raw/clotho/audio_selected")
    ap.add_argument("--out", default="clotho/train")
    ap.add_argument("--max-sd", type=float, default=None)
    ap.add_argument("--max-sd-ckpt", type=float, default=None)
    ap.add_argument("--max-sd-lvl", type=float, default=None)
    ap.add_argument("--no-music", action="store_true")
    ap.add_argument("--no-speech", action="store_true")
    ap.add_argument("--event-thr", type=float, default=0.5)
    ap.add_argument("--soundscape-only", action="store_true")
    ap.add_argument("--holdout", type=int, default=40, help="clips kept out of training")
    ap.add_argument("--chunk-seconds", type=float, default=10.0)
    ap.add_argument("--tolerance", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="only print what the filters keep")
    a = ap.parse_args()

    d = pd.read_csv(a.rated)
    if "dur" not in d:
        d["dur"] = d["sec"]
    d["soundscape_like"] = d.scape_tag.astype(bool) & (d.n_any >= 2) if "scape_tag" in d else False
    summary(d, "all rated")
    for col, thr, name in [("isop_sd", a.max_sd, "total sd"), ("isop_sd_ckpt", a.max_sd_ckpt, "checkpoint sd"),
                           ("isop_sd_lvl", a.max_sd_lvl, "level sd")]:
        if thr is not None:
            d = d[d[col] <= thr]
            summary(d, f"+ {name} <= {thr}")
    if a.no_music:
        d = d[d.ev_Music <= a.event_thr]
        summary(d, f"+ music <= {a.event_thr}")
    if a.no_speech:
        d = d[d.ev_Speech <= a.event_thr]
        summary(d, f"+ speech <= {a.event_thr}")
    if a.soundscape_only:
        d = d[d.soundscape_like]
        summary(d, "+ soundscape-like only")
    d = d.reset_index(drop=True)
    print(f"soundscape-like among kept: {100 * d.soundscape_like.mean():.0f}%")
    if a.dry_run or len(d) == 0:
        return
    hold = set(d.sample(min(a.holdout, len(d) // 5), random_state=a.seed).index) if a.holdout else set()

    out = Path(a.out)
    rated, ctrl = out / "rated", out / "norating"
    rated.mkdir(parents=True, exist_ok=True)
    ctrl.mkdir(parents=True, exist_ok=True)
    step, tol = int(round(a.chunk_seconds * SR)), int(round(a.tolerance * SR))
    meta_r, meta_c, clips, nfail = [], [], [], 0
    for i, r in d.iterrows():
        split = "holdout" if i in hold else "train"
        row = {"clip": f"cl{i:04d}", "file_name": r.file_name, "caption": r.caption, "isop_mean": round(r.isop_mean, 2),
               "isop_sd": r.isop_sd, "isop_sd_ckpt": r.isop_sd_ckpt, "isop_sd_lvl": r.isop_sd_lvl,
               "soundscape_like": bool(r.soundscape_like), "split": split, "chunks": 0}
        if split == "train":
            try:
                y, sr = sf.read(Path(a.audio_dir) / r.file_name, always_2d=True)
                y = condition_for_training(y, sr, target_sr=SR)
            except Exception as e:  # noqa: BLE001
                nfail += 1
                print(f"skipped {r.file_name}: {e}")
                continue
            base = str(r.caption).strip().rstrip(" .")
            n = (len(y) + tol) // step
            for k in range(n):
                chunk = y[k * step:(k + 1) * step]
                if len(chunk) < step:
                    chunk = np.pad(chunk, ((0, step - len(chunk)), (0, 0)))
                name = f"{row['clip']}_c{k}.wav"
                sf.write(rated / name, chunk, SR, subtype="PCM_16")
                dst = ctrl / name
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                os.symlink((rated / name).resolve(), dst)
                meta_r.append({"file": name, "caption": f"{base} [ISOPleasant: {fmt(r.isop_mean)}]"})
                meta_c.append({"file": name, "caption": base})
            row["chunks"] = n
        clips.append(row)

    for folder, meta in [(rated, meta_r), (ctrl, meta_c)]:
        with open(folder / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "caption"])
            w.writeheader()
            w.writerows(meta)
    adapter = Path(__file__).resolve().parents[1] / "generation" / "train" / "sa3_finetuning_data" / "metadata_adapter.py"
    write_config(rated, adapter, "clotho_rated")
    write_config(ctrl, adapter, "clotho_norating")
    pd.DataFrame(clips).to_csv(out / "clips.csv", index=False)
    pd.DataFrame([c for c in clips if c["split"] == "holdout"]).to_csv(out / "holdout.csv", index=False)
    if not meta_r:
        sys.exit("no chunks written; check --audio-dir")
    print(f"\n{sum(c['chunks'] > 0 for c in clips)} training clips -> {len(meta_r)} chunks "
          f"({len(meta_r) * a.chunk_seconds / 3600:.1f} h), {len(hold)} held out, {nfail} failed")
    print("example:", meta_r[0]["caption"], "|", meta_c[0]["caption"])
    print(f"train with:  DATASET_CONFIG={rated}/dataset_config.json   (and {ctrl}/dataset_config.json for the control)")


if __name__ == "__main__":
    main()
