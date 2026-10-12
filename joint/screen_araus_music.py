#!/usr/bin/env python3
"""Rank ARAUS stimuli by how music-like they sound (CLAP zero-shot), to find base soundscapes to drop.

ARAUS stimuli are  <base soundscape>|<masker file>|<SMR>.  Music in the background most likely belongs
to the base soundscape, so the scores are aggregated per base (median over its mixes). CLAP zero-shot is
a ranking aid, not a detector: listen to the top of the list AND to a random sample of the low scorers
to see how much it misses before you trust a threshold.

For every file, per 10 s window: softmax (temperature 100) of the cosine similarities between the audio
embedding and a prompt set containing music prompts and non-music contrast prompts; music_p = summed
probability of the music prompts. File score = mean (music_mean) and max (music_max) over windows.

    python joint/screen_araus_music.py --metadata data/generation/export_metadata.csv \
        --audio-dir data/generation/audio --out joint/araus_music_scores.csv
Run in the project .venv (needs transformers, torch). Writes <out>, <out>_bases.csv and prints the
bases with the highest scores.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsd.audio import TARGET_SR, read, resample, to_mono  # noqa: E402

MUSIC = ["music playing", "a person singing", "a musical instrument being played",
         "a radio playing music", "a street musician playing guitar", "electronic music"]
OTHER = ["birds singing", "wind in the trees", "flowing water", "road traffic noise",
         "people talking", "footsteps", "construction work", "quiet ambience of a park", "a siren"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", default="data/generation/export_metadata.csv")
    ap.add_argument("--audio-dir", default="data/generation/audio")
    ap.add_argument("--out", default="joint/araus_music_scores.csv")
    ap.add_argument("--model", default="laion/clap-htsat-unfused")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    from transformers import ClapModel, ClapProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClapModel.from_pretrained(a.model).to(dev).eval()
    proc = ClapProcessor.from_pretrained(a.model)
    prompts = [f"the sound of {p}" for p in MUSIC + OTHER]
    with torch.no_grad():
        t = proc.tokenizer(prompts, padding=True, return_tensors="pt").to(dev)
        o = model.get_text_features(**t)
        txt = (o if torch.is_tensor(o) else o.pooler_output)
        txt = torch.nn.functional.normalize(txt.float(), dim=-1)
    n_music = len(MUSIC)

    rows = list(csv.DictReader(open(a.metadata, encoding="utf-8")))
    if a.limit:
        rows = rows[: a.limit]
    out_rows = []
    for i, r in enumerate(rows):
        try:
            parts = r["stimulus_id"].split("|")
            base, masker, smr = parts[0], parts[1].split("_")[0], parts[2]
        except (KeyError, IndexError):
            continue
        rid = r.get("export_id") or r.get("id")
        path = Path(r["wav_path"]) if r.get("wav_path") else Path(a.audio_dir) / f"araus_{int(rid):07d}.wav"
        if not path.exists():
            continue
        y, sr = read(path)
        y = resample(to_mono(y), sr, TARGET_SR)
        step = 10 * TARGET_SR
        wins = [y[k * step:(k + 1) * step] for k in range(len(y) // step)] or [y]
        with torch.no_grad():
            inp = proc.feature_extractor(list(wins), sampling_rate=TARGET_SR, return_tensors="pt")
            inp = {k: v.to(dev) for k, v in inp.items()}
            o = model.get_audio_features(**inp)
            emb = torch.nn.functional.normalize((o if torch.is_tensor(o) else o.pooler_output).float(), dim=-1)
            p = torch.softmax(100.0 * emb @ txt.T, dim=-1)[:, :n_music].sum(dim=-1).cpu().numpy()
        out_rows.append({"id": rid, "base": base, "masker": masker, "smr": smr, "path": str(path),
                         "music_mean": float(p.mean()), "music_max": float(p.max())})
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    by_base = {}
    for r in out_rows:
        by_base.setdefault(r["base"], []).append(r)
    base_rows = []
    for b, rs in by_base.items():
        m = np.array([x["music_mean"] for x in rs])
        base_rows.append({"base": b, "n_mixes": len(rs), "music_median": float(np.median(m)),
                          "music_p90": float(np.percentile(m, 90))})
    base_rows.sort(key=lambda x: -x["music_median"])
    with open(out.with_name(out.stem + "_bases.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(base_rows[0]))
        w.writeheader()
        w.writerows(base_rows)
    print(f"\n{len(out_rows)} files, {len(base_rows)} base soundscapes. Highest music_median:")
    for r in base_rows[:25]:
        print(f"  {r['base']:40s} n={r['n_mixes']:3d} median={r['music_median']:.3f} p90={r['music_p90']:.3f}")
    print("music_median quantiles (10/50/90/99%):", np.round(np.percentile(
        [r["music_median"] for r in base_rows], [10, 50, 90, 99]), 3))


if __name__ == "__main__":
    main()
