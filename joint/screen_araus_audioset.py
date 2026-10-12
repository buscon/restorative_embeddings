#!/usr/bin/env python3
"""Screen ARAUS mixes for music (and other unwanted content) with an AudioSet tagger.

Why: the ARAUS training chunks contain music-like background and out-of-context birds, and the
earlier ARAUS fine-tunes reproduced them. This script scores every 10 s chunk (the unit the
model is trained on) with the AudioSet classifier MIT/ast-finetuned-audioset-10-10-0.4593
(527 classes, sigmoid outputs, so several classes can be high at once).

Per chunk it writes: the highest probability among music-related classes ("music"), among
bird classes ("bird"), among speech classes ("speech"), and the top-3 labels.
Per base soundscape (first part of stimulus_id) it also reports the median over all its
chunks: music in the base recording shows up in every mix built on it.

    python joint/screen_araus_audioset.py \
        --metadata data/generation/export_metadata.csv --audio-dir data/generation/audio \
        --out joint/araus_screen

Writes <out>/chunk_scores.csv, <out>/base_scores.csv, <out>/summary.txt and
<out>/listen_list.txt (chunks across the music-score range, for calibrating a threshold by ear).
Use --limit N for a quick test. The first run downloads the model from Hugging Face.
"""
import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import ASTFeatureExtractor, ASTForAudioClassification

MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR_MODEL = 16000

# exact AudioSet display names; missing names are reported at start-up, not silently ignored
MUSIC = ["Music", "Musical instrument", "Singing", "Song", "Plucked string instrument", "Guitar",
         "Electric guitar", "Bass guitar", "Acoustic guitar", "Piano", "Keyboard (musical)", "Synthesizer",
         "Organ", "Drum", "Drum kit", "Percussion", "Bowed string instrument", "Violin, fiddle", "Cello",
         "Wind instrument, woodwind instrument", "Flute", "Brass instrument", "Pop music", "Rock music",
         "Electronic music", "Ambient music", "New-age music", "Background music", "Soundtrack music",
         "Theme music", "Classical music", "Choir", "Humming", "Chant", "Mantra", "Harmonica", "Bell",
         "Wind chime", "Music of Asia", "Music of Africa", "Music of Latin America"]
BIRD = ["Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet", "Crow", "Owl", "Pigeon, dove",
        "Squawk", "Coo", "Caw", "Hoot", "Songbird"]
SPEECH = ["Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
          "Conversation", "Narration, monologue", "Babbling", "Shout", "Laughter", "Crowd"]


def indices(id2label, names, tag):
    rev = {v: k for k, v in id2label.items()}
    found = [rev[n] for n in names if n in rev]
    missing = [n for n in names if n not in rev]
    print(f"{tag}: {len(found)} classes used" + (f"; not in the model's label set: {missing}" if missing else ""))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", default="data/generation/export_metadata.csv")
    ap.add_argument("--audio-dir", default="data/generation/audio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-seconds", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="only the first N files (quick test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fe = ASTFeatureExtractor.from_pretrained(MODEL)
    model = ASTForAudioClassification.from_pretrained(MODEL).to(a.device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    i_music, i_bird, i_speech = (indices(id2label, MUSIC, "music"), indices(id2label, BIRD, "bird"),
                                 indices(id2label, SPEECH, "speech"))

    rows = []
    with open(a.metadata, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                parts = r["stimulus_id"].split("|")
                eid = r.get("export_id") or r.get("id")
                rows.append({"id": eid, "base": parts[0], "masker": parts[1].split("_")[0],
                             "smr": float(parts[2]), "p": float(r["ISOPleasant"]),
                             "wav": r.get("wav_path") or ""})
            except (KeyError, ValueError, IndexError):
                continue
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(rows)} files to score")

    def path_of(r):
        for c in ([Path(r["wav"])] if r["wav"] else []) + [Path(a.audio_dir) / f"{r['id']}.wav"]:
            if c.exists():
                return c
        digits = "".join(ch for ch in str(r["id"]) if ch.isdigit())
        c = Path(a.audio_dir) / f"araus_{int(digits):07d}.wav" if digits else None
        return c if c is not None and c.exists() else None

    pending, results = [], []   # pending: (meta, 16 kHz mono chunk)

    def flush():
        if not pending:
            return
        feats = fe([x for _, x in pending], sampling_rate=SR_MODEL, return_tensors="pt")
        with torch.no_grad():
            prob = torch.sigmoid(model(**{k: v.to(a.device) for k, v in feats.items()}).logits).cpu().numpy()
        for (m, _), p in zip(pending, prob):
            top = np.argsort(p)[::-1][:3]
            results.append({**m, "music": float(p[i_music].max()), "bird": float(p[i_bird].max()),
                            "speech": float(p[i_speech].max()),
                            "top3": "; ".join(f"{id2label[int(t)]} {p[t]:.2f}" for t in top)})
        pending.clear()

    n_missing = 0
    for n, r in enumerate(rows, 1):
        src = path_of(r)
        if src is None:
            n_missing += 1
            continue
        y, sr = sf.read(src, always_2d=True)
        y = y.mean(axis=1)
        step = int(round(a.chunk_seconds * sr))
        for i in range(len(y) // step):
            seg = y[i * step:(i + 1) * step]
            seg = resample_poly(seg, SR_MODEL, sr) if sr != SR_MODEL else seg
            pending.append(({"id": r["id"], "chunk": i, "base": r["base"], "masker": r["masker"],
                             "smr": r["smr"], "p": r["p"]}, seg.astype(np.float32)))
            if len(pending) >= a.batch:
                flush()
        if n % 200 == 0:
            print(f"  {n}/{len(rows)} files")
    flush()
    print(f"scored {len(results)} chunks ({n_missing} files without audio)")
    if not results:
        sys.exit("nothing scored: check --audio-dir")

    cols = ["id", "chunk", "base", "masker", "smr", "p", "music", "bird", "speech", "top3"]
    with open(out / "chunk_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)

    by_base = defaultdict(list)
    for r in results:
        by_base[r["base"]].append(r)
    base_rows = []
    for b, rs in by_base.items():
        base_rows.append({"base": b, "n_chunks": len(rs),
                          "music_median": float(np.median([x["music"] for x in rs])),
                          "music_p90": float(np.percentile([x["music"] for x in rs], 90)),
                          "bird_median": float(np.median([x["bird"] for x in rs])),
                          "speech_median": float(np.median([x["speech"] for x in rs]))})
    base_rows.sort(key=lambda r: -r["music_median"])
    with open(out / "base_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(base_rows[0]))
        w.writeheader()
        w.writerows(base_rows)

    # ---- summary and a list to calibrate by ear
    lines = []
    m = np.array([r["music"] for r in results])
    lines.append(f"chunks: {len(results)}, base soundscapes: {len(base_rows)}")
    lines.append("music score, quantiles over chunks: " +
                 ", ".join(f"{q}%={np.percentile(m, q):.3f}" for q in (10, 25, 50, 75, 90, 95, 99)))
    for thr in (0.05, 0.1, 0.2, 0.3, 0.5):
        lines.append(f"chunks with music > {thr}: {(m > thr).sum()} ({(m > thr).mean():.0%}); "
                     f"bases with median music > {thr}: {sum(r['music_median'] > thr for r in base_rows)}")
    lines.append("\nhighest-scoring base soundscapes (median over their chunks):")
    for r in base_rows[:15]:
        lines.append(f"  {r['base']:40s} music {r['music_median']:.2f}  bird {r['bird_median']:.2f}  "
                     f"speech {r['speech_median']:.2f}  ({r['n_chunks']} chunks)")
    lines.append("\nmean score by masker type:")
    for mk in sorted({r["masker"] for r in results}):
        rs = [r for r in results if r["masker"] == mk]
        lines.append(f"  {mk:12s} music {np.mean([x['music'] for x in rs]):.3f}  "
                     f"bird {np.mean([x['bird'] for x in rs]):.3f}  speech {np.mean([x['speech'] for x in rs]):.3f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    rng = random.Random(a.seed)
    order = sorted(results, key=lambda r: r["music"])
    bands = [("lowest 25%", 0.0, 0.25), ("25-75%", 0.25, 0.75), ("75-90%", 0.75, 0.9),
             ("90-99%", 0.9, 0.99), ("top 1%", 0.99, 1.0)]
    ll = ["Listen to these to decide where 'music' starts. File names = <id>_c<chunk>.wav in the audio dir (chunk "
          "number = which 10 s part of araus_<id>.wav).\n"]
    for name, lo, hi in bands:
        sel = order[int(lo * len(order)):max(int(hi * len(order)), int(lo * len(order)) + 1)]
        for r in rng.sample(sel, min(4, len(sel))):
            ll.append(f"[{name}] music {r['music']:.2f}  araus_{int(r['id']):07d}.wav chunk {r['chunk']}  "
                      f"masker {r['masker']} smr {r['smr']:g}  top3: {r['top3']}")
    (out / "listen_list.txt").write_text("\n".join(ll) + "\n")
    print(f"\nwritten to {out}/ (chunk_scores.csv, base_scores.csv, summary.txt, listen_list.txt)")


if __name__ == "__main__":
    main()
