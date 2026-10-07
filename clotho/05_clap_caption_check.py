#!/usr/bin/env python3
"""Score each clip's five Clotho captions against its audio with CLAP; keep the best one.

For every selected clip: embed the audio (same CLAP model and 10 s windowing as the rest of
the repo: laion/clap-htsat-unfused, mono, 48 kHz, up to three 10 s windows, averaged) and the
five captions. Raw cosine similarity is hard to read, so each caption is also expressed
relative to the other captions in the selected set:

    z   = (cos(audio_i, caption_ik) - mean) / std, where mean and std are taken over all
          captions of OTHER clips for the same audio (about 5 x N - 5 captions)
    pct = share of those other captions that score lower than this caption

z > 0 means "fits this audio better than a typical caption from the set". The comparison
captions come from the same soundscape-like set, so they are a harder baseline than random
text and z is conservative. The script prints a sanity check (the mean pct of the true
captions would be 0.50 if CLAP carried no information) and how many clips survive at
different thresholds; choose --min-z from that table, not from a rule of thumb.

Decision per clip: keep the caption with the highest z; drop the clip if even that z is below
--min-z. The kept caption is written to `caption` (the other four stay in caption_1..5), and
`good_captions` lists every caption at or above --min-z.

What this does NOT check: whether the clip is a soundscape at all (a clean walkie-talkie clip
scores high). That is the job of the keyword/tag selection in 02_select_candidates.py.

    python clotho/05_clap_caption_check.py --selected clotho/selected_clotho.csv \
        --audio-dir data/raw/clotho/audio_selected --out-dir clotho/checked [--min-z 1.0]

Run it in the stable-audio-tools venv (env B) or any env with torch, transformers, soundfile,
scipy, pandas and tqdm. A GPU is not needed for a few thousand clips but helps.
Outputs in --out-dir:
    caption_scores.csv   one row per (clip, caption): sim, z, pct
    selected_checked.csv one row per kept clip: original columns + caption (best), best_idx, best_z,
                         n_good, good_captions (all captions with z >= --min-z, joined by ' || ')
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsd.audio import TARGET_SR, prepare, read, windows  # noqa: E402

MODEL = "laion/clap-htsat-unfused"  # same as rsd/embed.py in archive/
CAPS = [f"caption_{i}" for i in range(1, 6)]


def as_tensor(out):
    return out if torch.is_tensor(out) else out.pooler_output  # transformers 4.x vs 5.x


@torch.no_grad()
def embed_audio(model, fe, paths, device, batch):
    embs, ok = [], []
    for i in tqdm(range(0, len(paths), batch), desc="audio"):
        wins, idx = [], []
        for j, p in enumerate(paths[i:i + batch]):
            try:
                x, sr = read(p)
                wins.append(windows(prepare(x, sr)))  # (3, 480000)
                idx.append(i + j)
            except Exception as e:  # missing or unreadable file
                print(f"skip {p}: {e}")
        if not wins:
            continue
        flat = np.concatenate(wins)  # (3*b, samples)
        inp = fe(list(flat), sampling_rate=TARGET_SR, return_tensors="pt")
        e = as_tensor(model.get_audio_features(**{k: v.to(device) for k, v in inp.items()}))
        e = e.float().cpu().numpy().reshape(len(wins), 3, -1).mean(axis=1)  # average the windows
        embs.append(e)
        ok += idx
    e = np.concatenate(embs)
    return ok, e / np.linalg.norm(e, axis=1, keepdims=True)


@torch.no_grad()
def embed_text(model, tok, texts, device, batch=128):
    out = []
    for i in tqdm(range(0, len(texts), batch), desc="text"):
        inp = tok(texts[i:i + batch], padding=True, truncation=True, max_length=77, return_tensors="pt")
        e = as_tensor(model.get_text_features(**{k: v.to(device) for k, v in inp.items()}))
        out.append(e.float().cpu().numpy())
    e = np.concatenate(out)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected", default="clotho/selected_clotho.csv")
    ap.add_argument("--audio-dir", default="data/raw/clotho/audio_selected")
    ap.add_argument("--out-dir", default="clotho/checked")
    ap.add_argument("--min-z", type=float, default=1.0, help="drop clips whose best caption has z below this")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=8, help="clips per audio batch")
    ap.add_argument("--limit", type=int, default=None, help="first N clips only (for tests)")
    ap.add_argument("--show", type=int, default=5, help="print this many best and worst clips")
    args = ap.parse_args()

    from transformers import ClapModel, ClapProcessor
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ClapModel.from_pretrained(args.model).to(device).eval()
    proc = ClapProcessor.from_pretrained(args.model)

    df = pd.read_csv(args.selected)
    if args.limit:
        df = df.head(args.limit)
    paths = [Path(args.audio_dir) / f for f in df.file_name]
    have = [p.exists() for p in paths]
    print(f"{sum(have)} of {len(df)} audio files found in {args.audio_dir}")
    df = df[have].reset_index(drop=True)
    paths = [p for p, h in zip(paths, have) if h]

    ok, A = embed_audio(model, proc.feature_extractor, paths, device, args.batch)
    df = df.iloc[ok].reset_index(drop=True)
    n = len(df)
    caps = df[CAPS].astype(str).values  # (n, 5)
    T = embed_text(model, proc.tokenizer, caps.reshape(-1).tolist(), device)  # (5n, d)

    sims = A @ T.T  # (n, 5n); caption j belongs to clip j // 5
    own = np.repeat(np.arange(n), 5)  # clip index of each caption
    rows = []
    z = np.zeros((n, 5))
    pct = np.zeros((n, 5))
    raw = np.zeros((n, 5))
    for i in range(n):
        others = np.delete(sims[i], np.arange(5 * i, 5 * i + 5))
        mu, sd = others.mean(), others.std() + 1e-9
        s = sims[i, 5 * i:5 * i + 5]
        raw[i] = s
        z[i] = (s - mu) / sd
        pct[i] = [(others < v).mean() for v in s]
    best = z.argmax(axis=1)
    best_z = z.max(axis=1)

    scores = pd.DataFrame({
        "file_name": np.repeat(df.file_name.values, 5),
        "caption_idx": np.tile(np.arange(1, 6), n),
        "caption": caps.reshape(-1),
        "sim": raw.reshape(-1), "z": z.reshape(-1), "pct": pct.reshape(-1),
    })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "caption_scores.csv", index=False)

    print(f"\nclips scored: {n}")
    print(f"sanity: mean pct of the true captions = {pct.mean():.3f} (0.500 = CLAP carries no information)")
    print(f"z of all true captions: " + ", ".join(f"p{q}={np.percentile(z, q):.2f}" for q in (10, 25, 50, 75, 90)))
    print(f"best z per clip:        " + ", ".join(f"p{q}={np.percentile(best_z, q):.2f}" for q in (10, 25, 50, 75, 90)))
    print("\nthreshold  clips kept  hours kept  (clips with at least one caption z >= threshold)")
    for t in (-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        keep = best_z >= t
        print(f"{t:8.1f}  {int(keep.sum()):10d}  {df.dur[keep].sum() / 3600:10.1f}")

    keep = best_z >= args.min_z
    out = df[keep].copy()
    out["caption"] = [caps[i, best[i]] for i in np.where(keep)[0]]
    out["best_idx"] = best[keep] + 1
    out["best_z"] = best_z[keep]
    out["n_good"] = (z[keep] >= args.min_z).sum(axis=1)
    # every caption that passes the threshold, for training on several captions per clip
    out["good_captions"] = [" || ".join(c for c, zz in zip(caps[i], z[i]) if zz >= args.min_z)
                            for i in np.where(keep)[0]]
    out.to_csv(out_dir / "selected_checked.csv", index=False)
    print(f"\n--min-z {args.min_z}: kept {len(out)} of {n} clips ({out.dur.sum() / 3600:.1f} h); "
          f"wrote {out_dir}/selected_checked.csv")

    if args.show:
        order = np.argsort(best_z)
        for title, ids in (("WORST", order[:args.show]), ("BEST", order[::-1][:args.show])):
            print(f"\n{title} clips by best z:")
            for i in ids:
                print(f"- {df.file_name[i]}  tags={str(df.keywords[i])[:50]}  best z={best_z[i]:.2f}")
                for k in range(5):
                    print(f"    z={z[i, k]:5.2f}  {caps[i, k]}")


if __name__ == "__main__":
    main()
