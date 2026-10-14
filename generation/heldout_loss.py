#!/usr/bin/env python3
"""Does the model use the rating in "[ISOPleasant: x.xx]"? Paired loss test, no listening needed.

For every file the diffusion training loss is computed with the SAME latent, noise and timestep under
different captions:
    correct     caption as written
    flip        number replaced by minus the number (+0.70 -> -0.70)
    mirror      number replaced by the number of the file at the opposite rank (lowest <-> highest rating);
                stays inside the value range of the data, which the plain flip does not for skewed data
    shuffle     number taken from another random file
    none        tag removed, only the description text
If the model uses the number, the correct caption gives the lower loss: (wrong - correct) > 0, on average over
files, and for most files. If the number does not matter the differences scatter around 0. Always run the same
test with the BASE model (--ckpt model.safetensors) as a control: it never saw the tag, so it should give ~0.

The loss follows the training step of stable-audio-tools (latent = pretransform.encode of the audio padded
to sample_size, objective 'v' or rectified flow, loss only on the non-padded latent frames, seconds_start 0,
seconds_total = true length). Timesteps are a fixed grid instead of random draws.

    python generation/heldout_loss.py --ckpt <model.ckpt> --config configs/model_config.json \\
        --data-dir <folder with wav + metadata.csv (file,caption)> --n-files 100 [--use-ema]

Use files the model was NOT trained on if you can (Clotho: the 40 held-out clips; joint set: ARAUS stimuli whose
base soundscape is not among the training files). On training files a positive difference can also come from
memorising which number belongs to which file, so it shows that the number is used, not that it generalises.
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from generate import load_weights  # noqa: E402

try:
    from rsd.captions import format_pleasantness  # noqa: E402
except Exception:  # pragma: no cover
    def format_pleasantness(v):
        s = f"{v:.2f}"
        return "0.00" if s == "-0.00" else s

TAG = re.compile(r"^(?P<text>.*?)\s*\[ISOPleasant:\s*(?P<val>[-+]?\d+(?:\.\d+)?)\]\s*$")
CONDS = ["correct", "flip", "mirror", "shuffle", "none"]


def load_audio(path, sr, size):
    wav, osr = torchaudio.load(str(path))
    if osr != sr:
        wav = torchaudio.functional.resample(wav, osr, sr)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    wav = wav[:2]
    n = min(wav.shape[1], size)
    out = torch.zeros(2, size)
    out[:, :n] = wav[:, :n]
    return out, n / sr


def bootstrap_ci(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--files-list", default=None, help="text file, one wav name per line (default: all files with a tag)")
    ap.add_argument("--n-files", type=int, default=100)
    ap.add_argument("--t-grid", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--draws", type=int, default=4, help="noise draws per file and timestep")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--min-abs", type=float, default=0.3, help="subset with |rating| >= this is reported separately")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="no mixed precision (training used mixed precision)")
    ap.add_argument("--out", default=None, help="per-file csv (default: <data-dir>/heldout_loss.csv)")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    sr, size = cfg["sample_rate"], cfg["sample_size"]
    data = Path(a.data_dir).expanduser()
    rows = []
    for r in csv.DictReader(open(data / "metadata.csv", encoding="utf-8")):
        m = TAG.match(r["caption"])
        if m and (data / r["file"]).exists():
            rows.append({"file": r["file"], "text": m["text"], "val": float(m["val"])})
    if a.files_list:
        keep = {x.strip() for x in open(a.files_list) if x.strip()}
        rows = [r for r in rows if r["file"] in keep]
    rng = np.random.default_rng(a.seed)
    if len(rows) > a.n_files:
        rows = [rows[i] for i in sorted(rng.choice(len(rows), a.n_files, replace=False))]
    n = len(rows)
    if n < 4:
        sys.exit(f"only {n} usable files with a tag in {data}")
    vals = np.array([r["val"] for r in rows])
    order = np.argsort(vals, kind="stable")
    mirror = np.empty(n)
    mirror[order] = vals[order][::-1]
    shuffle = vals[rng.permutation(n)]
    for i in range(n):                         # make sure the shuffled number differs
        if abs(shuffle[i] - vals[i]) < 0.05:
            j = (i + n // 2) % n
            shuffle[i], shuffle[j] = shuffle[j], shuffle[i]
    caps = {c: [] for c in CONDS}
    for i, r in enumerate(rows):
        f = format_pleasantness
        caps["correct"].append(f"{r['text']} [ISOPleasant: {f(r['val'])}]")
        caps["flip"].append(f"{r['text']} [ISOPleasant: {f(-r['val'])}]")
        caps["mirror"].append(f"{r['text']} [ISOPleasant: {f(mirror[i])}]")
        caps["shuffle"].append(f"{r['text']} [ISOPleasant: {f(shuffle[i])}]")
        caps["none"].append(r["text"])

    from stable_audio_tools.models import create_model_from_config
    model = create_model_from_config(cfg)
    if not load_weights(model, Path(a.ckpt).expanduser(), use_ema=a.use_ema):
        sys.exit(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval().requires_grad_(False)
    objective = getattr(model, "diffusion_objective", "v")
    ds = model.pretransform.downsampling_ratio
    amp = (device == "cuda") and not a.fp32
    print(f"{n} files, objective {objective}, t grid {a.t_grid}, {a.draws} draws, autocast {amp}", flush=True)

    # latents, true lengths
    Z, secs, nvalid = [], [], []
    with torch.no_grad():
        for r in rows:
            audio, s = load_audio(data / r["file"], sr, size)
            with torch.autocast(device, dtype=torch.float16, enabled=amp):
                z = model.pretransform.encode(audio[None].to(device))
            Z.append(z[0].float().cpu())
            secs.append(s)
            nvalid.append(min(z.shape[-1], int(math.ceil(s * sr / ds))))
    T, K = len(a.t_grid), a.draws
    L = {c: np.zeros((n, T, K)) for c in CONDS}
    with torch.no_grad():
        for b0 in range(0, n, a.batch):
            idx = list(range(b0, min(n, b0 + a.batch)))
            z = torch.stack([Z[i] for i in idx]).to(device)
            md = {c: [{"prompt": caps[c][i], "seconds_start": 0.0, "seconds_total": secs[i]} for i in idx] for c in CONDS}
            cond = {c: model.conditioner(md[c], device) for c in CONDS}
            for ti, t in enumerate(a.t_grid):
                tt = torch.full((len(idx),), float(t), device=device)
                if objective == "v":
                    al, sg = math.cos(t * math.pi / 2), math.sin(t * math.pi / 2)
                else:
                    al, sg = 1 - t, t
                for k in range(K):
                    g = torch.Generator(device="cpu")
                    g.manual_seed(a.seed * 1000003 + b0 * 101 + ti * 11 + k)
                    noise = torch.randn(z.shape, generator=g).to(device)
                    x = z * al + noise * sg
                    target = noise * al - z * sg if objective == "v" else noise - z
                    for c in CONDS:
                        with torch.autocast(device, dtype=torch.float16, enabled=amp):
                            out = model(x, tt, cond=cond[c], cfg_dropout_prob=0.0)
                        err = (out.float() - target) ** 2
                        for j, i in enumerate(idx):
                            L[c][i, ti, k] = err[j, :, : nvalid[i]].mean().item()
            print(f"  {min(n, b0 + a.batch)}/{n} files", flush=True)

    per_file = {c: L[c].mean(axis=(1, 2)) for c in CONDS}
    absv = np.abs(vals)
    sub = absv >= a.min_abs
    print(f"\nmean loss over files (correct caption): {per_file['correct'].mean():.4f}")
    print(f"values in the test files: min {vals.min():+.2f}, median {np.median(vals):+.2f}, max {vals.max():+.2f}; "
          f"{int(sub.sum())} files with |rating| >= {a.min_abs}\n")
    print("difference = loss(changed caption) - loss(correct caption); > 0 means the correct number is better")
    print(f"{'caption':8s} {'subset':>14s} {'mean diff':>11s} {'95% CI':>22s} {'files with diff>0':>18s}")
    for c in CONDS[1:]:
        for name, m in (("all files", np.ones(n, bool)), (f"|x|>={a.min_abs}", sub)):
            if m.sum() < 3:
                continue
            d = per_file[c][m] - per_file["correct"][m]
            lo, hi = bootstrap_ci(d)
            print(f"{c:8s} {name:>14s} {d.mean():+11.5f} [{lo:+9.5f},{hi:+9.5f}] {100 * (d > 0).mean():16.0f}%")
    print("\nby timestep t (all files), mean difference for flip / mirror / shuffle / none:")
    for ti, t in enumerate(a.t_grid):
        row = [(L[c][:, ti].mean(axis=1) - L["correct"][:, ti].mean(axis=1)).mean() for c in CONDS[1:]]
        print(f"  t={t:.2f}: " + "  ".join(f"{c} {v:+.5f}" for c, v in zip(CONDS[1:], row)))
    out = Path(a.out) if a.out else data / "heldout_loss.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "rating", "mirror_rating", "shuffle_rating"] + [f"loss_{c}" for c in CONDS])
        for i, r in enumerate(rows):
            w.writerow([r["file"], r["val"], mirror[i], shuffle[i]] + [per_file[c][i] for c in CONDS])
    print(f"\nper-file table: {out}")


if __name__ == "__main__":
    main()
