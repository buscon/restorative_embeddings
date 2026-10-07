#!/usr/bin/env python3
"""Systematic generation: every (seed x masker x pleasantness level) with one loaded model.

Writes <out>/s<seed>_<masker>_p<level>.wav and <out>/grid.csv (seed,masker,p,file,prompt),
which evaluate_grid.py reads. The model is loaded once; existing files are skipped, so an
interrupted run can be restarted.

    python generation/generate_grid.py --ckpt <model.ckpt> --config configs/model_config.json \
        --out out_grid_1500 [--seeds 1 2 3 4] [--levels -0.8 0 0.8] [--position background]

Prompts use the training caption format (rsd/captions.py).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torchaudio
from einops import rearrange

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from generate import load_weights  # noqa: E402
from rsd.captions import MASKER_WORDS, format_pleasantness, make_caption  # noqa: E402


def main() -> bool:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--levels", type=float, nargs="+", default=[-0.8, 0.0, 0.8])
    ap.add_argument("--maskers", nargs="+", default=list(MASKER_WORDS.values()),
                    help="masker phrases as used in training captions")
    ap.add_argument("--prompt-template", default=None,
                    help='custom prompt with {item} (each entry of --maskers) and {p}, e.g. '
                         '"soundscape at {item} in London [ISOPleasant: {p}]" for an ISD model')
    ap.add_argument("--position", choices=["foreground", "background"], default="background")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--sampler", default="dpmpp-3m-sde")
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--negative-prompt", default=None,
                    help='text the sampler is pushed away from, e.g. "music, melody, singing, instruments"')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    sr, size = cfg["sample_rate"], cfg["sample_size"]

    from stable_audio_tools.models import create_model_from_config
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    model = create_model_from_config(cfg)
    if not load_weights(model, Path(args.ckpt).expanduser(), use_ema=args.use_ema):
        return False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval().requires_grad_(False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, m, p) for s in args.seeds for m in args.maskers for p in args.levels]
    manifest = []
    for k, (seed, masker, p) in enumerate(jobs, 1):
        name = f"s{seed}_{masker.replace(' ', '_')}_p{p:.2f}.wav"
        if args.prompt_template:
            prompt = args.prompt_template.format(item=masker, p=format_pleasantness(p))
        else:
            prompt = make_caption(masker, args.position == "foreground", p)
        manifest.append({"seed": seed, "masker": masker, "p": p, "file": name, "prompt": prompt})
        path = out / name
        if path.exists():
            print(f"[{k}/{len(jobs)}] exists: {name}", flush=True)
            continue
        print(f"[{k}/{len(jobs)}] {name}  |  {prompt}", flush=True)
        neg = None
        if args.negative_prompt:
            neg = [{"prompt": args.negative_prompt, "seconds_start": 0.0, "seconds_total": args.seconds}]
        with torch.no_grad():
            audio = generate_diffusion_cond(
                model, steps=args.steps, cfg_scale=args.guidance,
                conditioning=[{"prompt": prompt, "seconds_start": 0.0, "seconds_total": args.seconds}],
                negative_conditioning=neg,
                sample_size=size, sampler_type=args.sampler, device=device, seed=seed)
        audio = rearrange(audio, "b d n -> d (b n)")[:, : int(args.seconds * sr)]
        audio = (audio / audio.abs().max().clamp(min=1e-8)).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        torchaudio.save(str(path), audio, sr)

    with open(out / "grid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "masker", "p", "file", "prompt"])
        w.writeheader()
        w.writerows(manifest)
    print(f"done: {len(jobs)} files in {out}")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
