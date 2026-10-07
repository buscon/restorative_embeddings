#!/usr/bin/env python3
"""
Generate audio with Stable Audio Open v1 from either:
  - the base model (.safetensors), or
  - a PyTorch Lightning fine-tuned checkpoint (.ckpt)

Checkpoint handling:
  - .safetensors: keys already match the model, loaded directly.
  - .ckpt: weights live under "state_dict"; keys carry a "diffusion." prefix
    (stripped here) and a duplicate set under "diffusion_ema." (EMA, skipped by
    default; use --use-ema to load those instead).

Usage:
  python3 generate.py --ckpt model.safetensors --config model_config.json \
      --prompt "park, quiet" --seconds 10 --output base.wav

  python3 generate.py --ckpt epoch=44-step=2000.ckpt --config model_config.json \
      --prompt "park, quiet" --seconds 10 --output finetuned.wav
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio
from einops import rearrange

# Fixed seed so base and fine-tuned outputs are directly comparable.
# Change it here in the script when you want a different sample.
SEED = 42


def load_weights(model, ckpt_path: Path, use_ema: bool = False) -> bool:
    print("\n📂 Loading weights...")

    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(str(ckpt_path))
        print(f"  safetensors file with {len(state_dict)} keys")
    else:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        raw = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        print(f"  checkpoint with {len(raw)} keys")

        keep_prefix = "diffusion_ema." if use_ema else "diffusion."
        skip_prefix = "diffusion." if use_ema else "diffusion_ema."
        state_dict = {}
        skipped = 0
        for key, value in raw.items():
            if key.startswith(keep_prefix):
                state_dict[key[len(keep_prefix):]] = value
            elif key.startswith(skip_prefix):
                skipped += 1
            else:
                state_dict[key] = value  # anything else, kept as-is
        print(f"  using {'EMA' if use_ema else 'regular'} weights, "
              f"skipped {skipped} keys, {len(state_dict)} remaining")

    result = model.load_state_dict(state_dict, strict=False)
    total = len(model.state_dict())
    loaded = total - len(result.missing_keys)
    print(f"✓ Loaded {loaded}/{total} weights "
          f"({len(result.unexpected_keys)} unexpected keys ignored)")

    if loaded < total * 0.95:
        print("❌ Too many weights missing; the model would produce noise.")
        print(f"   Sample missing: {result.missing_keys[:5]}")
        print(f"   Sample unexpected: {result.unexpected_keys[:5]}")
        return False
    return True


def main() -> bool:
    parser = argparse.ArgumentParser(description="Stable Audio Open v1 generation")
    parser.add_argument("--ckpt", required=True, help=".safetensors (base) or .ckpt (fine-tuned)")
    parser.add_argument("--config", required=True, help="Path to model_config.json")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--output", default="generated.wav", help="Output wav file")
    parser.add_argument("--seconds", type=float, default=10.0, help="Duration in seconds (max 47)")
    parser.add_argument("--steps", type=int, default=100, help="Diffusion steps")
    parser.add_argument("--guidance", type=float, default=7.0, help="CFG scale")
    parser.add_argument("--sampler", default="dpmpp-3m-sde", help="Sampler type")
    parser.add_argument("--use-ema", action="store_true",
                        help="For .ckpt: load EMA weights instead of the regular ones")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().absolute()
    ckpt_path = Path(args.ckpt).expanduser().absolute()
    for p in (config_path, ckpt_path):
        if not p.exists():
            print(f"❌ Not found: {p}")
            return False

    seed = SEED

    print("=" * 70)
    print("Stable Audio Open v1 - Generation")
    print("=" * 70)
    print(f"Config:     {config_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Prompt:     {args.prompt}")
    print(f"Seconds:    {args.seconds}")
    print(f"Seed:       {seed}")

    with open(config_path) as f:
        model_config = json.load(f)
    sample_rate = model_config["sample_rate"]
    max_sample_size = model_config["sample_size"]

    from stable_audio_tools.models import create_model_from_config
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    print("\n🏗️  Creating model...")
    model = create_model_from_config(model_config)

    if not load_weights(model, ckpt_path, use_ema=args.use_ema):
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval().requires_grad_(False)
    print(f"✓ Model on {device}")

    conditioning = [{
        "prompt": args.prompt,
        "seconds_start": 0.0,
        "seconds_total": float(args.seconds),
    }]

    # Always generate at the model's native length, then trim to the requested duration.
    print("\n🎵 Generating...")
    with torch.no_grad():
        output = generate_diffusion_cond(
            model,
            steps=args.steps,
            cfg_scale=args.guidance,
            conditioning=conditioning,
            sample_size=max_sample_size,
            sampler_type=args.sampler,
            device=device,
            seed=seed,
        )

    # (batch, channels, samples) -> (channels, batch*samples)
    output = rearrange(output, "b d n -> d (b n)")
    output = output[:, : int(args.seconds * sample_rate)]

    peak = output.abs().max().clamp(min=1e-8)
    output = (output / peak).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    out_path = Path(args.output).expanduser().absolute()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), output, sample_rate)

    print(f"\n💾 Saved: {out_path}")
    print(f"   {output.shape[1] / sample_rate:.1f}s @ {sample_rate} Hz, "
          f"{out_path.stat().st_size / 1024 / 1024:.1f} MB")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
