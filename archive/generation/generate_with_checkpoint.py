#!/usr/bin/env python3
"""
Generate audio using fine-tuned Stable Audio Open v1 checkpoint.
Loads a PyTorch Lightning checkpoint and generates audio with text prompts.
"""

import torch
import soundfile as sf
from pathlib import Path
import json
import sys
import argparse

from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.utils import copy_state_dict

def load_checkpoint(checkpoint_path: str, model_config_path: str, device: str = "cuda"):
    """Load a fine-tuned Stable Audio checkpoint."""
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)

    print(f"Creating model from config: {model_config_path}")
    model = create_model_from_config(model_config)
    model = model.to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"Found 'state_dict' in checkpoint (PyTorch Lightning format)")
        else:
            state_dict = checkpoint
            print(f"Loading checkpoint as direct state dict")

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]
                cleaned_state_dict[new_key] = value
            else:
                cleaned_state_dict[key] = value

        copy_state_dict(model, cleaned_state_dict)
        print(f"✓ Checkpoint loaded successfully\n")

    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        raise

    model.eval()
    return model, model_config

def generate_audio(model, model_config, prompt: str, output_path: Path, duration: float = 30.0, steps: int = 100, guidance_scale: float = 7.0, seed: int = 42, device: str = "cuda") -> bool:
    """Generate a single audio sample from a text prompt."""
    sample_rate = model_config.get('sample_rate', 44100)

    try:
        print(f"Generating: {prompt[:80]}...")

        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0.0,
            "seconds_end": duration
        }]

        audio = generate_diffusion_cond(
            model,
            conditioning=conditioning,
            steps=steps,
            cfg_scale=guidance_scale,
            sample_size=int(sample_rate * duration),
            device=device,
            seed=seed,
        )

        if hasattr(audio, 'cpu'):
            audio = audio.cpu()
        audio_np = audio.float().numpy()

        if audio_np.ndim == 3:
            audio_np = audio_np[0]

        if audio_np.ndim == 2:
            audio_np = audio_np.T
        elif audio_np.ndim == 1:
            pass

        max_val = abs(audio_np).max()
        if max_val > 1.0:
            audio_np = audio_np / max_val
            print(f"  → Normalized (peak: {max_val:.2f})")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio_np, sample_rate)
        duration_s = audio_np.shape[0] / sample_rate
        print(f"  ✓ Saved: {output_path} ({duration_s:.2f}s)\n")

        return True

    except Exception as e:
        print(f"  ✗ Error generating audio: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate audio using fine-tuned Stable Audio Open v1 checkpoint")
    parser.add_argument('--ckpt', '--checkpoint', type=str, required=True, help='Path to fine-tuned checkpoint (.ckpt file)')
    parser.add_argument('--config', type=str, required=True, help='Path to model_config.json')
    parser.add_argument('--prompt', type=str, action='append', required=True, help='Text prompt(s) for generation')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output directory')
    parser.add_argument('--duration', '-d', type=float, default=30.0, help='Duration in seconds')
    parser.add_argument('--steps', '-s', type=int, default=50, help='Number of diffusion steps')
    parser.add_argument('--guidance', '-g', type=float, default=7.0, help='Guidance scale')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None, help='Device to use')

    args = parser.parse_args()

    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    else:
        output_dir = checkpoint_path.parent / "generated_audio"

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Output directory: {output_dir}\n")

    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    if not config_path.exists():
        print(f"✗ Config not found: {config_path}")
        sys.exit(1)

    try:
        model, model_config = load_checkpoint(str(checkpoint_path), str(config_path), device=device)
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        sys.exit(1)

    print(f"Generating {len(args.prompt)} audio sample(s)...")
    print(f"Duration: {args.duration}s, Steps: {args.steps}, Guidance: {args.guidance}\n")

    successful = 0
    for idx, prompt in enumerate(args.prompt, 1):
        prompt_slug = prompt[:30].replace(' ', '_').replace('[', '').replace(']', '')
        output_filename = f"generated_{idx:02d}_{prompt_slug}.wav"
        output_path = output_dir / output_filename

        if generate_audio(model, model_config, prompt, output_path, duration=args.duration, steps=args.steps, guidance_scale=args.guidance, seed=args.seed + idx, device=device):
            successful += 1

    print(f"✓ Generation complete: {successful}/{len(args.prompt)} successful")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
