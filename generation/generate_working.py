#!/usr/bin/env python3
"""
Generate audio from Stable Audio Open v1 - Minimal Working Version

Uses only the imports that are confirmed to exist in your installation.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio


def main():
    parser = argparse.ArgumentParser(description="Generate audio from Stable Audio Open v1")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (.safetensors or .ckpt)")
    parser.add_argument("--config", required=True, help="Path to model_config.json")
    parser.add_argument("--prompt", required=True, help="Text prompt for generation")
    parser.add_argument("--output", default="generated.wav", help="Output file")
    parser.add_argument("--guidance", type=float, default=7.0, help="Guidance scale")
    parser.add_argument("--seconds", type=float, default=30, help="Duration in seconds")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--steps", type=int, default=250, help="Diffusion steps")

    args = parser.parse_args()

    # Set seed if provided (but not -1, which means random)
    if args.seed is not None and args.seed != -1:
        torch.manual_seed(args.seed)

    print("\n" + "="*70)
    print("Stable Audio Open v1 - Working Version")
    print("="*70)

    # Paths
    config_path = Path(args.config).expanduser().absolute()
    ckpt_path = Path(args.ckpt).expanduser().absolute()

    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return False

    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return False

    print(f"\n📝 Config: {config_path}")
    print(f"📦 Checkpoint: {ckpt_path}")

    # Load config
    with open(config_path) as f:
        model_config = json.load(f)
    print("✓ Config loaded")

    # Create model using the WORKING import on this system
    print(f"\n🏗️  Creating model...")
    try:
        from stable_audio_tools.models import create_model_from_config

        model = create_model_from_config(model_config)
        print("✓ Model created successfully")

    except ImportError as e:
        print(f"❌ Cannot import create_model_from_config: {e}")
        print(f"   This function should exist on your system")
        return False
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Load checkpoint - handle both formats
    print(f"\n📂 Loading checkpoint...")
    try:
        if ckpt_path.suffix == '.safetensors':
            from safetensors.torch import load_file
            state_dict = load_file(str(ckpt_path))
            print(f"✓ Loaded safetensors format")
        else:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            print(f"✓ Loaded PyTorch checkpoint")

        print(f"  Checkpoint has {len(state_dict)} keys")

        # Load into model
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        loaded_count = len(state_dict) - len(missing_keys)
        total_count = len(model.state_dict())

        print(f"✓ Loaded {loaded_count}/{total_count} weights ({100 * loaded_count // total_count}%)")

        if loaded_count < total_count * 0.7:
            print(f"\n❌ CRITICAL: Only {loaded_count}/{total_count} weights loaded!")
            print(f"   Model will produce noise.")
            return False

    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"\n✓ Model ready on device: {device}")

    # Generate audio
    print(f"\n🎵 Generating audio...")
    print(f"   Prompt: {args.prompt}")
    print(f"   Duration: {args.seconds}s")
    print(f"   Guidance: {args.guidance}")
    print(f"   Steps: {args.steps}")

    try:
        from stable_audio_tools.inference.generation import generate_diffusion_cond

        # Build conditioning as the library expects
        conditioning = [{
            "text": args.prompt,
            "seconds_total": args.seconds
        }]

        with torch.no_grad():
            # Only pass seed if it's a valid integer (not None or -1)
            generation_kwargs = {
                "model": model,
                "conditioning": conditioning,
                "steps": args.steps,
                "cfg_scale": args.guidance,
                "sample_rate": model_config.get("sample_rate", 16000),
                "sample_size": int(args.seconds * model_config.get("sample_rate", 16000)),
                "device": device,
            }
            
            # Only add seed if it's valid
            if args.seed is not None and args.seed != -1:
                generation_kwargs["seed"] = args.seed
            
            audio = generate_diffusion_cond(**generation_kwargs)

        print(f"\n✓ Generation complete!")

        # Process output
        audio = audio.cpu()
        if audio.dim() == 3:
            audio = audio.squeeze(0)
        elif audio.dim() == 1:
            audio = audio.unsqueeze(0)

        print(f"  Audio shape: {audio.shape}")
        print(f"  Duration: {audio.shape[-1] / model_config.get('sample_rate', 16000):.2f}s")

        # Save
        output_path = Path(args.output).expanduser().absolute()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sample_rate = model_config.get("sample_rate", 16000)
        torchaudio.save(str(output_path), audio, sample_rate)

        file_size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"\n✅ Saved to: {output_path}")
        print(f"   Size: {file_size_mb:.2f}MB")

        return True

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
