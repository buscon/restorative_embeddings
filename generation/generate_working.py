#!/usr/bin/env python3
"""
Generate audio from Stable Audio Open v1 - Handles fine-tuned checkpoints with diffusion. prefix
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

    if args.seed is not None and args.seed != -1:
        torch.manual_seed(args.seed)

    print("\n" + "="*70)
    print("Stable Audio Open v1 - Working Version")
    print("="*70)

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

    with open(config_path) as f:
        model_config = json.load(f)
    print("✓ Config loaded")

    print(f"\n🏗️  Creating model...")
    try:
        from stable_audio_tools.models import create_model_from_config
        model = create_model_from_config(model_config)
        print("✓ Model created successfully")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False

    print(f"\n📂 Loading checkpoint...")
    try:
        if ckpt_path.suffix == '.safetensors':
            from safetensors.torch import load_file
            state_dict = load_file(str(ckpt_path))
            print(f"✓ Loaded safetensors format")
        else:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            
            # Handle PyTorch Lightning checkpoints (from fine-tuning)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"✓ Loaded PyTorch Lightning checkpoint (extracted state_dict)")
            else:
                state_dict = checkpoint
                print(f"✓ Loaded PyTorch checkpoint")

        print(f"  Checkpoint has {len(state_dict)} keys")

        # Handle fine-tuned checkpoint structure:
        # Fine-tuned checkpoints have "diffusion." prefix and include EMA weights
        # Strip the prefix and ignore EMA weights
        processed_state_dict = {}
        
        for key, value in state_dict.items():
            # Skip EMA weights (training-only)
            if key.startswith('diffusion_ema.'):
                continue
            
            # Strip "diffusion." prefix if present
            if key.startswith('diffusion.'):
                new_key = key[len('diffusion.'):]
                processed_state_dict[new_key] = value
            else:
                # Keep as-is (base model or already stripped)
                processed_state_dict[key] = value
        
        print(f"  After processing: {len(processed_state_dict)} keys (removed EMA weights)")
        
        state_dict = processed_state_dict
        
        # Load into model
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        loaded_count = len(state_dict) - len(missing_keys)
        total_count = len(model.state_dict())

        print(f"✓ Loaded {loaded_count}/{total_count} weights ({100 * loaded_count // total_count}%)")

        if loaded_count < total_count * 0.7:
            print(f"\n❌ CRITICAL: Only {loaded_count}/{total_count} weights loaded!")
            print(f"   Expected at least 70% ({int(total_count * 0.7)}) weights")
            print(f"   Missing keys: {len(missing_keys)}")
            return False

    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"\n✓ Model ready on device: {device}")

    print(f"\n🎵 Generating audio...")
    print(f"   Prompt: {args.prompt}")
    print(f"   Duration: {args.seconds}s")
    print(f"   Guidance: {args.guidance}")
    print(f"   Steps: {args.steps}")

    try:
        from stable_audio_tools.inference.generation import generate_diffusion_cond

        # Include all required conditioning keys
        conditioning = [{
            "prompt": args.prompt,
            "seconds_start": 0.0,
            "seconds_total": args.seconds
        }]

        with torch.no_grad():
            generation_kwargs = {
                "model": model,
                "conditioning": conditioning,
                "steps": args.steps,
                "cfg_scale": args.guidance,
                "sample_rate": model_config.get("sample_rate", 16000),
                "sample_size": int(args.seconds * model_config.get("sample_rate", 16000)),
                "device": device,
            }
            
            if args.seed is not None and args.seed != -1:
                generation_kwargs["seed"] = args.seed
            
            audio = generate_diffusion_cond(**generation_kwargs)

        print(f"\n✓ Generation complete!")

        audio = audio.cpu()
        if audio.dim() == 3:
            audio = audio.squeeze(0)
        elif audio.dim() == 1:
            audio = audio.unsqueeze(0)

        print(f"  Audio shape: {audio.shape}")
        print(f"  Duration: {audio.shape[-1] / model_config.get('sample_rate', 16000):.2f}s")

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
