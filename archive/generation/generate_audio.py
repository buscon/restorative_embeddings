#!/usr/bin/env python3
"""
Generate audio from Stable Audio Open v1 - Simple & Fixed Version

Key fix: Checkpoint keys have DOUBLE "model." prefix
- Checkpoint: model.model.timestep_features.weight
- We need to strip "model." to get model.timestep_features.weight
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio


def load_safetensors_checkpoint(model, checkpoint_path):
    """Load from safetensors format."""
    try:
        from safetensors.torch import load_file

        print(f"\n📦 Loading safetensors checkpoint...")
        state_dict = load_file(checkpoint_path)

        # Load directly - keys should match
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        loaded_count = len(state_dict) - len(missing_keys)
        total_count = len(model.state_dict())

        print(f"✓ Loaded {loaded_count}/{total_count} weights from safetensors")

        if missing_keys and len(missing_keys) < 20:
            print(f"  Missing keys: {missing_keys[:5]}")

        return loaded_count > 0

    except Exception as e:
        print(f"❌ safetensors load failed: {e}")
        return False


def load_pytorch_checkpoint(model, checkpoint_path):
    """Load PyTorch Lightning checkpoint with proper prefix handling."""
    print(f"\n📦 Loading PyTorch checkpoint...")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✓ Loaded checkpoint file")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    print(f"  Checkpoint has {len(state_dict)} keys")

    # Show sample keys to understand format
    sample_keys = list(state_dict.keys())[:3]
    print(f"  Sample keys:")
    for key in sample_keys:
        print(f"    - {key}")

    # FIX: Strip ONE "model." prefix (checkpoint has model.model.*, we need model.*)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # Remove first 'model.' prefix
            cleaned_state_dict[new_key] = value
        else:
            cleaned_state_dict[key] = value

    # Load into model
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)

    loaded_count = len(cleaned_state_dict) - len(missing_keys)
    total_count = len(model.state_dict())

    print(f"\n✓ Loaded {loaded_count}/{total_count} weights")

    # Critical check
    if loaded_count < total_count * 0.7:
        print(f"\n❌ CRITICAL: Only loaded {loaded_count}/{total_count} weights!")
        print(f"   Model will produce noise with random weights.")

        # Debug info
        model_keys_set = set(model.state_dict().keys())
        cleaned_keys_set = set(cleaned_state_dict.keys())
        common = model_keys_set & cleaned_keys_set

        print(f"\n   Debug info:")
        print(f"   Model expects: {len(model_keys_set)} keys")
        print(f"   Cleaned checkpoint has: {len(cleaned_keys_set)} keys")
        print(f"   Keys that match: {len(common)}")

        if common:
            sample_match = list(common)[0]
            print(f"   Sample matching key: {sample_match}")

        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate audio from Stable Audio Open v1")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (.safetensors or .ckpt)")
    parser.add_argument("--config", required=True, help="Path to model_config.json")
    parser.add_argument("--prompt", required=True, help="Text prompt for generation")
    parser.add_argument("--output", default="generated.wav", help="Output file")
    parser.add_argument("--guidance", type=float, default=7.0, help="Guidance scale")
    parser.add_argument("--seconds", type=float, default=30, help="Duration in seconds")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print("\n" + "="*70)
    print("Stable Audio Open v1 - Generation (Fixed)")
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

    # Create model using stable-audio-tools
    print(f"\n🏗️  Creating model...")
    try:
        from stable_audio_tools.models import get_model_from_pretrained

        # Try to load model with checkpoint
        model = get_model_from_pretrained(
            model_type="diffusion",
            model_name="stabilityai/stable-audio-open-1.0"
        )
        print("✓ Model loaded from pretrained")

    except Exception as e:
        print(f"⚠️  Couldn't use pretrained loader: {e}")
        print(f"   Trying manual model creation...")

        try:
            # Manual model creation
            from stable_audio_tools.models.diffusion_uncond_cfm import DiffusionUnconditionedCFM

            model = DiffusionUnconditionedCFM(
                **model_config.get("model", {})
            )
            print("✓ Model created manually")

        except Exception as e2:
            print(f"❌ Error creating model: {e2}")
            return False

    # Load checkpoint
    print(f"\n📂 Loading checkpoint...")

    if ckpt_path.suffix == '.safetensors':
        success = load_safetensors_checkpoint(model, str(ckpt_path))
    else:
        success = load_pytorch_checkpoint(model, str(ckpt_path))

    if not success:
        print(f"\n❌ Failed to load checkpoint properly")
        return False

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"\n✓ Model on device: {device}")

    # Generate
    print(f"\n🎵 Generating audio...")
    print(f"   Prompt: {args.prompt}")
    print(f"   Duration: {args.seconds}s")
    print(f"   Guidance: {args.guidance}")

    try:
        with torch.no_grad():
            # Use stable-audio-tools generate interface
            from stable_audio_tools.inference.generation import generate_diffusion_cond

            # Get sample rate from model or config
            sample_rate = model_config.get("sample_rate", 16000)

            # Generate using the library's generation function
            audio = generate_diffusion_cond(
                model=model,
                prompt=args.prompt,
                seconds=args.seconds,
                guidance_scale=args.guidance,
                temperature=1.0,
                device=device,
                seed=args.seed,
            )

            print(f"✓ Generated {len(audio) / sample_rate:.1f}s of audio")

            # Save
            output_path = Path(args.output).expanduser().absolute()
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            torchaudio.save(str(output_path), audio.cpu(), sample_rate)

            print(f"\n💾 Saved to: {output_path}")
            print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f}MB")

            return True

    except Exception as e:
        print(f"❌ Generation error: {e}")
        import traceback
        traceback.print_exc()

        # Try fallback
        print(f"\n⚠️  Trying fallback generation method...")
        try:
            # Direct model generation if available
            if hasattr(model, 'sample'):
                audio = model.sample(num_samples=1, num_steps=50)
                audio = audio.squeeze(0)

                output_path = Path(args.output).expanduser().absolute()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                sample_rate = model_config.get("sample_rate", 16000)

                torchaudio.save(str(output_path), audio.cpu(), sample_rate)
                print(f"✓ Generated (fallback method)")
                return True
            else:
                print(f"❌ No fallback generation method available")
                return False

        except Exception as e3:
            print(f"❌ Fallback also failed: {e3}")
            return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
