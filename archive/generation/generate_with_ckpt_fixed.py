#!/usr/bin/env python3
"""
Generate audio from Stable Audio Open v1 checkpoint - FIXED VERSION

This version properly handles checkpoint loading by:
1. Trying safetensors format first (most reliable)
2. Falling back to PyTorch Lightning .ckpt with proper key handling
3. Providing detailed diagnostics of what weights are actually loaded
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio
from einops import rearrange


def load_checkpoint_safetensors(model, checkpoint_path):
    """Load checkpoint from safetensors format (most reliable)."""
    try:
        from safetensors.torch import load_file

        print(f"\n📦 Attempting to load safetensors format...")
        state_dict = load_file(checkpoint_path)

        # Load directly - safetensors keys should match exactly
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        loaded_count = len(state_dict) - len(missing_keys)
        total_count = len(model.state_dict())

        print(f"✓ Loaded {loaded_count}/{total_count} weights from safetensors")

        if missing_keys:
            print(f"⚠️  Missing {len(missing_keys)} keys")
            if len(missing_keys) <= 5:
                for key in missing_keys:
                    print(f"    - {key}")

        if unexpected_keys:
            print(f"⚠️  Unexpected {len(unexpected_keys)} keys in checkpoint")

        return loaded_count > 0

    except ImportError:
        print("⚠️  safetensors not available, will try PyTorch format")
        return False
    except Exception as e:
        print(f"⚠️  safetensors load failed: {e}")
        return False


def load_checkpoint_pytorch_lightning(model, checkpoint_path):
    """Load PyTorch Lightning checkpoint with proper key handling."""
    print(f"\n📦 Attempting to load PyTorch Lightning checkpoint...")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✓ Loaded checkpoint file")
    except Exception as e:
        print(f"❌ Error loading checkpoint file: {e}")
        return False

    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"✓ Found 'state_dict' in checkpoint")
    else:
        state_dict = checkpoint
        print(f"ℹ  No 'state_dict' key, using checkpoint directly")

    # Get sample keys to understand format
    sample_keys = list(state_dict.keys())[:5]
    print(f"\nSample checkpoint keys:")
    for key in sample_keys:
        print(f"  - {key}")

    # Try to find the right prefix transformation
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    # Strategy 1: Direct match
    direct_matches = model_keys & ckpt_keys
    if len(direct_matches) > len(model_keys) * 0.8:
        print(f"\n✓ Using direct key matching (found {len(direct_matches)} matches)")
        cleaned_state_dict = state_dict

    # Strategy 2: Remove 'model.' prefix
    else:
        stripped_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                stripped_dict[key[6:]] = value
            else:
                stripped_dict[key] = value

        stripped_matches = model_keys & set(stripped_dict.keys())
        if len(stripped_matches) > len(model_keys) * 0.8:
            print(f"\n✓ Using 'model.' prefix removal (found {len(stripped_matches)} matches)")
            cleaned_state_dict = stripped_dict
        else:
            # Strategy 3: Try to understand the mismatch
            print(f"\n⚠️  Key matching strategies failed!")
            print(f"    Model expects: {len(model_keys)} keys")
            print(f"    Checkpoint has: {len(ckpt_keys)} keys")
            print(f"    Direct matches: {len(direct_matches)}")
            print(f"    After 'model.' removal: {len(stripped_matches)}")

            # Show first mismatch
            model_sample = list(model_keys)[0]
            ckpt_sample = list(ckpt_keys)[0]
            print(f"\n    Model expects: {model_sample}")
            print(f"    Checkpoint has: {ckpt_sample}")

            return False

    # Load the cleaned state dict
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)

    loaded_count = len(cleaned_state_dict) - len(missing_keys)
    total_count = len(model.state_dict())

    print(f"\n📊 Loaded {loaded_count}/{total_count} weights")

    if missing_keys:
        print(f"⚠️  Missing {len(missing_keys)} keys")
        if len(missing_keys) <= 5:
            for key in missing_keys:
                print(f"    - {key}")
        elif len(missing_keys) <= 20:
            for key in list(missing_keys)[:10]:
                print(f"    - {key}")
            print(f"    ... and {len(missing_keys) - 10} more")

    if unexpected_keys:
        print(f"⚠️  Unexpected {len(unexpected_keys)} keys in checkpoint")

    # Critical check: did we load meaningful weights?
    if loaded_count < total_count * 0.8:
        print(f"\n❌ CRITICAL: Only loaded {loaded_count}/{total_count} weights!")
        print(f"    The model will run with mostly uninitialized random weights.")
        print(f"    This will produce pure noise output.")
        return False

    return True


def generate_audio(model, prompt, model_config, guidance_scale=7.0, seconds=30, seed=None):
    """Generate audio from the model."""

    if seed is not None:
        torch.manual_seed(seed)

    print(f"\n🎵 Generating audio...")
    print(f"  Prompt: {prompt}")
    print(f"  Duration: {seconds}s")
    print(f"  Guidance scale: {guidance_scale}")

    # Get sample rate from config
    sample_rate = model_config.get("sample_rate", 16000)

    # Prepare conditioning
    with torch.no_grad():
        # Use stable-audio-tools for proper conditioning
        from stable_audio_tools.models.utils import get_prompt_for_conditioning

        conditioning = get_prompt_for_conditioning(model, prompt)

        # Generate
        output = model.generate(
            prompt_tokens=conditioning,
            use_sampling=True,
            top_k=250,
            top_p=0,
            temperature=1.0,
            num_samples=1,
            sample_rate=sample_rate,
            seconds_start=0,
            seconds_total=seconds,
            use_conditioning=True,
            conditioning_length=4096,
        )

    # Process output
    audio = output[0].cpu()
    audio = audio.squeeze(0)

    print(f"\n✓ Generated audio")
    print(f"  Shape: {audio.shape}")
    print(f"  Min: {audio.min():.4f}, Max: {audio.max():.4f}")
    print(f"  Mean: {audio.mean():.4f}, Std: {audio.std():.4f}")

    return audio, sample_rate


def main():
    parser = argparse.ArgumentParser(description="Generate audio from SAO v1 checkpoint")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint")
    parser.add_argument("--config", required=True, help="Path to model config")
    parser.add_argument("--prompt", required=True, help="Generation prompt")
    parser.add_argument("--output", default="generated_audio.wav", help="Output file")
    parser.add_argument("--guidance", type=float, default=7.0, help="Guidance scale")
    parser.add_argument("--seconds", type=float, default=30, help="Duration in seconds")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    print("\n" + "="*70)
    print("Stable Audio Open v1 - Audio Generation (FIXED)")
    print("="*70)

    # Load config
    print(f"\n📝 Loading config: {args.config}")
    config_path = Path(args.config).expanduser().absolute()
    with open(config_path) as f:
        model_config = json.load(f)
    print("✓ Config loaded")

    # Create model
    print(f"\n🏗️  Creating model from config...")
    try:
        from stable_audio_tools.models.utils import create_model_from_config
        model = create_model_from_config(model_config)
        print("✓ Model created")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False

    # Load checkpoint
    print(f"\n📂 Loading checkpoint: {args.ckpt}")
    ckpt_path = Path(args.ckpt).expanduser().absolute()

    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return False

    # Try different loading strategies
    success = False

    # Strategy 1: Try safetensors if the path points to a .safetensors file
    if ckpt_path.suffix == '.safetensors' or ckpt_path.with_suffix('.safetensors').exists():
        safetensors_path = ckpt_path if ckpt_path.suffix == '.safetensors' else ckpt_path.with_suffix('.safetensors')
        if safetensors_path.exists():
            success = load_checkpoint_safetensors(model, str(safetensors_path))

    # Strategy 2: PyTorch Lightning .ckpt format
    if not success and ckpt_path.suffix == '.ckpt':
        success = load_checkpoint_pytorch_lightning(model, str(ckpt_path))

    if not success:
        print(f"\n❌ Failed to load checkpoint!")
        return False

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"\n✓ Model moved to device: {device}")

    # Generate audio
    try:
        audio, sample_rate = generate_audio(
            model,
            args.prompt,
            model_config,
            guidance_scale=args.guidance,
            seconds=args.seconds,
            seed=args.seed
        )
    except Exception as e:
        print(f"❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Save output
    print(f"\n💾 Saving to: {args.output}")
    output_path = Path(args.output).expanduser().absolute()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torchaudio.save(output_path, audio.unsqueeze(0), sample_rate)
    print(f"✓ Saved audio ({output_path.stat().st_size / 1024 / 1024:.1f}MB)")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
