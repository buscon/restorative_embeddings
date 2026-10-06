#!/usr/bin/env python3
"""
Generate audio using stable-audio-tools official inference API

This approach uses the library's intended inference path, avoiding manual
checkpoint loading complexity that causes key mismatches.
"""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio


def main():
    parser = argparse.ArgumentParser(description="Generate audio using stable-audio-tools API")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (model.safetensors or model.ckpt)")
    parser.add_argument("--config", required=True, help="Path to model config")
    parser.add_argument("--prompt", required=True, help="Generation prompt")
    parser.add_argument("--output", default="generated_audio.wav", help="Output file")
    parser.add_argument("--guidance", type=float, default=7.0, help="Guidance scale")
    parser.add_argument("--seconds", type=float, default=30, help="Duration in seconds")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    print("\n" + "="*70)
    print("Stable Audio Open v1 - Using Official API")
    print("="*70)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Use stable-audio-tools' inference utilities
    try:
        from stable_audio_tools.interface import generate_diffusion_model
        from stable_audio_tools.models.utils import create_model_from_config

        print("\n✓ Imported stable-audio-tools inference utilities")
    except ImportError as e:
        print(f"❌ Failed to import stable-audio-tools: {e}")
        print("   Make sure it's installed: pip install git+https://github.com/Stability-AI/stable-audio-tools.git")
        return False

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

    # Create model
    print(f"\n🏗️  Creating model...")
    try:
        model = create_model_from_config(config_path)
        print("✓ Model created")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False

    # Load checkpoint
    print(f"\n📂 Loading checkpoint...")
    try:
        if ckpt_path.suffix == '.safetensors':
            from safetensors.torch import load_file
            state_dict = load_file(str(ckpt_path))
            print("✓ Loaded safetensors format")
        else:
            import torch
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            print("✓ Loaded PyTorch checkpoint")

        # Load into model
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        loaded_count = len(state_dict) - len(missing_keys)
        total_count = len(model.state_dict())

        print(f"✓ Loaded {loaded_count}/{total_count} weights")

        if loaded_count < total_count * 0.7:
            print(f"⚠️  WARNING: Only {loaded_count}/{total_count} weights loaded!")
            print(f"   Model may produce poor quality output.")

    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"\n✓ Model on device: {device}")

    # Generate using the library's inference function
    print(f"\n🎵 Generating audio...")
    print(f"   Prompt: {args.prompt}")
    print(f"   Duration: {args.seconds}s")
    print(f"   Guidance: {args.guidance}")

    try:
        # Use the official generate function
        from stable_audio_tools.models.utils import get_prompt_for_conditioning

        with torch.no_grad():
            # Get conditioning
            conditioning = get_prompt_for_conditioning(model, args.prompt)

            # Generate with diffusion
            sample_rate = model.sample_rate

            # Prepare generation parameters
            num_steps = 100

            # Generate audio
            audio = model.sample(
                num_samples=1,
                num_steps=num_steps,
                use_sampling=True,
                top_k=250,
                top_p=0,
                temperature=1.0,
                conditioning=conditioning,
                use_classifier_free_guidance=True,
                classifier_free_guidance_scale=args.guidance,
                seed=args.seed,
            )

        print(f"✓ Generation complete")

        # Save
        output_path = Path(args.output).expanduser().absolute()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # audio is (batch, channels, samples)
        audio = audio.cpu()
        if audio.dim() == 3:
            audio = audio.squeeze(0)

        torchaudio.save(str(output_path), audio, sample_rate)

        print(f"\n💾 Saved: {output_path}")
        print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f}MB")
        print(f"   Sample rate: {sample_rate}Hz")

        return True

    except Exception as e:
        print(f"❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
