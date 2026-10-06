#!/usr/bin/env python3
"""
Diagnose the exact format of checkpoints to understand key mismatch.

This script will:
1. Inspect PyTorch Lightning checkpoint (.ckpt) format
2. Inspect safetensors format (if available)
3. Compare with model's expected state_dict keys
4. Identify the exact mismatch causing weight loading to fail
"""

import json
import sys
from pathlib import Path

def diagnose_model_keys():
    """Check what keys the model actually expects."""
    print("\n🔍 Checking SAO v1 model expected keys...")
    print("=" * 70)

    try:
        from stable_audio_tools.models.utils import create_model_from_config

        config_path = Path("~/stableaudio/models/stabilityai__stable-audio-open-1.0/model_config.json").expanduser()
        print(f"Loading model config from: {config_path}")

        with open(config_path) as f:
            config = json.load(f)

        print("✓ Config loaded")
        print(f"  Model type: {config.get('model_type', 'unknown')}")

        # Create model to see expected keys
        model = create_model_from_config(config)
        print("✓ Model created from config")

        model_keys = list(model.state_dict().keys())
        print(f"\n📊 Model expects {len(model_keys)} keys")
        print("\nFirst 10 expected keys:")
        for key in model_keys[:10]:
            print(f"  - {key}")

        print("\n✓ Last 5 expected keys:")
        for key in model_keys[-5:]:
            print(f"  - {key}")

        return model_keys

    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return None

def diagnose_ckpt_format(ckpt_path):
    """Inspect PyTorch Lightning checkpoint format."""
    print("\n🔍 Inspecting PyTorch Lightning checkpoint...")
    print("=" * 70)

    try:
        import torch

        ckpt_path = Path(ckpt_path).expanduser()
        print(f"Checkpoint: {ckpt_path}")

        if not ckpt_path.exists():
            print(f"❌ File not found")
            return None

        checkpoint = torch.load(ckpt_path, map_location='cpu')

        # Check structure
        print(f"\nCheckpoint top-level keys:")
        for key in checkpoint.keys():
            if key == 'state_dict':
                print(f"  - {key}: (contains model weights)")
            else:
                print(f"  - {key}")

        # Inspect state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            ckpt_keys = list(state_dict.keys())

            print(f"\n📊 Checkpoint state_dict has {len(ckpt_keys)} keys")
            print("\nFirst 10 checkpoint keys:")
            for key in ckpt_keys[:10]:
                print(f"  - {key}")

            print("\n✓ Last 5 checkpoint keys:")
            for key in ckpt_keys[-5:]:
                print(f"  - {key}")

            return ckpt_keys
        else:
            print("❌ No 'state_dict' found in checkpoint")
            return None

    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None

def diagnose_safetensors_format(safetensors_path):
    """Inspect safetensors format if available."""
    print("\n🔍 Inspecting safetensors format...")
    print("=" * 70)

    safetensors_path = Path(safetensors_path).expanduser()
    print(f"File: {safetensors_path}")

    if not safetensors_path.exists():
        print(f"❌ File not found")
        return None

    try:
        from safetensors.torch import load_file

        state_dict = load_file(safetensors_path)
        st_keys = list(state_dict.keys())

        print(f"✓ Loaded safetensors format")
        print(f"\n📊 safetensors has {len(st_keys)} keys")
        print("\nFirst 10 safetensors keys:")
        for key in st_keys[:10]:
            print(f"  - {key}")

        print("\n✓ Last 5 safetensors keys:")
        for key in st_keys[-5:]:
            print(f"  - {key}")

        return st_keys

    except Exception as e:
        print(f"❌ Error loading safetensors: {e}")
        return None

def compare_formats(model_keys, ckpt_keys, st_keys):
    """Compare the different formats."""
    print("\n📊 Format Comparison")
    print("=" * 70)

    if model_keys:
        print(f"Model expects: {len(model_keys)} keys")
    if ckpt_keys:
        print(f"Checkpoint has: {len(ckpt_keys)} keys")
    if st_keys:
        print(f"Safetensors has: {len(st_keys)} keys")

    if model_keys and ckpt_keys:
        print("\n🔍 Matching checkpoint keys to model keys...")

        # Try to find what prefix transformation is needed
        sample_ckpt_key = ckpt_keys[0]
        sample_model_key = model_keys[0]

        print(f"\nSample checkpoint key: {sample_ckpt_key}")
        print(f"Sample model key: {sample_model_key}")

        # Check if removing 'model.' helps
        if sample_ckpt_key.startswith('model.'):
            stripped = sample_ckpt_key[6:]
            print(f"After removing 'model.': {stripped}")
            if stripped == sample_model_key:
                print("✓ This matches! The 'model.' removal strategy should work.")
            else:
                print("❌ Still doesn't match after removing 'model.'")

        # Check exact match count
        direct_matches = sum(1 for k in ckpt_keys if k in model_keys)
        stripped_matches = sum(1 for k in ckpt_keys if k.startswith('model.') and k[6:] in model_keys)

        print(f"\nDirect matches: {direct_matches}/{len(ckpt_keys)}")
        print(f"Matches after removing 'model.': {stripped_matches}/{len(ckpt_keys)}")

        if stripped_matches == 0:
            print("\n⚠️  Neither strategy works!")
            print("\nLet's check for other common prefixes...")

            # Find common prefixes
            prefixes = set()
            for key in ckpt_keys[:20]:
                parts = key.split('.')
                if len(parts) > 1:
                    prefixes.add(parts[0])

            print(f"Common prefixes in checkpoint: {prefixes}")

def main():
    print("\n" + "="*70)
    print("Checkpoint Format Diagnostic")
    print("="*70)

    # Paths
    config_path = Path("~/stableaudio/models/stabilityai__stable-audio-open-1.0/model_config.json").expanduser()
    ckpt_path = Path("~/stableaudio/models/stabilityai__stable-audio-open-1.0/model.ckpt").expanduser()
    safetensors_path = Path("~/stableaudio/models/stabilityai__stable-audio-open-1.0/model.safetensors").expanduser()

    print(f"\n📝 Paths to check:")
    print(f"  Config: {config_path} {'✓' if config_path.exists() else '❌'}")
    print(f"  Checkpoint (.ckpt): {ckpt_path} {'✓' if ckpt_path.exists() else '❌'}")
    print(f"  Safetensors: {safetensors_path} {'✓' if safetensors_path.exists() else '❌'}")

    # Run diagnostics
    model_keys = diagnose_model_keys()
    ckpt_keys = diagnose_ckpt_format(ckpt_path) if ckpt_path.exists() else None
    st_keys = diagnose_safetensors_format(safetensors_path) if safetensors_path.exists() else None

    # Compare
    if model_keys:
        compare_formats(model_keys, ckpt_keys, st_keys)

    # Recommendation
    print("\n💡 Recommendation:")
    print("=" * 70)

    if st_keys and len(st_keys) > 0:
        print("✓ safetensors format is available and should work!")
        print("  Use: from safetensors.torch import load_file")
    elif ckpt_keys and len(ckpt_keys) > 0:
        print("⚠️  Must debug .ckpt format mismatch")
        print("  The stripped keys still don't match the model")
    else:
        print("❌ No valid checkpoint format found")

if __name__ == "__main__":
    main()
