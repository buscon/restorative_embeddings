#!/usr/bin/env python3
"""
Diagnose fine-tuned checkpoint structure
"""

import json
import torch
from pathlib import Path

# Paths
ckpt_path = Path.home() / "Documents/restorative_embeddings/generation/train/checkpoints/restorative_soundscapes_finetuning_360/cwafugzf/checkpoints/epoch=44-step=2000.ckpt"
config_path = Path.home() / "stableaudio/models/stabilityai__stable-audio-open-1.0/model_config.json"

print("="*70)
print("Fine-tuned Checkpoint Diagnostic")
print("="*70)

# Load checkpoint
print(f"\nLoading checkpoint: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location='cpu')

print(f"\nCheckpoint type: {type(checkpoint)}")
print(f"Checkpoint keys: {list(checkpoint.keys())[:10]}...")

# Extract state_dict
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    print(f"\n✓ Found 'state_dict' key in checkpoint")
else:
    state_dict = checkpoint
    print(f"\n✗ No 'state_dict' key, using entire checkpoint")

print(f"State dict has {len(state_dict)} keys")

# Show sample keys
print(f"\nSample keys from checkpoint:")
sample_keys = list(state_dict.keys())[:5]
for key in sample_keys:
    print(f"  - {key}")

# Load model
print(f"\n\nLoading model from config...")
from stable_audio_tools.models import create_model_from_config

with open(config_path) as f:
    model_config = json.load(f)

model = create_model_from_config(model_config)
model_keys = set(model.state_dict().keys())

print(f"Model expects {len(model_keys)} keys")
print(f"\nSample model keys:")
sample_model_keys = list(model_keys)[:5]
for key in sample_model_keys:
    print(f"  - {key}")

# Compare
checkpoint_keys = set(state_dict.keys())
matching_keys = checkpoint_keys & model_keys
only_in_checkpoint = checkpoint_keys - model_keys
only_in_model = model_keys - checkpoint_keys

print(f"\n{'='*70}")
print(f"Comparison:")
print(f"  Keys in checkpoint: {len(checkpoint_keys)}")
print(f"  Keys model expects: {len(model_keys)}")
print(f"  Keys that match: {len(matching_keys)}")
print(f"  Keys only in checkpoint: {len(only_in_checkpoint)}")
print(f"  Keys only in model: {len(only_in_model)}")

# Check for prefix issues
print(f"\n{'='*70}")
print(f"Checking for prefix mismatches...")

# Sample keys from checkpoint that don't match
if only_in_checkpoint:
    print(f"\nSample keys ONLY in checkpoint (first 10):")
    for key in list(only_in_checkpoint)[:10]:
        print(f"  - {key}")

if only_in_model:
    print(f"\nSample keys ONLY in model (first 10):")
    for key in list(only_in_model)[:10]:
        print(f"  - {key}")

# Try stripping prefixes
print(f"\n{'='*70}")
print(f"Trying to strip common prefixes...")

# Common prefixes in PyTorch Lightning: "model.", "diffusion_model.", etc.
prefixes_to_try = ["model.", "diffusion_model.", "unet.", "conditioning.", "pretransform.", ""]

for prefix in prefixes_to_try:
    matching = 0
    for key in checkpoint_keys:
        stripped_key = key
        if stripped_key.startswith(prefix):
            stripped_key = stripped_key[len(prefix):]
        if stripped_key in model_keys:
            matching += 1
    
    if matching > len(matching_keys):
        print(f"✓ Prefix '{prefix}' would match {matching} keys (better than current {len(matching_keys)})")
    elif matching == len(matching_keys):
        print(f"  Prefix '{prefix}' matches {matching} keys (same as current)")
    else:
        print(f"  Prefix '{prefix}' would match {matching} keys (worse than current {len(matching_keys)})")

