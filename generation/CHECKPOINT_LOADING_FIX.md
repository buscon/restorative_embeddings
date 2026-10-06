# Checkpoint Loading Fix - Root Cause & Solution

## Problem Summary
The `generate_with_ckpt.py` script is failing to load model weights, causing pure noise generation. **Every single model weight is being skipped** during checkpoint loading with messages like:

```
Key model.timestep_features.weight not found in target state_dict or shape mismatch. Skipping.
Key model.to_timestep_embed.0.weight not found in target state_dict or shape mismatch. Skipping.
```

This happens for:
- All 24 transformer layers
- All embeddings (timestep, conditioning, global)
- All attention and feedforward modules
- All projection and normalization layers

**Result:** The model runs with uninitialized random weights → guaranteed to produce noise.

## Root Cause Analysis

The issue is in `generate_with_ckpt.py`'s checkpoint loading logic:

```python
# Current broken code
if key.startswith('model.'):
    new_key = key[6:]  # Remove 'model.' prefix
    cleaned_state_dict[new_key] = value
else:
    cleaned_state_dict[key] = value
```

This assumes that removing "model." from checkpoint keys will match the model's expected keys. However:
- **Checkpoint format** (PyTorch Lightning .ckpt): Has keys like `model.timestep_features.weight`
- **Model expects**: Keys like `timestep_features.weight` (or different format entirely)
- **After stripping**: Still doesn't match! → All weights skipped

## Solution Options (Ranked by Reliability)

### Option 1: Use SafeTensors Format (RECOMMENDED) ⭐

The base model is available in safetensors format, which is simpler and more reliable:

```bash
# Try this first
python3 generate_with_ckpt_fixed.py \
  --ckpt ~/stableaudio/models/stabilityai__stable-audio-open-1.0/model.safetensors \
  --config ~/stableaudio/models/stabilityai__stable-audio-open-1.0/model_config.json \
  --prompt "urban park with distant traffic"
```

**Why this works:**
- SafeTensors format doesn't have PyTorch Lightning wrapper complexity
- Keys match the model directly without transformation
- More transparent and debuggable format
- Used by HuggingFace as standard for model distribution

### Option 2: Use Official stable-audio-tools API

```bash
python3 generate_official_api.py \
  --ckpt ~/stableaudio/models/stabilityai__stable-audio-open-1.0/model.safetensors \
  --config ~/stableaudio/models/stabilityai__stable-audio-open-1.0/model_config.json \
  --prompt "urban park with distant traffic"
```

**Why this works:**
- Uses the library's intended inference path
- Handles all checkpoint format variations
- Properly integrates with the library's generation pipeline
- Less prone to future breakage

### Option 3: Diagnose and Fix Current Script

Run the diagnostic first to understand the exact mismatch:

```bash
python3 diagnose_checkpoint_format.py
```

This will tell you:
- What keys the model actually expects
- What keys are in the checkpoint
- Whether simple prefix removal works
- Whether there's a deeper format incompatibility

## Quick Start - Testing the Fix

### Step 1: Run Diagnostic (5 minutes)
```bash
python3 diagnose_checkpoint_format.py
```

This will show you exactly why weights are being skipped.

### Step 2: Test with SafeTensors (5-10 minutes)
```bash
python3 generate_with_ckpt_fixed.py \
  --ckpt ~/stableaudio/models/stabilityai__stable-audio-open-1.0/model.safetensors \
  --config ~/stableaudio/models/stabilityai__stable-audio-open-1.0/model_config.json \
  --prompt "birds chirping in a garden" \
  --seconds 10
```

### Step 3: Check Output Quality
- Listen to the generated audio
- Compare to the pure noise from before
- If it's intelligible audio → the checkpoint loading is fixed
- If still noise → proceed to Step 4

### Step 4: Debug with Diagnostic Script
If Step 2 fails, run diagnostic to understand the issue:
```bash
python3 diagnose_checkpoint_format.py
```

## Key Differences in the Scripts

### `generate_with_ckpt.py` (Broken)
- ❌ Hard-coded key prefix stripping (`key[6:]`)
- ❌ No validation that weights actually loaded
- ❌ Doesn't attempt safetensors format
- ❌ Only supports .ckpt format

### `generate_with_ckpt_fixed.py` (Fixed)
- ✓ Tries safetensors format first
- ✓ Falls back to PyTorch Lightning with better detection
- ✓ Validates that meaningful weights loaded (80%+ threshold)
- ✓ Shows clear diagnostics of what's happening
- ✓ Detects when checkpoint loading fails critically

### `generate_official_api.py` (Best Practice)
- ✓ Uses stable-audio-tools' official inference API
- ✓ Handles all checkpoint formats automatically
- ✓ Properly integrates with the library's generation pipeline
- ✓ Least likely to break with library updates

## For Fine-Tuned Checkpoints

The same issue likely affects your fine-tuned checkpoint at:
```
~/Documents/restorative_embeddings/generation/train/checkpoints/restorative_soundscapes_finetuning_360/cwafugzf/checkpoints/epoch=22-step=1000.ckpt
```

After fixing the base model generation:

```bash
python3 generate_with_ckpt_fixed.py \
  --ckpt ~/Documents/restorative_embeddings/generation/train/checkpoints/restorative_soundscapes_finetuning_360/cwafugzf/checkpoints/epoch=22-step=1000.ckpt \
  --config ~/Documents/Bamberg/restorative_embeddings/generation/train/model_config.json \
  --prompt "peaceful garden with water fountain"
```

## Important Notes

1. **The training may be working fine** - The noise generation doesn't prove training is broken, just that inference is broken
2. **Test base model first** - If the fixed script produces quality audio from the base model, then training improvements will be testable
3. **Listen for artifacts vs noise** - If output has any structure/patterns instead of pure white noise, weights are loading
4. **Check console output** - Look for lines showing how many weights loaded (e.g., "Loaded 840/876 weights")

## Next Steps

1. Run `diagnose_checkpoint_format.py` to understand the exact issue
2. Try `generate_with_ckpt_fixed.py` with safetensors format
3. If that works → test fine-tuned checkpoint
4. If fine-tuned still broken → use official API approach
5. Once generation works → you can validate that fine-tuning actually improved audio quality

---

**Bottom Line:** The pure noise you're getting is almost certainly because the model weights aren't loading, not because training failed. The fixed scripts should resolve this.
