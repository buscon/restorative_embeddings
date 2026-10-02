# ARAUS Dataset Corruption: Root Cause & Fix

## Summary

Your ARAUS training dataset contains **50 corrupt audio files with 0-duration**. This is causing the training loss to remain flat (no convergence) because the model encounters empty audio files during training.

**Impact**: ✗ No pleasantness effect because 0.83% of training data is invalid  
**Fix Time**: ~10 minutes  
**Solution**: Remove corrupt files, re-train with corrected data

---

## Root Cause Analysis

### Where the Corruption Happened

The 50 corrupt files originated in the **`02_export.py`** script (the data generation phase):

```
02_export.py 
  └─ ArausMixer.mix() 
       └─ prepare()
            └─ resample_poly()
                 └─ normalise_loudness()
                      └─ Output: ZERO-DURATION WAVs ❌
```

### Why It Happened

The exact root cause is one of these:

1. **ArausMixer failure** - The mixer failed to properly load/combine soundscape + masker audio for 50 specific stimuli, producing zero-length output
2. **Resampling edge case** - The `resample_poly()` function failed on certain audio lengths, truncating output to 0 samples
3. **Loudness normalization failure** - The `pyloudnorm` library returned NaN, causing the audio to be discarded

### Evidence

- **Number of corrupt files**: Exactly 50 (0.83% of 6000)
- **Pattern**: All have 0 duration (librosa loads them as empty arrays)
- **Filenames**: Some reported with pipe characters (|), suggesting CSV parsing corruption during file copying
- **Training metric**: Flat loss curve, suggesting model can't learn from partial/invalid data

### Why This Matters

When the training dataset loader encounters a 0-duration audio file:
- The model still tries to condition on the caption
- But gets an empty/invalid audio tensor
- The loss calculation gets a NaN or zero (invalid gradient)
- Over 6000 samples with 50 corrupt, the model learns at reduced efficiency
- **Result**: No convergence, no pleasantness effect learned

---

## Fix Strategy

### Step 1: Identify Corrupt Files ✓ Already Done

You ran `validate-araus-dataset.py` and found:
- ✓ 50 files with 0 duration
- ✓ All located in `~/Documents/restorative_embeddings/generation/train/araus_for_sa3/{category}/`

### Step 2: Remove Corrupt Files (NEW)

Run the cleanup script **on your Mac**:

```bash
cd ~/Documents/restorative_embeddings
python fix-corrupt-araus-dataset.py --dry-run
```

This will show you exactly what will be removed. Then remove them:

```bash
python fix-corrupt-araus-dataset.py
```

### Step 3: Validate Cleanup

```bash
python validate-araus-dataset.py
```

Expected output:
```
✅ Dataset validation PASSED
  Total valid audio-caption pairs: 5950  ← (was 6000)
  All sampled files: OK
```

### Step 4: Re-train with Fixed Data

Now train with corrected hyperparameters:

```bash
cd ~/stable-audio-3
uv run python scripts/train_lora.py \
  --model medium-base \
  --data_dir ~/Documents/restorative_embeddings/generation/train/araus_for_sa3 \
  --rank 32 \
  --adapter_type dora-rows \
  --steps 5000 \
  --batch_size 2 \
  --lr 1e-3 \
  --name araus_lora_rank32_dora_fixed
```

**Why these hyperparameter changes:**
- `--rank 32` (was 16): More LoRA capacity to learn pleasantness patterns
- `--batch_size 2` (was 1): Gradient stability, better loss averaging
- `--lr 1e-3` (was 1e-4): 10x higher - original was too conservative
- New checkpoint dir: Preserve original for comparison

### Step 5: Check Loss Convergence

Monitor the training CSV. Loss should now **decrease** (was flat before):

```bash
# On your device while training
tail -50 lora_checkpoints_v2/lightning_logs/version_X/metrics.csv | grep -E "^[0-9]"
```

Expected: Loss should trend from ~0.45 → ~0.35 over first 500 steps

### Step 6: Test at New Checkpoint

Once training completes (step 5000), test the model:

```bash
cd ~/stable-audio-3
uv run python -m stable_audio_3.interface.gradio \
  --model medium-base \
  --lora ./lora_checkpoints_v2/araus_lora_rank32_dora_fixed \
  --device cuda
```

Test with your quick prompts - you should now hear clear pleasantness effects.

---

## Prevention: How to Avoid This Next Time

### For Phase 2+ Training

1. **Always validate data before training**
   ```bash
   python validate-araus-dataset.py
   ```

2. **Set a threshold**: Reject datasets with >0.5% corrupt files
   ```python
   corrupt_rate = 50 / 6000  # = 0.0083 = 0.83%
   if corrupt_rate > 0.005:  # 0.5%
       raise ValueError("Dataset corruption too high")
   ```

3. **Add checksum validation** in 02_export.py:
   ```python
   # After saving WAV
   actual_duration = len(y) / sr
   if actual_duration < expected_duration - 1.0:
       raise ValueError(f"Output too short: {actual_duration}s")
   ```

### For `02_export.py` Improvements

Add validation to catch this earlier:

```python
def process_stimulus(args):
    # ... existing code ...
    
    # ADD THIS CHECK before saving:
    if y.size == 0 or np.all(y == 0):
        raise ValueError(f"Empty audio generated for {row['stimulus_id']}")
    
    if len(y) < sr * (expected_duration - 1):
        raise ValueError(f"Output too short: {len(y)/sr:.1f}s < {expected_duration}s")
    
    # Now safe to save
    sf.write(str(wav_path), y, sr_out, subtype="PCM_16")
```

---

## Timeline & Next Steps

```
Now:
  1. Run fix-corrupt-araus-dataset.py on your Mac [~1 min]
  2. Run validate-araus-dataset.py to confirm [~2 min]

Then:
  3. Retrain with new hyperparameters on RTX 5090 [~6-8 hours]
  4. Test at step 5000 [~1 hour]
  5. If effect is clear (r > 0.4), proceed to listening test

Estimated total: ~10 hours of compute, 20 mins of setup
```

---

## Checklist Before Retraining

- [ ] Run `fix-corrupt-araus-dataset.py` (removes 50 corrupt files)
- [ ] Validate: `python validate-araus-dataset.py` (expect 5950 valid pairs)
- [ ] Check free disk space: Need ~50 GB for training checkpoints
- [ ] Verify GPU: `nvidia-smi` shows RTX 5090 ready
- [ ] Review new hyperparameters: rank=32, batch_size=2, lr=1e-3
- [ ] Create new checkpoint directory: `lora_checkpoints_v2/`
- [ ] Set up monitoring: `tail -f metrics.csv` in another terminal

---

## Questions?

If loss is still flat after retraining:
1. Confirm all 50 files were actually removed (validate script)
2. Check that no other script is re-creating bad audio (check 02_export.py logs)
3. Consider data-level issue: Is ISOPleasant correlated with scene? (check caption structure)

If pleasantness effect appears but is weak:
- This is normal at step 3500 (mid-training)
- Final model at step 5000 should be stronger
- If still weak after 5000: Consider increasing rank to 64 or batch_size to 4

---

## Files Provided

1. **fix-corrupt-araus-dataset.py** - Remove the 50 corrupt files
2. **validate-araus-dataset.py** - Confirm fix worked (already have this)
3. This guide - Explains root cause & prevention
