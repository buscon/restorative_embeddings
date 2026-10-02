# Quick Fix Checklist: 50 Audio Issues

## What Happened
✅ The export script was fixed for validation  
⚠️ But validation still found **50 problematic audio files**  
❌ This blocks training dataset preparation  

---

## The Problem (In Plain English)

The export pipeline has **3 fragile steps** that can silently produce bad audio:

1. **Resampling** (48 kHz → 44.1 kHz): Can truncate or corrupt audio
2. **Loudness normalization** (to -23 LUFS): Can fail on edge cases (silent, already-loud input)
3. **Mono-to-stereo conversion**: Can fail with NaN/Inf values

The previous fix added validation for **empty audio**, but not for:
- Audio that is the **wrong duration**
- Audio with **NaN/Inf values**
- Audio that **clips during normalization**

So 50 files still slip through with corrupted content.

---

## Fix: 4 Steps (45 minutes)

### Step 1: Backup Current Export (2 min)
```bash
cd ~/Documents/Bamberg/restorative_embeddings
cp -r data/generation/audio data/generation/audio.bak
# ✅ Now you have a safe copy
```

### Step 2: Update Export Script (10 min)

In your `generation/02_export.py`:

**Find these lines** (around line 50–100):
```python
def normalise_loudness(y, sr, target_lufs=-23.0):
    """Normalise audio to target loudness using pyloudnorm."""
```

**Replace with** the `normalise_loudness_robust()` function from **IMPROVED-EXPORT-FUNCTIONS.py**

**Find** the direct resampling call (looks like):
```python
y = scipy.signal.resample_poly(y, 441, 480)
```

**Replace with**:
```python
from improved_export_functions import resample_audio_safe
y = resample_audio_safe(y, 48000, 44100)
```

**Add these imports at the top**:
```python
import numpy as np
from scipy.signal import resample_poly
import warnings
```

**Save the file.**

### Step 3: Re-export (20 min)
```bash
cd ~/Documents/Bamberg/restorative_embeddings
python generation/02_export.py \
  --araus data/raw/araus \
  --processed data/processed \
  --captions data/generation/captions.csv \
  --out data/generation \
  --n-samples 6000 \
  --workers 8 2>&1 | tee export-fixed.log

# Watch for errors:
tail -50 export-fixed.log | grep -i error
```

### Step 4: Validate (10 min)
```bash
python "Claude outputs/validate-araus-dataset.py"
```

Expected output:
```
✅ Dataset validation PASSED
Total valid audio-caption pairs: 6000
All sampled files: OK
```

---

## If Validation Still Fails

### Check the error log:
```bash
grep "ERROR" export-fixed.log | head -20
```

**Common outputs and fixes:**

| Log message | Cause | Fix |
|---|---|---|
| `Resampling duration mismatch` | Input audio is truncated | Re-export with `--workers 1` |
| `Loudness is NaN` | Silent masker | Filter silence maskers (edit captions script) |
| `Clipping detected` | Input is already loud | Reduce master volume before normalization |
| `Cannot load file` | File is corrupted | This stimulus may be bad at source (skip it) |

### Identify exactly which 50 files are bad:
```bash
python3 << 'EOF'
import librosa
from pathlib import Path

audio_dir = Path.home() / "Documents/Bamberg/restorative_embeddings/data/generation/audio"
problem_files = []

for wav in sorted(audio_dir.glob("*.wav"))[:100]:  # Check first 100
    try:
        y, sr = librosa.load(str(wav), sr=None, mono=False)
        if y.size == 0 or sr != 44100 or y.ndim != 2:
            problem_files.append(wav.name)
    except:
        problem_files.append(wav.name)

print(f"Problems: {len(problem_files)} files")
for f in sorted(problem_files)[:20]:
    print(f"  {f}")
EOF
```

---

## Expected Timeline

| Phase | Time | Status |
|---|---|---|
| **Today: Fix export script** | 45 min | ← You are here |
| **Today: Validate** | 10 min | Then ↓ |
| **Tomorrow: Prepare training dataset** | 5 min | `python generation/train/02_prepare_training_data_sa3.py` |
| **Tomorrow: Run training** | 8–12 hours | On RTX 5090 |
| **Next day: Evaluate** | 1–2 hours | Check if pleasantness effect works |

---

## Troubleshooting

**Q: Export still fails after fix?**  
A: Run with `--workers 1` to remove threading parallelism (some edge cases only appear with workers):
```bash
python generation/02_export.py ... --workers 1
```

**Q: Validation passes but training fails later?**  
A: Validation checks *file structure*; training checks *content quality*. Re-run validation with stronger checks:
```bash
python "Claude outputs/validate-araus-dataset.py" --strict-audio-analysis
```

**Q: How do I know if the fix actually worked?**  
A: Compare logs:
```bash
grep "ERROR" export.log | wc -l  # Old: should be > 50
grep "ERROR" export-fixed.log | wc -l  # New: should be 0
```

---

## Files Provided

1. **DEBUG-50-AUDIO-ISSUES.md** → Full technical analysis (read if you want to understand what went wrong)
2. **IMPROVED-EXPORT-FUNCTIONS.py** → Ready-to-use replacement functions (copy these into 02_export.py)
3. **This file** → Quick checklist (you are here)

---

## Next Steps After Fix

Once validation passes with 6000 files:

```bash
# 1. Prepare training dataset (organize by pleasantness bins)
python generation/train/02_prepare_training_data_sa3.py \
  --metadata data/generation/export_metadata.csv \
  --audio-dir data/generation/audio \
  --output generation/train/araus_for_sa3

# 2. Run training on RTX 5090 (takes 8–12 hours)
python generation/train/03_train_lora_sa3.py

# 3. Evaluate pleasantness effect
python generation/train/04_generate_and_evaluate.py
```

---

## Questions?

If this doesn't work:
1. Save the `export-fixed.log` file
2. Check which specific errors appear
3. Run the diagnostic script from DEBUG-50-AUDIO-ISSUES.md to identify the exact 50 files
4. That will tell us what's still broken

Good luck! 🎵
