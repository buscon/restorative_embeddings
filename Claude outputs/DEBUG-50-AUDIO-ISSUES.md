# Debugging the 50 Audio Issues in ARAUS Export

## Executive Summary

After the fixed export script was deployed, validation still reported **50 audio problems**—the same count as before. This document:

1. **Identifies what those 50 issues are** (analysis of the export pipeline)
2. **Provides tools to pinpoint exactly which files and why** (diagnostic scripts)
3. **Shows how to fix the root cause** (code and workflow adjustments)

---

## Part 1: Why 50 Files Are Still Problematic

### The Export Pipeline

The `02_export.py` script follows this pipeline for each ARAUS stimulus:

```
ARAUS stimulus (soundscape + masker)
    ↓
prepare() → mono 48 kHz, RMS-normalized for CLAP
    ↓
ArausMixer.mix() → mix soundscape + masker at SMR, still 48 kHz mono
    ↓
resample_poly() → 48 kHz → 44.1 kHz (resampling)
    ↓
normalise_loudness() → -23 LUFS (loudness normalization)
    ↓
Convert mono → stereo (duplicate channel)
    ↓
Save as WAV (44.1 kHz, stereo)
```

### Previous Fix (Incomplete)

The fix added validation at each step to raise explicit errors instead of silently producing bad audio:

```python
# After prepare()
if y.size == 0:
    raise ValueError("prepare() returned empty audio")

# After resample_poly()
if y.size == 0:
    raise ValueError("resample_poly() returned empty audio")

# After normalise_loudness()
if y.size == 0:
    raise ValueError("normalise_loudness() returned empty audio")
```

**But this checks only for *empty* audio.** It doesn't check for:
- Audio that is **too short** (truncated during resampling)
- Audio that is **incorrectly resampled** (artifacts, wrong duration)
- Audio that **fails loudness measurement** but doesn't return empty
- Audio that **clips during loudness correction**
- Audio with **NaN/Inf values**

### Likely Root Causes of the 50 Issues

**Hypothesis 1: Resampling bugs (most likely)**
- `scipy.signal.resample_poly()` has edge cases with certain signal lengths
- Some files may be slightly off in duration after resampling
- This affects perhaps **20–30 files**

**Hypothesis 2: Loudness measurement failures**
- For some files, `pyloudnorm.Meter.integrated_loudness()` returns `-np.inf` or `NaN`
- The current fix raises an error (good), but the error may occur for a specific category
- Example: silent maskers or mono/stereo channel mismatches
- This affects perhaps **10–20 files**

**Hypothesis 3: ARAUS data quality**
- Some ARAUS stimuli or maskers are corrupted at the source
- The mixing produces invalid audio (zeros, NaNs, or extreme values)
- This affects perhaps **5–10 files**

**Hypothesis 4: Clipping after loudness normalization**
- For already-loud input, normalization to −23 LUFS may push peaks to > 1.0
- If not clipped, this creates distortion; if clipped, it creates artifacts
- This affects perhaps **3–7 files**

---

## Part 2: Tools to Identify the Exact Issues

### Step 1: Run Enhanced Validation

Use this script to identify which of the 50 files have problems and what **type** of problem each has:

```python
#!/usr/bin/env python3
"""Enhanced validation with detailed diagnostics for each problematic file."""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import librosa
    import numpy as np
    import soundfile as sf
except ImportError:
    print("Install: pip install librosa numpy soundfile")
    sys.exit(1)

def check_audio_file(path, expected_sr=44100, expected_duration_range=(9, 11)):
    """Detailed check of audio file. Returns (is_valid, issues_list)."""
    issues = []
    
    try:
        # File size check
        size_mb = path.stat().st_size / (1024**2)
        if size_mb < 0.01:
            return False, [f"File size suspiciously small: {size_mb:.3f} MB"]
        
        # Load audio
        y, sr = librosa.load(str(path), sr=None, mono=False)
        
        # Basic checks
        if y.size == 0:
            return False, ["Empty audio (0 samples)"]
        
        if sr != expected_sr:
            issues.append(f"Sample rate mismatch: {sr} Hz (expected {expected_sr})")
        
        # Duration check
        if y.ndim == 1:
            n_samples = len(y)
        else:
            n_samples = y.shape[1]  # For stereo
        
        duration = n_samples / sr
        min_dur, max_dur = expected_duration_range
        if duration < min_dur or duration > max_dur:
            issues.append(f"Duration out of range: {duration:.2f}s (expected {min_dur}–{max_dur}s)")
        
        # Audio quality checks
        if np.any(np.isnan(y)):
            issues.append("Contains NaN values")
        if np.any(np.isinf(y)):
            issues.append("Contains Inf values")
        
        # RMS/loudness check
        rms = np.sqrt(np.mean(y**2))
        if rms < 0.0001:
            issues.append(f"Nearly silent: RMS={rms:.2e}")
        elif rms > 0.95:
            issues.append(f"Very loud or clipping: RMS={rms:.4f}")
        
        # Clipping check
        peak = np.abs(y).max()
        if peak > 0.98:
            issues.append(f"Clipping detected: peak={peak:.4f}")
        
        return len(issues) == 0, issues
    
    except Exception as e:
        return False, [f"Error: {str(e)[:80]}"]

# Find and check all audio
audio_dir = Path.home() / "Documents/Bamberg/restorative_embeddings/data/generation/audio"
if not audio_dir.exists():
    print(f"Error: {audio_dir} not found")
    sys.exit(1)

wav_files = sorted(list(audio_dir.glob("*.wav")))
print(f"Checking {len(wav_files)} files...\n")

problematic = []
for i, path in enumerate(wav_files):
    if (i+1) % 500 == 0:
        print(f"Progress: {i+1}/{len(wav_files)}")
    
    is_valid, issues = check_audio_file(path)
    if not is_valid:
        problematic.append((path.name, issues))

# Report
print("\n" + "="*80)
print(f"RESULTS: {len(problematic)}/{len(wav_files)} files are problematic")
print("="*80)

if problematic:
    # Categorize
    categories = {}
    for filename, issues in problematic:
        for issue in issues:
            if "sample rate" in issue:
                cat = "Sample rate mismatch"
            elif "Duration" in issue:
                cat = "Wrong duration"
            elif "silent" in issue:
                cat = "Silent audio"
            elif "NaN" in issue:
                cat = "NaN values"
            elif "Inf" in issue:
                cat = "Inf values"
            elif "Clipping" in issue:
                cat = "Clipping"
            elif "Error" in issue:
                cat = "Load error"
            else:
                cat = "Other"
            
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(filename)
    
    print("\nBy Category:")
    for cat in sorted(categories.keys()):
        names = categories[cat]
        print(f"  {cat}: {len(names)} files")
        for name in sorted(names)[:3]:
            print(f"    • {name}")
        if len(names) > 3:
            print(f"    ... and {len(names)-3} more")
    
    print("\nDetailed List (first 20):")
    for filename, issues in sorted(problematic)[:20]:
        print(f"\n  {filename}")
        for issue in issues:
            print(f"    ⚠ {issue}")

print("\n" + "="*80)
```

### Step 2: Analyze the Failed Stimuli

Once you know which files fail, check if they have a pattern:

```bash
# Run the diagnostic
python3 diagnose-audio-issues.py > audio-issues-report.txt

# Check for patterns in filenames
grep "❌" audio-issues-report.txt | cut -d_ -f2 | sort | uniq -c | sort -rn
# (identifies which ARAUS stimulus IDs are problematic)
```

---

## Part 3: Root Cause Fixes

### Fix 1: Improve Resampling Validation

The resampling step is fragile. Add these checks **before and after**:

```python
def resample_audio_safe(y, sr_orig, sr_target=44100):
    """Resample with validation."""
    from scipy.signal import resample_poly
    
    # Check input
    if y.size == 0:
        raise ValueError("Cannot resample empty audio")
    
    n_samples_orig = len(y)
    expected_duration = n_samples_orig / sr_orig
    
    # Calculate resampling ratio
    gcd_val = np.gcd(int(sr_target), int(sr_orig))
    up, down = int(sr_target) // gcd_val, int(sr_orig) // gcd_val
    
    print(f"  Resampling: {sr_orig}Hz→{sr_target}Hz (ratio {up}:{down})")
    
    # Resample
    y_resampled = resample_poly(y, up, down)
    
    # Validate output
    if y_resampled.size == 0:
        raise ValueError("Resampling produced empty audio")
    
    n_samples_resampled = len(y_resampled)
    duration_resampled = n_samples_resampled / sr_target
    
    # Check if duration matches (within 0.1% tolerance)
    duration_match = abs(duration_resampled - expected_duration) / expected_duration
    if duration_match > 0.001:  # > 0.1% mismatch
        raise ValueError(
            f"Resampling duration mismatch: "
            f"expected {expected_duration:.3f}s, "
            f"got {duration_resampled:.3f}s "
            f"({duration_match*100:.1f}% error)"
        )
    
    return y_resampled, sr_target
```

### Fix 2: Improve Loudness Normalization

The `normalise_loudness()` function should handle edge cases:

```python
def normalise_loudness(y, sr, target_lufs=-23.0):
    """Normalize to target loudness with detailed error handling."""
    try:
        import pyloudnorm
    except ImportError:
        print("WARNING: pyloudnorm not available")
        if y.ndim == 1:
            y = np.stack([y, y], axis=1)
        return y
    
    # Input validation
    if y.size == 0:
        raise ValueError("Cannot normalize empty audio")
    if y.ndim != 1:
        raise ValueError(f"Expected 1D mono input, got shape {y.shape}")
    
    # Convert to stereo
    y_stereo = np.stack([y, y], axis=1).astype(np.float32)
    
    # Measure loudness
    meter = pyloudnorm.Meter(sr)
    loudness = meter.integrated_loudness(y_stereo)
    
    # Validate measurement
    if np.isnan(loudness):
        # Try a different approach: use mono measurement
        try:
            loudness_mono = meter.integrated_loudness(y.astype(np.float32))
            if np.isnan(loudness_mono):
                raise ValueError("Loudness is NaN even with mono")
            loudness = loudness_mono
        except:
            raise ValueError("Cannot measure loudness (audio may be corrupt)")
    
    if loudness == -np.inf:
        raise ValueError("Audio is silent (loudness = -inf)")
    
    if loudness > 0:
        raise ValueError(f"Loudness unusually high: {loudness:.1f} LUFS (clipping?)")
    
    # Normalize
    normalized = pyloudnorm.normalize(meter, y_stereo, target_lufs)
    
    # Validate output
    if normalized.size == 0:
        raise ValueError("Normalization produced empty audio")
    
    # Check for clipping
    peak = np.abs(normalized).max()
    if peak > 1.0:
        print(f"  WARNING: Clipping after normalization (peak={peak:.4f}). Soft-clipping applied.")
        normalized = np.tanh(normalized)  # Soft clip
    
    return normalized
```

### Fix 3: Enhanced Process-Stimulus Validation

Wrap the entire per-stimulus pipeline with comprehensive checks:

```python
def process_stimulus_robust(stim_id, araus, mixer, out_dir, metadata):
    """Process one stimulus with comprehensive error handling."""
    
    try:
        # Step 1: Prepare
        y = prepare(araus[stim_id], sr=48000)
        assert y.size > 0, "prepare() returned empty"
        expected_48k = int(30 * 48000)
        assert len(y) >= expected_48k * 0.95, f"prepare() truncated: {len(y)}/{expected_48k} samples"
        
        # Step 2: Mix
        y = mixer.mix(stim_id, y, masker_sr=48000)
        assert y.size > 0, "mix() returned empty"
        
        # Step 3: Resample
        y, sr = resample_audio_safe(y, 48000, 44100)
        expected_44k = int(30 * 44100)
        assert len(y) >= expected_44k * 0.95, f"resample() truncated: {len(y)}/{expected_44k} samples"
        
        # Step 4: Normalize loudness
        y = normalise_loudness(y, sr, target_lufs=-23.0)
        assert y.size > 0, "normalise_loudness() returned empty"
        assert y.ndim == 2, f"Expected stereo (2D), got {y.ndim}D"
        
        # Step 5: Save and validate
        wav_path = out_dir / f"{stim_id}.wav"
        sf.write(wav_path, y, sr)
        
        # Post-save validation
        assert wav_path.exists(), "WAV file was not created"
        size = wav_path.stat().st_size
        assert size > 10000, f"WAV file suspiciously small: {size} bytes"
        
        # Re-load and verify
        y_check, sr_check = librosa.load(str(wav_path), sr=None, mono=False)
        assert y_check.size > 0, "Re-loaded file is empty"
        assert sr_check == 44100, f"Saved file has wrong SR: {sr_check}"
        
        return {"status": "OK", "stim_id": stim_id, "file": str(wav_path)}
    
    except Exception as e:
        return {"status": "ERROR", "stim_id": stim_id, "error": str(e)}
```

---

## Part 4: Workflow to Fix the 50 Issues

### Option A: Re-export with fixes (Recommended)

1. **Backup current export:**
   ```bash
   cd ~/Documents/Bamberg/restorative_embeddings
   mv data/generation/audio data/generation/audio.bak
   ```

2. **Update 02_export.py** with the fixes above (Fixes 1–3)

3. **Re-run export with diagnostics:**
   ```bash
   python generation/02_export.py \
     --araus data/raw/araus \
     --processed data/processed \
     --captions data/generation/captions.csv \
     --out data/generation \
     --n-samples 6000 \
     --workers 8 2>&1 | tee export.log
   ```

4. **Check for errors in log:**
   ```bash
   grep -E "ERROR|Error|error|Exception" export.log | head -20
   ```
   This tells you which stimuli fail and why.

5. **Run validation:**
   ```bash
   python "Claude outputs/validate-araus-dataset.py"
   ```

### Option B: Fix existing export (If data must be preserved)

1. **Identify problematic files:**
   ```bash
   python3 diagnose-audio-issues.py > issues.txt
   ```

2. **For each category of issue, apply targeted fixes:**
   - **Wrong duration:** Re-export only those stimuli
   - **Silent audio:** Check if masker is "silence" type (filter it out)
   - **NaN/Clipping:** Apply soft-clipping or re-normalize

3. **Create a patch script:**
   ```python
   # For each problematic stimulus ID:
   # 1. Re-process just that one stimulus
   # 2. Overwrite the corrupted WAV with the fixed version
   ```

---

## Part 5: Expected Outcomes

### If 50 issues are fixed:

```
✅ Re-export complete: 6000 files processed
✅ Validation passed: 6000 valid audio-caption pairs
✅ Ready for training dataset preparation
```

### If issues persist:

1. **Same 50 files fail again** → Root cause is in ARAUS data itself
   - Action: Filter out those 50 problematic stimuli (use 5950 instead of 6000)

2. **Different files fail** → Root cause is non-deterministic (e.g., random seed, threading)
   - Action: Use single-worker mode (`--workers 1`) to debug

3. **Fewer than 50 fail** → Fixes worked partially
   - Action: Apply remaining fixes and re-export

---

## Next Steps

1. **Run the enhanced diagnostic script** on your current export to identify the exact 50 files and their issues
2. **Apply Fix 1, 2, and 3** to `02_export.py`
3. **Re-export** with the fixed script
4. **Validate** and confirm all 6000 files are good
5. **Proceed to training dataset preparation** (`02_prepare_training_data_sa3.py`)

The root cause is almost certainly in the resampling or loudness normalization steps. Once we pinpoint which of the 50 files have which problem, the fix is straightforward.

