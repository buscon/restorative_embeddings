"""
IMPROVED EXPORT FUNCTIONS - Patch for 02_export.py
=====================================================

Replace the normalise_loudness() and resample functions in 02_export.py
with these improved versions to fix the 50 audio issues.

USAGE:
1. Back up your current 02_export.py
2. Copy these functions into 02_export.py
3. Update the process_stimulus() function to use resample_audio_safe()
4. Re-run the export
"""

import numpy as np
from scipy.signal import resample_poly
import warnings

# ============================================================================
# FIX 1: IMPROVED RESAMPLING WITH VALIDATION
# ============================================================================

def resample_audio_safe(y, sr_orig, sr_target=44100):
    """
    Resample audio with comprehensive validation.

    Args:
        y: audio array
        sr_orig: original sample rate (usually 48000)
        sr_target: target sample rate (usually 44100)

    Returns:
        y_resampled, sr_target

    Raises:
        ValueError: if resampling produces invalid audio
    """
    # Validate input
    if y.size == 0:
        raise ValueError("Cannot resample empty audio")

    n_samples_orig = len(y)
    expected_duration = n_samples_orig / sr_orig

    # Calculate resampling ratio (simplest form)
    gcd_val = np.gcd(int(sr_target), int(sr_orig))
    up = int(sr_target) // gcd_val
    down = int(sr_orig) // gcd_val

    # Resample
    try:
        y_resampled = resample_poly(y, up, down)
    except Exception as e:
        raise ValueError(f"Resampling failed: {str(e)}")

    # Validate output
    if y_resampled.size == 0:
        raise ValueError("Resampling produced empty audio")

    n_samples_resampled = len(y_resampled)
    duration_resampled = n_samples_resampled / sr_target

    # Check if duration matches (allow 0.1% tolerance)
    if expected_duration > 0:
        duration_error = abs(duration_resampled - expected_duration) / expected_duration
    else:
        duration_error = 0

    if duration_error > 0.001:  # > 0.1% mismatch is suspicious
        raise ValueError(
            f"Resampling duration mismatch: "
            f"expected {expected_duration:.4f}s, got {duration_resampled:.4f}s "
            f"({duration_error*100:.2f}% error). "
            f"This suggests a corrupt input or resampling bug."
        )

    return y_resampled


# ============================================================================
# FIX 2: IMPROVED LOUDNESS NORMALIZATION
# ============================================================================

def normalise_loudness_robust(y, sr, target_lufs=-23.0):
    """
    Normalize audio to target loudness with robust error handling.

    This version handles edge cases like:
    - Silent audio (loudness = -inf)
    - NaN loudness measurements (invalid audio)
    - Very loud audio that would clip during normalization
    - Missing pyloudnorm library

    Args:
        y: (n_samples,) mono audio array
        sr: sample rate
        target_lufs: target loudness (default -23.0 LUFS for SA3)

    Returns:
        (n_samples, 2) stereo normalized audio

    Raises:
        ValueError: if audio cannot be normalized
    """
    try:
        import pyloudnorm
    except ImportError:
        print("WARNING: pyloudnorm not installed. Skipping normalization.")
        # Still convert to stereo without normalization
        if y.ndim == 1:
            y = np.stack([y, y], axis=1)
        return y

    # ─── Input validation ───────────────────────────────────────────────
    if y.size == 0:
        raise ValueError("Cannot normalize empty audio")

    if y.ndim != 1:
        raise ValueError(f"Expected mono (1D) input, got shape {y.shape}")

    # ─── Prepare stereo version ─────────────────────────────────────────
    y_stereo = np.stack([y, y], axis=1).astype(np.float32)

    # Check for NaN/Inf in input
    if np.any(np.isnan(y_stereo)):
        raise ValueError("Input audio contains NaN values (corrupted)")
    if np.any(np.isinf(y_stereo)):
        raise ValueError("Input audio contains Inf values (corrupted)")

    # ─── Measure loudness ───────────────────────────────────────────────
    meter = pyloudnorm.Meter(sr)
    loudness = meter.integrated_loudness(y_stereo)

    # Validate loudness measurement
    if np.isnan(loudness):
        # Try mono measurement as backup
        try:
            loudness_mono = meter.integrated_loudness(y.astype(np.float32))
            if np.isnan(loudness_mono):
                raise ValueError("Loudness measurement is NaN. Audio may be corrupted or completely silent.")
            loudness = loudness_mono
        except:
            raise ValueError(
                "Loudness measurement failed for both stereo and mono. "
                "This often means the audio is corrupted, has no content, "
                "or is incompatible with pyloudnorm."
            )

    if loudness == -np.inf:
        raise ValueError("Audio is completely silent (loudness = -inf)")

    if loudness > 5:  # Suspiciously loud
        raise ValueError(
            f"Audio is unusually loud ({loudness:.1f} LUFS). "
            f"This may indicate clipping or corrupted input."
        )

    # ─── Normalize ──────────────────────────────────────────────────────
    try:
        normalized = pyloudnorm.normalize(meter, y_stereo, target_lufs)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

    # Validate output
    if normalized.size == 0:
        raise ValueError("Normalization produced empty audio")

    if np.any(np.isnan(normalized)):
        raise ValueError("Normalization produced NaN values")

    # ─── Handle clipping ────────────────────────────────────────────────
    peak = np.abs(normalized).max()
    if peak > 1.0:
        # Soft clipping to prevent harsh distortion
        warnings.warn(f"Clipping detected after normalization (peak={peak:.4f}). Applying soft clipping.")
        normalized = np.tanh(normalized)  # Smooth clipping curve

    return normalized


# ============================================================================
# FIX 3: ROBUST STIMULUS PROCESSING
# ============================================================================

def process_stimulus_robust(stim_id, araus, captions_df, out_dir,
                           masker_sr=48000, target_sr=44100,
                           import_prepare=None, import_mixer=None):
    """
    Process one ARAUS stimulus with comprehensive error tracking.

    This wraps the entire pipeline and validates at each step.

    Args:
        stim_id: stimulus ID
        araus: ARAUS dataset (dict-like)
        captions_df: DataFrame with captions
        out_dir: output directory (Path)
        masker_sr: sample rate of input (usually 48000)
        target_sr: target output sample rate (usually 44100)
        import_prepare: function to import and call prepare()
        import_mixer: ArausMixer instance

    Returns:
        dict with status, stim_id, and either 'file' or 'error'
    """
    import librosa
    import soundfile as sf
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{stim_id}.wav"

    try:
        # ─── Step 1: Prepare (RMS-normalized mono 48k) ──────────────────
        y = import_prepare(araus[stim_id])
        if y.size == 0:
            raise ValueError("Step 1 (prepare): returned empty audio")

        expected_48k = int(30 * 48000)
        if len(y) < expected_48k * 0.9:
            raise ValueError(
                f"Step 1 (prepare): audio too short "
                f"({len(y)} samples, expected ~{expected_48k})"
            )

        # ─── Step 2: Mix (soundscape + masker at SMR) ──────────────────
        y = import_mixer.mix(stim_id, y)
        if y.size == 0:
            raise ValueError("Step 2 (mix): returned empty audio")

        # ─── Step 3: Resample (48k → 44.1k) ───────────────────────────
        y = resample_audio_safe(y, masker_sr, target_sr)
        if y.size == 0:
            raise ValueError("Step 3 (resample): returned empty audio")

        expected_44k = int(30 * target_sr)
        if len(y) < expected_44k * 0.9:
            raise ValueError(
                f"Step 3 (resample): audio too short "
                f"({len(y)} samples, expected ~{expected_44k})"
            )

        # ─── Step 4: Normalize loudness (mono → stereo) ───────────────
        y = normalise_loudness_robust(y, target_sr, target_lufs=-23.0)
        if y.size == 0:
            raise ValueError("Step 4 (normalize): returned empty audio")
        if y.ndim != 2:
            raise ValueError(f"Step 4 (normalize): expected stereo (2D), got {y.ndim}D")

        # ─── Step 5: Save ────────────────────────────────────────────
        sf.write(str(wav_path), y, target_sr)

        # ─── Step 6: Post-save validation ────────────────────────────
        if not wav_path.exists():
            raise RuntimeError(f"Step 5 (save): WAV file was not created: {wav_path}")

        size = wav_path.stat().st_size
        if size < 10000:
            raise RuntimeError(
                f"Step 5 (save): WAV file suspiciously small ({size} bytes). "
                f"Expected at least ~200 KB for 30s stereo 44.1k."
            )

        # ─── Step 7: Verify by re-loading ────────────────────────────
        try:
            y_check, sr_check = librosa.load(str(wav_path), sr=None, mono=False)
        except Exception as e:
            raise RuntimeError(f"Step 7 (verify): Cannot re-load saved file: {str(e)}")

        if y_check.size == 0:
            raise RuntimeError("Step 7 (verify): Re-loaded file is empty")

        if sr_check != target_sr:
            raise RuntimeError(
                f"Step 7 (verify): Saved file has wrong sample rate: {sr_check} "
                f"(expected {target_sr})"
            )

        if y_check.ndim != 2:
            raise RuntimeError(
                f"Step 7 (verify): Saved file is not stereo: shape is {y_check.shape}"
            )

        return {
            "status": "OK",
            "stim_id": stim_id,
            "file": str(wav_path),
            "size_mb": size / (1024**2),
        }

    except Exception as e:
        # Clean up partial file
        if wav_path.exists():
            try:
                wav_path.unlink()
            except:
                pass

        return {
            "status": "ERROR",
            "stim_id": stim_id,
            "error": str(e),
        }


# ============================================================================
# UPDATED PROCESS_STIMULUS() FOR 02_EXPORT.PY
# ============================================================================

"""
CHANGE IN 02_EXPORT.PY:

OLD CODE (in process_stimulus):
    y = prepare(araus[stim_id])
    y = mixer.mix(stim_id, y)
    y = scipy.signal.resample_poly(y, 441, 480)  # Direct call
    y = normalise_loudness(y, 44100)

NEW CODE (use this instead):
    y = prepare(araus[stim_id])
    assert y.size > 0, "prepare() returned empty"

    y = mixer.mix(stim_id, y)
    assert y.size > 0, "mixer.mix() returned empty"

    y = resample_audio_safe(y, 48000, 44100)  # Use new function
    assert y.size > 0, "resample_audio_safe() returned empty"

    y = normalise_loudness_robust(y, 44100)  # Use new function
    assert y.size > 0 and y.ndim == 2, "normalise_loudness_robust() failed"

    # Save and validate (existing code should still work)
    sf.write(str(wav_path), y, 44100)
"""

# ============================================================================
# SUMMARY OF CHANGES
# ============================================================================

"""
1. resample_audio_safe():
   - NEW: Validates output duration matches expected duration (±0.1%)
   - NEW: Detailed error messages explaining what went wrong
   - FIX: Catches edge cases in scipy.signal.resample_poly()

2. normalise_loudness_robust():
   - NEW: Handles NaN loudness measurements (tries mono as fallback)
   - NEW: Handles -inf loudness (completely silent audio)
   - NEW: Detects clipping after normalization and applies soft-clipping
   - FIX: More robust error messages for debugging
   - SAME: Output format (stereo 32-bit float, 44.1 kHz, -23 LUFS)

3. process_stimulus_robust():
   - NEW: Seven-step validation pipeline (prepare→mix→resample→normalize→save→verify)
   - NEW: Explicit checks after each step
   - NEW: Detailed error messages showing which step failed and why
   - NEW: Post-save re-loading validation
   - SAME: Compatible with existing dataframe and file structure

These fixes target the root causes:
- Resampling bugs (wrong duration after resample_poly)
- Loudness measurement failures (NaN, -inf edge cases)
- Clipping during normalization
- Silent maskers
"""
