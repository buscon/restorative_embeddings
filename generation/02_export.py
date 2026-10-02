"""Step 2 - Export stratified 6k ARAUS subset with captions and loudness normalisation.

Samples 6,000 stimuli from ARAUS, stratified by pleasantness bin and masker type,
to ensure balanced acoustic and perceptual diversity. Audio is:

* Rebuilt from soundscape + masker + SMR using ArausMixer (no disk reads of 132 GB)
* Converted to stereo at 44.1 kHz
* Normalised to −23 LUFS (loudness standard for AI training audio)
* Saved as 16-bit WAV

Output structure:
    data/generation/
    ├── audio/
    │   ├── araus_0000000.wav   (6k WAVs, stratified sample)
    │   └── ...
    └── export_metadata.csv     (id, stimulus_id, caption, ISOPleasant, bin, LA50, wav_path)

Requires: soundfile, pyloudnorm, rsd.data, rsd.audio, generation.01_captions

    python generation/02_export.py --araus <path> --processed data/processed --captions data/generation/captions.csv --out data/generation
"""

import argparse
import sys
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import gcd

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsd.audio import ArausMixer, prepare  # noqa: E402
from rsd.data import load_araus  # noqa: E402


def stratified_sample(stim, n_samples=6000, seed=42):
    """Sample stimuli stratified by pleasantness bin and masker type.

    Ensures each bin × masker combination is represented proportionally, with
    oversampling of rare combinations if needed.
    """
    rng = np.random.default_rng(seed)
    strata = stim.groupby(["bin", "masker_type"]).size()

    # Target counts per stratum (proportional sampling)
    target_per_stratum = strata / strata.sum() * n_samples

    sampled = []
    for (bin_name, masker), target_count in target_per_stratum.items():
        subset = stim[(stim.bin == bin_name) & (stim.masker_type == masker)]
        n = min(int(np.round(target_count)), len(subset))
        if n > 0:
            sampled.append(subset.sample(n=n, random_state=rng))

    result = pd.concat(sampled, ignore_index=False).sample(frac=1, random_state=rng)
    return result.reset_index(drop=True)


def resample_audio_safe(y, sr_orig, sr_target=44100):
    """Resample audio with comprehensive validation.
    
    Args:
        y: audio array
        sr_orig: original sample rate (usually 48000)
        sr_target: target sample rate (usually 44100)
    
    Returns:
        y_resampled
    
    Raises:
        ValueError: if resampling produces invalid audio
    """
    # Validate input
    if y.size == 0:
        raise ValueError("Cannot resample empty audio")
    
    n_samples_orig = len(y)
    expected_duration = n_samples_orig / sr_orig
    
    # Calculate resampling ratio (simplest form)
    gcd_val = gcd(int(sr_target), int(sr_orig))
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
            f"({duration_error*100:.2f}% error)"
        )
    
    return y_resampled


def normalise_loudness(y, sr, target_lufs=-23.0):
    """Normalise audio to target loudness using pyloudnorm.

    Robust version with edge-case handling for:
    - Silent audio (loudness = -inf)
    - NaN loudness measurements (invalid audio)
    - Clipping after normalization

    Args:
        y: (n_samples,) mono audio array (will be converted to stereo)
        sr: sample rate
        target_lufs: target loudness in LUFS

    Returns:
        (n_samples, 2) stereo normalised audio

    Raises:
        ValueError: if audio cannot be normalized
    """
    try:
        import pyloudnorm
    except ImportError:
        print("WARNING: pyloudnorm not installed, skipping loudness normalisation")
        # Still convert to stereo
        if y.ndim == 1:
            y = np.stack([y, y], axis=1)
        return y

    # Validate input
    if y.size == 0:
        raise ValueError("Input audio is empty (0 samples)")
    
    if y.ndim != 1:
        raise ValueError(f"Expected mono audio (1D), got shape {y.shape}")

    # Convert mono to stereo early
    y_stereo = np.stack([y, y], axis=1)

    # pyloudnorm expects float32 or float64 in [-1, 1]
    y_stereo = y_stereo.astype(np.float32)

    # Check for NaN/Inf in input
    if np.any(np.isnan(y_stereo)):
        raise ValueError("Input audio contains NaN values (corrupted)")
    if np.any(np.isinf(y_stereo)):
        raise ValueError("Input audio contains Inf values (corrupted)")

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
                "Loudness measurement failed. Audio may be corrupted or incompatible with pyloudnorm."
            )
    
    if loudness == -np.inf:
        raise ValueError("Audio is completely silent (loudness = -inf)")
    
    if loudness > 5:  # Suspiciously loud
        raise ValueError(
            f"Audio is unusually loud ({loudness:.1f} LUFS). "
            f"This may indicate clipping or corrupted input."
        )

    # Normalize
    y_norm = pyloudnorm.normalize.loudness(y_stereo, loudness, target_lufs)

    # Handle clipping
    peak = np.abs(y_norm).max()
    if peak > 1.0:
        warnings.warn(f"Clipping detected after normalization (peak={peak:.4f}). Applying soft clipping.")
        y_norm = np.tanh(y_norm)  # Smooth clipping curve
    
    # Final validation
    if y_norm.size == 0:
        raise ValueError("Normalization produced empty audio")

    if np.any(np.isnan(y_norm)):
        raise ValueError("Normalization produced NaN values")

    return y_norm


def process_stimulus(args):
    """Process a single stimulus: mix, resample, normalize, save.

    Returns dict with metadata for export_metadata.csv
    
    Raises:
        ValueError: if audio validation fails at any step
    """
    idx, row, audio_dir, araus_root = args

    # Filename: araus_XXXXXXX.wav (7-digit zero-padded index)
    wav_idx = f"{idx:07d}"
    wav_path = audio_dir / f"araus_{wav_idx}.wav"

    try:
        # Initialize mixer (per-worker instance)
        soundscapes = pd.read_csv(araus_root / "data" / "soundscapes.csv")
        maskers = pd.read_csv(araus_root / "data" / "maskers.csv")
        mixer = ArausMixer(
            soundscapes=soundscapes,
            maskers=maskers,
            soundscape_dir=araus_root / "soundscapes",
            masker_dir=araus_root / "maskers",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ArausMixer with araus_root={araus_root}: {e}")

    # Step 1: Mix audio (row is passed as dict for pickling)
    x, sr = mixer.mix(row['soundscape'], row['masker'], row['smr'])
    
    if x.size == 0:
        raise ValueError(f"ArausMixer produced empty audio for soundscape={row['soundscape']}, masker={row['masker']}, smr={row['smr']}")

    # Step 2: Prepare (mono, 48 kHz, 30 s, RMS norm)
    y = prepare(x, sr)
    
    if y.size == 0:
        raise ValueError("prepare() returned empty audio")
    
    expected_samples_48k = int(30 * 48000)
    if len(y) < expected_samples_48k * 0.9:
        raise ValueError(f"prepare() returned truncated audio: {len(y)} samples (expected ~{expected_samples_48k})")

    # Step 3: Resample to 44.1 kHz (using improved function)
    y = resample_audio_safe(y, 48000, 44100)
    sr_out = 44100
    
    expected_samples_44k = int(30 * 44100)
    if len(y) < expected_samples_44k * 0.9:
        raise ValueError(f"resample_audio_safe() returned truncated audio: {len(y)} samples (expected ~{expected_samples_44k})")

    # Step 4: Normalise to -23 LUFS (returns stereo)
    y = normalise_loudness(y, sr_out, target_lufs=-23.0)
    
    if y.size == 0:
        raise ValueError("normalise_loudness() returned empty audio")
    
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError(f"normalise_loudness() did not return stereo: shape is {y.shape}")

    # Step 5: Save as 16-bit WAV
    sf.write(str(wav_path), y, sr_out, subtype="PCM_16")
    
    # Step 6: Post-save validation
    if not wav_path.exists():
        raise RuntimeError(f"WAV file was not created: {wav_path}")
    
    file_size = wav_path.stat().st_size
    if file_size < 10000:
        raise RuntimeError(f"WAV file suspiciously small ({file_size} bytes)")
    
    # Step 7: Verify by re-loading
    try:
        import librosa
        y_check, sr_check = librosa.load(str(wav_path), sr=None, mono=False)
        if y_check.size == 0:
            raise RuntimeError("Re-loaded file is empty")
        if sr_check != sr_out:
            raise RuntimeError(f"Saved file has wrong SR: {sr_check} (expected {sr_out})")
    except ImportError:
        pass  # librosa not available, skip re-load check

    return {
        "id": wav_idx,
        "stimulus_id": row['stimulus_id'],
        "caption": row['caption'],
        "ISOPleasant": row['ISOPleasant'],
        "bin": row['bin'],
        "LA50": row['LA50'],
        "wav_path": str(wav_path.relative_to(Path.cwd())),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--araus", required=True, help="Path to ARAUS dataset root")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--captions", default="data/generation/captions.csv")
    ap.add_argument("--out", default="data/generation")
    ap.add_argument("--n-samples", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8,
                    help="Number of parallel workers (default: 8)")
    a = ap.parse_args()

    araus_root = Path(a.araus)
    proc_root = Path(a.processed)
    out = Path(a.out)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    print("Loading ARAUS stimuli...")
    araus = load_araus(araus_root)
    
    print("Loading captions...")
    captions = pd.read_csv(a.captions, index_col=0)
    
    # Stratify sample
    print(f"Sampling {a.n_samples} stratified stimuli...")
    sample = stratified_sample(araus["stimuli"], n_samples=a.n_samples, seed=a.seed)
    
    # Merge with captions
    sample = sample.merge(captions[["caption", "ISOPleasant"]], left_on="stimulus_id", right_index=True)
    
    print(f"\nDistribution:")
    print(sample['bin'].value_counts().sort_index())
    print(f"\nExporting {len(sample)} audio files...")

    # Process in parallel
    results = []
    failed = []
    
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futures = {
            ex.submit(process_stimulus, (idx, row.to_dict(), audio_dir, araus_root)): idx
            for idx, (_, row) in enumerate(sample.iterrows())
        }
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                if (len(results) + len(failed)) % 500 == 0:
                    print(f"  Progress: {len(results)}/{len(sample)} ✓, {len(failed)} ✗")
            except Exception as e:
                failed.append({"stimulus_idx": idx, "error": str(e)})
                print(f"  ✗ Stimulus {idx}: {str(e)[:80]}")

    # Write metadata
    if results:
        results_df = pd.DataFrame(results)
        metadata_path = out / "export_metadata.csv"
        results_df.to_csv(metadata_path, index=False)
        print(f"\n✅ Export complete: {len(results)}/{len(sample)} files successfully exported")
        print(f"   Metadata saved to {metadata_path}")
    else:
        print(f"\n❌ Export failed: no files were successfully processed")
        sys.exit(1)

    if failed:
        print(f"\n⚠️  {len(failed)} files failed:")
        for f in failed[:10]:
            print(f"   {f['stimulus_idx']}: {f['error'][:100]}")
        if len(failed) > 10:
            print(f"   ... and {len(failed) - 10} more")
        
        # Write error log
        error_log_path = out / "export_errors.csv"
        pd.DataFrame(failed).to_csv(error_log_path, index=False)
        print(f"   Full error log: {error_log_path}")
    
    print("\nExport summary:")
    print(f"  Total requested: {len(sample)}")
    print(f"  Successfully exported: {len(results)}")
    print(f"  Failed: {len(failed)}")
    
    sys.exit(0 if len(failed) == 0 else 1)
