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

import numpy as np
import pandas as pd
import soundfile as sf

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


def normalise_loudness(y, sr, target_lufs=-23.0):
    """Normalise audio to target loudness using pyloudnorm.

    Args:
        y: (n_samples,) mono audio array (will be converted to stereo)
        sr: sample rate
        target_lufs: target loudness in LUFS

    Returns:
        (n_samples, 2) stereo normalised audio

    Raises:
        ValueError: if audio is empty or completely silent
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

    meter = pyloudnorm.Meter(sr)
    loudness = meter.integrated_loudness(y_stereo)

    # Validate loudness measurement
    if np.isnan(loudness):
        raise ValueError("Loudness measurement returned NaN (corrupted or invalid audio)")
    
    if loudness == -np.inf:
        raise ValueError("Audio is completely silent (loudness = -inf)")

    # Normalize
    y_norm = pyloudnorm.normalize.loudness(y_stereo, loudness, target_lufs)

    # Clip to [-1, 1] to avoid distortion
    y_norm = np.clip(y_norm, -1.0, 1.0)
    
    # Final validation: ensure we didn't produce empty audio
    if y_norm.size == 0:
        raise ValueError("Normalization produced empty audio")

    return y_norm


def process_stimulus(args):
    """Process a single stimulus: mix, resample, normalize, save.

    Returns dict with metadata for export_metadata.csv
    
    Raises:
        ValueError: if audio validation fails at any step
    """
    idx, row, audio_dir, araus_root = args

    from scipy.signal import resample_poly
    from math import gcd

    # Filename: araus_XXXXXXX.wav (7-digit zero-padded index)
    wav_idx = f"{idx:07d}"
    wav_path = audio_dir / f"araus_{wav_idx}.wav"

    # Initialize mixer (per-worker instance)
    try:
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

    # Mix audio (row is passed as dict for pickling)
    x, sr = mixer.mix(row['soundscape'], row['masker'], row['smr'])
    
    # Validate mixer output
    if x.size == 0:
        raise ValueError(f"ArausMixer produced empty audio for soundscape={row['soundscape']}, masker={row['masker']}, smr={row['smr']}")

    # Prepare: mono, 48 kHz, 30 s, RMS norm
    y = prepare(x, sr)
    
    # Validate prepare output
    if y.size == 0:
        raise ValueError("prepare() returned empty audio")
    
    expected_samples_48k = int(30 * 48000)  # 30 seconds at 48 kHz
    if len(y) < expected_samples_48k * 0.9:  # Allow 10% tolerance
        raise ValueError(f"prepare() returned truncated audio: {len(y)} samples (expected ~{expected_samples_48k})")

    # Resample to 44.1 kHz
    g = gcd(int(48000), int(44100))
    y = resample_poly(y, 44100 // g, 48000 // g)
    sr_out = 44100
    
    # Validate resample output
    if y.size == 0:
        raise ValueError("resample_poly() returned empty audio")
    
    expected_samples_44k = int(30 * 44100)  # 30 seconds at 44.1 kHz
    if len(y) < expected_samples_44k * 0.9:  # Allow 10% tolerance
        raise ValueError(f"resample_poly() returned truncated audio: {len(y)} samples (expected ~{expected_samples_44k})")

    # Normalise to -23 LUFS (returns stereo)
    y = normalise_loudness(y, sr_out, target_lufs=-23.0)
    
    # Validate normalise output
    if y.size == 0:
        raise ValueError("normalise_loudness() returned empty audio")
    
    if y.shape[0] < expected_samples_44k * 0.9:
        raise ValueError(f"normalise_loudness() produced truncated audio: {y.shape[0]} samples (expected ~{expected_samples_44k})")
    
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError(f"normalise_loudness() did not return stereo: shape is {y.shape}")

    # Save as 16-bit WAV
    sf.write(str(wav_path), y, sr_out, subtype="PCM_16")
    
    # Post-save validation: verify file was created and has audio
    if not wav_path.exists():
        raise RuntimeError(f"WAV file was not created: {wav_path}")
    
    if wav_path.stat().st_size < 1000:  # WAV header is ~44 bytes; 1KB is a reasonable minimum
        raise RuntimeError(f"WAV file is suspiciously small ({wav_path.stat().st_size} bytes): {wav_path}")

    # Return metadata
    return {
        "export_id": wav_idx,
        "stimulus_id": row['stimulus_id'],
        "caption": row['caption'],
        "ISOPleasant": row['ISOPleasant'],
        "bin": row['bin'],
        "LA50": row['LA50'],
        "wav_path": str(wav_path),
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

    # Load ARAUS metadata
    araus_data = load_araus(araus_root)
    stim = araus_data["stimuli"]
    sounds = araus_data["soundscapes"]
    maskers = araus_data["maskers"]

    # Load captions and filter to ARAUS only
    captions = pd.read_csv(a.captions)
    captions_araus = captions[captions.dataset == "araus"].copy()

    # Add caption and bin info to stimuli
    # Merge on stimulus_id (from stim) = id (from captions)
    stim = stim.merge(
        captions_araus[["id", "caption", "bin"]],
        left_on="stimulus_id",
        right_on="id",
        how="inner"
    )

    # Stratified sample
    sample = stratified_sample(stim, n_samples=a.n_samples, seed=a.seed)

    # Export audio in parallel
    print(f"Exporting {len(sample)} stratified ARAUS stimuli to {audio_dir}...")
    print(f"Using {a.workers} parallel workers...")

    export_rows = []
    processed = 0
    failed_stimuli = []

    with ProcessPoolExecutor(max_workers=a.workers) as executor:
        # Prepare tasks
        tasks = [
            (idx, row, audio_dir, araus_root)
            for idx, (_, row) in enumerate(sample.iterrows())
        ]

        # Submit all tasks
        futures = {
            executor.submit(process_stimulus, task): task[0]
            for task in tasks
        }

        # Collect results as they complete
        for future in as_completed(futures):
            idx = futures[future]
            try:
                metadata = future.result()
                export_rows.append(metadata)
                processed += 1

                if processed % 500 == 0:
                    print(f"  ... {processed} / {len(sample)}")
            except Exception as e:
                failed_stimuli.append((idx, str(e)))
                print(f"SKIP stimulus {idx}: {e}")

    # Save metadata
    metadata = pd.DataFrame(export_rows)
    metadata.to_csv(out / "export_metadata.csv", index=False)

    print(f"\nExport complete:")
    print(f"  {len(metadata)} WAV files in {audio_dir}")
    print(f"  Metadata saved to {out}/export_metadata.csv")
    print(f"\n  Pleasantness bin distribution:")
    print(metadata.bin.value_counts().sort_index().to_string())
    
    if failed_stimuli:
        print(f"\n  ⚠️  {len(failed_stimuli)} stimuli skipped due to validation errors:")
        for idx, error in failed_stimuli[:10]:
            print(f"     - stimulus {idx}: {error}")
        if len(failed_stimuli) > 10:
            print(f"     ... and {len(failed_stimuli) - 10} more")
