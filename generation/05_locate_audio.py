"""Locate and organize audio files corresponding to captions for verification.

Maps captions to their original ARAUS stimuli, showing masker type, soundscape,
and SMR (signal-to-masker ratio) to help find the audio files in the dataset.

Output: CSV with [caption_id, stimulus_id, soundscape, masker_type, SMR, caption]
and creates symbolic links to audio files in a verification folder (optional).

    python generation/05_locate_audio.py \
      --processed data/processed \
      --captions data/generation/captions.csv \
      --araus data/raw/araus \
      --out data/generation/audio_verification
"""

import argparse
from pathlib import Path
import pandas as pd


def locate_audio_files(araus_root):
    """Scan ARAUS dataset and return stimulus → audio file mapping.

    ARAUS structure:
        data/soundscapes/           → .wav files
        data/maskers/               → .wav files
        data/maskers_v2/            → alternative masker files
        data/stimuli/               → pre-mixed stimuli (if available)
    """
    araus_root = Path(araus_root)

    # First, try to find pre-mixed stimuli
    stimuli_dir = araus_root / "data" / "stimuli"
    if stimuli_dir.exists():
        print(f"Found pre-mixed stimuli directory: {stimuli_dir}")
        stim_files = {f.stem: f for f in stimuli_dir.glob("*.wav")}
        return stim_files, stimuli_dir

    # Fallback: locate component directories for reconstruction
    soundscape_dir = araus_root / "data" / "soundscapes"
    masker_dir = araus_root / "data" / "maskers"

    if soundscape_dir.exists() and masker_dir.exists():
        print(f"Found component directories (soundscapes and maskers)")
        print(f"  Soundscapes: {soundscape_dir}")
        print(f"  Maskers: {masker_dir}")
        return None, (soundscape_dir, masker_dir)

    print("ERROR: Could not locate ARAUS audio directories")
    return None, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", default="data/processed",
                    help="Path to processed data directory")
    ap.add_argument("--captions", default="data/generation/captions.csv",
                    help="Path to captions CSV")
    ap.add_argument("--araus", required=True,
                    help="Path to ARAUS dataset root")
    ap.add_argument("--out", default="data/generation/audio_verification",
                    help="Output directory for verification files")
    a = ap.parse_args()

    proc_root = Path(a.processed)
    captions_path = Path(a.captions)
    araus_root = Path(a.araus)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # Load captions and ARAUS stimuli
    print("Loading captions and stimulus metadata...")
    captions = pd.read_csv(captions_path)
    stimuli = pd.read_csv(proc_root / "araus_stimuli.csv")

    # Filter to ARAUS captions only
    araus_captions = captions[captions.dataset == "araus"].copy()
    print(f"Found {len(araus_captions)} ARAUS captions")

    # Check which columns are available in stimuli
    available_cols = ["stimulus_id", "soundscape", "masker_type"]
    for col in ["SMR", "gain_dB", "n_ratings", "LA50", "ISOPleasant"]:
        if col in stimuli.columns:
            available_cols.append(col)

    print(f"Available stimulus columns: {available_cols}")

    # Merge with stimulus metadata (only use available columns)
    araus_captions = araus_captions.merge(
        stimuli[available_cols],
        left_on="id",
        right_on="stimulus_id",
        how="left"
    )

    # Locate audio files
    print("\nLocating ARAUS audio files...")
    stim_files, audio_info = locate_audio_files(araus_root)

    # Show how to find audio
    print("\n" + "="*80)
    print("AUDIO LOCATION GUIDE")
    print("="*80)

    if stim_files is not None:
        print(f"\n✓ Pre-mixed stimuli found at: {audio_info}")
        print(f"  Total stimulus files: {len(stim_files)}")
        print("\nTo find audio for a stimulus:")
        print("  1. Look up the stimulus_id (e.g., 'stim_001')")
        print(f"  2. File should be at: {audio_info}/<stimulus_id>.wav")
    else:
        print("\n⚠ Pre-mixed stimuli not found")
        if isinstance(audio_info, tuple):
            soundscape_dir, masker_dir = audio_info
            print(f"\nComponent files available for reconstruction:")
            print(f"  Soundscapes: {soundscape_dir}")
            print(f"  Maskers: {masker_dir}")
            print("\nTo reconstruct audio for a stimulus:")
            print("  1. Look up the stimulus record: soundscape, masker_type, SMR, gain_dB")
            print("  2. Load: <soundscape>.wav + <masker_type>_<gain_dB>dB.wav")
            print("  3. Mix at specified SMR ratio")

    # Save mapping for reference
    output_csv = out / "audio_location_mapping.csv"
    output_cols = ["id", "stimulus_id", "soundscape", "masker_type"]
    for col in ["SMR", "gain_dB", "LA50", "ISOPleasant", "n_ratings", "caption"]:
        if col in araus_captions.columns:
            output_cols.append(col)

    araus_captions[output_cols].to_csv(output_csv, index=False)
    print(f"\n✓ Mapping saved to: {output_csv}")
    print(f"  Columns: {', '.join(output_cols)}")

    # Sample captions with audio info
    print("\n" + "="*80)
    print("SAMPLE CAPTIONS WITH AUDIO INFO")
    print("="*80)
    sample = araus_captions.head(5)
    for idx, row in sample.iterrows():
        print(f"\nStimulus ID: {row['stimulus_id']}")
        print(f"Soundscape: {row['soundscape']}")
        print(f"Masker: {row['masker_type']}")
        if 'gain_dB' in row and pd.notna(row['gain_dB']):
            print(f"  Gain: {row['gain_dB']} dB, SMR: {row['SMR']} dB")
        if 'LA50' in row and pd.notna(row['LA50']):
            print(f"  Loudness: {row['LA50']:.1f} dB")
        print(f"Caption: {row['caption'][:80]}...")
