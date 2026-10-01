"""Locate and organize audio files corresponding to captions for verification.

Maps captions to their original ARAUS stimuli, showing masker type, soundscape,
and SMR (signal-to-masker ratio) to help find the audio files in the dataset.

Output: CSV with [caption_id, stimulus_id, soundscape, masker_type, caption]
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


def parse_stimulus_id(stimulus_id):
    """Parse ARAUS stimulus_id format: soundscape|masker|SMR

    Returns dict with components or None if format doesn't match.
    """
    if pd.isna(stimulus_id) or not isinstance(stimulus_id, str):
        return None

    parts = stimulus_id.split('|')
    if len(parts) == 3:
        return {
            'soundscape': parts[0],
            'masker': parts[1],
            'smr': parts[2]
        }
    return None


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

    # Parse stimulus IDs to extract components
    print("\nParsing stimulus IDs to extract soundscape and masker...")
    araus_captions['stimulus_components'] = araus_captions['stimulus_id'].apply(parse_stimulus_id)

    # Extract components into separate columns
    araus_captions['soundscape_file'] = araus_captions['stimulus_components'].apply(
        lambda x: x['soundscape'] if x else None
    )
    araus_captions['masker_file'] = araus_captions['stimulus_components'].apply(
        lambda x: x['masker'] if x else None
    )
    araus_captions['smr_value'] = araus_captions['stimulus_components'].apply(
        lambda x: x['smr'] if x else None
    )

    # Show how to find audio
    print("\n" + "="*80)
    print("AUDIO LOCATION GUIDE")
    print("="*80)

    if stim_files is not None:
        print(f"\n✓ Pre-mixed stimuli found at: {audio_info}")
        print(f"  Total stimulus files: {len(stim_files)}")
        print("\nTo find audio for a stimulus:")
        print("  1. Look up the stimulus_id (e.g., 'R0001_segment_binaural_44100_1.wav|bird_00001.wav|6')")
        print(f"  2. File should be at: {audio_info}/<stimulus_id>.wav")
    else:
        print("\n✓ Component files available for reconstruction")
        if isinstance(audio_info, tuple):
            soundscape_dir, masker_dir = audio_info
            print(f"\n  Soundscapes: {soundscape_dir}")
            print(f"  Maskers: {masker_dir}")
            print("\nTo reconstruct audio for a stimulus:")
            print("  1. Parse stimulus_id: '<soundscape>|<masker>|<SMR>'")
            print("  2. Load soundscape: <soundscape_dir>/<soundscape>")
            print("  3. Load masker: <masker_dir>/<masker>")
            print("  4. Mix at SMR ratio using ArausMixer or similar")

    # Save mapping for reference
    output_csv = out / "audio_location_mapping.csv"
    output_cols = ["id", "stimulus_id", "soundscape_file", "masker_file", "smr_value", "caption"]

    araus_captions[output_cols].to_csv(output_csv, index=False)
    print(f"\n✓ Mapping saved to: {output_csv}")
    print(f"  Total entries: {len(araus_captions)}")

    # Sample captions with audio info
    print("\n" + "="*80)
    print("SAMPLE CAPTIONS WITH AUDIO RECONSTRUCTION INFO")
    print("="*80)
    sample = araus_captions.head(5)
    for idx, row in sample.iterrows():
        print(f"\nCaption ID: {row['id']}")
        print(f"Stimulus ID: {row['stimulus_id']}")
        if row['soundscape_file'] and row['masker_file']:
            print(f"  Soundscape: {row['soundscape_file']}")
            print(f"  Masker: {row['masker_file']}")
            print(f"  SMR: {row['smr_value']} dB")
        print(f"Caption: {row['caption'][:80]}...")
