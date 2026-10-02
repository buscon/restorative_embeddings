#!/usr/bin/env python3
"""
Prepare ARAUS Dataset for Stable Audio 3 LoRA Fine-Tuning

Purpose: Convert ARAUS metadata and audio files into SA3-compatible format.
Creates paired audio + text caption files organized by pleasantness category.

SA3 LoRA expects:
  - data_dir/
    - stimulus_001.wav
    - stimulus_001.txt
    - stimulus_002.wav
    - stimulus_002.txt
    - ...

Usage:
    python 02_prepare_training_data_sa3.py \\
        --metadata /path/to/export_metadata.csv \\
        --audio-dir /path/to/audio \\
        --output-dir ./araus_for_sa3

Output:
    araus_for_sa3/ (organized audio + caption pairs for SA3 training)
"""

import argparse
import csv
from pathlib import Path
import shutil
from datetime import datetime


def load_metadata(csv_path):
    """Load ARAUS metadata from CSV."""
    metadata = {}

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                export_id = row.get('id')  # Use 'id' column (0000000, 0000004, etc.)
                if export_id:
                    metadata[export_id] = row
        print(f"✓ Loaded {len(metadata)} metadata entries from {csv_path}")
        return metadata
    except Exception as e:
        print(f"✗ Error loading metadata: {e}")
        return {}


def create_caption(row):
    """Create SA3 caption from metadata row with minimal pleasantness conditioning.
    
    Format: description + pleasantness level (0-100)
    Example: "urban park with birds and ambient activity. [pleasantness: 50]"
    """
    caption = row.get('caption', '')
    iso_pleasant = row.get('ISOPleasant', '')

    # Add ISO pleasantness rating (0-100 scale) if available
    if iso_pleasant and iso_pleasant.lower() != 'nan':
        try:
            iso_val = float(iso_pleasant)
            return f"{caption} [pleasantness: {iso_val:.0f}]"
        except:
            pass

    # Fallback to description only if pleasantness not available
    return caption


def prepare_dataset(metadata, audio_dir, output_dir):
    """
    Prepare SA3 dataset: copy audio files and create caption files.

    Args:
        metadata: Dictionary of metadata entries
        audio_dir: Path to source audio directory
        output_dir: Path to output directory for SA3 training
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = Path(audio_dir)

    # Track statistics
    total = 0
    copied = 0
    missing = 0
    failed = 0

    # Create pleasantness bin subdirectories
    bins = set()
    for row in metadata.values():
        bin_level = row.get('bin', 'unknown')
        if bin_level and bin_level.lower() != 'nan':
            bins.add(bin_level)

    bin_dirs = {}
    for bin_level in sorted(bins):
        bin_dir = output_dir / bin_level
        bin_dir.mkdir(parents=True, exist_ok=True)
        bin_dirs[bin_level] = bin_dir

    print(f"\n✓ Created {len(bin_dirs)} pleasantness category directories:")
    for bin_level in sorted(bins):
        print(f"    - {bin_level}")

    # Process each metadata entry
    print(f"\nProcessing {len(metadata)} audio files...")

    for export_id, row in metadata.items():
        total += 1

        # Get audio filename from metadata
        wav_path = row.get('wav_path', '')
        if not wav_path:
            print(f"  ⚠ [{export_id}] No wav_path in metadata")
            missing += 1
            continue

        # Extract just the filename (handle full paths in CSV)
        wav_filename = Path(wav_path).name

        # Resolve audio file path (try filename only first)
        audio_file = audio_dir / wav_filename
        if not audio_file.exists():
            # Try the full path from CSV (in case it's relative to a different base)
            audio_file = audio_dir / wav_path
            if not audio_file.exists():
                # Try absolute path
                audio_file = Path(wav_path)
                if not audio_file.exists():
                    print(f"  ✗ [{export_id}] Audio file not found: {wav_filename} (tried: {audio_dir / wav_filename})")
                    missing += 1
                    continue

        # Determine output directory (by pleasantness bin)
        bin_level = row.get('bin', 'unknown')
        if bin_level and bin_level.lower() != 'nan' and bin_level in bin_dirs:
            target_dir = bin_dirs[bin_level]
        else:
            target_dir = output_dir

        # Create output filenames (use 'id' column instead of 'stimulus_id' to avoid pipe chars)
        stimulus_id = row.get('id', export_id or 'unknown')  # Use clean numeric ID (0000000, 0000004, etc.)
        output_audio = target_dir / f"{stimulus_id}.wav"
        output_caption = target_dir / f"{stimulus_id}.txt"

        try:
            # Copy audio file
            shutil.copy2(audio_file, output_audio)

            # Create caption file
            caption = create_caption(row)
            with open(output_caption, 'w', encoding='utf-8') as f:
                f.write(caption)

            copied += 1
            if total % 100 == 0:
                print(f"  ... processed {total} files ({copied} copied)")

        except Exception as e:
            print(f"  ✗ [{export_id}] Error: {e}")
            failed += 1

    # Summary
    print("\n" + "="*70)
    print("Dataset Preparation Summary")
    print("="*70)
    print(f"Total metadata entries:  {total}")
    print(f"Files copied:            {copied}")
    print(f"Files missing:           {missing}")
    print(f"Errors:                  {failed}")
    print(f"\nOutput directory:        {output_dir}")

    # Count files by bin
    print("\nFiles per pleasantness category:")
    for bin_level in sorted(bin_dirs.keys()):
        bin_dir = bin_dirs[bin_level]
        wav_files = list(bin_dir.glob("*.wav"))
        txt_files = list(bin_dir.glob("*.txt"))
        print(f"  - {bin_level:20s}: {len(wav_files):5d} audio files, {len(txt_files):5d} captions")

    # Root directory files (if any)
    root_wav = list(output_dir.glob("*.wav"))
    root_txt = list(output_dir.glob("*.txt"))
    if root_wav or root_txt:
        print(f"  - (root directory):     {len(root_wav):5d} audio files, {len(root_txt):5d} captions")

    return copied > 0


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ARAUS dataset for Stable Audio 3 LoRA fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 02_prepare_training_data_sa3.py \\
    --metadata ~/Documents/Bamberg/restorative_embeddings/data/generation/export_metadata.csv \\
    --audio-dir ~/Documents/Bamberg/restorative_embeddings/data/generation/audio \\
    --output-dir ./araus_for_sa3
        """
    )

    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to ARAUS export_metadata.csv file"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        required=True,
        help="Path to directory containing audio files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./araus_for_sa3",
        help="Output directory for SA3-formatted dataset (default: ./araus_for_sa3)"
    )

    args = parser.parse_args()

    # Verify inputs
    metadata_path = Path(args.metadata)
    audio_dir = Path(args.audio_dir)

    print("\n" + "="*70)
    print("Prepare ARAUS Dataset for Stable Audio 3")
    print("="*70 + "\n")

    if not metadata_path.exists():
        print(f"✗ Metadata file not found: {metadata_path}")
        return False

    if not audio_dir.exists():
        print(f"✗ Audio directory not found: {audio_dir}")
        return False

    print(f"Metadata file:  {metadata_path}")
    print(f"Audio directory: {audio_dir}")
    print(f"Output directory: {args.output_dir}")

    # Load and process
    metadata = load_metadata(metadata_path)
    if not metadata:
        print("✗ No metadata loaded")
        return False

    success = prepare_dataset(metadata, audio_dir, args.output_dir)

    if success:
        print("\n✓ Dataset preparation completed successfully")
        print(f"\nNext step: Run SA3 LoRA training")
        print(f"  cd ~/stable-audio-3")
        print(f"  uv run python scripts/train_lora.py \\")
        print(f"    --model medium-base \\")
        print(f"    --data_dir {args.output_dir} \\")
        print(f"    --rank 16 \\")
        print(f"    --adapter_type dora-rows \\")
        print(f"    --steps 5000")
        print()
        return True
    else:
        print("\n✗ Dataset preparation failed")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
