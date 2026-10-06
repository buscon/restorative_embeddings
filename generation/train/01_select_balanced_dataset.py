#!/usr/bin/env python3
"""
Select balanced subset of ARAUS files for training.

Purpose: Sample from the full ARAUS dataset (5625 files) while maintaining
distribution across pleasantness bins.

Pleasantness bin distribution (full dataset):
  - neutral:        1455 files (25.9%)
  - pleasant:       1581 files (28.1%)
  - unpleasant:     1410 files (25.1%)
  - very_pleasant:   578 files (10.3%)
  - very_unpleasant: 661 files (11.8%)

Usage:
    python 01_select_balanced_dataset.py \\
        --metadata /path/to/export_metadata.csv \\
        --output-csv selected_metadata.csv \\
        --num-samples 360 \\
        --seed 42

The script will:
  1. Load full ARAUS metadata
  2. Group by pleasantness bin
  3. Sample proportionally from each bin
  4. Save filtered metadata CSV for training
"""

import argparse
import csv
import random
from pathlib import Path
from collections import defaultdict


def load_metadata(csv_path):
    """Load ARAUS metadata from CSV and group by pleasantness bin."""
    metadata_by_bin = defaultdict(list)

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                bin_level = row.get('bin', '')
                if bin_level and bin_level.lower() != 'nan':
                    metadata_by_bin[bin_level].append(row)

        print(f"✓ Loaded metadata from {csv_path}")
        print(f"\nFull dataset distribution:")
        total = sum(len(items) for items in metadata_by_bin.values())
        for bin_level in sorted(metadata_by_bin.keys()):
            count = len(metadata_by_bin[bin_level])
            pct = 100 * count / total
            print(f"  {bin_level:20s}: {count:5d} files ({pct:5.1f}%)")
        print(f"  {'Total':20s}: {total:5d} files")

        return metadata_by_bin, total

    except Exception as e:
        print(f"✗ Error loading metadata: {e}")
        return {}, 0


def select_balanced_subset(metadata_by_bin, total_files, num_samples, seed=42):
    """
    Select num_samples files, proportionally balanced across pleasantness bins.

    Args:
        metadata_by_bin: Dict mapping bin name -> list of metadata rows
        total_files: Total files in full dataset
        num_samples: Target number of files to select
        seed: Random seed for reproducibility

    Returns:
        List of selected metadata rows
    """
    random.seed(seed)
    selected = []

    print(f"\nSelecting {num_samples} files (proportional to bin sizes):")
    print(f"Random seed: {seed}\n")

    for bin_level in sorted(metadata_by_bin.keys()):
        bin_items = metadata_by_bin[bin_level]
        bin_size = len(bin_items)
        proportion = bin_size / total_files
        target_count = int(num_samples * proportion)

        # Ensure at least 1 file from each bin
        target_count = max(1, target_count)

        # Don't sample more than available
        target_count = min(target_count, bin_size)

        # Randomly select from this bin
        sampled = random.sample(bin_items, k=target_count)
        selected.extend(sampled)

        pct_of_total = 100 * target_count / num_samples
        print(f"  {bin_level:20s}: {target_count:5d} files ({pct_of_total:5.1f}% of {num_samples})")

    total_selected = len(selected)
    print(f"  {'Total':20s}: {total_selected:5d} files")

    return selected


def save_selection(selected_metadata, output_csv):
    """Save selected metadata rows to CSV."""
    if not selected_metadata:
        print("✗ No metadata to save")
        return False

    try:
        # Rename 'id' to 'export_id' for downstream compatibility with 02_prepare_training_data_sa3.py
        for row in selected_metadata:
            if 'id' in row and 'export_id' not in row:
                row['export_id'] = row.pop('id')

        # Get fieldnames from first row
        fieldnames = list(selected_metadata[0].keys())
        # Ensure export_id comes first for clarity
        if 'export_id' in fieldnames:
            fieldnames.remove('export_id')
            fieldnames.insert(0, 'export_id')

        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected_metadata)

        print(f"\n✓ Saved {len(selected_metadata)} selected items to {output_csv}")
        return True

    except Exception as e:
        print(f"✗ Error saving metadata: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Select balanced ARAUS subset for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Select 360 files (expanded from original 100)
  python 01_select_balanced_dataset.py \\
    --metadata ~/Documents/restorative_embeddings/data/generation/export_metadata.csv \\
    --output-csv selected_360.csv \\
    --num-samples 360

  # Back to 100 files
  python 01_select_balanced_dataset.py \\
    --metadata ~/Documents/restorative_embeddings/data/generation/export_metadata.csv \\
    --output-csv selected_100.csv \\
    --num-samples 100

  # Custom amount
  python 01_select_balanced_dataset.py \\
    --metadata export_metadata.csv \\
    --output-csv selected.csv \\
    --num-samples 500 \\
    --seed 12345
        """
    )

    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to full ARAUS export_metadata.csv"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Output CSV with selected metadata"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=360,
        help="Number of files to select (default: 360, original was 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Verify input
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"✗ Metadata file not found: {metadata_path}")
        return False

    print("\n" + "="*70)
    print("ARAUS Balanced File Selection")
    print("="*70 + "\n")

    # Load and process
    metadata_by_bin, total = load_metadata(metadata_path)
    if not metadata_by_bin:
        print("✗ No metadata loaded")
        return False

    # Validate num_samples
    if args.num_samples > total:
        print(f"\n⚠ Requested {args.num_samples} files but only {total} available")
        print(f"  Clamping to {total}")
        num_samples = total
    else:
        num_samples = args.num_samples

    # Select balanced subset
    selected = select_balanced_subset(metadata_by_bin, total, num_samples, args.seed)

    # Save
    success = save_selection(selected, args.output_csv)

    if success:
        print("\n" + "="*70)
        print("✓ Selection complete!")
        print("="*70)
        print(f"\nNext step: Use this CSV to prepare training data")
        print(f"  python 02_prepare_training_data_sa3.py \\")
        print(f"    --metadata {args.output_csv} \\")
        print(f"    --audio-dir ~/Documents/restorative_embeddings/data/generation/audio \\")
        print(f"    --output-dir ./araus_for_sa3")
        print()
        return True
    else:
        print("\n✗ Selection failed")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
