#!/usr/bin/env python3
"""
Fix Corrupt ARAUS Files: Remove 0-Duration Audio Files

Identifies and removes the 50 corrupt 0-duration files from araus_for_sa3 dataset.
Re-validates dataset after cleanup.

Usage:
    python fix-corrupt-araus-dataset.py [--dry-run]
    python fix-corrupt-araus-dataset.py --data-dir ~/path/to/araus_for_sa3 [--dry-run]
"""

import os
import sys
from pathlib import Path
import argparse

try:
    import librosa
    import numpy as np
except ImportError:
    print("Error: librosa required. Install with: pip install librosa")
    sys.exit(1)


def find_corrupt_files(dataset_dir, sample_rate=44100, expected_duration=10.0):
    """
    Identify all files with 0 duration or significantly wrong duration.

    Returns:
        dict with 'zero_duration', 'wrong_duration', 'other_issues'
    """
    dataset_dir = Path(dataset_dir)
    issues = {
        'zero_duration': [],
        'wrong_duration': [],
        'other_issues': [],
    }

    categories = ["neutral", "pleasant", "unpleasant", "very_pleasant", "very_unpleasant"]
    total_files = 0

    for category in categories:
        cat_dir = dataset_dir / category
        if not cat_dir.exists():
            continue

        wav_files = sorted(cat_dir.glob("*.wav"))

        for wav_file in wav_files:
            total_files += 1
            txt_file = wav_file.with_suffix(".txt")

            try:
                y, sr = librosa.load(str(wav_file), sr=None)
                duration = len(y) / sr

                if duration == 0.0:
                    issues['zero_duration'].append({
                        'file': wav_file.name,
                        'category': category,
                        'path': wav_file,
                        'txt_path': txt_file,
                        'duration': duration,
                    })
                elif abs(duration - expected_duration) > 1.0:  # More than 1s off
                    issues['wrong_duration'].append({
                        'file': wav_file.name,
                        'category': category,
                        'path': wav_file,
                        'duration': duration,
                    })

            except Exception as e:
                issues['other_issues'].append({
                    'file': wav_file.name,
                    'category': category,
                    'path': wav_file,
                    'error': str(e),
                })

    return issues, total_files


def remove_corrupt_files(issues, dry_run=False):
    """
    Remove corrupt files (both audio and caption).

    Args:
        issues: Dict from find_corrupt_files()
        dry_run: If True, only report what would be removed

    Returns:
        Number of files removed
    """
    removed_count = 0

    # Remove zero-duration files
    for issue in issues['zero_duration']:
        wav_path = issue['path']
        txt_path = issue['txt_path']

        if dry_run:
            print(f"  Would remove: {wav_path.name} (category: {issue['category']})")
            if txt_path.exists():
                print(f"               {txt_path.name}")
        else:
            try:
                if wav_path.exists():
                    wav_path.unlink()
                    print(f"  ✓ Removed: {wav_path.name}")

                if txt_path.exists():
                    txt_path.unlink()
                    print(f"  ✓ Removed: {txt_path.name}")

                removed_count += 1
            except Exception as e:
                print(f"  ✗ Error removing {wav_path.name}: {e}")

    return removed_count


def main():
    parser = argparse.ArgumentParser(
        description="Fix corrupt ARAUS dataset by removing 0-duration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fix-corrupt-araus-dataset.py [--dry-run]
    (uses default: ~/Documents/restorative_embeddings/generation/train/araus_for_sa3)

  python fix-corrupt-araus-dataset.py --data-dir ~/my-dataset [--dry-run]
    (uses custom path)
        """
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path.home() / "Documents/restorative_embeddings/generation/train/araus_for_sa3"),
        help="Path to araus_for_sa3 dataset"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing"
    )
    parser.add_argument(
        "--validate-after",
        action="store_true",
        help="Run validation after cleanup"
    )

    args = parser.parse_args()
    dataset_dir = Path(args.data_dir)

    print("\n" + "="*80)
    print("FIX CORRUPT ARAUS FILES")
    print("="*80 + "\n")

    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return False

    print(f"Dataset path: {dataset_dir}")
    print(f"Dry-run mode: {args.dry_run}\n")

    # Step 1: Find corrupt files
    print("[1/3] Scanning for 0-duration files...")
    issues, total_files = find_corrupt_files(dataset_dir)

    print(f"✓ Scanned {total_files} files\n")

    # Step 2: Report findings
    print("[2/3] Corrupt files found:")
    zero_count = len(issues['zero_duration'])
    wrong_count = len(issues['wrong_duration'])
    error_count = len(issues['other_issues'])

    if zero_count == 0 and wrong_count == 0 and error_count == 0:
        print("✅ No corrupt files detected!")
        return True

    if zero_count > 0:
        print(f"\n  🔴 ZERO DURATION: {zero_count} files")
        for issue in issues['zero_duration'][:5]:
            print(f"     - {issue['category']}/{issue['file']}")
        if zero_count > 5:
            print(f"     ... and {zero_count - 5} more")

    if wrong_count > 0:
        print(f"\n  🟡 WRONG DURATION: {wrong_count} files")
        for issue in issues['wrong_duration'][:5]:
            print(f"     - {issue['category']}/{issue['file']} ({issue['duration']:.1f}s, expected ~10s)")
        if wrong_count > 5:
            print(f"     ... and {wrong_count - 5} more")

    if error_count > 0:
        print(f"\n  🟠 LOAD ERRORS: {error_count} files")
        for issue in issues['other_issues'][:3]:
            print(f"     - {issue['category']}/{issue['file']}: {issue['error']}")
        if error_count > 3:
            print(f"     ... and {error_count - 3} more")

    print()

    # Step 3: Remove files (zero-duration only)
    if zero_count > 0:
        print(f"[3/3] {'[DRY-RUN] ' if args.dry_run else ''}Removing {zero_count} zero-duration files...")
        removed = remove_corrupt_files(issues, dry_run=args.dry_run)

        if not args.dry_run:
            print(f"\n✅ Removed {removed} files\n")

            # Re-validate if requested
            if args.validate_after:
                print("[4/4] Re-validating dataset...")
                issues_after, total_after = find_corrupt_files(dataset_dir)
                zero_after = len(issues_after['zero_duration'])

                if zero_after == 0:
                    print(f"✅ Validation passed! {total_after} files remain, all valid.\n")
                else:
                    print(f"⚠️  Still found {zero_after} corrupt files after removal.\n")
        else:
            print(f"\n[DRY-RUN] Would remove {removed} files")
            print("Run without --dry-run to actually remove files\n")

    # Summary
    print("="*80)
    if args.dry_run:
        print("DRY-RUN SUMMARY")
        print("="*80)
        print(f"Files to remove: {zero_count}")
        print(f"Files to keep: {total_files - zero_count}")
        print(f"\nRun without --dry-run to remove these files")
    else:
        print("CLEANUP SUMMARY")
        print("="*80)
        print(f"Files removed: {removed}")
        print(f"Files remaining: {total_files - removed}")

    print()
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
