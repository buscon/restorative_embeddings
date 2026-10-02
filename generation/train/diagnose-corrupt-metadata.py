#!/usr/bin/env python3
"""
Diagnose Corrupted Entries in ARAUS export_metadata.csv

Identifies which metadata entries have malformed wav_path values,
which suggests the root cause of 0-duration files or naming corruption.

Usage:
    python diagnose-corrupt-metadata.py <path-to-export_metadata.csv>
"""

import argparse
import csv
from pathlib import Path


def diagnose_metadata(csv_path):
    """Examine metadata CSV for corrupted entries."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        return

    print(f"📊 Diagnosing: {csv_path}\n")

    # Track issues
    total_rows = 0
    issues = {
        "pipe_in_path": [],      # Paths containing | (pipe character)
        "multiple_extensions": [],  # Paths like "file1.wav|file2.wav"
        "empty_wav_path": [],    # Missing wav_path field
        "suspicious_names": [],  # Other unusual patterns
    }

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row_num, row in enumerate(reader, start=2):  # Start at 2 (accounting for header)
                total_rows += 1

                wav_path = row.get('wav_path', '').strip()

                if not wav_path:
                    issues["empty_wav_path"].append({
                        'row': row_num,
                        'export_id': row.get('export_id', 'N/A'),
                        'stimulus_id': row.get('stimulus_id', 'N/A'),
                    })
                    continue

                # Check for pipe character (CSV corruption indicator)
                if '|' in wav_path:
                    issues["pipe_in_path"].append({
                        'row': row_num,
                        'export_id': row.get('export_id', 'N/A'),
                        'wav_path': wav_path,
                    })

                # Check for multiple file extensions (suggests concatenation error)
                if wav_path.count('.wav') > 1 or wav_path.count('.mp3') > 1:
                    issues["multiple_extensions"].append({
                        'row': row_num,
                        'export_id': row.get('export_id', 'N/A'),
                        'wav_path': wav_path,
                    })

                # Other suspicious patterns
                if ('|' not in wav_path and
                    wav_path.count('.wav') == 1 and
                    wav_path.count('.mp3') <= 1):
                    # Check for unusual patterns
                    if any(c in wav_path for c in ['\n', '\r', '\t', '\0']):
                        issues["suspicious_names"].append({
                            'row': row_num,
                            'export_id': row.get('export_id', 'N/A'),
                            'wav_path': repr(wav_path),  # Show escaped chars
                        })

    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # Report findings
    print(f"Total rows processed: {total_rows}\n")

    issue_count = sum(len(v) for v in issues.values())

    if issue_count == 0:
        print("✅ No obvious corruption detected in wav_path entries")
        return

    print(f"⚠️  Found {issue_count} corrupted entries:\n")

    # Report each issue type
    if issues["pipe_in_path"]:
        print(f"❌ PIPES IN PATH ({len(issues['pipe_in_path'])} entries):")
        print("   This is likely CSV parsing corruption (field not properly quoted)\n")
        for issue in issues["pipe_in_path"][:5]:
            print(f"   Row {issue['row']}: export_id={issue['export_id']}")
            print(f"      wav_path = {issue['wav_path']}\n")
        if len(issues["pipe_in_path"]) > 5:
            print(f"   ... and {len(issues['pipe_in_path']) - 5} more\n")

    if issues["multiple_extensions"]:
        print(f"❌ MULTIPLE EXTENSIONS ({len(issues['multiple_extensions'])} entries):")
        print("   This suggests file paths were concatenated incorrectly\n")
        for issue in issues["multiple_extensions"][:5]:
            print(f"   Row {issue['row']}: export_id={issue['export_id']}")
            print(f"      wav_path = {issue['wav_path']}\n")
        if len(issues["multiple_extensions"]) > 5:
            print(f"   ... and {len(issues['multiple_extensions']) - 5} more\n")

    if issues["empty_wav_path"]:
        print(f"❌ EMPTY WAV_PATH ({len(issues['empty_wav_path'])} entries):")
        for issue in issues["empty_wav_path"][:5]:
            print(f"   Row {issue['row']}: export_id={issue['export_id']}, stimulus_id={issue['stimulus_id']}\n")
        if len(issues["empty_wav_path"]) > 5:
            print(f"   ... and {len(issues['empty_wav_path']) - 5} more\n")

    if issues["suspicious_names"]:
        print(f"⚠️  SUSPICIOUS PATTERNS ({len(issues['suspicious_names'])} entries):\n")
        for issue in issues["suspicious_names"][:5]:
            print(f"   Row {issue['row']}: export_id={issue['export_id']}")
            print(f"      wav_path = {issue['wav_path']}\n")
        if len(issues["suspicious_names"]) > 5:
            print(f"   ... and {len(issues['suspicious_names']) - 5} more\n")

    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)

    if issues["pipe_in_path"]:
        print("\n🔴 ROOT CAUSE: CSV Parsing Error")
        print("   The export_metadata.csv file contains pipes (|) in wav_path values.")
        print("   This indicates a CSV quoting or escaping error during generation.")
        print("\n   Likely source: 02_export.py or intermediate CSV processing")
        print("\n   Next step: Check how export_metadata.csv was created")
        print("   - Verify that metadata.to_csv() uses proper quoting")
        print("   - Check if wav_path contains special characters that need escaping")

    elif issues["multiple_extensions"]:
        print("\n🔴 ROOT CAUSE: Path Concatenation Error")
        print("   The wav_path contains multiple .wav or .mp3 extensions.")
        print("   This suggests filename concatenation without proper delimiters.")
        print("\n   Likely source: 02_export.py when building export_id or output paths")

    else:
        print("\n🟡 ROOT CAUSE: Unknown")
        print("   Metadata appears valid, but files still have 0 duration.")
        print("   The issue may be in:")
        print("   - The actual audio files (empty or corrupted)")
        print("   - The ArausMixer (not generating valid audio)")
        print("   - The normalise_loudness or resample functions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?",
                       default=Path.home() / "Documents/restorative_embeddings/data/generation/export_metadata.csv",
                       help="Path to export_metadata.csv")

    args = parser.parse_args()
    diagnose_metadata(args.csv_path)
