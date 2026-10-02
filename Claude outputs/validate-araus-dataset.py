#!/usr/bin/env python3
"""
Validate ARAUS dataset for Stable Audio 3 LoRA training.
Checks audio quality, caption format, and data integrity without listening.
"""

import os
import sys
from pathlib import Path
import warnings

try:
    import librosa
    import numpy as np
except ImportError:
    print("Error: librosa required. Install with: pip install librosa")
    sys.exit(1)

warnings.filterwarnings('ignore')

# Configuration
DATASET_DIR = Path.home() / "Documents/restorative_embeddings/generation/train/araus_for_sa3"
EXPECTED_SR = 44100
EXPECTED_DURATION = 10  # seconds
EXPECTED_CHANNELS = 1  # mono (or 2 for stereo, we'll accept both)
CAPTION_PATTERN = r"^.+\. Loudness: .+ \(LA50: .+ dB\)\. Pleasantness: .+ \(ISOPleasant: .+\)\. \[pleasantness: .+\]$"

# Pleasantness categories
CATEGORIES = ["neutral", "pleasant", "unpleasant", "very_pleasant", "very_unpleasant"]

# Results storage
results = {
    "total_files": 0,
    "valid_pairs": 0,
    "invalid_pairs": [],
    "audio_issues": [],
    "caption_issues": [],
    "format_issues": [],
    "quality_stats": {cat: {"count": 0, "levels": [], "clipping": 0, "silent": 0} for cat in CATEGORIES}
}

print("=" * 80)
print("ARAUS DATASET VALIDATION")
print("=" * 80)
print(f"\nDataset path: {DATASET_DIR}")
print(f"Expected sample rate: {EXPECTED_SR} Hz")
print(f"Expected duration: {EXPECTED_DURATION}s")
print(f"Expected channels: {EXPECTED_CHANNELS}")

# ============================================================================
# STEP 1: Check directory structure
# ============================================================================
print("\n[1/5] Checking directory structure...")

if not DATASET_DIR.exists():
    print(f"❌ Dataset directory not found: {DATASET_DIR}")
    sys.exit(1)

missing_cats = [c for c in CATEGORIES if not (DATASET_DIR / c).exists()]
if missing_cats:
    print(f"⚠️  Missing categories: {missing_cats}")

for cat in CATEGORIES:
    cat_dir = DATASET_DIR / cat
    if cat_dir.exists():
        wav_count = len(list(cat_dir.glob("*.wav")))
        txt_count = len(list(cat_dir.glob("*.txt")))
        print(f"  {cat}: {wav_count} audio files, {txt_count} captions")

# ============================================================================
# STEP 2: Check file pairing
# ============================================================================
print("\n[2/5] Checking audio-caption pairing...")

for category in CATEGORIES:
    cat_dir = DATASET_DIR / category
    if not cat_dir.exists():
        continue

    wav_files = sorted(cat_dir.glob("*.wav"))

    for wav_file in wav_files:
        caption_file = wav_file.with_suffix(".txt")

        if not caption_file.exists():
            results["invalid_pairs"].append(f"{category}/{wav_file.name} → missing caption")
        else:
            results["valid_pairs"] += 1

        results["total_files"] += 1

print(f"  Valid pairs: {results['valid_pairs']}/{results['total_files']}")

if results["invalid_pairs"]:
    print(f"\n  ❌ Missing caption pairs ({len(results['invalid_pairs'])}):")
    for pair in results["invalid_pairs"][:5]:
        print(f"     - {pair}")
    if len(results["invalid_pairs"]) > 5:
        print(f"     ... and {len(results['invalid_pairs']) - 5} more")

# ============================================================================
# STEP 3: Check audio files
# ============================================================================
print("\n[3/5] Checking audio file integrity...")

audio_issues_count = 0

for category in CATEGORIES:
    cat_dir = DATASET_DIR / category
    if not cat_dir.exists():
        continue

    wav_files = sorted(cat_dir.glob("*.wav"))

    for wav_file in wav_files[:10]:  # Sample first 10 per category
        try:
            y, sr = librosa.load(str(wav_file), sr=None, mono=False)

            # Check sample rate
            if sr != EXPECTED_SR:
                results["audio_issues"].append(
                    f"{category}/{wav_file.name}: wrong SR ({sr} Hz, expected {EXPECTED_SR})"
                )
                audio_issues_count += 1

            # Check duration
            duration = len(y) / sr
            if abs(duration - EXPECTED_DURATION) > 0.5:
                results["audio_issues"].append(
                    f"{category}/{wav_file.name}: wrong duration ({duration:.1f}s, expected {EXPECTED_DURATION}s)"
                )
                audio_issues_count += 1

            # Check channels
            if y.ndim == 1:
                channels = 1
                y_mono = y
            else:
                channels = y.shape[0]
                y_mono = librosa.to_mono(y)

            # Audio statistics
            rms = np.sqrt(np.mean(y_mono ** 2))
            peak = np.max(np.abs(y_mono))

            # Detect clipping (peak near 1.0)
            if peak > 0.99:
                results["quality_stats"][category]["clipping"] += 1

            # Detect silence (RMS < -60dB)
            if rms < 0.001:  # ~-60 dB
                results["quality_stats"][category]["silent"] += 1

            results["quality_stats"][category]["levels"].append(20 * np.log10(rms + 1e-10))
            results["quality_stats"][category]["count"] += 1

        except Exception as e:
            results["audio_issues"].append(f"{category}/{wav_file.name}: {str(e)}")
            audio_issues_count += 1

if results["audio_issues"]:
    print(f"  ❌ Audio issues found ({len(results['audio_issues'])}):")
    for issue in results["audio_issues"][:5]:
        print(f"     - {issue}")
    if len(results["audio_issues"]) > 5:
        print(f"     ... and {len(results['audio_issues']) - 5} more")
else:
    print(f"  ✅ All sampled audio files valid")

# ============================================================================
# STEP 4: Check caption format
# ============================================================================
print("\n[4/5] Checking caption file format...")

caption_issues_count = 0
import re

for category in CATEGORIES:
    cat_dir = DATASET_DIR / category
    if not cat_dir.exists():
        continue

    txt_files = sorted(cat_dir.glob("*.txt"))

    for txt_file in txt_files[:10]:  # Sample first 10 per category
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                caption = f.read().strip()

            # Check if caption is empty
            if not caption:
                results["caption_issues"].append(f"{category}/{txt_file.name}: empty caption")
                caption_issues_count += 1
                continue

            # Check caption structure
            required_parts = ["Loudness:", "LA50:", "Pleasantness:", "ISOPleasant:", "[pleasantness:"]
            missing_parts = [p for p in required_parts if p not in caption]

            if missing_parts:
                results["caption_issues"].append(
                    f"{category}/{txt_file.name}: missing {missing_parts}"
                )
                caption_issues_count += 1

            # Try to extract LA50 value
            match = re.search(r'LA50: ([\d.]+)', caption)
            if not match:
                results["caption_issues"].append(
                    f"{category}/{txt_file.name}: cannot parse LA50 value"
                )
                caption_issues_count += 1

            # Try to extract ISOPleasant value
            match = re.search(r'ISOPleasant: ([-\d.]+)', caption)
            if not match:
                results["caption_issues"].append(
                    f"{category}/{txt_file.name}: cannot parse ISOPleasant value"
                )
                caption_issues_count += 1

        except UnicodeDecodeError:
            results["caption_issues"].append(f"{category}/{txt_file.name}: encoding error (not UTF-8)")
            caption_issues_count += 1
        except Exception as e:
            results["caption_issues"].append(f"{category}/{txt_file.name}: {str(e)}")
            caption_issues_count += 1

if results["caption_issues"]:
    print(f"  ❌ Caption issues found ({len(results['caption_issues'])}):")
    for issue in results["caption_issues"][:5]:
        print(f"     - {issue}")
    if len(results["caption_issues"]) > 5:
        print(f"     ... and {len(results['caption_issues']) - 5} more")
else:
    print(f"  ✅ All sampled captions valid format")

# ============================================================================
# STEP 5: Audio quality statistics
# ============================================================================
print("\n[5/5] Audio quality statistics by category...")

for category in CATEGORIES:
    stats = results["quality_stats"][category]
    if stats["count"] == 0:
        print(f"  {category}: (no files sampled)")
        continue

    levels_db = stats["levels"]
    if levels_db:
        mean_level = np.mean(levels_db)
        std_level = np.std(levels_db)
        min_level = np.min(levels_db)
        max_level = np.max(levels_db)

        print(f"\n  {category.upper()}:")
        print(f"    Files sampled: {stats['count']}")
        print(f"    RMS level: {mean_level:.1f} dB (±{std_level:.1f})")
        print(f"    Range: {min_level:.1f} to {max_level:.1f} dB")

        if stats["clipping"] > 0:
            print(f"    ⚠️  Clipping detected: {stats['clipping']} files")
        if stats["silent"] > 0:
            print(f"    ⚠️  Silent files: {stats['silent']} files")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

total_issues = len(results["invalid_pairs"]) + len(results["audio_issues"]) + len(results["caption_issues"])

if total_issues == 0:
    print("✅ Dataset validation PASSED")
    print(f"\n  Total valid audio-caption pairs: {results['valid_pairs']}")
    print(f"  All sampled files: OK")
    print("\n  Recommendation: Dataset appears valid. Training issues likely due to:")
    print("    - Learning rate too low (try 1e-3)")
    print("    - Batch size too small (try batch_size=2)")
    print("    - LoRA rank too small (try rank=32)")
else:
    print(f"⚠️  Dataset validation found {total_issues} ISSUES:\n")
    print(f"  Missing pairs: {len(results['invalid_pairs'])}")
    print(f"  Audio issues: {len(results['audio_issues'])}")
    print(f"  Caption issues: {len(results['caption_issues'])}")
    print("\n  Recommendation: Fix these issues before retraining")

print("\n" + "=" * 80)
