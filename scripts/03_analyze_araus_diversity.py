#!/usr/bin/env python3
"""
Analyze ARAUS dataset metadata to understand diversity and structure.
Identifies which scenes, maskers, and pleasantness levels are available.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter

# Adjust these paths
DATA_DIR = Path("/home/marcello/Documents/restorative_embeddings/data/raw/araus/data")
REPORT_FILE = Path("/home/marcello/Documents/restorative_embeddings/araus_diversity_report.txt")

print("=" * 70)
print("ARAUS DATASET DIVERSITY ANALYSIS")
print("=" * 70)

# Load CSV files
print("\nLoading ARAUS metadata CSVs...")
try:
    soundscapes = pd.read_csv(DATA_DIR / "soundscapes.csv")
    responses = pd.read_csv(DATA_DIR / "responses.csv")
    maskers = pd.read_csv(DATA_DIR / "maskers.csv")
    print(f"✓ Soundscapes: {len(soundscapes)} records")
    print(f"✓ Responses: {len(responses)} records")
    print(f"✓ Maskers: {len(maskers)} records")
except Exception as e:
    print(f"ERROR loading CSVs: {e}")
    exit(1)

# Flexible column detection
def find_column(df, candidates):
    """Find first existing column from candidates list"""
    for col in candidates:
        if col in df.columns:
            return col
    return None

# Identify key columns
stimulus_col = find_column(soundscapes, ['stimulus_id', 'id', 'stimID'])
scene_col = find_column(soundscapes, ['scene', 'scene_name', 'Scene', 'description', 'label'])
masker_col = find_column(maskers, ['masker_id', 'id', 'maskerID'])
masker_type_col = find_column(maskers, ['masker_type', 'type', 'masker', 'Type'])
response_stimulus_col = find_column(responses, ['stimulus_id', 'id', 'stimID'])
response_masker_col = find_column(responses, ['masker_id', 'id', 'maskerID'])
pleasantness_col = find_column(responses, ['pleasantness', 'ISOPleasant', 'pleasantness_rating', 'rating'])

print(f"\nColumn Mapping:")
print(f"  Soundscapes: stimulus_id='{stimulus_col}', scene='{scene_col}'")
print(f"  Maskers: masker_id='{masker_col}', type='{masker_type_col}'")
print(f"  Responses: stimulus_id='{response_stimulus_col}', masker_id='{response_masker_col}', pleasantness='{pleasantness_col}'")

if not all([stimulus_col, scene_col, masker_col, masker_type_col, response_stimulus_col, pleasantness_col]):
    print("ERROR: Could not find all required columns!")
    exit(1)

# Analysis 1: Scene Diversity
print("\n" + "=" * 70)
print("ANALYSIS 1: SCENE DIVERSITY")
print("=" * 70)

scene_counts = soundscapes[scene_col].value_counts()
print(f"\nTotal unique scenes: {len(scene_counts)}")
print(f"Top 10 scenes by frequency:")
for scene, count in scene_counts.head(10).items():
    pct = count / len(soundscapes) * 100
    print(f"  {scene:40s}: {count:4d} ({pct:5.1f}%)")

# Analysis 2: Masker Type Diversity
print("\n" + "=" * 70)
print("ANALYSIS 2: MASKER TYPE DIVERSITY")
print("=" * 70)

masker_type_counts = maskers[masker_type_col].value_counts()
print(f"\nTotal unique masker types: {len(masker_type_counts)}")
print(f"Masker type distribution:")
for mtype, count in masker_type_counts.items():
    pct = count / len(maskers) * 100
    print(f"  {mtype:30s}: {count:4d} ({pct:5.1f}%)")

# Analysis 3: Pleasantness Distribution
print("\n" + "=" * 70)
print("ANALYSIS 3: PLEASANTNESS DISTRIBUTION")
print("=" * 70)

print(f"\nPleasantness statistics:")
print(f"  Min: {responses[pleasantness_col].min():.3f}")
print(f"  Max: {responses[pleasantness_col].max():.3f}")
print(f"  Mean: {responses[pleasantness_col].mean():.3f}")
print(f"  Median: {responses[pleasantness_col].median():.3f}")
print(f"  Std Dev: {responses[pleasantness_col].std():.3f}")

# Histogram of pleasantness
bins = pd.cut(responses[pleasantness_col], bins=10)
bin_counts = bins.value_counts().sort_index()
print(f"\nPleasantness histogram (10 bins):")
for interval, count in bin_counts.items():
    pct = count / len(responses) * 100
    bar = "█" * int(pct / 2)
    print(f"  {interval}: {count:5d} ({pct:5.1f}%) {bar}")

# Analysis 4: Stimulus Coverage (responses per stimulus)
print("\n" + "=" * 70)
print("ANALYSIS 4: STIMULUS COVERAGE")
print("=" * 70)

stimulus_response_counts = responses[response_stimulus_col].value_counts()
print(f"\nResponses per stimulus:")
print(f"  Total unique stimuli with responses: {len(stimulus_response_counts)}")
print(f"  Responses per stimulus - Min: {stimulus_response_counts.min()}")
print(f"  Responses per stimulus - Max: {stimulus_response_counts.max()}")
print(f"  Responses per stimulus - Mean: {stimulus_response_counts.mean():.1f}")

# Analysis 5: Scene-Masker Combinations
print("\n" + "=" * 70)
print("ANALYSIS 5: SCENE-MASKER COMBINATIONS")
print("=" * 70)

# Build mappings
stimulus_to_scene = dict(zip(soundscapes[stimulus_col], soundscapes[scene_col]))
masker_to_type = dict(zip(maskers[masker_col], maskers[masker_type_col]))

# Count combinations
combinations = Counter()
for _, row in responses.iterrows():
    stimulus_id = row[response_stimulus_col]
    masker_id = row[response_masker_col]

    scene = stimulus_to_scene.get(stimulus_id, "unknown")
    masker_type = masker_to_type.get(masker_id, "unknown")

    combinations[(scene, masker_type)] += 1

print(f"\nTotal unique scene-masker combinations: {len(combinations)}")
print(f"Top 15 combinations by response count:")
for (scene, masker_type), count in combinations.most_common(15):
    pct = count / len(responses) * 100
    print(f"  {scene:35s} + {masker_type:15s}: {count:5d} ({pct:5.1f}%)")

# Generate Report
print(f"\n✓ Saving detailed report to {REPORT_FILE}...")

with open(REPORT_FILE, 'w') as f:
    f.write("ARAUS DATASET DIVERSITY ANALYSIS REPORT\n")
    f.write("=" * 80 + "\n")
    f.write(f"Analysis date: 2026-10-04\n")
    f.write(f"Soundscapes: {len(soundscapes)} | Responses: {len(responses)} | Maskers: {len(maskers)}\n\n")

    # Scene diversity
    f.write("SCENE DIVERSITY\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total unique scenes: {len(scene_counts)}\n\n")
    f.write("Top 20 scenes:\n")
    for scene, count in scene_counts.head(20).items():
        pct = count / len(soundscapes) * 100
        f.write(f"  {scene:45s}: {count:4d} ({pct:5.1f}%)\n")

    # Masker diversity
    f.write("\n\nMASTER TYPE DIVERSITY\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total unique masker types: {len(masker_type_counts)}\n\n")
    for mtype, count in masker_type_counts.items():
        pct = count / len(maskers) * 100
        f.write(f"  {mtype:40s}: {count:4d} ({pct:5.1f}%)\n")

    # Pleasantness distribution
    f.write("\n\nPLEASANTNESS DISTRIBUTION\n")
    f.write("-" * 80 + "\n")
    f.write(f"Min: {responses[pleasantness_col].min():.3f}\n")
    f.write(f"Max: {responses[pleasantness_col].max():.3f}\n")
    f.write(f"Mean: {responses[pleasantness_col].mean():.3f}\n")
    f.write(f"Median: {responses[pleasantness_col].median():.3f}\n")
    f.write(f"Std Dev: {responses[pleasantness_col].std():.3f}\n\n")

    f.write("Histogram (10 bins):\n")
    for interval, count in bin_counts.items():
        pct = count / len(responses) * 100
        bar = "█" * int(pct / 2)
        f.write(f"  {str(interval):30s}: {count:5d} ({pct:5.1f}%) {bar}\n")

    # Top combinations
    f.write("\n\nTOP SCENE-MASKER COMBINATIONS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total unique combinations: {len(combinations)}\n\n")
    for (scene, masker_type), count in combinations.most_common(20):
        pct = count / len(responses) * 100
        f.write(f"  {scene:35s} + {masker_type:15s}: {count:5d} ({pct:5.1f}%)\n")

    # Recommendations
    f.write("\n\nRECOMMENDATIONS FOR DIVERSE TRAINING SUBSET\n")
    f.write("-" * 80 + "\n")
    f.write("""
To avoid the homogeneity problem (model learns one scene pattern and ignores
pleasantness conditioning), the training subset should:

1. SCENE DIVERSITY:
   - Select samples across multiple scene types, not just dominant ones
   - Ensure representation from top 5-8 scenes
   - Include some rare scenes for variety

2. MASKER TYPE DIVERSITY:
   - Include all or most masker types
   - Balance distribution across masker types
   - Ensure model sees different acoustic characteristics per pleasantness level

3. PLEASANTNESS COVERAGE:
   - Stratify across full pleasantness range (0.0-1.0)
   - Use 5-6 pleasantness bins with equal representation
   - Ensure model learns distinct acoustic patterns per pleasantness level

4. COMBINATION DIVERSITY:
   - Avoid over-representation of top combinations
   - Sample combinations uniformly from the space
   - Ensure each combination appears with multiple pleasantness levels

NEXT STEPS:
- Run 04_select_diverse_subset.py to create stratified diverse subset
- Extract audio files for selected stimulus IDs
- Re-encode with diverse captions reflecting pleasantness levels
- Re-train LoRA on diverse subset
""")

print(f"✓ Report saved")
print(f"\nSummary:")
print(f"  - {len(scene_counts)} unique scenes")
print(f"  - {len(masker_type_counts)} masker types")
print(f"  - {len(combinations)} scene-masker combinations")
print(f"  - Pleasantness range: {responses[pleasantness_col].min():.3f} to {responses[pleasantness_col].max():.3f}")
