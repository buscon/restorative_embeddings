#!/usr/bin/env python3
"""
Analyze ARAUS dataset diversity and generate a diverse training subset.

Reads the raw ARAUS CSV files and creates:
1. Diversity analysis (scenes, maskers, pleasantness distribution)
2. A balanced subset for training (diverse scenes, equal pleasantness bins)
3. Unique captions for each file
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import json

# Paths
DATA_DIR = Path("/Users/marcellolussana/Documents/Bamberg/restorative_embeddings/data/raw/araus/data")
OUTPUT_DIR = Path("/Users/marcellolussana/Documents/Bamberg/restorative_embeddings")

# Load ARAUS CSVs
print("Loading ARAUS metadata...")
soundscapes = pd.read_csv(DATA_DIR / "soundscapes.csv")
responses = pd.read_csv(DATA_DIR / "responses.csv")
maskers = pd.read_csv(DATA_DIR / "maskers.csv")
participants = pd.read_csv(DATA_DIR / "participants.csv")

print(f"✓ Loaded {len(soundscapes)} soundscapes")
print(f"✓ Loaded {len(responses)} responses")
print(f"✓ Loaded {len(maskers)} masker types")
print(f"✓ Loaded {len(participants)} participants")

# Analyze soundscape types
print("\n=== SOUNDSCAPE DIVERSITY ===")
if 'scene' in soundscapes.columns:
    scene_counts = soundscapes['scene'].value_counts()
    print(f"Total unique scenes: {len(scene_counts)}")
    print(f"Top 15 scenes:")
    for scene, count in scene_counts.head(15).items():
        print(f"  {scene:30s}: {count:3d}")
elif 'description' in soundscapes.columns:
    print(f"Soundscape descriptions available")
    print(soundscapes[['id', 'description']].head(10))
else:
    print(f"Soundscapes columns: {soundscapes.columns.tolist()}")

# Analyze masker types
print("\n=== MASKER TYPE DIVERSITY ===")
if 'type' in maskers.columns:
    masker_counts = maskers['type'].value_counts()
    print(f"Total masker types: {len(masker_counts)}")
    for masker, count in masker_counts.items():
        print(f"  {masker:20s}: {count:3d}")
else:
    print(f"Masker columns: {maskers.columns.tolist()}")

# Analyze pleasantness ratings
print("\n=== PLEASANTNESS DISTRIBUTION ===")
if 'iso_pleasantness' in responses.columns:
    iso_col = 'iso_pleasantness'
elif 'ISOPleasant' in responses.columns:
    iso_col = 'ISOPleasant'
else:
    iso_col = [c for c in responses.columns if 'pleasant' in c.lower()][0] if any('pleasant' in c.lower() for c in responses.columns) else None

if iso_col:
    pleasantness = responses[iso_col].dropna()
    print(f"Total ratings: {len(pleasantness)}")
    print(f"  Mean: {pleasantness.mean():.3f}")
    print(f"  Std: {pleasantness.std():.3f}")
    print(f"  Min: {pleasantness.min():.3f}, Max: {pleasantness.max():.3f}")
    
    # Bin distribution
    bins = [-1.0, -0.5, -0.15, 0.15, 0.5, 1.0]
    bin_labels = ['Very Unpleasant', 'Unpleasant', 'Neutral', 'Pleasant', 'Very Pleasant']
    counts = pd.cut(pleasantness, bins=bins, labels=bin_labels).value_counts().sort_index()
    print(f"\nBin distribution:")
    for label, count in counts.items():
        pct = count / len(pleasantness) * 100
        print(f"  {label:20s}: {count:5d} ({pct:5.1f}%)")
else:
    print(f"Response columns: {responses.columns.tolist()}")

# Analyze stimulus coverage
print("\n=== STIMULUS COVERAGE ===")
if 'stimulus_id' in responses.columns:
    stim_counts = responses['stimulus_id'].value_counts()
    print(f"Total unique stimuli rated: {len(stim_counts)}")
    print(f"  Ratings per stimulus:")
    print(f"    Mean: {stim_counts.mean():.2f}")
    print(f"    Min: {stim_counts.min()}, Max: {stim_counts.max()}")
    print(f"    Median: {stim_counts.median():.1f}")
else:
    print(f"No stimulus_id column. Available: {responses.columns.tolist()}")

# Save analysis
print("\n=== SAVING ANALYSIS ===")
analysis_file = OUTPUT_DIR / "araus_analysis.txt"
with open(analysis_file, "w") as f:
    f.write("ARAUS DATASET ANALYSIS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total soundscapes: {len(soundscapes)}\n")
    f.write(f"Total responses: {len(responses)}\n")
    f.write(f"Total participants: {len(participants)}\n\n")
    
    if 'scene' in soundscapes.columns:
        f.write("SCENE DISTRIBUTION:\n")
        for scene, count in soundscapes['scene'].value_counts().items():
            f.write(f"  {scene}: {count}\n")
    
    if 'type' in maskers.columns:
        f.write("\nMASTER TYPES:\n")
        for masker, count in maskers['type'].value_counts().items():
            f.write(f"  {masker}: {count}\n")

print(f"✓ Saved analysis to: {analysis_file}")

# Next step: create subset selection logic
print("\n=== NEXT STEPS ===")
print("1. Analyze the above to identify diverse scenes and maskers")
print("2. Stratify by pleasantness bins (balanced representation)")
print("3. Select subset avoiding repetitive captions")
print("4. Generate varied descriptions for each pleasantness level")
