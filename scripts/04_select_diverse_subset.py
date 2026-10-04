#!/usr/bin/env python3
"""
Select a diverse subset of ARAUS data stratified by pleasantness levels.
Generates diverse captions for each sample to enable learning pleasantness conditioning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import random

# Adjust these paths
DATA_DIR = Path("/home/marcello/Documents/restorative_embeddings/data/raw/araus/data")
OUTPUT_CSV = Path("/home/marcello/Documents/restorative_embeddings/diverse_subset.csv")
CAPTION_REPORT = Path("/home/marcello/Documents/restorative_embeddings/caption_diversity_report.txt")

# Configuration
PLEASANTNESS_BINS = 5  # Number of pleasantness bins (0.0-0.2, 0.2-0.4, etc)
SAMPLES_PER_BIN = 20    # Target samples per pleasantness bin (adjust based on available data)
MIN_SCENES_PER_BIN = 3  # Minimum scene types per bin
MIN_MASKERS_PER_BIN = 3 # Minimum masker type variety per bin

print("=" * 60)
print("ARAUS DIVERSE SUBSET SELECTOR")
print("=" * 60)

# Load data
print("\nLoading ARAUS metadata CSVs...")
try:
    soundscapes = pd.read_csv(DATA_DIR / "soundscapes.csv")
    responses = pd.read_csv(DATA_DIR / "responses.csv")
    maskers = pd.read_csv(DATA_DIR / "maskers.csv")
except Exception as e:
    print(f"ERROR: Could not load CSVs: {e}")
    exit(1)

print(f"✓ Soundscapes: {len(soundscapes)} rows")
print(f"✓ Responses: {len(responses)} rows")
print(f"✓ Maskers: {len(maskers)} rows")

# ARAUS-specific column mappings
stimulus_col = 'soundscape'        # Soundscape ID
scene_col = 'soundscape'           # Same as stimulus (soundscape name is the scene)
masker_col = 'masker'              # Masker ID
masker_type_col = 'class'          # Masker class/type
response_stimulus_col = 'soundscape'  # Soundscape in responses
response_masker_col = 'masker'     # Masker in responses
pleasantness_col = 'pleasant'      # Pleasantness rating (ARAUS uses 'pleasant' rating)

print(f"\nColumn mapping (ARAUS dataset):")
print(f"  Soundscapes: stimulus='{stimulus_col}', scene='{scene_col}'")
print(f"  Maskers: masker_id='{masker_col}', type='{masker_type_col}'")
print(f"  Responses: stimulus='{response_stimulus_col}', masker='{response_masker_col}', pleasantness='{pleasantness_col}'")

# Verify columns exist
if pleasantness_col not in responses.columns:
    print(f"ERROR: Column '{pleasantness_col}' not found in responses!")
    print(f"Available columns: {list(responses.columns)}")
    exit(1)

# Build stimulus -> scene mapping
stimulus_to_scene = dict(zip(soundscapes[stimulus_col], soundscapes[stimulus_col]))
print(f"\n✓ Mapped {len(stimulus_to_scene)} soundscapes")

# Build masker -> type mapping
masker_to_type = dict(zip(maskers[masker_col], maskers[masker_type_col]))
print(f"✓ Mapped {len(masker_to_type)} maskers to types")

# Print sample masker types
print(f"\nSample masker types:")
for masker_id, mtype in list(masker_to_type.items())[:10]:
    print(f"  {masker_id}: {mtype}")

# Analyze response data
print(f"\nAnalyzing {len(responses)} responses...")
responses_copy = responses.copy()

# Handle pleasantness column - may be on different scale
print(f"\nPleasantness statistics:")
print(f"  Min: {responses_copy[pleasantness_col].min():.3f}")
print(f"  Max: {responses_copy[pleasantness_col].max():.3f}")
print(f"  Mean: {responses_copy[pleasantness_col].mean():.3f}")

# Normalize pleasantness to 0-1 if needed
if responses_copy[pleasantness_col].max() > 1.5:
    print(f"  Normalizing pleasantness from {responses_copy[pleasantness_col].min()}-{responses_copy[pleasantness_col].max()} to 0-1 scale...")
    responses_copy[pleasantness_col] = (responses_copy[pleasantness_col] - responses_copy[pleasantness_col].min()) / (responses_copy[pleasantness_col].max() - responses_copy[pleasantness_col].min())

# Create bins
responses_copy['pleasantness_bin'] = pd.cut(
    responses_copy[pleasantness_col],
    bins=PLEASANTNESS_BINS,
    labels=range(PLEASANTNESS_BINS)
).astype(int)

print(f"\nPleasantness distribution across {PLEASANTNESS_BINS} bins:")
bin_counts = responses_copy['pleasantness_bin'].value_counts().sort_index()
for bin_id in range(PLEASANTNESS_BINS):
    count = bin_counts.get(bin_id, 0)
    pct = count / len(responses_copy) * 100 if len(responses_copy) > 0 else 0
    print(f"  Bin {bin_id}: {count:5d} responses ({pct:5.1f}%)")

# Stratified sampling: select diverse samples across bins
selected_samples = []
scene_types_by_bin = defaultdict(set)
masker_types_by_bin = defaultdict(set)

print(f"\nSelecting {SAMPLES_PER_BIN} samples per bin (stratified)...")
for bin_id in range(PLEASANTNESS_BINS):
    bin_data = responses_copy[responses_copy['pleasantness_bin'] == bin_id]

    if len(bin_data) == 0:
        print(f"  Bin {bin_id}: No data available")
        continue

    # Group by (stimulus, masker) to ensure variety
    grouped = bin_data.groupby([response_stimulus_col, response_masker_col])
    unique_combos = list(grouped.groups.keys())

    # Sample diverse combos from this bin
    sample_size = min(SAMPLES_PER_BIN, len(unique_combos))
    sampled_combos = random.sample(unique_combos, sample_size)

    for stimulus_id, masker_id in sampled_combos:
        # Get all responses for this combo to compute avg pleasantness
        combo_data = bin_data[
            (bin_data[response_stimulus_col] == stimulus_id) &
            (bin_data[response_masker_col] == masker_id)
        ]

        avg_pleasantness = combo_data[pleasantness_col].mean()

        scene = stimulus_to_scene.get(stimulus_id, str(stimulus_id))
        masker_type = masker_to_type.get(masker_id, str(masker_id))

        selected_samples.append({
            'stimulus_id': stimulus_id,
            'masker_id': masker_id,
            'scene': scene,
            'masker_type': masker_type,
            'pleasantness_bin': bin_id,
            'pleasantness_rating': avg_pleasantness
        })

        scene_types_by_bin[bin_id].add(scene)
        masker_types_by_bin[bin_id].add(masker_type)

    print(f"  Bin {bin_id}: Selected {len(sampled_combos)} samples")
    print(f"    Masker types: {', '.join(sorted(list(masker_types_by_bin[bin_id])[:5]))}...")

print(f"\n✓ Selected {len(selected_samples)} total samples")

# Generate diverse captions
def generate_caption(row):
    """Generate a diverse caption based on scene, masker, and pleasantness level"""
    scene = row['scene']
    masker_type = row['masker_type']
    bin_id = row['pleasantness_bin']
    rating = row['pleasantness_rating']

    # Pleasantness descriptors
    pleasantness_phrases = {
        0: ["very unpleasant", "harsh and disruptive", "strongly unpleasant", "disturbing"],
        1: ["unpleasant", "uncomfortable", "somewhat unpleasant", "displeasing"],
        2: ["neutral", "moderate", "acceptable", "neither pleasant nor unpleasant"],
        3: ["pleasant", "enjoyable", "agreeable", "satisfying"],
        4: ["very pleasant", "highly enjoyable", "delightful", "extremely satisfying"]
    }

    # Scene descriptors (using soundscape/masker combination)
    scene_descriptors = [
        f"soundscape with {masker_type}",
        f"{masker_type} environment",
        f"audio with {masker_type} sounds",
        f"{masker_type} atmosphere",
        f"listening to {masker_type}",
    ]

    # Select descriptors
    pleasantness_phrase = random.choice(pleasantness_phrases.get(bin_id, ["neutral"]))
    scene_descriptor = random.choice(scene_descriptors)

    # Combine
    caption = f"{pleasantness_phrase} {scene_descriptor}"
    caption = caption[0].upper() + caption[1:]  # Capitalize
    caption += f". [ISOPleasant: {rating:.2f}]"

    return caption

random.seed(42)  # For reproducibility
selected_df = pd.DataFrame(selected_samples)
selected_df['new_caption'] = selected_df.apply(generate_caption, axis=1)

# Save subset
selected_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Saved diverse subset to: {OUTPUT_CSV}")
print(f"  Columns: {list(selected_df.columns)}")

# Generate report
with open(CAPTION_REPORT, 'w') as f:
    f.write("DIVERSE SUBSET CAPTION GENERATION REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Total samples selected: {len(selected_df)}\n")
    f.write(f"Pleasantness bins: {PLEASANTNESS_BINS}\n")
    f.write(f"Target samples per bin: {SAMPLES_PER_BIN}\n\n")

    f.write("SAMPLE BREAKDOWN BY PLEASANTNESS BIN:\n")
    f.write("-" * 70 + "\n")
    for bin_id in range(PLEASANTNESS_BINS):
        bin_samples = selected_df[selected_df['pleasantness_bin'] == bin_id]
        if len(bin_samples) > 0:
            maskers = bin_samples['masker_type'].unique()
            f.write(f"\nBin {bin_id} ({len(bin_samples)} samples):\n")
            f.write(f"  Masker types ({len(maskers)}): {', '.join(sorted(maskers)[:8])}\n")
            f.write(f"  Pleasantness range: {bin_samples['pleasantness_rating'].min():.2f} - {bin_samples['pleasantness_rating'].max():.2f}\n")

    f.write("\n\nSAMPLE CAPTIONS:\n")
    f.write("-" * 70 + "\n")
    for idx, row in selected_df.head(20).iterrows():
        f.write(f"\n{idx+1}. {row['masker_type']} (Bin {row['pleasantness_bin']}, rating={row['pleasantness_rating']:.2f}):\n")
        f.write(f"   {row['new_caption']}\n")

    f.write("\n\nNEXT STEPS:\n")
    f.write("-" * 70 + "\n")
    f.write("1. Review the subset to ensure diversity across pleasantness levels\n")
    f.write("2. Extract audio files for selected stimulus_id + masker_id combinations\n")
    f.write("3. Pre-encode selected files to latents using new captions\n")
    f.write("4. Create training dataset from diverse_subset.csv\n")
    f.write("5. Re-train LoRA with diverse dataset\n")

print(f"✓ Saved report to: {CAPTION_REPORT}")

# Print summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Selected {len(selected_df)} diverse samples across {PLEASANTNESS_BINS} pleasantness bins")
print(f"Output CSV: {OUTPUT_CSV}")
print(f"Report: {CAPTION_REPORT}")
print("\nNext steps:")
print("1. Extract audio for selected stimulus IDs from ARAUS dataset")
print("2. Pre-encode selected audio files with new diverse captions")
print("3. Re-train LoRA with diverse_subset.csv")
