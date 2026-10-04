#!/usr/bin/env python3
"""
Build diverse training dataset from diverse_subset.csv and encoded latents.

Takes the selected subset and maps it to encoded .json files, replacing captions
with diverse versions that reflect pleasantness levels.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Adjust these paths
DIVERSE_SUBSET_CSV = Path("/home/marcello/Documents/restorative_embeddings/diverse_subset.csv")
ENCODED_DIR = Path("/home/marcello/Documents/restorative_embeddings/generation/train/araus_for_sa3_encoded")
OUTPUT_DIR = Path("/home/marcello/Documents/restorative_embeddings/generation/train/diverse_training_data")

print("=" * 70)
print("BUILD DIVERSE TRAINING DATASET")
print("=" * 70)

# Load diverse subset
print(f"\nLoading diverse subset from {DIVERSE_SUBSET_CSV}...")
try:
    subset_df = pd.read_csv(DIVERSE_SUBSET_CSV)
    print(f"✓ Loaded {len(subset_df)} samples")
except Exception as e:
    print(f"ERROR: Could not load diverse_subset.csv: {e}")
    exit(1)

# Load all encoded files and index by stimulus_id + masker_id
print(f"\nIndexing encoded dataset from {ENCODED_DIR}...")
encoded_index = defaultdict(list)
encoded_data = {}

for json_file in sorted(ENCODED_DIR.glob("*.json")):
    try:
        with open(json_file) as f:
            data = json.load(f)

        # Extract stimulus_id and masker_id from filename or prompt
        # Filename format typically: stimulus_{id}_masker_{masker_id}.json
        filename = json_file.stem  # Remove .json
        parts = filename.split('_')

        # Try to extract from filename
        stimulus_id = None
        masker_id = None

        try:
            # Try parsing: "stimulus_XXX_masker_YYY"
            if 'stimulus' in filename and 'masker' in filename:
                stim_idx = parts.index('stimulus')
                mask_idx = parts.index('masker')
                if stim_idx + 1 < len(parts):
                    stimulus_id = parts[stim_idx + 1]
                if mask_idx + 1 < len(parts):
                    masker_id = parts[mask_idx + 1]
        except:
            pass

        # Fallback: try to extract from prompt
        if 'prompt' in data:
            prompt = data['prompt']
            # Extract from prompt if available
            if '[' in prompt:
                # Prompt might contain metadata
                pass

        # Store data
        encoded_data[json_file.stem] = data
        if stimulus_id and masker_id:
            encoded_index[(stimulus_id, masker_id)].append(json_file.stem)

    except Exception as e:
        print(f"  Warning: Error reading {json_file.name}: {e}")

print(f"✓ Indexed {len(encoded_data)} encoded files")
print(f"✓ Found {len(encoded_index)} unique stimulus-masker combinations")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n✓ Output directory: {OUTPUT_DIR}")

# Process diverse subset
print(f"\nBuilding diverse training dataset...")
matched = 0
unmatched = 0
created_files = []

for idx, row in subset_df.iterrows():
    stimulus_id = str(row['stimulus_id'])
    masker_id = str(row['masker_id'])
    new_caption = row['new_caption']

    # Find encoded file for this combination
    key = (stimulus_id, masker_id)
    if key in encoded_index:
        # Get first matching file (usually only one)
        encoded_filename = encoded_index[key][0]
        encoded_data_item = encoded_data[encoded_filename]

        # Create new JSON with same latent but new caption
        new_data = encoded_data_item.copy()
        new_data['prompt'] = new_caption

        # Save to output directory
        output_filename = OUTPUT_DIR / f"{encoded_filename}.json"
        with open(output_filename, 'w') as f:
            json.dump(new_data, f)

        created_files.append(output_filename)
        matched += 1

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(subset_df)}...")
    else:
        unmatched += 1
        print(f"  Warning: No encoded file found for stimulus={stimulus_id}, masker={masker_id}")

print(f"\n✓ Successfully created diverse training dataset:")
print(f"  - Matched: {matched} samples")
print(f"  - Unmatched: {unmatched} samples")
print(f"  - Output directory: {OUTPUT_DIR}")
print(f"  - Files: {len(created_files)}")

# Verify sample captions
print(f"\n=== SAMPLE CAPTIONS (first 10) ===")
for idx, row in subset_df.head(10).iterrows():
    print(f"\n{idx + 1}. {row['scene']} + {row['masker_type']} (Bin {row['pleasantness_bin']}):")
    print(f"   {row['new_caption']}")

# Create a metadata file
metadata = {
    'original_subset_csv': str(DIVERSE_SUBSET_CSV),
    'source_encoded_dir': str(ENCODED_DIR),
    'total_samples': len(subset_df),
    'matched_samples': matched,
    'unmatched_samples': unmatched,
    'output_files': len(created_files)
}

metadata_file = OUTPUT_DIR / "metadata.json"
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ Saved metadata to: {metadata_file}")

print(f"\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print(f"""
1. Verify generated captions look diverse:
   - Sample diverse_subset.csv should show different pleasantness language
   - Check caption_diversity_report.txt for examples

2. Train LoRA with diverse dataset:
   python train_lora.py --dataset {OUTPUT_DIR} --epochs 10

3. Test results:
   - Generate audio with different pleasantness levels
   - Verify birds appear in parks, not traffic
   - Verify acoustic changes with pleasantness level

4. If still not working:
   - Increase SAMPLES_PER_BIN in 04_select_diverse_subset.py
   - Adjust PLEASANTNESS_BINS (try 3 bins for more extreme differences)
   - Increase training epochs
""")
