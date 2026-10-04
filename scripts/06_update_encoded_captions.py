#!/usr/bin/env python3
"""
Update captions in existing encoded dataset to be more diverse.
Works directly with the pre-encoded JSON files organized by pleasantness.
Also copies corresponding .npy latent files to the output directory.
"""

import json
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Paths
ENCODED_DIR = Path("/home/marcello/Documents/restorative_embeddings/generation/train/araus_for_sa3_encoded")
OUTPUT_DIR = Path("/home/marcello/Documents/restorative_embeddings/generation/train/diverse_training_data")

print("=" * 70)
print("UPDATE ENCODED CAPTIONS - DIVERSE DATASET (WITH .NPY FILES)")
print("=" * 70)

# Load all encoded files grouped by pleasantness folder
print("\nLoading encoded dataset...")
files_by_pleasantness = defaultdict(list)
all_files = []

pleasantness_folder_map = {
    'very_unpleasant': 0,
    'unpleasant': 1,
    'neutral': 2,
    'pleasant': 3,
    'very_pleasant': 4
}

for json_file in sorted(ENCODED_DIR.glob("*.json")):
    try:
        with open(json_file) as f:
            data = json.load(f)

        # Extract pleasantness folder from relpath
        relpath = data.get('relpath', '')
        pleasantness_folder = relpath.split('/')[0] if '/' in relpath else 'unknown'
        bin_id = pleasantness_folder_map.get(pleasantness_folder, 2)  # default to neutral

        files_by_pleasantness[bin_id].append((json_file, data, pleasantness_folder))
        all_files.append((json_file, data, pleasantness_folder, bin_id))

    except Exception as e:
        print(f"  Error reading {json_file}: {e}")

print(f"✓ Loaded {len(all_files)} encoded files")
print("\nDistribution by pleasantness:")
for bin_id in range(5):
    count = len(files_by_pleasantness[bin_id])
    folder_names = [f for (_, _, f) in files_by_pleasantness[bin_id]]
    if folder_names:
        folder_name = folder_names[0]
        print(f"  Bin {bin_id} ({folder_name:20s}): {count:4d} files")

# Select diverse subset (20 per bin)
SAMPLES_PER_BIN = 20
selected_files = []

print(f"\nSelecting {SAMPLES_PER_BIN} samples per bin...")
for bin_id in range(5):
    files_in_bin = files_by_pleasantness[bin_id]
    if len(files_in_bin) == 0:
        print(f"  Bin {bin_id}: No files available")
        continue

    # Randomly sample from this bin
    sample_size = min(SAMPLES_PER_BIN, len(files_in_bin))
    sampled = random.sample(files_in_bin, sample_size)
    selected_files.extend([(f, d, b, bin_id) for f, d, b in sampled])

    print(f"  Bin {bin_id}: Selected {sample_size} files")

print(f"\n✓ Selected {len(selected_files)} total files")

# Generate diverse captions
def generate_diverse_caption(bin_id, current_caption):
    """Generate a diverse caption based on pleasantness bin"""

    # Pleasantness descriptors
    pleasantness_phrases = {
        0: [
            "Very unpleasant audio environment",
            "Harsh and disturbing soundscape",
            "Strongly unpleasant acoustic experience",
            "Disruptive and uncomfortable sound",
            "Highly unpleasant listening environment"
        ],
        1: [
            "Unpleasant sound environment",
            "Uncomfortable acoustic atmosphere",
            "Somewhat displeasing soundscape",
            "Moderately unpleasant audio",
            "Disagreeable listening experience"
        ],
        2: [
            "Neutral sound environment",
            "Moderate acoustic atmosphere",
            "Balanced soundscape",
            "Acceptable listening experience",
            "Neither pleasant nor unpleasant"
        ],
        3: [
            "Pleasant sound environment",
            "Enjoyable acoustic atmosphere",
            "Agreeable soundscape",
            "Satisfying listening experience",
            "Positive audio environment"
        ],
        4: [
            "Very pleasant sound environment",
            "Highly enjoyable acoustic atmosphere",
            "Delightful soundscape",
            "Extremely satisfying listening experience",
            "Wonderful audio environment"
        ]
    }

    # Extract pleasantness value from current caption if available
    pleasantness_val = None
    if "[ISOPleasant:" in current_caption:
        try:
            val_str = current_caption.split("[ISOPleasant:")[1].split("]")[0].strip()
            pleasantness_val = float(val_str)
        except:
            pass

    phrase = random.choice(pleasantness_phrases.get(bin_id, pleasantness_phrases[2]))

    # Create caption
    caption = phrase + "."
    if pleasantness_val is not None:
        caption += f" [ISOPleasant: {pleasantness_val:.2f}]"

    return caption

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n✓ Output directory: {OUTPUT_DIR}")

# Process and save files with updated captions and copy .npy files
print("\nUpdating captions and copying latent files...")
saved_count = 0
missing_npy_count = 0

for json_file, data, pleasantness_folder, bin_id in selected_files:
    try:
        # Update caption
        old_caption = data.get('prompt', '')
        new_caption = generate_diverse_caption(bin_id, old_caption)
        data['prompt'] = new_caption

        # Save updated JSON to output directory
        output_json_file = OUTPUT_DIR / json_file.name
        with open(output_json_file, 'w') as f:
            json.dump(data, f)

        # Copy corresponding .npy file (if it exists)
        npy_file = json_file.with_suffix('.npy')
        if npy_file.exists():
            output_npy_file = OUTPUT_DIR / npy_file.name
            shutil.copy2(npy_file, output_npy_file)
        else:
            missing_npy_count += 1
            print(f"  Warning: No .npy file for {json_file.name}")

        saved_count += 1
        if saved_count % 20 == 0:
            print(f"  Processed {saved_count}/{len(selected_files)}...")

    except Exception as e:
        print(f"  Error processing {json_file.name}: {e}")

print(f"\n✓ Saved {saved_count} files with updated captions")
if missing_npy_count > 0:
    print(f"  ⚠ Warning: {missing_npy_count} .npy files were missing")

# Print sample captions
print("\n" + "=" * 70)
print("SAMPLE CAPTIONS BY PLEASANTNESS BIN")
print("=" * 70)

bins_sampled = defaultdict(list)
for json_file, data, _, bin_id in selected_files[:30]:
    bins_sampled[bin_id].append(data['prompt'])

for bin_id in range(5):
    if bins_sampled[bin_id]:
        bin_names = {0: "Very Unpleasant", 1: "Unpleasant", 2: "Neutral", 3: "Pleasant", 4: "Very Pleasant"}
        print(f"\nBin {bin_id} ({bin_names[bin_id]}):")
        for caption in bins_sampled[bin_id][:3]:
            print(f"  - {caption}")

print("\n" + "=" * 70)
print("VERIFY DATASET")
print("=" * 70)

json_count = len(list(OUTPUT_DIR.glob("*.json")))
npy_count = len(list(OUTPUT_DIR.glob("*.npy")))
print(f"Files in output directory:")
print(f"  .json files: {json_count}")
print(f"  .npy files: {npy_count}")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print(f"""
1. Training data ready at: {OUTPUT_DIR}
   Contains {saved_count} file pairs (json + npy)

2. Train LoRA with this diverse dataset:
   python train_lora.py --model medium-base \\
     --data_dir {OUTPUT_DIR} \\
     --rank 16 --adapter_type dora-rows \\
     --steps 2000 --batch_size 16 --logger csv

3. Expected improvements:
   - Model learns acoustic differences per pleasantness level
   - Birds should NOT appear in unpleasant/traffic scenarios
   - Different pleasantness levels produce distinct sounds
""")
