#!/usr/bin/env python3
"""
Collect all prompts from the training dataset and save to a file for inspection.
Shows prompt diversity and frequency analysis.
"""

import json
import os
from pathlib import Path
from collections import Counter

# Adjust these paths
ENCODED_DIR = Path("/home/marcello/Documents/restorative_embeddings/generation/train/araus_for_sa3_encoded")
OUTPUT_FILE = Path("/home/marcello/Documents/restorative_embeddings/all_prompts.txt")
STATS_FILE = Path("/home/marcello/Documents/restorative_embeddings/prompt_stats.txt")

print("Reading prompts from encoded dataset...")

prompts = []
errors = 0

# Read all .json files
for json_file in sorted(ENCODED_DIR.glob("*.json")):
    try:
        with open(json_file) as f:
            data = json.load(f)
            if "prompt" in data:
                prompts.append(data["prompt"])
    except Exception as e:
        errors += 1
        print(f"  Error reading {json_file}: {e}")

print(f"✓ Found {len(prompts)} prompts ({errors} errors)")

# Save all prompts to file
with open(OUTPUT_FILE, "w") as f:
    for i, prompt in enumerate(prompts, 1):
        f.write(f"{i:5d}. {prompt}\n")

print(f"✓ Saved to: {OUTPUT_FILE}")

# Analyze diversity
print("\n=== PROMPT ANALYSIS ===")
print(f"Total prompts: {len(prompts)}")
print(f"Unique prompts: {len(set(prompts))}")
print(f"Diversity: {len(set(prompts)) / len(prompts) * 100:.1f}%")

# Count frequency of key phrases
print("\n=== MOST COMMON PHRASES ===")
phrases = Counter()
for prompt in prompts:
    # Extract words before [ISOPleasant
    if "[ISOPleasant" in prompt:
        caption = prompt.split("[ISOPleasant")[0].strip()
        # Count keywords
        for word in caption.lower().split():
            if len(word) > 3:  # Only words longer than 3 chars
                phrases[word] += 1

for phrase, count in phrases.most_common(20):
    pct = count / len(prompts) * 100
    print(f"  {phrase:20s}: {count:4d} times ({pct:5.1f}%)")

# Check for repetition patterns
print("\n=== REPETITION ANALYSIS ===")
prompt_counts = Counter(prompts)
most_repeated = prompt_counts.most_common(10)

print(f"Prompts appearing >1 time: {sum(1 for c in prompt_counts.values() if c > 1)}")
print(f"\nTop 10 repeated prompts:")
for prompt, count in most_repeated:
    if count > 1:
        print(f"  [{count:3d}x] {prompt[:80]}")

# Save stats
with open(STATS_FILE, "w") as f:
    f.write(f"DATASET STATISTICS\n")
    f.write(f"==================\n\n")
    f.write(f"Total prompts: {len(prompts)}\n")
    f.write(f"Unique prompts: {len(set(prompts))}\n")
    f.write(f"Diversity: {len(set(prompts)) / len(prompts) * 100:.1f}%\n\n")

    f.write(f"TOP 20 MOST COMMON PHRASES:\n")
    for phrase, count in phrases.most_common(20):
        pct = count / len(prompts) * 100
        f.write(f"  {phrase:20s}: {count:4d} times ({pct:5.1f}%)\n")

    f.write(f"\n\nTOP REPEATED PROMPTS:\n")
    for prompt, count in most_repeated:
        if count > 1:
            f.write(f"  [{count:3d}x] {prompt}\n")

print(f"✓ Saved stats to: {STATS_FILE}")
print(f"\nInspect prompts with: cat {OUTPUT_FILE}")
