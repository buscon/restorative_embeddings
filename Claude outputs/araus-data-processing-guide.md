# ARAUS Dataset Processing for Stable Audio 3 LoRA Training

Complete guide to preparing the ARAUS (Affective Reactions to Audio Urban Soundscapes) dataset for Stable Audio 3 LoRA fine-tuning.

## Overview

The ARAUS dataset consists of ~6000 restorative soundscape audio clips with paired captions describing acoustic characteristics and perceived pleasantness. The original dataset contains:

- **Base clips:** ~250 unique natural soundscapes (park, water, forest, etc.)
- **Acoustic modifications:** Each base clip mixed with different sound sources (traffic, construction, wind, water, birds) at varying levels (-6, 0, 3, 6 dB)
- **Pleasantness ratings:** Organized into 5 categories: very_unpleasant, unpleasant, neutral, pleasant, very_pleasant
- **Loudness measurements:** LA50 equivalent continuous sound pressure level (dB)
- **Caption metadata:** Structured descriptions with ISO pleasantness scores

## Directory Structure

Final organized dataset for training:

```
~/Documents/restorative_embeddings/generation/train/araus_for_sa3/
├── neutral/
│   ├── R0001_segment_binaural_44100_1.wav
│   ├── R0001_segment_binaural_44100_1.txt
│   ├── R0001_segment_binaural_44100_2.wav
│   ├── R0001_segment_binaural_44100_2.txt
│   ├── ...
├── pleasant/
│   ├── R0002_segment_binaural_44100_1.wav
│   ├── R0002_segment_binaural_44100_1.txt
│   └── ...
├── unpleasant/
│   └── ...
├── very_pleasant/
│   └── ...
└── very_unpleasant/
    └── ...
```

**Total:** ~1200 files per pleasantness category (~6000 total files)

## Raw Data Format

Original ARAUS dataset comes with:

1. **Audio files:** WAV format, 44.1 kHz, stereo (binaural), 10 seconds each
2. **Metadata files:** CSV containing measurements and pleasantness ratings
3. **Caption files (optional):** Text descriptions in various formats

Common original metadata columns:
- `clip_id`: Unique identifier (e.g., "R0001")
- `sound_source`: Base sound type (park, water, forest, etc.)
- `modification`: Added sound (birds, traffic, construction, wind, water)
- `modification_level`: SPL level (-6, 0, 3, 6 dB)
- `la50`: Loudness measurement (40-75 dB range)
- `iso_pleasantness`: Pleasantness score (-1.0 to 1.0)
- `perceived_pleasantness`: Category label (unpleasant, neutral, pleasant, etc.)

## Processing Steps

### Step 1: Audio Format Verification

Ensure all audio files meet requirements:

```bash
# Check audio properties
for file in *.wav; do
  ffprobe -v error -show_entries format=duration,sample_rate -of default=noprint_wrappers=1 "$file"
done
```

**Requirements:**
- ✅ Format: WAV
- ✅ Sample rate: 44.1 kHz (resample if different: `sox input.wav -r 44100 output.wav`)
- ✅ Duration: ~10 seconds (silence-trim and pad as needed)
- ✅ Channels: Stereo or mono (dowmix stereo to mono if necessary: `sox -M input.wav output.wav remix 1,2`)

### Step 2: Caption Generation

For each audio file, create a paired `.txt` caption file with the exact structure:

```
{scene_description}. Loudness: {loudness_descriptor} (LA50: {db_value} dB). Pleasantness: {pleasantness_descriptor} (ISOPleasant: {iso_score}). [pleasantness: {category_tag}]
```

#### Caption Components

**Scene Description (from original metadata):**
- Extract from `sound_source` and `modification` fields
- Format: `"{base_source} with {modification_type}"`
- Examples:
  - "quiet park with natural ambience and birdsong"
  - "urban street with traffic and pedestrian sounds"
  - "flowing water with ripples and splashing"
  - "natural forest ambience with wind rustling through leaves"

**Loudness Descriptor (from LA50 value):**

Map LA50 dB ranges to descriptive language:

| dB Range | Descriptor |
|----------|------------|
| 40-46 | quiet |
| 46-52 | moderately quiet |
| 52-58 | moderate |
| 58-64 | moderately loud |
| 64-70 | loud |
| 70+ | very loud |

**Loudness Examples:**
- `LA50: 48.1 dB → "quiet"`
- `LA50: 52.4 dB → "moderately quiet"`
- `LA50: 65.8 dB → "moderately loud"`

**Pleasantness Descriptor (from ISO score & category):**

Map ISO pleasantness scores to descriptive language:

| ISO Score | Category | Descriptor |
|-----------|----------|------------|
| -1.0 to -0.5 | very_unpleasant | very unpleasant pleasantness |
| -0.5 to -0.1 | unpleasant | unpleasant pleasantness |
| -0.1 to 0.1 | neutral | neutral |
| 0.1 to 0.5 | pleasant | pleasant |
| 0.5 to 1.0 | very_pleasant | very pleasant |

**Pleasantness Examples:**
- `ISOPleasant: -0.25 → "unpleasant pleasantness"`
- `ISOPleasant: 0.0 → "neutral"`
- `ISOPleasant: 0.25 → "pleasant"`
- `ISOPleasant: 0.63 → "very pleasant"`

#### Sample Caption Generation

**Input metadata:**
```
clip_id: R0003_segment_binaural_44100_1
sound_source: park
modification: birdsong
la50: 52.4
iso_pleasantness: 0.25
perceived_pleasantness: pleasant
```

**Generated caption:**
```
quiet park with natural ambience and birdsong. Loudness: moderately quiet (LA50: 52.4 dB). Pleasantness: pleasant (ISOPleasant: 0.25). [pleasantness: pleasant]
```

### Step 3: Pleasantness-Based Organization

Organize files into pleasantness subdirectories:

```bash
# Pseudocode for organizing by pleasantness
for each audio_file:
  pleasantness_category = metadata[audio_file].perceived_pleasantness
  destination_dir = araus_for_sa3/{pleasantness_category}/
  
  # Copy audio and caption
  cp audio_file destination_dir/
  cp caption_file destination_dir/
```

**Mapping:**
- ISOPleasant < -0.4 → `very_unpleasant/`
- -0.4 ≤ ISOPleasant < -0.1 → `unpleasant/`
- -0.1 ≤ ISOPleasant < 0.1 → `neutral/`
- 0.1 ≤ ISOPleasant < 0.5 → `pleasant/`
- ISOPleasant ≥ 0.5 → `very_pleasant/`

### Step 4: File Naming Conventions

Ensure consistent naming for audio/caption pairs:

```
BaseFilename.wav  ↔  BaseFilename.txt
```

Examples:
- `R0001_segment_binaural_44100_1.wav` + `R0001_segment_binaural_44100_1.txt`
- `park_birds_-6dB_moderate_loudness.wav` + `park_birds_-6dB_moderate_loudness.txt`

**Important:** Filenames must match exactly (case-sensitive on Linux) for the training script to find caption pairs.

### Step 5: Validation

Verify dataset completeness before training:

```bash
#!/bin/bash
# Count audio-caption pairs per category

for category in neutral pleasant unpleasant very_pleasant very_unpleasant; do
  audio_count=$(find araus_for_sa3/$category -name "*.wav" | wc -l)
  caption_count=$(find araus_for_sa3/$category -name "*.txt" | wc -l)
  
  echo "$category: $audio_count audio files, $caption_count captions"
  
  # Check for orphaned files
  for audio in araus_for_sa3/$category/*.wav; do
    base="${audio%.wav}"
    caption="${base}.txt"
    if [ ! -f "$caption" ]; then
      echo "  ⚠️  Missing caption for: $(basename $audio)"
    fi
  done
done
```

**Expected output:**
```
neutral: 1200 audio files, 1200 captions
pleasant: 1200 audio files, 1200 captions
unpleasant: 1200 audio files, 1200 captions
very_pleasant: 1200 audio files, 1200 captions
very_unpleasant: 1200 audio files, 1200 captions
```

## Data Statistics

After processing, dataset should contain:

| Pleasantness | Count | Approx. Hours |
|--------------|-------|---------------|
| very_unpleasant | ~1200 | 3.33h |
| unpleasant | ~1200 | 3.33h |
| neutral | ~1200 | 3.33h |
| pleasant | ~1200 | 3.33h |
| very_pleasant | ~1200 | 3.33h |
| **TOTAL** | **~6000** | **~16.67h** |

(10 seconds per clip × ~1200 clips per category)

## Common Issues and Fixes

### Issue: Missing Caption Files

**Symptom:** Training script fails to find captions

**Solution:**
```bash
# Find files with mismatched extensions
find araus_for_sa3/ -name "*.wav" | while read audio; do
  base="${audio%.wav}"
  if [ ! -f "${base}.txt" ]; then
    echo "Missing: ${base}.txt"
  fi
done
```

### Issue: Incorrect Audio Sample Rate

**Symptom:** Audio glitches or distortion during training

**Solution:** Resample all audio to 44.1 kHz
```bash
find araus_for_sa3/ -name "*.wav" -exec sox {} -r 44100 {}.temp \; -exec mv {}.temp {} \;
```

### Issue: Audio Duration Inconsistent

**Symptom:** Training crashes on certain files

**Solution:** Ensure all files are exactly 10 seconds
```bash
find araus_for_sa3/ -name "*.wav" -exec sox {} -p pad 0 5 | sox -p {} remix - \;
```

### Issue: Caption Encoding Problems

**Symptom:** Training fails with unicode errors

**Solution:** Ensure all caption files are UTF-8 encoded
```bash
find araus_for_sa3/ -name "*.txt" -exec file {} \; | grep -v UTF-8
# Convert non-UTF-8 files:
iconv -f ISO-8859-1 -t UTF-8 input.txt > output.txt
```

### Issue: Pleasantness Categories Imbalanced

**Symptom:** Model biases toward overrepresented category

**Solution:** Verify equal distribution or add upsampling in training config
```bash
for category in neutral pleasant unpleasant very_pleasant very_unpleasant; do
  count=$(ls araus_for_sa3/$category/*.wav | wc -l)
  percentage=$((count * 100 / 6000))
  echo "$category: $percentage%"
done
```

## Automation Script

Python script to automate caption generation and organization:

```python
#!/usr/bin/env python3
"""
Generate captions and organize ARAUS dataset for SA3 LoRA training.
"""

import csv
import os
from pathlib import Path

def loudness_descriptor(db_value):
    """Map LA50 dB value to descriptor."""
    if db_value < 46:
        return "quiet"
    elif db_value < 52:
        return "moderately quiet"
    elif db_value < 58:
        return "moderate"
    elif db_value < 64:
        return "moderately loud"
    elif db_value < 70:
        return "loud"
    else:
        return "very loud"

def pleasantness_descriptor(iso_score):
    """Map ISO pleasantness score to descriptor."""
    if iso_score < -0.4:
        return "very unpleasant pleasantness", "very_unpleasant"
    elif iso_score < -0.1:
        return "unpleasant pleasantness", "unpleasant"
    elif iso_score < 0.1:
        return "neutral", "neutral"
    elif iso_score < 0.5:
        return "pleasant", "pleasant"
    else:
        return "very pleasant", "very_pleasant"

def generate_caption(row):
    """Generate caption from metadata row."""
    scene = row['sound_source']
    modification = row['modification']
    la50 = float(row['la50'])
    iso_score = float(row['iso_pleasantness'])
    
    scene_desc = f"{scene} with {modification}"
    loudness_desc = loudness_descriptor(la50)
    pleasantness_desc, category = pleasantness_descriptor(iso_score)
    
    caption = (
        f"{scene_desc}. Loudness: {loudness_desc} (LA50: {la50:.1f} dB). "
        f"Pleasantness: {pleasantness_desc} (ISOPleasant: {iso_score:.2f}). "
        f"[pleasantness: {category}]"
    )
    
    return caption, category

def process_dataset(csv_file, audio_dir, output_dir):
    """Process ARAUS dataset."""
    output_dir = Path(output_dir)
    
    # Create subdirectories
    for category in ["very_unpleasant", "unpleasant", "neutral", "pleasant", "very_pleasant"]:
        (output_dir / category).mkdir(parents=True, exist_ok=True)
    
    # Read metadata and generate captions
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_file = Path(audio_dir) / f"{row['clip_id']}.wav"
            if not audio_file.exists():
                print(f"⚠️  Missing: {audio_file}")
                continue
            
            caption, category = generate_caption(row)
            
            # Copy audio file
            dest_audio = output_dir / category / f"{row['clip_id']}.wav"
            os.system(f"cp {audio_file} {dest_audio}")
            
            # Write caption file
            dest_caption = output_dir / category / f"{row['clip_id']}.txt"
            dest_caption.write_text(caption)
            
            print(f"✓ {category}/{row['clip_id']}")

if __name__ == "__main__":
    # Configuration
    CSV_FILE = "~/araus_metadata.csv"  # Your ARAUS metadata CSV
    AUDIO_DIR = "~/araus_raw_audio"    # Your raw audio directory
    OUTPUT_DIR = "~/Documents/restorative_embeddings/generation/train/araus_for_sa3"
    
    process_dataset(CSV_FILE, AUDIO_DIR, OUTPUT_DIR)
    print(f"\n✅ Dataset processing complete: {OUTPUT_DIR}")
```

## Caption Validation Checklist

Before training, verify captions follow the correct structure:

```bash
# Check first caption in each category
for category in neutral pleasant unpleasant very_pleasant very_unpleasant; do
  echo "=== $category ==="
  head -1 araus_for_sa3/$category/*.txt
done
```

**Expected structure in each caption:**
- ✅ Scene description (park/water/forest/urban/etc.)
- ✅ Loudness descriptor with LA50 value
- ✅ Pleasantness descriptor with ISOPleasant score
- ✅ [pleasantness: tag] at end
- ✅ Proper spacing and punctuation

## References

- ARAUS Dataset Paper: https://doi.org/10.1038/s41597-023-02614-z
- ARAUS Original Dataset: https://zenodo.org/record/7867196
- ISO 12913: Acoustic quality of urban soundscapes
- Stable Audio 3: https://github.com/stability-ai/stable-audio-3
