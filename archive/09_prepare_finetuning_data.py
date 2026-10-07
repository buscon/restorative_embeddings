#!/usr/bin/env python3
"""
Prepare data for Stable Audio fine-tuning following the DCASE approach.

This script:
1. Validates audio format (44.1 kHz, stereo, 16-bit)
2. Creates metadata CSV from .txt caption files
3. Optionally applies data augmentation
4. Outputs ready-to-train dataset
"""

import os
import csv
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from scipy import signal

# Paths
base_dir = os.path.expanduser("~/Documents/restorative_embeddings")
audio_dir = os.path.join(base_dir, "generation/train/diverse_audio_with_captions")
output_dir = os.path.join(base_dir, "generation/train/sa3_finetuning_data")
metadata_csv = os.path.join(output_dir, "metadata.csv")

os.makedirs(output_dir, exist_ok=True)

# Target audio specs (from tutorial)
TARGET_SR = 44100
TARGET_CHANNELS = 2
TARGET_DTYPE = np.int16

print("Preparing Stable Audio fine-tuning dataset...")
print(f"Source: {audio_dir}")
print(f"Output: {output_dir}\n")

# Collect all audio files and captions
audio_files = []
pleasantness_bins = ['very_unpleasant', 'unpleasant', 'neutral', 'pleasant', 'very_pleasant']

for bin_name in pleasantness_bins:
    bin_dir = os.path.join(audio_dir, bin_name)
    
    if not os.path.exists(bin_dir):
        print(f"WARNING: {bin_name} directory not found")
        continue
    
    wav_files = sorted([f for f in os.listdir(bin_dir) if f.endswith('.wav')])
    
    for wav_file in wav_files:
        base_name = os.path.splitext(wav_file)[0]
        txt_file = os.path.join(bin_dir, f"{base_name}.txt")
        
        if os.path.exists(txt_file):
            audio_files.append({
                'wav': os.path.join(bin_dir, wav_file),
                'txt': txt_file,
                'bin': bin_name
            })

print(f"Found {len(audio_files)} audio files with captions\n")

# Process audio files
processed_count = 0
metadata_rows = []

for idx, file_info in enumerate(audio_files, 1):
    wav_path = file_info['wav']
    txt_path = file_info['txt']
    bin_name = file_info['bin']
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    
    try:
        # Load audio
        y, sr = librosa.load(wav_path, sr=None, mono=False)
        
        # Ensure stereo
        if y.ndim == 1:
            y = np.stack([y, y])
        elif y.shape[0] > 2:
            y = y[:2]
        elif y.shape[0] == 1:
            y = np.vstack([y, y])
        
        # Resample to 44.1 kHz if needed
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        
        # Normalize to -1 dBFS (peak normalization)
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak * 0.999  # Slightly below 1.0 to avoid clipping
        
        # Convert to int16
        y_int16 = np.int16(y * 32767)
        
        # Save processed audio
        output_wav = os.path.join(output_dir, f"{base_name}.wav")
        sf.write(output_wav, y_int16.T, TARGET_SR, subtype='PCM_16')
        
        # Read caption
        with open(txt_path, 'r') as f:
            caption = f.read().strip()
        
        # Add to metadata
        metadata_rows.append({
            'file': f"{base_name}.wav",
            'caption': caption
        })
        
        processed_count += 1
        print(f"[{idx:3d}] {base_name:40s} ✓")
        
    except Exception as e:
        print(f"[{idx:3d}] {base_name:40s} ERROR: {e}")

print(f"\nSuccessfully processed: {processed_count}/{len(audio_files)}")

# Write metadata CSV
print(f"\nWriting metadata.csv...")
with open(metadata_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['file', 'caption'])
    writer.writeheader()
    writer.writerows(metadata_rows)

print(f"Metadata saved to: {metadata_csv}")
print(f"\nDataset ready for fine-tuning!")
print(f"Use with training script:")
print(f"  --audio_dir {output_dir}")
print(f"  --metadata_csv {metadata_csv}")

# Show sample metadata
print(f"\nSample metadata entries:")
for row in metadata_rows[:3]:
    print(f"  {row['file']}: {row['caption'][:60]}...")
