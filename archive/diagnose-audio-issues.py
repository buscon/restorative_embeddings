#!/usr/bin/env python3
"""
Diagnose specific audio issues in exported ARAUS dataset.
Identifies which files have problems and why.
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import librosa
    import numpy as np
    import soundfile as sf
    import pandas as pd
except ImportError as e:
    print(f"Error: {e}")
    print("Install with: pip install librosa numpy soundfile pandas")
    sys.exit(1)

def diagnose_audio_file(wav_path, expected_sr=44100, expected_duration_min=9, expected_duration_max=11):
    """Diagnose a single audio file for issues."""
    issues = []
    
    try:
        # Check file exists and size
        if not wav_path.exists():
            return ["File does not exist"]
        
        file_size = wav_path.stat().st_size
        if file_size < 1000:
            return [f"File suspiciously small ({file_size} bytes)"]
        
        # Load with librosa for detailed analysis
        y, sr = librosa.load(str(wav_path), sr=None, mono=False)
        
        # Check for empty audio
        if y.size == 0:
            return ["Audio is empty (0 samples)"]
        
        # Check sample rate
        if sr != expected_sr:
            issues.append(f"Wrong sample rate: {sr}Hz (expected {expected_sr}Hz)")
        
        # Check shape
        if y.ndim == 1:
            channels = 1
            n_samples = len(y)
        else:
            channels = y.shape[0]
            n_samples = y.shape[1]
        
        # Check duration
        duration_sec = n_samples / sr
        if duration_sec < expected_duration_min or duration_sec > expected_duration_max:
            issues.append(f"Wrong duration: {duration_sec:.2f}s (expected {expected_duration_min}-{expected_duration_max}s)")
        
        # Check for silence
        rms = np.sqrt(np.mean(y**2))
        if rms < 0.001:
            issues.append(f"Audio is silent or near-silent (RMS={rms:.6f})")
        
        # Check for clipping
        peak = np.abs(y).max()
        if peak > 0.99:
            issues.append(f"Audio is clipping (peak={peak:.4f})")
        
        # Measure loudness (simple LUFS estimation)
        try:
            # Simple loudness check: average power in dB
            power_db = 20 * np.log10(rms + 1e-10)
            if power_db < -40:  # Suspiciously quiet
                issues.append(f"Very low loudness ({power_db:.1f} dB)")
        except:
            pass
        
        # Check for NaN/Inf
        if np.any(np.isnan(y)):
            issues.append("Audio contains NaN values")
        if np.any(np.isinf(y)):
            issues.append("Audio contains Inf values")
        
        # Check channels
        if channels not in [1, 2]:
            issues.append(f"Unexpected channel count: {channels} (expected 1 or 2)")
        
        return issues
    
    except Exception as e:
        return [f"Error loading file: {str(e)[:100]}"]

def main():
    # Get base path from environment or argument
    base_path = os.path.expanduser("$HOME/mnt/restorative_embeddings")
    if not os.path.exists(base_path):
        # Try without the $HOME expansion
        base_path = Path.cwd()
    
    base_path = Path(base_path)
    
    # Find audio directory
    audio_dir = base_path / "data/generation/audio"
    training_dir = base_path / "generation/train/araus_for_sa3"
    
    # Check which one exists
    if audio_dir.exists():
        target_dir = audio_dir
        print(f"Found exported audio directory: {audio_dir}")
    elif training_dir.exists():
        target_dir = training_dir
        print(f"Found training dataset directory: {training_dir}")
    else:
        print(f"Error: Neither {audio_dir} nor {training_dir} exists")
        print(f"\nBase path: {base_path}")
        print(f"Checking subdirectories...")
        
        # List what's actually there
        if base_path.exists():
            for item in sorted(base_path.iterdir())[:20]:
                print(f"  {item.name}/")
        
        print("\nSearching for any .wav files...")
        import subprocess
        result = subprocess.run(['find', str(base_path), '-name', '*.wav', '-type', 'f'], 
                              capture_output=True, text=True, timeout=30)
        if result.stdout:
            files = result.stdout.strip().split('\n')[:10]
            for f in files:
                print(f"  Found: {f}")
        else:
            print("  No .wav files found")
        sys.exit(1)
    
    # Find all WAV files
    wav_files = sorted(list(target_dir.glob("**/*.wav")))
    
    if not wav_files:
        print(f"Error: No WAV files found in {target_dir}")
        sys.exit(1)
    
    print(f"\n✅ Found {len(wav_files)} audio files")
    print(f"Scanning for issues...\n")
    
    # Diagnose each file
    problem_files = []
    file_issues_map = {}
    
    for i, wav_path in enumerate(wav_files):
        if (i + 1) % 500 == 0:
            print(f"  Scanned {i+1}/{len(wav_files)}...", flush=True)
        
        issues = diagnose_audio_file(wav_path)
        if issues:
            problem_files.append(wav_path)
            file_issues_map[wav_path.name] = issues
    
    # Report results
    print("\n" + "="*80)
    print(f"DIAGNOSTIC REPORT: {len(problem_files)} problematic files found")
    print("="*80)
    
    if problem_files:
        # Group issues by type
        issue_types = {}
        for issues in file_issues_map.values():
            for issue in issues:
                # Categorize
                if "sample rate" in issue:
                    key = "Wrong sample rate"
                elif "duration" in issue:
                    key = "Wrong duration"
                elif "silent" in issue or "RMS" in issue:
                    key = "Silent/low volume"
                elif "clipping" in issue:
                    key = "Clipping"
                elif "loudness" in issue:
                    key = "Low loudness"
                elif "empty" in issue:
                    key = "Empty audio"
                elif "size" in issue:
                    key = "File size issue"
                elif "channels" in issue:
                    key = "Wrong channel count"
                elif "Error loading" in issue:
                    key = "Cannot load file"
                else:
                    key = "Other"
                
                if key not in issue_types:
                    issue_types[key] = []
                issue_types[key].append(issue)
        
        # Print by category
        print("\nISSUES BY CATEGORY:")
        for category, issues in sorted(issue_types.items(), key=lambda x: -len(x[1])):
            print(f"\n  {category}: {len(issues)} occurrences")
            # Show first 3 unique examples
            unique = list(set(issues))[:3]
            for issue in unique:
                print(f"    • {issue}")
        
        print(f"\n\nPROBLEMATIC FILES ({len(problem_files)} total):")
        for wav_path in sorted(problem_files)[:50]:  # Show first 50
            issues_str = " | ".join(file_issues_map[wav_path.name])
            print(f"  ❌ {wav_path.name}")
            print(f"     └─ {issues_str}")
        
        if len(problem_files) > 50:
            print(f"\n  ... and {len(problem_files) - 50} more problematic files")
    
    else:
        print("\n✅ All audio files look good! No issues detected.")
    
    print("\n" + "="*80)
    print(f"Summary: {len(problem_files)}/{len(wav_files)} files have issues")
    print("="*80)

if __name__ == "__main__":
    main()
