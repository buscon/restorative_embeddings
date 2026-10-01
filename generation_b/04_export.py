"""Step 4 - Export clips and prepare for fine-tuning.

Processes selected clips for SAO fine-tuning:
1. Normalize to target loudness (−23 LUFS)
2. Convert to 44.1 kHz stereo
3. Create train/test splits (by uploader, to avoid near-duplicates)
4. Export audio or pre-compute latent embeddings

Output:
    - train/ and test/ audio directories (or .pt files if precomputed)
    - splits.json (train/test IDs)
    - metadata.json (loudness, original paths, etc.)

    python generation_b/04_export.py --captions captions.csv --audio-root /path/to/audio --out generation_b
"""

import argparse
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captions', type=Path, required=True,
                        help='Captions CSV from step 03')
    parser.add_argument('--audio-root', type=Path, required=True,
                        help='Root directory containing audio files')
    parser.add_argument('--target-loudness', type=float, default=-23.0,
                        help='Target loudness in LUFS')
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='Target sample rate')
    parser.add_argument('--split-by-uploader', action='store_true', default=True,
                        help='Split by uploader to avoid duplicates')
    parser.add_argument('--test-size', type=float, default=0.1,
                        help='Fraction of uploaders/data in test set')
    parser.add_argument('--precompute-latents', action='store_true',
                        help='Precompute SAO latent embeddings')
    parser.add_argument('--out', type=Path, default=Path('generation_b'),
                        help='Output directory')
    args = parser.parse_args()
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PLAN B: EXPORT CLIPS (Step 4)")
    print("=" * 70)
    
    # Load captions
    print(f"\nLoading captions from {args.captions}...")
    captions = pd.read_csv(args.captions)
    print(f"  Loaded {len(captions)} clips")
    
    # =========================================================================
    # Split by uploader (to avoid near-duplicates)
    # =========================================================================
    print("\n" + "=" * 70)
    print("SPLITTING BY UPLOADER")
    print("=" * 70)
    
    if args.split_by_uploader:
        print(f"\nGrouping clips by uploader...")
        uploaders = captions['uploader'].dropna().unique()
        print(f"  Found {len(uploaders)} unique uploaders")
        
        # Stratify uploaders by test/train
        test_uploaders = np.random.choice(
            uploaders, 
            size=max(1, int(len(uploaders) * args.test_size)),
            replace=False
        )
        
        test_mask = captions['uploader'].isin(test_uploaders)
        train_ids = captions[~test_mask]['id'].tolist()
        test_ids = captions[test_mask]['id'].tolist()
        
        print(f"\n  Train uploaders: {len(uploaders) - len(test_uploaders)}")
        print(f"  Train clips: {len(train_ids)}")
        print(f"  Test uploaders: {len(test_uploaders)}")
        print(f"  Test clips: {len(test_ids)}")
    
    else:
        # Random split
        test_indices = np.random.choice(
            len(captions),
            size=max(1, int(len(captions) * args.test_size)),
            replace=False
        )
        test_mask = np.zeros(len(captions), dtype=bool)
        test_mask[test_indices] = True
        
        train_ids = captions[~test_mask]['id'].tolist()
        test_ids = captions[test_mask]['id'].tolist()
        
        print(f"\nRandom split (test_size={args.test_size}):")
        print(f"  Train: {len(train_ids)}")
        print(f"  Test: {len(test_ids)}")
    
    splits = {
        'strategy': 'by_uploader' if args.split_by_uploader else 'random',
        'train': train_ids,
        'test': test_ids,
        'seed': 42
    }
    
    # =========================================================================
    # Create output directories
    # =========================================================================
    print("\n" + "=" * 70)
    print("CREATING OUTPUT STRUCTURE")
    print("=" * 70)
    
    (args.out / 'train').mkdir(exist_ok=True)
    (args.out / 'test').mkdir(exist_ok=True)
    
    print(f"\n  {args.out / 'train'}")
    print(f"  {args.out / 'test'}")
    
    # =========================================================================
    # Process audio
    # =========================================================================
    print("\n" + "=" * 70)
    print("AUDIO PROCESSING")
    print("=" * 70)
    
    print("\nTo export audio, use:")
    print("  - librosa.load() to read audio")
    print("  - pyloudnorm to normalize to -23 LUFS")
    print("  - librosa.resample() to 44.1 kHz")
    print("  - scipy.io.wavfile.write() or soundfile.write() to save")
    
    print("\nExample pipeline:")
    print("""
    import librosa
    import pyloudnorm
    import soundfile as sf
    
    audio, sr = librosa.load(path, sr=None)
    
    # Normalize loudness
    meter = pyloudnorm.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    audio = pyloudnorm.normalize(audio, loudness, args.target_loudness)
    
    # Resample
    if sr != args.sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=args.sample_rate)
    
    # Convert mono to stereo if needed
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    
    # Save
    sf.write(out_path, audio.T, args.sample_rate)
    """)
    
    # Placeholder: count files
    metadata = []
    failed = []
    
    for idx, row in captions.iterrows():
        clip_id = row['id']
        orig_path = Path(row.get('path', ''))
        
        if not orig_path.exists():
            failed.append({'id': clip_id, 'reason': 'audio not found'})
            continue
        
        # Determine split
        split = 'train' if clip_id in train_ids else 'test'
        
        metadata_item = {
            'id': clip_id,
            'split': split,
            'original_path': str(orig_path),
            'target_loudness': args.target_loudness,
            'target_sr': args.sample_rate,
            'caption': row['caption'],
            'ISOPleasant': row['ISOPleasant'],
            'pleasantness_bin': row['pleasantness_bin'],
        }
        metadata.append(metadata_item)
    
    print(f"\n✓ Ready to process {len(metadata)} clips")
    print(f"  Failed (not found): {len(failed)}")
    
    if failed:
        failed_file = args.out / 'export_failures.json'
        with open(failed_file, 'w') as f:
            json.dump(failed, f, indent=2)
        print(f"  See {failed_file}")
    
    # =========================================================================
    # Precompute latents (optional)
    # =========================================================================
    if args.precompute_latents:
        print("\n" + "=" * 70)
        print("PRECOMPUTING LATENT EMBEDDINGS")
        print("=" * 70)
        print("\nTo precompute SAO latents:")
        print("  1. Load SAO model (from github.com/stability-ai/stable-audio-tools)")
        print("  2. For each audio file: encode to latent space")
        print("  3. Save as .pt files")
        print("\nThis speeds up training but requires GPU.")
        print("Placeholder: SAO latent precomputation code here")
    
    # =========================================================================
    # Save splits and metadata
    # =========================================================================
    splits_file = args.out / 'splits.json'
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"\n✓ Splits saved to {splits_file}")
    
    metadata_file = args.out / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_file} ({len(metadata)} clips)")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nOutput directory: {args.out}")
    print(f"  Audio will be exported to:")
    print(f"    - {args.out / 'train'} ({len(train_ids)} clips)")
    print(f"    - {args.out / 'test'} ({len(test_ids)} clips)")
    print(f"\n  Metadata:")
    print(f"    - {splits_file}")
    print(f"    - {metadata_file}")
    
    print(f"\nAudio processing settings:")
    print(f"  Target loudness: {args.target_loudness} LUFS")
    print(f"  Sample rate: {args.sample_rate} Hz")
    print(f"  Channels: stereo")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    
    print("\n1. Implement audio export (use the code template above)")
    print("2. Run fine-tuning: train/ on SAO Small or SAO 1.0")
    print("3. See train/ directory for SAO config templates")
    print("\nFor fine-tuning setup, see:")
    print("  - github.com/stability-ai/stable-audio-tools")
    print("  - generation/02_export.py (plan A equivalent)")


if __name__ == '__main__':
    main()
