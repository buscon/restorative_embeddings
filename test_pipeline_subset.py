"""
Test Pipeline on ISD Subset

Runs the full pipeline on a small subset of soundscapes to validate:
1. CLAP embedding extraction
2. k-means clustering
3. Acoustic feature extraction
4. Cluster characterization
5. Stratified sampling

This is faster than full dataset and helps identify any issues before
scaling to all ~1000 soundscapes.
"""

import os
import random
import shutil
from pathlib import Path
from restorative_soundscape_pipeline import main

def create_test_subset(audio_dir, subset_size=100, output_dir='./isd_test_subset'):
    """
    Create a test subset of audio files for quick pipeline validation.
    
    Args:
        audio_dir: Path to organized ISD audio directory
        subset_size: Number of files to sample (default 100 for ~5-10 min on GPU)
        output_dir: Where to copy subset files
    
    Returns:
        Path to test subset directory
    """
    audio_path = Path(audio_dir)
    subset_path = Path(output_dir)
    
    # Find all WAV files
    wav_files = list(audio_path.glob('**/*.wav'))
    
    if not wav_files:
        wav_files = list(audio_path.glob('*.wav'))
    
    print(f"Found {len(wav_files)} total WAV files")
    
    if len(wav_files) < subset_size:
        subset_size = len(wav_files)
        print(f"Dataset smaller than subset_size. Using all {subset_size} files.")
    
    # Random sample
    sampled = random.sample(wav_files, subset_size)
    print(f"Sampling {subset_size} files for testing")
    
    # Create output directory
    subset_path.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    print(f"Copying to {output_dir}...")
    for wav_file in sampled:
        dest = subset_path / wav_file.name
        shutil.copy2(wav_file, dest)
    
    print(f"✓ Test subset created: {subset_size} files")
    return subset_path


def test_pipeline(subset_dir, output_dir='./test_results', n_clusters=6):
    """
    Run full pipeline on test subset.
    
    Args:
        subset_dir: Path to test subset directory
        output_dir: Where to save results
        n_clusters: Number of clusters (smaller for test)
    """
    print("\n" + "=" * 60)
    print("Running Pipeline on Test Subset")
    print("=" * 60)
    
    # Run pipeline
    results = main(
        audio_dir=str(subset_dir),
        output_dir=output_dir,
        n_clusters=n_clusters,
        batch_size=16  # Adjust based on GPU memory
    )
    
    return results


def summarize_results(output_dir):
    """
    Print summary of pipeline results.
    """
    import json
    import pandas as pd
    import numpy as np
    
    results_path = Path(output_dir)
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    # Load characterization
    char_path = results_path / 'cluster_characterization.json'
    if char_path.exists():
        with open(char_path) as f:
            cluster_info = json.load(f)
        
        print("\nCluster Sizes:")
        for cid, info in cluster_info.items():
            print(f"  Cluster {cid}: {info['size']} samples")
    
    # Load listening samples
    listening_path = results_path / 'listening_samples.csv'
    if listening_path.exists():
        listening_df = pd.read_csv(listening_path)
        print(f"\nListening Samples: {len(listening_df)} files stratified across clusters")
        print(f"  Per cluster: {len(listening_df) // len(cluster_info.keys()):.0f} on average")
    
    # Load embeddings
    emb_path = results_path / 'embeddings.npy'
    if emb_path.exists():
        embeddings = np.load(emb_path)
        print(f"\nEmbeddings: {embeddings.shape}")
    
    # Load features
    feat_path = results_path / 'acoustic_features.csv'
    if feat_path.exists():
        features = pd.read_csv(feat_path)
        print(f"Acoustic Features: {features.shape}")
        print(f"\n  Columns: {', '.join(features.columns[:5])}... ({features.shape[1]} total)")
    
    print(f"\n✓ All results saved to: {output_dir}/")
    print("\nNext steps:")
    print("  1. Review cluster_characterization.json")
    print("  2. Listen to samples in listening_samples.csv")
    print("  3. Annotate restorativeness for each sample")
    print("  4. Run full pipeline on complete ISD dataset")


def main_test():
    """
    Full test workflow
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Test restorative soundscape pipeline')
    parser.add_argument('audio_dir', help='Path to organized ISD audio directory')
    parser.add_argument('--subset-size', type=int, default=100, 
                       help='Number of files for test subset (default: 100)')
    parser.add_argument('--n-clusters', type=int, default=6,
                       help='Number of clusters (default: 6)')
    parser.add_argument('--keep-subset', action='store_true',
                       help='Keep test subset directory after testing')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.audio_dir):
        print(f"Error: Audio directory not found: {args.audio_dir}")
        return
    
    # Create subset
    subset_dir = create_test_subset(args.audio_dir, args.subset_size)
    
    # Run pipeline
    results = test_pipeline(subset_dir, n_clusters=args.n_clusters)
    
    # Summarize
    summarize_results('./test_results')
    
    # Clean up subset if requested
    if not args.keep_subset:
        print(f"\nCleaning up subset directory...")
        shutil.rmtree(subset_dir)
        print("✓ Removed test subset (use --keep-subset to retain)")


if __name__ == '__main__':
    import sys
    
    # Simple usage without argparse for direct calls
    if len(sys.argv) == 1:
        print("Usage:")
        print("  python test_pipeline_subset.py <audio_dir> [--subset-size 100] [--n-clusters 6]")
        print("\nExample:")
        print("  python test_pipeline_subset.py ./isd_audio --subset-size 150 --n-clusters 8")
        print("\nOr directly in Python:")
        print("  subset = create_test_subset('./isd_audio', subset_size=100)")
        print("  test_pipeline(str(subset))")
    else:
        main_test()
