"""Step 3 - Generate captions and metadata table for rated clips.

Creates captions from FSD50K labels, SoundAQnet scene predictions, and
pleasantness ratings. Template:

    "{scene}, {sources}. {pleasantness} soundscape."

Example:
    "Forest park with birdsong and wind. Pleasant soundscape."

Output: captions.csv with columns:
    [id, dataset, caption, ISOPleasant, pleasantness_bin, sources, scene, 
     rater, license, uploader, fsd50k_labels]

    python generation_b/03_captions.py --ratings ratings.csv --selected fsd50k_selected.csv --out generation_b
"""

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')


# Pleasantness bins (fixed or quantile-based)
PLEASANTNESS_BINS_FIXED = {
    "very_unpleasant": (-np.inf, -0.50),
    "unpleasant": (-0.50, -0.15),
    "neutral": (-0.15, +0.15),
    "pleasant": (+0.15, +0.50),
    "very_pleasant": (+0.50, +np.inf),
}

PLEASANTNESS_LABELS = {
    "very_unpleasant": "very unpleasant",
    "unpleasant": "unpleasant",
    "neutral": "neutral",
    "pleasant": "pleasant",
    "very_pleasant": "very pleasant",
}


def map_fsd_label_to_source(label):
    """Map FSD50K labels to human-readable source descriptions."""
    mapping = {
        'bird': 'birdsong',
        'bird call': 'bird calls',
        'bird song': 'bird songs',
        'wind': 'wind',
        'wind flowing': 'wind',
        'water': 'water sounds',
        'water flowing': 'flowing water',
        'rain': 'rain',
        'traffic': 'traffic noise',
        'car': 'traffic',
        'dog': 'dog sounds',
        'cat': 'cat sounds',
        'insect': 'insect sounds',
        'footsteps': 'footsteps',
        'crowd': 'crowd sounds',
        'chatter': 'human voices',
        'construction': 'construction noise',
        'tool': 'tool sounds',
        'bell': 'bell sounds',
        'metal': 'metallic sounds',
        'wood': 'wood sounds',
    }
    
    label_lower = label.lower()
    for key, value in mapping.items():
        if key in label_lower:
            return value
    
    return label.lower()  # fallback


def get_pleasantness_label(iso_pleasant, bins='fixed'):
    """Assign pleasantness label from ISOPleasant score."""
    
    if pd.isna(iso_pleasant):
        return 'neutral', 'unknown'
    
    if bins == 'fixed':
        for bin_name, (lower, upper) in PLEASANTNESS_BINS_FIXED.items():
            if lower <= iso_pleasant < upper:
                return bin_name, PLEASANTNESS_LABELS[bin_name]
    
    elif bins == 'quantile':
        # Will be set per-dataset in main()
        return 'unknown', 'unknown'
    
    return 'neutral', 'neutral'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratings', type=Path, required=True,
                        help='Ratings CSV from step 02')
    parser.add_argument('--selected', type=Path, required=True,
                        help='Selected clips CSV from step 01')
    parser.add_argument('--pleasantness-bins', choices=['fixed', 'quantile'],
                        default='fixed', help='Binning strategy')
    parser.add_argument('--out', type=Path, default=Path('generation_b'),
                        help='Output directory')
    args = parser.parse_args()
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PLAN B: GENERATE CAPTIONS (Step 3)")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading ratings from {args.ratings}...")
    if not args.ratings.exists():
        print(f"ERROR: Ratings file not found")
        return
    
    ratings = pd.read_csv(args.ratings)
    print(f"  Loaded {len(ratings)} rated clips")
    
    print(f"\nLoading selected clips from {args.selected}...")
    if not args.selected.exists():
        print(f"ERROR: Selected clips file not found")
        return
    
    selected = pd.read_csv(args.selected)
    print(f"  Loaded {len(selected)} selected clips")
    
    # Merge
    print("\nMerging ratings with clip metadata...")
    merged = ratings.merge(selected, on=['id', 'dataset'], how='left')
    print(f"  Merged: {len(merged)} clips with full info")
    
    # =========================================================================
    # Determine pleasantness binning
    # =========================================================================
    print(f"\nApplying {args.pleasantness_bins} pleasantness binning...")
    
    if args.pleasantness_bins == 'fixed':
        print("  Using fixed bins:")
        for bin_name, (lower, upper) in PLEASANTNESS_BINS_FIXED.items():
            print(f"    {bin_name}: [{lower:.2f}, {upper:.2f})")
    
    elif args.pleasantness_bins == 'quantile':
        # Compute quantile bins per dataset
        print("  Computing quantile bins (quintiles)...")
        quantile_bins = {}
        for dataset in merged['dataset'].unique():
            subset = merged[merged['dataset'] == dataset]['ISOPleasant'].dropna()
            if len(subset) > 0:
                quantiles = np.quantile(subset, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
                quantile_bins[dataset] = quantiles
                print(f"    {dataset}: {quantiles}")
    
    # =========================================================================
    # Generate captions
    # =========================================================================
    print("\nGenerating captions...")
    
    captions = []
    for idx, row in merged.iterrows():
        cap_dict = {
            'id': row['id'],
            'dataset': row['dataset'],
            'ISOPleasant': row['ISOPleasant'],
        }
        
        # Pleasantness bin and label
        if args.pleasantness_bins == 'fixed':
            bin_name, label = get_pleasantness_label(row['ISOPleasant'], bins='fixed')
        else:
            # TODO: implement quantile lookup
            bin_name, label = 'unknown', 'unknown'
        
        cap_dict['pleasantness_bin'] = bin_name
        cap_dict['pleasantness_label'] = label
        
        # Sources from FSD50K labels
        sources = []
        if pd.notna(row.get('labels')):
            labels = row['labels']
            if isinstance(labels, str):
                labels = labels.split(';')
            sources = [map_fsd_label_to_source(l.strip()) for l in labels[:3]]
        
        sources_str = ' and '.join(sources) if sources else 'natural ambience'
        cap_dict['sources'] = sources_str
        
        # Scene (placeholder: would use SoundAQnet)
        scene = 'outdoor'  # placeholder
        cap_dict['scene'] = scene
        
        # Build caption
        caption = f"{scene} with {sources_str}. {label} soundscape."
        cap_dict['caption'] = caption
        
        # Metadata
        cap_dict['rater'] = row.get('rater')
        cap_dict['rater_version'] = row.get('rater_version')
        cap_dict['license'] = row.get('licenses')
        cap_dict['uploader'] = row.get('uploader')
        cap_dict['fsd50k_labels'] = row.get('labels')
        
        captions.append(cap_dict)
    
    captions_df = pd.DataFrame(captions)
    
    # =========================================================================
    # Save and review
    # =========================================================================
    captions_file = args.out / 'captions.csv'
    captions_df.to_csv(captions_file, index=False)
    print(f"\n✓ Captions saved to {captions_file}")
    
    # Sample 20 random captions for manual review
    print("\n" + "=" * 70)
    print("SAMPLE CAPTIONS (review these before next step)")
    print("=" * 70)
    
    sample_size = min(20, len(captions_df))
    sample_indices = np.random.choice(len(captions_df), sample_size, replace=False)
    
    for idx in sorted(sample_indices):
        row = captions_df.iloc[idx]
        print(f"\n[{row['id']}] {row['pleasantness_label'].upper()}")
        print(f"  Caption: {row['caption']}")
        print(f"  ISOPleasant: {row['ISOPleasant']:.3f}")
        print(f"  Labels: {row['fsd50k_labels']}")
    
    # Distribution
    print("\n" + "=" * 70)
    print("PLEASANTNESS DISTRIBUTION")
    print("=" * 70)
    
    bin_counts = captions_df['pleasantness_bin'].value_counts()
    print("\nClips per bin:")
    for bin_name in ['very_unpleasant', 'unpleasant', 'neutral', 'pleasant', 'very_pleasant']:
        count = bin_counts.get(bin_name, 0)
        pct = 100 * count / len(captions_df)
        print(f"  {bin_name:20s}: {count:5d} ({pct:5.1f}%)")
    
    # Save distribution
    dist = {
        'total_clips': int(len(captions_df)),
        'pleasantness_distribution': bin_counts.to_dict(),
        'binning_strategy': args.pleasantness_bins,
        'sample_reviews': int(sample_size),
    }
    dist_file = args.out / 'distribution.json'
    with open(dist_file, 'w') as f:
        json.dump(dist, f, indent=2)
    
    print(f"\n✓ Distribution saved to {dist_file}")
    
    print("\n" + "=" * 70)
    print("NEXT STEP: 04_export.py")
    print("=" * 70)
    print(f"\npython generation_b/04_export.py --captions {captions_file} --out {args.out}")


if __name__ == '__main__':
    main()
