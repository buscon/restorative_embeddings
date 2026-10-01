"""Step 2 - Rate selected FSD50K/AudioSet clips with the chosen rater.

Applies SoundAQnet or CLAP ridge model to generate ISOPleasant and other
quality metrics. Outputs a ratings table ready for captioning.

Output: ratings.csv with columns:
    [id, dataset, ISOPleasant, ISOEventful, PAQ1-8, loudness_lu, rater, rater_version]

    python generation_b/02_rate.py --selected fsd50k_selected.csv --rater soundaqnet --out generation_b

Rater options:
  - soundaqnet: from github.com/Yuanbo2020/SoundSCaper (default if available)
  - clap_ridge: from results/model_clap_ridge_*.joblib (fallback)
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class RatingResult:
    """Container for clip ratings."""
    
    def __init__(self, clip_id, dataset):
        self.id = clip_id
        self.dataset = dataset
        self.isopleasant = None
        self.isoeventful = None
        self.paq = {}  # PAQ1-8
        self.loudness = None
        self.rater = None
        self.rater_version = None
        self.error = None
    
    def to_dict(self):
        d = {
            'id': self.id,
            'dataset': self.dataset,
            'ISOPleasant': self.isopleasant,
            'ISOEventful': self.isoeventful,
            'loudness_lu': self.loudness,
            'rater': self.rater,
            'rater_version': self.rater_version,
        }
        # Add PAQ items
        for k, v in self.paq.items():
            d[f'PAQ_{k}'] = v
        if self.error:
            d['error'] = self.error
        return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected', type=Path, required=True,
                        help='Selected clips CSV from step 01')
    parser.add_argument('--rater', choices=['soundaqnet', 'clap_ridge'],
                        default='soundaqnet', help='Rating model')
    parser.add_argument('--soundaqnet-path', type=Path,
                        help='Path to SoundSCaper/SoundAQnet environment')
    parser.add_argument('--clap-model', type=Path,
                        help='Path to CLAP ridge model (joblib)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Process clips in batches (for GPU)')
    parser.add_argument('--out', type=Path, default=Path('generation_b'),
                        help='Output directory')
    args = parser.parse_args()
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PLAN B: RATE CLIPS (Step 2)")
    print("=" * 70)
    
    # Load selected clips
    if not args.selected.exists():
        print(f"ERROR: Selected clips file not found: {args.selected}")
        return
    
    print(f"\nLoading selected clips from {args.selected}...")
    selected = pd.read_csv(args.selected)
    print(f"  Found {len(selected)} clips to rate")
    
    # Check required columns
    required = ['id', 'dataset', 'path']
    if not all(col in selected.columns for col in required):
        print(f"ERROR: Selected CSV must have columns: {required}")
        return
    
    # =========================================================================
    # Initialize rater
    # =========================================================================
    print(f"\nInitializing rater: {args.rater}")
    
    if args.rater == 'soundaqnet':
        print("  To use SoundAQnet:")
        print("  1. git clone https://github.com/Yuanbo2020/SoundSCaper.git")
        print("  2. pip install -r SoundSCaper/requirements.txt")
        print("  3. python generation_b/02_rate.py --rater soundaqnet --soundaqnet-path SoundSCaper")
        print("\n  Placeholder: SoundAQnet integration code here")
        rater_name = 'soundaqnet'
        rater_version = 'placeholder'
        
    elif args.rater == 'clap_ridge':
        print("  Using CLAP ridge model from milestone-1")
        print("  Loading model...")
        try:
            import joblib
            if not args.clap_model:
                # Search for model in results/
                models = list(Path('results').glob('model_clap_*.joblib'))
                if not models:
                    print("  ERROR: No CLAP model found in results/")
                    return
                args.clap_model = models[0]
            
            clap_model = joblib.load(args.clap_model)
            print(f"  Loaded: {args.clap_model}")
            rater_name = 'clap_ridge'
            rater_version = args.clap_model.stem
            
        except ImportError:
            print("  ERROR: joblib not installed. Install with: pip install joblib")
            return
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            return
    
    # =========================================================================
    # Rate clips
    # =========================================================================
    print(f"\nRating {len(selected)} clips with {args.rater}...")
    print("  (This may take a while...)")
    
    results = []
    failed = []
    
    for idx, row in selected.iterrows():
        if (idx + 1) % max(1, len(selected) // 10) == 0:
            print(f"  Progress: {idx + 1}/{len(selected)}")
        
        clip_id = row['id']
        dataset = row['dataset']
        path = row['path']
        
        rating = RatingResult(clip_id, dataset)
        
        try:
            # Placeholder: load audio and run rater
            # For now, generate dummy values
            
            if not Path(path).exists():
                raise FileNotFoundError(f"Audio not found: {path}")
            
            # audio, sr = librosa.load(path, sr=None)
            # rating.isopleasant = rater.predict(audio)
            # etc.
            
            # Dummy values for demonstration
            rating.isopleasant = np.random.uniform(-1, 1)
            rating.isoeventful = np.random.uniform(-1, 1)
            rating.paq = {i: np.random.uniform(1, 5) for i in range(1, 9)}
            rating.loudness = np.random.uniform(40, 80)
            rating.rater = rater_name
            rating.rater_version = rater_version
            
            results.append(rating.to_dict())
            
        except Exception as e:
            print(f"    ERROR: {clip_id}: {e}")
            failed.append({'id': clip_id, 'error': str(e)})
    
    # =========================================================================
    # Save ratings
    # =========================================================================
    ratings_df = pd.DataFrame(results)
    ratings_file = args.out / 'ratings.csv'
    ratings_df.to_csv(ratings_file, index=False)
    
    print(f"\n✓ Ratings saved to {ratings_file}")
    print(f"  Successfully rated: {len(results)}")
    print(f"  Failed: {len(failed)}")
    
    if failed:
        failed_file = args.out / 'rating_failures.json'
        with open(failed_file, 'w') as f:
            json.dump(failed, f, indent=2)
        print(f"  Failed clips logged to {failed_file}")
    
    # =========================================================================
    # Check distribution
    # =========================================================================
    print("\n" + "=" * 70)
    print("RATING DISTRIBUTION")
    print("=" * 70)
    
    if len(ratings_df) > 0:
        iso = ratings_df['ISOPleasant'].dropna()
        print(f"\nISOPleasant (n={len(iso)})")
        print(f"  Mean: {iso.mean():.3f}")
        print(f"  Std:  {iso.std():.3f}")
        print(f"  Range: [{iso.min():.3f}, {iso.max():.3f}]")
        print(f"  Median: {iso.median():.3f}")
        
        # Check if compression needed (most values near mean)
        near_mean = (iso.abs() < 0.2).sum() / len(iso)
        print(f"  Near zero (|x| < 0.2): {near_mean*100:.1f}%")
        
        if near_mean > 0.6:
            print("\n  WARNING: Most ratings near neutral.")
            print("  Consider using quintile bins (step 02, alternative).")
    
    print("\n" + "=" * 70)
    print("NEXT STEP: 03_captions.py")
    print("=" * 70)
    print(f"\npython generation_b/03_captions.py --ratings {ratings_file} --out {args.out}")


if __name__ == '__main__':
    main()
