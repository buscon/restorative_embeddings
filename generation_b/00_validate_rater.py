"""Step 0 - Validate SoundAQnet and alternatives on held-out ISD/ARAUS data.

Tests the rater (SoundAQnet or CLAP ridge model) on audio it wasn't trained
on (ISD recordings, and ARAUS if held out). Checks:
- Recording and location-level Pearson r with human ISOPleasant ratings.
- Sensitivity to level changes (±20, ±10 dB).
- Sensitivity to clip length (5 s, 10 s cuts).
- Comparison with milestone-1 CLAP ridge model (no level input).

Decision rule (fixed before running):
- Use SoundAQnet if recording-level r on ISD >= CLAP ridge r (0.37),
  AND predictions are stable under 10s cut and ±10 dB gain changes.
- Otherwise use CLAP ridge model.
- If neither >= 0.3 at 10 s length, stop plan B (labels too noisy).

Output: validation_results.json with:
    - rater name (soundaqnet or clap_ridge)
    - scores at full/5s/10s length and original/±10 dB level
    - pearson r, p-value, bootstrap CI
    - decision rule verdict
    - recommendation for next step

    python generation_b/00_validate_rater.py --isd data/isd.csv --out generation_b
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Suppress warnings for clean output
warnings.filterwarnings('ignore', category=UserWarning)


def pearson_ci(x, y, n_bootstrap=1000, ci=0.95):
    """Pearson r with bootstrap confidence interval."""
    r, p = stats.pearsonr(x, y)
    
    # Bootstrap CI
    n = len(x)
    rs = []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        r_boot, _ = stats.pearsonr(x[idx], y[idx])
        rs.append(r_boot)
    
    rs = np.array(rs)
    lower = np.percentile(rs, (1 - ci) / 2 * 100)
    upper = np.percentile(rs, (1 + ci) / 2 * 100)
    
    return r, p, (lower, upper)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--isd', type=Path, required=True, help='ISD data CSV')
    parser.add_argument('--araus', type=Path, help='ARAUS processed CSV')
    parser.add_argument('--soundaqnet', type=Path, help='Path to SoundAQnet model/env')
    parser.add_argument('--clap-model', type=Path, help='Path to CLAP ridge model (joblib)')
    parser.add_argument('--out', type=Path, default=Path('generation_b'), help='Output dir')
    args = parser.parse_args()
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PLAN B: VALIDATE RATER (Step 0)")
    print("=" * 70)
    
    # Load ISD
    print(f"\nLoading ISD from {args.isd}...")
    try:
        isd_data = pd.read_csv(args.isd)
    except FileNotFoundError:
        print(f"ERROR: ISD file not found at {args.isd}")
        print("\nNote: This script requires processed ISD data.")
        print("To create it, run: python -c \"from rsd.data import load_isd; load_isd('path/to/isd/recordings')\"")
        return
    
    # Check for required columns
    required = ['id', 'path', 'ISOPleasant']
    if not all(col in isd_data.columns for col in required):
        print(f"ERROR: ISD CSV must have columns: {required}")
        print(f"Found: {list(isd_data.columns)}")
        return
    
    results = {
        'validation_date': pd.Timestamp.now().isoformat(),
        'isd_n': len(isd_data),
        'isd_isopleasant_mean': float(isd_data['ISOPleasant'].mean()),
        'isd_isopleasant_std': float(isd_data['ISOPleasant'].std()),
        'tests': {}
    }
    
    # =========================================================================
    # TEST 1: Full-length ISD at original level
    # =========================================================================
    print("\n[TEST 1] Full-length ISD at calibrated level...")
    print("  (Requires SoundAQnet or CLAP ridge model implementation)")
    print("  Placeholder: To implement, integrate:")
    print("    - SoundAQnet from github.com/Yuanbo2020/SoundSCaper")
    print("    - CLAP ridge from results/model_*.joblib")
    
    results['tests']['full_length'] = {
        'status': 'not_implemented',
        'note': 'Requires loading audio and running models'
    }
    
    # =========================================================================
    # TEST 2: ISD at 5s and 10s length
    # =========================================================================
    print("\n[TEST 2] Length sensitivity (5s and 10s cuts)...")
    print("  Placeholder: Load audio, cut to lengths, re-evaluate")
    
    results['tests']['length_sensitivity'] = {
        'status': 'not_implemented',
        'note': 'Requires audio processing'
    }
    
    # =========================================================================
    # TEST 3: Gain sensitivity (±20, ±10 dB)
    # =========================================================================
    print("\n[TEST 3] Level sensitivity (±20, ±10 dB)...")
    print("  Placeholder: Apply gain to audio, re-evaluate SoundAQnet predictions")
    
    results['tests']['level_sensitivity'] = {
        'status': 'not_implemented',
        'note': 'Requires SoundAQnet level input'
    }
    
    # =========================================================================
    # DECISION RULE
    # =========================================================================
    print("\n" + "=" * 70)
    print("DECISION RULE")
    print("=" * 70)
    
    print("\nTo use this validation, implement the three tests above:")
    print("1. Run SoundAQnet on full-length ISD (Pearson r >= 0.37 to beat CLAP)")
    print("2. Check 5s/10s cuts (stability required)")
    print("3. Check ±10 dB gains (stability required)")
    print("\nReference baseline (CLAP ridge on ISD): r = 0.37")
    print("\nRecommendation: Use SoundAQnet if:")
    print("  - r_full >= 0.37 AND")
    print("  - r_10s >= 0.35 AND")
    print("  - |Δr(±10dB)| < 0.05")
    print("Otherwise: Use CLAP ridge model or stop plan B if r_10s < 0.30")
    
    decision = {
        'recommendation': 'implement_tests_above',
        'next_step': 'Once tests pass, proceed to 01_select.py',
        'reference_baseline': {
            'model': 'clap_ridge',
            'isd_r': 0.37,
            'source': 'milestone-1 results'
        }
    }
    
    results['decision'] = decision
    
    # Save results
    out_file = args.out / 'validation_results.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Validation template saved to {out_file}")
    print("\nNext: Implement the three tests using:")
    print("  - github.com/Yuanbo2020/SoundSCaper for SoundAQnet")
    print("  - librosa/scipy for audio processing")
    print(f"  - Results saved to {args.out}")


if __name__ == '__main__':
    main()
