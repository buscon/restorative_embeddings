"""Step 4 - Sample and review captions before proceeding to model training.

Samples 50 captions across pleasantness bins and masker types from the full
ARAUS captions, and outputs them to a TSV for manual review. This is the
checkpoint before training: verify caption quality, diversity, and semantic
coherence. Only proceed to model training after review is complete.

Output: captions_review_sample.tsv with 50 representative rows

    python generation/04_review_captions.py --captions data/generation/captions.csv --out data/generation

Review checklist:
    1. Are caption templates consistent and grammatically correct?
    2. Do pleasantness labels match the mood (e.g., "very unpleasant" for low ISOPleasant)?
    3. Are source descriptions diverse and accurate for each masker type?
    4. Are loudness descriptions reasonable given the LA50 values?
    5. Does each caption describe a unique acoustic scene?
    6. Are there any duplicates or near-duplicates within bins?
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captions", required=True, help="Path to captions.csv")
    ap.add_argument("--out", default="data/generation")
    ap.add_argument("--n-sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    captions = pd.read_csv(a.captions)

    # Filter to ARAUS only for review (more acoustic variety)
    araus = captions[captions.dataset == "araus"].reset_index(drop=True)

    if len(araus) == 0:
        print("ERROR: No ARAUS captions found. Run generation/01_captions.py first.")
        exit(1)

    # Stratified sample across pleasantness bins
    rng = np.random.default_rng(a.seed)
    strata = araus.groupby("bin").size()
    target_per_stratum = strata / strata.sum() * a.n_sample

    sampled = []
    for bin_name, target_count in target_per_stratum.items():
        subset = araus[araus.bin == bin_name]
        n = min(int(np.round(target_count)), len(subset))
        if n > 0:
            sampled.append(subset.sample(n=n, random_state=rng))

    sample = pd.concat(sampled, ignore_index=False).sample(frac=1, random_state=rng)
    sample = sample.reset_index(drop=True)

    # Output as tab-separated for easy review in Excel/Sheets
    out_path = Path(a.out) / "captions_review_sample.tsv"
    sample[["bin", "caption", "ISOPleasant", "LA50"]].to_csv(out_path, sep="\t", index=True, index_label="idx")

    print(f"Sampled {len(sample)} captions for review")
    print(f"Bin distribution:")
    print(sample.bin.value_counts().sort_index().to_string())
    print(f"\nReview sample saved to {out_path}")
    print("\nReview checklist:")
    print("  1. Are caption templates consistent and grammatically correct?")
    print("  2. Do pleasantness labels match the mood?")
    print("  3. Are source descriptions diverse and accurate?")
    print("  4. Are loudness descriptions reasonable?")
    print("  5. Does each caption describe a unique acoustic scene?")
    print("  6. Are there duplicates or near-duplicates within bins?")
    print("\nOpen in Excel/Sheets and review. Once approved, commit changes and proceed to training.")
