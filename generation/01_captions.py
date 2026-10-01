"""Step 1 - Generate pleasantness-conditioned captions for ARAUS and ISD stimuli.

Creates captions with five pleasantness levels: very unpleasant, unpleasant,
neutral, pleasant, very pleasant. Binning thresholds are fixed post-hoc on
combined ARAUS + ISD response distributions to balance class sizes:

* very unpleasant: ISOPleasant ≤ −0.50
* unpleasant: −0.50 < ISOPleasant ≤ −0.15
* neutral: −0.15 < ISOPleasant < +0.15
* pleasant: +0.15 ≤ ISOPleasant < +0.50
* very pleasant: ISOPleasant ≥ +0.50

Caption template:
    "{scene_description}, {source_description}, {loudness}. {pleasantness} soundscape."

Examples:
    "Urban park, birds singing and people talking, moderately loud. Pleasant soundscape."
    "Street intersection, traffic and construction noise, very loud. Very unpleasant soundscape."

Output: captions.csv with columns
    [id, dataset, caption, ISOPleasant, bin, n_ratings, LA50]

    python generation/01_captions.py --processed data/processed --out data/generation
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Pleasantness binning (fixed from combined ARAUS + ISD distribution)
BINS = {
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

# Source type descriptions (common masker types in ARAUS)
SOURCE_DESC = {
    "bird": "birdsong",
    "water": "flowing water",
    "traffic": "traffic noise",
    "construction": "construction noise",
}

# Loudness descriptions
def loudness_desc(la50):
    """Bin LA50 into descriptive category."""
    if la50 < 50:
        return "quiet"
    elif la50 < 60:
        return "moderately quiet"
    elif la50 < 70:
        return "moderately loud"
    elif la50 < 80:
        return "loud"
    else:
        return "very loud"


def bin_pleasantness(x):
    """Assign ISOPleasant to pleasantness bin."""
    for bin_name, (lo, hi) in BINS.items():
        if lo < x <= hi:
            return bin_name
    # Edge case: x exactly at boundary
    if x == -0.50:
        return "unpleasant"
    elif x == -0.15:
        return "neutral"
    elif x == 0.15:
        return "pleasant"
    elif x == 0.50:
        return "very_pleasant"
    return "neutral"


def captions_araus(proc_root):
    """Generate captions for ARAUS stimuli.

    Returns:
        DataFrame with [stimulus_id, caption, ISOPleasant, bin, n_ratings, LA50]
    """
    proc_root = Path(proc_root)

    # Load ARAUS stimulus-level aggregates
    stim = pd.read_csv(proc_root / "araus_stimuli.csv")

    # Add scene descriptions from soundscape metadata
    # For now, use a simple heuristic: group by soundscape base
    # Real implementation would join with USotW_metadata.csv
    stim["scene"] = stim.soundscape.str.replace(r"_segment.*", "", regex=True)

    # Add source descriptions from masker type
    stim["source"] = stim.masker_type.map(SOURCE_DESC).fillna(stim.masker_type)
    stim["loudness"] = stim.LA50.apply(loudness_desc)

    # Bin pleasantness
    stim["bin"] = stim.ISOPleasant.apply(bin_pleasantness)
    stim["pleasantness_label"] = stim.bin.map(PLEASANTNESS_LABELS)

    # Template caption
    stim["caption"] = (
        stim.scene + ", " +
        stim.source + ", " +
        stim.loudness + ". " +
        stim.pleasantness_label + " soundscape."
    )

    return stim[["stimulus_id", "caption", "ISOPleasant", "bin", "n_ratings", "LA50"]].rename(
        columns={"stimulus_id": "id"}
    ).assign(dataset="araus")


def captions_isd(proc_root):
    """Generate captions for ISD recordings.

    Note: ISD does not have masker types or controlled acoustic variation.
    Captions are simpler: "{location}, {loudness}. {pleasantness} soundscape."

    Returns:
        DataFrame with [id, caption, ISOPleasant, bin, n_ratings, LA50]
    """
    proc_root = Path(proc_root)

    # Load ISD recording-level ratings
    recs = pd.read_csv(proc_root / "isd_recordings.csv")

    # Filter to recordings with ratings (has_audio column may not exist)
    if "has_audio" in recs.columns:
        recs = recs[(recs.has_audio == True) & (recs.n_ratings > 0)].copy()
    else:
        recs = recs[recs.n_ratings > 0].copy()

    recs["loudness"] = recs.LA50.apply(loudness_desc)
    recs["bin"] = recs.ISOPleasant.apply(bin_pleasantness)
    recs["pleasantness_label"] = recs.bin.map(PLEASANTNESS_LABELS)

    # Template caption (location instead of scene)
    recs["caption"] = (
        recs.LocationID + ", " +
        recs.loudness + ". " +
        recs.pleasantness_label + " soundscape."
    )

    return recs[["GroupID", "caption", "ISOPleasant", "bin", "n_ratings", "LA50"]].rename(
        columns={"GroupID": "id"}
    ).assign(dataset="isd")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/generation")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # Generate captions for both datasets
    araus = captions_araus(a.processed)
    isd = captions_isd(a.processed)

    # Combine and save
    combined = pd.concat([araus, isd], ignore_index=True)
    combined.to_csv(out / "captions.csv", index=False)

    # Summary statistics
    print(f"Generated {len(araus)} ARAUS captions, {len(isd)} ISD captions")
    print("\nPleasantness bin distribution (combined):")
    print(combined.bin.value_counts().sort_index())
    print(f"\nWrote {out}/captions.csv")
