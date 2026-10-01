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
    "{scene} with {source}. Loudness: {loudness} (LA50: {LA50} dB). Pleasantness: {pleasantness_label} (ISOPleasant: {ISOPleasant})."

Examples:
    "Urban park with birdsong. Loudness: moderately loud (LA50: 65.3 dB). Pleasantness: pleasant (ISOPleasant: 0.28)."
    "Street intersection with traffic noise. Loudness: very loud (LA50: 78.2 dB). Pleasantness: very unpleasant pleasantness (ISOPleasant: -0.68)."

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
    "very_unpleasant": "very unpleasant pleasantness",
    "unpleasant": "unpleasant pleasantness",
    "neutral": "neutral pleasantness",
    "pleasant": "pleasant",
    "very_pleasant": "very pleasant",
}

# Source type descriptions (masker types in ARAUS)
SOURCE_DESC = {
    "bird": "birdsong",
    "water": "flowing water",
    "traffic": "traffic noise",
    "construction": "construction noise",
    "wind": "wind noise",
    "silence": None,  # Will be filtered out
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


def captions_araus(proc_root, soundscape_captions_df=None):
    """Generate captions for ARAUS stimuli using audio-analyzed soundscape descriptions.

    Args:
        proc_root: Path to processed data directory
        soundscape_captions_df: Optional DataFrame with [soundscape, caption] from audio analysis

    Returns:
        DataFrame with [stimulus_id, caption, ISOPleasant, bin, n_ratings, LA50]
    """
    proc_root = Path(proc_root)

    # Load ARAUS stimulus-level aggregates
    stim = pd.read_csv(proc_root / "araus_stimuli.csv")

    # Filter out silence maskers (no acoustic information to model)
    stim = stim[stim.masker_type != "silence"].copy()

    # Add soundscape descriptions from audio captioning
    has_soundscape_captions = False
    if soundscape_captions_df is not None and not soundscape_captions_df.empty:
        stim = stim.merge(soundscape_captions_df[["soundscape", "caption"]], on="soundscape", how="left")
        stim["soundscape_desc"] = stim["caption"]
        has_soundscape_captions = True
    else:
        # Fallback: use soundscape ID
        stim["soundscape_desc"] = stim.soundscape.str.replace(r"_segment.*", "", regex=True) + " soundscape"

    # Add masker descriptions
    stim["masker_desc"] = stim.masker_type.map(SOURCE_DESC).fillna(stim.masker_type)
    stim["loudness"] = stim.LA50.apply(loudness_desc)

    # Bin pleasantness
    stim["bin"] = stim.ISOPleasant.apply(bin_pleasantness)
    stim["pleasantness_label"] = stim.bin.map(PLEASANTNESS_LABELS)

    # Caption template: Combine soundscape description + masker + acoustic properties
    # Format: "{soundscape_with_masker}. Loudness: {loudness}. Pleasantness: {label}."
    stim["caption"] = (
        stim.soundscape_desc + " with added " +
        stim.masker_desc + ". " +
        "Loudness: " + stim.loudness +
        " (LA50: " + stim.LA50.round(1).astype(str) + " dB). " +
        "Pleasantness: " + stim.pleasantness_label +
        " pleasantness soundscape (ISOPleasant: " + stim.ISOPleasant.round(2).astype(str) + ")."
    )

    return stim[["stimulus_id", "caption", "ISOPleasant", "bin", "n_ratings", "LA50"]].rename(
        columns={"stimulus_id": "id"}
    ).assign(dataset="araus")


def captions_isd(proc_root):
    """Generate captions for ISD recordings.

    Note: ISD does not have masker types or controlled acoustic variation.
    Captions include location and acoustic measurements for consistency with ARAUS.

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

    # Expanded template caption for consistency with ARAUS
    recs["caption"] = (
        recs.LocationID + ". " +
        "Loudness: " + recs.loudness +
        " (LA50: " + recs.LA50.round(1).astype(str) + " dB). " +
        "Pleasantness: " + recs.pleasantness_label +
        " (ISOPleasant: " + recs.ISOPleasant.round(2).astype(str) + ")."
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

    # Try to load soundscape captions from audio analysis
    soundscape_path = out / "soundscape_captions.csv"
    soundscape_captions_df = None
    if soundscape_path.exists():
        soundscape_captions_df = pd.read_csv(soundscape_path)
        print(f"Loaded audio-analyzed soundscape captions from {soundscape_path}")
    else:
        print(f"Note: Audio-analyzed soundscape captions not found at {soundscape_path}.")
        print(f"      Run 04_soundscape_captions.py first for accurate soundscape descriptions.")
        print(f"      Using fallback descriptions (soundscape IDs).")

    # Generate captions for both datasets
    araus = captions_araus(a.processed, soundscape_captions_df=soundscape_captions_df)
    isd = captions_isd(a.processed)

    # Combine and save
    combined = pd.concat([araus, isd], ignore_index=True)
    combined.to_csv(out / "captions.csv", index=False)

    # Summary statistics
    print(f"Generated {len(araus)} ARAUS captions, {len(isd)} ISD captions")
    print("\nPleasantness bin distribution (combined):")
    print(combined.bin.value_counts().sort_index())
    print(f"\nWrote {out}/captions.csv")
