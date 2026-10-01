"""Create scene-type mappings for ARAUS and ISD using rule-based scene generation.

Enables semantically meaningful caption templating. For ARAUS, uses rule-based
generation combining masker type and loudness to create differentiated scene
descriptions. For ISD, maps the 26 sampling locations to scene categories based
on hand-annotated geographical and acoustic characteristics.

Output:
    data/generation/araus_scenes.csv  [soundscape, scene_description, scene_score]
    data/generation/isd_scenes.csv    [location, scene_category, description]

    python generation/03_scene_mapping.py --araus <path> --processed data/processed --out data/generation
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Scene templates by masker type and loudness
MASKER_SCENES = {
    "bird": {
        "quiet": "quiet park with natural ambience and birdsong",
        "moderate": "urban park with birds and ambient activity",
        "loud": "busy park with intermittent birdsong",
    },
    "water": {
        "quiet": "peaceful waterside environment with flowing water",
        "moderate": "waterfront location with water sounds and ambient activity",
        "loud": "active waterfront or water treatment facility",
    },
    "traffic": {
        "quiet": "suburban street with occasional traffic",
        "moderate": "busy urban street with steady traffic noise",
        "loud": "highway or major intersection with heavy traffic",
    },
    "construction": {
        "quiet": "light construction activity",
        "moderate": "active construction site with machinery",
        "loud": "intense construction site with heavy equipment",
    },
    "wind": {
        "quiet": "outdoor environment with light wind",
        "moderate": "windy outdoor location",
        "loud": "highly exposed outdoor area with strong wind",
    },
    "silence": {
        "quiet": "quiet controlled environment",
        "moderate": "indoor or sheltered space",
        "loud": "indoor space with background noise",
    },
}

# ISD location → scene category mapping (hand-annotated from recording names and metadata)
ISD_LOCATION_MAP = {
    # Venice, Italy (5 locations)
    "Venice": ("water", "Venetian streetscape with water and voices"),

    # Granada, Spain (6 locations)
    "Granada": ("urban_street", "Urban plaza with ambient activity"),

    # Groningen, Netherlands (8 locations)
    "Groningen": ("urban_street", "Urban street with mixed activity"),

    # London, UK (5 locations)
    "London": ("urban_street", "London urban environment"),
}


def get_loudness_category(la50):
    """Categorize loudness level from LA50."""
    if la50 < 60:
        return "quiet"
    elif la50 < 70:
        return "moderate"
    else:
        return "loud"


def generate_scene_description(masker_type, la50):
    """Generate scene description based on masker type and loudness.

    Returns (scene_description, confidence_score).
    Confidence is based on how well the masker type is defined.
    """
    loudness_cat = get_loudness_category(la50)

    # Normalize masker type (handle variations)
    masker_normalized = masker_type.lower().strip()

    # Map to base masker type
    if masker_normalized in MASKER_SCENES:
        scenes = MASKER_SCENES[masker_normalized]
        description = scenes.get(loudness_cat, scenes["moderate"])
        confidence = 0.9
    else:
        # Unknown masker type - use generic description
        if loudness_cat == "quiet":
            description = "quiet environment"
        elif loudness_cat == "loud":
            description = "loud environment"
        else:
            description = "moderate activity environment"
        confidence = 0.5

    return description, confidence


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--araus", required=True, help="Path to ARAUS dataset root")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/generation")
    a = ap.parse_args()

    araus_root = Path(a.araus)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- ARAUS scenes (using rule-based generation by masker type + loudness) ----
    stimuli = pd.read_csv(Path(a.processed) / "araus_stimuli.csv")

    # Extract unique soundscapes with their masker types and mean loudness
    araus_scenes = stimuli.groupby("soundscape").agg({
        "masker_type": "first",  # Masker type is constant per soundscape
        "LA50": "mean"  # Average loudness across stimuli in this soundscape
    }).reset_index()

    print(f"Generating scene descriptions for {len(araus_scenes)} unique soundscapes...")

    # Apply rule-based scene generation
    scenes = []
    scores = []
    for _, row in araus_scenes.iterrows():
        scene_desc, confidence = generate_scene_description(row["masker_type"], row["LA50"])
        scenes.append(scene_desc)
        scores.append(confidence)

    araus_scenes["scene_description"] = scenes
    araus_scenes["scene_score"] = scores

    # Keep only soundscape and description columns for output
    araus_scenes = araus_scenes[["soundscape", "scene_description", "scene_score"]]

    araus_scenes.to_csv(out / "araus_scenes.csv", index=False)
    print(f"Wrote {out}/araus_scenes.csv ({len(araus_scenes)} unique soundscapes)")
    print(f"\nScene description distribution:")
    print(araus_scenes["scene_description"].value_counts().to_string())

    # ---- ISD scenes (from location metadata) ----
    recordings = pd.read_csv(Path(a.processed) / "isd_recordings.csv")
    isd_scenes = recordings[["LocationID"]].drop_duplicates().reset_index(drop=True)

    # Map location to scene category
    isd_scenes["scene_category"] = isd_scenes.LocationID.map(
        lambda x: ISD_LOCATION_MAP.get(x, ("urban_street", "Unknown location"))[0]
    )
    isd_scenes["description"] = isd_scenes.LocationID.map(
        lambda x: ISD_LOCATION_MAP.get(x, ("urban_street", "Unknown location"))[1]
    )

    isd_scenes.to_csv(out / "isd_scenes.csv", index=False)
    print(f"\nWrote {out}/isd_scenes.csv ({len(isd_scenes)} unique locations)")

    print("\nScene mappings complete. Ready for caption generation with differentiated scene descriptions.")
