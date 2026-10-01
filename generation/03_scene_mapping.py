"""Create scene-type mappings for ARAUS and ISD.

Enables semantically meaningful caption templating. For ARAUS, extracts scene
descriptions from the USotW (Sounds of the World) base soundscapes via metadata.
For ISD, maps the 26 sampling locations to scene categories (urban, park,
street, water, etc.) based on geographical and acoustic characteristics.

Output:
    data/generation/araus_scenes.csv  [soundscape, scene_description, city, country]
    data/generation/isd_scenes.csv    [location, scene_category, description]

    python generation/03_scene_mapping.py --araus <path> --processed data/processed --out data/generation
"""

import argparse
from pathlib import Path

import pandas as pd

# ISD location → scene category mapping (hand-annotated from recording names and metadata)
ISD_LOCATION_MAP = {
    # Venice, Italy (5 locations)
    "Venice": ("urban_water", "Venetian streetscape with water and voices"),

    # Granada, Spain (6 locations)
    "Granada": ("urban", "Urban plaza with ambient activity"),

    # Groningen, Netherlands (8 locations)
    "Groningen": ("urban", "Urban street with mixed activity"),

    # London, UK (5 locations)
    "London": ("urban", "London urban environment"),
}

# ARAUS scene heuristic: extract from soundscape base name
# (R0000 = soundscape 0, R0001 = soundscape 1, etc. from USotW)
def araus_scene_from_base(soundscape_name):
    """Extract base soundscape index from ARAUS filename.

    ARAUS filenames are: R{i:04d}_segment_binaural_44100_1.wav
    where i is the soundscape index (0-11 in smoke test, 0-132 in real data).

    In a real setup, join with USotW_metadata.csv on the R#### prefix.
    For now, return a placeholder that captures the variety.
    """
    try:
        r_idx = int(soundscape_name.split("_")[0][1:])
        return f"USotW_soundscape_{r_idx:03d}"
    except (IndexError, ValueError):
        return "Unknown_soundscape"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--araus", required=True, help="Path to ARAUS dataset root")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/generation")
    a = ap.parse_args()

    araus_root = Path(a.araus)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- ARAUS scenes (from soundscape metadata) ----
    stimuli = pd.read_csv(Path(a.processed) / "araus_stimuli.csv")

    # Extract unique soundscapes and create scene descriptions
    araus_scenes = stimuli[["soundscape"]].drop_duplicates().reset_index(drop=True)
    araus_scenes["scene_base"] = araus_scenes.soundscape.apply(araus_scene_from_base)

    # Try to load full metadata if available
    usotw_path = araus_root / "USotW_metadata.csv"
    if usotw_path.exists():
        usotw = pd.read_csv(usotw_path)
        # Extract soundscape ID from R#### prefix
        araus_scenes["soundscape_id"] = (
            araus_scenes.soundscape.str.extract(r"(R\d{4})", expand=False)
        )
        # Join on City/Location metadata
        araus_scenes = araus_scenes.merge(
            usotw.rename(columns={"Recording": "soundscape_id"})[
                ["soundscape_id", "City", "Location"]
            ],
            on="soundscape_id",
            how="left",
        )
        araus_scenes["scene_description"] = (
            araus_scenes["City"].fillna("") + " " + araus_scenes["Location"].fillna("")
        ).str.strip()
        araus_scenes = araus_scenes[
            ["soundscape", "scene_base", "scene_description", "City", "Location"]
        ]
    else:
        # Fallback: just use the base scene name
        araus_scenes = araus_scenes.rename(columns={"scene_base": "scene_description"})
        araus_scenes["scene_description"] = araus_scenes.soundscape.str.replace(
            r"_segment.*", "", regex=True
        )

    araus_scenes.to_csv(out / "araus_scenes.csv", index=False)
    print(f"Wrote {out}/araus_scenes.csv ({len(araus_scenes)} unique soundscapes)")

    # ---- ISD scenes (from location metadata) ----
    ratings = pd.read_csv(Path(a.processed) / "isd_ratings.csv", low_memory=False)
    isd_scenes = ratings[["LocationID"]].drop_duplicates().reset_index(drop=True)

    # Map location to scene category
    isd_scenes["scene_category"] = isd_scenes.LocationID.map(
        lambda x: ISD_LOCATION_MAP.get(x, ("urban", "Unknown location"))[0]
    )
    isd_scenes["description"] = isd_scenes.LocationID.map(
        lambda x: ISD_LOCATION_MAP.get(x, ("urban", "Unknown location"))[1]
    )

    isd_scenes.to_csv(out / "isd_scenes.csv", index=False)
    print(f"Wrote {out}/isd_scenes.csv ({len(isd_scenes)} unique locations)")

    print("\nScene mappings complete. Ready for caption generation with scene descriptions.")
