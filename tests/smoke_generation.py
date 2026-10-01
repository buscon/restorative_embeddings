"""End-to-end smoke test for caption generation and audio export pipeline.

Validates the generation pipeline (01_captions.py, 02_export.py, 03_scene_mapping.py)
on small synthetic data. Checks file formats and basic sanity constraints without
requiring real ARAUS/ISD downloads.

    python tests/smoke_generation.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]


def make_synthetic_isd(root: Path):
    """Create tiny synthetic ISD dataset for testing."""
    sr = 48000
    rows, rec = [], 0
    locations = {"Park": ("PK", 0.6, 2500)}
    for loc, (pre, base, f) in locations.items():
        d = root / "WAV_City_1" / loc
        d.mkdir(parents=True)
        for g in range(2):
            gid = f"{pre}{100 + g}"
            fname = f"{gid}.wav"
            x = 0.02 * np.sin(2 * np.pi * f * np.arange(sr * 35) / sr)
            sf.write(str(d / fname), np.stack([x, x], axis=1), sr, subtype="FLOAT")
            for _ in range(1):
                rec += 1
                rows.append({"LocationID": loc, "SessionID": f"{loc}1", "GroupID": gid, "RecordID": rec,
                             "pleasant": 4, "vibrant": 3, "eventful": 3, "chaotic": 2,
                             "annoying": 2, "monotonous": 2, "uneventful": 2, "calm": 4,
                             "sss01": 3, "ssi01": 2, "ssi02": 2, "ssi03": 3, "ssi04": 4, "sss05": 4,
                             "LAeq_L50(A)": 60, "LAeq_L10(A)": 64, "LAeq_L90(A)": 56,
                             "LCeq_L50(C)": 66, "N_N5": 20, "R_R": 0.2, "FS_F": 0.15,
                             "T_TonalityHMS": 0.1})
    pd.DataFrame(rows).to_csv(root / "ISD v1.0 Data.csv", index=False)


def make_synthetic_araus(root: Path):
    """Create tiny synthetic ARAUS dataset for testing."""
    (root / "data").mkdir(parents=True)
    (root / "soundscapes").mkdir()
    (root / "maskers").mkdir()

    sr = 44100

    # Create 2 soundscapes
    sounds = []
    for i in range(2):
        name = f"R{i:04d}_segment_binaural_44100_1.wav"
        x = 0.1 * np.sin(2 * np.pi * (100 + 50 * i) * np.arange(sr * 30) / sr)
        x = np.stack([x, x], axis=1)
        sf.write(str(root / "soundscapes" / name), x, sr)
        sounds.append({"soundscape": name, "gain_s": 1.0, "insitu_leq": 55 + i, "fold_s": 0})

    # Create 2 maskers
    maskers = []
    for mt, f in [("bird", 3000), ("traffic", 150)]:
        for j in range(1):
            name = f"{mt}_{j:05d}.wav"
            x = 0.05 * np.sin(2 * np.pi * f * np.arange(sr * 30) / sr)
            sf.write(str(root / "maskers" / name), x, sr)
            row = {"masker": name}
            # Create gain columns for all possible dB values (46-84)
            for db in range(46, 85):
                row[f"gain_{db}dB"] = 10 ** ((db - 70) / 20)
                row[f"leq_at_gain_{db}dB"] = db
            maskers.append(row)

    pd.DataFrame(sounds).to_csv(root / "data/soundscapes.csv", index=False)
    pd.DataFrame(maskers).to_csv(root / "data/maskers.csv", index=False)

    # Create responses with ISO coordinates and psychoacoustics
    resp = []
    for i in range(2):
        participant = f"SYNTH_{i:05d}"
        for k in range(2):
            s = sounds[k % len(sounds)]
            m = maskers[0]["masker"]
            smr = -3
            resp.append({
                "participant": participant,
                "fold_r": 0,
                "soundscape": s["soundscape"],
                "masker": m,
                "smr": smr,
                "stimulus_index": k + 2,
                "is_attention": 0,
                "pleasant": 4 if k == 0 else 2,
                "vibrant": 3,
                "eventful": 3,
                "chaotic": 2,
                "annoying": 2,
                "monotonous": 2,
                "uneventful": 2,
                "calm": 4,
                # Psychoacoustic measurements (synthesized)
                "LA50_r": 60 + k * 5,
                "LA10_r": 65 + k * 5,
                "LA90_r": 55 + k * 5,
                "LC50_r": 65 + k * 5,
                "N05_r": 20 + k * 2,
                "Ravg_r": 0.2,
                "Favg_r": 0.15,
                "Tavg_r": 0.1,
            })

    pd.DataFrame(resp).to_csv(root / "data/responses.csv", index=False)
    pd.DataFrame([{
        "participant": "SYNTH_00000",
        "age": 30,
        "gender": "x",
        "who": 15,
        "pss": 15,
        "wnss": 3,
        "panas_pos": 30,
        "panas_neg": 15,
    }, {
        "participant": "SYNTH_00001",
        "age": 25,
        "gender": "x",
        "who": 18,
        "pss": 10,
        "wnss": 4,
        "panas_pos": 35,
        "panas_neg": 10,
    }]).to_csv(root / "data/participants.csv", index=False)


def run(*args, env):
    print("\n$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=REPO, env=env, check=True)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_synthetic_araus(tmp / "araus")
        make_synthetic_isd(tmp / "isd")

        env = {**os.environ, "RSD_FAKE_EMBED": "1"}
        P = str(tmp / "proc")
        G = str(tmp / "gen")

        # Step 0: Prepare (from existing smoke test)
        run("scripts/01_prepare.py", "--araus", str(tmp / "araus"), "--isd", str(tmp / "isd"),
            "--out", P, env=env)

        # Step 1: Generate scene mappings
        run("generation/03_scene_mapping.py", "--araus", str(tmp / "araus"),
            "--processed", P, "--out", G, env=env)

        # Step 2: Generate captions
        run("generation/01_captions.py", "--processed", P, "--out", G, env=env)

        # Step 3: Export (subset of audio for speed)
        run("generation/02_export.py", "--araus", str(tmp / "araus"),
            "--processed", P, "--captions", f"{G}/captions.csv",
            "--out", G, "--n-samples", "2", env=env)

        # Validate outputs
        print("\n=== Validation ===")

        # Check captions.csv
        captions = pd.read_csv(Path(G) / "captions.csv")
        assert len(captions[captions.dataset == "araus"]) > 0, "No ARAUS captions"
        assert captions[["id", "caption", "ISOPleasant", "bin", "LA50"]].notna().all().all(), \
            "NaN in captions.csv"
        print(f"✓ captions.csv ({len(captions)} rows)")

        # Check scene mappings
        scenes = pd.read_csv(Path(G) / "araus_scenes.csv")
        assert len(scenes) > 0, "No ARAUS scene mappings"
        print(f"✓ araus_scenes.csv ({len(scenes)} soundscapes)")

        # Check export_metadata.csv
        metadata = pd.read_csv(Path(G) / "export_metadata.csv")
        assert len(metadata) == 2, f"Expected 2 exported audio files, got {len(metadata)}"
        assert all(Path(p).exists() for p in metadata.wav_path), "Missing WAV files"
        assert metadata[["stimulus_id", "caption", "ISOPleasant"]].notna().all().all(), \
            "NaN in export_metadata.csv"
        print(f"✓ export_metadata.csv ({len(metadata)} rows)")

        # Check audio files
        for wav_path in metadata.wav_path:
            x, sr = sf.read(wav_path)
            assert sr == 44100, f"Wrong sample rate: {sr}"
            assert x.ndim == 2, f"Expected stereo, got shape {x.shape}"
            assert x.shape[1] == 2, f"Expected stereo, got {x.shape[1]} channels"
            assert x.shape[0] > 0, "Empty audio"
        print(f"✓ {len(metadata)} audio files (stereo, 44.1 kHz)")

        # Sample 5 captions to verify quality
        print("\n=== Sample Captions (first 5) ===")
        for idx, row in captions.head(5).iterrows():
            print(f"  [{row['bin']}] {row['caption']}")

        print(f"\n✓ SMOKE TEST PASSED")
        print(f"  Generated {len(captions)} captions, exported {len(metadata)} audio files")
