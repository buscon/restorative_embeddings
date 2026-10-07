"""Loading ARAUS (training) and ISD (external test) into common tables.

Targets are the ISO 12913-3 circumplex coordinates ISOPleasant and
ISOEventful in [-1, 1], computed with soundscapy from the eight PAQ items.
Both datasets use identical PAQ wording and 5-point scales.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from soundscapy.surveys.processing import calculate_iso_coords
from soundscapy.surveys.survey_utils import rename_paqs

PAQ_LABELS = ["pleasant", "vibrant", "eventful", "chaotic", "annoying", "monotonous", "uneventful", "calm"]
TARGETS = ["ISOPleasant", "ISOEventful"]

# --- Harmonised psychoacoustic predictors -------------------------------------
# Seven Versümer-style predictors that exist in BOTH datasets. Relative Approach
# is not in ARAUS; sharpness is left out because the methods differ (ARAUS:
# DIN 45692, ISD: Aures). Level uses the 50 % exceedance level in both, since
# ARAUS reports a fast-averaged mean (LAavg) rather than LAeq.
# ARAUS column names: README of ntudsp/araus-dataset-baseline-models.
# ISD column names: `ISD v1.0 Data.csv`; methods in `ISD v1.0 Metadata.xlsx`.
# Remaining mismatch to keep in mind: channel aggregation (L/R) may differ.
PSYCHO = ["LA50", "LA10_LA90", "LC50_LA50", "N5", "R", "FS", "T"]


def _psycho_araus(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "LA50": df["LA50_r"],
        "LA10_LA90": df["LA10_r"] - df["LA90_r"],
        "LC50_LA50": df["LC50_r"] - df["LA50_r"],
        "N5": df["N05_r"],
        "R": df["Ravg_r"],
        "FS": df["Favg_r"],
        "T": df["Tavg_r"],
    }, index=df.index)


def _psycho_isd(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "LA50": df["LAeq_L50(A)"],
        "LA10_LA90": df["LAeq_L10(A)"] - df["LAeq_L90(A)"],
        "LC50_LA50": df["LCeq_L50(C)"] - df["LAeq_L50(A)"],
        "N5": df["N_N5"],
        "R": df["R_R"],
        "FS": df["FS_F"],
        "T": df["T_TonalityHMS"],
    }, index=df.index)


def add_iso(df: pd.DataFrame) -> pd.DataFrame:
    paq = rename_paqs(df[PAQ_LABELS].astype(float))
    pl, ev = calculate_iso_coords(paq)
    return df.assign(ISOPleasant=pl.values, ISOEventful=ev.values)


# --- ARAUS ----------------------------------------------------------------------

def load_araus(root: str | Path) -> dict[str, pd.DataFrame]:
    """root = clone of ntudsp/araus-dataset-baseline-models after download.py."""
    root = Path(root)
    d = root / "data"
    resp = pd.read_csv(d / "responses.csv")
    parts = pd.read_csv(d / "participants.csv")
    sounds = pd.read_csv(d / "soundscapes.csv")
    maskers = pd.read_csv(d / "maskers.csv")

    # ARAUSv1: fold 0 = independent test set, folds 1-5 = cross-validation.
    # Fold -1 holds the practice/consistency stimulus shared by everyone, and
    # attention stimuli carry forced answers - both are dropped, as the dataset
    # authors recommend.
    resp = resp[resp.fold_r.between(0, 5) & (resp.is_attention == 0)].copy()
    resp = add_iso(resp.dropna(subset=PAQ_LABELS))
    resp["stimulus_id"] = resp.soundscape + "|" + resp.masker + "|" + resp.smr.astype(int).astype(str)
    resp["masker_type"] = resp.masker.str.split("_").str[0]  # bird, water, traffic, ...
    resp = pd.concat([resp, _psycho_araus(resp)], axis=1)
    resp = resp.merge(parts[["participant", "age", "gender", "who", "pss", "wnss", "panas_pos", "panas_neg"]],
                      on="participant", how="left")

    stim = (resp.groupby("stimulus_id")
            .agg(soundscape=("soundscape", "first"), masker=("masker", "first"), smr=("smr", "first"),
                 masker_type=("masker_type", "first"), fold=("fold_r", "first"),
                 n_ratings=("participant", "size"),
                 ISOPleasant=("ISOPleasant", "mean"), ISOEventful=("ISOEventful", "mean"),
                 **{c: (c, "mean") for c in PSYCHO})
            .reset_index())
    return {"responses": resp, "stimuli": stim, "soundscapes": sounds, "maskers": maskers}


# --- ISD -----------------------------------------------------------------------

def index_isd_wavs(isd_root: str | Path) -> dict[str, Path]:
    """GroupID -> WAV path. Skips macOS resource forks (`__MACOSX/`, `._*`),
    which also end in .wav, and resolves the odd names in the Groningen
    archive (`NP125.hdf.wav`, duplicate `NP102.1.wav`) by preferring the
    shortest file name per GroupID."""
    found: dict[str, Path] = {}
    for p in Path(isd_root).rglob("*.wav"):
        if "__MACOSX" in p.parts or p.name.startswith("._"):
            continue
        gid = p.name.split(".")[0]
        if gid not in found or len(p.name) < len(found[gid].name):
            found[gid] = p
    return found


def load_isd(isd_root: str | Path) -> dict[str, pd.DataFrame]:
    isd_root = Path(isd_root)
    csv = next(isd_root.rglob("ISD v1.0 Data.csv"), None)
    if csv is None:
        raise FileNotFoundError(f"'ISD v1.0 Data.csv' not found under {isd_root}")
    raw = pd.read_csv(csv, low_memory=False)
    wavs = index_isd_wavs(isd_root)

    ratings = raw.dropna(subset=PAQ_LABELS).copy()
    ratings = add_iso(ratings)
    ratings = pd.concat([ratings, _psycho_isd(ratings)], axis=1)
    ratings["GroupID"] = ratings.GroupID.astype(str)
    ratings["has_audio"] = ratings.GroupID.isin(wavs)

    # One recording per GroupID; several people may have rated it.
    rated = (ratings[ratings.has_audio].groupby("GroupID")
             .agg(LocationID=("LocationID", "first"), SessionID=("SessionID", "first"),
                  n_ratings=("RecordID", "size"),
                  ISOPleasant=("ISOPleasant", "mean"), ISOEventful=("ISOEventful", "mean"),
                  sss01=("sss01", "mean"), sss05=("sss05", "mean"),
                  **{c: (c, "first") for c in PSYCHO}))
    # Recordings without survey answers (mostly the 2020 lockdown sessions) are
    # kept for embedding/clustering but carry no targets.
    recs = pd.DataFrame({"GroupID": list(wavs), "wav": [str(p) for p in wavs.values()],
                         "folder": [p.parent.name for p in wavs.values()]})
    recs = recs.merge(rated, left_on="GroupID", right_index=True, how="left")
    recs["n_ratings"] = recs.n_ratings.fillna(0).astype(int)
    recs["LocationID"] = recs.LocationID.fillna(recs.folder)  # archive folders are named by location
    return {"ratings": ratings, "recordings": recs.sort_values("GroupID").reset_index(drop=True)}
