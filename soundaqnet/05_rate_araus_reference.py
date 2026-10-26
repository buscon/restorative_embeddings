#!/usr/bin/env python3
"""Reference set for the generated-audio heatmaps: real ARAUS stimuli rated by SoundAQnet with the same code.

Rebuilds ARAUS stimuli in memory (soundscape + masker at the stimulus SMR, rsd.audio.ArausMixer, same as generation/02_export.py),
cuts them to 30 s, and rates every stimulus under TWO level conditions with every released checkpoint (mean):
  true   the calibrated ARAUS level (digital scale x 2.8358 Pa per unit): what SoundAQnet was trained and tested on
  65     mono mix scaled to 65 dB SPL rms, exactly as the generated clips are rated (assumed level)
Comparing the generated clips with "65" is like for like; comparing them with "true" shows what the assumed level changes.

Sampling: stimuli of the chosen folds (default fold 0 = ARAUS independent test set), silence-masker stimuli excluded, N stimuli
stratified over the masker types (equal numbers per type; a type with fewer stimuli gives all it has), fixed seed. The
number of fold-0 stimuli is printed: compare it with the 3,576 test clips of the SoundSCaper paper to see whether they are the
same set (I could not verify this). The human rating (ISOPleasant, ISOEventful, mean of the listeners) is kept in the output.

    python soundaqnet/05_rate_araus_reference.py --araus data/raw/araus --soundscaper third_party/SoundSCaper \
        --n 300 --workers 8 --device cuda --out ~/grids_3msde/araus_ref

Output: <out>/soundaqnet_full_true.csv and <out>/soundaqnet_full_65.csv (same columns as 04_rate_folder_full.py, plus stimulus_id,
masker_type, smr, fold, human_ISOPleasant, human_ISOEventful, n_ratings; the column "file" is the stimulus_id). Features cached in <out>/cache.
"""
import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from soundaqnet import features as F  # noqa: E402

_MIXER = {}


def _mixer(araus):
    if "m" not in _MIXER:
        from rsd.audio import ArausMixer
        root = Path(araus)
        _MIXER["m"] = ArausMixer(pd.read_csv(root / "data" / "soundscapes.csv"), pd.read_csv(root / "data" / "maskers.csv"),
                                 root / "soundscapes", root / "maskers")
    return _MIXER["m"]


def job(args):
    i, araus, soundscape, masker, smr, cache = args
    import torch
    torch.set_num_threads(1)
    p_true, p_65 = Path(cache) / f"{i:05d}_true.npz", Path(cache) / f"{i:05d}_65.npz"
    if p_true.exists() and p_65.exists():
        return str(p_true), str(p_65)
    x, sr = _mixer(araus).mix(soundscape, masker, float(smr))
    y = F.to_mono_44k(x, sr)[: F.CLIP_SECONDS * F.SR_WAV]          # digital scale of the ARAUS wavs
    y_pa = y * F.ARAUS_PA_PER_DIGITAL
    np.savez(p_true, mel=F.log_mel(y), loud=F.iso_loudness(y_pa))
    y65 = y * F.assume_level(y, rms_db_spl=65.0) / F.ARAUS_PA_PER_DIGITAL   # assumed level, digital scale
    np.savez(p_65, mel=F.log_mel(y65), loud=F.iso_loudness(y65 * F.ARAUS_PA_PER_DIGITAL))
    return str(p_true), str(p_65)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--araus", default="data/raw/araus")
    ap.add_argument("--soundscaper", default="third_party/SoundSCaper")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from rsd.data import load_araus
    st = load_araus(a.araus)["stimuli"]
    print("stimuli per fold:", st.groupby("fold").size().to_dict())
    print("masker types (all folds):", st.masker_type.value_counts().to_dict())
    st = st[st.fold.isin(a.folds) & (st.masker_type != "silence")]
    print(f"folds {a.folds} without silence maskers: {len(st)} stimuli (paper test set: 3,576 clips)")
    types = sorted(st.masker_type.unique())
    per = max(1, a.n // len(types))
    st = pd.concat([g.sample(min(per, len(g)), random_state=a.seed) for _, g in st.groupby("masker_type")]).reset_index(drop=True)
    print(f"{len(st)} stimuli sampled ({per} per masker type, {len(types)} types: {types})")

    out = Path(a.out).expanduser()
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(i, a.araus, r.soundscape, r.masker, r.smr, str(cache)) for i, r in enumerate(st.itertuples())]
    with ProcessPoolExecutor(a.workers) as ex:
        paths = list(tqdm(ex.map(job, jobs, chunksize=1), total=len(jobs), desc="features"))

    from soundaqnet.model import EVENT_LABELS, PAQ_NAMES, SCENE_LABELS, SoundAQnetRunner
    root = Path(a.soundscaper) / "Inferring_soundscape_clips_for_LLM/application/system/model"
    runners = [SoundAQnetRunner(a.soundscaper, ckpt=c.name, device=a.device) for c in sorted(root.glob("*.pth"))]
    print(f"{len(runners)} checkpoints")
    rows = {"true": [], "65": []}
    for i, r in enumerate(tqdm(list(st.itertuples()), desc="model")):
        for k, cond in enumerate(("true", "65")):
            z = np.load(paths[i][k])
            outs = [run.predict(z["mel"][None], z["loud"][None]) for run in runners]
            mean = lambda key: np.mean([o[key][0] for o in outs], axis=0)  # noqa: E731
            isop = [float(o["ISOPleasant"][0]) for o in outs]
            row = {"file": r.stimulus_id, "stimulus_id": r.stimulus_id, "masker_type": r.masker_type, "smr": r.smr,
                   "fold": r.fold, "human_ISOPleasant": r.ISOPleasant, "human_ISOEventful": r.ISOEventful,
                   "n_ratings": r.n_ratings, "ISOPleasant": np.mean(isop), "ISOEventful": float(mean("ISOEventful")),
                   "ISOPleasant_sd": float(np.std(isop, ddof=1)) if len(isop) > 1 else 0.0}
            row.update({f"paq_{k_}": float(mean(k_)) for k_ in PAQ_NAMES})
            ev, sc = mean("event"), mean("scene")
            row.update({f"ev_{e}": float(ev[j]) for j, e in enumerate(EVENT_LABELS)})
            row.update({f"p_{s}": float(sc[j]) for j, s in enumerate(SCENE_LABELS)})
            rows[cond].append(row)
    for cond, rr in rows.items():
        d = pd.DataFrame(rr)
        d.to_csv(out / f"soundaqnet_full_{cond}.csv", index=False)
        from scipy.stats import spearmanr
        print(f"{cond}: wrote {len(d)} rows; SoundAQnet ISOPleasant vs listeners: Spearman {spearmanr(d.ISOPleasant, d.human_ISOPleasant)[0]:+.2f}")


if __name__ == "__main__":
    main()
