#!/usr/bin/env python3
"""Rate every wav in a folder with SoundAQnet and keep ALL outputs (not only ISOPleasant).

Per file: mono, assumed level --levels dB SPL rms (default 65), every released checkpoint; the outputs are averaged over
checkpoints and levels. Columns: file, ISOPleasant, ISOEventful, ISOPleasant_sd (between checkpoints/levels), paq_<8 names>,
ev_<15 event classes> (sigmoid probabilities), p_<3 scenes>. Same names as soundaqnet/01_rate_isd.py. The features are cached in
<folder>/cache_soundaqnet (same cache as clotho/10_score_grid_soundaqnet.py, so a file rated there at the same level is not
recomputed; about 60 s per 30 s clip for the loudness on 2 cores).

    python soundaqnet/04_rate_folder_full.py ~/grids_3msde/base_ae_g12 --workers 8 --device cuda
    (also for a reference folder of real recordings, e.g. ARAUS stimuli: --out <folder>/soundaqnet_full.csv)
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
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soundaqnet import features as F  # noqa: E402


def job(args):
    wav, level, cache = args
    import torch
    torch.set_num_threads(1)
    path = Path(cache) / f"{Path(wav).stem}__{level:g}.npz"
    if path.exists():
        return str(path)
    x, sr = sf.read(wav, dtype="float64", always_2d=True)
    y = F.to_mono_44k(x, sr)[: F.CLIP_SECONDS * F.SR_WAV]
    y_pa = y * F.assume_level(y, rms_db_spl=level)
    np.savez(path, mel=F.log_mel(y_pa / F.ARAUS_PA_PER_DIGITAL), loud=F.iso_loudness(y_pa))
    return str(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--soundscaper", default="third_party/SoundSCaper")
    ap.add_argument("--levels", type=float, nargs="+", default=[65.0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="default <folder>/soundaqnet_full.csv")
    a = ap.parse_args()

    folder = Path(a.folder).expanduser()
    files = sorted(folder.glob("*.wav"))
    if a.limit:
        files = files[: a.limit]
    print(f"{len(files)} wav files in {folder}; levels {a.levels}")
    cache = folder / "cache_soundaqnet"
    cache.mkdir(exist_ok=True)
    jobs = [(str(p), lv, str(cache)) for p in files for lv in a.levels]
    with ProcessPoolExecutor(a.workers) as ex:
        paths = list(tqdm(ex.map(job, jobs, chunksize=1), total=len(jobs), desc="features"))
    paths = np.array(paths).reshape(len(files), len(a.levels))

    from soundaqnet.model import EVENT_LABELS, PAQ_NAMES, SCENE_LABELS, SoundAQnetRunner
    root = Path(a.soundscaper) / "Inferring_soundscape_clips_for_LLM/application/system/model"
    ckpts = sorted(root.glob("*.pth"))
    print(f"{len(ckpts)} checkpoints")
    runners = [SoundAQnetRunner(a.soundscaper, ckpt=c.name, device=a.device) for c in ckpts]
    rows = []
    for i, p in enumerate(tqdm(files, desc="model")):
        outs = []
        for j in range(len(a.levels)):
            z = np.load(paths[i, j])
            for run in runners:
                outs.append(run.predict(z["mel"][None], z["loud"][None]))
        mean = lambda key: np.mean([o[key][0] for o in outs], axis=0)  # noqa: E731
        isop = [float(o["ISOPleasant"][0]) for o in outs]
        row = {"file": p.name, "ISOPleasant": np.mean(isop), "ISOEventful": float(mean("ISOEventful")),
               "ISOPleasant_sd": float(np.std(isop, ddof=1)) if len(isop) > 1 else 0.0}
        row.update({f"paq_{k}": float(mean(k)) for k in PAQ_NAMES})
        ev, sc = mean("event"), mean("scene")
        row.update({f"ev_{e}": float(ev[k]) for k, e in enumerate(EVENT_LABELS)})
        row.update({f"p_{s}": float(sc[k]) for k, s in enumerate(SCENE_LABELS)})
        rows.append(row)
    out = Path(a.out).expanduser() if a.out else folder / "soundaqnet_full.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
