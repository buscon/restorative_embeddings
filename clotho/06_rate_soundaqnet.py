#!/usr/bin/env python3
"""Rate the CLAP-checked Clotho clips with SoundAQnet and measure how stable each rating is.

Clotho clips are uncalibrated, but SoundAQnet depends on absolute level. So each clip is set to
several assumed overall levels (--levels, dB SPL rms, unweighted), and rated by every released
checkpoint. The result is a grid of levels x checkpoints predictions per clip. Per clip we write

    isop_mean     mean ISOPleasant over the grid (used as the rating)
    isop_sd       standard deviation over the whole grid
    isop_sd_ckpt  sd over checkpoints (mean over levels first removed: average of per-level sds)
    isop_sd_lvl   sd over levels (of the checkpoint-mean)
    isop_paq      ISOPleasant recomputed from the 8 PAQ outputs (ISO 12913-3 formula, mean over grid)
    isop_gap      |isop_mean - isop_paq|   (internal consistency)
    isoe_mean, paq_*_mean, p_<scene>, ev_<event>   mean over the grid, for later inspection

Stability is not validity: a rating can be stable and still wrong for this kind of audio. This
script only measures whether the model gives the same answer under harmless changes. The
threshold is applied in 07_build_training_set.py.

Features (log-mel, ISO 532-1 loudness) are cached per clip and level, so the script can be stopped
and rerun. Loudness is the slow part (20 to 60 s per 30 s clip and level): use many --workers.

    python clotho/06_rate_soundaqnet.py --soundscaper third_party/SoundSCaper \
        --checked clotho/checked/selected_checked.csv --audio-dir data/raw/clotho/audio_selected \
        --out clotho/rated.csv --workers 12 --device cuda [--limit 20]
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soundaqnet import features as F  # noqa: E402


def job(args):
    name, wav, level, cache = args
    path = Path(cache) / f"{Path(name).stem}__{level:g}.npz"
    if path.exists():
        return str(path)
    x, sr = sf.read(wav, dtype="float64", always_2d=True)
    y = F.to_mono_44k(x, sr)[: F.CLIP_SECONDS * F.SR_WAV]
    k = F.assume_level(y, rms_db_spl=level)  # pascal per file unit
    y_pa = y * k
    np.savez(path, mel=F.log_mel(y_pa / F.ARAUS_PA_PER_DIGITAL), loud=F.iso_loudness(y_pa), sec=len(y) / F.SR_WAV)
    return str(path)


def isop_from_paq(p):
    """ISO 12913-3: (pl - an) cos45 + (ca - ch) + (vi - mo) cos45, scaled to [-1, 1] by 4 + sqrt(32)
    (the SoundSCaper code divides by this; the paper writes 8 + sqrt(32), see soundaqnet/README)."""
    c = np.sqrt(2) / 2
    return ((p["pleasant"] - p["annoying"]) * c + (p["calm"] - p["chaotic"]) + (p["vibrant"] - p["monotonous"]) * c) \
        / (4 + np.sqrt(32))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--soundscaper", default="third_party/SoundSCaper")
    ap.add_argument("--checked", default="clotho/checked/selected_checked.csv")
    ap.add_argument("--audio-dir", default="data/raw/clotho/audio_selected")
    ap.add_argument("--out", default="clotho/rated.csv")
    ap.add_argument("--cache", default="clotho/cache_features")
    ap.add_argument("--levels", type=float, nargs="+", default=[60.0, 65.0, 70.0],
                    help="assumed rms levels in dB SPL; the middle one is the nominal level")
    ap.add_argument("--min-z", type=float, default=1.0, help="CLAP z-score of the best caption")
    ap.add_argument("--soundscape-only", action="store_true",
                    help="only clips with a soundscape tag and >= 2 keyword captions (1,544 clips); about half the work")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    df = pd.read_csv(a.checked)
    df = df[df.best_z >= a.min_z].reset_index(drop=True)
    if a.soundscape_only:
        df = df[df.scape_tag & (df.n_any >= 2)].reset_index(drop=True)
    if a.limit:
        df = df.head(a.limit)
    print(f"{len(df)} clips (best_z >= {a.min_z}); levels {a.levels}")
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    jobs = [(r.file_name, str(Path(a.audio_dir) / r.file_name), lv, a.cache) for r in df.itertuples() for lv in a.levels]
    with ProcessPoolExecutor(a.workers) as ex:
        paths = list(tqdm(ex.map(job, jobs, chunksize=1), total=len(jobs), desc="features"))
    paths = np.array(paths).reshape(len(df), len(a.levels))

    from soundaqnet.model import EVENT_LABELS, PAQ_NAMES, SCENE_LABELS, SoundAQnetRunner
    root = Path(a.soundscaper) / "Inferring_soundscape_clips_for_LLM/application/system/model"
    ckpts = sorted(p.name for p in root.glob("*.pth"))
    print("checkpoints:", ckpts)
    runners = [SoundAQnetRunner(a.soundscaper, ckpt=c, device=a.device) for c in ckpts]

    rows = []
    for i, r in enumerate(tqdm(df.itertuples(), total=len(df), desc="model")):
        grid = []  # one dict of scalars/arrays per (level, ckpt)
        for lv_i in range(len(a.levels)):
            z = np.load(paths[i, lv_i])
            for run in runners:
                o = run.predict(z["mel"][None], z["loud"][None])
                grid.append({k: v[0] for k, v in o.items()})
        isop = np.array([[g["ISOPleasant"] for g in grid[l * len(runners):(l + 1) * len(runners)]]
                         for l in range(len(a.levels))])  # (levels, ckpts)
        paq = {k: np.mean([g[k] for g in grid]) for k in PAQ_NAMES}
        ip = float(np.mean([isop_from_paq({k: g[k] for k in PAQ_NAMES}) for g in grid]))
        scene = np.mean([g["scene"] for g in grid], axis=0)
        ev = np.mean([g["event"] for g in grid], axis=0)
        row = {"file_name": r.file_name, "sec": float(z["sec"]),
               "isop_mean": isop.mean(), "isop_sd": isop.std(ddof=1),
               "isop_sd_ckpt": isop.std(axis=1, ddof=1).mean(), "isop_sd_lvl": isop.mean(axis=1).std(ddof=1),
               "isop_paq": ip, "isop_gap": abs(isop.mean() - ip),
               "isoe_mean": float(np.mean([g["ISOEventful"] for g in grid])), "scene": SCENE_LABELS[scene.argmax()]}
        row.update({f"paq_{k}": v for k, v in paq.items()})
        row.update({f"p_{s}": scene[j] for j, s in enumerate(SCENE_LABELS)})
        row.update({f"ev_{e}": ev[j] for j, e in enumerate(EVENT_LABELS)})
        rows.append(row)
    out = df.merge(pd.DataFrame(rows), on="file_name")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} rows to {a.out}")
    print(out[["isop_mean", "isop_sd", "isop_sd_ckpt", "isop_sd_lvl", "isop_gap"]].describe().round(3))


if __name__ == "__main__":
    main()
