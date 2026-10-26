#!/usr/bin/env python3
"""Rate the SoundSCaper paper's own ARAUS test set (3,576 participant responses) and rebuild Fig. 6(a) and 6(b).

Why: Fig. 6 of the SoundSCaper repo is not computed on stimuli. It is computed on 3,576 individual responses
(Dataset_training_validation_test/test_set_*.txt: 85 participants, folds 0-5, one clip per response, one 1-5 rating per
response, binary event labels). Our earlier ARAUS reference used 42 fold-0 stimuli, averaged over listeners. This script uses
exactly the paper's rows, so what remains different from the paper is only our feature extraction and the checkpoint choice.

Steps
  1. read the shipped id list (fold_<f>_participant_<p>_stimulus_<s>) and match it to data/responses.csv
     (auto-detects the stimulus column; stops and prints the columns if fewer than 95% of the ids match)
  2. rebuild each clip (soundscape + masker at the SMR; silence masker = soundscape alone), 30 s, true calibrated level
  3. rate with every released checkpoint; keep the mean AND the default checkpoint (PAQ1027) separately
  4. Spearman matrices: (a) predicted events x predicted AQ, (b) shipped ground-truth events x shipped ratings,
     compared with the Fig. 6(a) values transcribed from the repo image

    python soundaqnet/06_rate_araus_papertest.py --araus data/raw/araus --soundscaper third_party/SoundSCaper \
        --workers 8 --device cuda --out ~/grids_3msde/araus_papertest
    add --limit 200 for a quick pipeline check (the comparison with the paper is then meaningless)
"""
import argparse
import os
import re
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from soundaqnet import features as F  # noqa: E402

# Fig. 6 order (rows) and columns, as in the SoundSCaper repo figure
EVENTS = ["Bird", "Animal", "Wind", "Water", "Natural sounds", "Vehicle", "Traffic", "Sounds of things",
          "Environment and background", "Outside, rural or natural", "Speech", "Human sounds", "Music", "Noise", "Silence"]
AQ = ["pleasant", "eventful", "chaotic", "vibrant", "uneventful", "calm", "annoying", "monotonous"]
# Fig. 6(a), transcribed from Figure/vertical_layout.PNG (rounded to 2 decimals; Fig. 6(b) is recomputed from the label files)
FIG6A = np.array([
    [.13, .03, -.06, .19, -.03, .12, -.08, -.19], [.18, .12, -.06, .27, -.11, .13, -.14, -.28],
    [-.17, .11, .14, .09, -.10, -.17, .10, 0], [-.09, 0, .04, -.01, -.04, -.07, .06, .11],
    [-.09, .03, .05, .03, -.06, -.08, .06, .08], [-.40, .27, .40, .02, -.21, -.40, .34, -.10],
    [-.10, .36, .19, .33, -.33, -.16, .05, -.31], [-.37, .29, .38, .06, -.24, -.39, .31, -.13],
    [-.05, -.17, -.03, -.09, .16, .01, 0, .22], [.27, .09, -.14, .27, -.10, .22, -.18, -.32],
    [.07, .66, .21, .65, -.62, -.13, -.03, -.69], [.06, .66, .22, .65, -.62, -.14, -.03, -.69],
    [-.10, .61, .34, .44, -.55, -.28, .19, -.62], [-.11, -.19, -.01, -.16, .19, -.03, .05, .28],
    [.25, -.42, -.37, -.33, .45, .37, -.25, .34]])

_MIXER = {}


def make_id(fold, participant, stim):
    p = int(re.sub(r"\D", "", str(participant)))
    return f"fold_{int(fold)}_participant_{p:05d}_stimulus_{int(stim):02d}"


def match_ids(resp, ids):
    """Find the (fold, stimulus) columns of responses.csv that reproduce the shipped ids; returns the matched frame."""
    idset = set(ids)
    folds = [c for c in resp.columns if c.lower() in ("fold_r", "fold")]
    stims = [c for c in resp.columns if "stim" in c.lower() and resp[c].dtype.kind in "iuf"]
    best = (0.0, None)
    for fc in folds:
        for sc in stims:
            try:
                made = [make_id(f, p, s) for f, p, s in zip(resp[fc], resp["participant"], resp[sc])]
            except (ValueError, TypeError):
                continue
            frac = len(idset & set(made)) / len(idset)
            print(f"  fold column {fc!r} + stimulus column {sc!r}: {100 * frac:.1f}% of the 3,576 ids found")
            if frac > best[0]:
                best = (frac, made)
    return best


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
    p = Path(cache) / f"{i:05d}_true.npz"
    if p.exists():
        return str(p)
    mx = _mixer(araus)
    if str(masker).lower().startswith("silence") or masker not in mx.m.index:
        from rsd.audio import read
        x_s, sr = read(mx.sdir / soundscape)
        x = float(mx.s.at[soundscape, "gain_s"]) * x_s            # no masker: the soundscape alone
    else:
        x, sr = mx.mix(soundscape, masker, float(smr))
    y = F.to_mono_44k(x, sr)[: F.CLIP_SECONDS * F.SR_WAV]
    np.savez(p, mel=F.log_mel(y), loud=F.iso_loudness(y * F.ARAUS_PA_PER_DIGITAL))
    return str(p)


def rho_matrix(E, P):
    out = np.full((E.shape[1], P.shape[1]), np.nan)
    for i in range(E.shape[1]):
        if np.nanstd(E[:, i]) > 0:
            for j in range(P.shape[1]):
                out[i, j] = spearmanr(E[:, i], P[:, j])[0]
    return out


def compare(name, M, ref):
    ok = ~np.isnan(M)
    print(f"{name}: cell rank-corr with Fig. 6(a) {spearmanr(M[ok], ref[ok])[0]:+.2f}, "
          f"mean |rho| {np.nanmean(np.abs(M)):.2f} (paper {np.abs(ref).mean():.2f}), "
          f"slope vs paper {np.polyfit(ref[ok], M[ok], 1)[0]:.2f}, same sign {100 * np.mean(np.sign(M[ok]) == np.sign(ref[ok])):.0f}%, "
          f"mean |diff| {np.nanmean(np.abs(M - ref)):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--araus", default="data/raw/araus")
    ap.add_argument("--soundscaper", default="third_party/SoundSCaper")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sc = Path(a.soundscaper) / "Dataset_training_validation_test"
    ids = [l.strip() for l in open(sc / "test_set_audio_file_ids.txt")]
    ev_lab = [l.rstrip("\n").split("\t") for l in open(sc / "test_set_audio_event_labels.txt")]
    paq_gt = np.loadtxt(sc / "test_set_PAQ_8D_AQs.txt")
    assert len(ids) == len(ev_lab) == len(paq_gt) == 3576, "unexpected test set size"

    resp = pd.read_csv(Path(a.araus) / "data" / "responses.csv")
    print("responses.csv columns:", list(resp.columns))
    frac, made = match_ids(resp, ids)
    if frac < 0.95:
        print("Could not match the shipped ids to responses.csv. First rows:\n", resp.head(3).T)
        sys.exit(2)
    resp = resp.assign(_id=made).drop_duplicates("_id").set_index("_id")
    miss = [i for i in ids if i not in resp.index]
    print(f"matched {len(ids) - len(miss)} of {len(ids)} ids; missing {len(miss)}")
    keep = [k for k, i in enumerate(ids) if i in resp.index]
    if a.limit:
        keep = keep[: a.limit]
    ids_k = [ids[k] for k in keep]
    rows = resp.loc[ids_k]
    print("masker types in the test set:", rows.masker.str.split("_").str[0].value_counts().to_dict())

    out = Path(a.out).expanduser()
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(i, a.araus, r.soundscape, r.masker, r.smr, str(cache)) for i, r in enumerate(rows.itertuples())]
    with ProcessPoolExecutor(a.workers) as ex:
        paths = list(tqdm(ex.map(job, jobs, chunksize=2), total=len(jobs), desc="features"))

    from soundaqnet.model import EVENT_LABELS, PAQ_NAMES, SoundAQnetRunner
    mdir = Path(a.soundscaper) / "Inferring_soundscape_clips_for_LLM/application/system/model"
    ckpts = sorted(mdir.glob("*.pth"))
    runners = {c.name: SoundAQnetRunner(a.soundscaper, ckpt=c.name, device=a.device) for c in ckpts}
    default = "SoundAQnet_ASC96_AEC94_PAQ1027.pth"
    print(f"{len(runners)} checkpoints, default {default}")
    recs = []
    for k, p in enumerate(tqdm(paths, desc="model")):
        z = np.load(p)
        o = {n: r.predict(z["mel"][None], z["loud"][None]) for n, r in runners.items()}
        rec = {"id": ids_k[k]}
        for tag, sel in (("mean", list(o)), ("def", [default])):
            for q in PAQ_NAMES:
                rec[f"{tag}_paq_{q}"] = float(np.mean([o[n][q][0] for n in sel]))
            ev = np.mean([o[n]["event"][0] for n in sel], axis=0)
            for j, e in enumerate(EVENT_LABELS):
                rec[f"{tag}_ev_{e}"] = float(ev[j])
        recs.append(rec)
    d = pd.DataFrame(recs)
    d.to_csv(out / "soundaqnet_papertest.csv", index=False)

    # ground truth (b) and predictions (a)
    gt = np.array([[e in ev_lab[k] for e in EVENTS] for k in keep], float)
    Gb = rho_matrix(gt, paq_gt[keep])
    pd.DataFrame(Gb, index=EVENTS, columns=AQ).to_csv(out / "rho_b_groundtruth.csv")
    print(f"\nrows rated: {len(d)}")
    for tag in ("mean", "def"):
        E = d[[f"{tag}_ev_{e}" for e in EVENTS]].values
        P = d[[f"{tag}_paq_{q}" for q in AQ]].values
        M = rho_matrix(E, P)
        pd.DataFrame(M, index=EVENTS, columns=AQ).to_csv(out / f"rho_a_{tag}.csv")
        compare(f"(a) predictions, {'mean of checkpoints' if tag == 'mean' else 'default checkpoint'}", M, FIG6A)
    print(f"\nwrote {out}/soundaqnet_papertest.csv, rho_a_mean.csv, rho_a_def.csv, rho_b_groundtruth.csv")


if __name__ == "__main__":
    main()
