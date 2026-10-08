#!/usr/bin/env python3
"""Check the DGL-free SoundAQnet port and the feature extraction against the demo that ships with
SoundSCaper. Run this first on every new machine.

    python soundaqnet/00_demo_check.py --soundscaper third_party/SoundSCaper [--device cuda]

Part 1 feeds the shipped log-mel and loudness features to the model: ISOPleasant, ISOEventful,
the eight PAQ values and the 15 event probabilities must match the shipped output files to
about 1e-5. Part 2 recomputes the features from the shipped wav (log-mel with torchlibrosa,
loudness with MoSQITo instead of the Windows ISO_532-1.exe) and compares the outputs; small
differences (about 0.01 on ISOPleasant) come from the resampler and the loudness code.
The loudness step takes about a minute for a 30 s clip on two cores.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soundaqnet import features as F  # noqa: E402
from soundaqnet.model import PAQ_NAMES, SCENE_LABELS, SoundAQnetRunner  # noqa: E402

NAME = "fold_1_participant_00056_stimulus_13"


def show(tag, o):
    print(f"{tag:14s} scene {SCENE_LABELS[o['scene'][0].argmax()]:13s} ISOP {o['ISOPleasant'][0]:.4f}  ISOE {o['ISOEventful'][0]:.4f}  "
          "PAQ8 " + " ".join(f"{o[k][0]:.3f}" for k in PAQ_NAMES))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--soundscaper", default="third_party/SoundSCaper")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-features", action="store_true", help="only part 1 (fast)")
    a = ap.parse_args()
    R = Path(a.soundscaper)
    run = SoundAQnetRunner(R, device=a.device)

    mel = np.load(R / "Feature_log_mel" / "Dataset_mel" / f"{NAME}.npy")[None]
    loud = np.load(R / "Feature_loudness_ISO532_1" / "Dataset_wav_loudness" / f"{NAME}.npy", allow_pickle=True)[None]
    o1 = run.predict(mel, loud)
    out = R / "Inferring_soundscape_clips_for_LLM" / "application"
    lines = (out / "SoundAQnet_scene_ISOPl_ISOEv_PAQ8DAQs" / f"{NAME}_scene_PAQ.txt").read_text().split("\n")
    isop, isoe = map(float, lines[1].split())
    paq = np.array(lines[2].split(), float)
    ev = np.loadtxt(out / "SoundAQnet_event_probability" / f"{NAME}_event.txt")
    show("shipped feats", o1)
    print(f"{'shipped output':14s} scene {lines[0]:13s} ISOP {isop:.4f}  ISOE {isoe:.4f}  PAQ8 " + " ".join(f"{v:.3f}" for v in paq))
    d = max(abs(o1["ISOPleasant"][0] - isop), abs(o1["ISOEventful"][0] - isoe),
            np.abs(np.array([o1[k][0] for k in PAQ_NAMES]) - paq).max(), np.abs(ev - o1["event"][0]).max())
    print(f"part 1: max abs difference to the shipped output {d:.2e}  ->", "OK" if d < 1e-4 else "MISMATCH")
    if a.skip_features:
        return

    x, sr = sf.read(R / "Feature_log_mel" / "Dataset_wav" / f"{NAME}.wav", dtype="float64")
    t = time.time()
    m2, l2 = F.extract(x, sr, F.ARAUS_PA_PER_DIGITAL)
    print(f"features recomputed in {time.time() - t:.0f} s")
    o2 = run.predict(m2[None], l2[None])
    show("own features", o2)
    print(f"part 2: ISOP difference {o2['ISOPleasant'][0] - isop:+.4f}, ISOE {o2['ISOEventful'][0] - isoe:+.4f}",
          "->", "OK" if abs(o2["ISOPleasant"][0] - isop) < 0.03 and abs(o2["ISOEventful"][0] - isoe) < 0.03 else "CHECK")


if __name__ == "__main__":
    main()
