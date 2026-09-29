"""End-to-end smoke test on small synthetic stand-ins for ARAUS and ISD.

Checks that every script runs and that the file formats line up. It does not
download anything; set RSD_FAKE_EMBED=0 to exercise the real CLAP model.

    python tests/smoke_test.py
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
PAQ = ["pleasant", "vibrant", "eventful", "chaotic", "annoying", "monotonous", "uneventful", "calm"]
rng = np.random.default_rng(1)


def tone_noise(sr, secs, f, noise, ch):
    t = np.arange(int(sr * secs)) / sr
    x = 0.1 * np.sin(2 * np.pi * f * t) * (1 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.2, 3) * t))
    x = x[:, None] + noise * rng.standard_normal((len(t), ch))
    return np.repeat(x, ch, axis=1)[:, :ch] if x.shape[1] == 1 else x


def paq_from(score):
    """Crude PAQ answers that move with a latent 'pleasantness' score."""
    p = np.clip(np.round(3 + 1.5 * score + rng.normal(0, 0.7, 8) * [1, 1, 1, -1, -1, 1, 1, 1]), 1, 5)
    p[3] = np.clip(6 - p[0] + rng.integers(-1, 2), 1, 5)   # chaotic ~ opposite of pleasant
    p[4] = np.clip(6 - p[0] + rng.integers(-1, 2), 1, 5)   # annoying
    return dict(zip(PAQ, p.astype(int)))


def psycho(level):
    return {"LA50": level, "LA10": level + 4, "LA90": level - 4, "LC50": level + 6, "N5": level / 3,
            "R": rng.uniform(0.02, 0.08), "F": rng.uniform(0.01, 0.05), "T": rng.uniform(0.1, 0.5)}


def make_araus(root: Path):
    (root / "data").mkdir(parents=True)
    (root / "soundscapes").mkdir()
    (root / "maskers").mkdir()
    sr = 44100
    sounds = []
    for i in range(12):
        name = f"R{i:04d}_segment_binaural_44100_1.wav"
        sf.write(str(root / "soundscapes" / name), tone_noise(sr, 30, 100 + 40 * i, 0.05, 2), sr)
        sounds.append({"soundscape": name, "gain_s": 1.0, "insitu_leq": 55 + i, "fold_s": i % 6})
    masker_types = {"bird": 3000, "water": 800, "traffic": 120, "construction": 400}
    maskers, pleasant_by_type = [], {"bird": 0.6, "water": 0.4, "traffic": -0.6, "construction": -0.8}
    for mt, f in masker_types.items():
        for j in range(2):
            name = f"{mt}_{j:05d}.wav"
            sf.write(str(root / "maskers" / name), tone_noise(sr, 30, f * (1 + 0.1 * j), 0.02, 1)[:, 0], sr)
            row = {"masker": name}
            for db in range(46, 84):
                row[f"gain_{db}dB"], row[f"leq_at_gain_{db}dB"] = 10 ** ((db - 70) / 20), db
            maskers.append(row)
    pd.DataFrame(sounds).to_csv(root / "data/soundscapes.csv", index=False)
    pd.DataFrame(maskers).to_csv(root / "data/maskers.csv", index=False)

    resp, parts = [], []
    for p in range(60):
        pid = f"ARAUS_{p:05d}"
        parts.append({"participant": pid, "age": 30, "gender": "x", "who": 15, "pss": 15, "wnss": 3,
                      "panas_pos": 30, "panas_neg": 15})
        fold = p % 6
        own = [s for s in sounds if s["fold_s"] == fold]
        for k in range(10):
            s = own[k % len(own)]
            m = maskers[rng.integers(len(maskers))]["masker"]
            smr = int(rng.choice([-6, -3, 0, 3, 6]))
            mt = m.split("_")[0]
            score = pleasant_by_type[mt] * (1 - smr / 12) + rng.normal(0, 0.5)
            ps = psycho(s["insitu_leq"])
            resp.append({"participant": pid, "fold_r": fold, "soundscape": s["soundscape"], "masker": m,
                         "smr": smr, "stimulus_index": k + 2, "is_attention": 0, **paq_from(score),
                         "LA50_r": ps["LA50"], "LA10_r": ps["LA10"], "LA90_r": ps["LA90"], "LC50_r": ps["LC50"],
                         "N05_r": ps["N5"], "Ravg_r": ps["R"], "Favg_r": ps["F"], "Tavg_r": ps["T"]})
        # practice stimulus (fold -1) and an attention stimulus - must be dropped by the loader
        resp.append({**resp[-1], "fold_r": -1, "stimulus_index": 1})
        resp.append({**resp[-1], "fold_r": fold, "is_attention": 1, "stimulus_index": 20})
    pd.DataFrame(resp).to_csv(root / "data/responses.csv", index=False)
    pd.DataFrame(parts).to_csv(root / "data/participants.csv", index=False)


def make_isd(root: Path):
    sr = 48000
    rows, rec = [], 0
    locations = {"Park": ("PK", 0.6, 2500), "Street": ("ST", -0.5, 150), "Square": ("SQ", 0.1, 600)}
    for loc, (pre, base, f) in locations.items():
        d = root / "WAV_City_1" / loc
        d.mkdir(parents=True)
        for g in range(8):
            gid = f"{pre}{100 + g}"
            fname = f"{gid}.hdf.wav" if g == 0 else f"{gid}.wav"  # mimic the odd Groningen name
            sf.write(str(d / fname), 0.02 * tone_noise(sr, 35, f * (1 + 0.05 * g), 0.1, 2), sr, subtype="FLOAT")
            for _ in range(2):
                rec += 1
                ps = psycho(60 + 10 * (base < 0) + rng.normal(0, 3))
                rows.append({"LocationID": loc, "SessionID": f"{loc}1", "GroupID": gid, "RecordID": rec,
                             **paq_from(base + rng.normal(0, 0.5)), "sss01": 3, "sss05": int(rng.integers(1, 6)),
                             "LAeq_L50(A)": ps["LA50"], "LAeq_L10(A)": ps["LA10"], "LAeq_L90(A)": ps["LA90"],
                             "LCeq_L50(C)": ps["LC50"], "N_N5": ps["N5"], "R_R": ps["R"], "FS_F": ps["F"],
                             "T_TonalityHMS": ps["T"]})
        # macOS junk that also ends in .wav and must be ignored
        junk = root / "__MACOSX" / "WAV_City_1" / loc
        junk.mkdir(parents=True)
        (junk / f"._{pre}100.wav").write_bytes(b"not audio")
    # an unrated recording (like the lockdown sessions) and a rating without audio
    d = root / "WAV_City_1" / "Park"
    sf.write(str(d / "PK900.wav"), 0.02 * tone_noise(sr, 35, 2000, 0.1, 2), sr, subtype="FLOAT")
    rows.append({**rows[0], "GroupID": "XX1", "RecordID": 9999})
    pd.DataFrame(rows).to_csv(root / "ISD v1.0 Data.csv", index=False)


def run(*args, env):
    print("\n$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=REPO, env=env, check=True)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_araus(tmp / "araus")
        make_isd(tmp / "isd")
        env = {**os.environ, "RSD_FAKE_EMBED": os.environ.get("RSD_FAKE_EMBED", "1")}
        P, F, R = str(tmp / "proc"), str(tmp / "feat"), str(tmp / "res")
        run("scripts/01_prepare.py", "--araus", str(tmp / "araus"), "--isd", str(tmp / "isd"), "--out", P, env=env)
        for ds in ["araus", "isd"]:
            run("scripts/02_embed.py", "--dataset", ds, "--araus", str(tmp / "araus"), "--processed", P,
                "--out", F, "--workers", "2", env=env)
        run("scripts/02_embed.py", "--dataset", "isd", "--processed", P, "--out", F, "--workers", "0", env=env)  # resume = no-op
        run("scripts/03_train_evaluate.py", "--processed", P, "--features", F, "--out", R, "--n-boot", "200", env=env)
        run("scripts/04_clusters.py", "--processed", P, "--features", F, "--out", R + "/clusters", "--pca", "10",
            "--min-cluster-size", "3", env=env)
        run("scripts/05_listening_sample.py", "--processed", P, "--clusters", R + "/clusters",
            "--araus", str(tmp / "araus"), "--out", R + "/listening", "--per-cluster", "2",
            "--araus-per-cluster", "1", env=env)
        n = len(pd.read_csv(Path(R) / "metrics.csv"))
        assert n > 0, "no metrics written"
        print(f"\nSMOKE TEST PASSED ({n} metric rows)")
