#!/usr/bin/env python3
"""Open (labelled) listening playlists, one folder per result slide of the talk.

Selection rule, fixed in advance and written to README.md in every folder: seed 1, the two extreme ratings (-0.7, +0.7),
and prompts picked by their effect (lowest, median, highest), not by how they sound.
  results_1_acoustic_features   fine-tuned ISD (3 of 7 prompts: lowest, median, highest sharpness effect in seed SD) and
                                fine-tuned ARAUS (2 of 5 prompts: lowest and highest |sharpness effect|)
  base_model_1                  the same 3 ISD prompts, base model (rating as number) next to the fine-tuned ISD model
  results_2_and_base_model_2    six common prompts: 3 of 6 (lowest, median, highest mean predicted-pleasantness effect of
                                ARAUS and ISD); base model (words), fine-tuned ARAUS, fine-tuned ISD
Each folder: playlist.m3u (relative paths), audio/ (the original wav files, copied unchanged), README.md.

    python3 generation/make_slide_playlists.py --grids ~/grids_3msde --features ~/deck/figures/features_effects.csv --out ~/deck/playlists
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

LV = [-0.7, 0.7]


def load(root, folders, suffix=""):
    out = []
    for f in folders:
        f = f + suffix
        m = pd.read_csv(root / f / "metrics.csv")
        m["p"] = m.p.round(2)
        m["grid"] = f
        m["path"] = [str(root / f / f"s{int(s)}_{k}_p{p:.2f}.wav") for s, k, p in zip(m.seed, m.masker, m.p)]
        out.append(m[["seed", "masker", "p", "sq_isop", "grid", "path"]])
    return pd.concat(out)


def pick3(series):
    s = series.sort_values()
    return [s.index[0], s.index[len(s) // 2], s.index[-1]]


def label(masker):
    return masker.replace("soundscape_with_", "").replace("_", " ")


def build(out, name, rows, text):
    d = Path(out) / name
    (d / "audio").mkdir(parents=True, exist_ok=True)
    lines = ["#EXTM3U"]
    for i, (title, path) in enumerate(rows, 1):
        safe = "".join(c if c.isalnum() or c in "+-." else "_" for c in title.replace(" ", "_"))
        dst = d / "audio" / f"{i:02d}_{safe}.wav"
        shutil.copyfile(Path(path).resolve(), dst)   # original file, no re-encoding
        lines += [f"#EXTINF:30,{title}", f"audio/{dst.name}"]
    (d / "playlist.m3u").write_text("\n".join(lines) + "\n", encoding="utf8")
    (d / "README.md").write_text(text + "\n\nOrder in the playlist:\n\n" + "\n".join(f"{i}. {t}" for i, (t, _) in enumerate(rows, 1)) + "\n", encoding="utf8")
    print(name, len(rows), "files")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grids", required=True)
    ap.add_argument("--features", required=True, help="features_effects.csv (plot_grid_features.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--suffix", default="", help="added to every grid folder name, e.g. _chirp (isd/18_birds_chirping.sh)")
    a = ap.parse_args()
    root = Path(a.grids).expanduser()
    fe = pd.read_csv(Path(a.features).expanduser())
    fe = fe[fe.metric == "sharp"]
    isd = load(root, ["numbers_5000_g12"], a.suffix); araus = load(root, ["araus_6000_g12"], a.suffix); base_isd = load(root, ["base_with_numbers_g12"], a.suffix)
    sel_isd = pick3(fe[fe.grid == "ISD"].set_index("masker").ratio)
    fa = fe[fe.grid == "ARAUS"].set_index("masker").ratio.abs().sort_values()
    sel_araus = [fa.index[0], fa.index[-1]]
    rule1 = ("Seed 1, ratings -0.7 and +0.7, original level (wav, not equalised). Prompts chosen by effect, not by sound: ISD = lowest, median, "
             "highest sharpness effect in seed standard deviations (the lowest is a prompt where sharpness falls); ARAUS = lowest and highest absolute effect.")
    rule_base = ("Seed 1, ratings -0.7 and +0.7, original level (wav, not equalised). The three ISD prompts of the Results (1) folder "
                 "(lowest, median, highest sharpness effect), base model with the rating as a number next to the fine-tuned ISD model.")
    rows = []
    for m in sel_isd:
        for p in LV:
            rows.append((f"ISD model | {label(m)} | {p:+.1f}", isd[(isd.masker == m) & (isd.seed == 1) & (isd.p == p)].path.iloc[0]))
    for m in sel_araus:
        for p in LV:
            rows.append((f"ARAUS model | {label(m)} | {p:+.1f}", araus[(araus.masker == m) & (araus.seed == 1) & (araus.p == p)].path.iloc[0]))
    build(a.out, "results_1_acoustic_features", rows, "# Results (1): acoustic features\n\nSlide claim: sharpness changes clearly with the rating in ISD (6 of 7 prompts sharper at +0.7), hardly any clear change in ARAUS.\n\n" + rule1)
    rows = []
    for m in sel_isd:
        for name, g in [("base model (number)", base_isd), ("ISD model", isd)]:
            for p in LV:
                rows.append((f"{name} | {label(m)} | {p:+.1f}", g[(g.masker == m) & (g.seed == 1) & (g.p == p)].path.iloc[0]))
    build(a.out, "base_model_1", rows, "# Base model (1): acoustic features without fine-tuning\n\nSlide claim: the base model already changes with the tag in some prompts, so only changes beyond it count.\n\n" + rule_base)
    ar = load(root, ["common_araus_6000_g12", "common6new_araus_6000_g12"], a.suffix); isd6 = load(root, ["common_numbers_5000_g12", "common6new_numbers_5000_g12"], a.suffix)
    bw = load(root, ["common6_base_words_g12"], a.suffix)
    eff = {}
    for k, d in [("ARAUS", ar), ("ISD", isd6)]:
        pv = d[d.p.isin(LV)].pivot_table(index=["masker", "seed"], columns="p", values="sq_isop")
        eff[k] = (pv[0.7] - pv[-0.7]).groupby("masker").mean()
    mean_eff = pd.DataFrame(eff).mean(axis=1)
    sel6 = pick3(mean_eff)
    rows = []
    for m in sel6:
        for name, g in [("base model (words)", bw), ("ARAUS model", ar), ("ISD model", isd6)]:
            for p in LV:
                rows.append((f"{name} | {label(m)} | {p:+.1f}", g[(g.masker == m) & (g.seed == 1) & (g.p == p)].path.iloc[0]))
    txt = ("# Results (2) and Base model (2): predicted pleasantness\n\nSlide claim: ARAUS and ISD rise above the base model (rating as words); on the six common prompts.\n\n"
           "Seed 1, ratings -0.7 and +0.7, original level (wav, not equalised). Prompts: lowest, median and highest mean predicted-pleasantness effect "
           "of the ARAUS and ISD models (" + "; ".join(f"{label(m)}: {mean_eff[m]:+.2f}" for m in sel6) + ").")
    build(a.out, "results_2_and_base_model_2", rows, txt)


if __name__ == "__main__":
    main()
