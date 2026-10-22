#!/usr/bin/env python3
"""Cut short, loudness-matched demo clips for the talk from the generated grids.

For every demo in DEMOS the script takes the files of one grid (and one prompt) at the given levels, cuts an excerpt
(default 10 s from second 5), sets all clips to the same loudness (default -23 LUFS) and writes numbered wav files, a
playlist (demo.m3u) and demo_list.csv (what each clip is, for your notes).

Seed: with --seed auto the script picks, per demo, the seed whose effect (sharpness at the high level minus at the low
level, from metrics.csv) is closest to the median over the seeds, i.e. a typical example and not the best one. With
--seed 1 (or 2, 3) it uses that seed for all demos. Say on the slide which you used.

    python generation/make_demo_clips.py --root ~/Downloads/Downloads/grids_3msde --out ~/demo_clips
"""
import argparse, csv, re, shutil, statistics, subprocess
from pathlib import Path

# (id, what it shows, grid, prompt substring, levels)
DEMOS = [
    ("1", "Base model with words: the word changes the sound (no dominant source)",
     "base_with_words", "no_dominant", [-0.7, 0.7]),
    ("2", "Fine-tuned numbers model, 5,000 steps: slight rise with the score (birdsong and wind)",
     "numbers_5000", "birdsong", [-0.7, 0.7]),
    ("3", "Same model: sharpness rises clearly, but no difference was heard (waves)",
     "numbers_5000", "waves", [-0.7, 0.7]),
    ("4", "Words model, 5,000 steps: the low score sounded more pleasant (natural + human sounds)",
     "words_5000", "natural_sounds_and_human", [-0.7, 0.7]),
    ("5a", "Numbers model: at -0.7 three different prompts sound alike, at +0.7 they differ more",
     "numbers_5000", None, [-0.7, 0.7]),
    ("5b", "Base model: the same three prompts differ at both levels",
     "base_with_numbers", None, [-0.7, 0.7]),
]
THREE = ["birdsong", "traffic_noise_and_human", "other_noise"]   # prompts for demo 5


def read_grid(folder):
    with open(folder / "grid.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["seed"], r["p"] = int(r["seed"]), float(r["p"])
        r["key"] = r["masker"].replace(" ", "_")
    return rows


def pick_seed(folder, rows, sub, lo, hi, want):
    if want != "auto":
        return int(want)
    mp = folder / "metrics.csv"
    seeds = sorted({r["seed"] for r in rows})
    if not mp.exists():
        return seeds[0]
    with open(mp, newline="", encoding="utf-8") as f:
        m = list(csv.DictReader(f))
    if not m or "sharp" not in m[0]:
        return seeds[0]
    eff = {}
    for s in seeds:
        v = {}
        for r in m:
            if sub in r["masker"] and int(r["seed"]) == s and r.get("sharp") not in (None, ""):
                v[round(float(r["p"]), 2)] = float(r["sharp"])
        if round(lo, 2) in v and round(hi, 2) in v:
            eff[s] = v[round(hi, 2)] - v[round(lo, 2)]
    if not eff:
        return seeds[0]
    med = statistics.median(eff.values())
    return min(eff, key=lambda s: abs(eff[s] - med))


def lufs(path):
    res = subprocess.run(["ffmpeg", "-nostats", "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
                         capture_output=True, text=True)
    m = re.findall(r"^\s*I:\s*(-?[\d.]+) LUFS", res.stderr, re.M)
    return float(m[-1]) if m else None


def cut(src, dst, start, secs, target):
    tmp = dst.with_suffix(".tmp.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-t", str(secs), "-i", str(src),
                    "-af", "afade=t=in:d=0.2,afade=t=out:st=%s:d=0.5" % (secs - 0.5), str(tmp)], check=True)
    g = 0.0 if target is None else target - (lufs(tmp) or target)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-af",
                    f"volume={g:.2f}dB,alimiter=limit=0.95", str(dst)], check=True)
    tmp.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder with the grids (numbers_5000, words_5000, ...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", default="auto", help="auto, 1, 2 or 3")
    ap.add_argument("--start", type=float, default=5.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--lufs", type=float, default=-23.0, help="use -1 to keep original levels")
    a = ap.parse_args()
    root, out = Path(a.root).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    target = None if a.lufs == -1 else a.lufs
    entries, table, n = [], [], 0
    for did, what, grid, sub, levels in DEMOS:
        folder = root / grid
        rows = read_grid(folder)
        subs = [sub] if sub else THREE
        seeds = {}
        for sb in subs:
            if not [r for r in rows if sb in r["key"]]:
                print(f"!! no prompt containing '{sb}' in {grid}")
                continue
            seeds[sb] = pick_seed(folder, rows, sb, min(levels), max(levels), a.seed)
        # one prompt: low then high level; several prompts (demo 5): all prompts at one level, then the next level
        order = [(sb, p) for sb in seeds for p in levels] if sub else [(sb, p) for p in levels for sb in seeds]
        for sb, p in order:
            seed = seeds[sb]
            r = next((r for r in rows if sb in r["key"] and r["seed"] == seed and abs(r["p"] - p) < 1e-6), None)
            if r is None:
                print(f"!! missing {grid} {sb} seed {seed} p {p}"); continue
            n += 1
            name = f"{n:02d}_demo{did}_{grid}_{r['key']}_p{p:+.1f}.wav"
            cut(folder / r["file"], out / name, a.start, a.seconds, target)
            entries.append((f"{did} | {sb} | p={p:+.1f} | seed {seed}", name))
            table.append({"clip": name, "demo": did, "what": what, "grid": grid, "prompt": r["masker"],
                          "p": p, "seed": seed, "source_file": r["file"]})
    with open(out / "demo.m3u", "w", encoding="utf-8") as f:
        f.write("#EXTM3U\n")
        for label, name in entries:
            f.write(f"#EXTINF:-1,{label}\n{name}\n")
    with open(out / "demo_list.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)
    print(f"{len(table)} clips -> {out}/demo.m3u  (notes: {out}/demo_list.csv)")


if __name__ == "__main__":
    main()
