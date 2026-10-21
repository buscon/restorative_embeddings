#!/usr/bin/env python3
"""Build listening playlists (.m3u) from generated grids.

Reads <grid>/grid.csv (columns seed, masker, p, file, ...) written by generate_prompt_range.py / generate_grid.py.

Modes
  quality  Check for artifacts. One .m3u per grid, files in the order prompt > seed > level (lo, mid, hi).
           Not blind, original files, no copying.
  pairs    Blind paired comparison of two levels (default -0.7 vs +0.7), same prompt, seed and model.
           Trials from all given grids are pooled and shuffled; A/B order is random. Files are copied under
           anonymous names. Writes playlist.m3u, key.csv (do NOT open before rating) and responses.csv.
  models   Blind comparison of several grids (e.g. numbers vs words) for the same prompt, seed and level.
           One trial per (prompt, seed, level); the grids are the candidates, in random order.

Examples
  python generation/make_listening_playlists.py quality --out ~/listen/quality \
      --grids ~/grids_3msde/numbers_5000 ~/grids_3msde/words_5000 ~/grids_3msde/base_with_words --seeds 1
  python generation/make_listening_playlists.py pairs --out ~/listen/pairs_nature \
      --grids ~/grids_3msde/numbers_5000 ~/grids_3msde/words_5000 --maskers birds traffic --seeds 1 2 3
  python generation/make_listening_playlists.py models --out ~/listen/models_hi \
      --grids ~/grids_3msde/numbers_5000 ~/grids_3msde/words_5000 ~/grids_3msde/score_5000 --levels 0.7
"""
import argparse, csv, random, re, shutil, subprocess
from pathlib import Path


def load(grid):
    grid = Path(grid).expanduser()
    with open(grid / "grid.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["seed"], r["p"] = int(r["seed"]), float(r["p"])
        r["path"], r["grid"] = grid / r["file"], grid.name
    return rows


def select(rows, maskers, seeds):
    out = []
    for r in rows:
        if seeds and r["seed"] not in seeds:
            continue
        hay = (r["masker"] + " " + r.get("prompt_id", "")).lower()
        if maskers and not any(m.lower() in hay for m in maskers):
            continue
        out.append(r)
    return out


def find(rows, masker, seed, p):
    for r in rows:
        if r["masker"] == masker and r["seed"] == seed and abs(r["p"] - p) < 1e-6:
            return r
    return None


def loudness_lufs(path):
    res = subprocess.run(["ffmpeg", "-nostats", "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
                         capture_output=True, text=True)
    m = re.findall(r"^\s*I:\s*(-?[\d.]+) LUFS", res.stderr, re.M)
    return float(m[-1]) if m else None


def copy_audio(src, dst, target):
    """Copy src to dst; with target (LUFS) apply a linear gain (+ limiter) so loudness is equal across files."""
    if target is None:
        shutil.copyfile(src, dst)
        return None
    lufs = loudness_lufs(src)
    gain = 0.0 if lufs is None else target - lufs
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-af",
                    f"volume={gain:.2f}dB,alimiter=limit=0.95", str(dst)], check=True)
    return lufs


def write_m3u(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        f.write("#EXTM3U\n")
        for label, p in entries:
            f.write(f"#EXTINF:-1,{label}\n{p}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["quality", "pairs", "models"])
    ap.add_argument("--grids", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--maskers", nargs="*", default=[], help="substrings of masker / prompt_id; default all")
    ap.add_argument("--seeds", nargs="*", type=int, default=[])
    ap.add_argument("--lo", type=float, default=-0.7)
    ap.add_argument("--mid", type=float, default=0.0)
    ap.add_argument("--hi", type=float, default=0.7)
    ap.add_argument("--levels", nargs="*", type=float, default=None, help="models mode: levels to compare")
    ap.add_argument("--lufs", type=float, default=None,
                    help="pairs/models: equalise loudness to this LUFS (e.g. -23). Default: keep original levels")
    ap.add_argument("--rng", type=int, default=0, help="shuffle seed (keep it to reproduce the order)")
    a = ap.parse_args()

    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.rng)
    grids = {Path(g).expanduser().name: select(load(g), a.maskers, a.seeds) for g in a.grids}

    if a.mode == "quality":
        for name, rows in grids.items():
            entries = []
            for masker in sorted({r["masker"] for r in rows}):
                for seed in sorted({r["seed"] for r in rows}):
                    for p in (a.lo, a.mid, a.hi):
                        r = find(rows, masker, seed, p)
                        if r:
                            entries.append((f"{name} | s{seed} p={p:+.1f} | {masker}", r["path"].resolve()))
            write_m3u(out / f"{name}.m3u", entries)
            print(f"{name}: {len(entries)} files -> {out / (name + '.m3u')}")
        return

    audio = out / "audio"
    audio.mkdir(exist_ok=True)
    key, entries = [], []

    if a.mode == "pairs":
        trials = []
        for name, rows in grids.items():
            for masker in sorted({r["masker"] for r in rows}):
                for seed in sorted({r["seed"] for r in rows}):
                    lo, hi = find(rows, masker, seed, a.lo), find(rows, masker, seed, a.hi)
                    if lo and hi:
                        trials.append((lo, hi))
        rng.shuffle(trials)
        for i, (lo, hi) in enumerate(trials, 1):
            first, second = (lo, hi) if rng.random() < 0.5 else (hi, lo)
            names = {}
            for tag, r in (("A", first), ("B", second)):
                dst = audio / f"t{i:03d}_{tag}.wav"
                lufs = copy_audio(r["path"], dst, a.lufs)
                names[tag] = (dst.name, r, lufs)
                entries.append((f"trial {i:03d} {tag}", f"audio/{dst.name}"))
            key.append({"trial": i, "grid": lo["grid"], "masker": lo["masker"], "seed": lo["seed"],
                        "A_p": first["p"], "B_p": second["p"],
                        "A_lufs_orig": names["A"][2], "B_lufs_orig": names["B"][2]})
        resp_cols = ["trial", "more_pleasant(A/B)", "more_high_freq(A/B)", "confidence(1-3)", "artifacts(none/A/B/both)", "notes"]
        resp = [{"trial": k["trial"]} for k in key]
    else:  # models
        levels = a.levels if a.levels is not None else [a.hi]
        names_list = list(grids)
        base = grids[names_list[0]]
        trials = []
        for masker in sorted({r["masker"] for r in base}):
            for seed in sorted({r["seed"] for r in base}):
                for p in levels:
                    cands = [(n, find(grids[n], masker, seed, p)) for n in names_list]
                    if all(c[1] for c in cands):
                        trials.append((masker, seed, p, cands))
        rng.shuffle(trials)
        for i, (masker, seed, p, cands) in enumerate(trials, 1):
            rng.shuffle(cands)
            row = {"trial": i, "masker": masker, "seed": seed, "p": p}
            for j, (n, r) in enumerate(cands, 1):
                dst = audio / f"t{i:03d}_X{j}.wav"
                row[f"X{j}_lufs_orig"] = copy_audio(r["path"], dst, a.lufs)
                row[f"X{j}"] = n
                entries.append((f"trial {i:03d} X{j}", f"audio/{dst.name}"))
            key.append(row)
        resp_cols = ["trial", "ranking_most_to_least_pleasant (e.g. X2>X1>X3)", "artifacts(which X)", "notes"]
        resp = [{"trial": k["trial"]} for k in key]

    if not key:
        raise SystemExit("no complete trials found; check --maskers/--seeds/levels")
    write_m3u(out / "playlist.m3u", entries)
    for fname, data, cols in (("key.csv", key, list(key[0])), ("responses.csv", resp, resp_cols)):
        with open(out / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, restval="")
            w.writeheader()
            w.writerows(data)
    print(f"{len(key)} trials, {len(entries)} files -> {out}/playlist.m3u  (key.csv = answers, open it only after rating)")


if __name__ == "__main__":
    main()
