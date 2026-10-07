#!/usr/bin/env python3
"""Unpack only the selected Clotho clips from the 7z archives.

The archives are solid 7z files, so py7zr still decompresses the whole stream, but only the
selected files are written to disk (a few GB instead of about 7 GB of wavs). Needs
`pip install py7zr`.

    python clotho/04_extract_selected_audio.py --selected clotho/selected_clotho.csv \
        --archives data/raw/clotho/archives --out data/raw/clotho/audio_selected

The csv needs the columns file_name and split (written by 02_select_candidates.py). Files
land flat in --out under their Clotho file names. Re-running skips files already there.
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected", default="clotho/selected_clotho.csv")
    ap.add_argument("--archives", default="data/raw/clotho/archives")
    ap.add_argument("--out", default="data/raw/clotho/audio_selected")
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips per split (for tests)")
    args = ap.parse_args()

    try:
        import py7zr
    except ImportError:
        sys.exit("py7zr is missing: pip install py7zr")

    df = pd.read_csv(args.selected)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    missing_total = []
    for split, g in df.groupby("split"):
        wanted = [f for f in g.file_name if not (out / f).exists()]
        if args.limit:
            wanted = wanted[: args.limit]
        print(f"{split}: {len(g)} selected, {len(wanted)} still to extract")
        if not wanted:
            continue
        arch = Path(args.archives) / f"clotho_audio_{split}.7z"
        if not arch.exists():
            print(f"  archive not found: {arch} (run clotho/03_download_audio.sh {split})")
            missing_total += wanted
            continue
        with py7zr.SevenZipFile(arch, "r") as z:
            by_base = {Path(n).name: n for n in z.getnames() if n.lower().endswith(".wav")}
            targets = [by_base[f] for f in wanted if f in by_base]
            not_found = [f for f in wanted if f not in by_base]
            z.reset()
            with tempfile.TemporaryDirectory(dir=out) as tmp:
                z.extract(path=tmp, targets=targets)
                for n in targets:
                    shutil.move(str(Path(tmp) / n), out / Path(n).name)
        print(f"  extracted {len(targets)}; not found in archive: {len(not_found)}")
        missing_total += not_found
    if missing_total:
        print(f"{len(missing_total)} clips missing, e.g. {missing_total[:3]}")
    print(f"{len(list(out.glob('*.wav')))} wavs in {out}")


if __name__ == "__main__":
    main()
