#!/usr/bin/env python3
"""Count and list soundscape-like Clotho v2.1 clips (captions + metadata only, no audio).

Clotho = Freesound clips of 15-30 s with five crowd-written captions each. The test split
has no public captions, so development + validation + evaluation are used (5929 clips).

A clip is a candidate if
  - at least --min-captions of its five captions contain a soundscape keyword, OR
  - it has a soundscape-like Freesound tag AND at least one caption with a keyword.
With --require-tag the clip must have a soundscape-like Freesound tag AND at least
--min-captions captions with a keyword (stricter; fewer clips, fewer single-event clips).
The keyword lists are a first pass, not tuned; "station", "train" and "bus" were removed
after a first sample showed single events (a train horn, a radio) getting through. Read a
sample of the selected clips (--show) before relying on the selection.

Durations come from start_end_samples in the metadata (44.1 kHz). That field is empty for
about 30% of the clips; for those the mean of the known durations is used in the hour
counts (an estimate).

Licence classes in Clotho: CC0, BY, BY-NC, Sampling+. The default keeps CC0 and BY only
(--licences), so the BY-NC and Sampling+ clips are dropped from the output.

    python clotho/02_select_candidates.py --clotho data/raw/clotho \
        --output-csv clotho/selected_clotho.csv [--licences CC0 BY] [--min-captions 2] [--require-tag] [--show 50]
"""
import argparse
import ast
import sys
from pathlib import Path

import pandas as pd

SPLITS = ["development", "validation", "evaluation"]
CAPS = [f"caption_{i}" for i in range(1, 6)]

KEYWORDS = {
    "water/weather": r"\brain|rainfall|thunder|storm|\bwind|stream|river|waterfall|\bwaves?\b|ocean|\bsea\b|shore|creek|drizzl",
    "animals": r"\bbirds?\b|chirp|tweet|song ?bird|crickets?|frogs?|insects?|cicada|\bowl|rooster",
    "people/urban": r"crowd|people (?:are )?(?:talk|chat|walk)|chatter|street|traffic|\bcars?\b|road|\bcity|market|restaurant|playground|children|construction|crosswalk|sidewalk",
    "places": r"forest|\bpark\b|beach|jungle|\bfield\b|village|outdoors?|farm|\blake\b|\bpond\b|\bwoods?\b",
}
SCAPE_TAGS = r"field-recording|field recording|ambience|ambient|ambiance|soundscape|atmosphere|atmos\b|nature|city|urban|environment"


def licence_class(url: str) -> str:
    if "publicdomain" in url:
        return "CC0"
    if "by-nc" in url:
        return "BY-NC"
    if "/by/" in url:
        return "BY"
    if "sampling" in url:
        return "Sampling+"
    return url


def load(clotho_dir: Path) -> pd.DataFrame:
    parts = []
    for s in SPLITS:
        # the files are not all UTF-8
        c = pd.read_csv(clotho_dir / f"clotho_captions_{s}.csv", encoding="latin-1")
        m = pd.read_csv(clotho_dir / f"clotho_metadata_{s}.csv", encoding="latin-1")
        x = c.merge(m, on="file_name")
        x["split"] = s
        parts.append(x)
    df = pd.concat(parts, ignore_index=True)

    se = df.start_end_samples.apply(lambda v: ast.literal_eval(v) if isinstance(v, str) else None)
    df["dur_known"] = se.notna()
    df["dur"] = se.apply(lambda t: (t[1] - t[0]) / 44100 if t else float("nan"))
    print(f"duration known for {int(df.dur_known.sum())} of {len(df)} clips; "
          f"mean of known {df.dur.mean():.1f} s (used for the rest)")
    df["dur"] = df.dur.fillna(df.dur.mean())

    df["licence_class"] = df.license.apply(licence_class)
    lower = df[CAPS].astype(str).apply(lambda col: col.str.lower())
    for k, p in KEYWORDS.items():
        df["n_" + k] = lower.apply(lambda col: col.str.contains(p, regex=True)).sum(axis=1)
    allp = "|".join(KEYWORDS.values())
    df["n_any"] = lower.apply(lambda col: col.str.contains(allp, regex=True)).sum(axis=1)
    df["scape_tag"] = df.keywords.fillna("").str.lower().str.contains(SCAPE_TAGS, regex=True)
    return df


def report(name: str, df: pd.DataFrame, mask: pd.Series, lic_ok: pd.Series) -> None:
    for lab, mm in [("all licences", mask), ("kept licences", mask & lic_ok)]:
        sub = df[mm]
        print(f"{name:50s} {lab:14s} clips={len(sub):5d}  hours={sub.dur.sum() / 3600:5.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clotho", default="data/raw/clotho", help="folder with the Clotho CSVs")
    ap.add_argument("--output-csv", default="clotho/selected_clotho.csv")
    ap.add_argument("--min-captions", type=int, default=2,
                    help="captions (of 5) that must contain a keyword (default 2)")
    ap.add_argument("--require-tag", action="store_true",
                    help="require a soundscape-like Freesound tag AND >= --min-captions "
                         "captions with a keyword (stricter selection)")
    ap.add_argument("--licences", nargs="+", default=["CC0", "BY"],
                    help="licence classes kept in the 'kept licences' counts and the output "
                         "(CC0 BY BY-NC Sampling+; default CC0 BY)")
    ap.add_argument("--show", type=int, default=8, help="print this many random selected clips")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load(Path(args.clotho))
    lic_ok = df.licence_class.isin(args.licences)

    print(f"\nclips {len(df)}  hours {df.dur.sum() / 3600:.1f}")
    print(df.dur.describe().round(1).to_string())
    print(df.licence_class.value_counts().to_string())
    print(f"kept licences {args.licences}: {int(lic_ok.sum())} clips, "
          f"{df[lic_ok].dur.sum() / 3600:.1f} h\n")

    for k in KEYWORDS:
        report(f"{k}: >=1 caption", df, df["n_" + k] >= 1, lic_ok)
    report("any keyword: >=1 caption", df, df.n_any >= 1, lic_ok)
    report("any keyword: >=2 captions", df, df.n_any >= 2, lic_ok)
    report("any keyword: >=3 captions", df, df.n_any >= 3, lic_ok)
    report("soundscape-like Freesound tag", df, df.scape_tag, lic_ok)
    report("tag AND >=1 caption keyword", df, df.scape_tag & (df.n_any >= 1), lic_ok)
    report("tag AND >=2 caption keywords", df, df.scape_tag & (df.n_any >= 2), lic_ok)

    if args.require_tag:
        sel = df.scape_tag & (df.n_any >= args.min_captions)
        label = f"SELECTED: tag AND >={args.min_captions} captions"
    else:
        sel = (df.n_any >= args.min_captions) | (df.scape_tag & (df.n_any >= 1))
        label = f"SELECTED: >={args.min_captions} captions OR (tag AND >=1)"
    print()
    report(label, df, sel, lic_ok)

    out = df[sel & lic_ok]
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"\nwrote {len(out)} clips ({out.dur.sum() / 3600:.1f} h) to {args.output_csv}")

    if args.show:
        print(f"\n{min(args.show, len(out))} random selected clips (check these by ear / by eye):")
        for _, r in out.sample(min(args.show, len(out)), random_state=args.seed).iterrows():
            print(f"- {r.file_name} [{r.dur:.0f}s, {r.licence_class}] tags={str(r.keywords)[:60]}")
            print(f"    {r.caption_1}")


if __name__ == "__main__":
    sys.exit(main())
