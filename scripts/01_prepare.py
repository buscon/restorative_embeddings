"""Step 1 - build the ARAUS and ISD tables used by every later step.

    python scripts/01_prepare.py --araus data/raw/araus --isd data/raw/isd
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsd.data import TARGETS, load_araus, load_isd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--araus", default="data/raw/araus")
ap.add_argument("--isd", default="data/raw/isd")
ap.add_argument("--out", default="data/processed")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

ar = load_araus(a.araus)
ar["responses"].to_csv(out / "araus_responses.csv", index=False)
ar["stimuli"].to_csv(out / "araus_stimuli.csv", index=False)
r, s = ar["responses"], ar["stimuli"]
print(f"ARAUS: {len(r)} responses, {r.participant.nunique()} participants, "
      f"{len(s)} unique stimuli ({(s.fold == 0).sum()} in test fold 0), "
      f"{s.soundscape.nunique()} base soundscapes")

isd = load_isd(a.isd)
isd["ratings"].to_csv(out / "isd_ratings.csv", index=False)
isd["recordings"].to_csv(out / "isd_recordings.csv", index=False)
rt, rc = isd["ratings"], isd["recordings"]
rated = rc[rc.n_ratings > 0]
print(f"ISD: {len(rc)} recordings with audio, {len(rated)} of them rated "
      f"({int(rated.n_ratings.sum())} ratings, {rated.LocationID.nunique()} locations); "
      f"{(~rt.has_audio).sum()} ratings have no matching recording")

for t in TARGETS:
    print(f"  {t}: ARAUS mean {r[t].mean():+.3f} sd {r[t].std():.3f} | "
          f"ISD mean {rt[rt.has_audio][t].mean():+.3f} sd {rt[rt.has_audio][t].std():.3f}")
