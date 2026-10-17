#!/usr/bin/env python3
"""Start and end value of train/mse_signal (loss on real audio) for training runs, from Weights & Biases.

    python generation/wandb_loss_table.py                       # the six runs listed in RUNS below
    python generation/wandb_loss_table.py --window 300          # mean over the first / last 300 steps
    python generation/wandb_loss_table.py --upto 5000           # ISD number run: only steps <= 5000 (it ran to about 5.2k)

Needs `pip install wandb pandas` and `wandb login` (or WANDB_API_KEY). The run ID is the name of the folder above
`checkpoints/` on the server (e.g. .../isd_all_whole/57jbv1r6/checkpoints). The entity is taken from --entity, or the
default entity of the logged-in account; every project of that entity is searched for the ID.

Start and end are means over a window of logged steps, because single values are noisy (random noise level per step).
The first report's numbers (0.652 to 0.632, 0.61 to 0.54) were probably made in the W&B plot; the window used there is not
known. Run this for ISD and ARAUS as well and use one method for all rows of the slide.
"""
import argparse

import pandas as pd
import wandb

# label -> (run id, last step to include or None)
RUNS = {
    "ISD number":        ("57jbv1r6", 5000),
    "ARAUS number":      ("l2xivkew", 6000),
    "Clotho number":     ("487z5lmc", 5000),
    "ISD words":         ("dap9tpf4", 6000),
    "ISD number input":  (None,       7000),   # score run: fill in the ID (ls ~/stableaudio/runs_isd_score/isd_score_whole/)
    "ISD + ARAUS":       ("nxf8qc76", 6000),
}
KEY = "train/mse_signal"
STEP = "trainer/global_step"   # optimiser steps, as in the checkpoint names (W&B _step counts log calls)


def find_run(api, entity, run_id):
    for p in api.projects(entity):
        try:
            return api.run(f"{entity}/{p.name}/{run_id}")
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default=None)
    ap.add_argument("--window", type=int, default=200, help="number of steps averaged at start and end")
    ap.add_argument("--upto", type=int, default=None, help="override the last step for all runs")
    ap.add_argument("--score-id", default=None, help="run ID of the score-input run")
    a = ap.parse_args()

    api = wandb.Api(timeout=60)
    entity = a.entity or api.default_entity
    rows = []
    for label, (rid, last) in RUNS.items():
        if label == "ISD number input" and a.score_id:
            rid = a.score_id
        if rid is None:
            rows.append((label, "no run ID given (--score-id)", "", "", ""))
            continue
        run = find_run(api, entity, rid)
        if run is None:
            rows.append((label, f"{rid} not found in {entity}", "", "", ""))
            continue
        last = a.upto or last
        h = run.history(keys=[KEY, STEP], samples=100000, pandas=True)
        h[STEP] = h[STEP].ffill().bfill()   # step column may be logged on neighbouring rows only
        h = h.dropna(subset=[KEY, STEP])
        step_col = STEP
        h = h[h[step_col] <= last].sort_values(step_col)
        first = h[h[step_col] < h[step_col].min() + a.window][KEY].mean()
        end = h[h[step_col] > h[step_col].max() - a.window][KEY].mean()
        rows.append((label, f"{first:.3f}", f"{end:.3f}", f"{100 * (end / first - 1):+.0f}%",
                     f"steps {int(h[step_col].min())}-{int(h[step_col].max())}, {len(h)} points"))
    print(pd.DataFrame(rows, columns=["run", "start", "end", "change", "range"]).to_string(index=False))


if __name__ == "__main__":
    main()
