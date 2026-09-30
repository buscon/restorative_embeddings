"""Step 2 - CLAP embeddings (+ ecoacoustic indices) for ARAUS or ISD.

    python scripts/02_embed.py --dataset araus      # ~25k stimuli, mixed on the fly
    python scripts/02_embed.py --dataset isd        # ~1.4k field recordings
    python scripts/02_embed.py --dataset isd --limit 50   # quick check

Audio loading, mixing and index computation run in DataLoader workers; the
GPU only sees ready 10 s windows. Output:
    data/features/{dataset}_clap.npz     ids, emb (N x 512)
    data/features/{dataset}_indices.csv  id + ecoacoustic indices
Re-running resumes: ids already in the output are skipped.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsd.audio import TARGET_SR, ArausMixer, prepare, read, windows  # noqa: E402
from rsd.embed import DEFAULT_MODEL, get_embedder  # noqa: E402
from rsd.indices import compute_indices  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["araus", "isd"], required=True)
ap.add_argument("--araus", default="data/raw/araus")
ap.add_argument("--processed", default="data/processed")
ap.add_argument("--out", default="data/features")
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--device", default=None)
ap.add_argument("--fp16", action="store_true")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--recordings-per-batch", type=int, default=16)
ap.add_argument("--no-indices", action="store_true")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--flush-every", type=int, default=2000)
a = ap.parse_args()

proc, out = Path(a.processed), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
emb_path, idx_path = out / f"{a.dataset}_clap.npz", out / f"{a.dataset}_indices.csv"

if a.dataset == "araus":
    table = pd.read_csv(proc / "araus_stimuli.csv").sort_values(["soundscape", "masker"])  # cache-friendly order
    ids = table.stimulus_id.tolist()
else:
    table = pd.read_csv(proc / "isd_recordings.csv")
    ids = table.GroupID.tolist()

done_ids, done_emb, done_idx = [], [], []
if emb_path.exists():
    z = np.load(emb_path, allow_pickle=False)
    done_ids, done_emb = list(z["ids"]), [z["emb"]]
    if idx_path.exists():
        done_idx = [pd.read_csv(idx_path)]
    print(f"resuming: {len(done_ids)} already embedded")
todo = table[~pd.Series(ids, index=table.index).isin(set(done_ids))]
if a.limit:
    todo = todo.head(a.limit)


class Items(Dataset):
    def __init__(self, rows):
        self.rows = rows.to_dict("records")
        self.mixer = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        try:
            if a.dataset == "araus":
                if self.mixer is None:  # built lazily inside each worker
                    root = Path(a.araus)
                    self.mixer = ArausMixer(pd.read_csv(root / "data/soundscapes.csv"),
                                            pd.read_csv(root / "data/maskers.csv"),
                                            root / "soundscapes", root / "maskers")
                x, sr = self.mixer.mix(row["soundscape"], row["masker"], row["smr"])
                rid = row["stimulus_id"]
            else:
                x, sr = read(row["wav"])
                rid = row["GroupID"]
            duration = len(x) / sr
            y = prepare(x, sr)
            ind = {} if a.no_indices else {"duration_s": duration, **compute_indices(y, TARGET_SR)}
            return rid, windows(y), ind, None
        except Exception as e:  # keep going; report at the end
            return row.get("stimulus_id", row.get("GroupID")), None, None, repr(e)


def collate(batch):
    return batch


embedder = get_embedder(a.model, a.device, a.fp16)
loader = DataLoader(Items(todo), batch_size=a.recordings_per_batch, num_workers=a.workers,
                    collate_fn=collate, persistent_workers=False)

new_ids, new_emb, new_idx, failures = [], [], [], []
since_flush = 0


def flush():
    all_ids = done_ids + new_ids
    all_emb = np.concatenate(done_emb + ([np.stack(new_emb)] if new_emb else []), axis=0)
    np.savez(emb_path, ids=np.array(all_ids), emb=all_emb.astype(np.float32))
    if not a.no_indices:
        pd.concat(done_idx + [pd.DataFrame(new_idx)], ignore_index=True).to_csv(idx_path, index=False)


for batch in tqdm(loader, desc=f"embedding {a.dataset}"):
    ok = [b for b in batch if b[1] is not None]
    failures += [(b[0], b[3]) for b in batch if b[1] is None]
    if not ok:
        continue
    wins = np.concatenate([b[1] for b in ok])
    per = ok[0][1].shape[0]
    e = embedder.embed_windows(wins).reshape(len(ok), per, -1).mean(axis=1)  # average the 3 windows
    for (rid, _, ind, _), v in zip(ok, e):
        new_ids.append(rid)
        new_emb.append(v)
        if not a.no_indices:
            new_idx.append({"id": rid, **ind})
    since_flush += len(ok)
    if since_flush >= a.flush_every:
        flush()
        since_flush = 0

if new_ids:
    flush()
print(f"embedded {len(new_ids)} new items -> {emb_path}")
if failures:
    print(f"{len(failures)} failures, first few:")
    for f in failures[:10]:
        print("  ", f)
