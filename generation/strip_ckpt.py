#!/usr/bin/env python3
"""Make a weights-only copy of a stable-audio-tools training checkpoint (about half the size or less).

A training .ckpt (~17 GB) holds the model, an EMA copy of the model AND the AdamW optimiser state. For generating and
evaluating (generate_prompt_range.py, generate.py, heldout_loss.py) only the weights are needed. The copy keeps
{"state_dict": ...} so generate.load_weights() reads it exactly like the original. It cannot be used to RESUME a run
(no optimiser state).

    python generation/strip_ckpt.py CKPT [CKPT ...]            # keeps regular + EMA weights  -> <name>.weights.ckpt
    python generation/strip_ckpt.py CKPT --no-ema              # also drops the EMA copy (grids so far did not use --use-ema)
    python generation/strip_ckpt.py CKPT --fp16                # stores float32 tensors as float16 (loses some precision)

The original is never modified or deleted. After writing, the copy is read back and its keys and tensor shapes are
compared with the original; "OK" is printed only when they match. Delete the original yourself afterwards, if you want.
"""
import argparse
import os
import sys
from pathlib import Path

import torch


def strip(path: Path, no_ema: bool, fp16: bool) -> bool:
    out = path.with_name(path.stem + ".weights.ckpt")
    if out.exists():
        print(f"skip {path.name}: {out.name} exists")
        return True
    print(f"loading {path} ({path.stat().st_size / 1e9:.1f} GB) ...", flush=True)
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    raw = ck["state_dict"] if "state_dict" in ck else ck
    sd = {}
    for k, v in raw.items():
        if no_ema and k.startswith("diffusion_ema."):
            continue
        if fp16 and torch.is_tensor(v) and v.dtype == torch.float32:
            v = v.half()
        sd[k] = v
    del ck
    tmp = out.with_suffix(".tmp")
    torch.save({"state_dict": sd}, str(tmp))
    # verify: read back and compare keys and shapes with what we meant to write
    back = torch.load(str(tmp), map_location="cpu", weights_only=False)["state_dict"]
    ok = set(back) == set(sd) and all(tuple(back[k].shape) == tuple(sd[k].shape) for k in sd if torch.is_tensor(sd[k]))
    if not ok:
        print(f"FAILED verification for {path.name}; {tmp.name} left in place, original untouched")
        return False
    os.replace(tmp, out)
    print(f"OK {out.name}: {out.stat().st_size / 1e9:.1f} GB ({len(sd)} keys, "
          f"{'no EMA' if no_ema else 'with EMA'}{', fp16' if fp16 else ''})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()
    ok = all([strip(Path(c).expanduser(), a.no_ema, a.fp16) for c in a.ckpts])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
