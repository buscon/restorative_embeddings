#!/usr/bin/env python3
"""Copy configs/model_config.json and add a LoRA block (stable-audio-tools, docs/lora.md) for a LoRA fine-tune.

Everything else (conditioning, demo prompts, sample size) is kept, so check the demo prompts first: they
must match the caption style of the dataset you train on (numbers or words).
The optimizer of this repo (AdamW, InverseLR) is taken from the config; only its lr is changed (--lr).

  python generation/make_lora_config.py --out configs/model_config_lora.json [--rank 16] [--alpha 16]
         [--adapter lora|dora|lora-xs] [--lr 1e-4] [--exclude conditioners.]
Default --exclude "conditioners." keeps all adapters in the diffusion transformer (DiT): in the stable-audio-tools
code, LoRA is also added to every Linear layer of the conditioner, including T5. T5 is frozen (no gradient),
so those adapters would stay at their zero start anyway; excluding them makes this explicit. Use --exclude ""
to switch the exclusion off.
"""
import argparse, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", default=str(REPO / "configs/model_config.json"))
ap.add_argument("--out", default=str(REPO / "configs/model_config_lora.json"))
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=int, default=None, help="default = rank")
ap.add_argument("--adapter", default="lora", choices=["lora", "dora", "lora-xs"])
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--exclude", nargs="*", default=["conditioners."])
a = ap.parse_args()
cfg = json.load(open(a.src, encoding="utf-8"))
lc = {"rank": a.rank, "alpha": a.alpha or a.rank, "adapter_type": a.adapter}
ex = [x for x in a.exclude if x]
if ex:
    lc["exclude"] = ex
cfg["training"]["lora_config"] = lc
cfg["training"]["optimizer_configs"]["diffusion"]["optimizer"]["config"]["lr"] = a.lr
json.dump(cfg, open(a.out, "w", encoding="utf-8"), indent=4)
print("wrote", a.out, "| lora_config:", lc, "| lr:", a.lr)
print("demo prompts:")
for d in cfg["training"].get("demo", {}).get("demo_cond", []):
    print("  ", d.get("prompt"))
