#!/usr/bin/env python3
"""Copy configs/model_config.json and add a number conditioner "pleasantness" (range -1..1) to the cross-attention.

The score becomes one extra cross-attention token next to the T5 tokens (like seconds_start / seconds_total), so no
pretrained weight changes shape; the new conditioner starts from random weights (not in the pretrained checkpoint).
Demo prompts are replaced by the caption WITHOUT tag plus a pleasantness value.
  python generation/make_score_config.py [--out configs/model_config_score.json] [--min-val -1] [--max-val 1]
         [--demo-item "soundscape with natural sounds"] [--fourier learned|expo]
Train with the FULL fine-tune (not LoRA): in LoRA mode the new embedder would stay frozen at random weights.
"""
import argparse, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", default=str(REPO / "configs/model_config.json"))
ap.add_argument("--out", default=str(REPO / "configs/model_config_score.json"))
ap.add_argument("--id", default="pleasantness")
ap.add_argument("--min-val", type=float, default=-1.0)
ap.add_argument("--max-val", type=float, default=1.0)
ap.add_argument("--fourier", default="learned", choices=["learned", "expo"])
ap.add_argument("--demo-item", default="soundscape with natural sounds")
ap.add_argument("--demo-values", type=float, nargs="+", default=[-0.7, 0.0, 0.7])
a = ap.parse_args()
cfg = json.load(open(a.src, encoding="utf-8"))
conds = cfg["model"]["conditioning"]["configs"]
if any(c["id"] == a.id for c in conds):
    raise SystemExit(f"{a.src} already has a conditioner '{a.id}'")
conf = {"min_val": a.min_val, "max_val": a.max_val}
if a.fourier != "learned":
    conf["fourier_features_type"] = a.fourier
conds.append({"id": a.id, "type": "number", "config": conf})
cfg["model"]["diffusion"]["cross_attention_cond_ids"].append(a.id)
cfg["training"]["demo"]["demo_cond"] = [
    {"prompt": a.demo_item, a.id: v, "seconds_start": 0, "seconds_total": 30} for v in a.demo_values]
cfg["training"]["demo"]["num_demos"] = len(a.demo_values)
json.dump(cfg, open(a.out, "w", encoding="utf-8"), indent=4)
print("wrote", a.out)
print("cross_attention_cond_ids:", cfg["model"]["diffusion"]["cross_attention_cond_ids"])
for d in cfg["training"]["demo"]["demo_cond"]:
    print("  demo:", d)
