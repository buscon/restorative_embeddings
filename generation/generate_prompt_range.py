#!/usr/bin/env python3
"""One checkpoint, several different prompts, a range of pleasantness numbers.

For every (prompt x seed x level) one file is written to <out>/. Prompts are
"<item> [ISOPleasant: <p>]", the training caption format without a city. The default items
(PROMPTS below) are 5 combinations of the source words used in the ISD training captions
("in_vocabulary") and 2 free-text scenes the model never saw as caption text ("free_text").
Replace them with --prompts-file (one item per line; optional "id|group|item").

Writes
  <out>/s<seed>_<item>_p<level>.wav   same naming as generate_grid.py
  <out>/grid.csv         one row per file: seed, masker (= item), p, file, prompt, plus prompt_id,
                         group, ckpt, step, seconds, diffusion_steps, guidance, sampler, use_ema,
                         negative_prompt, sample_rate, created. evaluate_grid.py reads this file.
  <out>/prompts.csv      file,prompt
Rows are written after every file, so an interrupted run keeps its list; existing files are skipped.

    python generation/generate_prompt_range.py --ckpt <model.ckpt> --config configs/model_config.json \
        --out ~/grids/isd_all_6000_range [--seeds 1 2 3] [--evaluate]
    python generation/generate_prompt_range.py --dry-run --out /tmp/x --ckpt x --config x   # list files only
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from rsd.captions import format_pleasantness  # noqa: E402

# (id, group, item). "in_vocabulary": words used in the ISD training captions
# (traffic noise / other noise / human sounds / natural sounds / no dominant sound source).
# "free_text": scenes that never appeared as caption text; they test whether the number still acts
# when the text is outside the training vocabulary (expect weaker or odder results).
PROMPTS = [
    ("traffic_human",    "in_vocabulary", "soundscape with traffic noise and human sounds"),
    ("natural_human",    "in_vocabulary", "soundscape with natural sounds and human sounds"),
    ("natural_traffic",  "in_vocabulary", "soundscape with natural sounds and traffic noise"),
    ("other_noise",      "in_vocabulary", "soundscape with other noise"),
    ("no_dominant",      "in_vocabulary", "soundscape with no dominant sound source"),
    ("birds_wind",       "free_text",     "soundscape with birdsong and wind in the trees"),
    ("waves_beach",      "free_text",     "soundscape with waves on a beach"),
]
FIELDS = ["seed", "masker", "p", "file", "prompt", "prompt_id", "group", "ckpt", "step", "seconds",
          "diffusion_steps", "guidance", "sampler", "use_ema", "negative_prompt", "sample_rate", "created"]


def read_prompts(path):
    out = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [x.strip() for x in line.split("|")]
        out.append(tuple(parts) if len(parts) == 3 else (f"prompt{i:02d}", "custom", parts[-1]))
    return out


def main() -> bool:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--levels", type=float, nargs="+", default=None,
                    help="default -0.7 -0.4 -0.1 0.1 0.4 0.7 (skips 0.0, which behaved oddly in earlier grids); "
                         "with --labels: -0.7 -0.35 0 0.35 0.7")
    ap.add_argument("--labels", action="store_true",
                    help="write the pleasantness as one of five words instead of a number (models trained with "
                         "isd/08_make_words_set.py). The numeric level only names the file and the table row; "
                         "the word is the nearest of -0.7 very unpleasant, -0.35 unpleasant, 0 neutral, "
                         "0.35 pleasant, 0.7 very pleasant")
    ap.add_argument("--prompts-file", default=None)
    ap.add_argument("--template", default="{item} [ISOPleasant: {p}]")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--sampler", default="dpmpp-3m-sde")
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--lora-ckpt", default=None,
                    help="LoRA checkpoint (.ckpt or .safetensors) from a stable-audio-tools LoRA run; --ckpt is then "
                         "the BASE model (model.safetensors). EMA does not exist for LoRA runs.")
    ap.add_argument("--negative-prompt", default=None)
    ap.add_argument("--evaluate", action="store_true", help="run evaluate_grid.py on the folder at the end")
    ap.add_argument("--dry-run", action="store_true", help="write grid.csv and print the file list; no model")
    args = ap.parse_args()

    if args.levels is None:
        args.levels = [-0.7, -0.35, 0.0, 0.35, 0.7] if args.labels else [-0.7, -0.4, -0.1, 0.1, 0.4, 0.7]
    centers = [-0.7, -0.35, 0.0, 0.35, 0.7]
    words = ["very unpleasant", "unpleasant", "neutral", "pleasant", "very pleasant"]

    def tag(p):
        if args.labels:
            return words[min(range(5), key=lambda i: abs(p - centers[i]))]
        return format_pleasantness(p)

    prompts = read_prompts(args.prompts_file) if args.prompts_file else PROMPTS
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    m = re.search(r"step=(\d+)", Path(args.ckpt).name)
    step = int(m.group(1)) if m else ""
    jobs = [(s, pid, grp, item, p) for (pid, grp, item) in prompts for s in args.seeds for p in args.levels]

    rows = {}
    grid_path, prompts_path = out / "grid.csv", out / "prompts.csv"
    if grid_path.exists():
        with open(grid_path, newline="", encoding="utf-8") as f:
            rows = {r["file"]: r for r in csv.DictReader(f)}

    def save():
        with open(grid_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows[k] for k in sorted(rows))
        with open(prompts_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "prompt"])
            w.writerows((k, rows[k]["prompt"]) for k in sorted(rows))

    def record(seed, pid, grp, item, p, sr):
        name = f"s{seed}_{item.replace(' ', '_')}_p{p:.2f}.wav"
        prompt = args.template.format(item=item, p=tag(p))
        rows[name] = {"seed": seed, "masker": item, "p": p, "file": name, "prompt": prompt,
                      "prompt_id": pid, "group": grp, "ckpt": str(Path(args.ckpt).expanduser()),
                      "step": step, "seconds": args.seconds, "diffusion_steps": args.steps,
                      "guidance": args.guidance, "sampler": args.sampler, "use_ema": args.use_ema,
                      "negative_prompt": args.negative_prompt or "", "sample_rate": sr,
                      "created": datetime.now().isoformat(timespec="seconds")}
        return name, prompt

    if args.dry_run:
        for seed, pid, grp, item, p in jobs:
            name, prompt = record(seed, pid, grp, item, p, "")
            print(f"{name}  |  {prompt}")
        save()
        print(f"dry run: {len(jobs)} files listed, {grid_path} written")
        return True

    import torch
    import torchaudio
    from einops import rearrange
    from generate import load_weights
    from stable_audio_tools.models import create_model_from_config
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    with open(args.config) as f:
        cfg = json.load(f)
    sr, size = cfg["sample_rate"], cfg["sample_size"]
    model = create_model_from_config(cfg)
    if not load_weights(model, Path(args.ckpt).expanduser(), use_ema=args.use_ema):
        return False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval().requires_grad_(False)
    if args.lora_ckpt:
        from stable_audio_tools.models.lora import load_and_apply_loras
        load_and_apply_loras(model, [str(Path(args.lora_ckpt).expanduser())], cfg["model_type"])

    neg = None
    if args.negative_prompt:
        neg = [{"prompt": args.negative_prompt, "seconds_start": 0.0, "seconds_total": args.seconds}]
    for k, (seed, pid, grp, item, p) in enumerate(jobs, 1):
        name, prompt = record(seed, pid, grp, item, p, sr)
        path = out / name
        if path.exists():
            print(f"[{k}/{len(jobs)}] exists: {name}", flush=True)
            continue
        print(f"[{k}/{len(jobs)}] {name}  |  {prompt}", flush=True)
        with torch.no_grad():
            audio = generate_diffusion_cond(
                model, steps=args.steps, cfg_scale=args.guidance,
                conditioning=[{"prompt": prompt, "seconds_start": 0.0, "seconds_total": args.seconds}],
                negative_conditioning=neg, sample_size=size, sampler_type=args.sampler,
                device=device, seed=seed)
        audio = rearrange(audio, "b d n -> d (b n)")[:, : int(args.seconds * sr)]
        audio = (audio / audio.abs().max().clamp(min=1e-8)).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        torchaudio.save(str(path), audio, sr)
        save()   # after every file, so an interrupted run keeps its list
    save()
    print(f"done: {len(jobs)} files in {out}  (table: {grid_path})")

    if args.evaluate:
        res = subprocess.run([sys.executable, str(HERE / "evaluate_grid.py"), str(out)],
                             capture_output=True, text=True)
        (out / "evaluation.txt").write_text(res.stdout + res.stderr, encoding="utf-8")
        print(res.stdout + res.stderr)
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
