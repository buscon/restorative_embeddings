"""stable-audio-tools custom_metadata_module: caption AND a numeric score from metadata.csv.

Columns of metadata.csv next to the wavs: file, caption, score. Returns {"prompt": caption, "pleasantness": score}.
The key "pleasantness" must match the id of the number conditioner in the model config (generation/make_score_config.py).
"""
import csv
from pathlib import Path

_cache = {}


def _load(folder):
    if folder not in _cache:
        with open(Path(folder) / "metadata.csv", encoding="utf-8") as f:
            _cache[folder] = {r["file"]: (r.get("caption", ""), float(r["score"])) for r in csv.DictReader(f)}
    return _cache[folder]


def get_custom_metadata(info, audio):
    p = Path(info["path"])
    caption, score = _load(str(p.parent))[p.name]
    return {"prompt": caption, "pleasantness": score}
