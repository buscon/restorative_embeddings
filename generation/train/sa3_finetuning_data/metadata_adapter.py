"""stable-audio-tools custom_metadata_module: reads captions from metadata.csv.

Referenced by dataset_config_360.json. For each wav it looks up the caption in the
metadata.csv that sits next to it (columns: file, caption).
"""
import csv
from pathlib import Path

_cache = {}


def _load(folder):
    if folder not in _cache:
        with open(Path(folder) / "metadata.csv", encoding="utf-8") as f:
            _cache[folder] = {r["file"]: r.get("caption", "") for r in csv.DictReader(f)}
    return _cache[folder]


def get_custom_metadata(info, audio):
    p = Path(info["path"])
    return {"prompt": _load(str(p.parent)).get(p.name, "")}
