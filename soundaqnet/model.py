"""Load SoundAQnet from a SoundSCaper checkout and run it, without DGL and without Windows tools.

The SoundSCaper code is imported from the checkout, not copied. Only the DGL calls are replaced
(dgl_shim.py) and framework.config is replaced by the constants the model needs, because the
original config.py fails on machines without the authors' drive paths.
"""
import io
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch

from . import dgl_shim

EVENT_LABELS = ['Silence', 'Human sounds', 'Wind', 'Water', 'Natural sounds', 'Traffic', 'Sounds of things',
                'Vehicle', 'Bird', 'Outside, rural or natural', 'Environment and background', 'Speech', 'Music',
                'Noise', 'Animal']
SCENE_LABELS = ['public_square', 'park', 'street_traffic']
PAQ_NAMES = ['pleasant', 'eventful', 'chaotic', 'vibrant', 'uneventful', 'calm', 'annoying', 'monotonous']
DEFAULT_CKPT = "SoundAQnet_ASC96_AEC94_PAQ1027.pth"
SUBDIR = "Inferring_soundscape_clips_for_LLM"


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """The normalisation files are pickles from a third-party repo: allow numpy arrays only."""

    def find_class(self, module, name):
        if module.split(".")[0] == "numpy" and name in {"_reconstruct", "ndarray", "dtype", "scalar", "_frombuffer"}:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def _load_norm(path):
    with open(path, "rb") as f:
        d = _NumpyOnlyUnpickler(io.BytesIO(f.read())).load()
    return np.asarray(d["mean"], dtype=np.float32), np.asarray(d["std"], dtype=np.float32)


class SoundAQnetRunner:
    def __init__(self, soundscaper_root, ckpt=DEFAULT_CKPT, device="cpu"):
        root = Path(soundscaper_root) / SUBDIR
        dgl_shim.install()
        cfg = types.ModuleType("framework.config")
        cfg.mel_bins, cfg.event_labels, cfg.scene_labels = 64, EVENT_LABELS, SCENE_LABELS
        cfg.each_emotion_class_num, cfg.cuda = 1, 0
        sys.modules["framework.config"] = cfg
        sys.path.insert(0, str(root))
        try:
            from framework.models_pytorch import SoundAQnet
        finally:
            sys.path.pop(0)
        self.device = torch.device(device)
        self.model = SoundAQnet(max_node_num=8, node_emb_dim=64, hidden_dim=32, out_dim=64)
        state = torch.load(root / "application" / "system" / "model" / ckpt, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
        self.model.to(self.device).eval()
        norm = root / "application" / "Dataset" / "0_normalization_files"
        self.mel_mean, self.mel_std = _load_norm(norm / "norm_log_mel.pickle")
        self.loud_mean, self.loud_std = _load_norm(norm / "norm_loudness.pickle")
        self.graph = dgl_shim.complete_graph(8, 64, device=self.device)

    @torch.no_grad()
    def predict(self, mel, loud, batch_size=32):
        """mel: (n, 3001, 64) log-mel; loud: (n, 15000, 1) ISO 532-1 loudness in sone.
        Returns a dict of arrays. Both features must be for exactly 30 s."""
        mel = (np.asarray(mel, np.float32) - self.mel_mean) / self.mel_std
        loud = (np.asarray(loud, np.float32) - self.loud_mean) / self.loud_std
        out = {k: [] for k in ["scene", "event", "ISOPleasant", "ISOEventful"] + PAQ_NAMES}
        for i in range(0, len(mel), batch_size):
            m = torch.from_numpy(mel[i:i + batch_size]).to(self.device)
            l = torch.from_numpy(loud[i:i + batch_size]).to(self.device)
            res = self.model(m, l, [self.graph] * len(m))
            scene, event, isop, isoe, *paq = res
            out["scene"].append(torch.softmax(scene, -1).cpu().numpy())
            out["event"].append(torch.sigmoid(event).cpu().numpy())
            out["ISOPleasant"].append(isop.cpu().numpy()[:, 0])
            out["ISOEventful"].append(isoe.cpu().numpy()[:, 0])
            for name, v in zip(PAQ_NAMES, paq):
                out[name].append(v.cpu().numpy()[:, 0])
        return {k: np.concatenate(v) for k, v in out.items()}
