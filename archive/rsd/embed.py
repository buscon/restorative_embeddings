"""CLAP audio embeddings (LAION CLAP via Hugging Face transformers)."""

from __future__ import annotations

import os

import numpy as np
import torch

from .audio import TARGET_SR

DEFAULT_MODEL = "laion/clap-htsat-unfused"


class ClapEmbedder:
    """Embeds 10 s windows and averages them per recording.

    Works with transformers 4.x and 5.x: the feature extractor is called
    positionally (the processor keyword changed from `audios` to `audio`), and
    `get_audio_features` may return a tensor (4.x) or an output object (5.x).
    """

    dim = 512

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None, fp16: bool = False):
        from transformers import ClapModel, ClapProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClapModel.from_pretrained(model_name).to(self.device).eval()
        self.fe = ClapProcessor.from_pretrained(model_name).feature_extractor
        self.fp16 = fp16 and self.device.startswith("cuda")

    @torch.no_grad()
    def embed_windows(self, wins: np.ndarray) -> np.ndarray:
        """(n_windows, samples) -> (n_windows, 512)."""
        inp = self.fe(list(wins), sampling_rate=TARGET_SR, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        with torch.autocast("cuda", enabled=self.fp16):
            out = self.model.get_audio_features(**inp)
        emb = out if torch.is_tensor(out) else out.pooler_output
        return emb.float().cpu().numpy()


class FakeEmbedder:
    """Deterministic stand-in (log band energies) for tests without CLAP.
    Enabled with RSD_FAKE_EMBED=1."""

    dim = 64

    def __init__(self, *_, **__):
        pass

    def embed_windows(self, wins: np.ndarray) -> np.ndarray:
        spec = np.abs(np.fft.rfft(wins, axis=1)) ** 2
        bands = np.array_split(spec, self.dim, axis=1)
        return np.log10(np.stack([b.mean(axis=1) for b in bands], axis=1) + 1e-12)


def get_embedder(model_name: str = DEFAULT_MODEL, device: str | None = None, fp16: bool = False):
    if os.environ.get("RSD_FAKE_EMBED") == "1":
        return FakeEmbedder()
    return ClapEmbedder(model_name, device, fp16)
