"""CLAP-based sound source recognition and scoring.

Zero-shot source scoring via text embeddings. Used for caption generation and
source descriptors in conditioning, plus post-hoc analysis of source alignment
between ARAUS lab stimuli and ISD field recordings.
"""

import numpy as np
import torch

# ---- Source categories and phrase sets ----------------------------------------

CATEGORIES = {  # ISD item -> phrases (from the questionnaire's own examples)
    "traffic": ("ssi01", ["cars driving past", "buses and trucks in street traffic", "a train passing",
                          "an airplane flying overhead", "road traffic noise"]),
    "other_noise": ("ssi02", ["a siren", "construction work", "industrial machinery",
                              "goods being loaded and unloaded", "a jackhammer"]),
    "human": ("ssi03", ["people having a conversation", "people laughing", "children playing",
                        "footsteps", "a crowd of people talking"]),
    "natural": ("ssi04", ["birds singing", "flowing water", "wind in the trees and leaves",
                          "a water fountain", "birdsong in a park"]),
}

ALT_PHRASES = {
    "alternative": {
        "traffic": ["a busy road with cars", "a motorcycle accelerating", "a tram passing by",
                    "heavy traffic on a highway", "an idling bus engine"],
        "other_noise": ["an ambulance siren", "a drill", "machines at a building site",
                        "a lorry being unloaded", "a generator humming"],
        "human": ["a group of people chatting", "a child shouting", "people walking on the pavement",
                  "voices in a busy square", "someone laughing loudly"],
        "natural": ["a bird chirping", "a stream of water", "leaves rustling in the wind",
                    "a fountain splashing", "birds in a garden"],
    },
    "labels": {
        "traffic": ["traffic noise"],
        "other_noise": ["other noise such as sirens, construction and industry"],
        "human": ["sounds from human beings"],
        "natural": ["natural sounds"],
    },
}

# ---- Text embedding --------------------------------------------------------

@torch.no_grad()
def text_embeddings(model, processor, phrase_sets=None):
    """Compute text embeddings for source categories.

    Args:
        model: ClapModel instance (expects .eval() already called)
        processor: ClapProcessor instance
        phrase_sets: dict of {category: [phrases]}. If None, uses CATEGORIES.

    Returns:
        dict mapping category to (n_phrases, 512) normalized embeddings
    """
    tok = processor.tokenizer
    res = {}
    phrase_sets = phrase_sets or {c: ph for c, (_, ph) in CATEGORIES.items()}
    for cat, phrases in phrase_sets.items():
        inp = tok([f"the sound of {p}" for p in phrases], padding=True, return_tensors="pt")
        o = model.get_text_features(**inp)
        e = (o if torch.is_tensor(o) else o.pooler_output).numpy()
        res[cat] = e / np.linalg.norm(e, axis=1, keepdims=True)
    return res


def scores(emb, txt):
    """Compute raw and relative source scores for audio embeddings.

    Args:
        emb: (n_audio, 512) audio embedding array
        txt: dict of {category: (n_phrases, 512)} text embeddings

    Returns:
        DataFrame with raw scores (traffic, other_noise, human, natural)
        and relative scores (subtract mean, suffix _rel).
    """
    import pandas as pd
    raw = pd.DataFrame({c: (emb @ t.T).mean(axis=1) for c, t in txt.items()})
    rel = raw.sub(raw.mean(axis=1), axis=0).add_suffix("_rel")
    return pd.concat([raw, rel], axis=1)
