#!/usr/bin/env python3
"""Does the frozen T5 text encoder of Stable Audio Open turn the rating in "[ISOPleasant: x.xx]" into an ordered signal?

stable-audio-tools encodes the prompt with a FROZEN T5 (t5-base; enable_grad is false), so what the diffusion
model can use is fixed by (1) how the tokenizer splits the number and (2) what the frozen encoder makes of the
tokens. This script looks at both for several ways of writing the rating. It needs only the tokenizer and the
encoder (CPU is enough), no checkpoint and no GPU.

Per format it prints
  - tokens of the number part for some example values and the number of tokens per value
  - Spearman correlation between the distance of two prompt embeddings and the difference of the two ratings
    (1 = embedding distance grows steadily with the rating difference; near 0 = no ordered signal)
  - the distance between neighbouring values (0.01 apart) at 0.00 against the median, and the largest jumps
  - d(-0.5,+0.5), d(-0.5,0), d(0,+0.5): in an ordered code the first is the largest

    python generation/probe_t5_tag.py [--template "soundscape with natural sounds [ISOPleasant: {}]"]
"""
import argparse
import collections

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, T5EncoderModel

FORMATS = {
    "signed_2dec (current)": lambda v: f"{v:.2f}",
    "percent 0-100 (v+1)*50": lambda v: f"{int(round((v + 1) * 50))}",
    "scale 0-10 (v+1)*5": lambda v: f"{int(round((v + 1) * 5))}",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default="soundscape with natural sounds [ISOPleasant: {}]")
    ap.add_argument("--model", default="t5-base")
    ap.add_argument("--lo", type=float, default=-0.8)
    ap.add_argument("--hi", type=float, default=0.8)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    enc = T5EncoderModel.from_pretrained(a.model).eval()
    v = np.array([round(x, 2) for x in np.arange(a.lo, a.hi + 1e-9, 0.01)])
    def number_tokens(text):
        """tokens between the last ':' and the closing ']' of the tag"""
        t = tok.convert_ids_to_tokens(tok(text)["input_ids"])
        k = max(i for i, x in enumerate(t) if x == ":")
        return t[k + 1:-2]

    for name, fmt in FORMATS.items():
        strs = [fmt(x) for x in v]
        toks = {}
        for s in strs:
            toks[s] = number_tokens(a.template.format(s))
        print(f"\n=== {name} ===")
        shown = [x for x in (-0.8, -0.7, -0.4, -0.1, -0.07, 0.0, 0.01, 0.07, 0.1, 0.4, 0.7, 0.8) if a.lo <= x <= a.hi]
        for x in shown:
            s = fmt(x)
            print(f"  v={x:+.2f} -> '{s}' -> {toks[s]}")
        cnt = collections.Counter(len(t) for t in toks.values())
        print("  tokens per value:", dict(sorted(cnt.items())))
        E = []
        with torch.no_grad():
            for s in strs:
                b = tok(a.template.format(s), return_tensors="pt")
                h = enc(**b).last_hidden_state[0]
                names = tok.convert_ids_to_tokens(b["input_ids"][0])
                i0 = names.index("▁[")
                E.append(h[i0:-1].mean(0).numpy())          # mean over the tag tokens
        E = np.array(E)
        D = np.linalg.norm(E[:, None] - E[None], axis=-1)
        dv = np.abs(v[:, None] - v[None])
        iu = np.triu_indices(len(v), 1)
        rho = spearmanr(D[iu], dv[iu]).correlation
        nb = np.array([D[i, i + 1] for i in range(len(v) - 1) if strs[i] != strs[i + 1]])
        med = np.median(nb)
        i0 = int(np.argmin(np.abs(v)))
        j = lambda x: int(np.argmin(np.abs(v - x)))
        around0 = np.mean([D[i0, i0 - 1], D[i0, i0 + 1]])
        jumps = np.argsort(-np.array([D[i, i + 1] for i in range(len(v) - 1)]))[:5]
        print(f"  Spearman(embedding distance, |rating difference|) = {rho:.2f}")
        if strs[i0 - 1] != strs[i0] != strs[i0 + 1]:
            print(f"  neighbour distance (0.01 apart): median {med:.3f}; around 0.00 {around0:.3f} (x{around0 / med:.1f})")
        else:
            print(f"  neighbour distance (0.01 apart, only where the written value changes): median {med:.3f}; 0.00 shares its string with its neighbours")
        print("  largest neighbour jumps:", [(strs[i], strs[i + 1], round(float(D[i, i + 1] / med), 1) if med else None) for i in jumps])
        print(f"  d(-0.5,+0.5)={D[j(-0.5), j(0.5)]:.3f}  d(-0.5,0)={D[j(-0.5), j(0.0)]:.3f}  d(0,+0.5)={D[j(0.0), j(0.5)]:.3f}  d(0.4,0.5)={D[j(0.4), j(0.5)]:.3f}")


if __name__ == "__main__":
    main()
