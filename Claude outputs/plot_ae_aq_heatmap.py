#!/usr/bin/env python3
"""Heatmap of audio events x affective qualities (SoundSCaper paper, Fig. 6) for generated audio.

Cells are Spearman rho. Rows: the 15 event classes, columns: the 8 SoundAQnet attributes (ISO 12913 PAQ).
Panels
  (a) predicted events vs predicted attributes of the generated clips (both from SoundAQnet; Fig. 6a style)
  (b) intended content (label of the prompt, 0/1) vs predicted attributes (closest to Fig. 6b, which uses labels)
  (c) optional reference: predicted events vs predicted attributes of real recordings rated with the same settings
Stars: adjusted p (default Benjamini-Hochberg over the 120 cells), * <0.05, ** <0.01, *** <0.001. With --unit clip the clips of
one prompt are treated as independent, which makes p too small; --unit prompt averages the seeds of a prompt first.
Also written: the matrices (csv) and similarity between the panels (rank correlation of the 120 cells, sign agreement, with a
bootstrap over prompts for the generated clips, over clips for the reference).

    python generation/plot_ae_aq_heatmap.py --grid ~/grids_3msde/base_ae_g12 --labels generation/ae_label_broad.csv \
        [--ref ~/grids_3msde/araus_ref/soundaqnet_full.csv] --out ~/plots_base_ae/base_ae
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, t as tdist

EVENTS = ['Silence', 'Human sounds', 'Wind', 'Water', 'Natural sounds', 'Traffic', 'Sounds of things', 'Vehicle', 'Bird',
          'Outside, rural or natural', 'Environment and background', 'Speech', 'Music', 'Noise', 'Animal']
PAQ = ['pleasant', 'eventful', 'chaotic', 'vibrant', 'uneventful', 'calm', 'annoying', 'monotonous']


def rho_matrix(X, Y):
    """Spearman rho between every column of X (n x a) and Y (n x b); nan for constant columns."""
    RX = np.apply_along_axis(rankdata, 0, X)
    RY = np.apply_along_axis(rankdata, 0, Y)
    RX = RX - RX.mean(0)
    RY = RY - RY.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (RX.T @ RY) / np.outer(np.sqrt((RX ** 2).sum(0)), np.sqrt((RY ** 2).sum(0)))


def p_matrix(r, n):
    with np.errstate(invalid="ignore", divide="ignore"):
        tt = r * np.sqrt((n - 2) / (1 - r ** 2))
    return 2 * tdist.sf(np.abs(tt), n - 2)


def adjust(p, method):
    if method == "none":
        return p
    flat = p.ravel()
    ok = np.isfinite(flat)
    q = np.full_like(flat, np.nan)
    v = flat[ok]
    m = len(v)
    if method == "bonferroni":
        q[ok] = np.minimum(v * m, 1)
    else:  # Benjamini-Hochberg
        order = np.argsort(v)
        adj = v[order] * m / (np.arange(m) + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        out = np.empty(m)
        out[order] = np.minimum(adj, 1)
        q[ok] = out
    return q.reshape(p.shape)


def stars(q):
    return "" if not np.isfinite(q) else "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def similarity(A, B):
    m = np.isfinite(A) & np.isfinite(B)
    if m.sum() < 5:
        return np.nan, np.nan
    rho = spearmanr(A[m], B[m])[0]
    sign = float((np.sign(A[m]) == np.sign(B[m])).mean())
    return rho, sign


def main():
    global EVENTS, PAQ
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True, help="folder with grid.csv and soundaqnet_full.csv")
    ap.add_argument("--labels", required=True, help="ae_label_primary.csv or ae_label_broad.csv")
    ap.add_argument("--ref", default=None, help="soundaqnet_full.csv of a reference set of real recordings")
    ap.add_argument("--ref-name", default="reference")
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--extra-iso", action="store_true",
                    help="add the columns ISOPleasant and ISOEventful to the eight attributes (SoundSCaper README figures 7 and 8)")
    ap.add_argument("--event-order", default=None, help="semicolon-separated event names, top to bottom (all 15; one name contains commas). Default: the model's order")
    ap.add_argument("--aq-order", default=None, help="comma-separated attribute names, left to right (all 8). Default: the model's order")
    ap.add_argument("--fig7", action="store_true",
                    help="extra figure: the attributes ISOPleasant and ISOEventful (rows) against the 15 events (columns); needs --extra-iso")
    ap.add_argument("--fig8", default=None, metavar="EVENT1;EVENT2",
                    help="extra figure: the named events (rows) against the 8 attributes (columns)")
    ap.add_argument("--unit", choices=["clip", "prompt"], default="clip")
    ap.add_argument("--correction", choices=["none", "bh", "bonferroni"], default="bh")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if a.event_order:
        EVENTS = [x.strip() for x in a.event_order.split(";")]
        assert sorted(EVENTS) == sorted(['Silence', 'Human sounds', 'Wind', 'Water', 'Natural sounds', 'Traffic', 'Sounds of things',
                                         'Vehicle', 'Bird', 'Outside, rural or natural', 'Environment and background', 'Speech',
                                         'Music', 'Noise', 'Animal']), "event order must list all 15 events once"
    if a.aq_order:
        PAQ = [x.strip() for x in a.aq_order.split(",")]
        assert sorted(PAQ) == sorted(['pleasant', 'eventful', 'chaotic', 'vibrant', 'uneventful', 'calm', 'annoying', 'monotonous'])

    grid = Path(a.grid).expanduser()
    g = pd.read_csv(grid / "grid.csv")[["file", "prompt_id", "seed"]]
    s = pd.read_csv(grid / "soundaqnet_full.csv")
    lab = pd.read_csv(a.labels)
    d = g.merge(s, on="file").merge(lab, on="prompt_id")
    print(f"{len(d)} generated clips, {d.prompt_id.nunique()} prompts")
    ev, pq, lb = [f"ev_{e}" for e in EVENTS], [f"paq_{p}" for p in PAQ], [f"lab_{e}" for e in EVENTS]
    names = list(PAQ)
    if a.extra_iso:
        pq, names = pq + ["ISOPleasant", "ISOEventful"], names + ["ISOPleasant", "ISOEventful"]
    if a.unit == "prompt":
        d = d.groupby("prompt_id", as_index=False)[ev + pq + lb].mean()
    sets = {"a": ("predicted events, generated", ev, d), "b": ("intended content (prompt), generated", lb, d)}
    if a.ref:
        r = pd.read_csv(Path(a.ref).expanduser())
        print(f"{len(r)} reference recordings")
        sets["c"] = (f"predicted events, {a.ref_name}", ev, r)

    res = {}
    for k, (title, xc, df) in sets.items():
        R = rho_matrix(df[xc].to_numpy(float), df[pq].to_numpy(float))
        Q = adjust(p_matrix(R, len(df)), a.correction)
        res[k] = (title, R, Q, len(df))
        pd.DataFrame(R, index=EVENTS, columns=names).to_csv(f"{a.out}_rho_{k}.csv")

    if "c" in res:   # difference generated minus reference, in rho units (cells with a star in neither panel are faint)
        Rd = res["a"][1] - res["c"][1]
        res["d"] = (f"difference (a) minus (c), in rho", Rd, np.full_like(Rd, np.nan), 0)
        pd.DataFrame(Rd, index=EVENTS, columns=names).to_csv(f"{a.out}_rho_diff_a_minus_c.csv")
    n = len(res)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n + 1.4, 7.4), squeeze=False)
    for ax, (k, (title, R, Q, nn)) in zip(axes[0], res.items()):
        im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for i in range(len(EVENTS)):
            for j in range(len(names)):
                if np.isfinite(R[i, j]):
                    ax.text(j, i, f"{R[i, j]:.2f}{stars(Q[i, j])}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(R[i, j]) > 0.6 else "black")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(EVENTS)), EVENTS if k == "a" else [""] * len(EVENTS), fontsize=8)
        ax.set_title(f"({k}) {title}" + (f"\nn = {nn} ({a.unit}s)" if nn else ""), fontsize=9)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.7, label="Spearman rho")
    fig.text(0.01, 0.01, f"stars: adjusted p ({a.correction}) * <0.05 ** <0.01 *** <0.001; empty cell = constant column",
             fontsize=7)
    fig.savefig(f"{a.out}_heatmap.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{a.out}_heatmap.pdf", bbox_inches="tight")

    def subfigure(rows, cols, fname, transpose):
        """Sub-matrix of the panels: rows/cols are names; transpose puts the attributes on the vertical axis."""
        keys = [k for k in ("a", "b", "c") if k in res]
        r_idx, c_idx = [EVENTS.index(e) for e in rows], [names.index(c) for c in cols]
        fig, axs = plt.subplots(len(keys), 1, figsize=(1.0 * (len(rows) if transpose else len(cols)) + 3.5, 2.3 * len(keys) + 1.2),
                                squeeze=False, layout="constrained")
        for ax, k in zip(axs[:, 0], keys):
            title, R, Q, nn = res[k]
            M, QQ = R[np.ix_(r_idx, c_idx)], Q[np.ix_(r_idx, c_idx)]
            if transpose:
                M, QQ = M.T, QQ.T
            im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.2f}{stars(QQ[i, j])}", ha="center", va="center", fontsize=7,
                                color="white" if abs(M[i, j]) > 0.6 else "black")
            xl, yl = (rows, cols) if transpose else (cols, rows)
            ax.set_xticks(range(len(xl)), xl, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(yl)), yl, fontsize=8)
            ax.set_title(f"({k}) {title}, n = {nn}", fontsize=9)
        fig.colorbar(im, ax=axs[:, 0].tolist(), shrink=0.8, label="Spearman rho")
        fig.savefig(fname, dpi=200, bbox_inches="tight")

    if a.fig7:
        if a.extra_iso:
            subfigure(EVENTS, ["ISOPleasant", "ISOEventful"], f"{a.out}_fig7_AQ_vs_AE.png", transpose=True)
        else:
            print("--fig7 needs --extra-iso; skipped")
    if a.fig8:
        subfigure([e.strip() for e in a.fig8.split(";")], PAQ, f"{a.out}_fig8_AE_vs_AQ.png", transpose=False)

    # similarity between panels, bootstrap over prompts (generated) and over clips (reference)
    rng = np.random.default_rng(a.seed)
    pairs = [("a", "b")] + ([("a", "c"), ("b", "c")] if a.ref else [])
    lines = [f"generated clips: {len(d)}, prompts: {g.prompt_id.nunique()}, unit: {a.unit}, correction: {a.correction}"]
    pid = d.prompt_id.unique() if a.unit == "clip" else None
    groups = {p: np.where(d.prompt_id.to_numpy() == p)[0] for p in (pid if pid is not None else d.prompt_id)}
    keys = list(groups)
    for x, y in pairs:
        rho0, sg0 = similarity(res[x][1], res[y][1])
        boots = []
        for _ in range(a.boot):
            idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
            Rb = {}
            for kk in (x, y):
                df = sets[kk][2]
                if kk == "c":
                    df = df.iloc[rng.integers(0, len(df), len(df))]
                else:
                    df = d.iloc[idx]
                Rb[kk] = rho_matrix(df[sets[kk][1]].to_numpy(float), df[pq].to_numpy(float))
            boots.append(similarity(Rb[x], Rb[y]))
        b = np.array(boots, float)
        lo, hi = np.nanpercentile(b[:, 0], [2.5, 97.5])
        slo, shi = np.nanpercentile(b[:, 1], [2.5, 97.5])
        lines.append(f"({x}) vs ({y}): rank correlation of the cells {rho0:+.2f} (95% bootstrap {lo:+.2f} to {hi:+.2f}); "
                     f"same sign in {100 * sg0:.0f}% of the cells ({100 * slo:.0f} to {100 * shi:.0f}%)")
    txt = "\n".join(lines)
    Path(f"{a.out}_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"wrote {a.out}_heatmap.png/.pdf, _rho_*.csv, _summary.txt")


if __name__ == "__main__":
    main()
