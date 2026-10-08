"""Numba version of MoSQITo's non-linear temporal decay step (ISO 532-1 time-varying loudness).

In MoSQITo 1.2 the step `_nl_loudness` is a Python loop over about 1.4 million time columns with
about 20 small numpy calls in each, and takes about 40 of the 46 seconds a 30 s clip needs. The
recurrence is independent for every row (loudness band), so it is rewritten here as a compiled loop with
exactly the same conditions. `install()` replaces the function inside MoSQITo; `check()` compares
the result with the original on random data. If numba is missing, install() does nothing and the
original code runs (slowly).
"""
import sys
from math import exp, sqrt

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _loop(uo, u2, ui, B):
    R, N = uo.shape
    for r in range(R):
        for col in range(N):
            p = col - 1 if col > 0 else N - 1  # numpy index -1 at col 0, as in the original
            uo_p = uo[r, p]
            u2_p = u2[r, p]
            u = ui[r, col]
            uo2 = uo_p * B[2] - u2_p * B[3]
            if uo_p > u2_p and uo2 >= u:
                uo[r, col] = uo2
            uo2 = uo_p * B[4]
            if uo_p <= u2_p and uo2 >= u:
                uo[r, col] = uo2
            u2[r, col] = uo[r, col]
            u22 = uo_p * B[0] - u2_p * B[1]
            if u < uo_p and uo_p > u2_p and u22 <= uo[r, col]:
                u2[r, col] = u22
            u2_2 = (u2_p - u) * B[5] + u
            if u >= uo_p and not (abs(u - uo_p) < 1e-5 and uo[r, col] <= u2_p):
                u2[r, col] = u2_2


_loop_jit = njit(cache=True)(_loop) if njit else None


def nl_loudness_fast(core_loudness):
    sample_rate, nl_iter = 2000, 24
    t_short, t_long, t_var = 0.005, 0.015, 0.075
    delta_t = 1 / (sample_rate * nl_iter)
    P = (t_var + t_long) / (t_var * t_short)
    Q = 1 / (t_short * t_var)
    l1 = -P / 2 + sqrt(P * P / 4 - Q)
    l2 = -P / 2 - sqrt(P * P / 4 - Q)
    den = t_var * (l1 - l2)
    e1, e2 = exp(l1 * delta_t), exp(l2 * delta_t)
    B = np.array([
        (e1 - e2) / den,
        ((t_var * l2 + 1) * e1 - (t_var * l1 + 1) * e2) / den,
        ((t_var * l1 + 1) * e1 - (t_var * l2 + 1) * e2) / den,
        (t_var * l1 + 1) * (t_var * l2 + 1) * (e1 - e2) / den,
        exp(-delta_t / t_long),
        exp(-delta_t / t_var),
    ])
    core = np.asarray(core_loudness, dtype=np.float64)
    nl = core.copy()
    delta = np.roll(core, -1, axis=1)
    delta[:, -1] = 0
    delta = (delta - nl) / nl_iter
    ui = np.zeros((core.shape[0], core.shape[1], nl_iter))
    ui[:, :, 0] = core
    for i in range(1, nl_iter):
        ui[:, :, i] = ui[:, :, i - 1] + delta
    ui = np.ascontiguousarray(ui.reshape(core.shape[0], core.shape[1] * nl_iter))
    uo = ui.copy()
    u2 = np.zeros_like(uo)
    mask = core[:, 0] >= 1e-5
    u2[mask, 0] = core[mask, 0] * (1 - B[5])
    _loop_jit(uo, u2, ui, B)
    return uo.reshape(core.shape[0], core.shape[1], nl_iter)[:, :, 0]


def install():
    """Patch MoSQITo. Returns True if the fast version is active."""
    if _loop_jit is None:
        return False
    import mosqito.sq_metrics.loudness.loudness_zwtv.loudness_zwtv  # noqa: F401
    mod = sys.modules["mosqito.sq_metrics.loudness.loudness_zwtv.loudness_zwtv"]
    mod._nl_loudness = nl_loudness_fast
    return True


def check(seed=0, rows=24, cols=3000):
    """Max abs difference to the original MoSQITo function on random 'loudness-like' data."""
    from mosqito.sq_metrics.loudness.loudness_zwtv._nonlinear_decay import _nl_loudness
    rng = np.random.default_rng(seed)
    x = np.abs(rng.normal(size=(rows, cols))) * (rng.random((rows, cols)) > 0.3)
    return float(np.abs(_nl_loudness(x) - nl_loudness_fast(x)).max())
