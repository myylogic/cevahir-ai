# -*- coding: utf-8 -*-
"""Statistics and decision rules of the KA study (PROTOCOL §6-§7). Pure functions, no I/O, no global RNG.

Unit = one world. Tests are exact sign-flip permutation tests over all 2^n sign patterns (CEV-11's frozen
``cev11.stats.signflip_pvalue``), Holm step-down within a family (``cev11.stats.holm``), Student-t intervals
(``cev11.stats.t_interval``).

* :func:`frontier` — the frontier statistic F*: the best FUSED transfer of any mixture of an address's
  configurations under the locality and efficacy constraints (a small linear programme solved exactly by
  enumerating basic feasible solutions).
* :func:`label_contrast` — one-sided contrast with a SESOI: supported / below SESOI / refuted / inconclusive.
* :func:`two_sided` — two-sided contrast with a SESOI: each direction is a margin-shifted one-sided test,
  the family p-value is 2 min(p+, p-).
* :func:`threshold_claim` — one-sided t-test of a mean against a fixed threshold.
* :func:`upper_bound_pass_probability` — the chance that a guard of the form "upper 95% bound <= limit" passes
  for a given true mean and across-world spread (design diagnostic of the KA-H1 locality guard).
* :func:`hier_bootstrap` — hierarchical bootstrap (worlds, then facts or edits) for descriptive intervals.
"""
from __future__ import annotations

import itertools
import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import KA_VERSION  # noqa: F401
from cev11.stats import holm, holm_adjusted, signflip_pvalue, student_t_ppf, student_t_sf, t_interval

ALPHA = 0.05
MIN_WORLDS = 6
BOOTSTRAP_N = 10_000          # hierarchical bootstrap resamples (descriptive intervals only)
BOOTSTRAP_SEED = 0
SUPPORTED, REFUTED, INCONCLUSIVE = "supported", "refuted", "inconclusive"
BELOW_SESOI = "below SESOI"
INSUFFICIENT = "inconclusive (insufficient data)"
NOT_TESTED = "not tested"
EQUIVALENT = "equivalent"

# Frontier constraints (PROTOCOL §3.4): X = Leak_RN - Leak_RD, R = Leak_RD, E = efficacy on the lived frame.
X_MAX, R_MAX, E_MIN, C_MAX = 0.10, 0.01, 0.80, 0.05


# ---------------------------------------------------------------------------------------------------- frontier LP
def frontier(cells: Sequence[Mapping[str, float]], *, x_max: float = X_MAX, r_max: float = R_MAX,
             e_min: float = E_MIN, c_max: Optional[float] = None) -> Tuple[float, List[float]]:
    """F* = max_w sum_k w_k Y_k  s.t.  w >= 0, sum w <= 1, sum w X <= x_max, sum w R <= r_max, sum w E >= e_min
    (and, with ``c_max``, sum w C <= c_max). Returns ``(F*, w)``; infeasible -> ``(0.0, [0, ...])``.

    Exact: the optimum of a bounded feasible linear programme is attained at a basic feasible solution, and all of
    them are enumerated (at most C(n + m, m) small linear systems for n configurations and m constraints).
    """
    n = len(cells)
    if n == 0:
        return 0.0, []
    Y = np.array([float(c["Y"]) for c in cells])
    rows = [np.ones(n), np.array([float(c["X"]) for c in cells]), np.array([float(c["R"]) for c in cells]),
            -np.array([float(c["E"]) for c in cells])]
    rhs = [1.0, float(x_max), float(r_max), -float(e_min)]
    if c_max is not None:
        rows.append(np.array([float(c["C"]) for c in cells]))
        rhs.append(float(c_max))
    m = len(rows)
    A = np.hstack([np.vstack(rows), np.eye(m)])          # [m, n + m]: w and one slack per constraint
    b = np.array(rhs)
    best, best_w = -1.0, None
    tol = 1e-12
    for basis in itertools.combinations(range(n + m), m):
        Bm = A[:, basis]
        if abs(np.linalg.det(Bm)) < 1e-14:
            continue
        xb = np.linalg.solve(Bm, b)
        if (xb < -1e-10).any():
            continue
        w = np.zeros(n + m)
        w[list(basis)] = np.clip(xb, 0.0, None)
        val = float(Y @ w[:n])
        if val > best + tol:
            best, best_w = val, w[:n].copy()
    if best_w is None:
        return 0.0, [0.0] * n
    return max(best, 0.0), [float(v) for v in best_w]


def mixture_value(cells: Sequence[Mapping[str, float]], w: Sequence[float], key: str) -> float:
    if len(cells) != len(w):
        raise ValueError("mixture and cells differ in length")
    return float(sum(float(wk) * float(c[key]) for wk, c in zip(w, cells)))


# ---------------------------------------------------------------------------------------------------- labels
def _arr(d: Sequence[float]) -> np.ndarray:
    x = np.asarray([float(v) for v in d], dtype=np.float64)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError(f"expected finite numbers, got {d!r}")
    return x


def interval(d: Sequence[float], conf: float = 0.95) -> Tuple[float, float]:
    """Two-sided t-interval of the mean; a constant sample gives the degenerate interval (mean, mean)."""
    x = _arr(d)
    if x.size >= 2 and float(x.std(ddof=1)) == 0.0:
        return float(x.mean()), float(x.mean())
    return t_interval(x.tolist(), conf)


def t_pvalue(d: Sequence[float], threshold: float, side: str) -> float:
    """One-sided t-test of the mean against ``threshold``.

    ``side="less"``: H0 mean >= threshold (claim: mean < threshold); ``side="greater"``: H0 mean <= threshold.
    A constant sample has no variance: p = 0 when the constant is strictly on the claimed side, else 1.
    """
    x = _arr(d)
    n = int(x.size)
    if n < 2:
        return 1.0
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    diff = mean - threshold if side == "greater" else threshold - mean
    if side not in ("less", "greater"):
        raise ValueError("side must be 'less' or 'greater'")
    if sd == 0.0:
        return 0.0 if diff > 0 else 1.0
    return float(student_t_sf(diff / (sd / math.sqrt(n)), n - 1))


def label_contrast(d: Sequence[float], sesoi: float, rejected: bool, min_worlds: int = MIN_WORLDS) -> str:
    """supported: Holm rejects the margin-shifted test (then mean > SESOI necessarily);
    below SESOI: the 95% t-interval lies inside (0, SESOI);
    refuted: the upper 95% bound is below SESOI and the lower bound is <= 0;
    inconclusive otherwise."""
    x = _arr(d)
    if x.size < min_worlds:
        return INSUFFICIENT
    if rejected:
        return SUPPORTED
    lo, hi = interval(x.tolist())
    if math.isfinite(hi) and hi < sesoi:
        return BELOW_SESOI if lo > 0 else REFUTED
    return INCONCLUSIVE


def two_sided(d: Sequence[float], sesoi: float) -> dict:
    """Two-sided SESOI contrast: p_plus tests mean(d) > +SESOI, p_minus tests mean(d) < -SESOI (exact sign-flip
    on d - SESOI and on -d - SESOI); the family p-value is min(1, 2 min(p_plus, p_minus))."""
    x = _arr(d)
    p_plus = signflip_pvalue(x.tolist(), sesoi)
    p_minus = signflip_pvalue((-x).tolist(), sesoi)
    return {"p_plus": p_plus, "p_minus": p_minus, "p": min(1.0, 2.0 * min(p_plus, p_minus))}


def label_two_sided(d: Sequence[float], sesoi: float, rejected: bool, test: Mapping[str, float],
                    names: Tuple[str, str] = ("A better", "B better"), min_worlds: int = MIN_WORLDS) -> str:
    """``names[0]`` if Holm rejects and the positive direction is the smaller p, ``names[1]`` for the mirror,
    'equivalent' if the 90% t-interval lies inside (-SESOI, +SESOI), else inconclusive."""
    x = _arr(d)
    if x.size < min_worlds:
        return INSUFFICIENT
    if rejected:
        return names[0] if test["p_plus"] <= test["p_minus"] else names[1]
    lo, hi = interval(x.tolist(), 0.90)
    if math.isfinite(lo) and math.isfinite(hi) and -sesoi < lo and hi < sesoi:
        return EQUIVALENT
    return INCONCLUSIVE


def threshold_claim(d: Sequence[float], threshold: float, side: str, rejected: Optional[bool] = None,
                    min_worlds: int = MIN_WORLDS) -> dict:
    """Threshold claim ``mean < threshold`` (side 'less') or ``mean > threshold`` (side 'greater').

    Returns ``{"p", "mean", "ci95", "label"}``. With ``rejected`` (the Holm outcome) the label is supported /
    refuted (the 95% interval lies wholly on the wrong side) / inconclusive; without it the label is None.
    """
    x = _arr(d)
    p = t_pvalue(x.tolist(), threshold, side)
    lo, hi = interval(x.tolist()) if x.size >= 2 else (math.nan, math.nan)
    out = {"p": p, "mean": float(x.mean()) if x.size else math.nan, "ci95": [lo, hi], "label": None,
           "threshold": threshold, "side": side}
    if rejected is not None:
        if x.size < min_worlds:
            out["label"] = INSUFFICIENT
        elif rejected:
            out["label"] = SUPPORTED
        elif (side == "less" and math.isfinite(lo) and lo > threshold) or \
                (side == "greater" and math.isfinite(hi) and hi < threshold):
            out["label"] = REFUTED
        else:
            out["label"] = INCONCLUSIVE
    return out


def upper_bound(d: Sequence[float], conf: float = 0.95) -> float:
    """Upper end of the two-sided t-interval of the mean (a conservative one-sided bound)."""
    return interval(d, conf)[1]


def lower_bound(d: Sequence[float], conf: float = 0.95) -> float:
    return interval(d, conf)[0]


def upper_bound_pass_probability(mean: float, sd: float, n: int, limit: float, conf: float = 0.95) -> float:
    """P(upper end of the two-sided ``conf`` t-interval of the mean of ``n`` values <= ``limit``) when the values
    are independent N(``mean``, ``sd``^2): the chance that a guard "upper bound <= limit" passes.

    With Z ~ N(0, 1) and S/sd ~ chi_{n-1}/sqrt(n-1) independent, the bound is
    ``mean + sd (Z + t S/sd) / sqrt(n)``, so the probability is ``E[Phi((limit - mean) sqrt(n)/sd - t S/sd)]``,
    integrated over the chi density on a fixed grid (deterministic: no random numbers). ``sd = 0`` gives 1 or 0.
    """
    n = int(n)
    if n < 2:
        raise ValueError("the bound needs at least two values")
    if not (math.isfinite(mean) and math.isfinite(sd) and math.isfinite(limit)) or sd < 0:
        raise ValueError("mean, sd and limit must be finite and sd >= 0")
    if sd == 0.0:
        return 1.0 if mean <= limit else 0.0
    nu = n - 1
    t = student_t_ppf(0.5 + conf / 2.0, nu)
    c = np.linspace(0.0, math.sqrt(nu) + 12.0, 24001)                     # chi_{nu} variable: S/sd = c / sqrt(nu)
    pdf = 2.0 * c ** (nu - 1) * np.exp(-c * c / 2.0) / (2.0 ** (nu / 2.0) * math.gamma(nu / 2.0))
    arg = (limit - mean) * math.sqrt(n) / sd - t * c / math.sqrt(nu)
    phi = np.array([0.5 * math.erfc(-a / math.sqrt(2.0)) for a in arg])
    y = phi * pdf
    return float(min(1.0, max(0.0, np.sum((y[1:] + y[:-1]) * np.diff(c)) / 2.0)))


def hier_bootstrap(per_world: Mapping[int, Sequence[Sequence[float]]],
                   combine: Optional[Callable[[np.ndarray], np.ndarray]] = None, n: int = BOOTSTRAP_N,
                   seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile interval of the mean over worlds of a per-world statistic, resampling worlds and then the
    world's units (facts or edits) with replacement.

    ``per_world[w]`` is a ``[units, k]`` table (one row per fact or edit, ``k`` columns, for example one column
    per arm). In every resample each drawn world slot redraws its units and takes the column means;
    ``combine`` maps the ``[draws, k]`` means to the per-world statistic (default: the first column). The random
    draws are those of CEV-11's frozen ``cev11.stats.hierarchical_bootstrap`` (same generator, same order), so
    with one column the two functions return the same interval.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    worlds = sorted(per_world)
    if not worlds:
        raise ValueError("hier_bootstrap needs at least one world")
    rng = np.random.Generator(np.random.PCG64(seed))
    n_worlds = len(worlds)
    slot_world = rng.integers(0, n_worlds, size=(n, n_worlds))
    per_slot = np.full((n, n_worlds), np.nan)
    for wi, w in enumerate(worlds):
        table = np.asarray(per_world[w], dtype=np.float64)
        if table.ndim == 1:
            table = table[:, None]
        if table.ndim != 2 or table.shape[0] == 0 or not np.all(np.isfinite(table)):
            raise ValueError(f"world {w}: expected a non-empty finite [units, k] table")
        mask = slot_world == wi
        k = int(mask.sum())
        if k == 0:
            continue
        m = table.shape[0]
        idx = rng.integers(0, m, size=(k, m))
        means = table[idx].mean(axis=1)                                    # [draws, columns]
        per_slot[mask] = means[:, 0] if combine is None else np.asarray(combine(means), dtype=np.float64)
    if not np.all(np.isfinite(per_slot)):
        raise ValueError("non-finite per-world statistic in the bootstrap")
    lo, hi = np.quantile(per_slot.mean(axis=1), [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    a, b = _arr(x), _arr(y)
    if a.size != b.size or a.size < 3 or float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def fisher_z(r: float) -> float:
    r = max(min(float(r), 1.0 - 1e-9), -1.0 + 1e-9)
    return math.atanh(r)


__all__ = ["ALPHA", "MIN_WORLDS", "BOOTSTRAP_N", "BOOTSTRAP_SEED", "SUPPORTED", "REFUTED", "INCONCLUSIVE",
           "BELOW_SESOI", "INSUFFICIENT", "NOT_TESTED", "EQUIVALENT", "X_MAX", "R_MAX", "E_MIN", "C_MAX", "frontier",
           "mixture_value", "interval", "t_pvalue", "label_contrast", "two_sided", "label_two_sided",
           "threshold_claim", "upper_bound", "lower_bound", "upper_bound_pass_probability", "hier_bootstrap",
           "pearson", "fisher_z", "holm", "holm_adjusted", "signflip_pvalue", "student_t_ppf", "student_t_sf",
           "t_interval"]
