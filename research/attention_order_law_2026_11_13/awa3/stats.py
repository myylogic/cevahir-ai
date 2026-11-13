# -*- coding: utf-8 -*-
"""Preregistered statistics of the AWA-3 study (PROTOCOL section 10). Pure Python, exact where stated.

The unit is the training run (seed); there is no bootstrap.

* ``exact_t4``            T4 = #{a_i with a_i - shift > max(b)}; exact permutation p under the location-shift null
                          (5 vs 5: T4 >= 4 has p = 6/252, T4 = 5 has p = 1/252). Tolerates one failed seed.
* ``mann_whitney_exact``  exact Mann-Whitney U by enumerating all splits (one- or two-sided, optional shift).
* ``hodges_lehmann``      median of the pairwise differences a_i - b_j, with exact rank-based intervals
                          (5 vs 5, one-sided 5%: the 5th smallest of the 25 differences, confidence 95.2%).
* ``holm``                Holm-adjusted p-values.
* ``t_upper`` / ``t_lower`` / ``t_test_greater``  one-sided Student-t bounds and test over seeds.
* ``fit_epsilon`` / ``predict_accuracy``  the per-hop error law acc(k) = h_k + (1 - h_k)(1 - eps)^(k - 1).
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

ALPHA = 0.05


def _finite(values: Sequence[float], what: str) -> List[float]:
    out = [float(v) for v in values]
    if not out or any(not math.isfinite(v) for v in out):
        raise ValueError(f"{what} must be a non-empty sequence of finite numbers")
    return out


def median(values: Sequence[float]) -> float:
    v = sorted(_finite(values, "values"))
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def mean(values: Sequence[float]) -> float:
    v = _finite(values, "values")
    return math.fsum(v) / len(v)


# ------------------------------------------------------------------------------- exact permutation
def _splits(n_total: int, n_a: int):
    indices = range(n_total)
    for chosen in itertools.combinations(indices, n_a):
        mask = set(chosen)
        yield chosen, [i for i in indices if i not in mask]


def t4_statistic(a: Sequence[float], b: Sequence[float]) -> int:
    top = max(b)
    return sum(1 for x in a if x > top)


def exact_t4(a: Sequence[float], b: Sequence[float], shift: float = 0.0) -> dict:
    """Exact permutation test of H0: (a - shift) and b are exchangeable, statistic T4.

    Returns {"T4", "p", "n_splits"}; p = share of the C(n+m, n) splits of the pooled shifted values whose T4 is
    at least the observed one.
    """
    a, b = _finite(a, "a"), _finite(b, "b")
    shifted = [x - shift for x in a]
    observed = t4_statistic(shifted, b)
    pooled = shifted + b
    hits = total = 0
    for ia, ib in _splits(len(pooled), len(a)):
        total += 1
        hits += t4_statistic([pooled[i] for i in ia], [pooled[i] for i in ib]) >= observed
    return {"T4": observed, "p": hits / total, "n_splits": total}


def u_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    """U = #{a_i > b_j} + 0.5 * #{a_i == b_j}."""
    return math.fsum(1.0 if x > y else 0.5 if x == y else 0.0 for x in a for y in b)


def mann_whitney_exact(a: Sequence[float], b: Sequence[float], shift: float = 0.0,
                       alternative: str = "greater") -> dict:
    """Exact Mann-Whitney test of (a - shift) against b by enumerating every split of the pooled values."""
    if alternative not in ("greater", "two-sided"):
        raise ValueError("alternative must be 'greater' or 'two-sided'")
    a, b = _finite(a, "a"), _finite(b, "b")
    shifted = [x - shift for x in a]
    observed = u_statistic(shifted, b)
    centre = 0.5 * len(a) * len(b)
    pooled = shifted + b
    hits = total = 0
    for ia, ib in _splits(len(pooled), len(a)):
        u = u_statistic([pooled[i] for i in ia], [pooled[i] for i in ib])
        total += 1
        if alternative == "greater":
            hits += u >= observed - 1e-12
        else:
            hits += abs(u - centre) >= abs(observed - centre) - 1e-12
    return {"U": observed, "p": hits / total, "n_splits": total, "alternative": alternative}


def u_null_cdf(n: int, m: int) -> List[float]:
    """P(U <= u) for u = 0..n*m under the null without ties (exact, by counting rank arrangements)."""
    counts = [0] * (n * m + 1)
    for chosen in itertools.combinations(range(n + m), n):
        # U = number of (a, b) pairs with a above b = sum over a of (#b below it)
        counts[sum(rank - i for i, rank in enumerate(chosen))] += 1
    total = sum(counts)
    cdf, running = [], 0
    for c in counts:
        running += c
        cdf.append(running / total)
    return cdf


def hodges_lehmann(a: Sequence[float], b: Sequence[float], alpha: float = ALPHA, two_sided: bool = False) -> dict:
    """Hodges-Lehmann estimate of the shift a - b with an exact rank-based confidence bound / interval.

    One-sided (default): "lower" is the c-th smallest pairwise difference with c the largest integer such that
    P(U <= c - 1) <= alpha; "confidence" is the achieved 1 - P(U <= c - 1). Two-sided: alpha / 2 in each tail,
    "lower" and "upper" are the c-th smallest and c-th largest differences. When no such c exists the bound is
    infinite (the sample is too small for that level).
    """
    a, b = _finite(a, "a"), _finite(b, "b")
    diffs = sorted(x - y for x in a for y in b)
    cdf = u_null_cdf(len(a), len(b))
    tail = alpha / 2 if two_sided else alpha
    c = 0
    while c < len(diffs) and cdf[c] <= tail:
        c += 1
    out = {"estimate": median(diffs), "n_pairs": len(diffs), "order_statistic": c,
           "confidence": (1.0 - (2 if two_sided else 1) * cdf[c - 1]) if c else None,
           "lower": diffs[c - 1] if c else -math.inf}
    if two_sided:
        out["upper"] = diffs[len(diffs) - c] if c else math.inf
    return out


def holm(pvals: Mapping[str, float]) -> Dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    for name, p in pvals.items():
        if not (isinstance(p, (int, float)) and 0.0 <= p <= 1.0):
            raise ValueError(f"p-value of {name!r} must be in [0, 1]")
    ordered = sorted(pvals.items(), key=lambda kv: (kv[1], kv[0]))
    m, running, out = len(ordered), 0.0, {}
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = running
    return out


# ------------------------------------------------------------------------------------- Student t
def _betacf(a: float, b: float, x: float) -> float:
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_sf(t: float, df: float) -> float:
    """P(T > t) for Student's t with df degrees of freedom."""
    if df <= 0:
        raise ValueError("df must be positive")
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    tail = 0.5 * betainc(0.5 * df, 0.5, df / (df + t * t))
    return tail if t >= 0 else 1.0 - tail


def student_t_ppf(q: float, df: float) -> float:
    """Quantile of Student's t (bisection on the survival function)."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    lo, hi = -1e6, 1e6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 - student_t_sf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _mean_se(values: Sequence[float]) -> Tuple[float, float, int]:
    v = _finite(values, "values")
    n = len(v)
    if n < 2:
        raise ValueError("a t-bound needs at least two seeds")
    m = math.fsum(v) / n
    var = math.fsum((x - m) ** 2 for x in v) / (n - 1)
    return m, math.sqrt(var / n), n


def t_upper(values: Sequence[float], conf: float = 0.95) -> float:
    """Upper one-sided confidence bound of the mean over seeds."""
    m, se, n = _mean_se(values)
    return m + student_t_ppf(conf, n - 1) * se


def t_lower(values: Sequence[float], conf: float = 0.95) -> float:
    """Lower one-sided confidence bound of the mean over seeds."""
    m, se, n = _mean_se(values)
    return m - student_t_ppf(conf, n - 1) * se


def t_test_greater(values: Sequence[float], mu0: float) -> float:
    """One-sided p-value of H0: mean <= mu0 (zero variance: 0 if the mean exceeds mu0, else 1)."""
    m, se, n = _mean_se(values)
    if se == 0.0:
        return 0.0 if m > mu0 else 1.0
    return student_t_sf((m - mu0) / se, n - 1)


# ------------------------------------------------------------------------------ per-hop error law
def predict_accuracy(k: int, floor: float, eps: float) -> float:
    """acc(k) = h_k + (1 - h_k) (1 - eps)^(k - 1)."""
    return floor + (1.0 - floor) * (1.0 - eps) ** (k - 1)


def fit_epsilon(acc_by_depth: Mapping[int, float], floors: Mapping[int, float], grid: int = 10_000) -> dict:
    """Least-squares per-hop error on the given depths (deterministic grid search, resolution 1 / grid)."""
    depths = sorted(acc_by_depth)
    if not depths or any(k < 2 for k in depths):
        raise ValueError("the per-hop law is fitted on depths >= 2")

    def sse(eps: float) -> float:
        return math.fsum((acc_by_depth[k] - predict_accuracy(k, floors[k], eps)) ** 2 for k in depths)

    best = min(range(grid + 1), key=lambda i: (sse(i / grid), i))
    return {"eps": best / grid, "sse": sse(best / grid), "depths": depths}


def classify_per_hop(y_endpoint: float, observed_plateau: float, predicted: float) -> str:
    """PROTOCOL per-hop classifier: precise / compounding / depth cliff / unclassified."""
    if y_endpoint >= 0.80:
        return "precise"
    if abs(observed_plateau - predicted) <= 0.07:
        return "compounding"
    if observed_plateau < predicted - 0.10:
        return "depth cliff"
    return "unclassified"


def majority(labels: Sequence[str]) -> Optional[str]:
    """The label held by more than half of the entries, else None."""
    for label in sorted(set(labels)):
        if 2 * sum(1 for x in labels if x == label) > len(labels):
            return label
    return None
