# -*- coding: utf-8 -*-
"""CEV-11 statistics and decision rules (PROTOCOL §10.1–10.2, §11, §13).

Everything here is a pure function of its inputs (explicit seeds, no global RNG, no I/O), so analyze.py and
verify.py can recompute every decision bit-for-bit.

Decision rule per hypothesis (PROTOCOL §11), applied by :func:`decide`:

* A world contributes only if it has at least ``MIN_EDITS_PER_WORLD`` (15) qualifying units; the caller passes the
  per-world effects ``d`` of the qualifying worlds only. Fewer than ``MIN_WORLDS`` (7) qualifying worlds ->
  ``"inconclusive (insufficient data)"``.
* Test: exact one-sided sign-flip permutation over all 2^n sign patterns of (d_w - m), H0: mean(d - m) <= 0
  (:func:`signflip_pvalue`); Holm step-down within each family (:func:`holm`).
* ``"supported"`` iff Holm rejects. ``"refuted"`` iff the upper bound of the two-sided 95% t-interval of mean(d) is
  below the margin m (this includes effects of the wrong sign). Otherwise ``"inconclusive"``.
* Manipulation check failed (H1/H2, PROTOCOL §3): a would-be refutation becomes
  ``"inconclusive (weak manipulation: cannot be refuted)"`` (``can_refute=False``).

Hypothesis-specific conjunctions (S1 non-inferiority, S3 Spearman condition) and overrides (not run, precondition,
map-only switch) are applied in analyze.py; their labels are defined here so that all labels live in one place.

Signature notes relative to INTERFACES.md (all backward compatible, positional arguments unchanged):
``decide`` takes keyword-only ``can_refute``, ``min_worlds`` and ``conf``; ``hierarchical_bootstrap`` takes
keyword-only ``stat`` (custom per-world statistic, used for the S4 median difference) and ``alpha``.
Note that ``decide(d, m, rejected)`` with fewer than 7 values returns the insufficient-data label, as the protocol
requires; unit tests of the three-way rule on small n pass ``min_worlds`` explicitly.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

# --- protocol constants (PROTOCOL §10–§11) -------------------------------------------------------------------
ALPHA = 0.05                 # one-sided family-wise level per family (Holm)
MIN_EDITS_PER_WORLD = 15     # a world contributes only with >= 15 qualifying units
MIN_WORLDS = 7               # fewer qualifying worlds -> insufficient data
T_CONF = 0.95                # two-sided t-interval used by the refutation rule
BOOTSTRAP_N = 10_000         # hierarchical bootstrap resamples (descriptive)
MAX_SIGNFLIP_N = 40          # exact enumeration limit (meet-in-the-middle keeps 2^20 work per half)

# --- decision labels ----------------------------------------------------------------------------------------
SUPPORTED = "supported"
REFUTED = "refuted"
INCONCLUSIVE = "inconclusive"
INSUFFICIENT = "inconclusive (insufficient data)"
WEAK_MANIPULATION = "inconclusive (weak manipulation: cannot be refuted)"
LOCALITY_COST = "not supported (locality cost)"
SPEARMAN_NOT_MET = "inconclusive (Spearman condition not met)"
PRECONDITION = "inconclusive (precondition)"
NOT_RUN = "not run"
NOT_TESTABLE = "not testable (map-only switch)"
LABELS = (SUPPORTED, REFUTED, INCONCLUSIVE, INSUFFICIENT, WEAK_MANIPULATION, LOCALITY_COST, SPEARMAN_NOT_MET,
          PRECONDITION, NOT_RUN, NOT_TESTABLE)


def _finite_array(values: Sequence[float], what: str) -> np.ndarray:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{what}: expected a flat sequence of numbers")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what}: non-finite value in {values!r}")
    return arr


# --- exact sign-flip permutation test -------------------------------------------------------------------------
def _signed_sums(values: np.ndarray) -> np.ndarray:
    """All 2^k sums sum_i s_i * v_i over sign patterns s in {+1, -1}^k (k = len(values); [0.0] for k = 0)."""
    sums = np.zeros(1, dtype=np.float64)
    for v in values:
        sums = np.concatenate([sums + v, sums - v])
    return sums


def signflip_pvalue(d: Sequence[float], margin: float) -> float:
    """Exact one-sided sign-flip permutation p-value for H0: mean(d - margin) <= 0.

    x_i = d_i - margin. Under H0 at its boundary the signs of x_i are exchangeable, so
    p = #{s in {+1,-1}^n : sum_i s_i x_i >= sum_i x_i} / 2^n, the identity pattern included (smallest p = 2^-n).
    Patterns whose statistic ties the observed one within a tiny relative tolerance (float round-off) are
    counted as ties, i.e. against rejection. The enumeration is exact (meet-in-the-middle over two halves).
    """
    x = _finite_array(d, "signflip_pvalue d") - float(margin)
    n = int(x.size)
    if n == 0:
        raise ValueError("signflip_pvalue needs at least one value")
    if n > MAX_SIGNFLIP_N:
        raise ValueError(f"exact enumeration limited to n <= {MAX_SIGNFLIP_N}, got {n}")
    t_obs = float(x.sum())
    tol = 1e-10 * max(1.0, float(np.abs(x).sum()))
    threshold = t_obs - tol
    half = n // 2
    left = _signed_sums(x[:half])
    right = np.sort(_signed_sums(x[half:]))
    # count pairs (a, b) with a + b >= threshold  <=>  b >= threshold - a
    first = np.searchsorted(right, threshold - left, side="left")
    count = int(np.sum(right.size - first))
    return count / float(2 ** n)


# --- Holm step-down ------------------------------------------------------------------------------------------
def _check_pvals(pvals: Mapping[str, float]) -> None:
    if not pvals:
        raise ValueError("holm needs at least one p-value")
    for k, p in pvals.items():
        if not (isinstance(p, (int, float)) and math.isfinite(p) and 0.0 <= p <= 1.0):
            raise ValueError(f"invalid p-value for {k!r}: {p!r}")


def holm(pvals: dict[str, float], alpha: float = ALPHA) -> dict[str, bool]:
    """Holm step-down: sort ascending, reject while p_(i) <= alpha / (m - i), stop at the first failure.

    m = len(pvals): pass every member of the preregistered family (an untested member gets p = 1.0), so the
    family size stays at its preregistered value. Ties are ordered by name, which cannot change the outcome.
    """
    _check_pvals(pvals)
    m = len(pvals)
    order = sorted(pvals, key=lambda k: (pvals[k], k))
    out = {k: False for k in pvals}
    for i, k in enumerate(order):
        if pvals[k] <= alpha / (m - i):
            out[k] = True
        else:
            break
    return out


def holm_adjusted(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (reported only): p~_(i) = max_{j<=i} min(1, (m - j) p_(j))."""
    _check_pvals(pvals)
    m = len(pvals)
    order = sorted(pvals, key=lambda k: (pvals[k], k))
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


# --- Student t quantile without scipy --------------------------------------------------------------------------
_FPMIN = 1e-300


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz; Numerical Recipes 6.4)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > _FPMIN else _FPMIN)
    h = d
    for m in range(1, 100_000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _FPMIN else _FPMIN)
        c = 1.0 + aa / c
        c = c if abs(c) > _FPMIN else _FPMIN
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _FPMIN else _FPMIN)
        c = 1.0 + aa / c
        c = c if abs(c) > _FPMIN else _FPMIN
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            return h
    raise RuntimeError(f"incomplete beta continued fraction did not converge (a={a}, b={b}, x={x})")


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) for a, b > 0, 0 <= x <= 1."""
    if a <= 0 or b <= 0:
        raise ValueError("betainc needs a, b > 0")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_sf(t: float, df: float) -> float:
    """Upper tail P(T > t) of Student's t with df degrees of freedom."""
    if df <= 0:
        raise ValueError("df must be positive")
    if t == 0.0:
        return 0.5
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t > 0 else 1.0 - tail


def student_t_ppf(q: float, df: float) -> float:
    """Quantile of Student's t (bisection on the upper tail; |relative error| ~ 1e-15)."""
    if not (0.0 < q < 1.0):
        raise ValueError("q must lie in (0, 1)")
    if q == 0.5:
        return 0.0
    if q < 0.5:
        return -student_t_ppf(1.0 - q, df)
    target = 1.0 - q                      # upper-tail probability, < 0.5
    lo, hi = 0.0, 1.0
    while student_t_sf(hi, df) > target:
        lo, hi = hi, hi * 2.0
        if hi > 1e300:
            raise RuntimeError("t quantile bracket overflow")
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if student_t_sf(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def t_interval(d: Sequence[float], conf: float = T_CONF) -> tuple[float, float]:
    """Two-sided t-interval of mean(d): mean +- t_{(1+conf)/2, n-1} * s / sqrt(n). (nan, nan) if n < 2."""
    x = _finite_array(d, "t_interval d")
    n = int(x.size)
    if n < 2:
        return (math.nan, math.nan)
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    half = student_t_ppf(0.5 + conf / 2.0, n - 1) * sd / math.sqrt(n)
    return (mean - half, mean + half)


# --- three-way decision ---------------------------------------------------------------------------------------
def decide(d: list[float], margin: float, rejected: bool, *, can_refute: bool = True,
           min_worlds: int = MIN_WORLDS, conf: float = T_CONF) -> str:
    """PROTOCOL §11 decision for one hypothesis from the per-world effects of its qualifying worlds.

    Returns SUPPORTED iff ``rejected`` (Holm); REFUTED iff the upper bound of the two-sided ``conf`` t-interval of
    mean(d) is below ``margin`` (WEAK_MANIPULATION instead when ``can_refute`` is False); INSUFFICIENT when fewer
    than ``min_worlds`` worlds are given; INCONCLUSIVE otherwise.
    """
    x = _finite_array(d, "decide d")
    if x.size < min_worlds:
        return INSUFFICIENT
    if rejected:
        if not float(x.mean()) > margin:
            raise ValueError(f"inconsistent input: rejected=True but mean(d)={float(x.mean())!r} <= margin={margin!r}")
        return SUPPORTED
    _, hi = t_interval(x.tolist(), conf)
    if math.isfinite(hi) and hi < margin:
        return REFUTED if can_refute else WEAK_MANIPULATION
    return INCONCLUSIVE


# --- hierarchical bootstrap (worlds, then units within the drawn world) ----------------------------------------
def hierarchical_bootstrap(per_world_edits: dict[int, list[float]], n: int = BOOTSTRAP_N, seed: int = 0, *,
                           stat: Callable[[list[Any]], float] | None = None,
                           alpha: float = 0.05) -> tuple[float, float]:
    """Percentile interval of the mean over worlds of a per-world statistic, resampling worlds and then units.

    Each resample draws len(worlds) worlds with replacement; every drawn slot resamples that world's units with
    replacement (independently per slot, also when a world is drawn twice) and computes the per-world statistic
    (``stat``; default the mean of float units). The resample value is the mean over slots. Returns the
    (alpha/2, 1 - alpha/2) quantiles (numpy 'linear' rule) of the n resample values. Deterministic in ``seed``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    worlds = sorted(per_world_edits)
    if not worlds:
        raise ValueError("hierarchical_bootstrap needs at least one world")
    rng = np.random.Generator(np.random.PCG64(seed))
    n_worlds = len(worlds)
    slot_world = rng.integers(0, n_worlds, size=(n, n_worlds))
    per_slot = np.full((n, n_worlds), np.nan)
    for wi, w in enumerate(worlds):
        items = list(per_world_edits[w])
        m = len(items)
        if m == 0:
            raise ValueError(f"world {w} has no units")
        mask = slot_world == wi
        k = int(mask.sum())
        if k == 0:
            continue
        idx = rng.integers(0, m, size=(k, m))
        if stat is None:
            vals = _finite_array(items, f"world {w} units")
            per_slot[mask] = vals[idx].mean(axis=1)
        else:
            per_slot[mask] = np.asarray([float(stat([items[j] for j in row])) for row in idx], dtype=np.float64)
    if not np.all(np.isfinite(per_slot)):
        raise ValueError("non-finite per-world statistic in the bootstrap")
    boots = per_slot.mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(lo), float(hi))


# --- Spearman with within-world permutation (S3) -------------------------------------------------------------------
def rankdata(values: Sequence[float]) -> np.ndarray:
    """1-based ranks with ties given their average rank."""
    x = _finite_array(values, "rankdata")
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(x.size, dtype=np.float64)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman_within_world(groups: Mapping[int, tuple[Sequence[float], Sequence[float]]], n_perm: int = 10_000,
                          seed: int = 0) -> dict:
    """Pooled within-world Spearman correlation and its one-sided within-world permutation p-value (H1: rho > 0).

    Within each world x and y are ranked (average ranks for ties) and centred, so between-world shifts cannot
    create correlation; rho is the Pearson correlation of the pooled centred ranks (equal to the mean per-world
    Spearman rho when worlds have equal size and no ties). The null permutes y within each world (n_perm seeded
    permutations); p = (1 + #{rho_perm >= rho}) / (1 + n_perm). Returns rho = None and p = 1.0 when the ranks have
    no variance (rho undefined).
    """
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    rxs, rys = [], []
    n_units = 0
    for w in sorted(groups):
        xs, ys = groups[w]
        x = _finite_array(xs, f"world {w} x")
        y = _finite_array(ys, f"world {w} y")
        if x.size != y.size:
            raise ValueError(f"world {w}: x and y differ in length ({x.size} vs {y.size})")
        n_units += int(x.size)
        if x.size < 2:
            continue
        rx = rankdata(x)
        ry = rankdata(y)
        rxs.append(rx - rx.mean())
        rys.append(ry - ry.mean())
    sxx = float(sum(float(np.dot(r, r)) for r in rxs))
    syy = float(sum(float(np.dot(r, r)) for r in rys))
    out = {"n": n_units, "n_worlds": len(groups), "n_perm": int(n_perm), "seed": int(seed)}
    if sxx <= 0.0 or syy <= 0.0:
        out.update(rho=None, p=1.0)
        return out
    denom = math.sqrt(sxx * syy)
    rho = float(sum(float(np.dot(a, b)) for a, b in zip(rxs, rys)) / denom)
    rng = np.random.Generator(np.random.PCG64(seed))
    num = np.zeros(n_perm, dtype=np.float64)
    for rx, ry in zip(rxs, rys):
        perms = rng.permuted(np.tile(ry, (n_perm, 1)), axis=1)
        num += perms @ rx
    rho_perm = num / denom
    hits = int(np.count_nonzero(rho_perm >= rho - 1e-12))
    out.update(rho=rho, p=(1 + hits) / (1 + n_perm))
    return out


def median(values: Sequence[float]) -> float:
    x = _finite_array(values, "median")
    if x.size == 0:
        raise ValueError("median of an empty sequence")
    return float(np.median(x))


def mean(values: Sequence[float]) -> float:
    x = _finite_array(values, "mean")
    if x.size == 0:
        raise ValueError("mean of an empty sequence")
    return float(x.mean())


__all__ = [
    "ALPHA", "MIN_EDITS_PER_WORLD", "MIN_WORLDS", "T_CONF", "BOOTSTRAP_N", "SUPPORTED", "REFUTED", "INCONCLUSIVE",
    "INSUFFICIENT", "WEAK_MANIPULATION", "LOCALITY_COST", "SPEARMAN_NOT_MET", "PRECONDITION", "NOT_RUN",
    "NOT_TESTABLE", "LABELS", "signflip_pvalue", "holm", "holm_adjusted", "betainc", "student_t_sf",
    "student_t_ppf", "t_interval", "decide", "hierarchical_bootstrap", "rankdata", "spearman_within_world",
    "median", "mean",
]
