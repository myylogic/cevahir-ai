#!/usr/bin/env python3
"""How far are the closed-form messages of bg_core.py from exact hierarchical Bayes?

Reference (exact up to quadrature): theta_i ~ Beta(k a, k(1-a)), a ~ U(0,1), k uniform on GRID,
counts (wp_i, wn_i) per member. Posterior mean of theta for (a) every member with data and
(b) a new member without data, by numerical integration over a (4001 points) and the grid.
Model under test: bg_core with outliers switched off (v fixed ~1) and certain memberships, so
that the two solve the same problem. Baselines: own counts with Laplace; pooled counts with Laplace.
Run: python3 check_exact_hbm.py [--split]   (about 1 minute; --split lets the model form sub-categories)
"""
import json
import sys
from math import lgamma

import numpy as np

from bg_core import GRID, HAS, ISA, Mind

A = (np.arange(4001) + 0.5) / 4001
LG = np.vectorize(lgamma)


def lbeta(a, b):
    return LG(a) + LG(b) - LG(a + b)


def exact(counts, target):
    """counts: list of (wp, wn); target: (wp, wn) of the member asked about (already excluded or not)."""
    logp = np.zeros((len(GRID), len(A)))
    for j, k in enumerate(GRID):
        for wp, wn in counts:
            logp[j] += lbeta(k * A + wp, k * (1 - A) + wn) - lbeta(k * A, k * (1 - A))
    w = np.exp(logp - logp.max())
    w /= w.sum()
    wp, wn = target
    post = (wp + GRID[:, None] * A[None, :]) / (wp + wn + GRID[:, None])
    return float((w * post).sum())


SPLIT = "--split" in sys.argv


def model(counts):
    m = Mind("chk", v_fixed=1 - 1e-9, certain_membership=True, allow_split=SPLIT)
    m.set_prior_trust("obs", 1e12, 0.0)
    for i, (wp, wn) in enumerate(counts):
        m.perceive(ISA, f"m{i}", "cat", 1, "obs", event=True)
        for _ in range(int(wp)):
            m.perceive(HAS, f"m{i}", "p", 1, "obs", event=True)
        for _ in range(int(wn)):
            m.perceive(HAS, f"m{i}", "p", 0, "obs", event=True)
    m.perceive(ISA, "new", "cat", 1, "obs", event=True)
    out = [m.answer(HAS, m.ix[f"m{i}"], m.ix["p"])["e"] for i in range(len(counts))]
    return out, m.answer(HAS, m.ix["new"], m.ix["p"])["e"], m


def main():
    rng = np.random.default_rng(20261114)
    rows = []
    nsplit = {}
    regimes = {"mixed": lambda: 0.5, "skewed": lambda: 0.8, "pure": lambda: float(rng.random() < 0.5),
               "hetero": lambda: rng.random()}
    for reg, draw in regimes.items():
        for M in (2, 4, 8):
            for rep in range(12):
                counts = []
                for _ in range(M):
                    n = int(rng.integers(1, 7))
                    kpos = int(rng.binomial(n, draw()))
                    counts.append((kpos, n - kpos))
                e_mod, e_new, mm = model(counts)
                nsplit[reg] = nsplit.get(reg, 0) + (len(mm.sub) > 0)
                tp, tn = sum(c[0] for c in counts), sum(c[1] for c in counts)
                for i, (wp, wn) in enumerate(counts):
                    ex = exact(counts, (wp, wn))      # exact joint posterior: all data, then member i
                    rows.append(dict(reg=reg, M=M, kind="member", exact=ex, model=e_mod[i],
                                     own=(wp + 1) / (wp + wn + 2), pooled=(tp + 1) / (tp + tn + 2)))
                ex = exact(counts, (0, 0))
                rows.append(dict(reg=reg, M=M, kind="new", exact=ex, model=e_new, own=0.5,
                                 pooled=(tp + 1) / (tp + tn + 2)))
    res = {}
    print(f"{'regime':8s} {'kind':7s} {'n':>4s} | mean abs error vs exact hierarchical Bayes:  model   own-Laplace  pooled-Laplace | max(model)")
    for reg in regimes:
        for kind in ("member", "new"):
            r = [x for x in rows if x["reg"] == reg and x["kind"] == kind]
            em = np.array([abs(x["model"] - x["exact"]) for x in r])
            eo = np.array([abs(x["own"] - x["exact"]) for x in r])
            ep = np.array([abs(x["pooled"] - x["exact"]) for x in r])
            res[f"{reg}/{kind}"] = dict(n=len(r), model=float(em.mean()), own=float(eo.mean()), pooled=float(ep.mean()),
                                        model_max=float(em.max()))
            print(f"{reg:8s} {kind:7s} {len(r):4d} |                                              {em.mean():.4f}   {eo.mean():.4f}       {ep.mean():.4f}        | {em.max():.4f}")
    em = np.array([abs(x["model"] - x["exact"]) for x in rows])
    eo = np.array([abs(x["own"] - x["exact"]) for x in rows])
    ep = np.array([abs(x["pooled"] - x["exact"]) for x in rows])
    res["all"] = dict(n=len(rows), model=float(em.mean()), own=float(eo.mean()), pooled=float(ep.mean()), model_max=float(em.max()))
    print(f"{'ALL':8s} {'':7s} {len(rows):4d} |                                              {em.mean():.4f}   {eo.mean():.4f}       {ep.mean():.4f}        | {em.max():.4f}")
    res["splits_per_regime_of_36"] = nsplit
    print("category splits allowed:", SPLIT, "| scenarios (of 36 per regime) in which a sub-category was formed:", nsplit)
    json.dump(res, open("check_exact_hbm_split.json" if SPLIT else "check_exact_hbm.json", "w"), indent=1)


if __name__ == "__main__":
    main()
