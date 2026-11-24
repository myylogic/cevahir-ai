# -*- coding: utf-8 -*-
"""The one rule of the evidence core, unchanged, and what it says about many outcomes.

A claim with W+ confirmations and W- contradictions (W = W+ + W-) is believed with

    f = W+ / W,    c = W / (W + k),    e = c (f - 1/2) + 1/2,    k = 1            (zihin/evidence.py)

which is the same number as

    e = (W+ + k/2) / (W + k).

With k = 1 this is the Krichevsky-Trofimov estimator: the posterior mean of a Bernoulli rate under the Jeffreys prior
Beta(1/2, 1/2). The rule of the theory is therefore a known estimator with known guarantees. For binary sequences, its
sequential code length exceeds that of the best fixed rate in hindsight by at most (1/2) log2 n + 1 bits.

Many outcomes. A context (or any origin) that has met n outcomes, t of them distinct, is asked what comes next. The
rule is applied to the novelty event, "what follows is new to this origin". Each distinct outcome was new once, so
W+ = t and W = n:

    novelty = (t + 1/2) / (n + 1).

The rest of the belief, 1 - novelty, is shared among the outcomes met, in proportion to their counts. The novelty mass
goes to a coarser origin: a shorter context, and at the root a symbol never met. No new constant enters.

The speaking line. An option racing alone must reach theta = 1 within T = 6 settling steps, so it is said only when
its belief is at least e* = 1/2 + theta / (2 T) ~ 0.583 (docs/research/living_evidence_mind_2026_11_24/PAPER.md, 2.4).
"""
from __future__ import annotations

K = 1.0                                   # conviction constant of the theory
THETA = 1.0                               # act threshold
T = 6                                     # settling steps
SPEAKING_LINE = 0.5 + THETA / (2.0 * T)   # e* ~ 0.5833


def belief(pos: float, total: float, k: float = K) -> float:
    """e = c (f - 1/2) + 1/2 with f = pos / total and c = total / (total + k); 1/2 when nothing was lived."""
    if total <= 0:
        return 0.5
    return (total / (total + k)) * (pos / total - 0.5) + 0.5


def novelty(distinct: int, total: float, k: float = K) -> float:
    """Belief that what follows an origin is new to it: the rule on the novelty event, W+ = distinct, W = total.

    An origin that has met nothing believes everything that follows is new (1)."""
    if total <= 0:
        return 1.0
    return belief(distinct, total, k)


ESCAPES = {
    # the rule (the default): the novelty event judged by the one rule
    "rule": lambda t, n: (t + 0.5) / (n + 1.0),
    # escape estimators of the compression literature, for comparison only
    "ppm-a": lambda t, n: 1.0 / (n + 1.0),          # Cleary and Witten (1984), method A
    "ppm-c": lambda t, n: t / (n + t),              # Moffat (1990), Witten and Bell (1991)
    "ppm-d": lambda t, n: t / (2.0 * n),            # Howard (1993)
}
