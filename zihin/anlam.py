# -*- coding: utf-8 -*-
"""Substitution Evidence (Yerine Geçme Kanıtı): meaning equivalence learned from lived substitutions and their
consequences, counted by the one rule. Behind ``Config.substitution`` (off by default until measured).

The owner asked how the mind can know that merhaba, selam, selamün aleyküm and hello mean the same. No list of
equivalent words is given. Two expressions mean the same, for this mind, to the extent that one stands where the
other was expected and what follows is what the other would have led to (same place, same consequences; the
distributional and the bisimulation views in one test):

* **A witness.** In a moment where the mind expected x on a channel (its expectation e(x) > 1/2, read at the moment
  before) and y occurred there instead (y not expected as much as x), y has stood where x stands.
* **Its judgement, at the next moment.** What occurs right after y is read against x's own lived continuations (its
  lag-1 evidence): if x led to it (e(x -> z) > 1/2), the substitution kept x's consequences and counts for "y can
  stand for x"; if x had a lived continuation (its best e > 1/2) and led to nothing that occurred, it counts against.
  The amount is how far the expectation stands from 1/2: 2 |e - 1/2|.
* **Where it is kept.** As evidence on y's membership of x's categories (the ABSTRACT ties y -> g for each category g
  of x, weighted by e(x -> g)), with the inferred provenance: what the category's members lived then speaks for y
  through the members' reading (F11), and the trust in inference is learned like any trust. Nothing is kept for a
  pair whose x has no category yet. Every increment goes through the store and the ledger, so it is retractable.

Opposites stand in the same places (big / small) and are not told apart by place alone; only consequences that
differ (what follows them) count against, so a pair that only shares places gathers evidence both ways and stays
near 1/2. The own speech and thoughts of the mind are not witnesses (they say nothing about how others use words).
"""
from __future__ import annotations

from typing import List, Tuple

from .events import ACTED, IMAGINED, INFERRED
from .evidence import expectation
from .store import ABSTRACT, ATOMIC, W_NEG, W_POS


def substitution(mind, items: List[Tuple[int, int]]) -> None:
    """Judge the witnesses of the moment before with what occurs now, then note this moment's witnesses."""
    from .form import categories_of
    from .learn import add_evidence1
    from .read import own_mass
    S, k = mind.S, mind.cfg.k
    occurred = [u for u, prov in items if prov not in (ACTED, IMAGINED)]
    pending = getattr(mind, "witnesses", None) or []
    for x, y in pending:
        if not occurred:
            continue
        follow = own_mass(mind, x, 1, mind.trust)
        if not follow:
            continue
        expect = {z: expectation(p, n, k) for z, (p, n) in follow.items()}
        got = max(expect.get(z, 0.5) for z in occurred)
        best = max(expect.values())
        if got > 0.5:
            table, amount = W_POS, 2.0 * (got - 0.5)
        elif best > 0.5:
            table, amount = W_NEG, 2.0 * (best - 0.5)
        else:
            continue
        for category, e in categories_of(mind, x):
            if e > 0.5 and category != y:
                add_evidence1(mind, table, INFERRED, S.tie(y, category, ABSTRACT, 0), amount * e)
    witnesses = []
    if occurred and mind.predicted:
        kinds, channels = S.arrays().kind, S.arrays().channel
        for y in occurred:
            if not kinds[y] & ATOMIC:
                continue
            ch = channels[y]
            e_y = mind.predicted.get(y, 0.5)
            best_x, best_e = -1, max(0.5, e_y)
            for x, e in mind.predicted.items():
                if e > best_e and x != y and channels[x] == ch and kinds[x] & ATOMIC:
                    best_x, best_e = x, e
            if best_x >= 0:
                witnesses.append((best_x, y))
    mind.witnesses = witnesses
