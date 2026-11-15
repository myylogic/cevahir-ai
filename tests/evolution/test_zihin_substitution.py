# -*- coding: utf-8 -*-
"""Zihin: Substitution Evidence (zihin/anlam.py) and the Substitute Reading (read.substitute_backoff), behind
Config.substitution (off by default). A word lived where another was expected, followed by what the other leads to,
gains inferred evidence of belonging to the other's categories; with the flag off nothing changes."""
from __future__ import annotations

from zihin import Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS
from zihin.events import INFERRED, TOLD
from zihin.form import categories_of


def life(substitution: bool) -> Mind:
    cfg = VARIANTS["speaking"]()
    cfg.substitution = substitution
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    for _ in range(4):
        s.live("bugün kedi süt içti .", TOLD)
        s.live("bugün köpek süt içti .", TOLD)
    s.live("bugün kuzu süt içti .", TOLD)
    return mind


def shared(mind: Mind, a: str, b: str) -> int:
    ua, ub = mind.unit_of(CHANNEL, a), mind.unit_of(CHANNEL, b)
    return len({g for g, e in categories_of(mind, ua) if e > 0.5} & {g for g, e in categories_of(mind, ub) if e > 0.5})


def test_substitution_adds_inferred_membership_only_when_on():
    off, on = life(False), life(True)
    def inferred(mind):
        lamb = mind.unit_of(CHANNEL, "kuzu")
        return sum(float(mind.S.w1(0, t)[INFERRED]) for t in mind.S.out.get((lamb, 2, 0), []))
    assert inferred(on) > 0.0 and inferred(off) == 0.0      # kuzu stood where kedi was expected, and süt followed
    assert shared(on, "kuzu", "kedi") >= shared(off, "kuzu", "kedi")
