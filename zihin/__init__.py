# -*- coding: utf-8 -*-
"""Zihin (mind): prototype of a non-Transformer core that learns while it lives.

One interface (events on channels), one substance (units and ties that grow), one rule (evidence on ties).
See ``research/living_mind_2026_11_14/NOTES.md`` for what is implemented as designed, what was changed and why.
"""
from .events import ACTED, IMAGINED, IMITATED, INFERRED, N_PROV, PROVENANCE, SENSED, TOLD, Event
from .mind import Config, Mind
from .store import ABSTRACT, ATOMIC, CATEGORY, COMPOSED, EXPECT, INDUCED, PART, SLOTTED, Substance

__all__ = ["ACTED", "IMAGINED", "IMITATED", "INFERRED", "N_PROV", "PROVENANCE", "SENSED", "TOLD", "Event",
           "Config", "Mind", "ABSTRACT", "ATOMIC", "CATEGORY", "COMPOSED", "EXPECT", "INDUCED", "PART",
           "SLOTTED", "Substance"]
