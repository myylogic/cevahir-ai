# -*- coding: utf-8 -*-
"""Thinking: the inner voice with a measured sense of reality (LAYERS.md L3, design section 6).

The mind brings a topic to mind and speaks inwardly with the very machinery it speaks with (the same word units, the
same reading: F11/F12 starts, F16 continuation), but nothing leaves it: each word it says inwardly comes back as its
own echo with provenance "imagined". Every step is scored for reality before it is taken: the expectation of that
word given the present inner context, read from all evidence EXCEPT what was imagined (what was lived, read, told
supports it or not). The thought is learned like any experience, in the class "imagined", so trust in its own
constructions is learned when sensed or told evidence later lands on the same ties (learn.judge).

A life clock (session) lets the mind think while nobody talks to it, and lets it speak first when a thought it had is
real enough and the silence has lasted; it never speaks into someone else's turn.
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from .events import ACTED, IMAGINED, N_PROV, Event


def _reality(mind, unit_symbol) -> float:
    """Expectation of the next inner word from everything but imagination (the reality score of that step)."""
    from .read import members_map
    if mind.context is None:
        return 0.5
    trust = mind.trust.clone()
    trust[IMAGINED] = 0.0
    trust[ACTED] = 0.0
    return members_map(mind, mind.context, trust).get(unit_symbol, 0.5)


def think(session, topic: str, limit: int = 15) -> Tuple[List[str], float]:
    """Think about ``topic`` inwardly. Returns the inner words and the reality score R (mean over steps)."""
    from .__main__ import CHANNEL, words
    mind = session.mind
    symbols = words(topic)
    if not symbols:
        return [], 0.0
    cid = mind.ch.id(CHANNEL)
    acts = []
    for i, symbol in enumerate(symbols):                   # the topic is brought to mind (imagined, not heard)
        acts = mind.step([Event(CHANNEL, symbol, IMAGINED)])
    thought, scores = [], []
    while acts and len(thought) < limit:
        said = [e for e in acts if e.channel == CHANNEL]
        if not said:
            break
        word = said[0].symbol
        index = mind.ch.symbol_index(cid, word, create=False)
        unit = mind.S.atom_index.get((cid, index), -1) if index >= 0 else -1
        scores.append(_reality(mind, unit) if unit >= 0 else 0.5)
        thought.append(str(word))
        # the inner word comes back as imagined, not as an act heard by others
        mind.pending = [Event(e.channel, e.symbol, IMAGINED, e.intensity) if e.channel == CHANNEL and e.prov == ACTED
                        else e for e in mind.pending]
        acts = mind.step([])
    mind.step([])                                          # the thought ends; the episode closes
    reality = sum(scores) / len(scores) if scores else 0.0
    return thought, reality


def topic_of(session) -> str:
    """What to think about: the last utterance's least known content word (the highest drive), else its last word."""
    from .__main__ import CHANNEL, words
    from .read import drive_of
    mind = session.mind
    if not session.history:
        return ""
    text = session.history[-1][1]
    if text.startswith("["):
        return ""
    cid = mind.ch.id(CHANNEL)
    best, best_drive = "", -1.0
    for w in words(text):
        if len(w) < 4 or not w.isalpha():
            continue
        index = mind.ch.symbol_index(cid, w, create=False)
        unit = mind.S.atom_index.get((cid, index), -1) if index >= 0 else -1
        d = drive_of(mind, unit) if unit >= 0 else 0.0
        if d > best_drive:
            best, best_drive = w, d
    return best
