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


def _unit(mind, channel_name: str, symbol: str) -> int:
    cid = mind.ch.id(channel_name)
    index = mind.ch.symbol_index(cid, symbol, create=False)
    return mind.S.atom_index.get((cid, index), -1) if index >= 0 else -1


def analogue(mind, unit: int, channel_name: str, tuples: int = 60) -> Tuple[int, float]:
    """The concept most like ``unit`` by structure (structure mapping, a first form): in the lived sequences that
    contain it, the other words at their places; a sequence of the same length with one of those words at the same
    place and another unit at this unit's place makes that unit its counterpart (ateş evi ısıtır / güneş dünyayı
    ısıtır: ateş <-> güneş). Each shared word counts by how telling it is (1 / how many sequences hold it there).
    Returns (counterpart, strength) or (-1, 0)."""
    from .store import PART
    S = mind.S
    cid = mind.ch.id(channel_name)
    mine = sorted((int(S.t_src[t]) for t in S.inc.get((unit, PART), [])),
                  key=lambda c: -float(S.u_use[c]))[:tuples]
    score = {}
    for container in mine:
        parts = S.parts(container)
        if unit not in parts:
            continue
        i = parts.index(unit)
        for j, w in enumerate(parts):
            if j == i or w == unit:
                continue
            holders = [t for t in S.inc.get((w, PART), []) if int(S.t_lag[t]) == j]
            weight = 1.0 / max(1, len(holders))
            for t in holders:
                other_parts = S.parts(int(S.t_src[t]))
                if len(other_parts) != len(parts):
                    continue
                y = other_parts[i]
                if y != unit and int(S.u_kind[y]) & 1 and int(S.u_channel[y]) == cid and y not in parts:
                    score[y] = score.get(y, 0.0) + weight
    if not score:
        return -1, 0.0
    y = max(score, key=score.get)
    return y, score[y]


def imagine(session, topic: str, limit: int = 12) -> Tuple[List[str], List[str], float, str]:
    """T8, a first form of imagination: think about X, find the concept Y most like X (analogue), carry X's thought
    over to Y, and score that construction for reality step by step from all evidence except imagined (an ungrounded
    leap scores low). The construction is lived inwardly (class imagined), so later evidence can judge it.
    Returns (thought about X, construction about Y, reality of the construction, Y)."""
    from .__main__ import CHANNEL, words
    mind = session.mind
    thought, _ = think(session, topic, limit)
    symbols = words(topic)
    if not symbols or not thought:
        return thought, [], 0.0, ""
    x = _unit(mind, CHANNEL, symbols[-1])
    if x < 0:
        return thought, [], 0.0, ""
    y, strength = analogue(mind, x, CHANNEL)
    if y < 0:
        return thought, [], 0.0, ""
    other = str(mind.symbol_of(y)[1])
    construction, scores = [other], []
    mind.step([Event(CHANNEL, other, IMAGINED)])
    for w in thought:                                      # X's thought, said of Y, step by step
        u = _unit(mind, CHANNEL, w)
        scores.append(_reality(mind, u) if u >= 0 else 0.5)
        mind.step([Event(CHANNEL, w, IMAGINED)])
        construction.append(w)
    mind.step([])
    reality = sum(scores) / len(scores) if scores else 0.0
    return thought, construction, reality, other
