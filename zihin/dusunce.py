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

from typing import Dict, List, Tuple

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


def topic_of(session, text: str = None) -> str:
    """What to think about: the utterance's (by default the last one's) least known content word (the highest
    drive), among the words the mind has met."""
    from .__main__ import CHANNEL, words
    from .read import drive_of
    mind = session.mind
    if text is None:
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


def sort_dreams(mind) -> Dict[str, float]:
    """Sorting what was imagined (in sleep): every tie that holds imagined evidence not yet judged is compared with
    the evidence on it that is not imagined (lived, read, told; own acts excluded), weighted by that evidence's own
    conviction. A positive imagined claim agrees in proportion to the real frequency and differs in proportion to its
    complement; a negative one the other way round. The agreement and the difference are evidence on the mind's trust
    in its own imagination (source "imagined" -> holds); judged amounts are marked so nothing is judged twice. Ties
    with no real evidence stay open: hypotheses, neither true nor false yet."""
    from .evidence import conviction
    from .events import ACTED, IMAGINED, SENSED
    from .store import C_NEG, C_POS, W_NEG, W_POS
    S = mind.S
    n = S.n_ties
    if n == 0 or mind.source_tie[IMAGINED] < 0:
        return {"true": 0.0, "false": 0.0, "open": 0.0, "trust": float(mind.trust[IMAGINED])}
    open_pos = S.t_w[W_POS, :n, IMAGINED] - S.t_w[C_POS, :n, IMAGINED]
    open_neg = S.t_w[W_NEG, :n, IMAGINED] - S.t_w[C_NEG, :n, IMAGINED]
    pending = ((open_pos > 0) | (open_neg > 0)).nonzero().flatten()
    if pending.numel() == 0:
        return {"true": 0.0, "false": 0.0, "open": 0.0, "trust": float(mind.trust[IMAGINED])}
    real = mind.trust.clone()
    real[IMAGINED] = 0.0
    real[ACTED] = 0.0
    w_pos = (S.t_w[W_POS, pending] * real).sum(-1)
    w_neg = (S.t_w[W_NEG, pending] * real).sum(-1)
    w = w_pos + w_neg
    has_real = w > 0
    ids = pending[has_real]
    true = false = 0.0
    if ids.numel():
        f = (w_pos[has_real] / w[has_real])
        c = conviction(w[has_real], mind.cfg.k)
        op, on = open_pos[ids].clamp(min=0), open_neg[ids].clamp(min=0)
        agree = c * (op * f + on * (1 - f))
        differ = c * (op * (1 - f) + on * f)
        S.begin(mind.t + 1, mind.ch.sizes())                # sorting is a moment of the mind's own life
        mind.t += 1
        S.add(C_POS, IMAGINED, ids, op)
        S.add(C_NEG, IMAGINED, ids, on)
        true, false = float(agree.sum()), float(differ.sum())
        S.add1(W_POS, SENSED, mind.source_tie[IMAGINED], true)
        S.add1(W_NEG, SENSED, mind.source_tie[IMAGINED], false)
        S.end()
        mind.trust = mind._trust()
    return {"true": true, "false": false, "open": float((~has_real).sum()), "trust": float(mind.trust[IMAGINED])}


def _real(mind) -> torch.Tensor:
    """Trust with imagination and own acts left out: what counts as real (lived, read, told, inferred)."""
    trust = mind.trust.clone()
    trust[IMAGINED] = 0.0
    trust[ACTED] = 0.0
    return trust


def _claims(mind, words_: List[str], lags=(1,)) -> List[Tuple[int, int, int]]:
    """What a sequence claims: each word is followed by the word that follows it there (the EXPECT ties a lived
    sequence teaches between words; longer reach is carried by composed units, not by word-to-word ties).
    Returned as (unit, unit, lag)."""
    from .__main__ import CHANNEL
    units = [_unit(mind, CHANNEL, w) for w in words_]
    out = []
    for i, u in enumerate(units):
        for lag in lags:
            if u >= 0 and i + lag < len(units) and units[i + lag] >= 0:
                out.append((u, units[i + lag], lag))
    return out


def _real_on(mind, claims) -> List[Tuple[float, float]]:
    """(W+, W-) of real evidence on the tie of each claim, trust-weighted; (0, 0) where no tie exists."""
    from .store import EXPECT, W_NEG, W_POS
    S = mind.S
    real = _real(mind)
    out = []
    for u, v, lag in claims:
        tie = next((t for t in S.out.get((u, EXPECT, lag), []) if int(S.t_dst[t]) == v), -1)
        if tie < 0:
            out.append((0.0, 0.0))
        else:
            out.append((float((S.t_w[W_POS, tie] * real).sum()), float((S.t_w[W_NEG, tie] * real).sum())))
    return out


def remember_dream(mind, about: str, other: str, words_: List[str], reality) -> dict:
    """A dream is kept as what it claimed and how much reality backed each claim when it was dreamt; the claims
    nothing real had touched then are its new part (what it imagined, not what it recalled)."""
    claims = _claims(mind, words_)
    record = {"t": mind.t, "about": about, "other": other, "words": list(words_), "reality": reality,
              "claims": [(u, v, lag, wp, wn) for (u, v, lag), (wp, wn) in zip(claims, _real_on(mind, claims))]}
    dreams = getattr(mind, "dreams", None)
    if dreams is None:
        dreams = mind.dreams = []
    dreams.append(record)
    del dreams[:-500]                                      # the oldest dreams fade
    return record


def verdict(mind, record) -> Tuple[str, int, int]:
    """Which of a dream's new claims reality has since judged, and how, by the core's own measure: a claim holds
    when the expectation real evidence gives it is above one half, fails when below. "dogru" when every new claim
    holds, "yanlis" when any fails, "acik" otherwise; "biliniyordu" when it had no new claim (it recalled, it did
    not imagine). Returns (verdict, judged, new)."""
    from .evidence import expectation
    new = [(u, v, lag) for u, v, lag, wp, wn in record["claims"] if wp + wn == 0]
    if not new:
        return "biliniyordu", 0, 0
    judged, backed, contradicted = 0, 0, 0
    for wp, wn in _real_on(mind, new):
        if wp + wn == 0:
            continue
        judged += 1
        e = float(expectation(torch.tensor(wp), torch.tensor(wn), mind.cfg.k))
        backed += e > 0.5
        contradicted += e < 0.5
    if contradicted:
        return "yanlis", judged, len(new)
    if backed == len(new):
        return "dogru", judged, len(new)
    return "acik", judged, len(new)


def sleep(session, cycles: int = 5) -> Dict[str, object]:
    """Sleep: dreams, then sorting. The mind dreams about the last topic and about what it is unsure of (words it
    uses but holds with little conviction): each dream is a thought, carried over to a counterpart where it has one
    (imagine), lived as imagined and kept with its claims. Then everything imagined is sorted against what is real:
    the evidence on its trust in its own imagination (sort_dreams), and each kept dream's verdict (true, false,
    still open). Returns the night's dreams, the verdicts of the dreams reality has judged, and the sorting."""
    from .__main__ import CHANNEL
    from .web import wanted_topics
    mind = session.mind
    topics = []
    last = topic_of(session)
    if last:
        topics.append(last)
    for w in wanted_topics(mind, CHANNEL, cycles * 2):
        if w not in topics:
            topics.append(w)
    before_night = len(getattr(mind, "dreams", None) or [])
    dreams = []
    for topic in topics[:cycles]:
        thought, construction, reality, other = imagine(session, topic)
        if construction:
            dreams.append(remember_dream(mind, topic, other, construction, round(reality, 2)))
        elif thought:
            dreams.append(remember_dream(mind, topic, "", [topic] + thought, None))
    before = float(mind.trust[IMAGINED])
    sorting = sort_dreams(mind)
    sorting["trust_before"] = before
    judged = []
    for record in (getattr(mind, "dreams", None) or [])[:before_night]:   # earlier dreams reality may have met
        v, n_judged, n_new = verdict(mind, record)
        if v in ("dogru", "yanlis") and record.get("said") != v:
            record["said"] = v
            judged.append((v, record))
    mind.nights = getattr(mind, "nights", 0) + 1
    return {"dreams": dreams, "judged": judged, "sorting": sorting}
