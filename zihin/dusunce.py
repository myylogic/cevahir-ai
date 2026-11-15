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


def _words(session, text: str) -> List[str]:
    """The session's text adapter (its mind's language convention), else the default one."""
    adapter = getattr(session, "words", None)
    if adapter is None:
        from .__main__ import words as adapter
    return adapter(text)


def think(session, topic: str, limit: int = 15) -> Tuple[List[str], float]:
    """Think about ``topic`` inwardly. Returns the inner words and the reality score R (mean over steps).

    The topic is brought to mind (imagined, not heard): a topic of several words is taken in word by word, the mind
    deciding nothing until its last word, as it takes in what it hears. From then on the inner voice is the mind's own
    decisions, each lived as imagined (go_on): it ends when the mind decides nothing, or when it comes round to a
    composed context this episode already lived; ``limit`` words are only a safety cap."""
    from .__main__ import CHANNEL
    mind = session.mind
    symbols = _words(session, topic)
    if not symbols:
        return [], 0.0
    acts = take_in(mind, [[Event(CHANNEL, symbol, IMAGINED)] for symbol in symbols])   # the topic is brought to mind
    thought, scores = [], []

    def heard(words_: List[str]) -> None:                  # each word as it is decided, before its echo comes back
        for word in words_:
            unit = mind.unit_of(CHANNEL, word)
            scores.append(_reality(mind, unit) if unit >= 0 else 0.5)
            thought.append(word)
    go_on(mind, acts, heard, limit, imagined=True)
    mind.step([])                                          # the thought ends; the episode closes (a silent moment:
    reality = sum(scores) / len(scores) if scores else 0.0  # no context, so nothing is decided in it)
    return thought, reality


def take_in(mind, moments: List[List[Event]]) -> List[Event]:
    """Live these moments one after another without deciding until the last (the session's way of taking in an
    utterance: the act threshold is raised while it lasts); returns what the mind decides in the last one."""
    for events in moments[:-1]:
        hear_without_deciding(mind, events)
    return mind.step(moments[-1]) if moments else []


def hear_without_deciding(mind, events: List[Event] = ()) -> None:
    """One moment in which the mind perceives (what is pending from the last moment, and ``events``) but decides
    nothing: the act threshold is raised for it, as while someone else speaks."""
    theta = mind.knob["theta"]
    mind.knob["theta"] = 1e9
    try:
        mind.step(list(events))
    finally:
        mind.knob["theta"] = theta


def comes_round(mind) -> bool:
    """Whether the echo of what the mind has just decided (its pending acts, perceived in the next moment) brings back a
    composed context this episode already lived: the episode tuple that moment would make (its last F items) is a
    composed unit already in the episode's trace."""
    items = []
    for e in mind.pending:
        if e.intensity <= 0 or not mind.ch.has(e.channel):
            continue
        channel = mind.ch.id(e.channel)
        if mind.ch.kind(channel) not in ("sense", "act"):
            continue
        unit = mind._atom(channel, e.symbol, create=False)
        if unit < 0:                                       # a symbol never met: a context never lived
            return False
        items.append(unit)
    key = tuple((list(mind.episode) + items)[-mind.knob["F"]:]) if items else ()
    if len(key) < 2:
        return False
    context = mind.S.tuple_index.get(key)
    return context is not None and context in mind.trace


def go_on(mind, acts: List[Event], out, limit: int, imagined: bool = False) -> None:
    """Carry on from what the mind has just decided: ``out`` gets the words of each decision (text channel) as they
    are decided; their echo comes back in the next moment (``imagined``: relabelled imagined, the inner voice; else as
    its own acts, speech), where it decides again. It ends when the mind decides nothing, or when the echo brings back
    a composed context this episode already lived (comes_round): the echo is then perceived while the mind does not
    decide again, so no echo is dropped and no decision is left unsaid. ``limit`` moments are a safety cap, ended the
    same way."""
    from .__main__ import CHANNEL
    moments = 0
    while acts:
        out([str(e.symbol) for e in acts if e.channel == CHANNEL])
        if imagined:                                       # the inner word comes back as imagined, not as speech
            mind.pending = [Event(e.channel, e.symbol, IMAGINED, e.intensity) if e.channel == CHANNEL and
                            e.prov == ACTED else e for e in mind.pending]
        moments += 1
        if moments >= limit or comes_round(mind):
            hear_without_deciding(mind)
            return
        acts = mind.step([])


FREQUENT = 200        # how many of the most used words carry the language rather than a topic, at most
SHARE = 10            # ... and in a smaller vocabulary at most one word in SHARE (sleep, see frequent_words)


def frequent_cut(vocabulary: int, k: int = FREQUENT, share: int = SHARE) -> int:
    """How many words count as frequent in a vocabulary of a given size when the cut is relative: the most used
    ceil(vocabulary / share), at most k. A fixed number of words is a share that grows as the vocabulary shrinks:
    200 is a tenth of 2000 words, all of a mind that met 200, and a mind whose every word is frequent has no topic
    left to dream about. Equal to k from k * share words on (2000 by default), so larger minds are cut as before."""
    return min(k, -(-vocabulary // share))


def frequent_words(mind, k: int = FREQUENT, relative: bool = False) -> set:
    """The k most used words of the mind's life (they carry the language, not a topic; frequency decides). With
    ``relative``, k is cut to the vocabulary (frequent_cut): the same words from 2000 words on."""
    from .__main__ import CHANNEL
    from .store import ATOMIC
    S = mind.S
    cid = mind.ch.id(CHANNEL)
    atoms = [u for u in range(S.n_units)
             if int(S.u_kind[u]) & ATOMIC and int(S.u_channel[u]) == cid and S.u_alive[u]]   # not what !unut took back
    if relative:
        k = frequent_cut(len(atoms), k)
    top = sorted(atoms, key=lambda u: -float(S.u_use[u]))[:k]
    return {str(mind.ch.symbol(cid, int(S.u_symbol[u]))) for u in top}


def topic_of(session, text: str = None) -> str:
    """What to think about: the least known content word (the highest drive) of an utterance, among the words the
    mind has met; by default of the last utterance it heard in the session, whatever the history entry it belongs to
    (a line told, a chat, a dialogue watched, the last sentence of a file read or taught; Session.last_heard)."""
    from .__main__ import CHANNEL
    from .read import drive_of
    mind = session.mind
    if text is None:
        last_heard = getattr(session, "last_heard", None)
        text = last_heard() if last_heard is not None else ""
    if not text:
        return ""
    cid = mind.ch.id(CHANNEL)
    best, best_drive = "", -1.0
    for w in _words(session, text):
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
    from .__main__ import CHANNEL
    mind = session.mind
    thought, _ = think(session, topic, limit)
    symbols = _words(session, topic)
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
    open_pos = S.column(W_POS, IMAGINED) - S.column(C_POS, IMAGINED)
    open_neg = S.column(W_NEG, IMAGINED) - S.column(C_NEG, IMAGINED)
    pending = ((open_pos > 0) | (open_neg > 0)).nonzero().flatten()
    if pending.numel() == 0:
        return {"true": 0.0, "false": 0.0, "open": 0.0, "trust": float(mind.trust[IMAGINED])}
    real = mind.trust.clone()
    real[IMAGINED] = 0.0
    real[ACTED] = 0.0
    w_pos = (S.w(W_POS, pending) * real).sum(-1)
    w_neg = (S.w(W_NEG, pending) * real).sum(-1)
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
            out.append((float((S.w1(W_POS, tie) * real).sum()), float((S.w1(W_NEG, tie) * real).sum())))
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


def seen_with(mind, word: str, limit: int = 1) -> List[str]:
    """The images lived with a word (the image perceived in the moment before the word: an EXPECT tie from an image
    gist on the eye channel to the word), most supported first."""
    from .__main__ import CHANNEL
    from .gorme import EYE
    from .store import EXPECT, W_POS
    S = mind.S
    if not mind.ch.has(EYE):
        return []
    eye = mind.ch.id(EYE)
    unit = _unit(mind, CHANNEL, word)
    if unit < 0:
        return []
    found = {}
    for tie in S.inc.get((unit, EXPECT), ()):
        src = int(S.t_src[tie])
        if int(S.u_channel[src]) == eye:
            found[src] = found.get(src, 0.0) + float((S.w1(W_POS, tie) * mind.trust).sum())
    best = sorted(found, key=lambda u: -found[u])[:limit]
    return [str(mind.symbol_of(u)[1]) for u in best if found[u] > 0.0]


def dream(session, topic: str, links: int = 3, frequent: set = None) -> List[dict]:
    """One dream: a chain of imaginings, each carried on from the word of the last one the mind is least sure of
    (free association). In each, what the mind has seen with the dreamt word is seen again (the image re-lived as
    imagined), and the imagining is kept with its claims (remember_dream). Returns the dream's records."""
    from .__main__ import CHANNEL
    from .gorme import EYE
    mind = session.mind
    check = getattr(session, "check", None)              # a dream stops between its imaginings when asked (Ctrl+C)
    records, visited = [], set()
    for _ in range(links):
        if not topic or topic in visited:
            break
        if check is not None:
            check()
        visited.add(topic)
        seen = seen_with(mind, topic)
        for gist in seen:                                  # it sees in its dream what it saw with this word, as it
            mind.step([Event(EYE, gist, IMAGINED)])        # saw it: the image first, then the word comes to mind
        kind = "benzetme"                                  # analogy: a thought carried to a counterpart
        thought, construction, reality, other = imagine(session, topic)
        if construction:
            record = remember_dream(mind, topic, other, construction, round(reality, 2))
        elif thought:
            kind = "düşünce"
            record = remember_dream(mind, topic, "", [topic] + thought, None)
        else:
            break
        record["kind"] = kind
        record["seen"] = seen
        records.append(record)
        rest = [w for w in record["words"][1:] if not frequent or w not in frequent]
        topic = topic_of(session, " ".join(rest))          # what this imagining leaves it least sure of
    return records


def tau(mind, unit: int) -> float:
    """How telling a unit is by the mind's own memory: 1 / (1 + the number of composed contexts holding it), the
    weight the core gives a part when it calls up lived situations (read.py, F23). A word every context holds tells
    little about any one of them; one that few contexts hold is particular. In a speaking mind the composed contexts
    are the utterance so far, so a word late in an utterance is held by fewer of them than one early in it, and the
    last words of sentences weigh most: what the memory holds, stated here, not corrected."""
    from .store import PART
    return 1.0 / (1 + len(mind.S.inc.get((unit, PART), ())))


def day_residue(session, n: int, entries: int = 12, leave_out=()) -> List[str]:
    """What the day was about: the words most characteristic of the session's recent history entries relative to the
    mind, by its own evidence: how often a word was lived in the last ``entries`` entries (heard or said: a line, a
    chat, a dialogue watched, a whole file read or taught, Entry.counts) times tau(u) = 1 / (1 + the number of composed
    contexts holding it). A word the day repeated and few contexts hold is what it was about; one every context holds
    is not. The same measure for every kind of entry; no word-shape rule (what a word looks like decides nothing).
    ``leave_out``: words not offered (sleep leaves out the frequent ones)."""
    from .__main__ import CHANNEL
    mind = session.mind
    if not mind.ch.has(CHANNEL):
        return []
    cid = mind.ch.id(CHANNEL)
    counts: Dict[str, int] = {}
    for entry in reversed(session.history[-entries:] if entries else []):      # the latest first: ties go to it
        for w, c in getattr(entry, "counts", {}).items():
            counts[w] = counts.get(w, 0) + c
    scored = []
    for w, c in counts.items():
        if w in leave_out:
            continue
        index = mind.ch.symbol_index(cid, w, create=False)
        unit = mind.S.atom_index.get((cid, index), -1) if index >= 0 else -1
        if unit >= 0:
            scored.append((c * tau(mind, unit), w))
    scored.sort(key=lambda sw: -sw[0])                     # stable: equal weights keep the latest entry's order
    return [w for _, w in scored[:n]]


def sleep(session, cycles: int = 3) -> Dict[str, object]:
    """Sleep: dreams, then sorting. The mind dreams about what the day was about (day_residue) and about what it is
    unsure of (words it uses but holds with little conviction, wanted_topics), in that order, until ``cycles`` dreams
    were dreamt or no topic is left: a topic it can think nothing after (a word that only ends what it heard, such as
    the last word of a sentence) gives no dream, and the next one is tried. Each dream is a chain of imaginings carried
    on by free association, and what it saw with a dreamt word it sees again. Then everything imagined is sorted
    against what is real: the evidence on its trust in its own imagination (sort_dreams), and each kept dream's
    verdict (true, false, still open). Returns the night's dreams (each a list of imaginings), the verdicts of the
    dreams reality has judged, and the sorting."""
    from .__main__ import CHANNEL
    from .web import wanted_topics
    mind = session.mind
    check = getattr(session, "check", None)              # stopped between dreams (Ctrl+C): the night is not sorted
    frequent = frequent_words(mind, relative=True)                              # a small mind keeps topics too
    topics = day_residue(session, cycles, leave_out=frequent)                   # what the day was about, then what
    for w in wanted_topics(mind, CHANNEL, cycles * 2):                         # it is unsure of
        if w not in topics and w not in frequent:
            topics.append(w)
    before_night = len(getattr(mind, "dreams", None) or [])
    dreams = []
    for topic in topics:
        if len(dreams) >= cycles:
            break
        if check is not None:
            check()
        mind.step([])                                      # each dream begins in silence
        chain = dream(session, topic, frequent=frequent)
        if chain:
            dreams.append(chain)
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
