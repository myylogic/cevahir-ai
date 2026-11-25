# -*- coding: utf-8 -*-
"""Thinking: the inner voice with a measured sense of reality (LAYERS.md L3, design section 6).

The mind brings a topic to mind and speaks inwardly with the very machinery it speaks with (the same word units, the
same reading: F11/F12 starts, F16 continuation), but nothing leaves it: each word it says inwardly comes back as its
own echo with provenance "imagined". Every step is scored for reality before it is taken: the expectation of that
word given the present inner context, read from all evidence EXCEPT what was imagined (what was lived, read, told
supports it or not). The thought is learned like any experience, in the class "imagined", so trust in its own
constructions is learned when sensed or told evidence later lands on the same ties (learn.judge).

A dream is a chain of such thoughts: no word is put into the mind's inner voice but the topic, and no reasoning form
is coded (the analogy that replayed a thought about X on its counterpart Y and the inference by a shared root are
gone); what the inner voice reasons with is what the core reads with, the mind's own generalisations. Every topic is
chosen by one measure of the mind's evidence (topic_score). The online life's clock lets the mind think while nobody
talks to it (an innate rhythm, see Life in __main__); a thought is never said for it.
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


def think(session, topic: str, limit: int = 15, before: List[Event] = (), reality: bool = True) -> Tuple[List[str], float]:
    """Think about ``topic`` inwardly. Returns the inner words and the reality score R (mean over steps); with
    ``reality`` False R is not read (0.0): it is a second reading of every step, by another trust, that what is
    thought does not depend on (an answer that is only wanted for its words need not pay for it).

    The topic is brought to mind (imagined, not heard): a topic of several words is taken in word by word, the mind
    deciding nothing until its last word, as it takes in what it hears; ``before``: what comes to mind in a moment
    before the topic, taken in the same way (a dream sees again what it saw with the word, then the word). From then on
    the inner voice is the mind's own decisions, each lived as imagined (go_on): it ends when the mind decides nothing,
    or when it comes round to a composed context this episode already lived; ``limit`` words are only a safety cap."""
    from .__main__ import CHANNEL
    mind = session.mind
    symbols = _words(session, topic)
    if not symbols:
        return [], 0.0
    moments = ([list(before)] if before else []) + [[Event(CHANNEL, symbol, IMAGINED)] for symbol in symbols]
    acts = take_in(mind, moments)                          # the topic is brought to mind
    thought, scores = [], []

    def heard(words_: List[str]) -> None:                  # each word as it is decided, before its echo comes back
        for word in words_:
            unit = mind.unit_of(CHANNEL, word)
            if reality:
                scores.append(_reality(mind, unit) if unit >= 0 else 0.5)
            thought.append(word)
    go_on(mind, acts, heard, limit, imagined=True)
    mind.step([])                                          # the thought ends; the episode closes (a silent moment:
    score = sum(scores) / len(scores) if scores else 0.0   # no context, so nothing is decided in it)
    return thought, score


def goes_to_an_end(mind, context: int, limit: int = 60):
    """F47: whether the mind, brought back to a lived situation, can say what follows it to the end. The items of
    ``context`` come to mind again one after another on a copy of the mind (imagined; nothing of it is lived), and
    from the last of them the copy goes on by its own decisions (the inner voice). True when it said something and,
    where it stopped, the end of a turn is what it expects (over 1/2): it stopped because it was done. False when it
    said nothing, or stopped where no end is expected: it does not know how to go on. None when this cannot be told
    (a mind that never lived the end of a turn, or a situation that is not made of single items)."""
    from .__main__ import TURN, TURN_END
    from .form import parts_of
    from .store import ATOMIC
    end = mind.unit_of(TURN, TURN_END)
    if end < 0 or context is None:
        return None
    v = mind.S.arrays()
    moments = []
    for part in parts_of(mind, context):
        if not v.kind[part] & ATOMIC:
            return None
        channel = int(v.channel[part])
        moments.append([Event(mind.ch.names[channel], mind.ch.symbol(channel, int(v.symbol[part])), IMAGINED)])
    if not moments:
        return None
    copy = mind.clone()
    copy.__dict__["_settling"] = True                      # (whether a claim is still open is what is being found out:
    copy.step([])                                          # act.holds_back does not go by what was open before)
    acts = take_in(copy, moments)                          # from rest: the situation begins a run of its own
    said: List[str] = []
    go_on(copy, acts, said.extend, limit, imagined=True)
    return bool(said) and copy.predicted.get(end, 0.5) > 0.5


def moments_of(mind, context: int) -> List[List[Event]]:
    """The items of a lived situation as moments that come to mind again (imagined), one item a moment; [] when the
    situation is not made of single items. A single item is a situation of one moment."""
    from .form import parts_of
    from .store import ATOMIC
    v = mind.S.arrays()
    moments = []
    for part in parts_of(mind, context) or (context,):
        if not v.kind[part] & ATOMIC:
            return []
        channel = int(v.channel[part])
        moments.append([Event(mind.ch.names[channel], mind.ch.symbol(channel, int(v.symbol[part])), IMAGINED)])
    return moments


def question_frames(mind) -> List[int]:
    """F51: the questions the mind knows, as frames: generalisations with one place whose last item is the end of a
    turn and whose instances were answered (the lived context of a member, with that end, went on with a word:
    someone's turn followed; a sentence that was read and left at that is not a question). The one answered of the
    most things first. Found by walking its units (done when it asks itself something, not every moment; kept
    while the mind's structure does not grow)."""
    from .__main__ import TURN, TURN_END
    from .form import parts_of
    from .read import kind_masks, mv_view, own_mass
    from .store import ABSTRACT, ATOMIC, CATEGORY, SLOTTED
    end = mind.unit_of(TURN, TURN_END)
    if end < 0:
        return []
    S = mind.S
    kept = mind.__dict__.get("_questions")
    if kept is not None and kept[0] == (S.n_units, S.n_ties):
        return kept[1]
    v = S.arrays()
    channels, act = mv_view(mind, "u_channel"), kind_masks(mind)[5]
    trust = mind.trust.clone()
    trust[IMAGINED] = 0.0                                  # (what it asked and answered inwardly was not answered)
    found = []
    for unit in range(S.n_units):
        if not v.ualive[unit] or not v.kind[unit] & SLOTTED:
            continue
        parts = parts_of(mind, unit)
        if len(parts) < 2 or parts[-1] != end:
            continue
        places = [j for j, p in enumerate(parts) if v.kind[p] & CATEGORY]
        if len(places) != 1 or not all(v.kind[p] & (ATOMIC | CATEGORY) for p in parts):
            continue
        j = places[0]
        answered = 0
        for tie in S.inc.get((parts[j], ABSTRACT), ()):
            lived = S.tuple_index.get(parts[:j] + (int(S.t_src[tie]),) + parts[j + 1:])
            if lived is not None and any(w_pos > 0.0 and act[channels[t]]
                                         for t, (w_pos, _) in own_mass(mind, lived, 1, trust).items()):
                answered += 1
        if answered:
            found.append((-answered, unit))
    out = [unit for _, unit in sorted(found)]
    mind.__dict__["_questions"] = ((S.n_units, S.n_ties), out)
    return out


def ask_itself(session, item: int, limit: int = 30):
    """F51: the mind asks itself about an item, in the question it knows best: the frame's items come to mind one
    after another with the item in the frame's place (imagined; nothing is said aloud), and from the end of the
    question the inner voice answers by the mind's own decisions. Returns (the question's words, the inner words,
    whether an unknown was found). An unknown: it said nothing, or the first word it said rests on what others share
    (``from_what_it_lived``'s test), or a relation of its frame named nothing for the item (F46 noted it in that
    very moment). The unknown is an open claim about the item at the end of the imagined question (written here when
    F46 did not write it), which the mind may look up by itself (F47); the item is kept (``Mind.wondered``) so that
    what the look shows settles what the asking was worth. With no question it knows, (None, [], False)."""
    from .__main__ import CHANNEL
    from .act import rests_on_others
    from .form import parts_of
    from .learn import note_unknown
    from .read import open_claim
    from .store import CATEGORY
    mind = session.mind
    frames = question_frames(mind)
    if not frames:
        return None, [], False
    v = mind.S.arrays()
    moments, asked = [], []
    for part in parts_of(mind, frames[0]):
        unit = item if v.kind[part] & CATEGORY else part
        channel = int(v.channel[unit])
        symbol = mind.ch.symbol(channel, int(v.symbol[unit]))
        moments.append([Event(mind.ch.names[channel], symbol, IMAGINED)])
        if mind.ch.names[channel] == CHANNEL:
            asked.append(str(symbol))
    symbol = mind.ch.symbol(int(v.channel[item]), int(v.symbol[item]))
    before = open_claim(mind, symbol)
    mind.step([])                                          # from rest: the question begins a run of its own
    acts = take_in(mind, moments)
    context = mind.context
    said: List[str] = []
    guessed = [False]

    def out(words_now: List[str]) -> None:
        made_now = mind.__dict__.get("_src_now") or {}
        for w in ([] if said else words_now):
            unit = mind.unit_of(CHANNEL, w)
            if mind.context is not None and rests_on_others(mind, mind.context, unit, made_now.get(unit)):
                guessed[0] = True
        said.extend(words_now)
    go_on(mind, acts, out, limit, imagined=True)
    mind.step([])                                          # the thought has ended
    unknown = not said or guessed[0] or open_claim(mind, symbol) > max(before, 0.5)
    if unknown:
        if not open_claim(mind, symbol) > max(before, 0.5) and context is not None:
            note_unknown(mind, item, context)              # (F46 did not write it: a plain claim, at the question's end)
        mind.wondered[item] = True
    return asked, said, unknown


def from_what_it_lived(mind, context: int, limit: int = 60, word_only: bool = False):
    """F50: whether the mind, brought back to a lived situation with nowhere to look, says what follows it to the
    end, and says the word that follows *that situation* from what it lived of the thing asked about: as
    ``goes_to_an_end``, and the first word it decides there does not rest only on what others share (hata's chain:
    a word made by a way that is not ``act.grounded``, at a context that has no record of its own of it). The
    words after it are the shape of the answer and are not asked this. ``word_only``: the claim was about that next
    word only (the mind had a word to say there, resting on what others share): it is met when the word it now
    says there rests on what it lived of the thing, whether or not it can say the rest. True, False, or None when
    it cannot be told. On a copy: nothing is lived."""
    from .__main__ import CHANNEL, TURN, TURN_END
    from .act import rests_on_others
    end = mind.unit_of(TURN, TURN_END)
    if end < 0 or context is None:
        return None
    moments = moments_of(mind, context)
    if not moments:
        return None
    copy = mind.clone()
    copy.__dict__["can_look"] = False
    copy.__dict__["_settling"] = True
    copy.step([])                                          # from rest: the situation begins a run of its own
    acts = take_in(copy, moments)
    said: List[str] = []
    guessed = [False]

    def out(words_now: List[str]) -> None:
        made_now = copy.__dict__.get("_src_now") or {}
        for w in ([] if said else words_now):              # only the word that follows the situation itself
            unit = copy.unit_of(CHANNEL, w)
            if copy.context is not None and rests_on_others(copy, copy.context, unit, made_now.get(unit)):
                guessed[0] = True
        said.extend(words_now)
    go_on(copy, acts, out, limit, imagined=True)
    if word_only:
        return bool(said) and not guessed[0]
    return bool(said) and copy.predicted.get(end, 0.5) > 0.5 and not guessed[0]


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


def tau(mind, unit: int) -> float:
    """How telling a unit is by the mind's own memory: 1 / (1 + the number of composed contexts holding it), the
    weight the core gives a part when it calls up lived situations (read.py, F23). A word every context holds tells
    little about any one of them; one that few contexts hold is particular. In a speaking mind the composed contexts
    are the utterance so far, so a word late in an utterance is held by fewer of them than one early in it: what the
    memory holds, stated here, not corrected."""
    from .store import PART
    return 1.0 / (1 + len(mind.S.inc.get((unit, PART), ())))


def topic_score(mind, unit: int) -> float:
    """How good a topic a symbol is, by the mind's own evidence: tau(u) x D(u). tau is how telling it is (above); the
    drive D(u) = the sum over its lag-1 ties of use x (1 - conviction) is how little is known of what follows it
    (read.drive_of). A good topic is telling and not yet well known; a symbol nothing ever followed scores 0 (there
    is nothing to think after it). The one measure for every topic the program chooses: what to think about, what
    the day was about, what to dream and what to look up. No word-shape rule (what a word looks like decides
    nothing) and no frequency cut-off (frequent words are held by many contexts, so their tau is small)."""
    from .read import drive_of
    return tau(mind, unit) * drive_of(mind, unit)


def topic_of(session, text: str = None) -> str:
    """What to think about: the symbol of an utterance that is the best topic by the mind's own evidence (topic_score,
    the first of equals), among the symbols it has met; "" when none scores above 0. By default the last utterance it
    heard in the session, whatever the history entry it belongs to (a line told, a chat, a dialogue watched, the last
    sentence of a file read or taught; Session.last_heard)."""
    from .__main__ import CHANNEL
    mind = session.mind
    if text is None:
        last_heard = getattr(session, "last_heard", None)
        text = last_heard() if last_heard is not None else ""
    if not text:
        return ""
    best, best_score = "", 0.0
    for w in dict.fromkeys(_words(session, text)):
        unit = mind.unit_of(CHANNEL, w)
        score = topic_score(mind, unit) if unit >= 0 else 0.0
        if score > best_score:
            best, best_score = w, score
    return best


def _unit(mind, channel_name: str, symbol: str) -> int:
    cid = mind.ch.id(channel_name)
    index = mind.ch.symbol_index(cid, symbol, create=False)
    return mind.S.atom_index.get((cid, index), -1) if index >= 0 else -1


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


DREAM_CAP = 12        # imaginings in one dream at most: only a safety cap (a dream ends when the mind's race falls silent)


def dream(session, topic: str, cap: int = DREAM_CAP) -> List[dict]:
    """One dream: a chain of imaginings, each the mind's own inner voice about a topic (think: what it decides, lived as
    imagined; only the topic is brought to mind), after what it saw with that word is seen again (the image re-lived
    as imagined, the moment before the word, as it was lived). The next topic is the best topic (topic_score) of what
    this imagining said that the dream has not dreamt of yet. The chain ends when the mind's race falls silent (it
    thinks nothing after the topic), when no topic is left, or at the safety cap. Each imagining is kept with its
    claims (remember_dream). Returns the dream's records."""
    from .__main__ import CHANNEL
    from .gorme import EYE
    mind = session.mind
    check = getattr(session, "check", None)              # a dream stops between its imaginings when asked (Ctrl+C)
    records, visited = [], set()
    while topic and topic not in visited and len(records) < cap:
        if check is not None:
            check()
        visited.add(topic)
        seen = seen_with(mind, topic)
        thought, reality = think(session, topic, before=[Event(EYE, gist, IMAGINED) for gist in seen])
        if not thought:                                    # its race fell silent: the dream ends
            break
        record = remember_dream(mind, topic, "", [topic] + thought, round(reality, 2))
        record["kind"] = "düşünce"
        record["seen"] = seen
        records.append(record)
        topic = topic_of(session, " ".join(w for w in thought if w not in visited))
    return records


def day_residue(session, n: int, entries: int = 12) -> List[str]:
    """What the day was about: the words of the session's recent history entries that are the best topics, weighted by
    how often the day lived them: how often a word was lived in the last ``entries`` entries (heard or said: a line, a
    chat, a dialogue watched, a whole file read or taught, Entry.counts) times topic_score(u) = tau(u) x D(u). A word
    the day repeated, few contexts hold and the mind does not yet know well is what it was about; a word nothing ever
    followed (the last word of every sentence it ended) scores 0 and is not offered. The same measure for every kind
    of entry; no word-shape rule and no frequency cut-off."""
    from .__main__ import CHANNEL
    mind = session.mind
    if not mind.ch.has(CHANNEL):
        return []
    counts: Dict[str, int] = {}
    for entry in reversed(session.history[-entries:] if entries else []):      # the latest first: ties go to it
        for w, c in getattr(entry, "counts", {}).items():
            counts[w] = counts.get(w, 0) + c
    scored = []
    for w, c in counts.items():
        unit = mind.unit_of(CHANNEL, w)
        score = c * topic_score(mind, unit) if unit >= 0 else 0.0
        if score > 0.0:
            scored.append((score, w))
    scored.sort(key=lambda sw: -sw[0])                     # stable: equal weights keep the latest entry's order
    return [w for _, w in scored[:n]]


def sleep(session, cycles: int = 3) -> Dict[str, object]:
    """Sleep: dreams, then sorting. The mind dreams about what the day was about (day_residue) and about what it most
    wants to know of all it holds (wanted_topics), both by the one measure topic_score (telling and not yet well
    known), in that order, until ``cycles`` dreams were dreamt or no topic is left: a topic it can think nothing after
    gives no dream, and the next one is tried. Each dream is a chain of imaginings carried
    on by free association, and what it saw with a dreamt word it sees again. Then everything imagined is sorted
    against what is real: the evidence on its trust in its own imagination (sort_dreams), and each kept dream's
    verdict (true, false, still open). Returns the night's dreams (each a list of imaginings), the verdicts of the
    dreams reality has judged, and the sorting."""
    from .__main__ import CHANNEL
    from .web import wanted_topics
    mind = session.mind
    check = getattr(session, "check", None)              # stopped between dreams (Ctrl+C): the night is not sorted
    topics = day_residue(session, cycles)                  # what the day was about, then what it wants to know
    for w in wanted_topics(mind, CHANNEL, cycles * 2):     # (the same measure, topic_score, over all it holds)
        if w not in topics:
            topics.append(w)
    before_night = len(getattr(mind, "dreams", None) or [])
    dreams = []
    for topic in topics:
        if len(dreams) >= cycles:
            break
        if check is not None:
            check()
        mind.step([])                                      # each dream begins in silence
        chain = dream(session, topic)
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
    mind.consolidate()                                 # the day's ties into the long-term store, the caches dropped
    return {"dreams": dreams, "judged": judged, "sorting": sorting}
