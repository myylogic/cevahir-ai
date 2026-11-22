# -*- coding: utf-8 -*-
"""Wholes (F43): an utterance held as parts and answered part by part; a first working memory. Told Apart (F45): what
a word never said alone answers with.

The mind decides every word from its last F items (its window). An answer longer than the window is said with the
question out of sight, and a long question does not fit the window at all. What holds more than the window is not a
longer window but larger items: a sequence known as a whole stands as one item, and so does what it was answered with.
Nothing here is told to the mind; all of it is read from what it lived.

* A whole. A sequence that was said alone and answered is a whole. Every dialogue the mind watches is lived once more
  as two items, the utterance and its answer (``said`` symbols), so what a whole was answered with is one lived
  continuation, however long it is (``lived_reply``).
* A summary. When the answer of a watched dialogue is made of the replies of wholes that stand in the utterance, the
  dialogue is lived once more in summary (``align``): each such whole as its form (its own summary, the wholes inside
  it as bare wholes: the form says how it is built, not of what; a whole in which no whole stands is a bare whole),
  each reply as a pointer to the place of its whole, the other words as they are. Of the ways an answer can be read
  so, the one that leaves the fewest words unexplained is taken, then the one with the fewest items. When two tie
  and every whole of the one lies inside a whole of the other, they are one reading at two grains and the larger
  wholes are taken ("turn left" and the "left" in it answer alike); when their wholes cross, they are two readings
  and nothing is concluded. A summary is one the mind can hold: it fits the window. It is lived once more as its
  construction, every whole bare: what is answered to a construction does not depend on the forms of its wholes, so
  a summary whose forms never met in it is answered as its construction was (``_heard_as``).
* Answering in parts. An utterance answered before is remembered whole (heard as the one item it was lived as). An
  utterance never answered is heard in its summary (``reading``): the one that was lived and answered as it is (the
  summary's context has lived continuations of its own); when none was, the one that was lived and answered as its
  construction, every item of it lived before (its words and the forms of its wholes, in some summary: what was
  never lived as an item is not known to stand as one). A whole inside it is a sequence answered before, or a
  sequence never met that has such a summary of its own (a whole by its form). When the mind decides a pointer,
  it says the reply of that part: it sets its working state aside, answers the part as if it had been said alone
  (what it remembers of a part answered before; a new part in its own summary, or word by word) and takes its working
  state up again where it left it. The echo of all it said meanwhile is one item, the pointer.
* Told apart (F45, ``Config.told_apart``). A word that was never said alone has no reply of its own, and stands in a
  summary as the word it is: what was lived with "left" says nothing of "right" in the same place. When two lived
  dialogues differ in one word of the utterance and their answers, of one length, differ in one pair of words wherever
  they differ, each of the two words answers with its own (the method of difference): that is lived as a dialogue of
  its own, inferred (``tell_apart``), and the word is a whole from then on. Only dialogues the mind can hold are
  compared. A word that was itself said alone and answered keeps the reply it was given. The residue: the difference
  is assigned only when what the two answers share is what the shared words answer with, and a word of the answer
  that a shared word answers with is not the differing word's share. A word told apart in several ways, none of
  them most of the evidence, has no reply.

The window bounds each level, not the answer: "jump around left thrice and walk twice" is five items and its answer
two, and each of the two is answered in a few items of its own.

A summary is another level, and it is lived on a channel of its own (``LEVEL``), with an end of the turn of its own:
its wholes and pointers, and the words that stand in it too. No item of a summary is an item of the words' level, so
no frame, category or resemblance of the words' level holds a summary, and what the mind reads word by word is what
it was before any summary was lived.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

from .events import ACTED, IMAGINED, INFERRED, SENSED, TOLD, Event
from .evidence import expectation
from .form import mv_view

SAID = "said"                   # ("said", w1, ...): a sequence lived as one item (an utterance, or what answered it)
WHOLE = ("whole",)              # a bare whole; ("whole", i1, ...): the form of a whole (its own summary)
REPLY = "reply"                 # ("reply", j): "the reply of the whole at place j of the utterance"
LEVEL = "ozet"                  # the channel of the summaries' items: wholes, pointers, and the words that stand in them
LEVEL_END = "⏎⏎"                # the end of a turn as a summary lives it (on the turn channel)
DEPTH = 8                       # how deep a part may be answered in parts (a safety cap)
CHOICES = 4096                  # how many choices of wholes are read for one utterance (a safety cap)
# the working state of the present episode (Mind._TRANSIENT without the clock, the knobs and what is not the episode's)
WORKING = ("focus", "last_focus", "gens", "last_gens", "carry", "episode", "trace", "context", "pending", "predicted",
           "act_hope", "hoping", "chunked", "idle", "prov_of", "unknown", "voice", "witnesses", "episode_start",
           "topic")


def said(words) -> tuple:
    return (SAID,) + tuple(words)


def is_said(symbol) -> bool:
    return isinstance(symbol, tuple) and symbol[:1] == (SAID,)


def is_whole(symbol) -> bool:
    return isinstance(symbol, tuple) and symbol[:1] == WHOLE


def is_pointer(symbol) -> bool:
    return isinstance(symbol, tuple) and len(symbol) == 2 and symbol[0] == REPLY


def show(symbol) -> str:
    """A symbol of the text channel as the chat shows it (``!neden``): a word as it is; a sequence held as one item in
    quotes; a pointer as the place it points to; a whole as its form."""
    if is_pointer(symbol):
        return "↪ %d. yerdeki bütünün cevabı" % (symbol[1] + 1)
    if is_said(symbol):
        return "«%s»" % " ".join(str(w) for w in symbol[1:])
    if is_whole(symbol):
        return "‹%s›" % (" ".join(show(x) for x in symbol[1:]) or "bütün")
    return str(symbol)


# ---------------------------------------------------------------------------------------------- reading what was lived
def went_on(mind, symbols, without=()) -> Dict[object, Tuple[float, float]]:
    """What the lived summary "these items, then the end of the turn" went on with, {symbol: (W+, W-)}: what was
    lived, not the mind's own acts and not what it imagined (``without``: nor these classes). Empty when no such
    summary was lived."""
    from .__main__ import TURN
    from .read import own_mass
    end = mind.unit_of(TURN, LEVEL_END)
    units = [mind.unit_of(LEVEL, x) for x in symbols]
    if end < 0 or any(u < 0 for u in units):
        return {}
    context = mind.S.tuple_index.get(tuple(units) + (end,))
    if context is None:
        return {}
    trust = mind.trust.clone()
    trust[ACTED] = 0.0
    trust[IMAGINED] = 0.0
    for prov in without:
        trust[prov] = 0.0
    level, channels = mind.ch.id(LEVEL), mv_view(mind, "u_channel")
    return {mind.symbol_of(v)[1]: (p, n) for v, (p, n) in own_mass(mind, context, 1, trust).items()
            if channels[v] == level and p > 0.0}


def lived_reply(mind, words) -> Optional[tuple]:
    """What a sequence, said alone, was answered with (its words), or None: the strongest lived continuation of the
    dialogue lived as two items, when it is expected over 1/2. F45: of a single word, when it is also most of what
    the word was answered with (a word told apart in several ways, none of them most of the evidence, has no reply)."""
    best, total = None, 0.0
    for symbol, (p, n) in went_on(mind, [said(words)]).items():
        if not is_said(symbol):
            continue
        total += p
        if expectation(p, n, mind.cfg.k) > 0.5 and (best is None or p > best[0]):
            best = (p, symbol[1:])
    if best is None or (len(words) == 1 and getattr(mind.cfg, "told_apart", False) and 2.0 * best[0] <= total):
        return None
    return best[1]


def _reply(mind, words, memo) -> Optional[tuple]:
    key = ("reply", tuple(words))
    if key not in memo:
        memo[key] = lived_reply(mind, words)
    return memo[key]


def _spans(n: int) -> List[Tuple[int, int]]:
    """Every proper run of places of an utterance of n items."""
    return [(i, j) for i in range(n) for j in range(i + 1, n + 1) if j - i < n]


def _disjoint(spans) -> List[List[Tuple[int, int]]]:
    """Every non-empty choice of runs no two of which overlap (at most ``CHOICES`` of them: a bound on effort for an
    utterance in which very many wholes stand)."""
    spans = sorted(spans)
    out: List[List[Tuple[int, int]]] = []

    def grow(start: int, chosen: list, free: int) -> None:
        for index in range(start, len(spans)):
            if len(out) >= CHOICES:
                return
            i, j = spans[index]
            if i >= free:
                now = chosen + [spans[index]]
                out.append(now)
                grow(index + 1, now, j)
    grow(0, [], 0)
    return out


def _cover(answer: tuple, replies: Dict[Tuple[int, int], tuple]):
    """The cheapest reading of an answer as replies of the given runs and words of its own: ((words left unexplained,
    items), the pieces in order (a run, or a word), whether it is the only reading that cheap), or None."""
    m = len(answer)
    far = (1 << 30, 1 << 30)
    best, ways, back = [far] * (m + 1), [0] * (m + 1), [None] * (m + 1)
    best[0], ways[0] = (0, 0), 1
    for k in range(1, m + 1):
        options = []
        if best[k - 1] != far:
            options.append(((best[k - 1][0] + 1, best[k - 1][1] + 1), k - 1, None))
        for span, reply in replies.items():
            start = k - len(reply)
            if start >= 0 and best[start] != far and answer[start:k] == reply:
                options.append(((best[start][0], best[start][1] + 1), start, span))
        if not options:
            continue
        best[k] = min(option[0] for option in options)
        top = [option for option in options if option[0] == best[k]]
        ways[k] = min(2, sum(ways[option[1]] for option in top))
        back[k] = (top[0][1], top[0][2])
    if best[m] == far:
        return None
    pieces, k = [], m
    while k > 0:
        k_before, span = back[k]
        pieces.append(span if span is not None else answer[k - 1])
        k = k_before
    return best[m], pieces[::-1], ways[m] == 1


def _summary(words: tuple, chosen, forms) -> Tuple[tuple, Dict[int, Tuple[int, int]]]:
    """An utterance with the chosen runs as their forms: (items, {place of a whole: its run})."""
    starts = {span[0]: span for span in chosen}
    items, where, i = [], {}, 0
    while i < len(words):
        span = starts.get(i)
        if span is None:
            items.append(words[i])
            i += 1
        else:
            where[len(items)] = span
            items.append(forms[span])
            i = span[1]
    return tuple(items), where


def form(mind, words: tuple, reply: Optional[tuple], F: int, memo: dict) -> tuple:
    """The form of a whole: its own summary with the wholes inside it as bare wholes (how it is built, not of what);
    a bare whole when it has no summary (one item, or a sequence in which no whole stands: nothing is known of how it
    is built but that it is a whole). ``reply``: what it was answered with (its summary is read from that), None for
    a sequence never answered (its summary is the one it would be heard in)."""
    if len(words) == 1:
        return WHOLE
    key = ("form", words, reply)
    if key not in memo:
        inner = align(mind, words, reply, F, memo) if reply is not None else reading(mind, words, F, memo)
        memo[key] = WHOLE if inner is None else WHOLE + tuple(WHOLE if is_whole(x) else x for x in inner[0])
    return memo[key]


def align(mind, words: tuple, answer: tuple, F: int, memo: dict):
    """The summary of a lived dialogue: (the utterance's items, the answer's items, {place: run}), or None.
    A proper run of the utterance that was said alone and answered before is a whole; the answer is read as the
    replies of some of them (no two overlapping, every one of them pointed to) and words of its own. The reading with
    the fewest words left unexplained, then the fewest items; none when two readings tie, or when the summary does not
    fit the window."""
    replies = {}
    for span in _spans(len(words)):
        reply = _reply(mind, words[span[0]:span[1]], memo)
        if reply is not None and any(answer[k:k + len(reply)] == reply for k in range(len(answer) - len(reply) + 1)):
            replies[span] = reply
    found = []
    for chosen in _disjoint(replies):
        cover = _cover(answer, {span: replies[span] for span in chosen})
        if cover is None:
            continue
        cost, pieces, only = cover
        if {p for p in pieces if isinstance(p, tuple)} != set(chosen):
            continue                                     # a whole not pointed to stays the words it is
        found.append((cost, only, chosen, pieces))
    if not found:
        return None
    cheapest = min(reading_[0] for reading_ in found)
    top = [reading_ for reading_ in found if reading_[0] == cheapest]
    if len(top) > 1:                                     # the same reading at two grains: the larger wholes
        def holds(coarse, fine) -> bool:
            return all(any(i <= a and b <= j for i, j in coarse[2]) for a, b in fine[2])
        coarsest = [r for r in top if all(holds(r, other) for other in top)]
        if len(coarsest) == 1:
            top = coarsest
    if len(top) != 1 or not top[0][1]:
        return None                                      # two readings tie: nothing is concluded
    _, _, chosen, pieces = top[0]
    forms = {span: form(mind, words[span[0]:span[1]], replies[span], F, memo) for span in chosen}
    items, where = _summary(words, chosen, forms)
    place = {span: j for j, span in where.items()}
    answered = tuple((REPLY, place[p]) if isinstance(p, tuple) else p for p in pieces)
    if len(items) + 1 + len(answered) > F:
        return None                                      # a summary is one the mind can hold
    return items, answered, where


def reading(mind, words: tuple, F: int, memo: dict):
    """The summary an utterance never answered is heard in: (items, {place: run}), or None.
    A proper run of it is a whole when it was said alone and answered, or (never met) when it has such a summary of
    its own. Of the summaries these wholes give, the one whose context was lived and went on with the most evidence
    (expected over 1/2) is taken, the shorter of equals; none when two different ones tie. When none was lived as it
    is: the same of the summaries whose construction was lived so, every item of the summary lived before (its words
    and the forms of its wholes, in some summary)."""
    key = ("reading", words)
    if key in memo:
        return memo[key]
    memo[key] = None                                     # (a run is never read through itself)
    forms = {}
    for span in _spans(len(words)):
        part = words[span[0]:span[1]]
        reply = _reply(mind, part, memo)
        if reply is not None:
            forms[span] = form(mind, part, reply, F, memo)
        elif len(part) > 1 and reading(mind, part, F, memo) is not None:
            forms[span] = form(mind, part, None, F, memo)
    best, tie, unlived = None, False, []

    def weigh(items: tuple, where, lived_as: tuple) -> bool:
        """Take a summary by how its lived context went on (the most evidence, the shorter of equals)."""
        nonlocal best, tie
        lived = [p for p, n in went_on(mind, lived_as).values() if expectation(p, n, mind.cfg.k) > 0.5]
        if not lived:
            return False
        rank = (max(lived), -len(items))
        if best is None or rank > best[0]:
            best, tie = (rank, items, where), False
        elif rank == best[0] and items != best[1]:
            tie = True
        return True
    for items, where in _summaries(mind, words, forms, F):
        if not weigh(items, where, items):
            unlived.append((items, where))
    if best is None:                                     # none was lived as it is: as its construction
        for items, where in unlived:
            bare = construction(items)
            if bare != items and all(mind.unit_of(LEVEL, x) >= 0 for x in items):
                weigh(items, where, bare)
    memo[key] = None if best is None or tie else (best[1], best[2])
    return memo[key]


def _summaries(mind, words: tuple, forms: Dict[Tuple[int, int], tuple], F: int):
    """Every summary of an utterance by the given wholes (at least one of them) that the mind can hold and whose
    construction may have been lived: (items, {place: run}). A summary lived as it is was lived as its construction
    too (``consolidate``), and every beginning of a lived sequence is a lived sequence: a choice of wholes is
    followed only while its construction so far was lived."""
    lived, starts = mind.S.tuple_index, {}
    for (i, j), shape in forms.items():
        starts.setdefault(i, []).append((j, shape))
    bare, out = mind.unit_of(LEVEL, WHOLE), []

    def grow(i: int, items: tuple, where: dict, units: tuple) -> None:
        if i == len(words):
            if where:
                out.append((items, dict(where)))
            return
        if len(items) + 3 > F:                           # (one more item, its end, and an answer)
            return
        for j, item, unit, whole in [(i + 1, words[i], mind.unit_of(LEVEL, words[i]), False)] + [
                (j, shape, bare, True) for j, shape in sorted(starts.get(i, ()), key=lambda found: found[0])]:
            now = units + (unit,)
            if unit < 0 or (len(now) > 1 and now not in lived):
                continue
            if whole:
                where[len(items)] = (i, j)
            grow(j, items + (item,), where, now)
            if whole:
                del where[len(items)]
    grow(0, (), {}, ())
    return out


# ------------------------------------------------------------------------------------------------------------- living
def _hear(s, items, prov: int, speaker: str = None) -> None:
    """The items of a summary, one a moment, while the mind does not decide (as the session hears). A summary is
    lived on a channel of its own, the words that stand in it too: no item of it is an item of the words' level, so
    what the mind reads word by word is what it was."""
    from .__main__ import PERSON
    mind = s.mind
    mind.add_channel(LEVEL, "act")
    theta = mind.knob["theta"]
    mind.knob["theta"] = 1e9
    try:
        for symbol in items:
            mind.step([Event(LEVEL, symbol, prov)] + ([Event(PERSON, speaker, SENSED)] if speaker else []))
    finally:
        mind.knob["theta"] = theta


def _end(s, prov: int, tone: str, listening: bool = True):
    """The end of a turn as a summary lives it: a perceived event of its own, with how it ended as a facet (as
    Session.end_of_turn); what the mind decides in that moment."""
    from .__main__ import TONE, TURN
    mind = s.mind
    theta = mind.knob["theta"]
    if listening:
        mind.knob["theta"] = 1e9
    try:
        events = [Event(TURN, LEVEL_END, prov)]
        if tone and s.tones:
            events.append(Event(TONE, tone, prov))
        return mind.step(events)
    finally:
        mind.knob["theta"] = theta


def _live(s, first, answer, prov: int, asker: Optional[str], tones) -> None:
    """A dialogue lived in summary as the session watches one: the utterance, its end, the answer, its end, a pause.
    ``asker`` None: no one said it (a dialogue the mind inferred)."""
    if asker is not None:
        s.company(asker)
    _hear(s, first, prov, asker)
    _end(s, prov, tones[0])
    _hear(s, answer, prov)
    _end(s, prov, tones[1])
    s.mind.step([])


def construction(items: tuple) -> tuple:
    """A summary as its construction: its words as they are, every whole bare (which wholes stand where, whatever
    their forms)."""
    return tuple(WHOLE if is_whole(x) else x for x in items)


def _heard_as(mind, items: tuple) -> tuple:
    """How a summary is heard to be answered: as it is when it was lived and answered so (or when its construction
    was not); as its construction when only that was lived and answered. The forms of the wholes say which reading
    of an utterance is its summary (``reading``); what is answered to a construction does not depend on them, so a
    summary whose forms never met in it is answered as the lived summaries of the same construction were."""
    def lived(summary: tuple) -> bool:
        return any(expectation(p, n, mind.cfg.k) > 0.5 for p, n in went_on(mind, summary).values())
    bare = construction(items)
    return bare if bare != items and not lived(items) and lived(bare) else items


def _holes(mind) -> dict:
    """{(the words before, the words after): the words that stood between them}, over the sequences lived as one item
    that an utterance the mind can hold with its answer may be (read from the alphabet of the summaries' channel and
    kept as it grows: nothing but what the mind lived)."""
    symbols = mind.ch.symbols[mind.ch.id(LEVEL)]
    longest = int(mind.knob["F"]) - 2
    kept = mind.__dict__.get("_whole_holes")
    if kept is None or kept[0] > len(symbols) or kept[2] != longest:
        kept = mind.__dict__["_whole_holes"] = [0, {}, longest]
    for symbol in symbols[kept[0]:]:
        if is_said(symbol) and len(symbol) - 1 <= longest:
            words = symbol[1:]
            for i, word in enumerate(words):
                kept[1].setdefault((words[:i], words[i + 1:]), {})[word] = True
    kept[0] = len(symbols)
    return kept[1]


def said_alone(mind, word) -> bool:
    """Whether a word was itself said alone and answered (in what the mind lived, not in what it inferred)."""
    return any(is_said(x) and expectation(p, n, mind.cfg.k) > 0.5
               for x, (p, n) in went_on(mind, [said((word,))], without=(INFERRED,)).items())


def tell_apart(s, words: tuple, answered: tuple, tones) -> None:
    """F45 (Told Apart): the method of difference on two lived dialogues. When this dialogue and one lived before
    differ in one word of the utterance, and their answers (of one length) differ in one pair of words wherever they
    differ, each of the two words answers with the word its dialogue's answer has there: that is lived as a dialogue
    of its own (inferred), unless the word was itself said alone and answered. The residue: what the two answers
    share must be what the shared words answer with (until then the difference cannot be assigned), and a word of
    the answer that a shared word answers with is not the differing word's share (in "walk twice" and "walk left"
    the answers differ in "i_walk" and "i_lturn", and "i_walk" is walk's). Only dialogues the mind can hold are
    compared (the utterance, its end and the answer fit the window), and only utterances of several words: the two
    agree in all but one word."""
    mind = s.mind
    if len(words) < 2 or len(words) + 1 + len(answered) > mind.knob["F"] or not mind.ch.has(LEVEL):
        return
    holes = _holes(mind)
    for i, word in enumerate(words):
        rest, known = words[:i] + words[i + 1:], None
        for other in list(holes.get((words[:i], words[i + 1:]), ())):
            if other == word:
                continue
            before = lived_reply(mind, words[:i] + (other,) + words[i + 1:])
            if before is None or len(before) != len(answered):
                continue
            apart = {(x, y) for x, y in zip(answered, before) if x != y}
            if len(apart) != 1:
                continue
            if known is None:                            # what the words the two utterances share answer with
                known = {lived_reply(mind, (x,)) for x in rest}
            if any((x,) not in known for x, y in zip(answered, before) if x == y):
                continue                                 # what the two answers share is not accounted for yet
            (mine, its), = apart
            for w, a in ((word, mine), (other, its)):
                if (a,) in known or said_alone(mind, w):
                    continue                             # (the residue: what another word of the utterance answers
                _live(s, (said((w,)),), (said((a,)),), INFERRED, None, tones)    # with is not this word's share)


def consolidate(s, first: str, answer: str, prov: int, asker: str) -> None:
    """After a dialogue was watched: it is lived once more as two items (the utterance, its answer), and, when its
    answer is made of the replies of wholes that stand in the utterance, once more in that summary and once more as
    the summary's construction (every whole bare). F45: before that, what its words answer with is told apart."""
    mind = s.mind
    words, answered = tuple(s.body(first)), tuple(s.body(answer))
    if not words or not answered:
        return
    tones = (s.tone(first), s.tone(answer))
    if getattr(mind.cfg, "told_apart", False):
        tell_apart(s, words, answered, tones)
    summary = align(mind, words, answered, mind.knob["F"], {})
    _live(s, (said(words),), (said(answered),), prov, asker, tones)
    if summary is not None:
        _live(s, summary[0], summary[1], prov, asker, tones)
        if construction(summary[0]) != summary[0]:       # ... and once more as its construction
            _live(s, construction(summary[0]), summary[1], prov, asker, tones)


# ---------------------------------------------------------------------------------------------------------- answering
def set_aside(mind) -> dict:
    """The working state of the present episode is set aside (the mind is as after a pause; what it knows stays)."""
    held = {name: copy.copy(getattr(mind, name)) for name in WORKING}
    mind.focus, mind.last_focus, mind.gens, mind.last_gens = [], [], [], []
    mind.carry, mind.episode, mind.trace, mind.context = {}, [], [], None
    mind.pending, mind.predicted, mind.act_hope, mind.hoping = [], {}, {}, None
    mind.chunked, mind.prov_of, mind.unknown, mind.voice, mind.witnesses = [], {}, 0.0, {}, []
    mind.topic, mind.idle = None, 1
    return held


def take_up(mind, held: dict) -> None:
    """The working state set aside is taken up again where it was left."""
    for name, value in held.items():
        setattr(mind, name, value)


def reply_in_parts(s, text: str, limit: int, say=None) -> Optional[List[str]]:
    """Session.reply for an utterance of several words that is remembered as a whole or has a summary: the words the
    mind says, or None when it has neither (the usual way then). An utterance answered before, whose answer is
    expected over the speaking line, is heard as the one item it was lived as, and the answer the mind decides is said
    word by word (what was answered before is remembered whole, however long it was). An utterance never answered is
    heard in its summary; the mind's decisions are said as they are made, a pointer by answering its part
    (``_go_on``)."""
    mind = s.mind
    words = tuple(s.body(text))
    memo: dict = {}
    if len(words) < 2:
        return None
    if _reply(mind, words, memo) is not None:
        line = 0.5 + mind.knob["theta"] / (2.0 * mind.cfg.T)
        firm = max((expectation(p, n, mind.cfg.k) for x, (p, n) in went_on(mind, [said(words)]).items()
                    if is_said(x)), default=0.0)
        if firm <= line:
            return None
        found = ((said(words),), {})
    else:
        found = reading(mind, words, mind.knob["F"], memo)
        if found is None:
            return None
    spoken: List[str] = []

    def out(words_now: List[str]) -> None:
        spoken.extend(words_now)
        if say is not None:
            for w in words_now:
                say(w)
    mind.__dict__["_part_depth"] = 0
    try:
        _hear(s, _heard_as(mind, found[0]), TOLD, s.partner)
        acts = _end(s, TOLD, s.tone(text), listening=False)
        _go_on(s, acts, words, found[1], out, [limit], 0, memo, LEVEL)
    finally:
        mind.__dict__.pop("_part_depth", None)
    return spoken


def _answer(s, words: tuple, out, budget: list, depth: int, memo: dict) -> None:
    """A part answered as if it had been said alone (brought to mind, not heard): a part answered before, by what the
    mind remembers of it as a whole; a new one in its summary, when it has one; else word by word."""
    from .__main__ import CHANNEL
    mind = s.mind
    found = None
    if _reply(mind, words, memo) is not None:
        found = ((said(words),), {})
    elif depth < DEPTH and len(words) > 1:
        found = reading(mind, words, mind.knob["F"], memo)
    mind.__dict__["_part_depth"] = depth                 # (for !neden: how deep in parts a word was decided)
    try:
        if found is not None:                            # remembered whole, or heard in its summary
            _hear(s, _heard_as(mind, found[0]), IMAGINED)
            acts = _end(s, IMAGINED, ".", listening=False)
            _go_on(s, acts, words, found[1], out, budget, depth, memo, LEVEL)
        else:                                            # word by word, as the session hears and the mind answers
            s.listen(" ".join(words), IMAGINED)
            acts = s.end_of_turn(IMAGINED, listening=False, tone=".")
            _go_on(s, acts, words, {}, out, budget, depth, memo, CHANNEL)
    finally:
        mind.__dict__["_part_depth"] = depth - 1


def _go_on(s, acts, words: tuple, where, out, budget: list, depth: int, memo: dict, channel: str) -> None:
    """Carry on from what the mind has just decided on ``channel`` (dusunce.go_on, with wholes): a word is said; a
    sequence held as one item is said word by word; a pointer is said by answering its part, the working state set
    aside meanwhile. The echo of each decision comes back in the next moment as the one item it was decided as."""
    from .dusunce import comes_round, hear_without_deciding
    mind = s.mind
    while acts:
        for e in acts:
            if e.channel != channel:
                continue
            symbol = e.symbol
            if is_pointer(symbol):
                span = where.get(symbol[1])
                if span is not None and budget[0] > 0:
                    held = set_aside(mind)
                    try:
                        _answer(s, words[span[0]:span[1]], out, budget, depth + 1, memo)
                    finally:
                        take_up(mind, held)
            elif is_said(symbol):
                out([str(w) for w in symbol[1:]])
            elif not is_whole(symbol):
                out([str(symbol)])
        budget[0] -= 1
        if budget[0] <= 0 or comes_round(mind):
            hear_without_deciding(mind)
            return
        acts = mind.step([])
