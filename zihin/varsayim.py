# -*- coding: utf-8 -*-
"""F59 Conjectures (Varsayım): what a kind of link has been found to carry, the mind carries further by itself.

Until here everything the mind said of a thing it had been told of that thing, or read from it when asked. Nothing
made it think a statement nobody had told it and nobody had asked for. Here it does, in its own time, and keeps
what it comes to apart from what it knows.

* **A link.** A thing x that was told to go on with an item y, where y is itself something told about (sentences
  began with it): "x y …". What follows y in that sentence is the link's *tail* (the words the link was made with).
* **A candidate.** Every statement told of y, with x in y's place. Nobody said it of x.
* **What a link carries is learned.** Where the candidate *was* told of x too, the link carried that statement to
  that thing (one experience for, booked once). Where x was told the same statement with one item different, it
  did not (one against). The records are kept on the mind's own channel and read level by level, each level the
  vote of the one below it, every member once, as everywhere in the mind: this statement through links with this
  tail (the things it held of, the things it failed of); links with this tail (the statements for, against); links
  of any kind. A level has decided when its record is for or against by more than the weight of having no record
  (k: one lucky agreement is not knowledge; F50's rule). The most specific level that has decided speaks, and a
  level that leans against without having decided keeps the more general ones from speaking for it.
* **A conjecture.** A candidate that was not told of x, whose way has decided *for*: the mind thinks it (its inner
  voice lives the statement, imagined as every thought is: not told, not sensed), and keeps it with its source: the
  link, the statement it was carried from, the record it rests on. It is a conjecture, not knowledge: the provenance
  says so, sleep sorts it against what is lived like everything imagined (F22), and what the mind reads from it
  weighs as much as it has learned to trust its own imagination.
* **Settled by what is told later.** When x is told that statement, the conjecture held; when x is told it with
  another item at one place, it failed. Either way the record of its way moves, in every field at once: a kind of
  link that fails in one field is believed less in the others.
* **In its own time.** What it was told of a thing is something to think over; it does so in a silent moment in
  which nothing else was decided, when its drive to (one thing unthought: u = 1) and what thinking things over has
  come to before reach the line it speaks at: rate = (2u - 1) + (2 e - 1), as for any act. One thing a moment.

Nothing here names a word, a field or a kind of link. The mind is not told that "sınıfındandır" makes a class: it
finds that what is told of the item before that word is told of the thing, and that what is told of the item before
another word is not.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .events import ACTED, IMAGINED, INFERRED, Event
from .store import EXPECT, SLOTTED, W_NEG, W_POS

CHANNEL = "varsayim"         # the mind's own channel for what its links have carried
HELD = "tuttu"               # the unit the ties (way -> held) point at
MADE = "vardığım"            # the way of the conjectures it made itself: what they came to when they met what it
                             # was told (what thinking things over is worth)
ANY = ("v",)                 # links of any kind
KEPT = 256                   # the conjectures kept for the report (the evidence is in the ties and in what it thought)
QUEUE = 256                  # the things it has still to think over (the oldest go first)
MOST = 32                    # the statements of an item read at once
LINKS = 4096                 # the items whose linked things are kept


def told(mind):
    """The trust the mind reads what it was *told* with: its own words, thoughts and inferences left out. (One
    tensor a moment, kept with the moment's readings: they are remembered by the trust they were made with.)"""
    return _trust(mind, "told")


def imagined(mind):
    """The trust that reads only what the mind's inner voice said."""
    return _trust(mind, "imagined")


def _trust(mind, which: str):
    from .read import _moment_memo
    memo = _moment_memo(mind)
    key = ("varsayim_trust", which)
    trust = memo.get(key) if memo is not None else None
    if trust is None:
        trust = mind.trust.clone()
        if which == "told":
            trust[ACTED] = trust[IMAGINED] = trust[INFERRED] = 0.0
        else:
            trust[:] = 0.0
            trust[IMAGINED] = 1.0
        if memo is not None:
            memo[key] = trust
    return trust


def statements(mind, y: int, trust) -> List[tuple]:
    """The sequences that began with ``y`` and came to an end, as they were told (not what was only asked of it):
    tuples of items on y's channel."""
    from .read import _moment_memo, _world_pos, kind_masks, mv_view, only_asked, start_mass
    memo = _moment_memo(mind)
    key = ("statements", y, id(trust))
    if memo is not None and key in memo:
        return memo[key]
    S, F = mind.S, mind.knob["F"]
    kinds, channels = mv_view(mind, "u_kind"), mv_view(mind, "u_channel")
    sense = kind_masks(mind)[4]
    channel = channels[y]
    out: List[tuple] = []
    stack = [(y, v) for v, (b, _) in start_mass(mind, y, trust).items()
             if b > 0.0 and channels[v] == channel and not only_asked(mind, y, v)]
    while stack and len(out) < MOST:
        sequence = stack.pop()
        unit = S.tuple_index.get(sequence)
        if unit is None or kinds[unit] & SLOTTED:
            continue
        ended = False
        for v, p in zip(*_world_pos(mind, unit, trust)):
            if p <= 0.0:
                continue
            if sense[channels[v]]:
                ended = True                               # its turn ended here
            elif channels[v] == channel and len(sequence) < F:
                stack.append(sequence + (v,))
        if ended:
            out.append(sequence)
    out.sort()
    if memo is not None:
        memo[key] = out
    return out


def _mass(mind, way) -> Tuple[float, float]:
    from .read import mass1
    unit = mind.unit_of(CHANNEL, way)
    if unit < 0:
        return 0.0, 0.0
    tie = mind.S.find_tie(unit, mind.unit_of(CHANNEL, HELD), EXPECT, 0)
    return mass1(mind, tie) if tie >= 0 else (0.0, 0.0)


def _settle(mind, way, held: bool) -> None:
    """One experience of a way: the candidate it made was told so (held) or told otherwise. The mind's own percept."""
    from .learn import add_evidence1
    channel = mind.add_channel(CHANNEL, "knob")
    tie = mind.S.tie(mind._atom(channel, way), mind._atom(channel, HELD), EXPECT, 0)
    add_evidence1(mind, W_POS if held else W_NEG, ACTED, tie, 1.0, relief=False)
    from .read import _moment_memo
    memo = _moment_memo(mind)
    if memo:
        memo.pop(("varsayim_votes",), None)


def tried(mind, way, x: int) -> bool:
    """Whether what this way would carry to ``x`` has met what x was told (booked once: the tie is the record)."""
    return _mass(mind, way + (x,)) != (0.0, 0.0)


def votes(mind) -> Dict[object, Tuple[float, float]]:
    """The records, level by level, each level the vote of the one below it, every member once: a statement
    through links with a tail ("v", tail, S): (the things it was carried to and held of, those it failed of); links
    with a tail ("v", tail) and links of any kind ("v",): (the statements for, the statements against), each by the
    sign of its record. Read once a moment."""
    from .read import _moment_memo
    memo = _moment_memo(mind)
    found = memo.get(("varsayim_votes",)) if memo is not None else None
    if found is None:
        found = {}
        if mind.ch.has(CHANNEL):
            channel = mind.ch.id(CHANNEL)
            for index in range(mind.ch.sizes()[channel]):
                symbol = mind.ch.symbol(channel, index)
                if not isinstance(symbol, tuple) or len(symbol) != 4:
                    continue
                w_pos, w_neg = _mass(mind, symbol)
                if w_pos + w_neg <= 0.0:
                    continue
                n_for, n_against = found.get(symbol[:3], (0.0, 0.0))
                found[symbol[:3]] = (n_for + (w_pos > w_neg), n_against + (w_neg > w_pos))
            for way, (n_for, n_against) in list(found.items()):
                for level in (ANY, way[:2]):
                    l_for, l_against = found.get(level, (0.0, 0.0))
                    found[level] = (l_for + (n_for > n_against), l_against + (n_against > n_for))
        if memo is not None:
            memo[("varsayim_votes",)] = found
    return found


def decided(mind, way) -> Tuple[Optional[bool], tuple, float, float]:
    """What the record says of a way ("v", tail, S), the most specific level that has decided speaking: (for,
    against or None; the level; its record). A level has decided when it is for or against by more than k; a level
    that leans against without having decided keeps the more general levels from speaking for it. (A statement's
    own record is one of the votes of the levels above it: a statement that held of one thing is not believed less
    than one that was never tried. Found by measuring.)"""
    found = votes(mind)
    k = mind.cfg.k
    for level in (way, way[:2], ANY):
        n_for, n_against = found.get(level, (0.0, 0.0))
        if abs(n_for - n_against) > k:
            return n_for > n_against, level, n_for, n_against
        if n_against > n_for:                              # it leans against, not yet by more than k: nothing more
            return None, level, n_for, n_against           # general speaks for it (a statement that failed of two
    return None, ANY, 0.0, 0.0                             # things and held of one is not carried to a fourth)


def candidates(mind, x: int):
    """What the links of ``x`` would carry to it: [(y, tail, S, C, state), ...]. S a statement told of y, C the same
    with x in y's place; state: "told" (x was told C), "otherwise" (x was told C with one item different), "open"."""
    trust = told(mind)
    own = statements(mind, x, trust)
    known = set(own)
    out = []
    for link in own:
        y, tail = link[1], link[2:]
        if y == x:
            continue
        for S in statements(mind, y, trust):
            C = (x,) + S[1:]
            if C == link or x in S:
                continue
            if C in known:
                state = "told"
            elif any(len(T) == len(C) and sum(a != b for a, b in zip(T, C)) == 1 for T in own):
                state = "otherwise"
            else:
                state = "open"
            out.append((y, tail, S, C, state))
    return out


def took_in(mind, items) -> None:
    """An utterance the mind was told has ended: the item it began with is something to think over, and so is every
    thing linked to that item (what is told of it may be carried to them)."""
    from .read import kind_masks, mv_view
    sense, channels = kind_masks(mind)[4], mv_view(mind, "u_channel")
    if not any(sense[channels[u]] and prov not in (ACTED, IMAGINED, INFERRED) for u, prov in items):
        return
    first = next((u for u in mind.episode if not sense[channels[u]]), None)
    if first is None:
        return
    queue = mind.pondering
    for unit in (first,) + tuple(mind.linked.get(first, ())):
        queue.pop(unit, None)
        queue[unit] = mind.t
    while len(queue) > QUEUE:
        queue.pop(next(iter(queue)))


def worth(mind) -> float:
    """What thinking things over has come to: the expectation that a conjecture it made holds, over those that
    have met what it was told (1/2 before the first)."""
    from .evidence import expectation
    w_pos, w_neg = _mass(mind, MADE)
    return float(expectation(w_pos, w_neg, mind.cfg.k)) if w_pos + w_neg > 0.0 else 0.5


def think_over(mind, x: int) -> List[dict]:
    """The mind thinks over what it was told of ``x``: what its links carried or did not carry is booked on their
    records (once a thing), what it had conjectured of x is settled, and the candidates whose way has decided for
    are the new conjectures: returned for the inner voice to think."""
    new = []
    held = mind.conjectures
    thought = set(statements(mind, x, imagined(mind)))     # what its inner voice has already said of x
    for link in statements(mind, x, told(mind)):           # the items it is linked to: when something is told of
        y = link[1]                                        # one of them, x is thought over again
        if y != x and x not in mind.linked.get(y, ()):
            mind.linked[y] = (mind.linked.pop(y, ()) + (x,))[-QUEUE:]
            while len(mind.linked) > LINKS:                # (the items least lately linked go first)
                mind.linked.pop(next(iter(mind.linked)))
    found = candidates(mind, x)
    for y, tail, S, C, state in found:                     # first what its links carried or did not: the records
        if state == "open":                                # (so that what is then decided of x rests on them too)
            continue
        way = ("v", tail, S)
        key = (x, y, S)
        if not tried(mind, way, x):
            _settle(mind, way + (x,), state == "told")
            if C in thought:                               # a conjecture of its own has met what it was told
                _settle(mind, MADE, state == "told")
            for other in mind.waiting.pop(way[:2], ()):    # the record of this kind of link has moved: what waited
                if other != x:                             # for it is thought over again
                    mind.pondering.pop(other, None)
                    mind.pondering[other] = mind.t
        entry = held.get(key)
        if entry is not None and entry["state"] == "open":
            held[key] = dict(entry, state="held" if state == "told" else "failed", settled=mind.t)
    for y, tail, S, C, state in found:                     # then what nobody told of x
        way = ("v", tail, S)
        key = (x, y, S)
        if state != "open" or key in held or C in thought:
            continue
        verdict, level, n_for, n_against = decided(mind, way)
        if verdict:
            entry = {"t": mind.t, "x": x, "y": y, "tail": tail, "from": S, "statement": C, "level": level,
                     "for": n_for, "against": n_against, "state": "open", "settled": None}
            held[key] = entry
            new.append(entry)
        elif (verdict is None or level != way) and x not in mind.waiting.get(way[:2], ()):
            mind.waiting[way[:2]] = (mind.waiting.pop(way[:2], ()) + (x,))[-QUEUE:]   # not decided against by this
            while len(mind.waiting) > QUEUE:               # statement's own record: it waits for the record of its
                mind.waiting.pop(next(iter(mind.waiting)))  # kind of link to move
    while len(held) > KEPT:
        held.pop(next(iter(held)))
    return new


def think(mind, statement: tuple, end=()) -> None:
    """The inner voice lives a conjecture: the statement's items one after another and its end (``end``: the events
    an utterance ends with), imagined as every thought of the inner voice is (not told, not sensed: what sleep sorts
    against what is lived, and what the mind's trust in its own imagination weighs), a run of its own. Nothing is
    decided while it lasts."""
    from .dusunce import hear_without_deciding
    v = mind.S.arrays()
    mind.step([])                                          # from rest: the thought begins a run of its own
    for unit in statement:
        channel = int(v.channel[unit])
        hear_without_deciding(mind, [Event(mind.ch.names[channel], mind.ch.symbol(channel, int(v.symbol[unit])),
                                           IMAGINED)])
    if end:
        hear_without_deciding(mind, [Event(channel, symbol, IMAGINED) for channel, symbol in end])
    mind.step([])                                          # the thought has ended


def carry_out(session, say=None) -> List[dict]:
    """The session lets the mind's decision to think something over happen (``Mind.supposing``): the records are
    booked, and each new conjecture is thought by the inner voice. Returns the new conjectures."""
    from .__main__ import TONE, TURN, TURN_END
    mind = session.mind
    x = mind.supposing
    mind.pondering.pop(x, None)
    new = think_over(mind, x)
    end = [(TURN, TURN_END)] + ([(TONE, ".")] if session.tones else [])
    for entry in new:
        think(mind, entry["statement"], end if session.turns else ())
        if say is not None:
            say("\n  zihin (kendi kendine vardı): %s   [\"%s\" ve \"%s\" üzerinden]" % (
                shown(mind, entry["statement"]), shown(mind, entry["from"]),
                shown(mind, (entry["x"], entry["y"]) + entry["tail"])))
    return new


def shown(mind, sequence: tuple) -> str:
    return " ".join(str(mind.symbol_of(u)[1]) for u in sequence)


def report(mind, most: int = 12) -> List[str]:
    lines = ["Varsayımlarım: bana söylenmemiş, kendi zamanımda vardığım cümleler; hangi bağın neyi taşıdığını "
             "yaşadıklarımdan öğrendim."]
    for entry in list(mind.conjectures.values())[-most:]:
        state = {"open": "sınanmadı", "held": "doğrulandı", "failed": "çürütüldü"}[entry["state"]]
        lines.append("  an %d: %s  [%s]  kaynak: \"%s\" ve \"%s %s %s\" ; dayanak: %g lehte, %g aleyhte" % (
            entry["t"], shown(mind, entry["statement"]), state, shown(mind, entry["from"]),
            shown(mind, (entry["x"],)), shown(mind, (entry["y"],)), shown(mind, entry["tail"]), entry["for"],
            entry["against"]))
    if len(lines) == 1:
        lines.append("  (henüz bir varsayımım yok)")
    return lines
