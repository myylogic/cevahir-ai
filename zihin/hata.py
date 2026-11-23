# -*- coding: utf-8 -*-
"""F49 Fault Tracing (Hata İzi): where the mind's own computation broke, and what that teaches it about its ways.

What the core does with a failed expectation is local: the tie that expected gets evidence against (W-). But most
of what the mind says of something it never lived is not a tie's expectation. It is a *reading*: what the members
of a frame lived in the same place (F11), by which relation (the same word, the asked word itself, its continuation,
what is known of it, ...), the lived continuation of the last words (F16), the form of a word never met (F36), the
most similar lived situations (F23). A reading has no tie of its own, so when it fails nothing is written anywhere:
the mind corrects the one instance (its own record now says otherwise) and reads the next one exactly as before.

Here every way of reading has ties of its own on the mind's own channel (``okuma``), and is judged as everything
is judged, by what was then lived:

* **The chain is kept where it is computed.** ``read.members_map`` notes, for every claim it makes, which way made
  it and how strongly (``src``): the frame and the kind of its claim, or the name of the reading. This is the
  computation itself, not a reconstruction of it.
* **A claim is a prediction whether or not it is said.** At a moment in which the mind is free to speak, what each
  way claimed most is kept with its context (``Mind.claimed``) when the reading made it over the line the mind
  speaks at, or when it is the word the mind says. Rivals that kept each other silent are all kept. The strength is
  the reading's own (the neutral claim, as if nothing had been learned about the way): a way the mind has stopped
  believing is still tested, so it can earn its credit back. Not where the context has a record of its own of what
  follows (there memory speaks, and the core judges memory on its own tie).
* **Reality locates the break.** When that same context is lived again and goes on from outside (someone shows or
  says what follows), each claim kept for it is settled, on the ties of
  its way (the frame's claim of that kind, or the reading): (way -> holds) gets W+ if it held and W- if it
  failed, one experience; (way -> claimed) gets what the way had expected, W+ by its strength e0 and W- by 1 - e0.
  The words of an answer before the break are confirmed by the same pass (their contexts are lived again with the
  same words), the word at the break is the fault, and what the mind said after it is never judged (those contexts
  are not lived). A value signal ("good", "bad") on an answer settles the claims the mind said in it: all held; or
  it failed somewhere, and each link takes the blame the chain itself gives it: (1 - e_i) / (1 - prod e_j), the
  link the mind was least sure of most.
* **The record is read back with the core's own rule.** Where the claims of a level were made with f_claimed on
  average and held f_held of the time, a claim of strength e0 is worth, by that level,
  e_level = 1/2 + (f_held - 1/2) / (f_claimed - 1/2) (e0 - 1/2) (kept within 0 and 1): its advantage over 1/2
  rescaled by how far reality fell short of, or exceeded, what the level claimed. The levels are read general ->
  specific as everywhere: e <- e0; e <- c e_level + (1 - c) e. The general levels (every way; the kind of claim)
  are the vote of the ways that have a record of their own, each counted once with its conviction as a member is
  in a frame, this one left out: what an untested frame's claim is worth is how frames like it have fared, not how
  many answers were given in one of them. With nothing settled the claim is the reading's, bit for bit. A way
  whose claims held as often as it claimed leaves them as they are; one that held more often is believed more (the
  mind grows bolder where its ways hold); one that held less often than not brings its claims under 1/2: the mind
  is silent where it was confidently wrong.

So a failure of a frame's claim makes the mind unsure for the *next* filler of that frame (a local repair of the
structure, not only of the instance); failures across frames make the first failure of the next frame of that kind
enough (a principle the mind takes from its own errors: how frames like this one have fared); a relation that
kept holding keeps its credit while a value vote loses its own (which of two readings to believe, where they
disagree, is learned from what was lived); and confirmations give a silenced frame its voice back. A less specific reading does not step in for one the
mind has learned to doubt (``read._had``), and what the mind then does not say of an item is an open claim about
it (F46: what it may look up by itself, F47).

* **A way never tried.** It has only the vote of the ways like it. Where the vote is for it, it is believed more.
  Where the vote is against it, holding the claim back is one thing the mind can do and saying it once to find out
  is the other, and which is right depends on the life: where someone then shows what it did not say, holding back
  costs nothing; where it is only told "good" or "bad", a claim held back is never settled and a frame that would
  have held is lost for good (both measured). So the mind weighs the two by its own experience of holding back,
  e = a e_vote + (1 - a) e0, a = e(held back -> holds): W+ whenever a claim it did not say is settled by what it
  then lives, W- whenever it is called on again and a claim it held back earlier is still waiting.

* **Restructuring.** A frame's category is every item that ever filled its place. When the frame's own vote has
  been found wanting (``wanting``: its claims held less often than it claimed; with F67, ``failed``: or its record
  holds more than one experience against it, whatever it also got right), the vote is taken again among the
  members that share another lived place with the filler (``read.narrower_vote``): the category is split where the
  error showed it had to be, by a place the mind lived. That claim is a way of its own (kind ``NARROWER``), settled
  and read back like the others, so the split is kept only as far as its claims hold.

Nothing here names a word, a frame or a domain; no number is chosen (one experience per settled claim, the core's
k, the line the mind already speaks at). ``report`` says what the mind has learned about its own ways and where its
last faults were.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .events import ACTED, IMAGINED
from .evidence import conviction, frequency
from .store import EXPECT, W_NEG, W_POS

CHANNEL = "okuma"            # the mind's own channel for its ways of reading
ALL = "*"                    # every way (what its readings in general have been worth)
EXPECTED = "beklenen"        # the unit the ties (way -> claimed) point at: what the way itself expected
HELD = "sustu"               # holding a claim back: the tie (held back -> holds) says how often that was answered
NARROWER = "dar"             # the kind of a frame's claim made through a narrower place of its filler (restructuring)
KEPT = 1024                  # the contexts whose claims are kept until reality speaks of them (the oldest go first)
FAULTS = 32                  # the last faults kept for the report (a record for the owner; the evidence is in the ties)

NAMES = {0: "üyelerin aynı kelimesi", "dar": "başka bir yeri de paylaşan üyelerin kelimesi",
         -1: "sorulan kelimenin kendisi", -2: "kelimenin devamı", -3: "dizinin başı",
         -4: "yerine geçen kelime", -5: "tek başına aldığı cevap", -6: "türünden kalan", -7: "hakkında bilinen",
         -8: "hakkında bilinenden sorunun diğer kelimesine giden", -9: "sorunun diğer kelimesinin kendisi",
         -10: "bilinen dizide sorulan yerde duran"}
READINGS = {"aile": "çerçeve ailesinin oyu",
            "dizi": "bilinen dizinin devamı", "devam": "son kelimelerin yaşanmış devamı",
            "yerine": "yerine geçen kelimenin devamı", "biçim": "kelimenin biçimi",
            "fark": "tek kelimesi değişen durum", "durum": "en benzer yaşanmış durumlar"}


def frame_ways(key, frame: int) -> tuple:
    """The way of a frame's claim, general -> specific: every way; the kind of claim (the members' own word, or the
    relation it names); this frame's claim of that kind."""
    kind = key if key == NARROWER or (isinstance(key, int) and key < 0) else 0
    return ALL, ("g", kind), ("g", kind, int(frame))


def family_ways(kind) -> tuple:
    """F53: the way of a family's claim, general -> specific: every way; the family's claim of that kind (a relation
    it names, or a word every frame's members said)."""
    return ALL, ("aile", kind if isinstance(kind, int) and kind < 0 else 0)


def wanting(mind, ways) -> bool:
    """Whether this way's own record has found it wanting: its claims held less often than it claimed. (A way never
    settled has not been found anything.)"""
    if not mind.ch.has(CHANNEL):
        return False
    own = votes(mind).get(ways[-1])
    return own is not None and own[1] < own[2]


def failed(mind, ways) -> bool:
    """F67 (The Category Is Split Where It Failed, ``Config.split_failed``): whether this way's own record holds
    more than one experience against it (W- over k), whatever it also got right. ``wanting`` asks whether the way
    held less often than it claimed, on the whole; a vote that always holds for some fillers and never for others
    can be calibrated on the whole (it claims 0.6 and holds 0.6) and wrong in each part. That it failed at all,
    by more than the weight of no record, is what says the category may have to be split."""
    if not mind.ch.has(CHANNEL):
        return False
    return _mass(mind, ways[-1])[1] > mind.cfg.k


def reading_ways(name: str) -> tuple:
    """The way of a reading that is not a frame's: every way; the reading."""
    return ALL, (name,)


def _mass(mind, symbol, target: Optional[int] = None) -> Tuple[float, float]:
    """Trust-weighted (W+, W-) of the tie (way -> holds), or (way -> ``target``); zeros when never judged."""
    from .read import mass1
    unit = mind.unit_of(CHANNEL, symbol)
    tie = mind.S.find_tie(unit, mind.holds if target is None else target, EXPECT, 0) if unit >= 0 else -1
    return mass1(mind, tie) if tie >= 0 else (0.0, 0.0)


def _own(mind, symbol, expected: int) -> Optional[Tuple[float, float, float]]:
    """The record of one way that has ties of its own (a frame's claim of a kind, or a reading): (conviction, how
    often its claims held, the mean of what it claimed), or None when it was never judged."""
    w_pos, w_neg = _mass(mind, symbol)
    if w_pos + w_neg <= 0.0:
        return None
    x_pos, x_neg = _mass(mind, symbol, expected) if expected >= 0 else (0.0, 0.0)
    return (conviction(w_pos + w_neg, mind.cfg.k), frequency(w_pos, w_neg),
            frequency(x_pos, x_neg) if x_pos + x_neg > 0.0 else 0.5)


def votes(mind) -> Dict[object, Tuple[float, float, float]]:
    """Every way that has a record of its own votes once, as a member does in a frame (F11), with its conviction
    as its weight: for every way (``ALL``) and for each kind of a frame's claim, (the summed weight, the weighted
    sum of how often the members' claims held, of what they claimed). A frame asked about a hundred times is one
    member, as a frame asked about twice is: what an untested frame's claim is worth is how frames like it have
    fared, not how many answers were given in one of them. Read once a moment."""
    from .read import _moment_memo
    memo = _moment_memo(mind)
    found = memo.get(("ways",))
    if found is None:
        found = memo[("ways",)] = {}
        if mind.ch.has(CHANNEL):
            channel = mind.ch.id(CHANNEL)
            expected = mind.unit_of(CHANNEL, EXPECTED)
            for index in range(mind.ch.sizes()[channel]):
                symbol = mind.ch.symbol(channel, index)
                if symbol == EXPECTED or symbol == HELD:
                    continue
                own = _own(mind, symbol, expected)
                if own is None:
                    continue
                c, held, claimed = own
                found[symbol] = own
                for level in (ALL, symbol[:2]) if symbol[0] == "g" else (ALL,):
                    n, h, x = found.get(level, (0.0, 0.0, 0.0))
                    found[level] = (n + c, h + c * held, x + c * claimed)
    return found


def record(mind, ways, levels=None) -> list:
    """For every level of ``ways`` that has been judged, general -> specific: (conviction, how often its claims
    held, the mean of what they claimed). The general levels are the vote of the ways that have a record of their
    own, this one left out (as the instance read is left out of its frame's vote); the last level is its own."""
    out = []
    if not mind.ch.has(CHANNEL):
        return out
    found = votes(mind)
    own = found.get(ways[-1])
    k = mind.cfg.k
    for i, level in enumerate(ways[:-1]):
        if levels is not None and i not in levels:
            continue
        n, h, x = found.get(level, (0.0, 0.0, 0.0))
        if own is not None:
            n, h, x = n - own[0], h - own[0] * own[1], x - own[0] * own[2]
        if n > 1e-12:
            out.append((conviction(n, k), h / n, x / n))
    if own is not None and (levels is None or len(ways) - 1 in levels):
        out.append(own)
    return out


def answered(mind) -> float:
    """How far holding a claim back has been answered: the expectation of the tie (held back -> holds), 1/2 while
    it has never held one back. W+ when a claim it did not say is settled by what it then lives (someone showed it);
    W- when it is called on again and a claim it held back earlier is still waiting (no one did)."""
    from .evidence import expectation
    w_pos, w_neg = _mass(mind, HELD)
    return expectation(w_pos, w_neg, mind.cfg.k)


def proven(mind, ways) -> float:
    """How far the claims of one way are known to hold, by its own record alone: W+ / (W+ + W- + k) of the tie
    (way -> holds), the frequency with which its claims held, held with the conviction of that record (c f: what
    the record does not yet say counts for nothing). 0 while none of its claims was ever settled; over 1/2 only
    when they held more often than they failed by more than one experience (the weight of having no record)."""
    w_pos, w_neg = _mass(mind, ways[-1])
    return w_pos / (w_pos + w_neg + mind.cfg.k)


def withheld(mind, context: int) -> None:
    """F50: the word the mind decided at ``context`` was not said after all (it looks first, or holds it): the
    claims kept for that moment were held back, not said, and wait to be answered like any claim held back."""
    claims = mind.claimed.get(context)
    if not claims or not any(claim[2] is True for claim in claims):
        return
    mind.claimed[context] = tuple((ways, unit, False if said is True else said, e0, e)
                                  for ways, unit, said, e0, e in claims)
    mind.__dict__.setdefault("_held_back", []).append((context, mind.t))


def read_as(found, e0: float) -> float:
    """A claim of strength ``e0`` read with the record ``found`` of its way, general -> specific as everywhere:
    e <- e0; at every judged level e <- c e_level + (1 - c) e. e_level is what that level says a claim of this
    strength is worth: 1/2 + (f_held - 1/2) / (f_claimed - 1/2) (e0 - 1/2), kept within 0 and 1: where the level's
    claims were made with f_claimed on average and held f_held of the time, the advantage over 1/2 of this claim
    is rescaled by how far the advantage its claims held with fell short of, or exceeded, the one they claimed."""
    e = e0
    for c, held, claimed in found:
        if claimed > 0.5:
            level = min(1.0, max(0.0, 0.5 + (held - 0.5) / (claimed - 0.5) * (e0 - 0.5)))
            e = c * level + (1.0 - c) * e
    return e


def believed(mind, ways, e0: float) -> float:
    """A claim of strength ``e0`` (the reading's own: the neutral claim) read with what the claims of its way have
    come to (``read_as``). A way whose claims held as often as it claimed leaves the claim as it is; one that held
    more often is believed more; one that held less often than not brings its claims under 1/2. Only a claim (over
    1/2) is read so: the record is of what the way claimed. While nothing was ever settled the claim is the neutral
    one, bit for bit. The record is read once a moment.

    A way that was never tried has only the vote of the ways like it. Where that vote is for it, it is believed
    more. Where the vote is against it, the mind is between two things: holding the claim back (what frames like
    this one came to), and saying it once to find out (a try teaches). It weighs them by its own experience of
    holding back: e = a e_vote + (1 - a) e0, a = how far what it held back was then answered (``answered``). Where
    someone shows it what it did not say, holding back costs nothing and the principle rules before the first try;
    where no one does, a claim held back is never settled, and it tries."""
    if e0 <= 0.5 or not mind.ch.has(CHANNEL):
        return e0
    from .read import _moment_memo
    memo = _moment_memo(mind)
    key = ("record", ways)
    found = memo.get(key)
    if found is None:
        untested = votes(mind).get(ways[-1]) is None
        found = memo[key] = (record(mind, ways, mind.__dict__.get("_fault_levels")), untested)
    levels, untested = found
    if not levels:
        return e0
    e = read_as(levels, e0)
    if untested and e < e0:
        how = mind.__dict__.get("_fault_untested")        # research: "vote" (always held back), "try" (never)
        a = 1.0 if how == "vote" else 0.0 if how == "try" else memo.get(("answered",))
        if a is None:
            a = memo[("answered",)] = answered(mind)
        e = a * e + (1.0 - a) * e0
    return e


def watch(mind) -> None:
    """While the mind only listens (it cannot speak now): at a context it has no record of its own of what follows,
    it makes the reading it would have decided with, and what each way claims there over the line it speaks at, and
    the word it would say were it its turn, are kept as any claim is (``note``), to be settled by what it hears
    next. Such a claim was neither said nor held back (it was not the mind's turn): it says nothing about whether
    holding back is answered."""
    from . import read as R
    context = mind.context
    if context is None:
        return
    trust = mind.trust.clone()
    trust[ACTED] = 0.0
    act = R.kind_masks(mind)[5]
    channels = R.mv_view(mind, "u_channel")
    outside = trust.clone()
    outside[IMAGINED] = 0.0
    if any(w_pos > 0.0 and act[channels[v]] for v, (w_pos, _) in R.own_mass(mind, context, 1, outside).items()):
        return                                             # it has lived this context: memory speaks, nothing to test
    mind.__dict__.pop("_src_last", None)
    reading = R.members_map(mind, context, trust)
    found = mind.__dict__.pop("_src_last", None)
    if found is not None and found[0] == context:
        from .act import _race                             # the word it would say here, were it its turn (the same
        cfg = mind.cfg                                     # race on the same reading; nothing is said)
        candidates = {unit: e for unit, e in reading.items() if act[channels[unit]] and (e > 0.5 or cfg.race)}
        would = _race(mind, candidates, cfg.theta, cfg.T, dry=True) if cfg.race else []
        src = found[1]
        if getattr(cfg, "relied_on", False):               # F63: where the thing's own record speaks, what the
            src = relied_on(mind, context, src, would)     # others share is not what the mind would go by
        note(mind, src, would, listening=True)


def relied_on(mind, context: int, src: Dict[int, list], would: List[int]) -> Dict[int, list]:
    """F63 (On Trial Where It Is Relied On, ``Config.relied_on``; with silent claims): the claims of a moment in
    which the mind only listens that are a test of their way.

    A way of reading is judged by what its claims came to when the mind went by them. While it listens the mind
    goes by nothing aloud; what it would go by is the word it would say were it its turn. Where that word is
    something it lived of the very thing spoken of (a relation of the frame that reads the thing's own record
    claims it, more for than against; or it is how a sentence the mind lived of that thing went on), the mind
    knows, and the ways that only carry over what others share are not what it would go by there: that they agree
    with what it already knew is not a prediction that held, and that they differ from it is not one that failed.
    So only the claims that read the thing's own record are kept (they are tested: what the mind knows of a thing
    can be wrong). Where the word it would say rests on what others share, or it would say nothing, every claim is
    kept, as before: there the mind would have to go by what others share, and what it hears next says how far
    that holds.

    Found when everything was turned on together: a mind that had read a sentence of each of eight things and
    then watched them asked took the agreement of its generalisation with what it had read as that generalisation
    proven, and of the next thing, of which it had read nothing, it asserted what the eight share instead of
    looking it up (F50)."""
    from .act import grounded, lived_so
    if not any(grounded(src.get(unit, ())) or lived_so(mind, context, unit) for unit in would):
        return src
    kept: Dict[int, list] = {}
    for unit, made in src.items():
        mine = [entry for entry in made if grounded([entry])]
        if mine:
            kept[unit] = mine
    return kept


def note(mind, src: Dict[int, list], said: List[int], listening: bool = False) -> None:
    """At a moment in which the mind was free to speak: what each way claimed most for the present context, when
    that claim is over the line the mind speaks at as the reading made it (it would have been said had it no rival
    and had the mind learned nothing about its ways) or is the word the mind says; kept with the context, the way,
    the strength the reading gave it and the strength it was believed with, until that context is lived to go on.
    Rivals that kept each other silent are all kept: what is lived then says which way was right. Only where the
    context has no record of its own of what follows from outside (there memory speaks, and the core judges
    memory on its own tie). ``src``: unit -> [(ways, the neutral claim, the claim), ...]."""
    from . import read as R
    context = mind.context
    if context is None or not src:
        return
    cfg = mind.cfg
    line = 0.5 + cfg.theta / (2.0 * cfg.T)
    channels = R.mv_view(mind, "u_channel")
    act = R.kind_masks(mind)[5]
    top: Dict[tuple, Tuple[float, int, float]] = {}
    for unit, made in src.items():
        if not act[channels[unit]]:
            continue
        for ways, e0, e in made:
            if ways not in top or e0 > top[ways][0]:
                top[ways] = (e0, unit, e)
    def how(unit: int):                                    # True: it said it; False: it held it back; None: it was
        return None if listening else unit in said         # not its turn (``said`` is then what it would have said)
    claims = [(ways, unit, how(unit), e0, e) for ways, (e0, unit, e) in top.items()
              if e0 > line or (unit in said and e0 > 0.5)]
    for unit in said:                                      # the word it says, by every way that claimed it (a
        for ways, e0, e in src.get(unit, ()):              # claim: more for than against)
            if e0 > 0.5 and all(c[0] != ways for c in claims):
                claims.append((ways, unit, how(unit), e0, e))
    if not claims:
        return
    trust = mind.trust.clone()
    trust[ACTED] = 0.0
    trust[IMAGINED] = 0.0
    used = {int(channels[c[1]]) for c in claims}
    for v, (w_pos, _) in R.own_mass(mind, context, 1, trust).items():
        if w_pos > 0.0 and int(channels[v]) in used:
            return                                         # the context was lived to go on: its own record speaks
    claimed = mind.claimed
    waiting = mind.__dict__.get("_held_back")
    if waiting and not mind.__dict__.get("_came_to_mind"):   # called on again: what it held back in an earlier run
        start = getattr(mind, "episode_start", 0)          # of items and is still waiting was not answered (F50: not
                                                           # in a run that only comes to mind: no one called on it)
        still = [(c, t) for c, t in waiting if t >= start]
        for c, t in waiting:
            if t < start and any(claim[2] is False for claim in claimed.get(c, ())):
                _held(mind, False)
        mind.__dict__["_held_back"] = still
    claimed.pop(context, None)
    claimed[context] = tuple(claims)
    if any(claim[2] is False for claim in claims):
        mind.__dict__.setdefault("_held_back", []).append((context, mind.t))
    while len(claimed) > KEPT:
        claimed.pop(next(iter(claimed)))


def _held(mind, was_answered: bool) -> None:
    """One claim the mind held back was answered by what it then lived, or was still waiting when it was called on
    again: evidence on the tie (held back -> holds), the mind's own percept, no relief."""
    from .learn import add_evidence1
    channel = mind.add_channel(CHANNEL, "knob")
    tie = mind.S.tie(mind._atom(channel, HELD), mind.holds, EXPECT, 0)
    add_evidence1(mind, W_POS if was_answered else W_NEG, ACTED, tie, 1.0, relief=False)


def settle(mind, ways, held: bool, e0: float, amount: float = 1.0) -> None:
    """One claim of this way met reality. On the way's own ties (the frame's claim of that kind, or the reading):
    (way -> holds) gets W+ (it held) or W- (it failed), and (way -> claimed) what it had claimed: W+ by e0, W- by
    1 - e0 (so the mean of what the way claimed can be read beside how often it held). The general levels have no
    ties: they are the vote of these (``votes``). Booked as the mind's own percept (acted: self-produced, it judges
    nothing the mind was told) and no relief."""
    from .learn import add_evidence1
    S = mind.S
    channel = mind.add_channel(CHANNEL, "knob")
    expected = mind._atom(channel, EXPECTED)
    unit = mind._atom(channel, ways[-1])
    add_evidence1(mind, W_POS if held else W_NEG, ACTED, S.tie(unit, mind.holds, EXPECT, 0), amount, relief=False)
    tie = S.tie(unit, expected, EXPECT, 0)
    add_evidence1(mind, W_POS, ACTED, tie, amount * e0, relief=False)
    add_evidence1(mind, W_NEG, ACTED, tie, amount * (1.0 - e0), relief=False)
    memo = mind.__dict__.get("_read_memo")
    if memo:
        for key in [key for key in memo if type(key) is tuple and key and key[0] in ("ways", "record")]:
            del memo[key]


def _read_now(mind, ways, e0: float) -> float:
    """What a claim of strength e0 by this way is read as with the record as it stands this instant."""
    from .read import _moment_memo
    _moment_memo(mind).pop(("ways",), None)
    return read_as(record(mind, ways), e0)


def judge(mind, items) -> None:
    """The context of the last moment has gone on with ``items`` [(unit, provenance)]: the claims kept for it are
    settled by what came from outside on the claim's own channel (a channel carries one item a moment: the core's
    rule for a miss); what the mind itself said or thought there settles nothing. A claim that failed is a fault:
    kept for the report with the way that made it."""
    claims = mind.claimed.get(mind.context) if mind.claimed and mind.context is not None else None
    if not claims:
        return
    from .read import mv_view
    channels = mv_view(mind, "u_channel")
    lived = {int(channels[u]): u for u, p in items if p not in (ACTED, IMAGINED)}
    if not lived:
        return
    rest = []
    for claim in claims:
        ways, target, said, e0, _ = claim
        v = lived.get(int(channels[target]))
        if v is None:
            rest.append(claim)
            continue
        before = _read_now(mind, ways, e0)
        settle(mind, ways, v == target, e0)
        if said is False:
            _held(mind, True)                              # what it held back was answered by what it lives
        if v != target:
            _fault(mind, mind.context, target, v, ways, said, e0, before, _read_now(mind, ways, e0))
    if rest:
        mind.claimed[mind.context] = tuple(rest)
    else:
        del mind.claimed[mind.context]


def valued(mind, sign: float) -> None:
    """A value signal on the answer whose path is the episode's trace: the claims the mind *said* on that path are
    settled. Good: each held. Bad: at least one of them failed and nothing says which. The chain says how likely
    each was: a claim believed with strength e fails with 1 - e, so, given that the answer failed, claim i failed
    with (1 - e_i) / (1 - prod_j e_j): the link the mind was least sure of takes most of the blame, and a link it
    was sure of nearly none. Each claim is settled as failed by that amount (all equal when it was sure of all)."""
    trace = getattr(mind, "trace", None)
    if not trace or not mind.claimed:
        return
    found = []
    for context in dict.fromkeys(trace):
        claims = mind.claimed.get(context)
        if not claims:
            continue
        kept = tuple(c for c in claims if c[2] is not True)
        found += [(context, c) for c in claims if c[2] is True]
        if kept:
            mind.claimed[context] = kept
        else:
            del mind.claimed[context]
    if not found:
        return
    if sign > 0.0:
        for context, (ways, target, _, e0, _) in found:
            settle(mind, ways, True, e0)
        return
    whole = 1.0
    for _, claim in found:
        whole *= min(1.0, max(0.0, claim[4]))
    for context, (ways, target, _, e0, e) in found:
        blame = (1.0 - min(1.0, max(0.0, e))) / (1.0 - whole) if whole < 1.0 else 1.0 / len(found)
        if blame <= 0.0:
            continue
        before = _read_now(mind, ways, e0)
        settle(mind, ways, False, e0, min(1.0, blame))
        _fault(mind, context, target, None, ways, True, e0, before, _read_now(mind, ways, e0))


def _fault(mind, context: int, claimed: int, lived: Optional[int], ways, said: bool, e0: float, before: float,
           after: float) -> None:
    """A claim that failed, kept for the report: where, what was claimed and lived, by which way, and what a claim
    of that strength by that way was read as before and is read as now."""
    faults = mind.faults
    faults.append({"t": mind.t, "context": int(context), "claimed": int(claimed),
                   "lived": None if lived is None else int(lived), "ways": ways, "said": said,
                   "e0": e0, "before": before, "after": after})
    del faults[:-FAULTS]


def way_name(mind, ways) -> str:
    """A way in words (for the report)."""
    last = ways[-1]
    if last == ALL:
        return "her okuma"
    if last[0] == "g":
        kind = last[1]
        if isinstance(kind, int) and kind <= -100 and -100 - kind < len(getattr(mind, "relation_paths", ())):
            named = "kendi kurduğu: " + " sonra ".join(NAMES.get(r, str(r)) for r in mind.relation_paths[-100 - kind])
        else:                                              # (F58: a relation the mind built, by its steps)
            named = NAMES.get(kind, "ilişki %s" % (kind,))
        name = "genelleme: %s" % named
        if len(last) == 3:
            name += " (çerçeve: %s)" % _shown(mind, last[2])
        return name
    return READINGS.get(last[0], str(last[0]))


def _shown(mind, unit: int) -> str:
    """A unit in words: its parts' symbols, a category as "X"."""
    from .form import mv_view, parts_of
    from .store import ATOMIC, CATEGORY
    kinds = mv_view(mind, "u_kind")
    out = []
    for p in (parts_of(mind, unit) if not kinds[unit] & ATOMIC else (unit,)):
        if kinds[p] & CATEGORY:
            out.append("X")
        elif kinds[p] & ATOMIC:
            out.append(str(mind.symbol_of(p)[1]))
        else:
            out.append(_shown(mind, p))
    return " ".join(out)


def report(mind, most: int = 12) -> List[str]:
    """What the mind has learned about its own ways of reading, and where its last faults were."""
    lines: List[str] = []
    if not mind.ch.has(CHANNEL):
        return ["  henüz hiçbir okuması gerçekle karşılaşmadı (her okuma yolu doğuştaki gibi okunuyor)"]
    found = votes(mind)
    general = sorted((key for key in found if key == ALL or len(key) == 2 and key[0] == "g"),
                     key=lambda key: (key != ALL, -found[key][0]))
    lines.append("  okuma yolları (kendi sicili olan yolların oyu: kaç yol; iddia ettiği -> tuttuğu oran):")
    for key in general:
        n, h, x = found[key]
        lines.append("    %-58s %5.1f yol   %.2f -> %.2f" % (way_name(mind, (key,)), n, x / n, h / n))
    own = sorted((key for key in found if key not in general), key=lambda key: -found[key][0])
    lines.append("  kendi sicili olan yollar (tuttu / tutmadı; iddia ettiği -> tuttuğu oran; öyle bir iddia şimdi "
                 "nasıl okunuyor):")
    for key in own[:most]:
        ways = (ALL, key[:2], key) if key[0] == "g" else (ALL, key)
        w_pos, w_neg = _mass(mind, key)
        lines.append("    %-58s %4.1f / %-4.1f  %.2f -> %.2f  şimdi %.2f" % (
            way_name(mind, ways), w_pos, w_neg, found[key][2], found[key][1],
            read_as(record(mind, ways), found[key][2])))
    if len(own) > most:
        lines.append("    ... ve %d yol daha" % (len(own) - most))
    if mind.faults:
        lines.append("  son yanılgılar (kırılmanın yeri):")
        for f in mind.faults[-5:]:
            lived = "yaşanan '%s'" % _shown(mind, f["lived"]) if f["lived"] is not None else "cevap kötü karşılandı"
            lines.append("    '%s' ardından '%s' %s; %s. Kaynağı: %s; böyle bir iddiayı %.2f okuyordu, şimdi %.2f" % (
                _shown(mind, f["context"]), _shown(mind, f["claimed"]), "dedi" if f["said"] else "diyecekti"
                if f["said"] is False else "bekliyordu (dinlerken)", lived,
                way_name(mind, f["ways"]), f["before"], f["after"]))
    waiting = sum(len(c) for c in mind.claimed.values())
    if waiting:
        lines.append("  gerçekle henüz karşılaşmamış %d iddia bekliyor" % waiting)
    w_pos, w_neg = _mass(mind, HELD)
    if w_pos + w_neg > 0.0:
        lines.append("  söylemeyip tuttuğu iddialar: %.0f kez sonradan gösterildi, %.0f kez cevapsız kaldı; hiç denenmemiş "
                     "bir yolu, benzerleri tutmadıysa, %.2f oranında tutuyor (gerisinde bir kez deniyor)"
                     % (w_pos, w_neg, answered(mind)))
    return lines
