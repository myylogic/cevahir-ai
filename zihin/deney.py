# -*- coding: utf-8 -*-
"""F57 Trials (Kendi Üzerinde Deney): the mind's ways of reading are on trial against its own configuration, and
the configuration follows what the trials show.

Until here the mind learned *what* is so, and (F49) how far each of its ways of reading deserves to be believed. Which
ways of reading it has at all was decided outside it: a mechanism was measured by its builders in worlds they made,
turned on by hand or left off. Every mechanism that is studied and off is a way of learning the mind does not use.

Here that decision is the mind's, taken as its builders took it: by a controlled experiment, one thing changed.

* **A situation.** A context at which the mind decided what to say (it was its turn), and which then goes on from
  outside on a channel it speaks on: someone shows what the answer was. The teacher's loop, a correction, the next
  word of something it is being told.
* **The experiment.** At that moment, *before what came is taken in*, the mind reads the same context once more as
  it would have decided it (the reading it speaks with, its own words left out, the same race run dry: nothing is
  said or set), and once more for each of its ways with that one way changed: on if it is off, off if it is on. The
  memory is the same, the situation is the same, one thing differs. What came then says which of the two was
  better: saying what came is better than saying nothing, and saying nothing is better than saying another word.
* **The record.** Where the two differ, the way gets one experience on a tie of its own (on the mind's own channel
  ``deney``): W+ where the configuration *with* the way did better, W- where the one without it did. Nothing is
  written where they say the same.
* **A way that sleeps is read as one in use would be.** The mind reads every claim with what claims of its way have
  come to (F49), and a way never judged is read by how ways like it have fared: where the mind's ways in use have
  been failing (which is when it needs another), a sleeping way would be read as one of them and could never show
  what it is worth. So what a sleeping way alone claims in its trial is settled by what came, on that way's own
  ties, exactly as a claim the mind made and did not say is settled: its F49 record is the one it would have had
  in use. Found by measuring (one case in fifty rounds before, taken into use after two failures with it).
* **The configuration follows.** A way that is off is taken into use when its record is proven,
  W+ / (W+ + W- + k) > 1/2: it did better more often than worse by more than one experience (one lucky hit is not
  knowledge: F50's rule). A way that is in use is put away by the same rule the other way round,
  W- / (W+ + W- + k) > 1/2. In between nothing changes. A way taken into use stays on trial: the same experiment,
  now asking whether the mind would do better without it, so what it learned earlier is watched for as long as it
  lives (a way that came to break what the mind could do is put away again).
* **One change at a time.** Every way was tried against the configuration of that moment, alone. Two ways that each
  mend the same failure both look proven, and nothing was ever shown about the two together. So the configuration
  changes by one way at a time (of those whose record calls for it, the one with the firmest record; among equals
  the first in the mind's list of ways), and with it the question has changed: what a sleeping way did against the
  configuration that is no more says nothing about the one there is now, and its trial begins again (its record
  counts from that moment; the evidence stays in its tie). A way that is still worth something shows it again; one
  that only did what the way just taken does makes no difference any more and sleeps on. Found by measuring: where
  answers name the thing, four of six ways had been taken in pairs, on the same cases.
* **Kept.** The change is to the mind's own configuration, saved with it. Every case in which a way made a
  difference, and every change, is kept with its context, what came, and what each configuration would have said:
  why the mind reads as it does now can be read from its own record (``report``; the chat's ``!deney``).

* **Two together.** One way is changed at a time, so a gain that needs two sleeping ways together is not seen:
  each alone changes nothing that is said. Where the configuration did not say what came and no single way mended
  it, a sleeping way that alone changed what the mind *read* there (though not what it would say) is tried once
  more with each other sleeping way; a pair that then says what came is a way to try, kept and tried like any
  other (both on against the configuration), with a record of its own; when the record proves it the two are
  taken into use together, as one change. Found by measuring: a life in which frames of frames was never taken
  because what kept its claim down was another way's to mend.

What a way *is*: the mechanisms of reading the mind was built with (``WAYS``), and (F58, ``propose``) the relations
it builds itself: where it did not say what came, a chain of two of its own relations that names it for a filler
of the context is a relation nobody wrote, and it is tried like any other way; one it has taken into use is a
relation of its own from then on, and a step of the next chain. What is decided here is which of them this mind
uses, by what it lived.
One way is changed at a time: a gain that needs two ways together is not found until one of them has been taken
into use for a reason of its own.

Nothing here names a word or a world, and no number is chosen: one experience a case, the core's k, the race the
mind decides with.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .events import ACTED, IMAGINED
from .store import EXPECT, W_NEG, W_POS

CHANNEL = "deney"            # the mind's own channel for the trials of its ways
BETTER = "iyi"               # the unit the ties (way -> better) point at
WAYS = ("known", "families", "lived_apart", "asked_place", "echo_exact", "inherit", "witness", "difference",
        "unseen_share")
KEPT = 1024                  # the contexts at which it decided, kept until what follows is shown (the oldest go first)
CASES = 256                  # the last cases in which a way made a difference (a record for the report)
NAMES = {"known": "hakkında bilineni söylemek (F41)", "families": "çerçevelerin çerçevesi (F53)",
         "lived_apart": "ayrı yaşanmış ilişki (F54)", "asked_place": "sorulan yer (F55)",
         "echo_exact": "aynı kelimelerin yankısı (F52)", "inherit": "türünden kalan (F37)",
         "witness": "tanığın güvenilirliği", "difference": "fark yöntemi (F39)",
         "unseen_share": "görülmemiş pay (F60)"}


PAIR = "birlikte"             # the mark of two ways tried together: (PAIR, a, b)


def singles(mind) -> Tuple[object, ...]:
    """The ways on trial one at a time: the mechanisms of reading the configuration has, and the relations the
    mind has built (``Mind.relations_tried``, F58)."""
    cfg = mind.cfg
    return tuple(way for way in WAYS if hasattr(cfg, way)) + tuple(getattr(mind, "relations_tried", ()))


def is_pair(way) -> bool:
    return isinstance(way, tuple) and len(way) == 3 and way[0] == PAIR


def ways(mind) -> Tuple[object, ...]:
    """The ways on trial in this mind: the single ones, and the pairs it found to do together what neither does
    alone, for as long as both sleep (``Mind.pairs_tried``)."""
    return singles(mind) + tuple(pair for pair in getattr(mind, "pairs_tried", ())
                                 if not in_use(mind, pair[1]) and not in_use(mind, pair[2]))


def in_use(mind, way) -> bool:
    if isinstance(way, str):
        return bool(getattr(mind.cfg, way, False))
    if is_pair(way):
        return in_use(mind, way[1]) and in_use(mind, way[2])
    return way in getattr(mind, "relations", ())


def _set(mind, way, on: bool) -> None:
    if is_pair(way):
        _set(mind, way[1], on)
        _set(mind, way[2], on)
        return
    if isinstance(way, str):
        setattr(mind.cfg, way, on)
    else:
        held = [r for r in getattr(mind, "relations", ()) if r != way]
        mind.relations = tuple(held + [way]) if on else tuple(held)
    for name in ("_between_memo", "_episode_memo", "_situation_cache"):
        mind.__dict__.pop(name, None)                      # what was read with the other configuration


def record(mind, way) -> Tuple[float, float]:
    """(W+, W-) of the tie (way -> better): the cases in which the configuration with the way did better, and
    those in which the one without it did."""
    from .read import mass1
    unit = mind.unit_of(CHANNEL, way)
    if unit < 0:
        return 0.0, 0.0
    tie = mind.S.find_tie(unit, mind.unit_of(CHANNEL, BETTER), EXPECT, 0)
    return mass1(mind, tie) if tie >= 0 else (0.0, 0.0)


def standing(mind, way) -> Tuple[float, float]:
    """(W+, W-) of a way against the configuration as it is: for a way that sleeps, its record since the
    configuration last changed (``Mind.trial_mark``: what its record was then); for a way in use, the record it was
    taken into use with and everything since."""
    w_pos, w_neg = record(mind, way)
    was = getattr(mind, "trial_mark", {}).get(way)
    return (w_pos - was[0], w_neg - was[1]) if was else (w_pos, w_neg)


def _settle(mind, way, better: bool) -> None:
    """One case: booked as the mind's own percept (it judges nothing it was told), no relief."""
    from .learn import add_evidence1
    channel = mind.add_channel(CHANNEL, "knob")
    tie = mind.S.tie(mind._atom(channel, way), mind._atom(channel, BETTER), EXPECT, 0)
    add_evidence1(mind, W_POS if better else W_NEG, ACTED, tie, 1.0, relief=False)


def called(mind, acts) -> None:
    """The mind decided at this context (it was not listening): kept until what follows it is shown."""
    context = mind.context
    if context is None:
        return
    held = mind.called
    held.pop(context, None)
    held[context] = mind.t
    while len(held) > KEPT:
        held.pop(next(iter(held)))


_MEMOS = ("_read_memo", "_between_memo", "_episode_memo", "_situation_cache", "_src_last", "_src_now")


class _aside:
    """A stretch in which the mind reads as when it speaks and keeps nothing: the memos of the moment are put aside
    and put back, and the threshold raised while it listens is the one it speaks at."""

    def __init__(self, mind) -> None:
        self.mind = mind

    def __enter__(self):
        mind = self.mind
        self.saved = {name: mind.__dict__[name] for name in _MEMOS if name in mind.__dict__}
        self.theta = mind.knob["theta"]
        mind.knob["theta"] = mind.cfg.theta
        for name in _MEMOS:
            mind.__dict__.pop(name, None)
        mind.__dict__["_read_memo"] = {}
        return self

    def __exit__(self, *exc) -> None:
        mind = self.mind
        mind.knob["theta"] = self.theta
        for name in _MEMOS:
            mind.__dict__.pop(name, None)
        mind.__dict__.update(self.saved)


def would_say(mind, context: int, chain: Optional[dict] = None,
              read: Optional[dict] = None) -> Optional[Dict[int, int]]:
    """What the mind would say at ``context`` with its configuration as it stands, by channel: the reading it
    decides with (its own words left out) and the same race, run dry. Nothing is said, set or kept. ``chain``: a
    dict that is given F49's chain of this reading (unit -> [(ways, the reading's own claim, the claim), ...]).
    ``read``: a dict that is given the reading itself (what it expects of each word it could say)."""
    from . import read as R
    from .act import _race
    cfg = mind.cfg
    if cfg.support != "members" or not cfg.bind_repeats or not cfg.race:
        return None
    with _aside(mind):
        trust = mind.trust.clone()
        trust[ACTED] = 0.0
        reading = R.members_map(mind, context, trust)
        found = mind.__dict__.get("_src_last")
        if chain is not None and found is not None and found[0] == context:
            chain.update(found[1])
        act, channels = R.kind_masks(mind)[5], R.mv_view(mind, "u_channel")
        candidates = {unit: e for unit, e in reading.items() if act[channels[unit]]}
        if read is not None:
            read.update(candidates)
        return {int(channels[unit]): int(unit) for unit in _race(mind, candidates, cfg.theta, cfg.T, dry=True)}


def _claims(mind, chain: dict, said) -> Dict[tuple, Tuple[int, float]]:
    """What each way claimed most in a reading, as F49 keeps it (``hata.note``): over the line the mind speaks at
    as the reading made it, or the word it would say: {ways: (the claimed unit, the reading's own strength)}."""
    from . import read as R
    cfg = mind.cfg
    line = 0.5 + cfg.theta / (2.0 * cfg.T)
    act, channels = R.kind_masks(mind)[5], R.mv_view(mind, "u_channel")
    top: Dict[tuple, Tuple[int, float]] = {}
    for unit, made in chain.items():
        if not act[channels[unit]]:
            continue
        for ways, e0, _ in made:
            if (e0 > line or (unit in said and e0 > 0.5)) and (ways not in top or e0 > top[ways][1]):
                top[ways] = (unit, e0)
    return top


TRIED = 32                   # the relations it has proposed and tries (one that never made a difference goes first)


def propose(mind, context: int, came: Dict[int, int], now: Dict[int, int]) -> None:
    """F58: where the mind's configuration did not say what came. For a filler of the context (the place of one of
    its frames), a chain of two of the mind's relations that names what came is a relation to try: read before what
    came is taken in, so the chain names it from what was lived apart. Whether it is a relation of the frame is for
    the frame's members to say (they vouch for it or not when it is tried).

    The relations a chain is made of are the mind's own: those it was built with (``read.STEPS``) and those it built
    and took into use. A relation it built is a step like any other, so a chain of a chain is found when the shorter
    one has proven itself: the relations grow out of the ones that worked, to any length."""
    missed = {v for channel, v in came.items() if now.get(channel) != v}
    if not missed:
        return
    from . import read as R
    from .store import CATEGORY, COMPOSED
    kinds = R.mv_view(mind, "u_kind")
    if not kinds[context] & COMPOSED:
        return
    held = R.parts_of(mind, context)
    tried = mind.relations_tried
    with _aside(mind):
        trust = mind.trust.clone()
        trust[ACTED] = 0.0
        for frame, _ in R.generalisations(mind, context):
            parts = R.parts_of(mind, frame)
            cats = {p for p in parts if kinds[p] & CATEGORY}
            if len(cats) != 1:
                continue
            category = next(iter(cats))
            where = tuple(j for j, p in enumerate(parts) if p == category)
            if not where or len(held) <= where[0]:
                continue
            fixed = tuple(p for p in parts if p != category)
            filler = held[where[0]]
            steps = tuple((relation,) for relation in R.STEPS) + tuple(mind.relations)
            for first in steps:
                for second in steps:
                    path = first + second
                    if path in tried or path in mind.relations:
                        continue
                    if R.built_value(mind, path, filler, context, where[0], fixed, category, trust) in missed:
                        R.built_code(mind, path)
                        tried.append(path)
    while len(tried) > TRIED:
        idle = next((p for p in tried if p not in mind.relations and sum(record(mind, p)) <= 0.0), None)
        if idle is None:
            break
        tried.remove(idle)


def pairs(mind, readers) -> List[tuple]:
    """The pairs to try where no single way mended a failure: each sleeping way that alone changed what was read
    there, with each other sleeping way (pairs the mind already keeps are tried with the rest of its ways)."""
    asleep = [way for way in singles(mind) if not in_use(mind, way)]
    out = []
    for a in readers:
        for b in asleep:
            if a == b:
                continue
            pair = (PAIR,) + tuple(sorted((a, b), key=repr))
            if pair not in out and pair not in mind.pairs_tried:
                out.append(pair)
    return out


def _worth(said: Optional[int], came: Optional[int]) -> int:
    """What an answer was worth: what came (1), nothing (0), another word (-1). F66: where nothing came (the other
    ended there), saying nothing is what came (1) and a word is another word (-1)."""
    if came is None:
        return 1 if said is None else -1
    return 0 if said is None else (1 if said == came else -1)


def trial(mind, items) -> None:
    """The context of the last moment goes on with ``items`` [(unit, provenance)], and they are not yet taken in.
    If the mind decided at that context and something came from outside on a channel it speaks on: the experiment
    (see the module's text), the records, and the configuration follows.

    F66 (An End Is an Outcome, ``Config.trial_ends``): where it went on from outside with no word on any channel the
    mind speaks on (the other ended there: the end of a turn it watches), nothing is what came. The configuration
    that would have said nothing there did better and one that would have said a word did worse, so a way that
    makes the mind go on past the end of an answer is found by the trials like a way that makes it say a wrong
    word. (Before, such a moment was no outcome, and the harm could only be repaired from outside: F62.)"""
    context = mind.context
    if context is None or context not in mind.called:
        return
    from . import read as R
    act, channels = R.kind_masks(mind)[5], R.mv_view(mind, "u_channel")
    came = {int(channels[u]): int(u) for u, prov in items if prov not in (ACTED, IMAGINED) and act[channels[u]]}
    ended = False
    if not came:
        if not getattr(mind.cfg, "trial_ends", False) or not any(prov not in (ACTED, IMAGINED) for _, prov in items):
            return
        ended = True                                       # F66: it went on from outside, and with no word: on
        came = {c: None for c in range(len(act)) if act[c]}   # every channel the mind speaks on, nothing came
    del mind.called[context]
    faults = getattr(mind.cfg, "faults", False)
    chain_now: dict = {}
    read_now: dict = {}
    now = would_say(mind, context, chain_now if faults else None, read_now)
    if now is None:
        return
    claimed_now = _claims(mind, chain_now, set(now.values())) if faults else {}
    asleep: Dict[tuple, Tuple[int, float]] = {}            # what only a sleeping way claims here (settled below)
    if getattr(mind.cfg, "built_relations", False) and not ended:   # (nothing came: there is nothing to name)
        propose(mind, context, came, now)
    if not mind.trial_base:
        mind.trial_base = {way: in_use(mind, way) for way in singles(mind)}
    k = mind.cfg.k
    differed = []
    readers = []                                           # sleeping ways that alone changed what was read here
    missed = [channel for channel, v in came.items() if now.get(channel) != v]
    mended = False                                         # whether some one way would have said what it missed
    for way in ways(mind):                                 # every way is tried against the same configuration: the
        on = in_use(mind, way)                             # one the mind has at this moment
        if not is_pair(way):
            mind.trial_base.setdefault(way, on)
        chain: dict = {}
        read: dict = {}
        _set(mind, way, not on)
        try:
            other = would_say(mind, context, chain if faults and not on else None, read)
        finally:
            _set(mind, way, on)
        if not on and not is_pair(way) and read != read_now:
            readers.append(way)
        mended = mended or any(other.get(channel) == came[channel] for channel in missed)
        if faults and not on:                              # F49 for a way that sleeps: what it alone claims here is
            for claimant, claim in _claims(mind, chain, set(other.values())).items():   # a prediction too
                if claimant not in claimed_now and claimant not in asleep:
                    asleep[claimant] = claim
        with_way, without = (now, other) if on else (other, now)
        for channel, v in came.items():
            a, b = _worth(with_way.get(channel), v), _worth(without.get(channel), v)
            if a == b:
                continue
            _settle(mind, way, a > b)
            if way not in differed:
                differed.append(way)
            mind.trial_cases.append({"t": mind.t, "way": way, "context": int(context), "came": v,
                                     "with": with_way.get(channel), "without": without.get(channel),
                                     "better": a > b})
    if missed and readers and not mended:
        for pair in pairs(mind, readers):                  # two together: no single way mended what it missed
            _set(mind, pair, True)
            try:
                other = would_say(mind, context)
            finally:
                _set(mind, pair, False)
            for channel in missed:
                v = came[channel]
                if _worth(other.get(channel), v) > _worth(now.get(channel), v):
                    if pair not in mind.pairs_tried:
                        mind.pairs_tried.append(pair)
                    _settle(mind, pair, True)
                    if pair not in differed:
                        differed.append(pair)
                    mind.trial_cases.append({"t": mind.t, "way": pair, "context": int(context), "came": v,
                                             "with": other.get(channel), "without": now.get(channel),
                                             "better": True})
        del mind.pairs_tried[:-TRIED]
    del mind.trial_cases[:-CASES]
    if asleep:                                             # a way that sleeps is read with F49's record like any
        from . import hata                                 # other, and it cannot earn one unless what it claims is
        for claimant, (unit, e0) in asleep.items():        # settled: as a claim the mind made and did not say is
            v = came.get(int(channels[unit]))
            if v is not None:
                hata.settle(mind, claimant, v == unit, e0)
    change = None                                          # the configuration changes after all were tried, by
    for way in differed:                                   # one way: the firmest record, the first among equals
        on = in_use(mind, way)
        w_pos, w_neg = standing(mind, way)
        firm = (w_neg if on else w_pos) / (w_pos + w_neg + k)
        if firm > 0.5 and (change is None or firm > change[0]):
            change = (firm, way, not on, w_pos, w_neg)
    if change is not None:
        _, way, on, w_pos, w_neg = change
        _set(mind, way, on)
        mind.trial_changes.append({"t": mind.t, "way": way, "on": on, "better": w_pos, "worse": w_neg})
        for other in ways(mind):                           # what sleeps was tried against a configuration that is no
            if not in_use(mind, other):                    # more: its trial begins again
                mind.trial_mark[other] = record(mind, other)


def own_choice(mind, way) -> Optional[bool]:
    """What the mind's own trials last decided about a way (taken into use: True, put away: False); None when it
    never changed it, or when its owner changed it after: how the way stands then was set from outside."""
    choice = None
    for change in getattr(mind, "trial_changes", ()):
        if change["way"] == way or (is_pair(change["way"]) and way in change["way"][1:]):
            choice = None if change.get("owner") else bool(change["on"])
    return choice


def set_from_outside(mind, changed: Dict[object, bool]) -> None:
    """The mind's owner changed which ways it uses (the program: at loading, --ac, --kapat). The ways are set; where
    the trials have begun the change is an entry of their record, marked as the owner's (``sync`` rebuilds the
    configuration from that record after a moment was undone); and the question has changed, as it does when the
    mind changes a way itself: what a way did against the configuration that is no more says nothing about the one
    there is now, so the trial of every way that sleeps, and of every way that was changed, begins again."""
    changed = {way: bool(on) for way, on in changed.items() if in_use(mind, way) != bool(on)}
    if not changed:
        return
    for way, on in changed.items():
        _set(mind, way, on)
        if mind.trial_base:
            mind.trial_changes.append({"t": mind.t, "way": way, "on": on, "better": 0.0, "worse": 0.0,
                                       "owner": True})
    for way in ways(mind):
        if way in changed or not in_use(mind, way):
            was = record(mind, way)
            if was != (0.0, 0.0) or way in mind.trial_mark:
                mind.trial_mark[way] = was


def sync(mind) -> None:
    """The configuration as the mind's own record of its changes says it is (after a moment was undone)."""
    state = dict(getattr(mind, "trial_base", {}) or {})
    for change in getattr(mind, "trial_changes", ()):
        state[change["way"]] = change["on"]
    for way, on in state.items():
        if in_use(mind, way) != on:
            _set(mind, way, on)


def way_name(mind, way) -> str:
    if isinstance(way, str):
        return NAMES.get(way, way)
    if is_pair(way):
        return "birlikte: %s + %s" % (way_name(mind, way[1]), way_name(mind, way[2]))
    from . import hata
    return "kendi kurduğu ilişki: " + " sonra ".join(hata.NAMES.get(r, str(r)) for r in way)


def report(mind, most: int = 8) -> List[str]:
    """What the trials of its ways have come to, and the last cases (for the owner; the evidence is in the ties)."""
    from . import hata
    lines = ["Kendi üzerinde deney: her okuma yolu, o yol açık ve kapalıyken aynı durumda ne söyleyeceğime bakılarak "
             "sınanır; gelen cevap hangisinin daha iyi olduğunu söyler."]
    for way in ways(mind):
        w_pos, w_neg = record(mind, way)
        if w_pos + w_neg <= 0.0 and not in_use(mind, way):
            continue
        now = standing(mind, way)
        lines.append("  %-44s %s ; daha iyi %g, daha kötü %g%s" % (
            way_name(mind, way), "kullanımda" if in_use(mind, way) else "uykuda", w_pos, w_neg,
            "" if now == (w_pos, w_neg) else " (şimdiki yapıya karşı: daha iyi %g, daha kötü %g)" % now))
    for change in mind.trial_changes[-most:]:
        if change.get("owner"):                            # set from outside (the program, at its owner's word)
            lines.append("  an %d: %s: sahibi %s" % (change["t"], way_name(mind, change["way"]),
                                                    "açtı" if change["on"] else "kapattı"))
            continue
        lines.append("  an %d: %s %s (daha iyi %g, daha kötü %g)" % (
            change["t"], way_name(mind, change["way"]), "kullanıma alındı" if change["on"] else "bırakıldı",
            change["better"], change["worse"]))
    for case in mind.trial_cases[-most:]:
        def word(unit):
            return "(sessiz)" if unit is None else str(mind.symbol_of(unit)[1])
        lines.append("  an %d, %s | gelen: %s | yol açıkken: %s | kapalıyken: %s | %s" % (
            case["t"], hata._shown(mind, case["context"]), word(case["came"]), word(case["with"]),
            word(case["without"]), way_name(mind, case["way"])))
    if len(lines) == 1:
        lines.append("  (henüz hiçbir yol bir fark yaratmadı)")
    return lines
