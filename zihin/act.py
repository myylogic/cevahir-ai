# -*- coding: utf-8 -*-
"""Acting, imagining and the control knobs (design sections 2.4, 5, 6, 7, 8).

Acting. An act unit is an atomic unit on a channel of kind ``act``. The same unit is active when another
emits the symbol (provenance told / imitated) and when the mind emits it (provenance acted), so "others do X"
and "I do X" are separated by provenance, not by different units:

* others do X in situation S:  EXPECT S -> X, evidence in the class ``imitated``;
* what my doing X in S brings:  EXPECT (S, X) -> relief, evidence of the mind's own outcomes.

The expectation the mind acts on pools the two: W+ = imitated W+(S -> X) + W+((S, X) -> relief),
W- = imitated W-(S -> X) + W-((S, X) -> relief). Own repetitions of X (class ``acted`` on S -> X) are a
record of behaviour and deliberately NOT evidence for doing it again. The act accumulates 2e - 1 per settling
iteration and is emitted when the sum crosses the threshold theta within T iterations; among the acts of one
channel the first to cross wins. With nothing crossing the mind stays silent.

Imagination (flag, simple first version) and control knobs as internal acts (flag, simple first version) are
at the end of the file.
"""
from __future__ import annotations

import math
from typing import Hashable, List

import torch

from .events import ACTED, IMAGINED, IMITATED, N_PROV, Event
from .evidence import conviction, expectation
from .form import kind_masks, mass1, mass1_total, mv_view
from .learn import add_evidence1
from .read import drive_of, effective, generalisations, own_mass
from .store import ABSTRACT, CATEGORY, EXPECT, F64, SLOTTED, W_NEG, W_POS

KNOB_ACTS = (("F", 1), ("F", -1), ("theta", 1), ("theta", -1), ("gain", 1), ("gain", -1))
KNOB_RANGE = {"F": (3, 15)}


def act_expectation(mind, act_unit: int, imitated_pos: float, imitated_neg: float) -> float:
    """Pooled expectation of doing ``act_unit`` now: imitated evidence plus the mind's own outcomes."""
    S = mind.S
    outcome = act_unit if mind.context is None else S.tuple_index.get(tuple(mind.episode) + (act_unit,))
    if outcome is not None:
        tie = S.find_tie(outcome, mind.feel_unit["relief"], EXPECT, 1)
        if tie >= 0:
            w_pos, w_neg = mass1(mind, tie)
            imitated_pos += w_pos
            imitated_neg += w_neg
    return expectation(imitated_pos, imitated_neg, mind.cfg.k)


def consequence(mind, act_unit: int) -> float:
    """L5: sum over value signals of sign x (2 e - 1) for the path "this episode, then this act", read from the
    EXPECT lag-2 ties that the episode trace collected when such a path was taken before (0 when never taken
    or no value signal is defined). Signs come from the adapter (mind.value_sign). F35 (``purpose``): what a path
    never taken would lead to is read from the paths its frames hold (``general_consequences``), and the path's own
    record comes last."""
    signs = getattr(mind, "value_sign", None)
    if not signs or not mind.episode:
        return 0.0
    S = mind.S
    parts = tuple(mind.episode) + (act_unit,)
    path = S.tuple_index.get(parts)
    values = []
    for (channel, symbol), sign in signs.items():
        if not mind.ch.has(channel):
            continue
        cid = mind.ch.id(channel)
        index = mind.ch.symbol_index(cid, symbol, create=False)
        value_unit = S.atom_index.get((cid, index), -1) if index >= 0 else -1
        if value_unit >= 0:
            values.append((value_unit, sign))
    if not values:
        return 0.0
    general = general_consequences(mind, parts, values) if getattr(mind.cfg, "purpose", False) else {}
    if path is None and not general:
        return 0.0
    total = 0.0
    for value_unit, sign in values:
        e = general.get(value_unit, 0.5)
        tie = S.find_tie(path, value_unit, EXPECT, 2) if path is not None else -1
        if tie >= 0:
            w_pos, w_neg = _in_company(mind, tie, *mass1(mind, tie))
            if general:                                    # the path's own record comes last
                c = w_pos + w_neg
                c = c / (c + mind.cfg.k)
                e = c * (w_pos / (w_pos + w_neg) if w_pos + w_neg > 0.0 else 0.5) + (1.0 - c) * e
            else:
                e = expectation(w_pos, w_neg, mind.cfg.k)
        elif not general:
            continue
        total += sign * (2.0 * e - 1.0)
    return total


def general_consequences(mind, parts, values) -> dict:
    """F35 (Generalised Consequences): what a path leads to, read as its frames' members lived it. For each frame of
    the path (an item of the path replaced by a category it is a member of, as the frames of a context are found),
    each member whose path was lived votes the share of its value evidence (lag-2 ties to the value signals) that each
    value received (F11):  W+(v) = sum_x e_x s_x(v),  W-(v) = sum_x e_x (1 - s_x(v)). The frame whose members agree
    most speaks: e(v) = c f + (1 - c) 1/2, c = lam e(path -> frame) conviction(members). {value unit: e}."""
    from .form import categories_of, parts_of
    S, k = mind.S, mind.cfg.k
    value_units = [v for v, _ in values]
    valued = _valued_paths(mind, value_units)           # the paths that led to good or bad: few
    if not valued:
        return {}
    kinds = mv_view(mind, "u_kind")
    best = None
    for j, part in enumerate(parts):
        if kinds[part] & (CATEGORY | SLOTTED):
            continue
        for category, e_abs in categories_of(mind, part):
            if e_abs <= 0.5:
                continue
            where = [i for i, q in enumerate(parts) if q == part] if getattr(S, "bind_repeats", False) else [j]
            if S.tuple_index.get(tuple(category if i in where else q for i, q in enumerate(parts))) is None:
                continue
            shares = {v: 0.0 for v in value_units}
            members = 0.0
            for instance, masses in valued.items():     # the valued paths that are this frame's instances
                held = parts_of(mind, instance)
                if len(held) != len(parts) or any(held[i] != q for i, q in enumerate(parts) if i not in where):
                    continue
                x = held[where[0]]
                if x == part or any(held[i] != x for i in where):
                    continue
                tie = S.find_tie(x, category, ABSTRACT, 0)
                if tie < 0:
                    continue
                total = sum(masses.get(v, 0.0) for v in value_units)
                if total <= 0.0:
                    continue
                e_x = expectation(*mass1(mind, tie), k)
                members += e_x
                for v in value_units:
                    shares[v] += e_x * masses.get(v, 0.0) / total
            if members <= 0.0:
                continue
            top = max(shares.values()) / members
            score = e_abs * conviction(members, k) * top
            if best is None or score > best[0]:
                best = (score, e_abs, members, shares)
    if best is None:
        return {}
    _, e_abs, members, shares = best
    c = mind.cfg.lam * e_abs * conviction(members, k)
    return {v: c * (w / members) + (1.0 - c) * 0.5 for v, w in shares.items()}


def _in_company(mind, tie: int, w_pos: float, w_neg: float):
    """F33 with F35: what a path led to with other people is theirs (what pleased one person is an expectation of that
    relationship); read in present company when the mind keeps its ledger, else as it is."""
    if getattr(mind.cfg, "company", False) and "personal" in mind.__dict__ and getattr(mind.cfg, "purpose", False):
        from .iliski import company_mass1
        return company_mass1(mind, tie, w_pos, w_neg, None)
    return w_pos, w_neg


def _valued_paths(mind, value_units) -> dict:
    """F35: the composed paths that were later followed by a value signal (EXPECT lag 2), with their positive mass
    toward each value: {path: {value unit: W+}}. Read once a moment."""
    from .read import _moment_memo
    memo = _moment_memo(mind)
    key = ("valued", tuple(value_units))
    if key in memo:
        return memo[key]
    S = mind.S
    kinds, src, lag = mv_view(mind, "u_kind"), mv_view(mind, "t_src"), mv_view(mind, "t_lag")
    from .store import COMPOSED
    found: dict = {}
    for v in value_units:
        for tie in S.inc.get((v, EXPECT), ()):
            u = src[tie]
            if lag[tie] != 2 or not kinds[u] & COMPOSED or kinds[u] & SLOTTED:
                continue
            w_pos = _in_company(mind, tie, *mass1(mind, tie))[0]
            if w_pos > 0.0:
                found.setdefault(u, {})[v] = w_pos
    memo[key] = found
    return found


def _explore(mind, ranked, theta: float, T: int):
    """Hesitation ends in a try (L5 with curiosity), where a try can teach something.

    Only when some alternative is well supported (it would have been chosen alone: 2 (e - 1/2) T >= theta) is the
    silence hesitation rather than not knowing. A try is made when the mind has just been called on (its turn begins:
    the last thing in the episode is a perceived event) or when some alternative's consequences are already known
    here (it has met good or bad on these paths); in the middle of its own utterance with nothing at stake it does
    not wander. It tries, among the well supported alternatives whose known consequences are not bad, the one whose
    consequences it knows least (the evidence on its path's value ties; a path never taken knows nothing). If every
    well supported alternative has led to bad, and only then, it tries among everything else it has lived after this
    (e > 1/10). Ties: better known consequence, then the race order. Returns the chosen (score, e, unit) or None."""
    if not any(2.0 * (r[1] - 0.5) * T >= theta for r in ranked):
        return None                                        # nothing well supported: not knowing, not hesitation
    S = mind.S
    known = {r[2]: _known_consequence(mind, r[2]) for r in ranked}
    at_stake = any(k > 0.0 for k in known.values()) or (           # good or bad was met on these paths, or in
        mind.context is not None and _value_evidence(mind, mind.context) > 0.0)   # this very situation before
    last = mind.episode[-1] if mind.episode else -1
    called_on = last >= 0 and kind_masks(mind)[4][mv_view(mind, "u_channel")[last]]
    if not (called_on or at_stake):
        return None
    value = {r[2]: consequence(mind, r[2]) for r in ranked}
    supported = [r for r in ranked if 2.0 * (r[1] - 0.5) * T >= theta and value[r[2]] >= 0.0]
    pool = supported or ([r for r in ranked if r[1] > 0.1] if at_stake else [])
    if not pool:
        return None
    return min(pool, key=lambda r: (known[r[2]], -value[r[2]], -r[0], r[2]))


def _known_consequence(mind, act_unit: int) -> float:
    """How much the mind knows of what this path led to: the evidence on the value ties of "this episode, then
    this act" (0 when the path was never taken)."""
    if not getattr(mind, "value_sign", None) or not mind.episode:
        return 0.0
    path = mind.S.tuple_index.get(tuple(mind.episode) + (act_unit,))
    return 0.0 if path is None else _value_evidence(mind, path)


def _value_evidence(mind, unit: int) -> float:
    """The evidence on a unit's EXPECT lag-2 ties to the value signals (what it was later followed by: good, bad)."""
    signs = getattr(mind, "value_sign", None)
    if not signs:
        return 0.0
    S = mind.S
    known = 0.0
    for (channel, symbol) in signs:
        if not mind.ch.has(channel):
            continue
        cid = mind.ch.id(channel)
        index = mind.ch.symbol_index(cid, symbol, create=False)
        value_unit = S.atom_index.get((cid, index), -1) if index >= 0 else -1
        tie = S.find_tie(unit, value_unit, EXPECT, 2) if value_unit >= 0 else -1
        if tie >= 0:
            known += mass1_total(mind, tie)
    return known


def _race(mind, candidates, theta: float, T: int) -> List[Event]:
    """F19: one act per channel per moment, so the candidates of a channel are alternatives that exclude each
    other. Each scores its expectation plus half its consequences (L5); the leader accumulates its advantage over
    the strongest rival (over 1/2, "saying nothing", when it has none) and is emitted if that advantage reaches the
    threshold within T iterations: rate = 2 (score_leader - score_rival). With no rival this is the rule as before
    (2e - 1); with rivals that are all equally (un)supported nothing leads and the mind stays silent."""
    S = mind.S
    channels = mv_view(mind, "u_channel")
    by_channel = {}
    for unit, e in candidates.items():
        score = e + 0.5 * consequence(mind, unit)
        by_channel.setdefault(channels[unit], []).append((score, e, unit))
    acts: List[Event] = []
    for channel in sorted(by_channel):
        ranked = sorted(by_channel[channel], reverse=True)
        score, e, unit = ranked[0]
        rival = ranked[1][0] if len(ranked) > 1 else 0.5
        if getattr(mind.cfg, "urge", False) and len(ranked) > 1 and rival > 0.5 + theta / (2.0 * T):
            u = _urge(mind, unit, ranked[1][2])        # F40: how far the two are things known about the same item
            rival = (1.0 - u) * rival + u * 0.5        # (another thing to say, not an alternative): against silence
        rate = 2.0 * (score - max(rival, 0.5 if score <= 0.5 else rival))
        if rate <= 0.0 or math.ceil(theta / rate - 1e-12) > T:
            tried = _explore(mind, ranked, theta, T) if getattr(mind.cfg, "explore", False) else None
            if tried is None:
                continue
            score, e, unit = tried
        acts.append(Event(mind.ch.names[channel], mind.ch.symbol(channel, int(S.u_symbol[unit])), ACTED))
        mind.hoping = (unit, e)
        if mind.__dict__.get("_explain_on"):           # the chat's !neden: where this word came from
            from .neden import attribute
            mind.__dict__.setdefault("_why", []).append(attribute(mind, unit, consequence(mind, unit)))
    return acts


def _urge(mind, a: int, b: int) -> float:
    """F40 (Urge to Go On): how far two expected words are two things known about the same item rather than two
    alternatives, u in [0, 1]: over the items of the present context, the largest share of an item's lived
    continuations on the words' channel (W+) that the two hold together. "uçar" and "fare" after "… baykuş" are most of
    what "baykuş" was lived to go on with (two facts of baykuş: saying either is right, they compete only for the
    order); "hamsi" and "levrek" after "… balıktır ." are almost nothing of what any item there went on with (only one
    can be right). The rival is then weighed as (1 - u) rival + u 1/2: the race of F19 for alternatives, a race against
    silence for things to say. A rival that is not itself over the speaking line is never weighed so (the mind would
    not have said it)."""
    from .form import parts_of
    from .read import _unit_arrays
    from .store import COMPOSED
    if mind.context is None:
        return 0.0
    channels = mv_view(mind, "u_channel")
    channel = channels[a]
    items = parts_of(mind, mind.context) if mv_view(mind, "u_kind")[mind.context] & COMPOSED else (mind.context,)
    best = 0.0
    for x in set(items):
        found = _unit_arrays(mind, x, 1)
        if found is None:
            continue
        dst, pos, _ = found
        total = both = 0.0
        hit_a = hit_b = False
        for v, p in zip(dst, pos):
            p = float(p)
            if p <= 0.0 or channels[v] != channel:      # what x went on with on the words' channel
                continue
            total += p
            if v == a:
                both += p
                hit_a = True
            elif v == b:
                both += p
                hit_b = True
        if hit_a and hit_b and total > 0.0:
            best = max(best, both / total)
    return best


def decide(mind) -> List[Event]:
    """Step 4: accumulate to threshold; at most one act per act channel; silence is legal."""
    S, cfg = mind.S, mind.cfg
    theta, T = mind.knob["theta"], cfg.T
    candidates = {}
    if mind.context is not None and cfg.support == "members" and cfg.bind_repeats:
        from .read import members_map                      # F12: speak with what the frame's members lived
        not_self = mind.trust.clone()
        not_self[ACTED] = 0.0                              # own repetitions are a record, not a reason (section 7)
        channels, act = mv_view(mind, "u_channel"), kind_masks(mind)[5]
        for unit, e in members_map(mind, mind.context, not_self).items():
            if act[channels[unit]] and (e > 0.5 or cfg.race):
                candidates[unit] = e
    elif mind.context is not None:
        only_imitated = torch.zeros(N_PROV, dtype=F64)
        only_imitated[IMITATED] = mind.trust[IMITATED]
        channels, act = mv_view(mind, "u_channel"), kind_masks(mind)[5]
        for unit, (w_pos, w_neg) in effective(mind, mind.context, trust=only_imitated).items():
            if act[channels[unit]] and w_pos > 0.0:
                candidates[unit] = act_expectation(mind, unit, w_pos, w_neg)
    elif cfg.knobs and mind.ch.has("knob"):               # knob acts: only their own outcomes speak
        channel = mind.ch.id("knob")
        for index in range(len(mind.ch.symbols[channel])):
            unit = S.atom_index.get((channel, index), -1)
            if unit >= 0 and S.find_tie(unit, mind.feel_unit["relief"], EXPECT, 1) >= 0:
                candidates[unit] = act_expectation(mind, unit, 0.0, 0.0)
    if cfg.race:
        return _race(mind, candidates, theta, T)
    best = {}
    for unit, e in candidates.items():
        rate = 2.0 * e - 1.0 + consequence(mind, unit)  # L5 will: the drive plus what this path led to
        if rate <= 0.0:
            continue
        latency = math.ceil(theta / rate - 1e-12)
        if latency > T:
            continue
        channel = int(S.u_channel[unit])
        entry = (latency, -e, unit)
        if channel not in best or entry < best[channel]:
            best[channel] = entry
    acts: List[Event] = []
    for channel in sorted(best):
        latency, neg_e, unit = best[channel]
        acts.append(Event(mind.ch.names[channel], mind.ch.symbol(channel, int(S.u_symbol[unit])), ACTED))
        mind.hoping = (unit, -neg_e)
    return acts


# ---------------------------------------------------------------------- control knobs (section 8, simple)
def apply_knob(mind, symbol: Hashable) -> None:
    """Setting a knob is an internal act: the echo of the act changes the knob."""
    name, direction = symbol
    if name == "F":
        low, high = KNOB_RANGE["F"]
        mind.knob["F"] = max(low, min(high, mind.knob["F"] + int(direction)))
    elif name in ("theta", "gain"):
        mind.knob[name] = mind.knob[name] * 2.0 ** (0.25 * int(direction))


def babble(mind) -> List[Event]:
    """Exploration of the knobs: nobody shows the mind how to turn them, so no imitated evidence can start
    them. Every ``babble_every`` idle moments the next knob act of a fixed round is emitted (deterministic)."""
    cfg = mind.cfg
    if cfg.babble_every <= 0 or mind.idle == 0 or mind.idle % cfg.babble_every:
        return []
    mind.add_channel("knob", "knob")
    symbol = KNOB_ACTS[mind.babbled % len(KNOB_ACTS)]
    mind.babbled += 1
    return [Event("knob", symbol, ACTED)]


# ---------------------------------------------------------------------- imagination (section 6, simple)
def imagine(mind) -> int:
    """Idle moment: construct conclusions from support alone and record them as imagined.

    Start from the units of the last focus with the highest drive (at most H). A conclusion is a tie i -> v that the
    instance does not hold itself but one generalisation g expects (chain i -> g -> v). Its reality score R
    is the expectation of i -> v from all evidence that is NOT on that chain (other generalisations; 1/2 when
    there is none). It is written with provenance ``imagined`` and weight iota * R. Later direct experience on
    the same tie judges it and so moves the trust in the mind's own constructions (``learn.judge``).
    """
    S, cfg = mind.S, mind.cfg
    starts = sorted((-drive_of(mind, u), u) for u, _, grounded in mind.last_focus
                    if grounded and not int(S.u_kind[u]) & (CATEGORY | SLOTTED))
    made = 0
    for _, instance in starts[:cfg.H]:
        own = own_mass(mind, instance)
        for general, _ in generalisations(mind, instance):
            for target, (w_pos, w_neg) in own_mass(mind, general).items():
                if expectation(w_pos, w_neg, cfg.k) <= 0.5 or sum(own.get(target, (0.0, 0.0))) > 0.0:
                    continue
                tie = S.find_tie(instance, target, EXPECT, 1)
                if tie >= 0 and float(S.w1(W_POS, tie)[IMAGINED] + S.w1(W_NEG, tie)[IMAGINED]) > 0.0:
                    continue
                rest = effective(mind, instance, skip=general).get(target, (0.0, 0.0))
                reality = expectation(rest[0], rest[1], cfg.k) if cfg.reality else 1.0
                add_evidence1(mind, W_POS, IMAGINED, S.tie(instance, target, EXPECT, 1), cfg.iota * reality)
                made += 1
    mind.imagined += made
    return made
