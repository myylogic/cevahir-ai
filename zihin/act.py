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
from .evidence import expectation
from .learn import add_evidence1
from .read import drive_of, effective, generalisations, own_mass
from .store import CATEGORY, EXPECT, F64, SLOTTED, W_NEG, W_POS

KNOB_ACTS = (("F", 1), ("F", -1), ("theta", 1), ("theta", -1), ("gain", 1), ("gain", -1))
KNOB_RANGE = {"F": (3, 15)}


def act_expectation(mind, act_unit: int, imitated_pos: float, imitated_neg: float) -> float:
    """Pooled expectation of doing ``act_unit`` now: imitated evidence plus the mind's own outcomes."""
    S = mind.S
    outcome = act_unit if mind.context is None else S.tuple_index.get(tuple(mind.episode) + (act_unit,))
    if outcome is not None:
        tie = S.find_tie(outcome, mind.feel_unit["relief"], EXPECT, 1)
        if tie >= 0:
            imitated_pos += float((S.t_w[W_POS, tie] * mind.trust).sum())
            imitated_neg += float((S.t_w[W_NEG, tie] * mind.trust).sum())
    return expectation(imitated_pos, imitated_neg, mind.cfg.k)


def consequence(mind, act_unit: int) -> float:
    """L5: sum over value signals of sign x (2 e - 1) for the path "this episode, then this act", read from the
    EXPECT lag-2 ties that the episode trace collected when such a path was taken before (0 when never taken
    or no value signal is defined). Signs come from the adapter (mind.value_sign)."""
    signs = getattr(mind, "value_sign", None)
    if not signs or not mind.episode:
        return 0.0
    S = mind.S
    path = S.tuple_index.get(tuple(mind.episode) + (act_unit,))
    if path is None:
        return 0.0
    total = 0.0
    for (channel, symbol), sign in signs.items():
        if not mind.ch.has(channel):
            continue
        cid = mind.ch.id(channel)
        index = mind.ch.symbol_index(cid, symbol, create=False)
        value_unit = S.atom_index.get((cid, index), -1) if index >= 0 else -1
        if value_unit < 0:
            continue
        tie = S.find_tie(path, value_unit, EXPECT, 2)
        if tie >= 0:
            w_pos = float((S.t_w[W_POS, tie] * mind.trust).sum())
            w_neg = float((S.t_w[W_NEG, tie] * mind.trust).sum())
            total += sign * (2.0 * expectation(w_pos, w_neg, mind.cfg.k) - 1.0)
    return total


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
    called_on = last >= 0 and int(S.u_channel[last]) >= 0 and mind.ch.kind(int(S.u_channel[last])) == "sense"
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
            known += float(((S.t_w[W_POS, tie] + S.t_w[W_NEG, tie]) * mind.trust).sum())
    return known


def _race(mind, candidates, theta: float, T: int) -> List[Event]:
    """F19: one act per channel per moment, so the candidates of a channel are alternatives that exclude each
    other. Each scores its expectation plus half its consequences (L5); the leader accumulates its advantage over
    the strongest rival (over 1/2, "saying nothing", when it has none) and is emitted if that advantage reaches the
    threshold within T iterations: rate = 2 (score_leader - score_rival). With no rival this is the rule as before
    (2e - 1); with rivals that are all equally (un)supported nothing leads and the mind stays silent."""
    S = mind.S
    by_channel = {}
    for unit, e in candidates.items():
        score = e + 0.5 * consequence(mind, unit)
        by_channel.setdefault(int(S.u_channel[unit]), []).append((score, e, unit))
    acts: List[Event] = []
    for channel in sorted(by_channel):
        ranked = sorted(by_channel[channel], reverse=True)
        score, e, unit = ranked[0]
        rival = ranked[1][0] if len(ranked) > 1 else 0.5
        rate = 2.0 * (score - max(rival, 0.5 if score <= 0.5 else rival))
        if rate <= 0.0 or math.ceil(theta / rate - 1e-12) > T:
            tried = _explore(mind, ranked, theta, T) if getattr(mind.cfg, "explore", False) else None
            if tried is None:
                continue
            score, e, unit = tried
        acts.append(Event(mind.ch.names[channel], mind.ch.symbol(channel, int(S.u_symbol[unit])), ACTED))
        mind.hoping = (unit, e)
    return acts


def decide(mind) -> List[Event]:
    """Step 4: accumulate to threshold; at most one act per act channel; silence is legal."""
    S, cfg = mind.S, mind.cfg
    theta, T = mind.knob["theta"], cfg.T
    candidates = {}
    if mind.context is not None and cfg.support == "members" and cfg.bind_repeats:
        from .read import members_map                      # F12: speak with what the frame's members lived
        not_self = mind.trust.clone()
        not_self[ACTED] = 0.0                              # own repetitions are a record, not a reason (section 7)
        for unit, e in members_map(mind, mind.context, not_self).items():
            channel = int(S.u_channel[unit])
            if channel >= 0 and mind.ch.kind(channel) == "act" and (e > 0.5 or cfg.race):
                candidates[unit] = e
    elif mind.context is not None:
        only_imitated = torch.zeros(N_PROV, dtype=F64)
        only_imitated[IMITATED] = mind.trust[IMITATED]
        for unit, (w_pos, w_neg) in effective(mind, mind.context, trust=only_imitated).items():
            channel = int(S.u_channel[unit])
            if channel >= 0 and mind.ch.kind(channel) == "act" and w_pos > 0.0:
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
                if tie >= 0 and float(S.t_w[W_POS, tie, IMAGINED] + S.t_w[W_NEG, tie, IMAGINED]) > 0.0:
                    continue
                rest = effective(mind, instance, skip=general).get(target, (0.0, 0.0))
                reality = expectation(rest[0], rest[1], cfg.k) if cfg.reality else 1.0
                add_evidence1(mind, W_POS, IMAGINED, S.tie(instance, target, EXPECT, 1), cfg.iota * reality)
                made += 1
    mind.imagined += made
    return made
