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


def decide(mind) -> List[Event]:
    """Step 4: accumulate to threshold; at most one act per act channel; silence is legal."""
    S, cfg = mind.S, mind.cfg
    theta, T = mind.knob["theta"], cfg.T
    candidates = {}
    if mind.context is not None and cfg.support == "members" and cfg.bind_repeats:
        from .read import members_map                      # F12: speak with what the frame's members lived
        for unit, e in members_map(mind, mind.context).items():
            channel = int(S.u_channel[unit])
            if channel >= 0 and mind.ch.kind(channel) == "act" and e > 0.5:
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
    best = {}
    for unit, e in candidates.items():
        rate = 2.0 * e - 1.0
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
