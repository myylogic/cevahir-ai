# -*- coding: utf-8 -*-
"""The one learning rule of Zihin (design C3, section 2 step 5, section 4 "up", section 1 on trust).

Everything here is local to ties whose source was in the focus, and every change is an additive, logged
increment. Three uses of the same rule:

* ``learn_follow``  EXPECT lag 1: what was in the last focus is tied to what occurs now; what a tie expected
  on a channel that reported something else gets negative evidence weighted by the tie's own expectation.
* ``learn_together`` EXPECT lag 0: grounded units that are in the focus together.
* induction "up": a generalisation (category or frame) in the last focus acquires, with provenance
  ``inferred``, what its instance acquired, each instance counting at most ``induct_cap``.

Learned trust. When direct experience (a reference class) adds evidence to a tie, the evidence of the other
classes on that tie that has not been judged yet is judged: what agrees with the direct experience is positive
evidence on the tie (source of that class -> "holds"), what disagrees is negative. Judged mass is recorded in
the ``checked`` tables so that every claim is judged once.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .events import INFERRED, REFERENCE, SENSED
from .evidence import conviction, expectation
from .store import ABSTRACT, C_NEG, C_POS, CATEGORY, EXPECT, F64, I64, SLOTTED, W_NEG, W_POS


def add_evidence(mind, table: int, slot: int, ids: torch.Tensor, amounts: torch.Tensor) -> torch.Tensor:
    """Add evidence through the store; keep the relief bookkeeping; let direct experience judge the rest."""
    S = mind.S
    if ids.numel() == 0:
        return amounts
    amounts = S.add(table, slot, ids, amounts)  # snapped to the grid, aligned with ids
    if table in (W_POS, W_NEG):
        mind._gained.append((ids, amounts * float(mind.trust[slot])))  # for relief (section 5)
        if slot in REFERENCE and mind.cfg.learn_trust:
            judge(mind, table, ids, amounts)
    return amounts


def add_evidence1(mind, table: int, slot: int, tie: int, amount: float) -> None:
    add_evidence(mind, table, slot, torch.tensor([tie], dtype=I64), torch.tensor([amount], dtype=F64))


def judge(mind, table: int, ids: torch.Tensor, amounts: torch.Tensor) -> None:
    """Direct experience on ties ``ids`` judges the not yet judged evidence of the learned-trust classes."""
    S = mind.S
    share = torch.clamp(amounts, max=1.0)  # one full direct experience judges everything outstanding
    for slot in range(S.P):
        if slot in REFERENCE or mind.source_tie[slot] < 0:
            continue
        open_pos = (S.t_w[W_POS, ids, slot] - S.t_w[C_POS, ids, slot]) * share
        open_neg = (S.t_w[W_NEG, ids, slot] - S.t_w[C_NEG, ids, slot]) * share
        if float(open_pos.sum()) == 0.0 and float(open_neg.sum()) == 0.0:
            continue
        open_pos = S.add(C_POS, slot, ids, open_pos)
        open_neg = S.add(C_NEG, slot, ids, open_neg)
        agree = float(open_pos.sum()) if table == W_POS else float(open_neg.sum())
        differ = float(open_neg.sum()) if table == W_POS else float(open_pos.sum())
        S.add1(W_POS, SENSED, mind.source_tie[slot], agree)
        S.add1(W_NEG, SENSED, mind.source_tie[slot], differ)


def tie_mass(mind, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trust-weighted (W+, W-) of ties."""
    S = mind.S
    return (S.t_w[W_POS, ids] * mind.trust).sum(-1), (S.t_w[W_NEG, ids] * mind.trust).sum(-1)


def learn_follow(mind, percepts: List[Tuple[int, float, int, int]]) -> None:
    """EXPECT lag 1 between the last focus and what occurs now (percepts: unit, intensity, slot, channel)."""
    S, cfg = mind.S, mind.cfg
    g = mind.knob["gain"]
    occurred: Dict[int, Tuple[float, int]] = {}
    slot_of_channel: Dict[int, int] = {}
    for unit, intensity, slot, channel in percepts:
        if unit not in occurred or intensity > occurred[unit][0]:
            occurred[unit] = (intensity, slot)
        slot_of_channel.setdefault(channel, slot)
    if not mind.last_focus or not occurred:
        return
    changes: Dict[int, Dict[int, list]] = {}  # instance -> target -> raw masses before/after and slots
    for unit, a_u, grounded in mind.last_focus:
        if not grounded or a_u <= 0.0 or int(S.u_kind[unit]) & (CATEGORY | SLOTTED):
            continue
        record: Dict[int, list] = {}
        ties = S.out.get((unit, EXPECT, 1), [])
        if ties:
            ids = torch.tensor(ties, dtype=I64)
            dst = S.t_dst[ids]
            w_pos, w_neg = tie_mass(mind, ids)
            e = expectation(w_pos, w_neg, cfg.k)
            hope = mind.act_hope.get(unit)
            if hope is not None:  # an own act is an expectation of relief (changed rule, see NOTES.md)
                e = torch.where(dst == mind.feel_unit["relief"], torch.clamp(e, min=hope), e)
            raw_pos, raw_neg = S.t_w[W_POS, ids].sum(-1), S.t_w[W_NEG, ids].sum(-1)
            channel = S.u_channel[dst]
            present = torch.tensor([occurred[d][0] if d in occurred else 0.0 for d in dst.tolist()], dtype=F64)
            weight = e if cfg.neg_weight == "e" else torch.ones_like(e)
            miss = g * a_u * weight * (1.0 - present)
            reporting = torch.zeros(ids.numel(), dtype=torch.bool)
            for ch, slot in slot_of_channel.items():
                on = channel == ch
                if bool(on.any()):
                    reporting |= on
                    got = add_evidence(mind, W_NEG, slot, ids[on], miss[on])
                    for d, rp, rn, m in zip(dst[on].tolist(), raw_pos[on].tolist(), raw_neg[on].tolist(),
                                            got.tolist()):
                        record[d] = [rp, rp, rn, rn + m, slot, slot]
            S.use_ties(ids[reporting])
        for v, (intensity, slot) in occurred.items():
            if intensity <= 0.0:
                continue
            tie = S.tie(unit, v, EXPECT, 1)
            if v not in record:
                rp, rn = float(S.t_w[W_POS, tie].sum()), float(S.t_w[W_NEG, tie].sum())
                record[v] = [rp, rp, rn, rn, slot, slot]
            got = add_evidence(mind, W_POS, slot, torch.tensor([tie], dtype=I64),
                               torch.tensor([g * a_u * intensity], dtype=F64))
            record[v][1] += float(got[0])
            record[v][4] = slot
        changes[unit] = record
    induce(mind, changes)


def induce(mind, changes: Dict[int, Dict[int, list]]) -> None:
    """Section 4 "up": a generalisation in the last focus acquires what its instance acquired.

    Each instance tie counts at most ``induct_cap`` (raw evidence), discounted by the expectation of the
    ABSTRACT relation. The evidence keeps the provenance class of the event that gave it to the instance.
    """
    S, cap = mind.S, mind.cfg.induct_cap
    for general, instance, e_abs in mind.last_gens:
        record = changes.get(instance)
        if not record or e_abs <= 0.5:
            continue
        for v, (pos0, pos1, neg0, neg1, slot_pos, slot_neg) in record.items():
            up = (min(pos1, cap) - min(pos0, cap)) * e_abs
            down = (min(neg1, cap) - min(neg0, cap)) * e_abs
            if up > 0.0:
                add_evidence1(mind, W_POS, slot_pos, S.tie(general, v, EXPECT, 1), up)
            if down > 0.0:
                tie = S.find_tie(general, v, EXPECT, 1)
                if tie >= 0:
                    add_evidence1(mind, W_NEG, slot_neg, tie, down)


def learn_together(mind, focus: List[Tuple[int, float, bool]], slot: int) -> None:
    """EXPECT lag 0 among the grounded units of the focus; expected but absent partners get negative evidence."""
    S, cfg = mind.S, mind.cfg
    g = mind.knob["gain"]
    present = {unit: a for unit, a, grounded in focus if grounded and a > 0.0}
    for unit, a_u in present.items():
        ties = S.out.get((unit, EXPECT, 0), [])
        if ties:
            ids = torch.tensor(ties, dtype=I64)
            dst = S.t_dst[ids]
            here = torch.tensor([present.get(d, 0.0) for d in dst.tolist()], dtype=F64)
            w_pos, w_neg = tie_mass(mind, ids)
            e = expectation(w_pos, w_neg, cfg.k)
            weight = e if cfg.neg_weight == "e" else torch.ones_like(e)
            add_evidence(mind, W_NEG, slot, ids, g * a_u * weight * (1.0 - here))
            S.use_ties(ids)
        for other, a_v in present.items():
            if other != unit:
                add_evidence1(mind, W_POS, slot, S.tie(unit, other, EXPECT, 0), g * a_u * a_v)
