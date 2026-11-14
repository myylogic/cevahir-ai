# -*- coding: utf-8 -*-
"""Reading the substance (design sections 2.3, 4 "down", 5): effective evidence, predictions, drive.

Nothing here writes. Support from a generalisation to an instance is computed when the instance is read, so
it always reflects the present evidence of the generalisation and needs no retraction of its own.

Effective evidence of an instance tie i -> v (section 4):

    as written ("raw"):   W_eff = W_own + lam * e(i -> g) * W(g -> v)          summed over generalisations g
    changed   ("capped"): W_eff = W_own + lam * e(i -> g) * W(g -> v) / max(1, W(g -> v) / k)

In the changed form one generalisation counts at most as much as lam * e(i -> g) lived experiences, however
much evidence the generalisation has collected (see NOTES.md, finding on exceptions).

Generalisations of a unit: its categories (ABSTRACT ties) and, for a composed unit, the existing composed
units that have one of its parts replaced by a category of that part (frames).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .evidence import conviction, expectation, frequency
from .form import categories_of
from .store import CATEGORY, COMPOSED, EXPECT, I64, SLOTTED, W_NEG, W_POS


def generalisations(mind, unit: int) -> List[Tuple[int, float]]:
    """(generalisation unit, expectation that ``unit`` is its instance), only where that exceeds 1/2."""
    S = mind.S
    found = categories_of(mind, unit)
    kind = int(S.u_kind[unit])
    if kind & COMPOSED and not kind & SLOTTED:
        parts = S.parts(unit)
        for pos, part in enumerate(parts):
            for category, e in categories_of(mind, part):
                frame = S.tuple_index.get(parts[:pos] + (category,) + parts[pos + 1:])
                if frame is not None:
                    found.append((frame, e))
    return [(general, e) for general, e in found if e > 0.5]


def own_mass(mind, unit: int, lag: int = 1, trust: Optional[torch.Tensor] = None) -> Dict[int, Tuple[float, float]]:
    """Trust-weighted (W+, W-) of the EXPECT ties leaving ``unit``, by target."""
    S = mind.S
    ties = S.out.get((unit, EXPECT, lag), [])
    if not ties:
        return {}
    ids = torch.tensor(ties, dtype=I64)
    trust = mind.trust if trust is None else trust
    w_pos = (S.t_w[W_POS, ids] * trust).sum(-1).tolist()
    w_neg = (S.t_w[W_NEG, ids] * trust).sum(-1).tolist()
    return {d: (p, n) for d, p, n in zip(S.t_dst[ids].tolist(), w_pos, w_neg)}


def effective(mind, unit: int, trust: Optional[torch.Tensor] = None, support: Optional[str] = None,
              skip: Optional[int] = None) -> Dict[int, Tuple[float, float]]:
    """Effective (W+, W-) of the lag-1 ties of ``unit``: own evidence plus support from its generalisations.

    ``skip`` leaves one generalisation out (used by the reality score of imagination: leave-path-out).
    """
    cfg = mind.cfg
    support = cfg.support if support is None else support
    mass = own_mass(mind, unit, 1, trust)
    kind = int(mind.S.u_kind[unit])
    if support == "none" or kind & (CATEGORY | SLOTTED):
        return mass
    for general, e_abs in generalisations(mind, unit):
        if general == skip:
            continue
        for target, (w_pos, w_neg) in own_mass(mind, general, 1, trust).items():
            factor = cfg.lam * e_abs
            if support == "capped":
                factor /= max(1.0, (w_pos + w_neg) / cfg.k)
            old = mass.get(target, (0.0, 0.0))
            mass[target] = (old[0] + factor * w_pos, old[1] + factor * w_neg)
    return mass


def predictions(mind, mode: Optional[str] = None) -> Dict[int, float]:
    """Expectations for the next moment (section 2.3).

    "votes" (as written): every unit of the focus votes a_u * (2 e - 1) for the targets of its lag-1 ties and
    the votes are summed, as in the settling formula. "specific" (changed): the most specific unit of the
    present context (the composed unit of the episode, or its single item) is read with its support.
    """
    cfg = mind.cfg
    mode = cfg.predict if mode is None else mode
    if mode == "specific":
        if mind.context is None:
            return {}
        return {v: expectation(p, n, cfg.k) for v, (p, n) in effective(mind, mind.context).items()}
    votes: Dict[int, float] = {}
    for unit, a_u, grounded in mind.focus:
        if a_u <= 0.0:
            continue
        for v, (p, n) in effective(mind, unit).items():
            votes[v] = votes.get(v, 0.0) + a_u * (2.0 * expectation(p, n, cfg.k) - 1.0)
    return {v: min(1.0, max(0.0, 0.5 + 0.5 * vote)) for v, vote in votes.items()}


def belief(mind, unit: int, target: int) -> Dict[str, float]:
    """White-box reading of one lag-1 tie: own and effective frequency, conviction, expectation."""
    cfg = mind.cfg
    own = own_mass(mind, unit).get(target, (0.0, 0.0))
    eff = effective(mind, unit).get(target, (0.0, 0.0))
    out = {}
    for name, (p, n) in (("own", own), ("eff", eff)):
        out[name + "_pos"], out[name + "_neg"] = p, n
        out[name + "_f"] = frequency(p, n)
        out[name + "_c"] = conviction(p + n, cfg.k)
        out[name + "_e"] = expectation(p, n, cfg.k)
    return out


def drive_of(mind, unit: int) -> float:
    """D(u) = sum over the lag-1 ties of u of use * (1 - conviction) (section 5)."""
    S = mind.S
    ties = S.out.get((unit, EXPECT, 1), [])
    if not ties:
        return 0.0
    ids = torch.tensor(ties, dtype=I64)
    total = ((S.t_w[W_POS, ids] + S.t_w[W_NEG, ids]) * mind.trust).sum(-1)
    return float((S.t_use[ids] * (1.0 - conviction(total, mind.cfg.k))).sum())


def total_drive(mind) -> float:
    """Drive over the recently used units: the units of the present focus."""
    return sum(drive_of(mind, unit) for unit, a, grounded in mind.focus if grounded)


def conflict_of(mind, unit: int) -> float:
    """Mass of the lag-1 ties of ``unit`` with frequency near 1/2 and high conviction (section 2.6)."""
    total = 0.0
    for p, n in own_mass(mind, unit).values():
        total += conviction(p + n, mind.cfg.k) * (1.0 - abs(2.0 * frequency(p, n) - 1.0))
    return total
