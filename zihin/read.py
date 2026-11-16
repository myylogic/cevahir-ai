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

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .evidence import conviction, expectation, frequency
from .form import (SMALL, categories_of, expectations, kind_masks, mass1, mass1_total, masses, masses_py, mv_view,
                   np_view, parts_of, trust_entry)
from .store import ABSTRACT, ATOMIC, CATEGORY, COMPOSED, EXPECT, F64, I64, PART, RELY, SLOTTED, W_NEG, W_POS


def _memo_key(mind, kind: str, unit: int, lag: int, trust: Optional[torch.Tensor]):
    """The open read memo of the mind and a key for one reading, or (None, None). ``Mind.step`` opens the memo for
    a stretch of the moment in which nothing is written (predicting and deciding, and feeling when no writer ran),
    so a reading that depends only on the substance, the trust and the channels is the same each time it is made
    there; ``trust`` enters the key by its bytes."""
    memo = mind.__dict__.get("_read_memo")
    if memo is None:
        return None, None
    return memo, (kind, unit, lag, trust_entry(mind, mind.trust if trust is None else trust)[0])


def generalisations(mind, unit: int) -> List[Tuple[int, float]]:
    """(generalisation unit, expectation that ``unit`` is its instance), only where that exceeds 1/2."""
    memo, key = _memo_key(mind, "gen", unit, 0, None)
    if memo is not None:
        hit = memo.get(key)
        if hit is not None:
            return list(hit)
    found = _generalisations(mind, unit)
    if memo is not None:
        memo[key] = tuple(found)
    return found


def _generalisations(mind, unit: int) -> List[Tuple[int, float]]:
    S = mind.S
    found = categories_of(mind, unit)
    kind = mv_view(mind, "u_kind")[unit]
    if kind & COMPOSED and not kind & SLOTTED:
        parts = parts_of(mind, unit)
        for pos, part in enumerate(parts):
            for category, e in categories_of(mind, part):
                if getattr(S, "bind_repeats", False):
                    frame = S.tuple_index.get(tuple(category if q == part else q for q in parts))
                else:
                    frame = S.tuple_index.get(parts[:pos] + (category,) + parts[pos + 1:])
                if frame is not None:
                    found.append((frame, e))
    return [(general, e) for general, e in found if e > 0.5]


def own_mass(mind, unit: int, lag: int = 1, trust: Optional[torch.Tensor] = None,
             whole: bool = False) -> Dict[int, Tuple[float, float]]:
    """Trust-weighted (W+, W-) of the EXPECT ties leaving ``unit``, by target (recalled in present company, F33; with
    ``whole``, the whole life's evidence, whoever it was lived with)."""
    found = _unit_arrays(mind, unit, lag, trust, whole)
    if found is None:
        return {}
    dst, w_pos, w_neg = found
    if isinstance(dst, list):
        return dict(zip(dst, zip(w_pos, w_neg)))
    return dict(zip(dst.tolist(), zip(w_pos.tolist(), w_neg.tolist())))


def _unit_arrays(mind, unit: int, lag: int = 1, trust: Optional[torch.Tensor] = None, whole: bool = False):
    """(targets, W+, W-) of ``own_mass`` in tie order (lists for a few ties, read-only arrays for more), or None when
    ``unit`` has no EXPECT tie of this lag."""
    S = mind.S
    ties = S.out.get((unit, EXPECT, lag))
    if not ties:
        return None
    company = getattr(mind.cfg, "company", False) and "personal" in mind.__dict__ and not whole
    memo, key = _memo_key(mind, "own" if company or not whole else "whole", unit, lag, trust)
    if memo is not None:
        hit = memo.get(key)
        if hit is not None:
            return hit
    rivals = lag == 1 and getattr(mind.cfg, "rivals", False)
    if len(ties) <= SMALL and not rivals:
        dst = mv_view(mind, "t_dst")
        w_pos, w_neg = masses_py(mind, ties, trust)
        if company:                                    # F33: recalled in present company
            from .iliski import company_view
            seen = company_view(mind, unit, EXPECT, lag, ties, w_pos, w_neg, trust)
            if seen is not None:
                w_pos, w_neg = seen
        found = ([dst[t] for t in ties], w_pos, w_neg)
    else:
        ids, dst = S.out_ids((unit, EXPECT, lag))
        w = masses(mind, ids, trust)
        w_pos, w_neg = w[0], w[1]
        if company:                                    # F33: recalled in present company
            from .iliski import company_view
            seen = company_view(mind, unit, EXPECT, lag, ties, w_pos, w_neg, trust)
            if seen is not None:
                w_pos, w_neg = np.array(seen[0]), np.array(seen[1])
        if rivals:
            w_pos, w_neg = rival_mass(mind, dst, torch.from_numpy(w_pos), torch.from_numpy(w_neg))
            w_pos, w_neg = w_pos.numpy(), w_neg.numpy()
        found = (dst.numpy(), w_pos, w_neg)
        for a in found:
            a.flags.writeable = False
    if memo is not None:
        memo[key] = found
    return found


def _world_pos(mind, unit: int, trust: Optional[torch.Tensor] = None):
    """(targets, W+) of the lag-1 ties of ``unit`` on sense/act channels with W+ > 0, in tie order: the items of
    {v: p for v, (p, n) in own_mass(mind, unit, 1, trust).items() if _world(mind, v) and p > 0.0}."""
    memo, key = _memo_key(mind, "world+", unit, 1, trust)
    if memo is not None:
        hit = memo.get(key)
        if hit is not None:
            return hit
    found = _unit_arrays(mind, unit, 1, trust)
    if found is None:
        out = ((), ())
    elif isinstance(found[0], list):
        world, channel = kind_masks(mind)[3], mv_view(mind, "u_channel")
        pairs = [(v, p) for v, p in zip(found[0], found[1]) if world[channel[v]] and p > 0.0]
        out = (tuple(v for v, _ in pairs), tuple(p for _, p in pairs))
    else:
        d, p, _ = found
        keep = kind_masks(mind)[0][np_view(mind, "u_channel")[d]] & (p > 0.0)
        out = (tuple(d[keep].tolist()), tuple(p[keep].tolist()))
    if memo is not None:
        memo[key] = out
    return out


def _continuing(mind, unit: int, exclude=()) -> set:
    """The targets of ``continuations(mind, unit, exclude)`` (its keys), without their expectations."""
    targets, _ = _world_pos(mind, unit)
    exclude = set(exclude)
    return {v for v in targets if v != unit and v not in exclude}


def start_mass(mind, unit: int, trust: Optional[torch.Tensor] = None) -> Dict[int, Tuple[float, float]]:
    """F26 (Sequence Starts): (W+, W-) of "a sequence that starts with ``unit`` goes on with v", by world target v.

    An episode begins after a pause, and its context after two items is the composed unit (unit, v): that unit is
    recognised once in every episode that began with ``unit`` and then v, and each recognition is evidence on its PART
    ties (form.confirm_parts). Their trust-weighted mass B(v) counts those beginnings. What began with ``unit`` and
    went on otherwise counts against v, as its rivals' share (the reading F25 b gives lag-1 ties):
        W+(v) = B(v),   W-(v) = sum over v' != v of B(v')
    The atom's own lag-1 ties hold what followed ``unit`` anywhere; this holds what followed it where a sequence began
    with it (a sentence read, an utterance after silence). Nothing new is stored for it; reads only."""
    memo, key = _memo_key(mind, "start", unit, 0, trust)
    if memo is not None:
        hit = memo.get(key)
        if hit is not None:
            return dict(hit)
    S = mind.S
    kinds, arity = mv_view(mind, "u_kind"), mv_view(mind, "u_arity")
    src, lag = mv_view(mind, "t_src"), mv_view(mind, "t_lag")
    world, channel = kind_masks(mind)[3], mv_view(mind, "u_channel")
    began: Dict[int, float] = {}
    company = getattr(mind.cfg, "company", False) and "personal" in mind.__dict__
    if company:
        from .iliski import company_mass1
    for tie in S.inc.get((unit, PART), ()):
        c = src[tie]
        if lag[tie] != 0 or arity[c] != 2 or kinds[c] & SLOTTED or not kinds[c] & COMPOSED:
            continue
        v = parts_of(mind, c)[1]
        if v == unit or not world[channel[v]]:
            continue
        w_pos, w_neg = mass1(mind, tie, trust)
        if company:                                      # F33: beginnings lived with others are theirs
            w_pos, w_neg = company_mass1(mind, tie, w_pos, w_neg, trust)
        if w_pos > 0.0:
            began[v] = began.get(v, 0.0) + w_pos
    total = sum(began.values())
    out = {v: (b, max(0.0, total - b)) for v, b in began.items()}
    if memo is not None:
        memo[key] = tuple(out.items())
    return out


def excluded(mind, out: Dict[int, float], own: Dict[int, Tuple[float, float]]) -> None:
    """F27 (Lived Exclusion): one item per channel per moment, so what the instance itself was lived to be followed by
    on a channel is evidence against every target there that it was never followed by. A target that a generalisation
    proposes (it is in ``out``) and that the instance never went on with, on a world channel where the instance went
    on with something, is read with W+ = 0 and W- = the instance's own positive mass on that channel (the full record:
    each of those moments was a moment it was not followed by the target), nested last as own evidence is:
        e <- c * 0 + (1 - c) e,   c = conviction(W-)
    Targets the instance has its own evidence on keep that evidence (F14 records their failures); nothing is written."""
    k = mind.cfg.k
    channel = mv_view(mind, "u_channel")
    totals: Dict[int, float] = {}
    for target, (w_pos, _) in own.items():
        if w_pos > 0.0 and _world(mind, target):
            totals[channel[target]] = totals.get(channel[target], 0.0) + w_pos
    if not totals:
        return
    for target in list(out):
        if target in own or not _world(mind, target):
            continue
        total = totals.get(channel[target], 0.0)
        if total > 0.0:
            out[target] = (1.0 - conviction(total, k)) * out[target]


def corroborated(mind, unit: int, trust, frame: int, claims: Dict[int, Tuple[float, float, float]],
                 out: Dict[int, float], tail: Optional[int] = None) -> None:
    """F30 (Corroboration): independent lived evidence about the same claim adds up (the revision rule of NARS).
    The claims of the frame that spoke (for each value: its conviction and frequency) are revised with what the lived
    situations most like the present one (F23) say about the same value, counting only lived situations the frame's
    reading did not count (its members' instances and every frame are left out, so no lived context is counted twice):
        w = k c / (1 - c) for each reading;  W+ = f1 w1 + f2 w2,  W- = (1 - f1) w1 + (1 - f2) w2
    f2 is the value's share of the situations' votes: situations that went on otherwise count against it. Values only
    the situations claim are left to F23's own reading (it speaks when nothing else does). Writes into ``out``."""
    k = mind.cfg.k
    focus = None
    if getattr(mind.cfg, "replies", False):              # F34: about the claim's own context
        n, frame_parts = len(parts_of(mind, unit)), parts_of(mind, frame)
        category = frame_category(mind, frame)
        focus = {n - 1 - j for j, p in enumerate(frame_parts) if p != category}
        for i, entry in enumerate(tail or ()):
            if entry >= 0:
                focus.add(n - 1 - (len(frame_parts) + i))
        focus = frozenset(d for d in focus if d >= 0)
    c_sit, shares = situation_evidence(mind, unit, trust, frozenset(_member_instances(mind, frame, unit, tail)),
                                       lived_only=True, focus=focus)
    if c_sit <= 0.0:
        return
    w2 = k * c_sit / (1.0 - c_sit)
    for target, (_, held, f) in claims.items():
        w1 = k * held / (1.0 - held)
        f2 = shares.get(target, 0.0)
        out[target] = expectation(f * w1 + f2 * w2, (1.0 - f) * w1 + (1.0 - f2) * w2, k)


def _begins(mind, unit: int) -> bool:
    """F26: ``unit`` is the present context and the whole present episode, one item after a pause: the moment is the
    beginning of a sequence."""
    episode = getattr(mind, "episode", None)
    return (unit == getattr(mind, "context", None) and episode is not None and len(episode) == 1
            and episode[0] == unit and bool(mv_view(mind, "u_kind")[unit] & ATOMIC))


def begins_with(mind, unit: int, v: int) -> bool:
    """F26: a sequence was lived that began with ``unit`` and then ``v`` (the composed unit of the two exists)."""
    found = mind.S.tuple_index.get((unit, v))
    return found is not None and not mv_view(mind, "u_kind")[found] & SLOTTED


def start_continuations(mind, unit: int, exclude=()) -> Dict[int, float]:
    """F26: expectations of what a sequence that starts with ``unit`` goes on with (``start_mass`` by the one rule),
    without ``exclude`` (the fixed words of a frame)."""
    k = mind.cfg.k
    return {v: expectation(p, n, k) for v, (p, n) in start_mass(mind, unit).items() if v not in exclude}


def rival_mass(mind, dst: torch.Tensor, w_pos: torch.Tensor, w_neg: torch.Tensor):
    """F25 (b), the normalised reading: for a target on a sense or act channel the negative mass is the positive mass
    of the source's other targets on that channel, so f(v | x) = W+(x -> v) / sum over v' of W+(x -> v') and the
    conviction is that of everything x was followed by there (rivals share one mass instead of each holding its own).
    Targets on the mind's own channels keep their own negative evidence."""
    S = mind.S
    n = len(mind.ch.names)
    channel = S.u_channel[dst].long()
    channel = torch.where(channel >= 0, channel, torch.full_like(channel, n))   # no channel: a slot of its own
    world = torch.tensor([mind.ch.kind(c) in ("sense", "act") for c in range(n)] + [False], dtype=torch.bool)
    total = torch.zeros(n + 1, dtype=w_pos.dtype).index_add_(0, channel, w_pos)
    return w_pos, torch.where(world[channel], (total[channel] - w_pos).clamp(min=0.0), w_neg)


def effective(mind, unit: int, trust: Optional[torch.Tensor] = None, support: Optional[str] = None,
              skip: Optional[int] = None) -> Dict[int, Tuple[float, float]]:
    """Effective (W+, W-) of the lag-1 ties of ``unit``: own evidence plus support from its generalisations.

    ``skip`` leaves one generalisation out (used by the reality score of imagination: leave-path-out).
    """
    cfg = mind.cfg
    support = cfg.support if support is None else support
    if support in ("nested", "choice", "reliable", "trusted", "members"):                    # masses are only defined for the additive forms
        support = "capped"
    mass = own_mass(mind, unit, 1, trust)
    kind = mv_view(mind, "u_kind")[unit]
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


def extension(mind, general: int) -> int:
    """How many instances a generalisation covers: the fillers of its category (of the category that stands
    in, for a frame). Smaller = more specific."""
    S = mind.S
    kinds = mv_view(mind, "u_kind")
    if kinds[general] & CATEGORY:
        return len(S.inc.get((general, ABSTRACT), ()))
    sizes = [len(S.inc.get((p, ABSTRACT), ())) for p in parts_of(mind, general) if kinds[p] & CATEGORY]
    return min(sizes) if sizes else 0


def expect_map(mind, unit: int) -> Dict[int, float]:
    """Effective expectation of every lag-1 tie of ``unit``, by target.

    With support "raw" / "capped" / "none" this is the expectation of the effective masses (section 4 as
    written, or capped). With support "nested" (changed, see NOTES.md) the generalisations are read from the
    most general to the most specific, each one moving the estimate towards its own frequency in proportion
    to how much it has lived, and the instance's own evidence comes last:

        e <- 1/2;   for g general -> specific:  e <- c_g f_g + (1 - c_g) e,   c_g = lam e(i -> g) conviction_g
                    own evidence:               e <- c_own f_own + (1 - c_own) e

    Without any generalisation this is exactly e = c (f - 1/2) + 1/2. The form is idempotent for copies of
    the same category, lets the specific override the general, and lets own evidence override both.
    """
    cfg = mind.cfg
    general = mv_view(mind, "u_kind")[unit] & (CATEGORY | SLOTTED)
    if cfg.support == "members" and not general:
        return members_map(mind, unit)
    if cfg.support == "trusted" and not general:
        return trusted_map(mind, unit)
    if cfg.support == "reliable" and not general:
        return reliable_map(mind, unit)
    if cfg.support == "choice" and not general:
        return choice_map(mind, unit)
    if cfg.support != "nested" or general:
        return {v: expectation(p, n, cfg.k) for v, (p, n) in effective(mind, unit).items()}
    own = own_mass(mind, unit)
    gens = sorted(((extension(mind, g), g, e_abs) for g, e_abs in generalisations(mind, unit)),
                  key=lambda x: (-x[0], x[1]))
    out: Dict[int, float] = {}
    for _, general, e_abs in gens:
        for target, (w_pos, w_neg) in own_mass(mind, general).items():
            c = cfg.lam * e_abs * conviction(w_pos + w_neg, cfg.k)
            out[target] = c * frequency(w_pos, w_neg) + (1.0 - c) * out.get(target, 0.5)
    for target, (w_pos, w_neg) in own.items():
        c = conviction(w_pos + w_neg, cfg.k)
        out[target] = c * frequency(w_pos, w_neg) + (1.0 - c) * out.get(target, 0.5)
    return out


def reliability(mind, general: int) -> float:
    """Expectation of the RELY self-tie of a generalisation (F9); 1/2 when it has never been judged."""
    S = mind.S
    ties = S.out.get((general, RELY, 0), [])
    if not ties:
        return 0.5
    ids = torch.tensor(ties, dtype=I64)
    pos, neg = S.w2(ids)
    w_pos = float((pos * mind.trust).sum())
    w_neg = float((neg * mind.trust).sum())
    return expectation(w_pos, w_neg, mind.cfg.k)


def reliable_map(mind, unit: int) -> Dict[int, float]:
    """support = "reliable" (F9): the generalisation that has best predicted what its instances lived speaks.

        g* = argmax over generalisations with sense/act targets and reliability(g) > 1/2 of  e(i -> g) reliability(g)
        (none such: no generalisation speaks; what is not known stays at 1/2, "I do not know")
        e <- c f + (1 - c) 1/2 for every target of g*,  c = lam e(i -> g*) conviction(g* -> v)
        own evidence of the instance last:  e <- c_own f_own + (1 - c_own) e
    """
    cfg, S = mind.cfg, mind.S
    best = None
    for general, e_abs in generalisations(mind, unit):
        mass = {t: pn for t, pn in own_mass(mind, general).items() if _world(mind, t)}
        if not mass:
            continue
        r = reliability(mind, general)
        if mind.cfg.proven_only and r <= 0.5:              # F10b (measured worse): unproven ones have no voice
            continue
        score = e_abs * r
        if best is None or score > best[0]:
            best = (score, e_abs, general)
    out: Dict[int, float] = {}
    if best is not None:
        _, e_abs, general = best
        for target, (w_pos, w_neg) in own_mass(mind, general).items():
            c = cfg.lam * e_abs * conviction(w_pos + w_neg, cfg.k)
            out[target] = c * frequency(w_pos, w_neg) + (1.0 - c) * 0.5
    for target, (w_pos, w_neg) in own_mass(mind, unit).items():
        c = conviction(w_pos + w_neg, cfg.k)
        out[target] = c * frequency(w_pos, w_neg) + (1.0 - c) * out.get(target, 0.5)
    return out


def _world(mind, unit: int) -> bool:
    """The unit is on a sense or act channel."""
    return kind_masks(mind)[3][mv_view(mind, "u_channel")[unit]]


SELF, NEXT = -1, -2   # F12 relational targets of a frame: "the filler of my slot", "what the filler goes on with"
START = -3            # F26: "what a sequence that starts with the filler goes on with" (more specific than NEXT)
ALIKE = -4            # F31: "what stands where the filler stands" (the paradigmatic axis; NEXT and START are syntagmatic)
REPLY = -5            # F34: "what the filler, said on its own, was answered with" (the continuation of filler + event)


def resolve(mind, target: int, filler: Optional[int], fixed, category: Optional[int] = None) -> Optional[int]:
    """A frame's relational target read for one instance: SELF is the filler itself, START the strongest continuation
    of sequences that began with the filler (F26), NEXT the filler's own strongest continuation (F12), ALIKE the item
    that stands most where the filler stands (F31, ``stand_in``; not one of the rival fillers of the frame's own place,
    ``category``); the fixed words of the frame are never what the relation names. Any other target is itself."""
    if target == SELF:
        return filler
    if target == ALIKE:
        return stand_in(mind, filler, fixed, category) if filler is not None else None
    if target == REPLY:
        return reply_of(mind, filler, fixed) if filler is not None else None
    if target in (NEXT, START):
        if filler is None:
            return None
        memo = mind.__dict__.get("_read_memo")         # a moment's reading resolves the same member many times
        key = ("resolve", target, filler, tuple(fixed))
        if memo is not None and key in memo:
            return memo[key]
        goes_on = start_continuations(mind, filler, fixed) if target == START else continuations(mind, filler, fixed)
        found = max(goes_on, key=goes_on.get) if goes_on else None
        if memo is not None:
            memo[key] = found
        return found
    return target


def continuations(mind, unit: int, exclude=()) -> Dict[int, float]:
    """Own lag-1 expectations of a unit on sense/act targets, without ``exclude`` (the fixed words of a frame)."""
    found = _unit_arrays(mind, unit)
    if found is None:
        return {}
    d, w_pos, w_neg = found
    if isinstance(d, list):
        world, channel, k = kind_masks(mind)[3], mv_view(mind, "u_channel"), mind.cfg.k
        return {v: expectation(p, n, k) for v, p, n in zip(d, w_pos, w_neg)
                if world[channel[v]] and v not in exclude and v != unit and p > 0.0}
    keep = kind_masks(mind)[0][np_view(mind, "u_channel")[d]] & (w_pos > 0.0) & (d != unit)
    if exclude:
        keep &= ~np.isin(d, np.fromiter(exclude, dtype=np.int64))
    return dict(zip(d[keep].tolist(), expectations(w_pos[keep], w_neg[keep], mind.cfg.k).tolist()))


def member_mass(mind, frame: int, exclude: int, trust: Optional[torch.Tensor] = None):
    """F11: what the members of a frame's category have lived in its slot, read now from their own evidence.

    For a frame whose category C fills one slot (one or, with bind_repeats, several positions), each member x of
    C (ABSTRACT x -> C with expectation e_x) whose instance (the frame with x in the slot) exists contributes its
    own belief: the share of its own positive evidence on sense/act targets that each value v received, s_x(v).
        W+(v) = sum_x e_x s_x(v),   W-(v) = sum_x e_x (1 - s_x(v))
    Every member counts once, with both signs; the instance being read is left out.
    F12 (bind_repeats): a value that IS the member itself counts under SELF, a value that is one of the member's
    own continuations (not a fixed word of the frame) counts under NEXT, so members that agree on a relation
    ("the answer repeats the asked word", "then comes what that word goes on with") agree although their words
    differ. Returns (masses by target or SELF/NEXT, members, slot positions, fixed words).
    """
    out = member_votes(mind, frame, exclude, trust)
    return out[:4]


def _value_keys(mind, x: int, targets, fixed, relational: bool, starts: bool, exact: bool,
                category: Optional[int] = None, inline: bool = True) -> List[List[int]]:
    """The keys under which each value a member x lived counts, in the order of ``targets``: the value itself, or the
    relation it stands in to x (F12 SELF and NEXT, F26 START: the first that holds, before the value). F28 (``exact``):
    the value itself always, and with it the most specific relation that names exactly that value as it would be
    resolved for the read instance (SELF, then START, then NEXT). F31 (``paradigmatic``, with ``category``, the frame's
    place): last, ALIKE, where the value is the item that stands most where x stands (``inline``; the members' reading
    adds it afterwards, where it can hold, see ``member_votes``)."""
    alike = inline and relational and category is not None and getattr(mind.cfg, "paradigmatic", False)
    replies = relational and getattr(mind.cfg, "replies", False)
    if exact:
        named: Dict[int, int] = {}
        if relational:
            if x in targets:
                named[x] = SELF
            if starts:
                best = resolve(mind, START, x, fixed)
                if best is not None and best not in named:
                    named[best] = START
            best = resolve(mind, NEXT, x, fixed)
            if best is not None and best not in named:
                named[best] = NEXT
            if alike and _may_stand_in(mind, x, targets, fixed, named):
                best = resolve(mind, ALIKE, x, fixed, category)
                if best is not None and best not in named:
                    named[best] = ALIKE
            if replies:
                best = resolve(mind, REPLY, x, fixed)
                if best is not None and best not in named:
                    named[best] = REPLY
        return [[v, named[v]] if v in named else [v] for v in targets]
    goes_on = _continuing(mind, x, fixed) if relational else ()
    stands = resolve(mind, ALIKE, x, fixed, category) if alike and _may_stand_in(mind, x, targets, fixed, {}) else None
    answered = resolve(mind, REPLY, x, fixed) if replies else None
    keys = []
    for v in targets:
        if relational and v == x:
            keys.append([SELF])
        elif starts and v not in fixed and begins_with(mind, x, v):
            keys.append([START])                   # F26: v is how a sequence that began with x went on
        elif v in goes_on:
            keys.append([NEXT])
        elif v == stands:
            keys.append([ALIKE])                   # F31: v stands where x stands
        elif v == answered:
            keys.append([REPLY])                   # F34: v is what x on its own was answered with
        else:
            keys.append([v])
    return keys


def reply_of(mind, x: int, fixed=()) -> Optional[int]:
    """F34 (Reply Relation): what x, said on its own, was answered with: the strongest world continuation of the
    composed unit (x, E), E the perceived event among the frame's fixed items (the end of a turn), without the frame's
    fixed words. NEXT is what x goes on with; this is what x as a whole utterance was followed by."""
    memo = _moment_memo(mind)
    key = ("reply", x, tuple(fixed))
    if key in memo:
        return memo[key]
    found = None
    sense, channels = kind_masks(mind)[4], mv_view(mind, "u_channel")
    event = next((p for p in fixed if sense[channels[p]]), None)
    if event is not None:
        unit = mind.S.tuple_index.get((x, event))
        if unit is not None:
            goes_on = continuations(mind, unit, fixed)
            if goes_on:
                found = max(goes_on, key=goes_on.get)
                if goes_on[found] <= 0.5:
                    found = None
    memo[key] = found
    return found


def _may_stand_in(mind, x: int, targets, fixed, named) -> bool:
    """F31, a bound on effort: the item that stands most where x stands is sought only when one of the values x lived
    (not x, not a fixed word, not already named by another relation) has filled a place with x at all."""
    since = getattr(mind, "episode_start", mind.t)
    places = memberships_before(mind, x, since)
    if not places:
        return False
    return any(v != x and v not in fixed and v not in named and places.keys() & memberships_before(mind, v, since).keys()
               for v in targets)


def stand_ins(mind, y: int, since: int) -> Dict[int, Tuple[float, float]]:
    """F31: (W+, W-) of "z and y are interchangeable" for every z that has filled a lived place with y before ``since``,
    in one pass over y's places (each pair as ``interchangeability`` counts it; the sums in another order). Reads only."""
    keys, w_pos, w_neg, _ = _stand_arrays(mind, y, since)
    return dict(zip(keys.tolist(), zip(w_pos.tolist(), w_neg.tolist())))


_NO_STAND = (np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0), np.zeros(0))


def _moment_memo(mind) -> dict:
    """The reading memo of the moment, or, read between moments (the inner voice reads without living), a memo kept
    while the moment does not change (nothing is written without a new moment; as ``situation_backoff`` keeps its
    reading)."""
    memo = mind.__dict__.get("_read_memo")
    if memo is not None:
        return memo
    held = mind.__dict__.get("_between_memo")
    if held is None or held[0] != mind.t:
        held = mind.__dict__["_between_memo"] = (mind.t, {})
    return held[1]


def _stand_arrays(mind, y: int, since: int):
    """``stand_ins`` as arrays (candidates, W+, W-, expectation), all of y's places in one pass; memoised for the moment."""
    memo = _moment_memo(mind)
    key = ("stand", y, since)
    if memo is not None and key in memo:
        return memo[key]
    S, k = mind.S, mind.cfg.k
    places, sizes, factors = [], [], []
    for g, e_y in memberships_before(mind, y, since).items():
        ties = S.inc.get((g, ABSTRACT), ())
        n = len(ties)
        if n < 2 or e_y == 0.5:
            continue
        places.append(ties)
        sizes.append(n)
        factors.append((2.0 * e_y - 1.0) / (n - 1))
    found = _NO_STAND
    if places:                                         # all places in one pass: their ties side by side
        idx = np.fromiter((t for ties in places for t in ties), dtype=np.int64, count=sum(sizes))
        factor = np.repeat(np.array(factors), sizes)
        z = np_view(mind, "t_src")[idx]
        keep = (z != y) & (np_view(mind, "t_birth")[idx] < since)
        idx, z, factor = idx[keep], z[keep], factor[keep]
        if len(idx):
            w = masses(mind, idx)
            e_z = expectations(w[0], w[1], k)
            m = factor * (2.0 * e_z - 1.0)             # (2 e_y - 1)(2 e_z - 1) / (n - 1)
            y_held = factor > 0.0                      # e_y > 1/2 (factor has its sign)
            agree = y_held & (e_z > 0.5)
            differ = np.where(y_held, e_z < 0.5, e_z > 0.5)
            some = agree | differ
            if some.any():
                keys, where = np.unique(z[some], return_inverse=True)
                w_pos = np.bincount(where, weights=np.where(agree, m, 0.0)[some], minlength=len(keys))
                w_neg = np.bincount(where, weights=np.where(differ, np.abs(m), 0.0)[some], minlength=len(keys))
                found = (keys, w_pos, w_neg, expectations(w_pos, w_neg, k))
    if memo is not None:
        memo[key] = found
    return found


def stand_in(mind, y: int, fixed=(), category: Optional[int] = None) -> Optional[int]:
    """F31 (Paradigmatic Relation): the item that stands most where y stands, read from the lived places both filled
    before the present run of items (``stand_ins``, the expectation of the evidence), if it is held (above 1/2) and
    no other is held as much. Neither y, nor a fixed word of the frame, nor a rival filler of the frame's own place
    (a member of ``category``: the place's fillers are what the frame tells apart, not what it relates them to)."""
    memo = _moment_memo(mind)
    since = getattr(mind, "episode_start", mind.t)
    key = ("stand_in", y, tuple(fixed), category, since)
    if memo is not None and key in memo:
        return memo[key]
    keys, _, _, held = _stand_arrays(mind, y, since)
    found = None
    if len(keys):
        out = np.zeros(len(keys), dtype=bool)
        if category is not None:                       # the rival fillers of the frame's own place
            ties = mind.S.inc.get((category, ABSTRACT), ())
            out |= np.isin(keys, np_view(mind, "t_src")[np.fromiter(ties, dtype=np.int64, count=len(ties))])
        if fixed:
            out |= np.isin(keys, np.fromiter(fixed, dtype=np.int64, count=len(fixed)))
        e = np.where(out, 0.5, held)                   # what is left out is held as ignorance
        i = int(np.argmax(e))
        best_e = float(e[i])
        second = float(np.max(np.delete(e, i))) if len(e) > 1 else 0.5
        if best_e > 0.5 and best_e > second:           # held, and no other held as much
            found = int(keys[i])
    if memo is not None:
        memo[key] = found
    return found


def member_analogy(mind, frame: int, exclude: int, trust: Optional[torch.Tensor] = None) -> Dict[int, Tuple[float, float]]:
    """F29 (Analogy by Interchangeability): what the members of a frame's category lived, carried over to the read
    instance through the interchangeability of its filler y with each member x (read from the places both filled
    before the present run of items, ``interchangeability``), by the analogy of NARS: from "x's instance went on with
    v", (f1 = v's share of what it went on with, c1 = conviction of how much it lived), and "y and x are
    interchangeable", (f2, c2):
        f = f1 f2,   c = c1 c2 f2,   evidence w = k c / (1 - c), split by f
    and the members' evidence adds (revision; each member's instance is one lived source). Keyed as member_votes keys
    (values and relations). It says how firmly the frame's claims hold for this instance, never what they are: the
    members' reading keeps the frequency of every claim (what all members agree on), and the conviction of the claim
    for the instance is the stronger of the members' breadth (F11) and this depth (the conviction of the evidence
    carried over). So a value one interchangeable member lived and the others did not stays a value the members
    disagree on. A member never seen to fill a place with y carries nothing."""
    S, k = mind.S, mind.cfg.k
    kinds = mv_view(mind, "u_kind")
    if exclude is None or not kinds[exclude] & COMPOSED:
        return {}
    parts = parts_of(mind, frame)
    cats = {p for p in parts if kinds[p] & CATEGORY}
    if len(cats) != 1:
        return {}
    category = cats.pop()
    where = tuple(j for j, p in enumerate(parts) if p == category)
    held = parts_of(mind, exclude)
    if not where or len(held) <= where[0]:
        return {}
    y = held[where[0]]
    fixed = tuple(p for p in parts if p != category)
    relational = getattr(S, "bind_repeats", False)
    starts = relational and getattr(mind.cfg, "starts", False)
    exact = getattr(mind.cfg, "exact_relations", False)
    since = getattr(mind, "episode_start", mind.t)
    src = mv_view(mind, "t_src")
    pooled: Dict[int, List[float]] = {}
    for tie in S.inc.get((category, ABSTRACT), ()):
        x = src[tie]
        if x == y:
            continue
        instance = S.tuple_index.get(tuple(x if j in where else p for j, p in enumerate(parts)))
        if instance is None or instance == exclude:
            continue
        w_pos, w_neg = interchangeability(mind, y, x, since)
        if w_pos <= 0.0:
            continue
        targets, own = _world_pos(mind, instance, trust)
        total = sum(own)
        if total <= 0.0:
            continue
        f2 = frequency(w_pos, w_neg)
        c = conviction(total, k) * conviction(w_pos + w_neg, k) * f2
        if c <= 0.0:
            continue
        w = k * c / (1.0 - c)
        for v, p, keys in zip(targets, own, _value_keys(mind, x, targets, fixed, relational, starts, exact, category)):
            f = (p / total) * f2
            for key in keys:
                acc = pooled.setdefault(key, [0.0, 0.0])
                acc[0] += f * w
                acc[1] += (1.0 - f) * w
    return {key: (p, n) for key, (p, n) in pooled.items()}


def _member_instances(mind, frame: int, exclude: int, tail: Optional[int] = None) -> set:
    """F30: the lived contexts the members' reading of ``frame`` counts (its members' instances, the read one left out;
    with ``tail``, F32, each instance ends with the item the relation names for its member)."""
    S = mind.S
    parts = parts_of(mind, frame)
    kinds = mv_view(mind, "u_kind")
    cats = {p for p in parts if kinds[p] & CATEGORY}
    if len(cats) != 1:
        return set()
    category = cats.pop()
    where = tuple(j for j, p in enumerate(parts) if p == category)
    fixed = tuple(p for p in parts if p != category)
    src = mv_view(mind, "t_src")
    found = set()
    for tie in S.inc.get((category, ABSTRACT), ()):
        instance = _instance(mind, parts, where, src[tie], tail, fixed, category)
        if instance is not None and instance != exclude:
            found.add(instance)
    return found


def _instance(mind, parts, where, x: int, tail, fixed, category: int) -> Optional[int]:
    """The lived instance of a frame with x in its place; with ``tail`` (F32, F34), followed by what each tail entry
    names for x: a relation (ALIKE, REPLY) resolved for x, or an item every member has there (None when a relation
    names nothing or that instance was never lived)."""
    held = tuple(x if j in where else p for j, p in enumerate(parts))
    if tail is not None:
        for entry in (tail if isinstance(tail, tuple) else (tail,)):
            if entry < 0:
                related = resolve(mind, entry, x, fixed, category)
                if related is None:
                    return None
                held += (related,)
            else:
                held += (entry,)
    return mind.S.tuple_index.get(held)


def bound_frames(mind, unit: int) -> List[Tuple[int, float, tuple]]:
    """F32 (Relational Binding), F34 (several places): the frames of a prefix of the unit whose filler the items after
    the prefix are bound to: each of them the filler's stand-in (F31, ALIKE), what the filler on its own was answered
    with (F34, REPLY), or an item every member has there; at least one a relation. Returns (frame, e, tail) for the
    longest prefix that has such frames, tail the entries in order (relations as their keys, items as themselves).
    A place binds its repeats (bind_repeats: the filler itself, SELF); it binds what stands for its filler, and what
    its filler was answered with, the same way: "jump twice ⏎ i_jump" is read through each member's "walk twice ⏎
    i_walk". A continuation of the filler (NEXT, START) is not bound: it is what follows the filler, read by the
    context's own evidence (bound too, it lowered the teacher protocol, NOTES)."""
    kinds = mv_view(mind, "u_kind")
    paradigmatic = getattr(mind.cfg, "paradigmatic", False)
    replies = getattr(mind.cfg, "replies", False)
    if not kinds[unit] & COMPOSED or kinds[unit] & SLOTTED or not getattr(mind.S, "bind_repeats", False) \
            or not (paradigmatic or replies):
        return []
    relations = ([ALIKE] if paradigmatic else []) + ([REPLY] if replies else [])
    parts = parts_of(mind, unit)
    shortest = 2 if replies else len(parts) - 1          # F31 alone binds the last item only
    for k in range(len(parts) - 1, max(shortest, 2) - 1, -1):
        prefix = mind.S.tuple_index.get(parts[:k])
        if prefix is None:
            continue
        found = []
        for frame, e in generalisations(mind, prefix):
            if not kinds[frame] & COMPOSED:
                continue
            category = frame_category(mind, frame)
            if category is None:
                continue
            frame_parts = parts_of(mind, frame)
            where = [j for j, p in enumerate(frame_parts) if p == category]
            fixed = tuple(p for p in frame_parts if p != category)
            filler = parts[where[0]]
            tail, bound = [], False
            for item in parts[k:]:
                named = None
                if item != filler and item not in fixed:
                    named = next((r for r in relations if resolve(mind, r, filler, fixed, category) == item), None)
                if named is not None:
                    tail.append(named)
                    bound = True
                elif item == filler or not replies:
                    tail = None                          # the filler again is bound by bind_repeats already
                    break
                else:
                    tail.append(item)                    # an item every member has there
            if tail is not None and bound:
                found.append((frame, e, tuple(tail)))
        if found:
            return found
    return []


def member_votes(mind, frame: int, exclude: int, trust: Optional[torch.Tensor] = None, tail: Optional[int] = None):
    """``member_mass`` and, fifth, how many members voted (the numerosity of the voice, F24). ``tail`` (F32): each
    member's instance is followed by the item the relation names for that member (``bound_frames``)."""
    S, k = mind.S, mind.cfg.k
    parts = parts_of(mind, frame)
    kinds = mv_view(mind, "u_kind")
    cats = {p for p in parts if kinds[p] & CATEGORY}
    if len(cats) != 1:
        return {}, 0.0, (), (), 0
    category = cats.pop()
    where = tuple(j for j, p in enumerate(parts) if p == category)
    fixed = tuple(p for p in parts if p != category)
    relational = getattr(S, "bind_repeats", False)
    starts = relational and getattr(mind.cfg, "starts", False)
    exact = getattr(mind.cfg, "exact_relations", False)
    shares: Dict[int, float] = {}
    members = voted = 0.0
    voters = 0
    looks = None                                       # F18: the read instance's filler and its facets
    if getattr(mind.cfg, "resemblance", False) and exclude is not None and kinds[exclude] & COMPOSED:
        from .learn import facet_set, facet_similarity
        held = parts_of(mind, exclude)
        filler = held[where[0]] if where and len(held) > where[0] else None
        looks = (filler, facet_set(mind, filler)) if filler is not None else None
        if looks is not None and not looks[1]:
            looks = None
    src = mv_view(mind, "t_src")
    alike = relational and getattr(mind.cfg, "paradigmatic", False)
    if alike:                                          # F31: values that may stand where their member stands
        since = getattr(mind, "episode_start", mind.t)
        rivals = {src[t] for t in S.inc.get((category, ABSTRACT), ())}
        may, may_share = [], 0.0
    for tie in S.inc.get((category, ABSTRACT), ()):
        x = src[tie]
        if tail is None:
            instance = S.tuple_index.get(tuple(x if j in where else p for j, p in enumerate(parts)))
        else:
            instance = _instance(mind, parts, where, x, tail, fixed, category)
        if instance is None or instance == exclude:
            continue
        targets, own = _world_pos(mind, instance, trust)   # its sense/act targets with W+ > 0, and their W+
        total = sum(own)
        if total <= 0.0:
            continue
        e_x = expectation(*mass1(mind, tie), k)
        weight = e_x
        if looks is not None:                          # members that look like the item speak louder (GCM):
            sim = facet_similarity(mind, looks[0], x, looks[1])   # vote x exp(-sensitivity x (1 - similarity))
            if sim <= 0.0:
                continue
            weight = e_x * math.exp(-mind.knob["F"] * (1.0 - sim))
        members += e_x
        voted += weight
        voters += 1
        places = memberships_before(mind, x, since) if alike else None
        loose = []
        for v, p, keys in zip(targets, own, _value_keys(mind, x, targets, fixed, relational, starts, exact, category,
                                                        inline=False)):
            share = weight * p / total
            for key in keys:
                shares[key] = shares.get(key, 0.0) + share
            if places and keys == [v] and v != x and v not in fixed and v not in rivals \
                    and places.keys() & memberships_before(mind, v, since).keys():
                loose.append((v, share))               # named by no other relation, and it has filled a place with x
        if loose:
            may.append((x, loose))
            may_share += sum(share for _, share in loose)
    if voted <= 0.0:
        return {}, 0.0, where, fixed, 0
    if alike and may and may_share * members / voted > 0.5 * members:
        for x, loose in may:                           # F31: the relation is sought where it can hold for the members
            named = stand_in(mind, x, fixed, category)  # (above 1/2: at most this much of their evidence can name it)
            for v, share in loose:
                if v == named:
                    shares[ALIKE] = shares.get(ALIKE, 0.0) + share
                    break
    scale = members / voted                            # frequencies from the weighted votes, conviction from how
    return ({v: (w * scale, members - w * scale) for v, w in shares.items()}, members, where, fixed,  # many members
            voters)


VOICE_CHANNEL = "sesler"   # F24: the mind's own channel for the kinds of voices (not "ses": the hearing facets use it)
VOICE_ALL = "*"         # F24: the unit of every voice (what voices in general have been worth)


def voice_kind(voters: int, top: float) -> Tuple[int, int]:
    """F24: the kind of a voice as the mind perceives it, in two magnitudes: how many members speak (numerosity in
    doubling bands: 1, 2, 3-4, 5-8, 9-16, ...) and how much they agree (the frequency of their top value in quarters,
    so that one half, where a prediction starts to hold, is a band edge; all agreeing is a band of its own). Nothing
    in it names a word, a class or a domain."""
    many = (max(1, voters) - 1).bit_length() + 1
    agree = 4 if top >= 1.0 else max(0, min(3, int(4.0 * top)))
    return many, agree


def _holds(mind, symbol) -> Tuple[float, float]:
    """Trust-weighted (W+, W-) of the tie (voice unit -> holds); zeros when never lived."""
    unit = mind.unit_of(VOICE_CHANNEL, symbol)
    tie = mind.S.find_tie(unit, mind.holds, EXPECT, 0) if unit >= 0 else -1
    if tie < 0:
        return 0.0, 0.0
    return mass1(mind, tie)


def voice_credibility(mind, kind: Tuple[int, int]) -> float:
    """F24: how far a voice of this kind can be believed, read general -> specific as everywhere (``expect_map``,
    nested): e <- e(every voice -> holds), then e <- c f + (1 - c) e with the kind's own (f, c). A kind the mind has
    never relied on is worth what voices in general have been worth; with no such experience at all, 1/2. Reads
    only; units and ties are made when the first outcome is lived (learn.judge_voices)."""
    if not mind.ch.has(VOICE_CHANNEL):
        return 0.5
    k = mind.cfg.k
    w_pos, w_neg = _holds(mind, VOICE_ALL)
    e = expectation(w_pos, w_neg, k)
    w_pos, w_neg = _holds(mind, kind)
    c = conviction(w_pos + w_neg, k)
    return c * frequency(w_pos, w_neg) + (1.0 - c) * e


def _claim(mind, unit: int, mass, where, fixed, category: Optional[int] = None) -> Tuple[Optional[int], float]:
    """F24: what a voice claims for ``unit``: its top value (SELF, NEXT, START and ALIKE resolved as in
    ``members_map``) and its frequency f."""
    filler = parts_of(mind, unit)[where[0]] if where and mv_view(mind, "u_kind")[unit] & COMPOSED else None
    claim, claim_f = None, -1.0
    for target, (w_pos, w_neg) in mass.items():
        target = resolve(mind, target, filler, fixed, category)
        f = frequency(w_pos, w_neg)
        if target is not None and f > claim_f:
            claim, claim_f = target, f
    return claim, claim_f


def frame_category(mind, frame: int) -> Optional[int]:
    """The one category of a frame (its place), or None."""
    kinds = mv_view(mind, "u_kind")
    cats = {p for p in parts_of(mind, frame) if kinds[p] & CATEGORY}
    return cats.pop() if len(cats) == 1 else None


def members_map(mind, unit: int, trust: Optional[torch.Tensor] = None) -> Dict[int, float]:
    """support = "members" (F11): a category speaks for an instance with what its members have lived.

    Among the frames of the instance, the one whose members agree most on a value speaks
    (score = e(i -> g) conviction(members) max_v f(v)); its expectations are e <- c f + (1 - c) 1/2 with
    c = lam e(i -> g) conviction(members); SELF resolves to the word in the instance's slot and NEXT to that word's
    own strongest continuation (F12), START to the strongest continuation of sequences that began with that word (F26,
    ``starts``: the members agree that the value is how a sequence beginning with them went on); the instance's own
    evidence comes last, as everywhere.
    F24 (voices): each score is weighted by the credibility the mind has learned for voices of that kind (how many
    members speak, how much they agree; with no such experience every kind reads 1/2 and the choice is exactly
    F11's). For the context read with the full trust, the voice that spoke and every voice that could have spoken
    are noted with their claims (top value, frequency), so that what comes next judges them (learn.judge_voices).
    """
    cfg, S = mind.cfg, mind.S
    voices = getattr(cfg, "voices", False)
    best = None
    heard = []                                         # F24: every voice that could speak: (kind, claim, f)
    frames = [(general, e_abs, None) for general, e_abs in generalisations(mind, unit)]
    if getattr(cfg, "bind_relations", False):
        frames += bound_frames(mind, unit)               # F32: frames of the prefix bound to their filler's stand-in
    for general, e_abs, tail in frames:
        mass, members, where, fixed, voters = member_votes(mind, general, unit, trust, tail)
        if not mass:
            continue
        c_members = conviction(members, cfg.k)
        top = max(frequency(p, n) for p, n in mass.values())
        score = e_abs * c_members * top
        kind = None
        if voices:
            kind = voice_kind(voters, top)
            score *= voice_credibility(mind, kind)
            heard.append((kind,) + _claim(mind, unit, mass, where, fixed, frame_category(mind, general)))
        if best is None or score > best[0]:
            best = (score, e_abs, c_members, mass, where, fixed, kind, len(heard) - 1, general, tail)
    out: Dict[int, float] = {}
    if best is not None:
        _, e_abs, c_members, mass, where, fixed, kind, spoke, frame, tail = best
        c = cfg.lam * e_abs * c_members
        filler = parts_of(mind, unit)[where[0]] if where and mv_view(mind, "u_kind")[unit] & COMPOSED else None
        carried = member_analogy(mind, frame, unit, trust) if getattr(cfg, "analogy", False) and tail is None else {}
        category = frame_category(mind, frame)
        corroborate = getattr(cfg, "corroborate", False) and getattr(cfg, "situations", False)
        claims: Dict[int, Tuple[float, float, float]] = {}
        for target, (w_pos, w_neg) in mass.items():
            held = c                               # F29: what the members agree on is theirs (the frequency, all
            if target in carried:                  # members); how firmly it holds for this instance is the stronger
                held = max(c, conviction(sum(carried[target]), cfg.k))   # of breadth and interchangeable depth
            f = frequency(w_pos, w_neg)
            e = held * f + (1.0 - held) * 0.5
            target = resolve(mind, target, filler, fixed, category)
            if target is None:
                continue
            out[target] = max(out.get(target, 0.0), e)
            if corroborate and (target not in claims or e > claims[target][0]):
                claims[target] = (e, held, f)
        if claims:
            corroborated(mind, unit, trust, frame, claims, out, tail)
        if voices and trust is None:
            mind.__dict__.setdefault("voice", {})[unit] = (heard[spoke], heard)
    own = own_mass(mind, unit, 1, trust)
    for target, (w_pos, w_neg) in own.items():
        c = conviction(w_pos + w_neg, cfg.k)
        out[target] = c * frequency(w_pos, w_neg) + (1.0 - c) * out.get(target, 0.5)
    lived = own_mass(mind, unit, 1, trust, whole=True) \
        if getattr(cfg, "company", False) and "personal" in mind.__dict__ else own
    if getattr(cfg, "exclusive", False) and lived:     # F27 reads the whole life (F33: that the instance went on
        excluded(mind, out, lived)                     # otherwise is known whoever it was lived with; what it went
                                                       # on with is recalled in present company)
    if getattr(cfg, "starts", False) and _begins(mind, unit):
        began = start_mass(mind, unit, trust)            # F26: the present moment is a sequence that began with this
        total = sum(p for p, _ in began.values())        # item, more specific than the item anywhere: its lived
        if total > 0.0:                                  # evidence comes last; what never went on a sequence that
            for target in set(out) | set(began):         # began so counts against (its rivals' share)
                if not _world(mind, target):
                    continue
                w_pos, w_neg = began.get(target, (0.0, total))
                c = conviction(w_pos + w_neg, cfg.k)
                out[target] = c * frequency(w_pos, w_neg) + (1.0 - c) * out.get(target, 0.5)
    if cfg.backoff and not any(e > 0.5 for t, e in out.items() if _world(mind, t)):
        for target, e in continuation_backoff(mind, unit, trust).items():
            out[target] = max(out.get(target, 0.0), e)
    if getattr(cfg, "substitution", False) and not any(e > 0.5 for t, e in out.items() if _world(mind, t)):
        for target, e in substitute_backoff(mind, unit, trust).items():
            out[target] = max(out.get(target, 0.0), e)
    line = 0.5 + mind.knob["theta"] / (2.0 * cfg.T)     # F36: when nothing would be said (the speaking line), a
    if getattr(cfg, "forms", False) and not any(e > line for t, e in out.items() if _world(mind, t)):
        for target, e in form_reading(mind, unit, trust).items():   # word never met is read as the form it is of
            out[target] = max(out.get(target, 0.0), e)
    if getattr(cfg, "situations", False) and not any(e > 0.5 for t, e in out.items() if _world(mind, t)):
        for target, e in situation_backoff(mind, unit, trust).items():
            out[target] = max(out.get(target, 0.0), e)
    return out


def form_base(mind, w: int, since: int):
    """F36 (Word Forms): the lived word a word never met is a form of, and how far: (base, f, c) or None.

    The words are read as letters. Among the words lived before ``since`` on the same channel, the base shares the
    longest beginning (stem) with w, and the stem is a boundary of the lived words themselves (Harris's successor
    variety): it is a lived word, or the lived words that begin with it go on with at least two different letters.
    A beginning only one lived word has, cut where w happens to differ ("mas" of "masa" for "maske"), is not a stem.
    Ties of length go to the most lived.
    f = the share of w the stem covers (its letters over the longer word's); c = conviction(the lived words that begin
    with the stem): a stem many forms begin with is a stem."""
    memo = _moment_memo(mind)
    key = ("form", w, since)
    if key in memo:
        return memo[key]
    found = None
    channels, symbols = mv_view(mind, "u_channel"), mv_view(mind, "u_symbol")
    channel = channels[w]
    word = str(mind.ch.symbol(channel, symbols[w]))
    if word.isalpha() and len(word) >= 3:
        S = mind.S
        cid = channel
        lexicon = mind.__dict__.get("_lexicon")
        size = len(mind.ch.symbols[cid])
        if lexicon is None or lexicon[0] != (cid, size, since):
            words = []
            for index, sym in enumerate(mind.ch.symbols[cid]):
                u = S.atom_index.get((cid, index), -1)
                text = str(sym)
                if u >= 0 and text.isalpha() and int(S.u_birth[u]) < since and float(S.u_use[u]) > 0.0:
                    words.append((text, u, float(S.u_use[u])))
            lexicon = mind.__dict__["_lexicon"] = ((cid, size, since), words)
        words = lexicon[1]
        best = None
        for text, u, use in words:
            if u == w:
                continue
            n = 0
            for a, b in zip(word, text):
                if a != b:
                    break
                n += 1
            if n < 2 or n == len(word):
                continue
            if best is None or n > best[0] or (n == best[0] and use > best[2]):
                best = (n, u, use, text)
        if best is not None:
            n, base, _, text = best
            stem = word[:n]
            forms = [t for t, _, _ in words if t.startswith(stem)]
            follow = {t[n] if len(t) > n else "" for t in forms}
            if "" in follow or len(follow) >= 2:
                f = n / max(len(word), len(text))
                found = (base, f, conviction(float(len(forms)), mind.cfg.k))
    memo[key] = found
    return found


def form_reading(mind, unit: int, trust: Optional[torch.Tensor] = None) -> Dict[int, float]:
    """F36: a context holding words never met (before the present run) is read as the lived context with each of
    them replaced by the lived word it is a form of (``form_base``), when that context was lived: its members' reading
    carried over with conviction c f (the analogy of NARS), e <- c f (e' - 1/2) + 1/2."""
    kinds = mv_view(mind, "u_kind")
    if not kinds[unit] & COMPOSED or kinds[unit] & SLOTTED:
        return {}
    since = getattr(mind, "episode_start", mind.t)
    born = mv_view(mind, "u_birth")
    parts = parts_of(mind, unit)
    act = kind_masks(mind)[5]
    channels = mv_view(mind, "u_channel")
    held, weight = [], 1.0
    for p in parts:
        if kinds[p] & ATOMIC and act[channels[p]] and born[p] >= since:
            found = form_base(mind, p, since)
            if found is None:
                return {}
            base, f, c = found
            held.append(base)
            weight *= c * f
        else:
            held.append(p)
    if weight >= 1.0:
        return {}
    lived = mind.S.tuple_index.get(tuple(held))
    if lived is None or lived == unit:
        return {}
    memo = _moment_memo(mind)
    key = ("form_read", lived, id(trust) if trust is not None else None)
    if key in memo:
        return memo[key]
    memo[key] = {}                                       # no recursion through itself
    read = members_map(mind, lived, trust)
    out = {t: weight * (e - 0.5) + 0.5 for t, e in read.items() if _world(mind, t)}
    memo[key] = out
    return out


def memberships_before(mind, unit: int, since: int) -> Dict[int, float]:
    """F29: the categories ``unit`` was lived as a member of before the moment ``since``, with the expectation of each
    membership: a membership born in the present run of items (the utterance being read) is left out, so that what is
    being read cannot vouch for itself."""
    memo = _moment_memo(mind)
    key = ("memberships", unit, since)
    if memo is not None and key in memo:
        return memo[key]
    S = mind.S
    ties = S.out.get((unit, ABSTRACT, 0))
    if not ties:
        return {}
    k = mind.cfg.k
    if len(ties) > SMALL:                              # many memberships: the same numbers, element by element
        idx = np.fromiter(ties, dtype=np.int64, count=len(ties))
        idx = idx[np_view(mind, "t_birth")[idx] < since]
        w = masses(mind, idx)
        found = dict(zip(np_view(mind, "t_dst")[idx].tolist(), expectations(w[0], w[1], k).tolist()))
    else:
        dst, born = mv_view(mind, "t_dst"), mv_view(mind, "t_birth")
        w_pos, w_neg = masses_py(mind, ties)
        found = {dst[t]: expectation(p, n, k) for t, p, n in zip(ties, w_pos, w_neg) if born[t] < since}
    if memo is not None:
        memo[key] = found
    return found


def interchangeability(mind, y: int, x: int, since: int) -> Tuple[float, float]:
    """F29: (W+, W-) of "y and x are interchangeable", read from the lived places both have filled before ``since``.

    A category is a lived place: the fillers of one slot of contexts that differ only there. Each place both fill is
    one piece of evidence that they stand for each other, as strong as both memberships are above 1/2 (m = 2e - 1, the
    one rule's evidence above ignorance) and as telling as the place is specific: it is shared among its n members,
    so a pair receives 1/(n - 1) of it (a place of two is a whole piece, one holding everything that ends an utterance
    is almost none). A place where one is a member and the other is held not to be (e < 1/2) counts against:
        W+ = sum_g (m_y m_x) / (n_g - 1)  over places with both m > 0
        W- = sum_g |m_y m_x| / (n_g - 1)  over places with opposite signs
    Reads only; memoised for the moment."""
    memo = mind.__dict__.get("_read_memo")
    key = ("interchange", y, x, since)
    if memo is not None and key in memo:
        return memo[key]
    S = mind.S
    cy, cx = memberships_before(mind, y, since), memberships_before(mind, x, since)
    w_pos = w_neg = 0.0
    for g in cy.keys() & cx.keys():
        n = len(S.inc.get((g, ABSTRACT), ()))
        if n < 2:
            continue
        m = (2.0 * cy[g] - 1.0) * (2.0 * cx[g] - 1.0) / (n - 1)
        if cy[g] > 0.5 and cx[g] > 0.5:
            w_pos += m
        elif (cy[g] - 0.5) * (cx[g] - 0.5) < 0.0:
            w_neg -= m
    out = (w_pos, w_neg)
    if memo is not None:
        memo[key] = out
    return out


def substitute_backoff(mind, unit: int, trust: Optional[torch.Tensor] = None) -> Dict[int, float]:
    """Substitute Reading (with Substitution Evidence, zihin/anlam.py): a context never lived as a whole is read
    through the lived contexts that differ from it in one word only, a word that shares a category with the one it
    replaces. The strength of a replacement is s = (2 e(p -> g) - 1)(2 e(q -> g) - 1) for the shared category g that
    gives the most (both memberships read by the one rule), and the replacement context speaks with its own lived
    evidence held with conviction lam * s * conviction(its mass): e <- c f + (1 - c) 1/2. So "ismin ne ?" is met with
    what followed "adın ne ?" only as far as ismin and adın are known to stand in the same places; a weak equivalence
    cannot make the mind speak. Bounded effort: at most MEMBERS members of each category are tried."""
    S, cfg = mind.S, mind.cfg
    kinds = mv_view(mind, "u_kind")
    if not kinds[unit] & COMPOSED:
        return {}
    parts = parts_of(mind, unit)
    out: Dict[int, float] = {}
    for pos, p in enumerate(parts):
        if not kinds[p] & ATOMIC:
            continue
        strength: Dict[int, float] = {}
        for g, e_pg in categories_of(mind, p):
            if e_pg <= 0.5:
                continue
            for tie in list(S.inc.get((g, ABSTRACT), ()))[:MEMBERS]:
                q = int(S.t_src[tie])
                if q == p or not kinds[q] & ATOMIC:
                    continue
                e_qg = expectation(*mass1(mind, tie, trust), cfg.k)
                if e_qg > 0.5:
                    st = (2.0 * e_pg - 1.0) * (2.0 * e_qg - 1.0)
                    if st > strength.get(q, 0.0):
                        strength[q] = st
        for q, st in strength.items():
            other = S.tuple_index.get(parts[:pos] + (q,) + parts[pos + 1:])
            if other is None:
                continue
            for target, (w_pos, w_neg) in own_mass(mind, other, 1, trust).items():
                c = cfg.lam * st * conviction(w_pos + w_neg, cfg.k)
                e = c * frequency(w_pos, w_neg) + (1.0 - c) * 0.5
                if e > out.get(target, 0.0):
                    out[target] = e
    return out


MEMBERS = 200        # members of a category tried by the substitute reading (bounded effort)


def situation_backoff(mind, unit: int, trust: Optional[torch.Tensor] = None) -> Dict[int, float]:
    """F23: a situation never lived as a whole is met with what followed the lived situations most like it (see
    ``situation_evidence``): e = c f + (1 - c) 1/2 for every value the voters lived."""
    if not mv_view(mind, "u_kind")[unit] & COMPOSED:
        return {}
    key = (unit, mind.t, id(trust) if trust is not None else None)
    cache = getattr(mind, "_situation_cache", None)
    if cache is not None and cache[0] == key:
        return cache[1]
    c, shares = situation_evidence(mind, unit, trust, lived_only=getattr(mind.cfg, "lived_situations", False))
    out = {v: c * share + (1.0 - c) * 0.5 for v, share in shares.items()}
    mind._situation_cache = (key, out)
    return out


def situation_evidence(mind, unit: int, trust: Optional[torch.Tensor] = None, leave_out=frozenset(),
                       lived_only: bool = False, focus=None):
    """F23: a situation never lived as a whole is met with what followed the lived situations most like it.

    The situation is the present context: someone's utterance up to its perceived end (the turn end), and what the
    mind has said since, if anything. Lived contexts that end with the same part and hold the same perceived event at
    the same place from the end are compared with it part by part from the end. A part counts by how telling it is,
    1 / (1 + the number of those contexts that hold it), so a word nothing lived holds weighs most. The similarity s
    of a lived context is the weight of the parts both have at the same place over the weight of all parts either
    has; it must share a word of the utterance, not only punctuation and the end. Each such context votes its own
    evidence on what came next, weighted exp(-F (1 - s)) as exemplars are (GCM, as F18). The votes are held with
    conviction lam * s * conviction(voters), s the voters' weighted similarity: a weak resemblance (an unknown word
    in the utterance) cannot make the mind speak and leaves room for its curiosity.
    Returns (c, {value: its share f of the votes}); (0, {}) when no lived situation votes. ``leave_out``: lived
    contexts not to count (F30: those another reading has already counted). ``lived_only``: frames are left out too
    (a frame is a generalisation, not a lived situation: its own evidence is its members', literal values only).
    ``focus`` (F34): the places (distances from the end) of a frame's fixed items. The lived situations are then the
    lived suffixes of the present context (the context without its front: what followed the claim's own context when
    less of it was there), and the similarity is how much of the frame's fixed items a suffix holds, by the same
    weights (evidence about a frame's claim is what followed its own context, however much of it was lived)."""
    S, cfg = mind.S, mind.cfg
    if not mv_view(mind, "u_kind")[unit] & COMPOSED:
        return 0.0, {}
    heard = parts_of(mind, unit)
    n = len(heard)
    channels = mv_view(mind, "u_channel")
    sense, act = kind_masks(mind)[4:6]
    event = -1                                          # the last perceived event of the situation (the turn end)
    for j, p in enumerate(heard):
        if sense[channels[p]]:
            event = j
    if n >= 2 and event >= 1:
        d_event = n - 1 - event
        symbols = mv_view(mind, "u_symbol")

        def held(p):                                     # how many lived contexts hold the part
            return len(S.inc.get((p, PART), ()))

        def weight(p):
            return 1.0 / (1 + held(p))

        def word(p):
            c = channels[p]
            return act[c] and str(mind.ch.symbol(c, symbols[p])).isalpha()

        # the lived situations are called up through the rarest of the parts they must share at the same place from
        # the end (the last part, or a word of the utterance): the contexts holding it there, ending with the same part
        # and holding the event at its place. The cost grows with how often that part was lived, not with the size of
        # the whole memory.
        anchors = sorted(((n - 1 - j, heard[j]) for j in range(event) if word(heard[j])), key=lambda x: held(x[1]))
        if focus:                                        # F34: the lived suffixes of the present context
            anchors = []
        lived, seen, budget = [], {unit}, 20000
        if focus:
            for L in range(n - 1, 1, -1):
                suffix = S.tuple_index.get(heard[n - L:])
                if suffix is not None and suffix not in seen and L > d_event:
                    seen.add(suffix)
                    lived.append((heard[n - L:], suffix))
        src, lag = mv_view(mind, "t_src"), mv_view(mind, "t_lag")
        for d_anchor, anchor in anchors:                 # each word of the utterance calls up its situations, the
            for tie in S.inc.get((anchor, PART), ()):     # rarest word first, within a bounded effort
                budget -= 1
                c = src[tie]
                if c in seen:
                    continue
                t = parts_of(mind, c)
                L = len(t)
                if L > max(d_anchor, d_event) and lag[tie] == L - 1 - d_anchor and t[-1] == heard[-1] \
                        and t[L - 1 - d_event] == heard[event]:
                    seen.add(c)
                    lived.append((t, c))
                if budget <= 0:
                    break
            if budget <= 0:
                break

        heard_set = set(heard)
        heard_weight = sum(weight(p) for p in heard_set)
        shares: Dict[int, float] = {}
        voted = similar = 0.0
        voters = 0
        kinds = mv_view(mind, "u_kind")
        focus_weight = sum(weight(heard[n - 1 - d]) for d in focus) if focus else 0.0
        for t, u in lived:
            if u in leave_out or (lived_only and kinds[u] & SLOTTED):
                continue
            L = len(t)
            same = [d for d in range(min(n, L)) if t[L - 1 - d] == heard[n - 1 - d]]
            if focus:
                same = [d for d in same if d in focus]
            if not any(word(heard[n - 1 - d]) for d in same if d > d_event):   # a word of the utterance itself
                continue
            targets, own = _world_pos(mind, u, trust)
            total = sum(own)
            if total <= 0.0:
                continue
            if focus:
                s = sum(weight(heard[n - 1 - d]) for d in same) / focus_weight
            else:
                shared = {heard[n - 1 - d] for d in same}
                union = heard_weight + sum(weight(p) for p in set(t) - heard_set)
                s = sum(weight(p) for p in shared) / union
            g = math.exp(-mind.knob["F"] * (1.0 - s))
            voted += g
            similar += g * s
            voters += 1
            for v, p in zip(targets, own):
                shares[v] = shares.get(v, 0.0) + g * p / total
        if voted > 0.0:
            c = cfg.lam * (similar / voted) * conviction(float(voters), cfg.k)
            return c, {v: w / voted for v, w in shares.items()}
    return 0.0, {}


def continuation_backoff(mind, unit: int, trust: Optional[torch.Tensor] = None) -> Dict[int, float]:
    """F16: starting needs a reason, continuing follows the words. When nothing in the whole present situation
    expects a next symbol, the end of the sequence that is going on now (the items after the last perceived event,
    e.g. the mind's own utterance after its partner's ended) is read instead: its longest suffix that has lived
    continuations speaks, with its own evidence scaled by lam. It never reaches back across a perceived event, so
    it cannot start an utterance, only carry one on.
    F26 (``starts``): a composed suffix of fewer than F items is a sequence that began with those items (only an
    episode's beginning makes such a unit); an utterance of one item so far is read the same way, as how sequences
    that began with it went on (``start_mass``), before what followed that item anywhere."""
    S, cfg = mind.S, mind.cfg
    if not mv_view(mind, "u_kind")[unit] & COMPOSED:
        return {}
    parts = parts_of(mind, unit)
    channels, sense = mv_view(mind, "u_channel"), kind_masks(mind)[4]
    start = 0
    for j, p in enumerate(parts):
        if sense[channels[p]]:
            start = j + 1
    tail = parts[start:]
    heard = set(parts[:start])                          # what the partner just said is an echo, not knowledge
    first = 1 if len(tail) == len(parts) else 0         # the whole tuple is the unit itself, read already
    starts = getattr(cfg, "starts", False)
    for i in range(first, len(tail)):
        suffix = tail[i:]
        if not suffix:
            continue
        if len(suffix) == 1:
            readings = [own_mass(mind, suffix[0], 1, trust)]
            if starts and i == 0:                       # the whole utterance so far is this one item
                readings.insert(0, start_mass(mind, suffix[0], trust))
        else:
            piece = S.tuple_index.get(suffix)
            if piece is None:
                continue
            readings = [own_mass(mind, piece, 1, trust)]
        for mass in readings:
            out = {}
            for target, (w_pos, w_neg) in mass.items():
                if _world(mind, target) and w_pos > 0.0 and target not in heard:
                    c = cfg.lam * conviction(w_pos + w_neg, cfg.k)
                    out[target] = c * frequency(w_pos, w_neg) + (1.0 - c) * 0.5
            if any(e > 0.5 for e in out.values()):
                return out
    return {}


def trusted_map(mind, unit: int) -> Dict[int, float]:
    """support = "trusted" (F10): generalisations are sources, pooled by the trust they have earned.

    The same pooling as for provenance classes (W = sum_pi tau_pi w), with the trust of a generalisation
    tau_g = max(0, 2 reliability(g) - 1): one that has predicted no better than chance has no voice. Each speaks
    with at most lam e(i -> g) lived experiences (capped, as in section 4 changed):
        W+_gen(v) = sum_g tau_g lam e(i -> g) W+(g -> v) / max(1, W(g -> v) / k),   W-_gen likewise
        e <- c_gen f_gen + (1 - c_gen) 1/2;   own evidence last: e <- c_own f_own + (1 - c_own) e
    """
    cfg = mind.cfg
    pooled: Dict[int, List[float]] = {}
    for general, e_abs in generalisations(mind, unit):
        tau = max(0.0, 2.0 * reliability(mind, general) - 1.0)
        if tau <= 0.0:
            continue
        for target, (w_pos, w_neg) in own_mass(mind, general).items():
            factor = tau * cfg.lam * e_abs / max(1.0, (w_pos + w_neg) / cfg.k)
            acc = pooled.setdefault(target, [0.0, 0.0])
            acc[0] += factor * w_pos
            acc[1] += factor * w_neg
    out = {t: expectation(p, n, cfg.k) for t, (p, n) in pooled.items()}
    for target, (w_pos, w_neg) in own_mass(mind, unit).items():
        c = conviction(w_pos + w_neg, cfg.k)
        out[target] = c * frequency(w_pos, w_neg) + (1.0 - c) * out.get(target, 0.5)
    return out


def choice_map(mind, unit: int) -> Dict[int, float]:
    """support = "choice" (F7): for each target, the generalisation that has lived most about it speaks.

        best g = argmax over generalisations of  lam e(i -> g) conviction(g -> v)
        e <- c_g f_g + (1 - c_g) 1/2;   own evidence:  e <- c_own f_own + (1 - c_own) e

    Competing generalisations are not summed and not ordered by size: the most experienced one is chosen (the
    choice rule of NARS between competing beliefs). Own evidence of the instance always comes last.
    """
    cfg = mind.cfg
    best: Dict[int, Tuple[float, float]] = {}
    for general, e_abs in generalisations(mind, unit):
        for target, (w_pos, w_neg) in own_mass(mind, general).items():
            c = cfg.lam * e_abs * conviction(w_pos + w_neg, cfg.k)
            if c > best.get(target, (-1.0, 0.0))[0]:
                best[target] = (c, frequency(w_pos, w_neg))
    out = {t: c * f + (1.0 - c) * 0.5 for t, (c, f) in best.items()}
    for target, (w_pos, w_neg) in own_mass(mind, unit).items():
        c = conviction(w_pos + w_neg, cfg.k)
        out[target] = c * frequency(w_pos, w_neg) + (1.0 - c) * out.get(target, 0.5)
    return out


def predictions(mind, mode: Optional[str] = None) -> Dict[int, float]:
    """Expectations for the next moment (section 2.3).

    "votes" (as written): every unit of the focus votes a_u * (2 e - 1) for the targets of its lag-1 ties and
    the votes are summed, as in the settling formula. "specific" (changed): the most specific unit of the
    present context (the composed unit of the episode, or its single item) is read with its support.
    """
    cfg = mind.cfg
    mode = cfg.predict if mode is None else mode
    if mode == "own":                                  # the present context's own evidence only (no generalisation)
        found = _unit_arrays(mind, mind.context) if mind.context is not None else None
        if found is None:
            return {}
        dst, w_pos, w_neg = found
        if isinstance(dst, list):
            return {v: expectation(p, n, cfg.k) for v, p, n in zip(dst, w_pos, w_neg)}
        return dict(zip(dst.tolist(), expectations(w_pos, w_neg, cfg.k).tolist()))
    if mode == "specific":
        if mind.context is None:
            return {}
        return expect_map(mind, mind.context)
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
    out["eff_e"] = expect_map(mind, unit).get(target, 0.5)
    return out


def drive_of(mind, unit: int) -> float:
    """D(u) = sum over the lag-1 ties of u of use * (1 - conviction) (section 5).

    Only ties to units of sense and act channels count: the drive is uncertainty about the world, not about
    the mind's own feelings (otherwise feeling something would itself be a relief; see NOTES.md).
    """
    S = mind.S
    ties = S.out.get((unit, EXPECT, 1))
    if not ties:
        return 0.0
    k = mind.cfg.k
    if len(ties) <= SMALL:                             # a few ties: the terms in plain numbers
        world, channel, dst, use = kind_masks(mind)[3], mv_view(mind, "u_channel"), mv_view(mind, "t_dst"), \
            mv_view(mind, "t_use")
        terms = []
        for t in ties:
            if world[channel[dst[t]]]:
                total = mass1_total(mind, t)
                terms.append(use[t] * (1.0 - total / (total + k)))
        return float(torch.tensor(terms, dtype=F64).sum())   # the same values summed by torch in the same order
    ids, dst = S.out_ids((unit, EXPECT, 1))
    world = kind_masks(mind)[0][np_view(mind, "u_channel")[dst.numpy()]]
    total = masses(mind, ids, total=True)
    terms = np_view(mind, "t_use")[ids.numpy()] * (1.0 - total / (total + k))
    return float(torch.from_numpy(terms[world]).sum())  # the same values summed by torch in the same order


def total_drive(mind) -> float:
    """Drive over the recently used units: the units of the present focus."""
    return sum(drive_of(mind, unit) for unit, a, grounded in mind.focus if grounded)


def conflict_of(mind, unit: int) -> float:
    """Mass of the lag-1 ties of ``unit`` with frequency near 1/2 and high conviction (section 2.6)."""
    total = 0.0
    found = _unit_arrays(mind, unit)
    if found is None:
        return total
    _, p, n = found
    k = mind.cfg.k
    if isinstance(p, list):
        for a, b in zip(p, n):
            total += conviction(a + b, k) * (1.0 - abs(2.0 * frequency(a, b) - 1.0))
        return total
    w = p + n
    some = w > 0
    terms = w / (w + mind.cfg.k) * (1.0 - np.abs(2.0 * np.where(some, p / np.where(some, w, 1.0), 0.5) - 1.0))
    for term in terms.tolist():                        # added one by one in tie order, as before
        total += term
    return total
