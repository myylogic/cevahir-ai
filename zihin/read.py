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
    Targets the instance has its own evidence on keep that evidence (F14 records their failures); nothing is written.
    F44 (``exclusive_world``): the record of an instance holds everything that followed it, on every channel, so a
    target that is not in it followed the instance in none of its lived moments, whatever its channel: the largest of
    the instance's positive masses on a world channel (its lived moments, counted on the channel that held an item in
    most of them) counts against every world target it never went on with. An instance that always went on with the
    end of the turn is evidence against a word there."""
    k = mind.cfg.k
    channel = mv_view(mind, "u_channel")
    totals: Dict[int, float] = {}
    for target, (w_pos, _) in own.items():
        if w_pos > 0.0 and _world(mind, target):
            totals[channel[target]] = totals.get(channel[target], 0.0) + w_pos
    if not totals:
        return
    across = max(totals.values()) if getattr(mind.cfg, "exclusive_world", False) else 0.0
    for target in list(out):
        if target in own or not _world(mind, target):
            continue
        total = max(totals.get(channel[target], 0.0), across)
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
    the situations claim are left to F23's own reading (it speaks when nothing else does). Writes into ``out``.
    F38 (``witness``): the lived suffixes that leave out a fixed item of the claim add, value by value, what they
    witness of it (``witnessed``: NARS deduction), as a third reading in the same revision."""
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
    counted = frozenset(_member_instances(mind, frame, unit, tail))
    c_sit, shares = situation_evidence(mind, unit, trust, counted, lived_only=True, focus=focus)
    witnessed_by = witnessed(mind, unit, trust, counted, focus) if focus and getattr(mind.cfg, "witness", False) \
        else {}
    if c_sit <= 0.0 and not witnessed_by:
        return
    w2 = k * c_sit / (1.0 - c_sit) if c_sit > 0.0 else 0.0
    for target, (_, held, f) in claims.items():
        w1 = k * held / (1.0 - held)
        f2 = shares.get(target, 0.0)
        w_pos, w_neg = f * w1 + f2 * w2, (1.0 - f) * w1 + (1.0 - f2) * w2
        for v_pos, v_neg in witnessed_by.get(target, ()):
            w_pos, w_neg = w_pos + v_pos, w_neg + v_neg
        if c_sit > 0.0 or target in witnessed_by:
            out[target] = expectation(w_pos, w_neg, k)


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
INHERIT = -6          # F37: "what the filler's classes go on with, its own record last" (deduction with exceptions)
KNOWN = -7            # F41: "how the filler's own lived sequences go on from here, not said yet" (a set of things)


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
    if target == INHERIT:
        return inherited(mind, filler, fixed) if filler is not None else None
    if target == KNOWN:
        return None                                    # a set: read where the frame's votes are (known_claim)
    if target in (NEXT, START):
        if filler is None:
            return None
        memo = mind.__dict__.get("_read_memo")         # a moment's reading resolves the same member many times
        key = ("resolve", target, filler, tuple(fixed))
        if memo is not None and key in memo:
            return memo[key]
        goes_on = start_continuations(mind, filler, fixed) if target == START else continuations(mind, filler, fixed)
        found = max(goes_on, key=goes_on.get) if goes_on else None
        if found is not None and getattr(mind.cfg, "known", False):
            top = goes_on[found]                       # F41: of several equally strong continuations none is "the
            if sum(1 for e in goes_on.values() if e >= top - 1e-12) > 1:   # strongest" (they are the things known:
                found = None                           # KNOWN reads them as a set)
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
    inherits = relational and getattr(mind.cfg, "inherit", False)
    memo = mind.__dict__.get("_read_memo")             # the same member, values and frame are read more than once in
    if memo is not None:                               # a moment (under each trust, and by every frame of the place):
        memo_key = ("value_keys", x, tuple(targets), tuple(fixed), relational, starts, exact, category, alike,
                    replies, inherits)                 # nothing is written while the memo is open, and no caller
        hit = memo.get(memo_key)                       # changes the lists it is given
        if hit is not None:
            return hit
        found = _value_keys_read(mind, x, targets, fixed, relational, starts, exact, category, alike, replies, inherits)
        memo[memo_key] = found
        return found
    return _value_keys_read(mind, x, targets, fixed, relational, starts, exact, category, alike, replies, inherits)


def _value_keys_read(mind, x: int, targets, fixed, relational: bool, starts: bool, exact: bool,
                     category: Optional[int], alike: bool, replies: bool, inherits: bool) -> List[List[int]]:
    """``_value_keys``, read from the substance (its switches already read)."""
    if exact:
        named: Dict[int, int] = {}
        if relational:
            if x in targets:
                named[x] = SELF
            if starts:
                best = resolve(mind, START, x, fixed)
                if best is not None and best not in named:
                    named[best] = START
            if inherits and _may_inherit(mind, x, targets):   # F37: x's property (its own record last) before
                best = resolve(mind, INHERIT, x, fixed)         # whatever followed x anywhere
                if best is not None and best not in named:
                    named[best] = INHERIT
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
    deduced = resolve(mind, INHERIT, x, fixed) if inherits and _may_inherit(mind, x, targets) else None
    keys = []
    for v in targets:
        if relational and v == x:
            keys.append([SELF])
        elif starts and v not in fixed and begins_with(mind, x, v):
            keys.append([START])                   # F26: v is how a sequence that began with x went on
        elif v == deduced:
            keys.append([INHERIT])                 # F37: v is x's property (what x's classes go on with)
        elif v in goes_on:
            keys.append([NEXT])
        elif v == stands:
            keys.append([ALIKE])                   # F31: v stands where x stands
        elif v == answered:
            keys.append([REPLY])                   # F34: v is what x on its own was answered with
        else:
            keys.append([v])
    return keys


def stem_mates(mind, p: int, since: int) -> List[Tuple[int, float]]:
    """F37: the lived words that share the longest stem with ``p`` (a common beginning of at least three letters; two
    lived words part there, so it is a boundary of the lived words, F36), with the share of the longer word the stem
    covers. Only the longest: "hayvanlar" for "hayvandır", not "hayır" (a shorter beginning is another stem, as F36's
    base is the word sharing the longest one)."""
    memo = _episode_memo(mind)
    key = ("mates", p, since)
    if key in memo:
        return memo[key]
    found: List[Tuple[int, float]] = []
    channels, symbols = mv_view(mind, "u_channel"), mv_view(mind, "u_symbol")
    word = str(mind.ch.symbol(channels[p], symbols[p]))
    if word.isalpha() and len(word) >= 3:
        _lexicon(mind, channels[p], since)
        lexicon = mind.__dict__.get("_lexicon")
        if lexicon is not None and lexicon[0][0] == channels[p]:
            index = mind.__dict__.get("_lexicon_index")
            if index is None or index[0] != lexicon[0]:   # the words by their first three letters, once
                by_head: Dict[str, list] = {}
                for entry in lexicon[1]:
                    by_head.setdefault(entry[0][:3], []).append(entry)
                index = mind.__dict__["_lexicon_index"] = (lexicon[0], by_head)
            shared = []
            for text, u, _ in index[1].get(word[:3], ()):
                if u == p:
                    continue
                n = 0
                for a, b in zip(word, text):
                    if a != b:
                        break
                    n += 1
                shared.append((n, u, n / max(len(word), len(text))))
            longest = max((n for n, _, _ in shared), default=0)
            found = [(u, cover) for n, u, cover in shared if n == longest]
    memo[key] = found
    return found


def inherited(mind, x: int, fixed=()) -> Optional[int]:
    """F37 (Inheritance): what x's classes go on with, x's own record last: deduction with exceptions.

    x is a member of the places it filled (before the present run). The most telling fixed word of each place's frame
    (1 / (1 + the contexts that hold it), its share over the frame's words) names it. A lived word sharing the longest
    stem with that word (``stem_mates``: "kuşlar" for "kuştur") is the class spoken of on its own, and what it goes on
    with is a property of the class, except the form of being of a class (``_forms_of``: "bir" of "kuşlar bir hayvandır
    ."); what it inherits from the classes it is said to be of counts too, one step more (``_inheritance``, depth 2:
    member, class, kind). By the deduction of NARS, from "x is a member" (f1, c1) and "the class goes on with v" (f2 =
    v's share of its continuations, or the frequency of what it inherits; c2 = their conviction): f = f1 f2,
    c = f1 f2 c1 c2 f_n, the naming of the place by the word (f_n = its share x the stem's cover) weighing how sure the
    derivation is, not whether the property holds; w = k c / (1 - c), and the derivations add (revision). x's own
    record comes last: what x itself was lived to go on with that its class-mates (the members of the same places)
    mostly did not (what every member goes on with, "bir" of "X bir kuştur", is the class's form, not x's own) nests
    over the deduced evidence, and a deduced property x never went on with while it went on with such a thing of its
    own counts against: an exception ("penguen yüzer" over "kuşlar uçar"). The value with the highest expectation
    over 1/2, never a fixed word of the frame."""
    memo = _moment_memo(mind)
    key = ("inherit", x, tuple(fixed))
    if key in memo:
        return memo[key]
    since = getattr(mind, "episode_start", mind.t)
    order, values, e_deduced, lived, k = _inheritance(mind, x, since)[:5]
    fixed_set = set(fixed)
    lived_mass = sum(p for v, (p, _) in lived.items() if v not in fixed_set)
    factor = 1.0 - conviction(lived_mass, k) if lived_mass > 0.0 else 1.0
    best, best_e = None, 0.5
    for i in order:                                    # the deduced values by expectation, the best not fixed nor own
        v = values[i]
        if v in fixed_set or v in lived:
            continue
        if factor * e_deduced[i] > best_e:             # an exception: x went on with another thing of its own
            best, best_e = v, factor * e_deduced[i]
        break
    position = {v: i for i, v in enumerate(values)} if lived else {}
    for v, (p, n) in lived.items():                    # x's own record comes last
        if v in fixed_set:
            continue
        c = conviction(p + n, k)
        i = position.get(v)
        e = c * frequency(p, n) + (1.0 - c) * (e_deduced[i] if i is not None else 0.5)
        if e > best_e:
            best, best_e = v, e
    memo[key] = best
    return best


def _may_inherit(mind, x: int, targets) -> bool:
    """F37: whether a value x lived at a place can be its inherited property at all: one of the classes of the places
    x filled goes on with it, or x itself went on with it (its own record). When not, the deduction is not read (it
    would name a value the member never lived there, which counts for nothing)."""
    since = getattr(mind, "episode_start", mind.t)
    channel = mv_view(mind, "u_channel")[x]
    own = None
    for category, e_m in memberships_before(mind, x, since).items():
        if e_m <= 0.5:
            continue
        profile = _class_profile(mind, category, channel, since, 2)
        if profile is None:
            continue
        if own is None:
            own = _continuation_set(mind, x)
        named = profile[3]
        if any(v in named or v in own for v in targets):
            return True
    return False


def _inheritance(mind, x: int, since: int, depth: int = 2):
    """F37: (the deduced values in order of expectation, the values, their expectations, x's own record
    {v: (W+, W-)}, k, the values' W+, their W-) for ``inherited``; read once a run. The deduced evidence of each class
    word is read as arrays and the derivations are added value by value (revision). depth 2: a class word's own
    classes count too (what it inherits, one step), so a conclusion two steps away is reached (member, class, kind)."""
    memo = _episode_memo(mind)
    key = ("inheritance", x, since, depth)
    if key in memo:
        return memo[key]
    S, k = mind.S, mind.cfg.k
    kinds, channels = mv_view(mind, "u_kind"), mv_view(mind, "u_channel")
    world, act = kind_masks(mind)[3], kind_masks(mind)[5]
    pieces = []
    mates = set()
    src = mv_view(mind, "t_src")
    for category, e_m in memberships_before(mind, x, since).items():
        if e_m <= 0.5:
            continue
        tie = S.find_tie(x, category, ABSTRACT, 0)
        w_pos, w_neg = mass1(mind, tie) if tie >= 0 else (0.0, 0.0)
        f1, c1 = frequency(w_pos, w_neg), conviction(w_pos + w_neg, k)
        profile = _class_profile(mind, category, channels[x], since, depth)
        if profile is None:
            continue
        v_arr, f2_arr, g_arr, named = profile
        if x not in named:                               # what the class words go on with, carried to x at once
            mates.add(category)
            f = f1 * f2_arr
            c = np.minimum(f * c1 * g_arr, 1.0 - 1e-9)
            w = k * c / (1.0 - c)
            pieces.append((v_arr, f * w, (1.0 - f) * w))
            continue
        frame = category + 1                             # x is itself a class word or a property here: one by one
        words = [q for q in parts_of(mind, frame) if q != category and kinds[q] & ATOMIC and act[channels[q]]]
        telling = [1.0 / (1 + len(S.inc.get((q, PART), ()))) for q in words]
        total = sum(telling)
        most = max(range(len(words)), key=lambda i: telling[i])
        for q, t in [(words[most], telling[most])]:
            for mate, cover in stem_mates(mind, q, since):
                if mate == x:
                    continue
                goes, mass_all = _class_goes(mind, mate, channels[x])
                if goes is None:
                    continue
                v_arr, p_arr, present = goes
                own = p_arr[v_arr == x].sum() if x in present else 0.0
                mass = mass_all - float(own)
                if mass <= 0.0:
                    continue
                if own:
                    keep = v_arr != x
                    v_arr, p_arr = v_arr[keep], p_arr[keep]
                c2, f_n = conviction(mass, k), (t / total) * cover
                mates.add(category)
                f = f1 * (p_arr / mass)
                c = np.minimum(f * c1 * c2 * f_n, 1.0 - 1e-9)
                w = k * c / (1.0 - c)
                pieces.append((v_arr, f * w, (1.0 - f) * w))
    lived = {}
    values, e_deduced, order, pos_list, neg_list = [], [], [], [], []
    if pieces:
        v_all = np.concatenate([pc[0] for pc in pieces])
        uniq, inverse = np.unique(v_all, return_inverse=True)
        w_pos = np.bincount(inverse, weights=np.concatenate([pc[1] for pc in pieces]), minlength=uniq.size)
        w_neg = np.bincount(inverse, weights=np.concatenate([pc[2] for pc in pieces]), minlength=uniq.size)
        e = (w_pos + 0.5 * k) / (w_pos + w_neg + k)      # the expectation of each deduced value
        values, e_deduced, pos_list, neg_list = uniq.tolist(), e.tolist(), w_pos.tolist(), w_neg.tolist()
        order = np.argsort(-e, kind="stable").tolist()
        members = {src[t] for g in mates for t in list(S.inc.get((g, ABSTRACT), ()))[:MEMBERS]} - {x}
        own = _unit_arrays(mind, x, 1)                  # x's own record: what it went on with that its class-mates
        if own is not None:                             # mostly did not
            sets = [_continuation_set(mind, y) for y in members]
            for v, p, n in zip(own[0], own[1], own[2]):
                if not world[channels[v]] or channels[v] != channels[x] or p <= 0.0:
                    continue
                shared = sum(1 for went in sets if v in went)
                if shared * 2 < max(1, len(members)):
                    lived[v] = (float(p), float(n))
    found = (order, values, e_deduced, lived, k, pos_list, neg_list)
    memo[key] = found
    return found


def _class_profile(mind, category: int, channel: int, since: int, depth: int = 1):
    """F37: what the class words of a place go on with, for any member: (items, f2 = the item's share of its class
    word's continuations, g = c2 f_n, every class word and item named), one entry per class word and item (a member's
    deduction is then f = f1 f2, c = f c1 g); None when the place names no class. Read once a run."""
    memo = _episode_memo(mind)
    key = ("class_profile", category, channel, since, depth)
    if key in memo:
        return memo[key]
    S, k = mind.S, mind.cfg.k
    kinds, channels = mv_view(mind, "u_kind"), mv_view(mind, "u_channel")
    act = kind_masks(mind)[5]
    found = None
    frame = category + 1                                 # a place's frame is born right after it (induce_slots)
    if frame < S.n_units and kinds[frame] & SLOTTED:
        words = [q for q in parts_of(mind, frame) if q != category and kinds[q] & ATOMIC and act[channels[q]]]
        if words:
            telling = [1.0 / (1 + len(S.inc.get((q, PART), ()))) for q in words]
            total = sum(telling)
            vs, f2s, gs, named = [], [], [], set()
            most = max(range(len(words)), key=lambda i: telling[i])      # the class is named by the most telling
            for q, t in [(words[most], telling[most])]:                  # fixed word of the place's frame
                for mate, cover in stem_mates(mind, q, since):
                    goes, mass = _class_goes(mind, mate, channel)
                    if goes is not None and mass > 0.0:
                        v_arr, p_arr, present = goes
                        named.add(mate)
                        named.update(present)
                        vs.append(v_arr)
                        f2s.append(p_arr / mass)
                        gs.append(np.full(v_arr.size, conviction(mass, k) * (t / total) * cover))
                    if depth >= 2 and _said_of_a_class(mind, mate, since):   # what the class word inherits
                        inh = _inheritance(mind, mate, since, depth - 1)        # from the classes it is said to be of
                        values, pos, neg = inh[1], inh[5], inh[6]
                        if values:
                            w = np.array(pos) + np.array(neg)
                            named.add(mate)
                            named.update(values)
                            vs.append(np.array(values, dtype=np.int64))
                            f2s.append(np.array(pos) / w)
                            gs.append(w / (w + k) * ((t / total) * cover))
            if vs:
                found = (np.concatenate(vs), np.concatenate(f2s), np.concatenate(gs), frozenset(named))
    memo[key] = found
    return found


def _class_goes(mind, mate: int, channel: int):
    """F37: what a class word went on with on ``channel`` (world items, W+ > 0, not itself, not the form of being of
    a class): ((items, W+) as arrays, their set), with the total W+; (None, 0) when nothing. Read once a run, for every
    member it is a class of."""
    memo = _episode_memo(mind)
    key = ("class_goes", mate, channel)
    hit = memo.get(key)
    if hit is None:
        hit = (None, 0.0)
        found = _unit_arrays(mind, mate, 1)
        if found is not None:
            world, channels = kind_masks(mind)[3], mv_view(mind, "u_channel")
            dst, pos, _ = found
            forms = _forms_of(mind, mate)
            kept = [(int(v), float(p)) for v, p in zip(dst, pos)
                    if p > 0.0 and world[channels[v]] and channels[v] == channel and v != mate and v not in forms]
            if kept:
                v_arr = np.array([v for v, _ in kept], dtype=np.int64)
                p_arr = np.array([p for _, p in kept], dtype=np.float64)
                hit = ((v_arr, p_arr, frozenset(v_arr.tolist())), float(p_arr.sum()))
        memo[key] = hit
    return hit


def _said_of_a_class(mind, m: int, since: int) -> bool:
    """F37: whether a word filled a place whose most telling fixed word shares a stem with a lived word that went on
    with something (a place that can name a class): without one, nothing can be inherited by it. Read once a run."""
    memo = _episode_memo(mind)
    key = ("said_of", m, since)
    if key in memo:
        return memo[key]
    S = mind.S
    kinds, channels = mv_view(mind, "u_kind"), mv_view(mind, "u_channel")
    act = kind_masks(mind)[5]
    found = False
    for place, e_m in memberships_before(mind, m, since).items():
        if e_m <= 0.5:
            continue
        frame = place + 1
        if frame >= S.n_units or not kinds[frame] & SLOTTED:
            continue
        words = [q for q in parts_of(mind, frame) if q != place and kinds[q] & ATOMIC and act[channels[q]]]
        if not words:
            continue
        telling = [1.0 / (1 + len(S.inc.get((q, PART), ()))) for q in words]
        q = words[max(range(len(words)), key=lambda i: telling[i])]
        if any(S.out.get((mate, EXPECT, 1)) for mate, _ in stem_mates(mind, q, since)):
            found = True
            break
    memo[key] = found
    return found


def _forms_of(mind, m: int) -> frozenset:
    """F37: the fixed words of the places a class word filled that say it is of a class ("bir", "hayvandır" of "kuşlar
    bir hayvandır ."): what the class word went on with there is the form of being of a class, not a property of its
    own. A place says so when its most telling fixed word shares a stem with a lived word that went on with something
    (a class spoken of on its own). Read once a run."""
    memo = _episode_memo(mind)
    key = ("forms", m)
    if key in memo:
        return memo[key]
    S = mind.S
    since = getattr(mind, "episode_start", mind.t)
    kinds, channels = mv_view(mind, "u_kind"), mv_view(mind, "u_channel")
    act = kind_masks(mind)[5]
    found = set()
    for place, e_m in memberships_before(mind, m, since).items():
        if e_m <= 0.5:
            continue
        frame = place + 1                                # a place's frame is born right after it (induce_slots)
        if frame >= S.n_units or not kinds[frame] & SLOTTED:
            continue
        words = [q for q in parts_of(mind, frame) if q != place and kinds[q] & ATOMIC and act[channels[q]]]
        if not words:
            continue
        telling = [1.0 / (1 + len(S.inc.get((q, PART), ()))) for q in words]
        q = words[max(range(len(words)), key=lambda i: telling[i])]
        if any(S.out.get((mate, EXPECT, 1)) for mate, _ in stem_mates(mind, q, since) if mate != m):
            found.update(words)
    found = frozenset(found)
    memo[key] = found
    return found


def _continuation_set(mind, y: int) -> frozenset:
    """F37: the items y has a lag-1 EXPECT tie to (what it went on with, whatever the evidence); once a run."""
    memo = _episode_memo(mind)
    key = ("goes_to", y)
    hit = memo.get(key)
    if hit is None:
        ties = mind.S.out.get((y, EXPECT, 1))
        dst = mv_view(mind, "t_dst")
        hit = frozenset(dst[t] for t in ties) if ties else frozenset()
        memo[key] = hit
    return hit


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


def _episode_memo(mind) -> dict:
    """F37: a memo kept while the present run of items lasts (from ``episode_start``): what the classes of the places
    a word filled go on with is read once a run, as it was known when it was first read in the run. Dropped when a
    new run begins, and with the between-moments memo (undo, retraction)."""
    since = getattr(mind, "episode_start", mind.t)
    held = mind.__dict__.get("_episode_memo")
    if held is None or held[0] != since:
        held = mind.__dict__["_episode_memo"] = (since, {})
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
    with (F34, REPLY), or an item every member has there; at least one a relation. Returns (frame, e, tail) for every
    prefix that has such frames, longest first, tail the entries in order (relations as their keys, items as
    themselves): which of them speaks is the members' to say (a longer prefix whose members never lived on says
    nothing, and must not hide a shorter one whose members did).
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
    inherit = getattr(mind.cfg, "inherit", False)
    memo = mind.__dict__.get("_read_memo")             # read under each trust of a moment: the same frames (they do
    if memo is not None:                               # not depend on the trust; nothing is written meanwhile)
        memo_key = ("bound_frames", unit, paradigmatic, replies, inherit)
        hit = memo.get(memo_key)
        if hit is None:
            hit = memo[memo_key] = tuple(_bound_frames_read(mind, unit, kinds, paradigmatic, replies, inherit))
        return list(hit)
    return _bound_frames_read(mind, unit, kinds, paradigmatic, replies, inherit)


def _bound_frames_read(mind, unit: int, kinds, paradigmatic: bool, replies: bool, inherit: bool):
    """``bound_frames``, read from the substance."""
    relations = ([ALIKE] if paradigmatic else []) + ([REPLY] if replies else []) + \
        ([INHERIT] if inherit else [])
    parts = parts_of(mind, unit)
    shortest = 2 if replies else len(parts) - 1          # F31 alone binds the last item only
    every = []
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
        every += found
    return every


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
    known = relational and tail is None and bool(where) and getattr(mind.cfg, "known", False)   # F41: the filler's
    if known:                                          # last place is in the turn that is going on (what a filler in
        channels, sense = mv_view(mind, "u_channel"), kind_masks(mind)[4]   # the other's turn was answered with is
        known = not any(sense[channels[p]] for p in parts[where[-1] + 1:])  # REPLY's to say, F34)
    known_of: Dict[int, float] = {}
    not_known: Dict[int, float] = {}
    own_name: Dict[int, float] = {}
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
        known_x: Dict[int, float] = {}
        if known:                                      # F41: how x's own lived sequence goes on from its last
            known_x = known_unsaid(mind, x, instance, where[-1], trust)              # place in the instance
        places = memberships_before(mind, x, since) if alike else None
        loose = []
        for v, p, keys in zip(targets, own, _value_keys(mind, x, targets, fixed, relational, starts, exact, category,
                                                        inline=False)):
            share = weight * p / total
            for key in keys:
                shares[key] = shares.get(key, 0.0) + share
            if known:                                  # F41: said by a member of which it was known (with the lived
                if v == x:                             # sequence of the member that it carried on), or not; a
                    own_name[v] = own_name.get(v, 0.0) + share   # member that said itself
                if v in known_x:
                    key = (v, S.tuple_index.get(tuple(x if j in where else p for j, p in enumerate(parts))
                                                [where[-1]:] + (v,), -1))
                    known_of[key] = known_of.get(key, 0.0) + share
                else:
                    not_known[v] = not_known.get(v, 0.0) + share
            if places and keys == [v] and v != x and v not in fixed and v not in rivals \
                    and places.keys() & memberships_before(mind, v, since).keys():
                loose.append((v, share))               # named by no other relation, and it has filled a place with x
        if loose:
            may.append((x, loose))
            may_share += sum(share for _, share in loose)
    if voted <= 0.0:
        return {}, 0.0, where, fixed, 0
    if known:                                          # F41: how far the frame went on with things known of its
        if known_of:                                   # members, each word's share of it, the words that only members
            shares[KNOWN] = sum(known_of.values())     # that knew them said, and the members that said only themselves
        _moment_memo(mind)[("known_of", frame, exclude)] = (
            {key: w / voted for key, w in known_of.items()},
            frozenset(v for v, _ in known_of if not_known.get(v, 0.0) <= 0.0),
            frozenset(v for v, w in own_name.items() if shares.get(v, 0.0) <= w + 1e-12))
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
OPEN_CHANNEL = "acik"   # F46: the mind's own channel for what it found it did not know of an item (open claims)


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


def _note_claim(mind, filler: int, e: float, is_open: bool) -> None:
    """F46 (Open Claims): the frame that speaks at the present moment claims, with expectation ``e``, that what
    follows is a relation of its filler's own record (how its sequences go on, what it was answered with, what
    stands where it stands, what is known of it). When the relation names nothing for this filler, the claim is
    open: the mind's own evidence says something follows here, as it did for every member, and of this item it has
    nothing. Kept for the moment: of the claims over 1/2 the strongest, and whether it is open (``Mind.step``
    writes it, ``learn.note_claims``)."""
    if e <= 0.5:
        return
    memo = _moment_memo(mind)
    held = memo.get(("claim", filler))
    if held is None or e > held[0]:
        memo[("claim", filler)] = (e, is_open)


def claim_again(mind, context: int, filler: int) -> Optional[Tuple[float, bool]]:
    """F47: the claim that was left open at ``context`` about ``filler``, read again as the mind stands now: (e,
    whether it is still open), or None when no frame claims a relation of the filler's record there any more (or the
    context is gone). Reads only."""
    S = mind.S
    v = S.arrays()
    if context is None or context >= S.n_units or not v.ualive[context] or not v.kind[context] & COMPOSED \
            or filler not in parts_of(mind, context):
        return None
    mind.__dict__.pop("_between_memo", None)           # read afresh: nothing noted before counts
    memo = mind.__dict__.get("_read_memo")
    if memo is not None:
        memo.pop(("claim", filler), None)
    mind.__dict__["_claim_again"] = context
    try:
        members_map(mind, context)
        return _moment_memo(mind).pop(("claim", filler), None)
    finally:
        mind.__dict__.pop("_claim_again", None)
        mind.__dict__.pop("_between_memo", None)


LOOK_ALL = "*"          # F47: the unit of every look (what looking things up has led to in general)
THINK_CHANNEL = "dusun"  # F48: the mind's own channel for what its thoughts have led to
THINK_ALL = "*"         # F48: the unit of every thought


def think_expectation(mind) -> float:
    """F48: what thinking is expected to lead to: the expectation of the tie (every thought -> relief); 1/2 while
    the mind has not thought by its own decision yet."""
    unit = mind.unit_of(THINK_CHANNEL, THINK_ALL)
    tie = mind.S.find_tie(unit, mind.feel_unit["relief"], EXPECT, 1) if unit >= 0 else -1
    if tie < 0:
        return 0.5
    w_pos, w_neg = mass1(mind, tie)
    return expectation(w_pos, w_neg, mind.cfg.k)


WONDER = "sor"          # F51: the unit of every question the mind asked itself (on its channel for thoughts)


def wonder_expectation(mind) -> float:
    """F51: what asking itself about a new item is expected to lead to: the expectation of the tie (asking itself ->
    relief); 1/2 while it has never done so."""
    unit = mind.unit_of(THINK_CHANNEL, WONDER)
    tie = mind.S.find_tie(unit, mind.feel_unit["relief"], EXPECT, 1) if unit >= 0 else -1
    if tie < 0:
        return 0.5
    w_pos, w_neg = mass1(mind, tie)
    return expectation(w_pos, w_neg, mind.cfg.k)


def look_expectation(mind, symbol) -> float:
    """F47: what looking ``symbol`` up is expected to lead to, read general -> specific as everywhere: e <- e(every
    look -> relief), then e <- c f + (1 - c) e with the item's own record (f, c). 1/2 when nothing was ever looked
    up."""
    k, relief = mind.cfg.k, mind.feel_unit["relief"]
    e = 0.5
    for name, own in ((LOOK_ALL, False), (symbol, True)):
        unit = mind.unit_of(OPEN_CHANNEL, name)
        tie = mind.S.find_tie(unit, relief, EXPECT, 1) if unit >= 0 else -1
        if tie < 0:
            continue
        w_pos, w_neg = mass1(mind, tie)
        if own:
            c = conviction(w_pos + w_neg, k)
            e = c * frequency(w_pos, w_neg) + (1.0 - c) * e
        else:
            e = expectation(w_pos, w_neg, k)
    return e


def open_claim(mind, symbol) -> float:
    """F46: how far the mind holds that claims about the item ``symbol`` are left open: the expectation of the tie
    (its unit on the mind's own channel -> holds), 1/2 when no claim about it was ever left open."""
    unit = mind.unit_of(OPEN_CHANNEL, symbol)
    tie = mind.S.find_tie(unit, mind.holds, EXPECT, 0) if unit >= 0 else -1
    if tie < 0:
        return 0.5
    w_pos, w_neg = mass1(mind, tie)
    return expectation(w_pos, w_neg, mind.cfg.k)


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
    F41 (``known``): a frame's value may be how its filler's own lived sequences go on from here, a thing known of
    the filler and not said yet in the episode (KNOWN: ``known_unsaid``, ``known_claim``); a word the members said of
    themselves is not claimed for another filler (``explained``); and in an answer that tells what is known of its
    topic, where no word would be said, the members' own known sequences say how to go on (``sequence_reading``).
    F24 (voices): each score is weighted by the credibility the mind has learned for voices of that kind (how many
    members speak, how much they agree; with no such experience every kind reads 1/2 and the choice is exactly
    F11's). For the context read with the full trust, the voice that spoke and every voice that could have spoken
    are noted with their claims (top value, frequency), so that what comes next judges them (learn.judge_voices).
    """
    cfg, S = mind.cfg, mind.S
    voices = getattr(cfg, "voices", False)
    faults = getattr(cfg, "faults", False)             # F49: which way made each claim, and how far it is believed
    src: Optional[Dict[int, list]] = {} if faults else None    # target -> [(ways, the neutral claim, the claim)]
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
        if KNOWN in mass and len(mass) > 1:            # F41: a frame is chosen by the values it names (that its
            top = max(frequency(p, n) for key, (p, n) in mass.items() if key != KNOWN)   # members said things known
        score = e_abs * c_members * top                # of them is true of many frames)
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
        known_on = getattr(cfg, "known", False) and filler is not None and tail is None
        echo_on = known_on or (getattr(cfg, "echo", False) and filler is not None and tail is None)
        known_here: Dict[int, float] = {}
        said_known, only_known, own_names = {}, frozenset(), frozenset()
        open_on = getattr(cfg, "open_claims", False) and filler is not None and trust is None \
            and (unit == mind.context or mind.__dict__.get("_claim_again") == unit)   # F46: the claims of the present
                                                   # moment, read with the full trust (F47: or a claim read again)
        if filler is not None and tail is None and getattr(cfg, "find_out", False):
            _moment_memo(mind)[("filler", unit)] = filler   # F50: the item the frame that speaks is about
        topic = getattr(mind, "topic", None)
        telling = where and (where[-1] < len(parts_of(mind, unit)) - 1 or (
            topic is not None and topic[0] == filler and topic[2] == getattr(mind, "episode_start", None)))
        if known_on:                               # F41: how the filler's own lived sequence goes on from here
            if KNOWN in mass:
                known_here = known_unsaid(mind, filler, unit, where[-1], trust, episode=unit == mind.context)
            said_known, only_known, own_names = _moment_memo(mind).get(("known_of", frame, unit),
                                                                       ({}, frozenset(), frozenset()))

        def explained(t: int) -> bool:
            """F41: a word the members said because of what they are or of what was known of them is that
            relation's evidence, not the word's, and is not claimed for another filler:
            * a member's own name, said by no one but itself (SELF);
            * a word that only members of which it was known said, where the filler's own knowledge stands against
              it: inside the filler's own lived sequence, when that sequence goes on otherwise; just after the
              filler's name in an answer that tells what is known of it (it is the topic), when another thing of
              the word's kind is known of it (``goes_alike`` over 1/2). Elsewhere what the members agree on is the
              frame's to say (a property a thing has from its kind, never told of it)."""
            if t in own_names:
                return True
            if not telling or not known_here or t in known_here or t not in only_known:
                return False
            if where[-1] < len(parts_of(mind, unit)) - 1:
                return True                        # its own lived sequence goes on otherwise here
            lived = [seq for word, seq in said_known if word == t and seq >= 0]
            here = parts_of(mind, unit)[where[-1]:]
            for v in known_here:
                alike = goes_alike(mind, v, t, trust)
                carried = S.tuple_index.get(here + (v,), -1) if alike <= 0.5 and lived else -1
                if alike > 0.5 or (carried >= 0 and any(goes_alike(mind, carried, seq, trust) > 0.5 for seq in lived)):
                    return True
            return False
        for target, (w_pos, w_neg) in mass.items():
            held = c                               # F29: what the members agree on is theirs (the frequency, all
            if target in carried:                  # members); how firmly it holds for this instance is the stronger
                held = max(c, conviction(sum(carried[target]), cfg.k))   # of breadth and interchangeable depth
            f = frequency(w_pos, w_neg)
            e = held * f + (1.0 - held) * 0.5
            if target == KNOWN:                    # F41: the thing known of the filler most like what the members
                target, f = known_claim(mind, known_here, said_known, mass, trust,   # said of themselves here
                                        parts_of(mind, unit)[where[-1]:])
                if open_on:
                    _note_claim(mind, filler, e, target is None)
                if target is not None:
                    e = held * f + (1.0 - held) * 0.5
                    if faults:
                        src.setdefault(target, []).append((KNOWN, e))   # F49: the kind of claim that named it
                    out[target] = max(out.get(target, 0.0), e)
                    _moment_memo(mind)[("known_claim", unit)] = (filler, category, target)
                continue
            if known_on and target >= 0 and target != filler and explained(target):
                continue                           # F41: what a member said of itself is not said of another filler
            asked = target
            if echo_on and target in (NEXT, START) and unit == mind.context:
                target = resolve(mind, target, filler, tuple(fixed) + tuple(sorted(episode_said(mind, filler))),
                                 category)         # F42: what followed the filler in this episode is an echo
            else:
                target = resolve(mind, target, filler, fixed, category)
            if open_on and asked < SELF:           # F46: a relation that reads what was lived of the filler itself
                _note_claim(mind, filler, e, target is None)
            if target is None:
                continue
            if faults:
                src.setdefault(target, []).append((asked, e))   # F49: the kind of claim that named this value
            out[target] = max(out.get(target, 0.0), e)
            if corroborate and (target not in claims or e > claims[target][0]):
                claims[target] = (e, held, f)
        if claims:
            corroborated(mind, unit, trust, frame, claims, out, tail)
        if faults:                                 # F49: each claim of the frame, read with what claims of its
            from . import hata                     # way have come to (the frame's reading is the neutral claim)
            doubted = 0.0
            for target, keys in src.items():       # (the strongest kind of claim carries the corroborated one)
                top = max(e for _, e in keys)
                made = []
                for key, e0 in keys:
                    e0 = out[target] if e0 == top else e0
                    ways = hata.frame_ways(key, frame)
                    made.append((ways, e0, hata.believed(mind, ways, e0)))
                made.sort(key=lambda claim: -claim[2])
                src[target] = made
                out[target] = made[0][2]
                if made[0][2] <= 0.5 < made[0][1]:
                    doubted = max(doubted, made[0][1])
            if filler is not None and tail is None and hata.wanting(mind, hata.frame_ways(0, frame)) \
                    and mind.__dict__.get("_fault_narrower", True):   # (research: False leaves this step out)
                ways = hata.frame_ways(hata.NARROWER, frame)   # the frame's own vote was found wanting: read it
                for target, e0 in narrower_vote(mind, frame, unit, filler, c, trust).items():   # through a
                    made = src.setdefault(target, [])          # narrower place of its filler
                    made.append((ways, e0, hata.believed(mind, ways, e0)))
                    made.sort(key=lambda claim: -claim[2])
                    out[target] = max(out.get(target, 0.0), made[0][2])
            if getattr(cfg, "find_out", False) and filler is not None:
                _own_record_against_shared(mind, out, src)   # F50: what was lived of the thing itself stands
                                                             # against what the others share (the members of a
                                                             # narrower place of it are others too)
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
    if getattr(cfg, "known", False) and unit == mind.context and getattr(mind, "topic", None) is not None:
        said_line = 0.5 + cfg.theta / (2.0 * cfg.T)     # F41: when no word would be said, how the members went on
        words = mv_view(mind, "u_channel")              # where their own known sequences stood as the topic's does
        had = out.items() if getattr(cfg, "find_out", False) else _had(out, src)   # (F50: what the others share,
        if not any(e > said_line and words[t] == words[mind.topic[0]] for t, e in had):   # found wanting, does not
            _take(mind, out, src, sequence_reading(mind, unit, trust), "dizi")   # stand in the way of what is known
                                                                                 # of the thing itself)
    if cfg.backoff and not any(e > 0.5 for t, e in _had(out, src) if _world(mind, t)):
        _take(mind, out, src, continuation_backoff(mind, unit, trust), "devam")
    if getattr(cfg, "substitution", False) and not any(e > 0.5 for t, e in _had(out, src) if _world(mind, t)):
        _take(mind, out, src, substitute_backoff(mind, unit, trust), "yerine")
    line = 0.5 + mind.knob["theta"] / (2.0 * cfg.T)     # F36: when nothing would be said (the speaking line), a
    if getattr(cfg, "forms", False) and not any(e > line for t, e in _had(out, src) if _world(mind, t)):
        _take(mind, out, src, form_reading(mind, unit, trust), "biçim")   # word never met is read as its form
    own_line = 0.5 + cfg.theta / (2.0 * cfg.T)          # F39 only when nothing would be said at the mind's own
    if getattr(cfg, "difference", False) and not any(e > own_line for t, e in _had(out, src) if _world(mind, t)):
        _take(mind, out, src, replacement_reading(mind, unit, trust), "fark")   # line (not the one raised to listen)
    if getattr(cfg, "situations", False) and not any(e > 0.5 for t, e in _had(out, src) if _world(mind, t)):
        _take(mind, out, src, situation_backoff(mind, unit, trust), "durum")
    if src:
        mind.__dict__["_src_last"] = (unit, src)         # F49: the chain of this reading (act.decide keeps its own)
        if best is not None and open_on and doubted > own_line and \
                not any(e > own_line for t, e in out.items() if _world(mind, t)):
            _note_claim(mind, filler, doubted, True)     # F46: what its frame would have said of this filler it has
    return out                                           # learned not to believe, and nothing else is said: open


OWN_RECORD = (-3, -5, -7)       # F50: the relations of a frame that read a record of the filler lived apart from
                                # the frame's own instances: how the sequences that began with it went on, what it
                                # was answered with alone, what is known of it. Not its continuation anywhere
                                # (NEXT): for a member that includes the very dialogues the frame generalises, so
                                # it holds for every member whatever else was lived of it, and says nothing of
                                # whether what was read of a new thing answers the question


def _own_record_against_shared(mind, out: Dict[int, float], src: Dict[int, list]) -> None:
    """F50: the claims of the frame that speaks, where a relation that reads the filler's own record names a word
    r (the members vouch for the relation: for each of them it named what that member lived) and the members' own
    word proposes another word v of the same channel. A channel carries one item a moment, so "r follows", held
    with conviction c_r, is "v does not follow" held as firmly: as the context's own record stands against what a
    generalisation proposes there (F27), nested last,
        e_v <- c_r * 0 + (1 - c_r) e_v,   c_r = max(0, 2 e_r - 1)
    (the conviction of a claim e = c f + (1 - c)/2 that all its members agree on). A relation the mind has learned
    not to believe (F49: e_r under 1/2) excludes nothing. What the thing itself was lived to go on with is more
    specific than what the things of its kind share; before this rule the two tied, and the mind that had just read
    the answer about the thing itself said nothing (measured)."""
    channels = mv_view(mind, "u_channel")
    own: Dict[int, float] = {}                               # channel -> the firmest claim of the filler's own record
    named = set()
    for target, made in src.items():
        for ways, _, e in made:
            kind = ways[1]
            if len(kind) == 2 and kind[0] == "g" and kind[1] in OWN_RECORD and e > 0.5:
                named.add(target)
                own[channels[target]] = max(own.get(channels[target], 0.0), 2.0 * e - 1.0)
    if not own:
        return
    for target, made in src.items():
        c = own.get(channels[target], 0.0)
        if target in named or c <= 0.0:
            continue
        src[target] = [(ways, e0, (1.0 - c) * e) for ways, e0, e in made]
        out[target] = (1.0 - c) * out[target]


def narrower_vote(mind, frame: int, unit: int, y: int, c_frame: float, trust=None) -> Dict[int, float]:
    """F49 (restructuring): the claims of ``frame`` for the instance ``unit`` read through a narrower place of its
    filler ``y``. A frame's category is every item that ever filled its place; when its members' vote was found
    wanting (the mind's own record of it, ``hata.wanting``), the vote is taken again among the members that share
    *another* lived place with the filler (a category both were members of before the present run of items: the
    place of things said to be of one kind, of things something else was done to, ...). Each such place is a frame
    of its own in effect (this frame, among those fillers only), and as among frames (F11) the one whose members
    agree most speaks: score = conviction(members) max_v f(v), its claims
        e = c f + (1 - c) 1/2,   c = c_frame conviction(members there) / conviction(all members)
    (``c_frame`` is the frame's own lam e(i -> g) conviction(members): the same claim with the members counted
    anew). A place that holds every member is not narrower and says nothing. So the category is split by what its
    members share elsewhere, where the error showed that it had to be; whether such readings hold is settled like
    any other way's (the record of ("g", NARROWER, frame)). Reads only."""
    S, k = mind.S, mind.cfg.k
    kinds = mv_view(mind, "u_kind")
    parts = parts_of(mind, frame)
    cats = {p for p in parts if kinds[p] & CATEGORY}
    if len(cats) != 1:
        return {}
    category = cats.pop()
    where = tuple(j for j, p in enumerate(parts) if p == category)
    src = mv_view(mind, "t_src")
    lived: Dict[int, Dict[int, float]] = {}                # member -> its instance's shares of what it went on with
    for tie in S.inc.get((category, ABSTRACT), ()):
        x = src[tie]
        if x == y:
            continue
        instance = S.tuple_index.get(tuple(x if j in where else p for j, p in enumerate(parts)))
        if instance is None or instance == unit:
            continue
        targets, own = _world_pos(mind, instance, trust)
        total = sum(own)
        if total > 0.0:
            lived[x] = {v: p / total for v, p in zip(targets, own)}
    if len(lived) < 2:
        return {}
    since = getattr(mind, "episode_start", mind.t)
    best = None
    for place, e_in in memberships_before(mind, y, since).items():
        if place == category or e_in <= 0.5:
            continue
        there = [x for x in (src[tie] for tie in S.inc.get((place, ABSTRACT), ())) if x in lived]
        if not there or len(there) == len(lived):
            continue                                       # no member shares it, or every member does: not narrower
        mass: Dict[int, float] = {}
        for x in there:
            for v, share in lived[x].items():
                mass[v] = mass.get(v, 0.0) + share
        n = float(len(there))
        score = conviction(n, k) * max(mass.values()) / n
        if best is None or score > best[0] or (score == best[0] and place < best[3]):
            best = (score, n, mass, place)
    if best is None:
        return {}
    _, n, mass, _ = best
    c = c_frame * conviction(n, k) / conviction(float(len(lived)), k)
    return {v: c * (m / n) + (1.0 - c) * 0.5 for v, m in mass.items()}


def _had(out: Dict[int, float], src: Optional[Dict[int, tuple]]):
    """The claims a less specific reading steps back for. F49: a reading the mind has learned not to believe has
    still spoken (the mind is unsure, it is not without a reading): its neutral claim counts here, so that a cruder
    reading does not take the place of one that was found wanting."""
    if not src:
        return out.items()
    return [(t, max([e] + [claim[1] for claim in src[t]]) if t in src else e) for t, e in out.items()]


def _take(mind, out: Dict[int, float], src: Optional[Dict[int, tuple]], reading: Dict[int, float], name: str) -> None:
    """What a reading that is not a frame's claims goes into ``out`` where it is the strongest. F49 (``src`` given):
    read with what claims of that way of reading have come to (``hata.believed``), and noted as the way that made
    the claim."""
    if src is None:
        for target, e in reading.items():
            out[target] = max(out.get(target, 0.0), e)
        return
    from . import hata
    ways = hata.reading_ways(name)
    for target, e0 in reading.items():
        e = hata.believed(mind, ways, e0)
        made = src.setdefault(target, [])
        made.append((ways, e0, e))
        made.sort(key=lambda claim: -claim[2])
        out[target] = max(out.get(target, 0.0), e)


def said_after(x: int, sequence) -> set:
    """F41: the items that follow ``x`` in a sequence of items (what has been said of x there)."""
    return {sequence[i + 1] for i in range(len(sequence) - 1) if sequence[i] == x}


def episode_said(mind, x: int) -> set:
    """F41: what has followed ``x`` in the present episode (heard or said), read from the episode's trace: every
    lived context of the episode holds its last items in order."""
    said = set()
    kinds = mv_view(mind, "u_kind")
    for context in list(getattr(mind, "trace", ())) + ([mind.context] if mind.context is not None else []):
        if kinds[context] & COMPOSED:
            said |= said_after(x, parts_of(mind, context))
    return said


def known_unsaid(mind, x: int, context: int, last: int, trust: Optional[torch.Tensor] = None,
                 episode: bool = False) -> Dict[int, float]:
    """F41 (Unsaid Known): how the own lived sequence of ``x`` goes on from its last place in a context (position
    ``last``), as shares {v: q} of the words that may come next.
    * x is the context's last item: how the lived sequences that began with x went on (``start_mass``, F26: a
      sentence read, an utterance after silence; not what followed x in the middle of other sequences), without what
      follows x in the context (with ``episode``, the present moment: in the whole episode, heard or said) and
      without what was only ever asked of it (``only_asked``).
    * words follow x: what the lived sequence that began with x and went on with those words went on with (the
      composed unit of that suffix, lived as a sequence of its own; the context itself is not its own knowledge).
    Empty when nothing is known to come next (the known sequence ended there, or it was never lived)."""
    kinds, channels = mv_view(mind, "u_kind"), mv_view(mind, "u_channel")
    channel = channels[x]
    parts = parts_of(mind, context) if kinds[context] & COMPOSED else (context,)
    suffix = parts[last:]
    found = {}
    if len(suffix) == 1:
        said = episode_said(mind, x) if episode else said_after(x, parts)
        for v, (began, _) in start_mass(mind, x, trust).items():
            if channels[v] == channel and v not in said and not only_asked(mind, x, v):
                found[v] = began
    else:
        sequence = mind.S.tuple_index.get(suffix)
        if sequence is None or sequence == context or kinds[sequence] & SLOTTED:
            return {}
        for v, p in zip(*_world_pos(mind, sequence, trust)):
            if channels[v] == channel:
                found[v] = p
    total = sum(found.values())
    return {v: p / total for v, p in found.items()} if total > 0.0 else {}


def only_asked(mind, x: int, v: int) -> bool:
    """F41: ``v`` is what was asked of ``x``, not what is known of it: every lived sequence that began with x and
    went on with v was answered (its turn ended and the words of another turn followed). A sequence that was read or
    told and left at that (nothing followed its end), or one too long to follow, makes v known; so does v never having
    begun a sequence with x (nothing says it was asked)."""
    memo = _moment_memo(mind)
    key = ("asked", x, v)
    if key in memo:
        return memo[key]
    S, F = mind.S, mind.knob["F"]
    kinds, channels = mv_view(mind, "u_kind"), mv_view(mind, "u_channel")
    sense, act = kind_masks(mind)[4:6]
    first = S.tuple_index.get((x, v))
    asked = told = False
    stack = [first] if first is not None and not kinds[first] & SLOTTED else []
    steps = 0
    while stack and steps < 64 and not told:
        steps += 1
        u = stack.pop()
        parts = parts_of(mind, u)
        went_on = False
        for t in _world_pos(mind, u)[0]:
            longer = S.tuple_index.get(parts + (t,)) if len(parts) < F else None
            if sense[channels[t]]:                      # the turn ended here: was it answered?
                went_on = True
                if longer is not None and any(act[channels[w]] for w in _world_pos(mind, longer)[0]):
                    asked = True
                else:
                    told = True
            elif longer is not None and not kinds[longer] & SLOTTED:
                went_on = True
                stack.append(longer)
            elif len(parts) >= F:
                told = True                             # too long to follow
        if not went_on:
            told = True
    memo[key] = asked and not told
    return memo[key]


def unsaid_share(mind, y: int, trust: Optional[torch.Tensor] = None) -> float:
    """F41 (the urge to go on): the unsaid share of what is known of ``y``, in [0, 1]: of the evidence of how the
    lived sequences that began with y went on (on its channel, not what was only asked of it), the share of what has
    not followed it in the present episode."""
    channels = mv_view(mind, "u_channel")
    said = episode_said(mind, y)
    total = unsaid = 0.0
    for v, (p, _) in start_mass(mind, y, trust).items():
        if channels[v] != channels[y] or only_asked(mind, y, v):
            continue
        total += p
        if v not in said:
            unsaid += p
    return unsaid / total if total > 0.0 else 0.0


def lived_sequence(mind, parts, j: int) -> Tuple[int, int]:
    """F41: the longest lived sequence that begins at place ``j`` of ``parts``: (its length, its unit); the item
    itself, (1, parts[j]), when no sequence that began with it went on so (a sequence lived from its beginning left a
    unit for each of its beginnings)."""
    S, kinds, channels = mind.S, mv_view(mind, "u_kind"), mv_view(mind, "u_channel")
    found = (1, parts[j])
    for length in range(2, len(parts) - j + 1):
        u = S.tuple_index.get(parts[j:j + length])
        if u is None or kinds[u] & SLOTTED or channels[parts[j + length - 1]] != channels[parts[j]]:
            break                                      # (a sequence of the item's own channel: it does not reach
        found = (length, u)                            # across a perceived event, the end of a turn)
    return found


def sequence_ended(mind, sequence: int, channel: int, trust: Optional[torch.Tensor] = None) -> bool:
    """F41: a lived sequence that was never followed by a word of its channel and was followed by something else
    (its turn ended there)."""
    channels = mv_view(mind, "u_channel")
    targets, _ = _world_pos(mind, sequence, trust)
    return bool(targets) and not any(channels[v] == channel for v in targets)


def topic_state(mind, unit: int, trust: Optional[torch.Tensor] = None):
    """F41: where the present answer stands with what is known of its topic: None when there is no topic in the
    context, else (y, place j of its last mention, state, the items after its known sequence), state one of
    "start" (y is the last item), "inside" (the lived sequence from y goes on with a word), "ended" (it ended)."""
    topic = getattr(mind, "topic", None)
    kinds = mv_view(mind, "u_kind")
    if topic is None or topic[2] != getattr(mind, "episode_start", None) or not kinds[unit] & COMPOSED:
        return None
    y = topic[0]
    parts = parts_of(mind, unit)
    if y not in parts:
        return None
    j = len(parts) - 1 - parts[::-1].index(y)
    length, sequence = lived_sequence(mind, parts, j)
    extra = parts[j + length:]
    if length == 1:
        return (y, j, "start", extra) if not extra else None
    if sequence_ended(mind, sequence, mv_view(mind, "u_channel")[y], trust):
        return (y, j, "ended", extra)
    return (y, j, "inside", extra) if not extra else None


def told_so_far(mind, parts, x: int, channel: int, trust: Optional[torch.Tensor] = None) -> tuple:
    """F41: a context as an answer that tells what is known of ``x``, its places from the last backwards. What was
    said before the last perceived event (the other's turn: the question) is itself, x there a place of its own; in
    the turn that follows, every lived sequence of x (x and the words of its own lived sequence that follow it) is one
    place, marked by how it stands ("ended", or "open": cut short or going on), and the other items are
    themselves."""
    channels, sense = mv_view(mind, "u_channel"), kind_masks(mind)[4]
    turn = 0
    for i, p in enumerate(parts):
        if sense[channels[p]]:
            turn = i + 1
    out = [("filler",) if p == x else p for p in parts[:turn]]
    i = turn
    while i < len(parts):
        if parts[i] == x:
            length, sequence = lived_sequence(mind, parts, i)
            ended = length > 1 and sequence_ended(mind, sequence, channel, trust)
            out.append(("known", "ended" if ended else "open"))
            i += length
        else:
            out.append(parts[i])
            i += 1
    return tuple(out[::-1])


def _same_share(a, b) -> float:
    """F41: the share of two runs of places that are the same, place by place (the filler's own place counted: 1 for
    two empty runs)."""
    return (1.0 + sum(1 for p, q in zip(a, b) if p == q)) / (1.0 + max(len(a), len(b)))


def sequence_reading(mind, unit: int, trust: Optional[torch.Tensor] = None) -> Dict[int, float]:
    """F41 (Known Sequences): the answer tells what is known of its topic y (``mind.topic``: the filler of the place
    where such a claim was acted on). The lived contexts of the place's other members are read in which their own
    lived sequence stood as y's does in the present context: just named, going on, or ended with the same items after
    it; the item before the member the same. Each votes what it went on with, the member read as y; a thing known of
    the member, as the thing known of y most like it (``known_claim``). They are weighed as lived situations are
    (exp(-F (1 - s))): s is the share of the places before the filler that are the same when each context is read as
    an answer about its own filler (``told_so_far``: what was told of the member stands where what was told of y
    does), and they are held with lam * s * conviction(members).
    The urge decides between going on and ending: an answer that tells what is known of its topic ends when nothing
    known of it is left unsaid. What a member had left to say cannot be read from a lived context (it keeps only its
    last items), but an answer that ends for having nothing left has gone on at least once for every end. So while
    something known of y is unsaid (urge > 0), as many of the members' ends of turn as their goings-on are taken to be
    ends of that kind and are not counted; the ends beyond that count, as an expected end of the turn (members that
    stopped with things left unsaid: the answer to a question about one thing). When nothing is unsaid (urge 0),
    nothing is claimed."""
    state = topic_state(mind, unit, trust)
    if state is None:
        return {}
    memo = _moment_memo(mind)
    key = ("sequence", unit, id(trust) if trust is not None else None)
    if key in memo:
        return memo[key]
    S, cfg = mind.S, mind.cfg
    y, j, where, extra = state
    wish = unsaid_share(mind, y, trust) if where != "inside" else 1.0
    found: Dict[int, float] = {}
    if wish > 0.0:
        category = mind.topic[1]
        parts = parts_of(mind, unit)
        kinds, channels = mv_view(mind, "u_kind"), mv_view(mind, "u_channel")
        channel = channels[y]
        before = told_so_far(mind, parts[:j], y, channel, trust)
        here = lived_sequence(mind, parts, j)[0]
        src = mv_view(mind, "t_src")
        mates = {src[t] for t in S.inc.get((category, ABSTRACT), ())}
        votes: Dict[int, float] = {}
        ended: Dict[int, float] = {}
        said_known: Dict[Tuple[int, int], float] = {}
        voted = similar = ends = 0.0
        voters, seen, budget = set(), {unit}, 20000
        for x in mates:
            if x == y:
                continue
            for tie in S.inc.get((x, PART), ()):
                budget -= 1
                c = src[tie]
                if budget < 0 or c in seen or kinds[c] & SLOTTED:
                    continue
                seen.add(c)
                t = parts_of(mind, c)
                jx = len(t) - 1 - t[::-1].index(x)
                length, sequence = lived_sequence(mind, t, jx)
                if t[jx + length:] != extra:
                    continue
                if where == "start":
                    if length != 1:
                        continue
                elif where == "inside":                 # as far into a lived sequence of its own that goes on
                    if length != here or sequence_ended(mind, sequence, channel, trust):
                        continue
                elif length < 2 or not sequence_ended(mind, sequence, channel, trust):
                    continue
                held = told_so_far(mind, t[:jx], x, channel, trust)
                if before and held and before[0] != held[0]:
                    continue                            # the place before the filler is the same
                targets, masses_ = _world_pos(mind, c, trust)
                total = sum(masses_)
                if total <= 0.0:
                    continue
                s = _same_share(before, held)
                g = math.exp(-mind.knob["F"] * (1.0 - s))
                voted += g
                similar += g * s
                voters.add(x)
                known_x = known_unsaid(mind, x, c, jx, trust) if where != "ended" else {}
                for v, p in zip(targets, masses_):
                    share = g * p / total
                    if channels[v] != channel:
                        ends += share                   # the turn ended there (or another channel went on)
                        ended[v] = ended.get(v, 0.0) + share
                    elif v in known_x:
                        said = (v, S.tuple_index.get(t[jx:] + (v,), -1))
                        said_known[said] = said_known.get(said, 0.0) + share
                    elif v == x:
                        votes[y] = votes.get(y, 0.0) + share
                    elif v not in mates:
                        votes[v] = votes.get(v, 0.0) + share
        if voted > 0.0:
            spared = min(ends, voted - ends)            # ends for having nothing left: at most as many as goings-on
            counted = voted - spared
            if counted > 1e-12:
                c = cfg.lam * (similar / voted) * conviction(float(len(voters)), cfg.k)
                for v, w in votes.items():
                    found[v] = c * min(1.0, w / counted) + (1.0 - c) * 0.5
                for v, w in ended.items():              # the ends that count: the turn is expected to end here
                    found[v] = c * (w / ends) * (ends - spared) / counted + (1.0 - c) * 0.5
                if said_known:
                    known_here = known_unsaid(mind, y, unit, j, trust, episode=True)
                    target, f = known_claim(mind, known_here, {k: w / counted for k, w in said_known.items()}, {},
                                            trust, parts[j:])
                    if target is not None:
                        found[target] = max(found.get(target, 0.0), c * f + (1.0 - c) * 0.5)
    memo[key] = found
    return found


def goes_alike(mind, a: int, b: int, trust: Optional[torch.Tensor] = None) -> float:
    """F41: how far two items go on alike: the overlap of the shares of what each was lived to be followed by (1 for
    the same item). Words of one kind (what a thing does, what it eats) go on alike; words of different kinds do not."""
    if a == b:
        return 1.0
    memo = _moment_memo(mind)
    key = ("alike", min(a, b), max(a, b), id(trust) if trust is not None else None)
    if key not in memo:
        ta, wa = _world_pos(mind, a, trust)
        tb, wb = _world_pos(mind, b, trust)
        sa, sb = sum(wa), sum(wb)
        if sa <= 0.0 or sb <= 0.0:
            memo[key] = 0.0
        else:
            share_b = {v: w / sb for v, w in zip(tb, wb)}
            memo[key] = sum(min(w / sa, share_b.get(v, 0.0)) for v, w in zip(ta, wa))
    return memo[key]


def known_claim(mind, known_here: Dict[int, float], said_known, mass, trust=None, sequence=()):
    """F41: (value, f) of the frame's claim "what is known of the filler comes next" for the context being read.
    The members went on, here, with things known of them (``said_known``: the share of the frame's votes of each
    word, with the member's lived sequence it carried on); of the things known of this filler, the one most like what
    they said is claimed (among equals, the more known, then the one the members said most, then the one learned
    first), with
        f = sum_t said(t) * max_t alike(v, t):
    how far the members went on with things known of them, times how like the claimed thing is to something they
    said. A filler with one thing left to say says it when the members ever said its like here, however many other
    kinds they also said (they had those left to say; it has not). Two words are alike as far as they go on alike, as
    words anywhere or as the lived sequences they carry on (``goes_alike``; the same word is itself): what a thing
    does is like what another does, not like what it eats."""
    if not known_here or not said_known:
        return None, 0.0
    carried = {v: mind.S.tuple_index.get(tuple(sequence) + (v,), -1) for v in known_here}

    def alike(v, said):
        t, lived = said
        if v == t:
            return 1.0
        words = goes_alike(mind, v, t, trust)
        if words < 1.0 and lived >= 0 and carried[v] >= 0:
            words = max(words, goes_alike(mind, carried[v], lived, trust))
        return words
    fit = {v: sum(share * alike(v, said) for said, share in said_known.items()) for v in known_here}

    def rank(v):
        return (round(fit[v], 9), round(known_here[v], 9), frequency(*mass[v]) if v in mass else 0.0, -v)
    best = max(known_here, key=rank)
    return best, min(1.0, sum(said_known.values())) * max(alike(best, said) for said in said_known)


def _lexicon(mind, cid: int, since: int) -> list:
    """F36/F37: the words lived on channel ``cid`` before ``since``: (text, unit, use), kept while neither changes."""
    lexicon = mind.__dict__.get("_lexicon")
    size = len(mind.ch.symbols[cid])
    if lexicon is None or lexicon[0] != (cid, size, since):
        S = mind.S
        words = []
        for index, sym in enumerate(mind.ch.symbols[cid]):
            u = S.atom_index.get((cid, index), -1)
            text = str(sym)
            if u >= 0 and text.isalpha() and int(S.u_birth[u]) < since and float(S.u_use[u]) > 0.0:
                words.append((text, u, float(S.u_use[u])))
        lexicon = mind.__dict__["_lexicon"] = ((cid, size, since), words)
    return lexicon[1]


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
        words = _lexicon(mind, channel, since)
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


def replacement_reading(mind, unit: int, trust: Optional[torch.Tensor] = None) -> Dict[int, float]:
    """F39 (Replacement Witness, Mill's method of difference): when nothing would be said (no world target over the
    speaking line), the lived situations that differ from the present one by one replacement (a run of items A of the
    present context, B in the situation, the same length and everything else the same, the perceived end of the turn
    included) are witnesses of it, as far as the replacement made no difference where it was lived: the lived pairs
    of contexts that differ by the same replacement at the same place went on alike (the overlap of their lived
    continuations; f2 = the agreeing share, c2 = the conviction of their number). "güneşli … iyi eğlenceler ⏎"
    witnesses "güneşli … şemsiye al ⏎" when the two advices went on alike on cloudy days; "bulutlu … şemsiye al ⏎"
    does not, for cloudy and sunny days with the same advice went on otherwise.
    What a witness says is carried as F38's third form carries it: its own frequency, its reliability scaling only how
    firmly it holds (f = f1, c = f1 f2 c1 c2), so it says nothing of a value it never went on with and counts against
    nothing; witnesses add up (revision). {value: e} over 1/2 only. The present context is never its own evidence."""
    S, cfg, k = mind.S, mind.cfg, mind.cfg.k
    if not mv_view(mind, "u_kind")[unit] & COMPOSED:
        return {}
    memo = _moment_memo(mind)
    key = ("difference", unit)
    if key in memo:
        return memo[key]
    heard = parts_of(mind, unit)
    n = len(heard)
    kinds, src, lag = mv_view(mind, "u_kind"), mv_view(mind, "t_src"), mv_view(mind, "t_lag")
    witnesses: Dict[int, tuple] = {}
    budget = 20000
    for j in sorted(range(n), key=lambda i: len(S.inc.get((heard[i], PART), ()))):   # the rarer items first
        for tie in S.inc.get((heard[j], PART), ()):
            budget -= 1
            if budget < 0:
                break
            u = src[tie]
            if u == unit or u in witnesses or kinds[u] & SLOTTED or lag[tie] != j:
                continue
            t = parts_of(mind, u)
            if len(t) != n or t[-1] != heard[-1]:
                continue
            diff = [i for i in range(n) if t[i] != heard[i]]
            if not diff or max(diff) - min(diff) + 1 != len(diff) or len(diff) > n - 2:
                continue
            witnesses[u] = (min(diff), max(diff) + 1)
        if budget < 0:
            break
    evidence: Dict[int, List[float]] = {}
    for u, (lo, hi) in witnesses.items():
        targets, own = _world_pos(mind, u, trust)
        total = sum(own)
        if total <= 0.0:
            continue
        agree, pairs = _replacement_pairs(mind, heard[lo:hi], parts_of(mind, u)[lo:hi], lo, n, unit, u, trust)
        if agree <= 0.0:
            continue
        f2, c2, c1 = agree / pairs, conviction(pairs, k), conviction(total, k)
        for v, p in zip(targets, own):
            f1 = p / total
            c = min(f1 * f2 * c1 * c2, 1.0 - 1e-9)
            w = k * c / (1.0 - c)
            rec = evidence.setdefault(v, [0.0, 0.0])
            rec[0] += f1 * w
            rec[1] += (1.0 - f1) * w
    found = {v: expectation(p, q, k) for v, (p, q) in evidence.items()}
    found = {v: e for v, e in found.items() if e > 0.5}
    memo[key] = found
    return found


def _replacement_pairs(mind, A: tuple, B: tuple, lo: int, n: int, unit: int, witness: int, trust=None):
    """F39: (W+, W) of "replacing A by B at this place makes no difference": the lived contexts of length ``n`` that
    hold A at ``lo`` and whose copy with B there was lived too; each pair agrees by the overlap of the two lived
    continuations. Neither the present context nor the witness is counted."""
    memo = _moment_memo(mind)
    key = ("replacement", A, B, lo, n, unit, witness)
    if key in memo:
        return memo[key]
    S = mind.S
    kinds, src, lag = mv_view(mind, "u_kind"), mv_view(mind, "t_src"), mv_view(mind, "t_lag")
    agree = pairs = 0.0
    budget = 20000
    for tie in S.inc.get((A[0], PART), ()):
        budget -= 1
        if budget < 0:
            break
        c1 = src[tie]
        if lag[tie] != lo or c1 == unit or c1 == witness or kinds[c1] & SLOTTED:
            continue
        p1 = parts_of(mind, c1)
        if len(p1) != n or p1[lo:lo + len(A)] != A:
            continue
        c2 = S.tuple_index.get(p1[:lo] + B + p1[lo + len(A):])
        if c2 is None or c2 == unit or c2 == witness or kinds[c2] & SLOTTED:
            continue
        t1, w1 = _world_pos(mind, c1, trust)
        t2, w2 = _world_pos(mind, c2, trust)
        s1, s2 = sum(w1), sum(w2)
        if s1 <= 0.0 or s2 <= 0.0:
            continue
        share2 = {v: w / s2 for v, w in zip(t2, w2)}
        pairs += 1.0
        agree += sum(min(w / s1, share2.get(v, 0.0)) for v, w in zip(t1, w1))
    found = (agree, pairs)
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
    lived suffixes of the present context that reach every fixed item of the frame: the claim's own context, whatever
    filled the frame's place at the front, or nothing. A suffix that leaves out a fixed word is another context ("walk
    twice ⏎" of "jump left and walk twice ⏎": what follows it is the answer to another command); F38 reads such a
    suffix as a witness (``witnessed``)."""
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
        if focus:                                        # a suffix that reaches every fixed item of the claim
            for L in range(n - 1, max(max(focus) + 1, 2) - 1, -1):
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


def witness_reliability(mind, q: int, t: tuple, suffix: int, unit: int, trust=None) -> Tuple[float, float]:
    """F38 (Witness Reliability): the evidence (W+, W) that a lived suffix ``t`` of the present context, which leaves
    out a fixed item of a frame's claim (``q`` is the item just before it), goes on as the longer context does. The
    evidence is the lived contexts that end with (q, t): each agrees as far as it went on as t went on, by the
    overlap of the two lived continuations, sum over v of min(P_u(v), P_t(v)); W is their number. None lived: (0, 0),
    the suffix is another context (F34's rule). It is a source's reliability (the trust of a provenance, F24's
    voices) for the part of a context a reading leaves out, measured on the mind's own lived contexts, never on the
    claim being read (the present context is not among them)."""
    memo = _moment_memo(mind)
    key = ("witness", q, suffix, unit)
    if key in memo:
        return memo[key]
    S, k = mind.S, mind.cfg.k
    kinds = mv_view(mind, "u_kind")
    targets, own = _world_pos(mind, suffix, trust)
    total = sum(own)
    agree = pairs = 0.0
    if total > 0.0:
        p_t = {v: p / total for v, p in zip(targets, own)}
        L = len(t)
        src, lag = mv_view(mind, "t_src"), mv_view(mind, "t_lag")
        by_q, by_t = S.inc.get((q, PART), ()), S.inc.get((t[0], PART), ())
        through_q = len(by_q) <= len(by_t)               # call the contexts up through the rarer of the two items
        budget = 20000
        for tie in (by_q if through_q else by_t):
            budget -= 1
            if budget < 0:
                break
            u = src[tie]
            if u == unit or u == suffix or kinds[u] & SLOTTED:
                continue
            parts = parts_of(mind, u)
            m = len(parts)
            if m < L + 1 or lag[tie] != (m - L - 1 if through_q else m - L) or parts[m - L - 1] != q \
                    or parts[m - L:] != t:
                continue
            u_targets, u_own = _world_pos(mind, u, trust)
            u_total = sum(u_own)
            if u_total <= 0.0:
                continue
            pairs += 1.0
            agree += sum(min(p / u_total, p_t.get(v, 0.0)) for v, p in zip(u_targets, u_own))
    found = (agree, pairs)
    memo[key] = found
    return found


def witnessed(mind, unit: int, trust, counted, focus) -> Dict[int, List[Tuple[float, float]]]:
    """F38: what the lived suffixes of ``unit`` that leave out a fixed item of a frame's claim witness of it, value by
    value: {value: [(W+, W-) of each witness]}. By the deduction of NARS, from "t goes on with v" (f1 = v's share of
    t's lived continuations, c1 = their conviction) and "the longer context goes on as t does" (f2 = the agreeing
    share of the lived contexts that end with the left-out item and t, c2 = the conviction of their number): the
    witness keeps its own frequency and its reliability says only how firmly it holds (as F29 and F36 carry a
    reading), f = f1, c = f1 f2 c1 c2, w = k c / (1 - c). A witness says nothing of a value it never went on with
    (f1 = 0 gives c = 0) and cannot count against its own value: it cannot silence a claim. (The deduction's own
    frequency, f = f1 f2, let an unreliable witness count against its own value; lived with, that cost SCAN 22.) The
    suffix must hold a word of the utterance and the perceived event; the frame's member instances (``counted``) are
    not witnesses (no lived context counts twice)."""
    S, k = mind.S, mind.cfg.k
    heard = parts_of(mind, unit)
    n = len(heard)
    channels, symbols = mv_view(mind, "u_channel"), mv_view(mind, "u_symbol")
    sense, act = kind_masks(mind)[4:6]
    event = -1
    for j, p in enumerate(heard):
        if sense[channels[p]]:
            event = j
    if event < 1:
        return {}
    d_event = n - 1 - event
    reach = max(focus) + 1
    kinds = mv_view(mind, "u_kind")
    out: Dict[int, List[Tuple[float, float]]] = {}
    for L in range(min(reach, n) - 1, d_event + 1, -1):
        suffix = S.tuple_index.get(heard[n - L:])
        if suffix is None or suffix in counted or kinds[suffix] & SLOTTED:
            continue
        if not any(act[channels[p]] and str(mind.ch.symbol(channels[p], symbols[p])).isalpha()
                   for p in heard[n - L:event]):
            continue
        agree, pairs = witness_reliability(mind, heard[n - L - 1], heard[n - L:], suffix, unit, trust)
        if agree <= 0.0:
            continue
        targets, own = _world_pos(mind, suffix, trust)
        total = sum(own)
        if total <= 0.0:
            continue
        f2, c2, c1 = agree / pairs, conviction(pairs, k), conviction(total, k)
        for v, p in zip(targets, own):
            f1 = p / total
            f = f1                                       # the witness's own frequency; its reliability only says
            c = min(f1 * f2 * c1 * c2, 1.0 - 1e-9)       # how firmly it holds (F29, F36)
            w = k * c / (1.0 - c)
            out.setdefault(v, []).append((f * w, (1.0 - f) * w))
    return out


def continuation_backoff(mind, unit: int, trust: Optional[torch.Tensor] = None) -> Dict[int, float]:
    """F16: starting needs a reason, continuing follows the words. When nothing in the whole present situation
    expects a next symbol, the end of the sequence that is going on now (the items after the last perceived event,
    e.g. the mind's own utterance after its partner's ended) is read instead: its longest suffix that has lived
    continuations speaks, with its own evidence scaled by lam. It never reaches back across a perceived event, so
    it cannot start an utterance, only carry one on.
    F26 (``starts``): a composed suffix of fewer than F items is a sequence that began with those items (only an
    episode's beginning makes such a unit); an utterance of one item so far is read the same way, as how sequences
    that began with it went on (``start_mass``), before what followed that item anywhere.
    F52 (``echo_exact``): the rule "what the partner just said is an echo" in its exact form. Without it no word of
    the partner's utterance is read as a continuation, wherever it stood there. With it a continuation is an echo
    only where it followed these very words in what is present (the same rule F42 holds for one item over the
    whole episode): a sentence the mind read elsewhere is carried on although the question holds one of its
    words."""
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
    exact = getattr(cfg, "echo_exact", False)
    heard = set() if exact else set(parts[:start])      # what the partner just said is an echo, not knowledge
    first = 1 if len(tail) == len(parts) else 0         # the whole tuple is the unit itself, read already
    starts = getattr(cfg, "starts", False)
    for i in range(first, len(tail)):
        suffix = tail[i:]
        if not suffix:
            continue
        if exact:                                       # F52: ... where it followed these very words (not any word
            n = len(suffix)                             # the partner said, wherever it stood)
            heard = {parts[j + n] for j in range(len(parts) - n) if parts[j:j + n] == suffix}
        if len(suffix) == 1:
            readings = [own_mass(mind, suffix[0], 1, trust)]
            if starts and i == 0:                       # the whole utterance so far is this one item
                readings.insert(0, start_mass(mind, suffix[0], trust))
            if (getattr(cfg, "echo", False) or getattr(cfg, "known", False)) and unit == mind.context:
                heard = heard | episode_said(mind, suffix[0])   # F42: what followed the item earlier in this episode
                                                                # is an echo too
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
