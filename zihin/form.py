# -*- coding: utf-8 -*-
"""Concept formation of Zihin (design section 3): recruitment, slots and frames, chunking.

* Recruitment. The focus holds the items of the current episode in order (an episode is a run of moments with
  outside events; a silent moment ends it). The tuple of these items is matched against the composed units;
  an unmatched tuple becomes a new composed unit with PART ties to its members. Every recognition adds
  evidence to the PART ties, so a composed unit that is met again does not depend on the moment that made it.
* Slots. Two composed units of equal length that differ in exactly one position give a category unit for
  that position and a frame (a composed unit whose part there is the category); the fillers get ABSTRACT ties
  to the category, with evidence added every time a filler is met in that place.
* Typed frames. A category that is active because one of the episode's items is its instance stands in for
  that item: the tuple with the category in place of the item is recognised, or recruited when it wins a
  place in the focus (see ``Mind._settle``). This is how an existing category becomes a context of its own.
* Chunking. Two successive items whose EXPECT(lag 1) tie is reliable (expectation >= e_chunk) and used
  (use >= m_chunk) are fused into one composed unit that then counts as a single item of the episode.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

from .evidence import expectation
from .learn import add_evidence, add_evidence1, tie_mass
from .store import ABSTRACT, ATOMIC, CATEGORY, COMPOSED, EXPECT, F64, I64, INDUCED, PART, SLOTTED, W_NEG, W_POS, WIDE

# ---------------------------------------------------------------------- reading helpers (speed only, exact)
# Shared by form, read, act and mind. They give bit for bit what the assembled store rows give, with fewer and
# smaller operations: elementwise IEEE operations are the same in numpy, Python and torch; every sum of more than
# one nonzero term is still done by torch on the same values in the same order. Their caches live on the Mind and
# are never saved (Mind.__getstate__ drops them); each is checked against the object it was taken from.
READ_CACHES = ("_np_views", "_mv_views", "_trust_luts", "_kind_masks", "_parts_cache", "_read_memo")
SMALL = 24          # up to this many ties a reading is done in plain numbers (fewer library calls), else in numpy


def np_view(mind, name: str) -> np.ndarray:
    """numpy view (same memory) of the store tensor ``name``; taken again when growth replaces the tensor."""
    t = getattr(mind.S, name)
    cache = mind.__dict__.get("_np_views")
    if cache is None:
        cache = mind.__dict__["_np_views"] = {}
    hit = cache.get(name)
    if hit is not None and hit[0] is t:
        return hit[1]
    a = t.numpy()
    cache[name] = (t, a)
    return a


def mv_view(mind, name: str) -> memoryview:
    """memoryview (same memory) of the store tensor ``name``: one element reads as a Python int or float."""
    t = getattr(mind.S, name)
    cache = mind.__dict__.get("_mv_views")
    if cache is None:
        cache = mind.__dict__["_mv_views"] = {}
    hit = cache.get(name)
    if hit is not None and hit[0] is t:
        return hit[1]
    m = memoryview(t.numpy())
    cache[name] = (t, m)
    return m


def trust_entry(mind, trust: torch.Tensor):
    """(key, lut, values) of a trust vector: its bytes (a key for memos), lut[1 + p] = trust[p] and 0 elsewhere (no
    class, a full row), and its values as floats; lut and values are None when a trust is not finite (then a product
    with a zero is not zero and the fast reading would not be exact). Kept per tensor while it is not changed in
    place (its version counter)."""
    cache = mind.__dict__.get("_trust_luts")
    if cache is None or len(cache) > 64:
        cache = mind.__dict__["_trust_luts"] = {}
    hit = cache.get(id(trust))
    if hit is not None and hit[0] is trust and hit[1] == trust._version:
        return hit[2]
    values = trust.tolist()
    entry = (trust.numpy().tobytes(), None, None)
    if all(math.isfinite(v) for v in values):
        lut = np.zeros(128, dtype=np.float64)
        lut[1:1 + len(values)] = values
        entry = (entry[0], lut, values)
    cache[id(trust)] = (trust, trust._version, entry)
    return entry


def masses(mind, ids, trust: Optional[torch.Tensor] = None, total: bool = False, table: Optional[int] = None):
    """Trust-weighted masses of the ties ``ids`` (int64 tensor or array), bit for bit what the assembled rows give:
    [2, n] numpy with (S.w2(ids)[i] * trust).sum(-1); with ``table`` only that table's [n]; with ``total`` [n] with
    ((W+ rows + W- rows) * trust).sum(-1). A tie of one class holds one nonzero term, so its six-term sum is that
    term (+ 0.0 makes a -0.0 the +0.0 the sum gives); a tie with a full row is summed by torch as before."""
    S = mind.S
    trust = mind.trust if trust is None else trust
    idx = ids.numpy() if isinstance(ids, torch.Tensor) else ids
    _, lut, _ = trust_entry(mind, trust)
    if lut is None:                                        # not finite: the assembled rows themselves
        pos, neg = S.w2(torch.from_numpy(np.ascontiguousarray(idx, dtype=np.int64)))
        if total:
            return ((pos + neg) * trust).sum(-1).numpy()
        if table is not None:
            return ((pos if table == W_POS else neg) * trust).sum(-1).numpy()
        return torch.stack(((pos * trust).sum(-1), (neg * trust).sum(-1))).numpy()
    cls = np_view(mind, "t_cls")[idx]
    tr = lut[cls]
    ev = np_view(mind, "t_ev")
    if total:
        w = ev[W_POS, idx] + ev[W_NEG, idx]
        w *= tr
    elif table is not None:
        w = ev[table, idx] * tr
    else:
        w = ev[:, idx] * tr
    w += 0.0
    if S.n_rows > 1 and cls.size and cls.max() == WIDE:
        wide = np.flatnonzero(cls == WIDE)
        r = torch.from_numpy(np_view(mind, "t_row")[idx[wide]].astype(np.int64))
        if table is not None and not total:
            w[wide] = (S.ex[table].index_select(0, r) * trust).sum(-1).numpy()
        else:
            rows = S.ex[W_POS:W_NEG + 1].index_select(1, r)
            if total:
                w[wide] = ((rows[0] + rows[1]) * trust).sum(-1).numpy()
            else:
                w[:, wide] = (rows * trust).sum(-1).numpy()
    return w


def masses_py(mind, ties, trust: Optional[torch.Tensor] = None):
    """(W+, W-) lists of a few ties (ids as ints) in plain numbers: the values of ``masses``."""
    trust = mind.trust if trust is None else trust
    _, lut, values = trust_entry(mind, trust)
    if lut is None:
        w = masses(mind, np.array(ties, dtype=np.int64), trust)
        return w[0].tolist(), w[1].tolist()
    cls, ev = mv_view(mind, "t_cls"), mv_view(mind, "t_ev")
    w_pos, w_neg = [], []
    for t in ties:
        c = cls[t]
        if c == 0:
            w_pos.append(0.0)
            w_neg.append(0.0)
        elif c == WIDE:
            a, b = mass1(mind, t, trust)
            w_pos.append(a)
            w_neg.append(b)
        else:
            v = values[c - 1]
            w_pos.append(ev[W_POS, t] * v + 0.0)
            w_neg.append(ev[W_NEG, t] * v + 0.0)
    return w_pos, w_neg


def mass1(mind, tie: int, trust: Optional[torch.Tensor] = None) -> Tuple[float, float]:
    """(W+, W-) of one tie: float((S.w1(W_POS, tie) * trust).sum()) and the same for W-, bit for bit."""
    trust = mind.trust if trust is None else trust
    _, lut, values = trust_entry(mind, trust)
    c = mv_view(mind, "t_cls")[tie]
    if c == WIDE or lut is None:
        S = mind.S
        return float((S.w1(W_POS, tie) * trust).sum()), float((S.w1(W_NEG, tie) * trust).sum())
    if c == 0:
        return 0.0, 0.0
    ev, v = mv_view(mind, "t_ev"), values[c - 1]
    return ev[W_POS, tie] * v + 0.0, ev[W_NEG, tie] * v + 0.0


def mass1_total(mind, tie: int, trust: Optional[torch.Tensor] = None) -> float:
    """float(((S.w1(W_POS, tie) + S.w1(W_NEG, tie)) * trust).sum()), bit for bit."""
    trust = mind.trust if trust is None else trust
    _, lut, values = trust_entry(mind, trust)
    c = mv_view(mind, "t_cls")[tie]
    if c == WIDE or lut is None:
        S = mind.S
        return float(((S.w1(W_POS, tie) + S.w1(W_NEG, tie)) * trust).sum())
    if c == 0:
        return 0.0
    ev = mv_view(mind, "t_ev")
    return (ev[W_POS, tie] + ev[W_NEG, tie]) * values[c - 1] + 0.0


def masses_list(mind, ties, trust: Optional[torch.Tensor] = None):
    """[(W+, W-)] of a few ties (ids as ints) as floats: the values of ``masses``, in plain numbers."""
    w_pos, w_neg = masses_py(mind, ties, trust)
    return list(zip(w_pos, w_neg))


def expectations(w_pos: np.ndarray, w_neg: np.ndarray, k: float) -> np.ndarray:
    """evidence.expectation on arrays: the same operations, element by element."""
    total = w_pos + w_neg
    some = total > 0
    return total / (total + k) * (np.where(some, w_pos / np.where(some, total, 1.0), 0.5) - 0.5) + 0.5


def kind_masks(mind):
    """By channel id, with one more entry (False) that index -1 (no channel) reads: (world = sense or act, sense,
    act) as numpy masks, then the same three as tuples (for one unit at a time). Kept while no channel is added
    (channels are never removed or changed)."""
    kinds = mind.ch.kinds
    hit = mind.__dict__.get("_kind_masks")
    if hit is not None and hit[0] is kinds and hit[1] == len(kinds):
        return hit[2]
    lists = ([k in ("sense", "act") for k in kinds] + [False], [k == "sense" for k in kinds] + [False],
             [k == "act" for k in kinds] + [False])
    masks = tuple(np.array(x, dtype=bool) for x in lists) + tuple(tuple(x) for x in lists)
    mind.__dict__["_kind_masks"] = (kinds, len(kinds), masks)
    return masks


def parts_of(mind, unit: int) -> Tuple[int, ...]:
    """S.parts(unit), kept per unit while its PART list is the same list of the same length: a unit's parts never
    change while it lives, and undoing it or rebuilding the indices replaces the list."""
    S = mind.S
    ties = S.out.get((unit, PART, -1))
    if not ties:
        return ()
    cache = mind.__dict__.get("_parts_cache")
    if cache is None or len(cache) > 200000:
        cache = mind.__dict__["_parts_cache"] = {}
    hit = cache.get(unit)
    if hit is not None and hit[0] is ties and hit[1] == len(ties):
        return hit[2]
    lag, dst = mv_view(mind, "t_lag"), mv_view(mind, "t_dst")   # S.parts with the same keys and the same sort
    parts = tuple(dst[t] for t in sorted(ties, key=lambda t: lag[t]))
    cache[unit] = (ties, len(ties), parts)
    return parts


def compose(mind, parts: Tuple[int, ...], extra_kind: int = 0) -> Tuple[int, bool]:
    """The composed unit with these parts; recruited (with PART ties) when no unit matches."""
    S = mind.S
    unit = S.tuple_index.get(parts)
    if unit is not None:
        return unit, False
    unit = S.new_unit(COMPOSED | extra_kind, arity=len(parts))
    S.tuple_index[parts] = unit
    for position, part in enumerate(parts):
        S.tie(unit, part, PART, position)
    return unit, True


def confirm_parts(mind, unit: int, slot: int) -> None:
    """One more encounter of a composed unit: evidence on its PART ties."""
    if not mind.S.out.get((unit, PART, -1)):
        return
    ids, _ = mind.S.out_ids((unit, PART, -1))
    add_evidence(mind, W_POS, slot, ids, torch.full((ids.numel(),), mind.knob["gain"], dtype=F64))


def recognise(mind, slot: int) -> Optional[int]:
    """Recognise or recruit the composed unit of the current episode tuple; induce slots from it."""
    parts = tuple(mind.episode)
    if len(parts) < 2:
        return None
    unit, _ = compose(mind, parts)
    confirm_parts(mind, unit, slot)
    if mind.cfg.slots:
        induce_slots(mind, unit, parts, slot)
    return unit


def slot_keys(S, parts: Tuple[int, ...]):
    """``S.slot_keys(parts)``, the same list in the same order, built in one pass over the tuple."""
    if not getattr(S, "bind_repeats", False):
        return [((j, parts[:j] + (-1,) + parts[j + 1:]), (j,)) for j in range(len(parts))]
    positions = {}
    for j, q in enumerate(parts):                          # the values in order of their first position
        positions.setdefault(q, []).append(j)
    out = []
    for q, where in positions.items():
        if len(where) == 1:
            j = where[0]
            out.append(((j, parts[:j] + (-1,) + parts[j + 1:]), (j,)))
        else:
            where = tuple(where)
            out.append(((where, tuple(-1 if x == q else x for x in parts)), where))
    return out


def induce_slots(mind, unit: int, parts: Tuple[int, ...], slot: int) -> None:
    """Frames and categories from composed units that differ in exactly one position."""
    S, g = mind.S, mind.knob["gain"]
    for key, where in slot_keys(S, parts):
        pos = where[0]
        frame = S.frame_of.get(key)
        if frame is None:
            first = S.frame_first.get(key)
            if first is None:
                S.frame_first[key] = unit
                continue
            if first == unit:
                continue
            category = S.new_unit(CATEGORY)
            frame_parts = tuple(category if j in where else q for j, q in enumerate(parts))
            frame = S.new_unit(COMPOSED | SLOTTED | INDUCED, arity=len(parts))  # always category + 1
            S.tuple_index[frame_parts] = frame
            for position, part in enumerate(frame_parts):
                S.tie(frame, part, PART, position)
            S.frame_of[key] = frame
            confirm_parts(mind, frame, slot)
            earlier = parts_of(mind, first)[pos]
            add_evidence1(mind, W_POS, slot, S.tie(earlier, category, ABSTRACT, 0), g)
        else:
            category = frame - 1
        add_evidence1(mind, W_POS, slot, S.tie(parts[pos], category, ABSTRACT, 0), g)


def categories_of(mind, unit: int):
    """(category, expectation of the ABSTRACT tie) for every category the unit is an instance of."""
    S = mind.S
    ties = S.out.get((unit, ABSTRACT, 0))
    if not ties:
        return []
    k = mind.cfg.k
    if len(ties) <= SMALL:                                 # a few ties: the same values in plain numbers
        dst = mv_view(mind, "t_dst")
        w_pos, w_neg = masses_py(mind, ties)
        return [(dst[t], expectation(p, n, k)) for t, p, n in zip(ties, w_pos, w_neg)]
    ids, dst = S.out_ids((unit, ABSTRACT, 0))
    w = masses(mind, ids)
    return list(zip(dst.tolist(), expectations(w[0], w[1], k).tolist()))


def maybe_chunk(mind, slot: int) -> None:
    """Fuse the last two items of the episode into one composed item when their succession is reliable."""
    S, cfg = mind.S, mind.cfg
    if not cfg.chunking or len(mind.episode) < 2:
        return
    first, second = mind.episode[-2], mind.episode[-1]
    tie = S.find_tie(first, second, EXPECT, 1)
    if tie < 0 or float(S.t_use[tie]) < cfg.m_chunk:
        return
    w_pos, w_neg = tie_mass(mind, torch.tensor([tie], dtype=I64))
    if float(expectation(w_pos, w_neg, cfg.k)) < cfg.e_chunk:
        return
    chunk, _ = compose(mind, (first, second))
    confirm_parts(mind, chunk, slot)
    mind.episode[-2:] = [chunk]
    mind.chunked.append(chunk)
