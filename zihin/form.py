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

from typing import Optional, Tuple

import torch

from .evidence import expectation
from .learn import add_evidence, add_evidence1, tie_mass
from .store import ABSTRACT, ATOMIC, CATEGORY, COMPOSED, EXPECT, F64, I64, INDUCED, PART, SLOTTED, W_POS


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


def induce_slots(mind, unit: int, parts: Tuple[int, ...], slot: int) -> None:
    """Frames and categories from composed units that differ in exactly one position."""
    S, g = mind.S, mind.knob["gain"]
    for key, where in S.slot_keys(parts):
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
            earlier = S.parts(first)[pos]
            add_evidence1(mind, W_POS, slot, S.tie(earlier, category, ABSTRACT, 0), g)
        else:
            category = frame - 1
        add_evidence1(mind, W_POS, slot, S.tie(parts[pos], category, ABSTRACT, 0), g)


def categories_of(mind, unit: int):
    """(category, expectation of the ABSTRACT tie) for every category the unit is an instance of."""
    S = mind.S
    if not S.out.get((unit, ABSTRACT, 0)):
        return []
    ids, dst = S.out_ids((unit, ABSTRACT, 0))
    w_pos, w_neg = tie_mass(mind, ids)
    e = expectation(w_pos, w_neg, mind.cfg.k)
    return list(zip(dst.tolist(), e.tolist()))


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
