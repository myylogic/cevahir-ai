# -*- coding: utf-8 -*-
"""ELIMINATED. Not part of the mind (it was taken out of ``zihin/`` again); kept here as the record of a hypothesis
that was measured and rejected, so that the measurement can be repeated.

The hypothesis: copies of one mind ("siblings") live different things at the same time, and what each learned is
taken into the one mind by adding, with the units and ties found by their content, the increments each copy's
ledger holds. The adding itself is exact (one sibling absorbed leaves exactly the copy's evidence, tie by tie; the
sums do not depend on the order of the siblings). But a mind that absorbed siblings is not the mind that lived the
same things in turn: while it read, a sibling predicted, missed and generalised with what it knew when it parted,
not with what its siblings were learning meanwhile, and what a moment writes depends on what it expected. Measured
(``t_siblings.py``, ``results/``): no loss where the siblings share a beginning that already carries the structure
of what they read (a generated library, SCAN's add-jump); on natural text recall falls (the owner's documents, 200
sampled sentences, two-word cue: exact 85 -> 75 and valid 148 -> 95 from an empty mind; 87 -> 82 and 146 -> 122
with the first half read before; 90 -> 80 and 146 -> 122 when the siblings part and are absorbed in 8 rounds).
Parting more often makes the loss smaller; it does not make it nothing. The owner's requirement is no loss at all,
so the technique is rejected: a life is lived in turn. What can be done at the same time without any loss is what
writes nothing (``zihin/paralel.py``).

The functions: ``apart`` (a copy starts a ledger of its own), ``learned`` (what it learned since, as plain data),
``absorb`` (the mind takes it in as one moment of its life), ``live_apart`` (the whole with processes).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from zihin.events import IMITATED
from zihin.form import compose, confirm_parts, mass1_total, parts_of, slot_keys
from zihin.learn import add_evidence1
from zihin.store import ABSTRACT, ATOMIC, CATEGORY, COMPOSED, INDUCED, PART, SLOTTED, W_POS, frame_key

_MIND = None        # the mind the children of ``live_apart`` are forked with


def apart(twin) -> None:
    """The copy parts from the mind here: from now it keeps a ledger of its own, in memory."""
    S = twin.S
    S.log, S.keep_log = {}, True
    twin.__dict__["_parted"] = (S.n_units, S.n_ties, twin.t)


def learned(twin) -> Dict[str, object]:
    """What a copy learned since it parted (``apart``), as plain data: the channels, the units it made (an atom: its
    channel and symbol; a composed unit: its kind and parts; a category: with the frame that follows it), the ties it
    made (ends, form, lag) and, for every moment it lived, the increments it wrote."""
    S = twin.S
    U0, T0, t0 = twin.__dict__["_parted"]
    v = S.arrays()
    units: List[Optional[tuple]] = []
    for u in range(U0, S.n_units):
        if not v.ualive[u]:
            units.append(None)
            continue
        kind = int(v.kind[u])
        if kind & ATOMIC:
            channel = int(v.channel[u])
            units.append((kind, twin.ch.names[channel], twin.ch.kinds[channel],
                          twin.ch.symbol(channel, int(v.symbol[u]))))
        elif kind & COMPOSED:
            parts = tuple(int(p) for p in S.parts(u))      # lived as an episode (its slots were induced from) or
            lived = not kind & SLOTTED and any(            # made otherwise (a chunk): only the first is an instance
                key in S.frame_of or key in S.frame_first for key, _ in slot_keys(S, parts))
            units.append((kind, parts, lived))
        else:
            units.append((kind,))
    n = S.n_ties
    ties = (v.src[T0:n].copy(), v.dst[T0:n].copy(), v.form[T0:n].copy(), v.lag[T0:n].copy(), v.alive[T0:n].copy())

    def flat(x) -> np.ndarray:
        return (x.numpy() if isinstance(x, torch.Tensor) else np.atleast_1d(np.asarray(x))).copy()

    moments = []
    for m in sorted(S.log):
        entry = S.log[m]
        if m <= t0 or entry.retracted:
            continue
        moments.append(([(int(table), int(slot), flat(ids).astype(np.int64), flat(amounts).astype(np.float64))
                         for table, slot, ids, amounts in entry.ev],
                        [(flat(ids).astype(np.int64), flat(a).astype(np.float64)) for ids, a in entry.tie_use],
                        [(flat(ids).astype(np.int64), flat(a).astype(np.float64)) for ids, a in entry.unit_use]))
    return {"parted": (U0, T0, t0), "channels": list(zip(twin.ch.names, twin.ch.kinds)), "units": units,
            "ties": ties, "moments": moments, "lived": twin.t - t0}


def absorb(mind, what: Dict[str, object], slot: int = IMITATED) -> Dict[str, object]:
    """Take what a sibling learned into this mind (see the module's text), as one moment of the mind's own life.
    ``slot``: the provenance of what is induced across siblings (what they lived was read: imitated). The mind must
    be the one the sibling parted from, with every unit and tie it had then still in its place. Returns counts."""
    started = time.perf_counter()
    S = mind.S
    U0, T0, _ = what["parted"]
    if S.n_units < U0 or S.n_ties < T0:
        raise ValueError("this mind is not the one the sibling parted from")
    for name, kind in what["channels"]:
        if not mind.ch.has(name):
            mind.add_channel(name, kind)
    units, (src, dst, form, lag, alive) = what["units"], what["ties"]
    company, mind.company = getattr(mind, "company", None), None      # (nothing here is lived with anyone)
    units_before, ties_before = S.n_units, S.n_ties
    S.begin(mind.t + 1, mind.ch.sizes())
    mind.t += 1
    mind._gained = []
    induced = 0
    try:
        new_u = [-1] * len(units)

        def U(u: int) -> int:
            return u if u < U0 else new_u[u - U0]

        instances, to_join = [], []
        i = 0
        while i < len(units):
            rec = units[i]
            if rec is None:
                i += 1
                continue
            kind = rec[0]
            if kind & ATOMIC:
                new_u[i] = mind._atom(mind.ch.id(rec[1]), rec[3])
            elif kind & CATEGORY and not kind & COMPOSED:              # a category and, next, the frame it was made with
                frame = units[i + 1] if i + 1 < len(units) else None
                if frame is None or not frame[0] & INDUCED:
                    i += 1
                    continue
                parts_b = frame[1]
                where = tuple(j for j, p in enumerate(parts_b) if p == U0 + i)
                masked = tuple(-1 if j in where else U(p) for j, p in enumerate(parts_b))
                if where and all(p >= 0 or j in where for j, p in enumerate(masked)):
                    key = frame_key(where if len(where) > 1 else where[0], masked)
                    f = S.frame_of.get(key)
                    if f is None:                                      # (as form.induce_slots makes them)
                        category = S.new_unit(CATEGORY)
                        frame_parts = tuple(category if j in where else p for j, p in enumerate(masked))
                        f = S.new_unit(COMPOSED | SLOTTED | INDUCED, arity=len(frame_parts))
                        S.tuple_index[frame_parts] = f
                        for position, part in enumerate(frame_parts):
                            S.tie(f, part, PART, position)
                        S.frame_of[key] = f
                        first = S.frame_first.get(key)
                        if first is not None:                          # an instance the mind had, waiting for a second
                            to_join.append((first, where[0], f - 1))
                    new_u[i], new_u[i + 1] = f - 1, f
                i += 2
                continue
            elif kind & COMPOSED:
                parts = tuple(U(p) for p in rec[1])
                if all(p >= 0 for p in parts):
                    unit, made = compose(mind, parts, kind & SLOTTED)
                    new_u[i] = unit
                    if rec[2]:
                        instances.append(unit)
            i += 1
        new_t = np.full(len(src), -1, dtype=np.int64)
        for j in np.flatnonzero(alive).tolist():
            a, b = U(int(src[j])), U(int(dst[j]))
            if a >= 0 and b >= 0:
                new_t[j] = S.tie(a, b, int(form[j]), int(lag[j]))
        new_u_arr = np.array(new_u, dtype=np.int64) if new_u else np.zeros(0, dtype=np.int64)

        def mapped(ids: np.ndarray, base: int, table: np.ndarray) -> np.ndarray:
            out = ids.copy()
            late = ids >= base
            if late.any():
                out[late] = table[ids[late] - base]
            return out

        for ev, tie_use, unit_use in what["moments"]:
            for table, cls, ids, amounts in ev:
                ids = mapped(ids, T0, new_t)
                keep = ids >= 0
                if keep.any():
                    S.add(table, cls, torch.from_numpy(ids[keep]), torch.from_numpy(amounts[keep]))
            for ids, amounts in tie_use:
                ids = mapped(ids, T0, new_t)
                for a in np.unique(amounts).tolist():
                    pick = (amounts == a) & (ids >= 0)
                    if pick.any():
                        S.use_ties(torch.from_numpy(ids[pick]), float(a))
            for ids, amounts in unit_use:
                ids = mapped(ids, U0, new_u_arr)
                for a in np.unique(amounts).tolist():
                    pick = (amounts == a) & (ids >= 0)
                    if pick.any():
                        S.use_units(torch.from_numpy(ids[pick]), float(a))
        mind.trust = mind._trust()
        if mind.cfg.slots:                                # what living in turn would have induced across siblings
            g = mind.knob["gain"]

            def join(filler: int, category: int) -> None:
                tie = S.find_tie(filler, category, ABSTRACT, 0)
                if tie < 0 or mass1_total(mind, tie) <= 0.0:   # (a membership the sibling lived is already counted)
                    add_evidence1(mind, W_POS, slot, S.tie(filler, category, ABSTRACT, 0), g, relief=False)

            for first, pos, category in to_join:
                join(parts_of(mind, first)[pos], category)
            for unit in instances:
                parts = parts_of(mind, unit)
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
                        frame = S.new_unit(COMPOSED | SLOTTED | INDUCED, arity=len(parts))
                        S.tuple_index[frame_parts] = frame
                        for position, part in enumerate(frame_parts):
                            S.tie(frame, part, PART, position)
                        S.frame_of[key] = frame
                        confirm_parts(mind, frame, slot)
                        join(parts_of(mind, first)[pos], category)
                        induced += 1
                    join(parts[pos], frame - 1)
    finally:
        S.end()
        mind.company = company
        mind.trust = mind._trust()
        for name in ("_between_memo", "_episode_memo", "_situation_cache"):
            mind.__dict__.pop(name, None)
    return {"moments": len(what["moments"]), "units_made": S.n_units - units_before,
            "ties_made": S.n_ties - ties_before, "frames_induced": induced, "moment": mind.t,
            "seconds": time.perf_counter() - started}


def _child(args):
    job, work = args
    torch.set_num_threads(1)
    mind = _MIND
    apart(mind)
    work(mind, job)
    return learned(mind)


def live_apart(mind, jobs: List[object], work: Callable[[object, object], None],
               workers: Optional[int] = None) -> List[Dict[str, object]]:
    """Siblings in processes: for each job a forked copy of the mind lives ``work(copy, job)`` by itself (``work`` a
    function of a module, so that it can be handed to a process), and the mind absorbs what each learned, in the
    order of the jobs. Returns the reports of ``absorb``. Each sibling parts from the mind as it is now."""
    import multiprocessing
    import os
    global _MIND
    if not jobs:
        return []
    _MIND = mind
    try:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(min(workers or os.cpu_count() or 1, len(jobs)), maxtasksperchild=1) as pool:
            results = pool.map(_child, [(job, work) for job in jobs], chunksize=1)
    finally:
        _MIND = None
    return [absorb(mind, what) for what in results]
