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

Speed, bit for bit. The readings of a moment's learning take the evidence of a tie of one class from its two stored
values (``S.views``) instead of assembling rows of six classes: such a row holds one value v and five zeros, so its
trust-weighted sum is v t[p] + 0.0 and its plain sum v + 0.0 (the + 0.0 turns -0.0 into +0.0, as the six-term sum
does; a trust that is not finite would give 0 x inf = nan in the zeros, and then the rows are assembled as before). A
tie with a full row is summed by the same torch reduction as before. Elementwise steps (expectation, misses) are
the same IEEE operations in the same order, in numpy where the arrays are small; every sum over several values stays
a torch reduction. The place of a known target in a tie list is found by the tie index and bisection (tie ids grow
along a list), checked, else by comparing targets as before. Nothing is computed in another order, so every
stored value, record and relief entry is the same.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import torch

from .events import ACTED, INFERRED, JUDGES, REFERENCE, SENSED
from .evidence import conviction, expectation
from .store import ABSTRACT, ATOMIC, C_NEG, C_POS, CATEGORY, COMPOSED, EXPECT, F64, I64, RELY, SLOTTED, W_NEG, W_POS, WIDE


def add_evidence(mind, table: int, slot: int, ids: torch.Tensor, amounts: torch.Tensor,
                 relief: bool = True, all_expect: bool = False) -> torch.Tensor:
    """Add evidence through the store; keep the relief bookkeeping; let direct experience judge the rest.

    ``relief`` False: the evidence is the echo of the mind's own act and says nothing about the world, so it
    does not count as a decrease of the drive (see NOTES.md, finding on self-made relief).
    ``all_expect``: the caller took ``ids`` from an EXPECT key of ``S.out`` (every one is an expectation).
    """
    n = ids.numel()
    if n == 0:
        return amounts
    if n == 1:                                  # one tie: the same steps in plain numbers
        return torch.tensor([add_evidence1(mind, table, slot, int(ids), float(amounts), relief)], dtype=F64)
    return _add_many(mind, table, slot, ids, amounts, relief, all_expect)


def _add_many(mind, table: int, slot: int, ids: torch.Tensor, amounts: torch.Tensor, relief: bool,
              all_expect: bool) -> torch.Tensor:
    """``add_evidence`` for two or more ties."""
    S = mind.S
    amounts = S.add(table, slot, ids, amounts)  # snapped to the grid, aligned with ids
    if table <= W_NEG and mind.__dict__.get("company") is not None:   # F33: noted with the person present
        from .iliski import note
        note(mind, table, slot, ids.tolist(), amounts.tolist())
    if table in (W_POS, W_NEG):
        if all_expect:
            claims, claimed = ids, amounts
        else:
            expect = S.views()[0][ids.numpy()] == EXPECT
            if not expect.any():
                return amounts
            if expect.all():
                claims, claimed = ids, amounts
            else:
                mask = torch.from_numpy(expect)
                claims, claimed = ids[mask], amounts[mask]
        if relief:
            mind._gained.append((claims, claimed * mind.trust.tolist()[slot]))  # for relief (section 5)
        if slot in JUDGES and mind.cfg.learn_trust:
            judge(mind, table, claims, claimed)  # only expectations are claims about the world
    return amounts


def _add_np(mind, table: int, slot: int, ids: torch.Tensor, idn: np.ndarray, sel: np.ndarray, amounts: np.ndarray,
            relief: bool) -> list:
    """``add_evidence`` (all expectations) for the ties ``idn[sel]`` with ``amounts[sel]``; the snapped amounts as a list.
    ``ids`` is the tensor of ``idn``, passed on whole when every tie is selected."""
    k = sel.size
    if k == 1:
        p = int(sel[0])
        return [add_evidence1(mind, table, slot, int(idn[p]), float(amounts[p]), relief)]
    if k == idn.size:
        part, amt = ids, torch.from_numpy(amounts)
    else:
        part, amt = torch.from_numpy(idn[sel]), torch.from_numpy(amounts[sel])
    return _add_many(mind, table, slot, part, amt, relief, True).tolist()


def add_evidence1(mind, table: int, slot: int, tie: int, amount: float, relief: bool = True) -> float:
    """``add_evidence`` for one tie in plain numbers: the same snapped increment, relief entry and judging, in
    the same order. Returns the snapped amount."""
    S = mind.S
    got = S.add1(table, slot, tie, amount)
    if table <= W_NEG and mind.__dict__.get("company") is not None:   # F33: noted with the person present
        from .iliski import note
        note(mind, table, slot, (tie,), (got,))
    if table <= W_NEG and S.views()[0][tie] == EXPECT:
        if relief:
            mind._gained.append((tie, got * mind.trust.tolist()[slot]))   # one tie: (id, gain), see Mind._feel
        if slot in JUDGES and mind.cfg.learn_trust:
            judge1(mind, table, tie, got)
    return got


def judge1(mind, table: int, tie: int, amount: float) -> None:
    """``judge`` for one tie in plain numbers (a one-element sum is that element).

    A tie of one class holds nothing in the other classes, so with a finite share only its own class can be open:
    when that class is direct experience (or none), nothing is judged and nothing is written."""
    S = mind.S
    share = min(amount, 1.0)                                # torch.clamp(amount, max=1.0)
    v = S.views()
    c = v[1][tie]
    if c != WIDE and math.isfinite(share) and (c == 0 or (c - 1) in REFERENCE or mind.source_tie[c - 1] < 0):
        return
    for slot in range(S.P):
        if slot in REFERENCE or mind.source_tie[slot] < 0:
            continue
        cls, ev, row, ex = v[1:5]                           # taken again after a write: it can widen the tie
        c = cls[tie]
        if c == WIDE:
            r = row[tie]
            w_pos, w_neg, c_pos, c_neg = ex[W_POS, r, slot], ex[W_NEG, r, slot], ex[C_POS, r, slot], ex[C_NEG, r, slot]
        elif c == slot + 1:                                 # its one class: nothing judged yet
            w_pos, w_neg, c_pos, c_neg = ev[W_POS, tie], ev[W_NEG, tie], 0.0, 0.0
        else:
            w_pos = w_neg = c_pos = c_neg = 0.0
        open_pos = (float(w_pos) - float(c_pos)) * share
        open_neg = (float(w_neg) - float(c_neg)) * share
        if open_pos == 0.0 and open_neg == 0.0:
            continue
        open_pos = S.add1(C_POS, slot, tie, open_pos)
        open_neg = S.add1(C_NEG, slot, tie, open_neg)
        agree = open_pos if table == W_POS else open_neg
        differ = open_neg if table == W_POS else open_pos
        S.add1(W_POS, SENSED, mind.source_tie[slot], agree)
        S.add1(W_NEG, SENSED, mind.source_tie[slot], differ)
        if "personal" in mind.__dict__:                 # F33: whose word it was
            from .iliski import attribute
            attribute(mind, slot, ((tie, agree, differ, float(w_pos) + float(w_neg)),))
        v = S.views()


def judge(mind, table: int, ids: torch.Tensor, amounts: torch.Tensor) -> None:
    """Direct experience on ties ``ids`` judges the not yet judged evidence of the learned-trust classes.

    The four tables of the ties are read once: judging a class writes only its own checked column, after reading
    it, and the source ties (which are never judged). A class with nothing open on any of the ties is passed over
    (its open amounts are all zero) unless a share or a held value is not finite.

    The values are read from the store's layout (``S.views``) as ``S.w4`` would assemble them: a tie of one class p
    holds (w+, w-) in class p and zeros elsewhere (nothing judged), a widened tie its full row."""
    S = mind.S
    P = S.P
    judged = _judged_classes(mind)
    if not judged[0] or not ids.numel():
        return
    cls_v, ev_v, row_v, ex_v = S.views()[1:5]
    idn = ids.numpy()
    c = cls_v[idn]
    ev = ev_v.take(idn, axis=1)                             # copies: the values before any write below
    sh = np.minimum(amounts.numpy(), 1.0)                   # torch.clamp(amounts, max=1.0): one full direct
    wide = block = None                                     # experience judges everything outstanding
    if S.n_rows > 1 and c.max() == WIDE:
        wide = np.flatnonzero(c == WIDE)
        block = ex_v.take(row_v[idn[wide]], axis=1)         # [4, k, P]
    finite = bool(np.isfinite(sh).all()) and \
        (bool(np.isfinite(ev).all()) or bool(np.isfinite(ev[:, (c >= 1) & (c <= P)]).all())) and \
        (block is None or bool(np.isfinite(block).all()))
    nz = ev[W_POS] != 0.0
    nz |= ev[W_NEG] != 0.0
    mine = judged[1].take(c) if c.size >= 64 else judged[1][c]
    mine &= nz
    open_cls = set(c[mine].tolist()) if mine.any() else ()
    open_wide = ((block[W_POS] != block[C_POS]) | (block[W_NEG] != block[C_NEG])).any(0).tolist() \
        if block is not None else None
    for slot in judged[0]:
        if finite and (slot + 1) not in open_cls and not (open_wide is not None and open_wide[slot]):
            continue
        mine = c == slot + 1
        h_pos = np.where(mine, ev[W_POS], 0.0)
        h_neg = np.where(mine, ev[W_NEG], 0.0)
        c_pos = np.zeros(idn.size)
        c_neg = np.zeros(idn.size)
        if block is not None:
            h_pos[wide], h_neg[wide] = block[W_POS, :, slot], block[W_NEG, :, slot]
            c_pos[wide], c_neg[wide] = block[C_POS, :, slot], block[C_NEG, :, slot]
        open_pos = torch.from_numpy((h_pos - c_pos) * sh)
        open_neg = torch.from_numpy((h_neg - c_neg) * sh)
        if float(open_pos.sum()) == 0.0 and float(open_neg.sum()) == 0.0:
            continue
        open_pos = S.add(C_POS, slot, ids, open_pos)
        open_neg = S.add(C_NEG, slot, ids, open_neg)
        agree = float(open_pos.sum()) if table == W_POS else float(open_neg.sum())
        differ = float(open_neg.sum()) if table == W_POS else float(open_pos.sum())
        S.add1(W_POS, SENSED, mind.source_tie[slot], agree)
        S.add1(W_NEG, SENSED, mind.source_tie[slot], differ)
        if "personal" in mind.__dict__:                 # F33: whose word it was
            by_tie = mind.personal[0]
            noted_at = [j for j, tie in enumerate(idn.tolist()) if tie in by_tie]
            if noted_at:
                from .iliski import attribute
                a, d = (open_pos, open_neg) if table == W_POS else (open_neg, open_pos)
                attribute(mind, slot, [(int(idn[j]), float(a[j]), float(d[j]), float(h_pos[j] + h_neg[j]))
                                       for j in noted_at])


_JUDGED: Dict[tuple, tuple] = {}


def _judged_classes(mind) -> tuple:
    """The classes whose trust is learned (not direct experience, with a source tie) in class order, and a table by
    class byte that is True for them."""
    key = (mind.S.P, tuple(mind.source_tie))
    hit = _JUDGED.get(key)
    if hit is None:
        slots = [slot for slot in range(mind.S.P) if slot not in REFERENCE and mind.source_tie[slot] >= 0]
        table = np.zeros(128, dtype=bool)
        table[[1 + slot for slot in slots]] = True
        hit = _JUDGED[key] = (slots, table)
    return hit


def rows(S, table: int, ids: torch.Tensor) -> torch.Tensor:
    """The per-class values of one table for the ties ``ids`` ([n, P], the same values as S.t_w[table, ids])."""
    return S.w(table, ids)


_TRUST: list = [None, None, None]   # the trust tensor last seen, its values, its lookup tables (or None)


def _trust_tables(mind):
    """Lookup tables by class byte for the trust of this moment: ``lut[1 + p] = trust[p]`` and ``one[1 + p] = 1``, 0 for
    no class and for WIDE (full rows are summed apart). None when a trust value is not finite (the zeros of a row
    would then give 0 x inf = nan in the six-term sum)."""
    trust = mind.trust
    vals = trust.tolist()
    if _TRUST[0] is trust and _TRUST[1] == vals:
        return _TRUST[2]
    P = mind.S.P
    tables = None
    if len(vals) == P and all(math.isfinite(x) for x in vals):
        lut, one = np.zeros(128), np.zeros(128)
        lut[1:P + 1] = vals
        one[1:P + 1] = 1.0
        tables = (lut, one)
    _TRUST[0], _TRUST[1], _TRUST[2] = trust, vals, tables
    return tables


def _sums(mind, ids: torch.Tensor, raw: bool):
    """For the ties ``ids`` (n >= 1): ``W`` = [W+, W-] trust-weighted ([2, n] numpy) and, with ``raw``, the plain sums
    [w+, w-] ([2, n]), bit for bit ``(rows * trust).sum(-1)`` and ``rows.sum(-1)`` of the rows ``S.w2`` assembles. A tie of
    one class p holds v and five zeros: W = v trust[p] + 0.0, raw = v + 0.0; a full row is summed by the same torch
    reduction as before. None when the fast path does not apply (see ``_trust_tables``)."""
    tables = _trust_tables(mind)
    if tables is None:
        return None
    lut, one = tables
    S = mind.S
    cls_v, ev_v, row_v = S.views()[1:4]
    idn = ids.numpy()
    c = cls_v[idn]
    ev = ev_v.take(idn, axis=1)
    big = c.size >= 64
    w = ev * (lut.take(c) if big else lut[c])
    w += 0.0
    r = None
    if raw:
        r = ev * (one.take(c) if big else one[c])
        r += 0.0
    if S.n_rows > 1 and c.max() == WIDE:
        wide = np.flatnonzero(c == WIDE)
        full = S.ex[W_POS:W_NEG + 1].index_select(1, torch.from_numpy(row_v[idn[wide]]))   # [2, k, P]
        w[:, wide] = (full * mind.trust).sum(-1).numpy()
        if raw:
            r[:, wide] = full.sum(-1).numpy()
    return w, r


def _expect_np(w_pos: np.ndarray, w_neg: np.ndarray, k: float) -> np.ndarray:
    """``evidence.expectation`` step for step (the same IEEE operations): f with 1/2 where nothing was lived, c = W / (W + k).
    Silent like torch where a mass is not finite or W + k is 0 (only with trust set outside [0, 1])."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        total = w_pos + w_neg
        pos = total > 0
        f = np.where(pos, w_pos / np.where(pos, total, 1.0), 0.5)
        return (total / (total + k)) * (f - 0.5) + 0.5


def _raw1(S, table: int, tie: int) -> float:
    """``float(S.w1(table, tie).sum())``: a tie of one class holds one value and zeros, whose sum is that value."""
    cls, ev = S.views()[1:3]
    c = cls[tie]
    if c == WIDE:
        return float(S.w1(table, tie).sum())
    if c == 0 or table > W_NEG:
        return 0.0
    return float(ev[table, tie]) + 0.0


def _places(S, src: int, lag: int, idn: np.ndarray, dstn: np.ndarray, targets) -> list:
    """For each target, the positions in the tie list of (src, EXPECT, lag) (``idn``, its targets ``dstn``) whose target
    it is. The tie (src, v) comes from the tie index and its place from bisection (tie ids grow along a list), checked;
    if a check fails, by comparing the targets (as ``dst == v`` did)."""
    ties = [S.find_tie(src, v, EXPECT, lag) for v in targets]
    found = [t for t in ties if t >= 0]
    if not found:
        return [()] * len(ties)
    at = idn.searchsorted(found).tolist()
    n, out, j = idn.size, [], 0
    for t in ties:
        if t < 0:
            out.append(())
            continue
        p = at[j]
        j += 1
        if p >= n or idn[p] != t:
            return [np.flatnonzero(dstn == v).tolist() for v in targets]
        out.append((p,))
    return out


def tie_mass(mind, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trust-weighted (W+, W-) of ties."""
    sums = _sums(mind, ids, False) if ids.numel() else None
    if sums is None:
        w_pos, w_neg = mind.S.w2(ids)
        return (w_pos * mind.trust).sum(-1), (w_neg * mind.trust).sum(-1)
    w = torch.from_numpy(sums[0])
    return w[0], w[1]


def _present_of(dst: torch.Tensor, occurred: Dict[int, Tuple[float, int]]) -> torch.Tensor:
    """The intensity with which each target occurred now (0 for those that did not)."""
    if dst.numel() <= 64:
        return torch.tensor([occurred[d][0] if d in occurred else 0.0 for d in dst.tolist()], dtype=F64)
    present = torch.zeros(dst.numel(), dtype=F64)
    for unit, (intensity, _) in occurred.items():
        present[dst == unit] = intensity
    return present


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
    # channels on which the mind hears only itself in this moment (the echo of its own act)
    echo = {ch for ch, slot in slot_of_channel.items() if slot == ACTED and mind.ch.kind(ch) in ("act", "knob")}
    changes: Dict[int, Dict[int, list]] = {}  # instance -> target -> raw masses before/after and slots
    was_present = None
    if cfg.next_only_present:               # F13: "comes next" is learned by what WAS the present a moment ago
        was_present = {u for u in (mind.context, mind.episode[-1] if mind.episode else None) if u is not None}
    # With general_miss and without reliability, induce reads only the records of targets that occurred with
    # intensity > 0 (only their positive evidence grows); the other records are not written.
    acquired = None
    if cfg.general_miss and not cfg.reliability:
        acquired = [v for v, (intensity, _) in occurred.items() if intensity > 0.0]
    born_failures = getattr(cfg, "born_failures", False)
    kinds, channels = S.u_kind.numpy(), S.u_channel.numpy()      # no unit is made in this loop
    for unit, a_u, grounded in mind.last_focus:
        if not grounded or a_u <= 0.0 or int(kinds[unit]) & (CATEGORY | SLOTTED):
            continue
        if was_present is not None and unit not in was_present:
            continue
        record: Dict[int, list] = {}
        rivals = None                          # F25: (targets' channels, positive rows) of the ties held before now
        if S.out.get((unit, EXPECT, 1)):
            ids, dst = S.out_ids((unit, EXPECT, 1))
            sums = None if born_failures else _sums(mind, ids, True)
            if sums is None:
                rivals = _follow_rows(mind, unit, a_u, ids, dst, occurred, slot_of_channel, echo, acquired,
                                      born_failures, record)
            else:
                _follow(mind, unit, a_u, ids, dst, sums, occurred, slot_of_channel, echo, acquired, record, channels)
        for v, (intensity, slot) in occurred.items():
            if intensity <= 0.0:
                continue
            born = born_failures and S.find_tie(unit, v, EXPECT, 1) < 0
            tie = S.tie(unit, v, EXPECT, 1)
            if v not in record:
                rp, rn = _raw1(S, W_POS, tie), _raw1(S, W_NEG, tie)
                record[v] = [rp, rp, rn, rn, slot, slot]
            got = add_evidence1(mind, W_POS, slot, tie, g * a_u * intensity, relief=int(channels[v]) not in echo)
            record[v][1] += got
            record[v][4] = slot
            if born and rivals is not None:
                born_failure(mind, tie, v, rivals, record[v])
        changes[unit] = record
    induce(mind, changes, occurred, slot_of_channel)


def _follow(mind, unit: int, a_u: float, ids: torch.Tensor, dst: torch.Tensor, sums, occurred, slot_of_channel,
            echo, acquired, record: Dict[int, list], channels: np.ndarray) -> None:
    """The misses of one unit of the last focus on the channels that reported now, and the records of its targets
    (``learn_follow``), from the fused sums: the same values, ties and order as the rows and masks of
    ``_follow_rows``."""
    S, cfg = mind.S, mind.cfg
    w, r = sums
    e = _expect_np(w[W_POS], w[W_NEG], cfg.k)
    hope = mind.act_hope.get(unit)
    if hope is not None:  # an own act is an expectation of relief (changed rule, see NOTES.md)
        et = torch.from_numpy(e)
        e = torch.where(dst == mind.feel_unit["relief"], torch.clamp(et, min=hope), et).numpy()
    idn, dstn = ids.numpy(), dst.numpy()
    targets = list(occurred)
    places = _places(S, unit, 1, idn, dstn, targets)
    miss = (mind.knob["gain"] * a_u) * (e if cfg.neg_weight == "e" else np.ones(idn.size))
    for v, at in zip(targets, places):                      # x (1 - present); 1 - 0 = 1 leaves the others
        for p in at:
            miss[p] = miss[p] * (1.0 - occurred[v][0])
    expected = (e >= cfg.neg_floor) if cfg.neg_floor > 0.0 else None
    channel = channels[dstn]
    mine = None
    if acquired is not None:
        mine = sorted(p for v, at in zip(targets, places) if v in acquired for p in at)
    reporting = None
    for ch, slot in slot_of_channel.items():
        on = channel == ch
        if expected is not None:
            on &= expected
        sel = np.flatnonzero(on)
        if not sel.size:
            continue
        reporting = on if reporting is None else reporting | on
        got = _add_np(mind, W_NEG, slot, ids, idn, sel, miss, ch not in echo)
        if mine is not None:
            at = [p for p in mine if on[p]]
            if not at:
                continue
            where = sel.tolist()
            pairs = [(p, got[where.index(p)]) for p in at]
        else:
            pairs = zip(sel.tolist(), got)
        for p, m in pairs:
            rp, rn = float(r[W_POS, p]), float(r[W_NEG, p])
            record[int(dstn[p])] = [rp, rp, rn, rn + m, slot, slot]
    if reporting is not None:
        sel = np.flatnonzero(reporting)
        S.use_ties(ids if sel.size == idn.size else torch.from_numpy(idn[sel]))


def _follow_rows(mind, unit: int, a_u: float, ids: torch.Tensor, dst: torch.Tensor, occurred, slot_of_channel, echo,
                 acquired, born_failures: bool, record: Dict[int, list]):
    """``_follow`` from the assembled rows (F25 needs them; also when the trust is not finite). Returns the rivals."""
    S, cfg = mind.S, mind.cfg
    g = mind.knob["gain"]
    if acquired is not None:
        acquired = torch.tensor(acquired, dtype=I64)
    pos_rows, neg_rows = S.w2(ids)
    e = expectation((pos_rows * mind.trust).sum(-1), (neg_rows * mind.trust).sum(-1), cfg.k)
    hope = mind.act_hope.get(unit)
    if hope is not None:  # an own act is an expectation of relief (changed rule, see NOTES.md)
        e = torch.where(dst == mind.feel_unit["relief"], torch.clamp(e, min=hope), e)
    raw_pos, raw_neg = pos_rows.sum(-1), neg_rows.sum(-1)
    channel = S.u_channel[dst]
    present = _present_of(dst, occurred)
    weight = e if cfg.neg_weight == "e" else torch.ones_like(e)
    miss = g * a_u * weight * (1.0 - present)
    expected = (e >= cfg.neg_floor) if cfg.neg_floor > 0.0 else torch.ones_like(present, dtype=torch.bool)
    reporting = torch.zeros(ids.numel(), dtype=torch.bool)
    for ch, slot in slot_of_channel.items():
        on = (channel == ch) & expected
        if bool(on.any()):
            reporting |= on
            got = add_evidence(mind, W_NEG, slot, ids[on], miss[on], relief=ch not in echo, all_expect=True)
            d_on, rp_on, rn_on = dst[on], raw_pos[on], raw_neg[on]
            if acquired is not None:
                mine = torch.isin(d_on, acquired)
                if not bool(mine.any()):
                    continue
                d_on, rp_on, rn_on, got = d_on[mine], rp_on[mine], rn_on[mine], got[mine]
            for d, rp, rn, m in zip(d_on.tolist(), rp_on.tolist(), rn_on.tolist(), got.tolist()):
                record[d] = [rp, rp, rn, rn + m, slot, slot]
    S.use_ties(ids[reporting])
    if born_failures:
        return (channel, pos_rows)   # positive evidence is not changed by the misses above
    return None


def born_failure(mind, tie: int, v: int, rivals, entry: list) -> None:
    """F25: the failures a lag-1 tie X -> v born now would hold had it existed all along.

    Before this moment v never followed X, or the tie would exist. Every earlier moment in which X learned what
    comes next and v's channel reported something else gave that something positive evidence on X's tie to it, in
    the class and with the weight (g a_X) of that moment; ``rivals`` holds those rows, read before this moment's
    positive evidence. The full record (each such moment one failure of weight g a_X, as F14 records it, without
    its floor and its weighting by the expectation) would hold the same amount as negative evidence on X -> v, per
    class; the new tie starts with it, so its frequency is the frequency v has had after X and F14 governs what comes
    later. Only targets on sense and act channels (the world, one event per moment). The amounts are marked as
    judged in the classes that learned trust judges: they claim nothing new (each such moment is already a claim on
    the rival's tie, open to judging there) and would otherwise be judged a second time, with a mass that grows with
    how often X was lived. Logged like every increment, so retracting this moment removes them exactly.
    ``entry`` is the induce record of v: the negative evidence after the moment includes them, in the class that holds
    most of them."""
    S = mind.S
    ch = int(S.u_channel[v])
    if ch < 0 or mind.ch.kind(ch) not in ("sense", "act"):
        return
    channel, pos_rows = rivals
    same = channel == ch
    if not bool(same.any()):
        return
    held = pos_rows[same].sum(0).tolist()          # per class: what X's other successors on this channel received
    best, most, total = -1, 0.0, 0.0
    for slot, amount in enumerate(held):
        if amount <= 0.0:
            continue
        got = S.add1(W_NEG, slot, tie, amount)
        if slot not in REFERENCE:
            S.add1(C_NEG, slot, tie, got)
        if mind.__dict__.get("company") is not None:   # F33: noted with the person present
            from .iliski import note
            note(mind, W_NEG, slot, (tie,), (got,))
        total += got
        if got > most:
            best, most = slot, got
    if total > 0.0:
        entry[3] += total
        entry[5] = best


def induce(mind, changes: Dict[int, Dict[int, list]], occurred, slot_of_channel) -> None:
    """Section 4 "up": a generalisation in the last focus acquires what its instance acquired.

    Each instance tie counts at most ``induct_cap`` (raw evidence), discounted by the expectation of the
    ABSTRACT relation. The evidence keeps the provenance class of the event that gave it to the instance.
    """
    S, cfg, cap = mind.S, mind.cfg, mind.cfg.induct_cap
    if cfg.reliability:
        judge_generalisations(mind, changes)
    if getattr(cfg, "voices", False):
        judge_voices(mind, changes)
    activation = {unit: a for unit, a, _ in mind.last_focus}
    missed = set()
    gens = list(mind.last_gens)
    if cfg.miss_scope == "gens":         # F7: every generalisation the instance is read from is judged by it
        from .read import generalisations
        listed = {(g, i) for g, i, _ in gens}
        for instance in changes:
            for general, e_abs in generalisations(mind, instance):
                if (general, instance) not in listed:
                    listed.add((general, instance))
                    gens.append((general, instance, e_abs))
    for general, instance, e_abs in gens:
        record = changes.get(instance)
        if record is None or e_abs <= 0.5:
            continue
        for v, (pos0, pos1, neg0, neg1, slot_pos, slot_neg) in record.items():
            up = (min(pos1, cap) - min(pos0, cap)) * e_abs
            down = (min(neg1, cap) - min(neg0, cap)) * e_abs
            if up > 0.0:
                add_evidence1(mind, W_POS, slot_pos, S.tie(general, v, EXPECT, 1), up)
            if down > 0.0 and not cfg.general_miss:
                tie = S.find_tie(general, v, EXPECT, 1)
                if tie >= 0:
                    add_evidence1(mind, W_NEG, slot_neg, tie, down)
        if cfg.general_miss and general not in missed and (general in activation or cfg.miss_scope == "gens"):
            missed.add(general)
            strength = activation[general] if general in activation else e_abs
            if not S.out.get((general, EXPECT, 1)):
                continue
            ids, dst = S.out_ids((general, EXPECT, 1))
            sums = _sums(mind, ids, False)
            if sums is None:
                _general_miss_rows(mind, ids, dst, strength, occurred, slot_of_channel)
                continue
            w = sums[0]
            e = _expect_np(w[W_POS], w[W_NEG], cfg.k)
            idn, dstn = ids.numpy(), dst.numpy()
            targets = list(occurred)
            miss = (mind.knob["gain"] * strength) * e
            for v, at in zip(targets, _places(S, general, 1, idn, dstn, targets)):
                for p in at:
                    miss[p] = miss[p] * (1.0 - occurred[v][0])
            expected = (e >= cfg.neg_floor) if cfg.neg_floor > 0.0 else None
            channel = S.u_channel.numpy()[dstn]
            for ch, slot in slot_of_channel.items():
                on = channel == ch
                if expected is not None:
                    on &= expected
                sel = np.flatnonzero(on)
                if sel.size:
                    _add_np(mind, W_NEG, slot, ids, idn, sel, miss, True)
                    S.use_ties(ids if sel.size == idn.size else torch.from_numpy(idn[sel]))


def _general_miss_rows(mind, ids: torch.Tensor, dst: torch.Tensor, strength: float, occurred, slot_of_channel) -> None:
    """The misses of a generalisation (``induce``) from the assembled rows (when the trust is not finite)."""
    S, cfg = mind.S, mind.cfg
    w_pos, w_neg = tie_mass(mind, ids)
    e = expectation(w_pos, w_neg, cfg.k)
    channel = S.u_channel[dst]
    present = _present_of(dst, occurred)
    miss = mind.knob["gain"] * strength * e * (1.0 - present)
    expected = (e >= cfg.neg_floor) if cfg.neg_floor > 0.0 else torch.ones_like(present, dtype=torch.bool)
    for ch, slot in slot_of_channel.items():
        on = (channel == ch) & expected
        if bool(on.any()):
            add_evidence(mind, W_NEG, slot, ids[on], miss[on], all_expect=True)
            S.use_ties(ids[on])


def judge_generalisations(mind, changes: Dict[int, Dict[int, list]]) -> None:
    """F9: the same rule applied to the mind's own generalisations.

    For a composed instance that has just lived a value v on a sense/act channel, every generalisation g it is
    read from is judged by the expectation it held for v before this moment: w+ += e(i->g) e(g->v),
    w- += e(i->g) (1 - e(g->v)) on the self-tie g -> g of form RELY (provenance "inferred"). Its expectation is
    how reliably g has predicted what its instances lived. Nothing says which classification fits which
    dimension; that is learned here, per generalisation, from the mind's own life.
    """
    from .read import generalisations, own_mass
    S, k = mind.S, mind.cfg.k
    for instance, record in changes.items():
        kind = int(S.u_kind[instance])
        if not kind & COMPOSED or kind & SLOTTED:
            continue
        lived = [v for v, r in record.items() if r[1] > r[0] and r[0] == 0.0 and r[2] == 0.0  # first time only
                 and int(S.u_channel[v]) >= 0
                 and mind.ch.kind(int(S.u_channel[v])) in ("sense", "act")]
        if not lived:
            continue
        judged = _judged(mind)
        for general, e_abs in generalisations(mind, instance):
            for v in lived:
                judged.add((general, instance, v))
                verdict = _verdict(mind, general, v)
                if verdict is None:
                    continue
                tie = S.tie(general, general, RELY, 0)
                add_evidence1(mind, W_POS, INFERRED, tie, e_abs * verdict[0])
                add_evidence1(mind, W_NEG, INFERRED, tie, e_abs * verdict[1])


def judge_voices(mind, changes: Dict[int, Dict[int, list]]) -> None:
    """F24: what comes next judges the voices that could speak for a context (learning how far to believe a voice).

    When the mind read a context (F11), every generalisation whose members had lived something in its slot was a
    voice for it; each voice's claim was noted: the value it expected most, with frequency f, and the kind of the
    voice (read.voice_kind: how many members spoke, how much they agreed). When that context then lives a value for
    the first time (its tie had no evidence before, so the prediction rested on the voices and not on its own past),
    a claim held if the voice expected that value above one half and most (f > 1/2, the core's measure of a holding
    prediction); otherwise it failed. Every voice is judged, not only the one that spoke: they were all present when
    the context was read, and judging only the one relied on would let what was chosen decide what is learned. Each
    claim is one experience (gain g) on the tie (kind -> holds) and on the tie (every voice -> holds) of the mind's
    own channel ``ses``, class sensed, logged like all evidence and retractable; ``read.voice_credibility`` reads
    them general -> specific. Nothing here names a word, a class or a domain, so what one life taught about voices
    is there when the mind meets the next.
    """
    from .read import VOICE_ALL, VOICE_CHANNEL
    spoken = getattr(mind, "voice", None)
    if not spoken:
        return
    S = mind.S
    for instance, record in changes.items():
        voice = spoken.get(instance)
        if voice is None:
            continue
        lived = [v for v, r in record.items() if r[1] > r[0] and r[0] == 0.0 and r[2] == 0.0
                 and int(S.u_channel[v]) >= 0 and mind.ch.kind(int(S.u_channel[v])) in ("sense", "act")]
        if not lived:
            continue
        channel = mind.add_channel(VOICE_CHANNEL, "knob")
        for kind, claim, claim_f in voice[1]:
            held = claim in lived and claim_f > 0.5
            for symbol in (VOICE_ALL, kind):
                tie = S.tie(mind._atom(channel, symbol), mind.holds, EXPECT, 0)
                add_evidence1(mind, W_POS if held else W_NEG, SENSED, tie, mind.knob["gain"])


def _verdict(mind, general: int, v: int, leave_out: float = 0.0):
    """How generalisation g fared on a lived value v: (w+, w-) for its RELY tie, or None if g had no opinion.

    g's opinion on the channel of v is its sense/act ties on that channel; with none g is not tested.
    verdict "opinion" (F10): tested with the frequency it gave v, f(g -> v) (0 if v never followed g), weighted
    by the conviction of its strongest tie on that channel: w+ = c f, w- = c (1 - f).
    verdict "share" (F10b, measured worse, NOTES.md): tested with the share of its positive evidence on that
    channel that went to v, weighted by the conviction of that evidence.
    ``leave_out`` raw positive evidence on g -> v is removed first (what g received from this very instance).
    """
    S, k = mind.S, mind.cfg.k
    if not S.out.get((general, EXPECT, 1)):
        return None
    ids, dst = S.out_ids((general, EXPECT, 1))
    dstn = dst.numpy()
    sel = np.flatnonzero(S.u_channel.numpy()[dstn] == int(S.u_channel[v]))   # its ties on the channel of v
    if not sel.size:
        return None
    idn = ids.numpy()[sel]
    best = f = total = mine = 0.0
    for d, w_pos, w_neg in zip(dstn[sel].tolist(), _raw_np(S, W_POS, idn).tolist(), _raw_np(S, W_NEG, idn).tolist()):
        if d == v:
            w_pos = max(0.0, w_pos - leave_out)
            f = w_pos / (w_pos + w_neg) if w_pos + w_neg > 0.0 else 0.0
            mine = w_pos
        total += w_pos
        best = max(best, (w_pos + w_neg) / (w_pos + w_neg + k))
    if mind.cfg.verdict == "share":
        if total <= 0.0:
            return None
        p, c = mine / total, total / (total + k)
        return c * p, c * (1.0 - p)
    if best <= 0.0:
        return None
    return best * f, best * (1.0 - f)


def _judged(mind) -> set:
    """(generalisation, instance, value) triples already judged: every lived value tests a generalisation once."""
    if not hasattr(mind, "judged"):
        mind.judged = set()
    return mind.judged


def reflect(mind, budget: int = 1) -> None:
    """F10, idle moments: re-read lived instances with the generalisations the mind has NOW.

    A generalisation that was formed after its instances were lived (a class is found only once enough members
    were met) was never tested on them. Here, while nothing comes from outside, the mind takes the next lived
    instance (round robin over composed units with own evidence on a sense/act value) and judges every present
    generalisation g of it on each value v the instance lived, once per (g, i, v), with the expectation g holds for
    v LEAVING OUT what g received from this instance (induction gives at most e(i->g) x min(W_own+, cap) of raw
    positive evidence per instance; that much is subtracted), so that g is tested as a prediction, not as an echo.
    The evidence goes to the RELY self-tie, provenance "inferred", logged in this idle moment.
    """
    from .read import generalisations
    S, cfg = mind.S, mind.cfg
    judged = _judged(mind)
    n = S.n_units
    cursor = getattr(mind, "reflect_cursor", -1)
    done = tried = 0
    kinds, alive = S.u_kind.numpy(), S.u_alive.numpy()          # no unit is made here
    world = [c for c, kind in enumerate(mind.ch.kinds) if kind in ("sense", "act")]
    while done < budget and tried < n:
        cursor = (cursor + 1) % n
        tried += 1
        kind = int(kinds[cursor])
        if not kind & COMPOSED or kind & SLOTTED or not bool(alive[cursor]):
            continue
        lived = {}
        if world and S.out.get((cursor, EXPECT, 1)):          # its lived values on sense/act channels, in list order
            ids, dst = S.out_ids((cursor, EXPECT, 1))
            dstn = dst.numpy()
            channel = S.u_channel.numpy()[dstn]
            on = channel == world[0]
            for c in world[1:]:
                on |= channel == c
            sel = np.flatnonzero(on)
            if sel.size:
                for p, raw in zip(sel.tolist(), _raw_np(S, W_POS, ids.numpy()[sel]).tolist()):
                    if raw > 0.0:
                        lived[int(dstn[p])] = raw
        if not lived:
            continue
        done += 1
        for general, e_abs in generalisations(mind, cursor):
            rely = None
            for v, raw_own in lived.items():
                if (general, cursor, v) in judged:
                    continue
                judged.add((general, cursor, v))
                verdict = _verdict(mind, general, v, leave_out=e_abs * min(raw_own, cfg.induct_cap))
                if verdict is None:
                    continue
                if rely is None:
                    rely = S.tie(general, general, RELY, 0)
                add_evidence1(mind, W_POS, INFERRED, rely, e_abs * verdict[0])
                add_evidence1(mind, W_NEG, INFERRED, rely, e_abs * verdict[1])
    mind.reflect_cursor = cursor


def learn_value(mind, percepts) -> None:
    """L5, consequences: a signal on a channel of kind "value" (good or bad for this mind, as its body adapter
    defines) is evidence for every step of the episode that led to it: each composed context in the episode's
    trace gets EXPECT lag 2 ("later in the same episode") to the value unit. The trace is the only memory a
    consequence needs; nothing names what is good or bad here."""
    S, g = mind.S, mind.knob["gain"]
    trace = getattr(mind, "trace", [])
    if not trace:
        return
    for unit, intensity, slot, channel in percepts:
        if intensity <= 0.0 or mind.ch.kind(channel) != "value":
            continue
        for step_unit in dict.fromkeys(trace):
            add_evidence1(mind, W_POS, slot, S.tie(step_unit, unit, EXPECT, 2), g * intensity)


def facet_set(mind, unit: int) -> set:
    """The facets an item has been perceived with (its lag-0 ties to units of facet channels, positive evidence).
    Cached per unit until its number of lag-0 ties changes (evidence on a facet tie only grows)."""
    S = mind.S
    ties = S.out.get((unit, EXPECT, 0), [])
    cache = mind.__dict__.setdefault("_facet_cache", {})
    hit = cache.get(unit)
    if hit is not None and hit[0] == len(ties):
        return hit[1]
    out = set()
    facets = [c for c, kind in enumerate(mind.ch.kinds) if kind == "facet"]
    if ties and facets:
        ids, dst = S.out_ids((unit, EXPECT, 0))
        dstn = dst.numpy()
        channel = S.u_channel.numpy()[dstn]
        on = channel == facets[0]
        for c in facets[1:]:
            on |= channel == c
        sel = np.flatnonzero(on)
        if sel.size:                                        # in list order, as the ties were walked
            idn = ids.numpy()
            for p, raw in zip(sel.tolist(), _raw_np(S, W_POS, idn[sel]).tolist()):
                if raw > 0.0:
                    out.add(int(dstn[p]))
    cache[unit] = (len(ties), out)
    return out


_ONES: Dict[int, np.ndarray] = {}


def _raw_np(S, table: int, idn: np.ndarray) -> np.ndarray:
    """``S.w(table, ids).sum(-1)`` for W+ or W- (numpy): a tie of one class holds one value and zeros (sum v + 0.0); a full
    row is summed by the same torch reduction."""
    one = _ONES.get(S.P)
    if one is None:
        one = _ONES[S.P] = np.zeros(128)
        one[1:S.P + 1] = 1.0
    cls_v, ev_v, row_v = S.views()[1:4]
    c = cls_v[idn]
    r = ev_v[table].take(idn) * one.take(c)
    r += 0.0
    if S.n_rows > 1 and c.size and c.max() == WIDE:
        wide = np.flatnonzero(c == WIDE)
        r[wide] = S.ex[table].index_select(0, torch.from_numpy(row_v[idn[wide]])).sum(-1).numpy()
    return r


def _facet_weight(mind, facet: int) -> float:
    """How telling a facet is: 1 / the number of items it was perceived with (a facet everything has tells nothing)."""
    return 1.0 / max(1, len(mind.S.out.get((facet, EXPECT, 0), [])))


def facet_similarity(mind, a: int, b: int, fa: set = None) -> float:
    """Weighted Jaccard overlap of the facet sets of two items, each facet weighted by how telling it is (1 / the
    number of items it was perceived with). A facet no other item has ever had cannot be shared by anything, so it
    says nothing about resemblance and is left out of the comparison. 0 when either has no comparable facet."""
    fa = facet_set(mind, a) if fa is None else fa
    fb = facet_set(mind, b)
    degree = {f: len(mind.S.out.get((f, EXPECT, 0), [])) for f in fa | fb}
    fa = {f for f in fa if degree[f] > 1}
    fb = {f for f in fb if degree[f] > 1}
    if not fa or not fb:
        return 0.0
    w = {f: 1.0 / degree[f] for f in fa | fb}
    union = sum(w.values())
    return sum(w[f] for f in fa & fb) / union if union > 0 else 0.0


def join_by_resemblance(mind, unit: int, slot: int, neighbours: int = 20) -> None:
    """F18: a newly perceived item joins the categories of the items it resembles. Candidates are the items that
    share facets with it; for each category of those items the similarity of the nearest member is the evidence
    (provenance inferred) that the new item is an instance of it. Nothing is said about what the facets mean."""
    from .form import categories_of
    S = mind.S
    mine = facet_set(mind, unit)
    if not mine:
        return
    shared = {}
    kinds = S.u_kind.numpy()
    for f in mine:
        if S.out.get((f, EXPECT, 0)):
            dstn = S.out_ids((f, EXPECT, 0))[1].numpy()
            for x in dstn[(dstn != unit) & ((kinds[dstn] & ATOMIC) != 0)].tolist():   # in list order
                shared[x] = shared.get(x, 0) + 1
    best = {}
    for x, _ in sorted(shared.items(), key=lambda kv: -kv[1])[:neighbours]:
        sim = facet_similarity(mind, unit, x, mine)
        if sim <= 0.0:
            continue
        for category, e in categories_of(mind, x):
            if e > 0.5 and sim > best.get(category, 0.0):
                best[category] = sim
    from .events import INFERRED
    for category, sim in best.items():
        if S.find_tie(unit, category, ABSTRACT, 0) < 0:
            add_evidence1(mind, W_POS, INFERRED, S.tie(unit, category, ABSTRACT, 0), sim)


def learn_facets(mind, items: List[int], facets: List[Tuple[int, float, int]], slot: int) -> None:
    """Facets (parts or features of an item on a channel of kind "facet", e.g. subword pieces of a word, later
    features of a seen object) hold together with the items of the same moment: EXPECT lag 0 in both directions,
    positive evidence g x intensity. They never enter the episode or the focus; the context learns to expect them
    in the next moment like any percept (learn_follow), so a context can expect a suffix of a word it never met."""
    g = mind.knob["gain"]
    for item in items:
        new = float(mind.S.u_use[item]) <= 1.0              # met for the first time (use counts this moment)
        for facet, intensity, _ in facets:
            add_evidence1(mind, W_POS, slot, mind.S.tie(item, facet, EXPECT, 0), g * intensity)
            add_evidence1(mind, W_POS, slot, mind.S.tie(facet, item, EXPECT, 0), g * intensity)
        if new and mind.cfg.resemblance:
            join_by_resemblance(mind, item, slot)


def learn_together(mind, focus: List[Tuple[int, float, bool]], slot: int) -> None:
    """EXPECT lag 0 among the grounded units of the focus; expected but absent partners get negative evidence."""
    S, cfg = mind.S, mind.cfg
    g = mind.knob["gain"]
    present = {unit: a for unit, a, grounded in focus if grounded and a > 0.0}
    for unit, a_u in present.items():
        if S.out.get((unit, EXPECT, 0)):
            ids, dst = S.out_ids((unit, EXPECT, 0))
            sums = _sums(mind, ids, False)
            if sums is None:
                _together_rows(mind, unit, a_u, ids, dst, present, slot)
            else:
                w = sums[0]
                e = _expect_np(w[W_POS], w[W_NEG], cfg.k)
                idn = ids.numpy()
                partners = list(present)
                places = _places(S, unit, 0, idn, dst.numpy(), partners)
                amounts = (g * a_u) * (e if cfg.neg_weight == "e" else np.ones(idn.size))
                for d, at in zip(partners, places):     # x (1 - here); 1 - 0 = 1 leaves the others
                    for p in at:
                        amounts[p] = amounts[p] * (1.0 - present[d])
                if cfg.neg_floor > 0.0:
                    keep = e >= cfg.neg_floor
                    for at in places:                   # here > 0: every present partner is kept
                        for p in at:
                            keep[p] = True
                    sel = np.flatnonzero(keep)
                else:
                    sel = np.arange(idn.size)
                if sel.size:
                    _add_np(mind, W_NEG, slot, ids, idn, sel, amounts, True)
                    S.use_ties(ids if sel.size == idn.size else torch.from_numpy(idn[sel]))
        for other, a_v in present.items():
            if other != unit:
                add_evidence1(mind, W_POS, mind.prov_of.get(other, SENSED), S.tie(unit, other, EXPECT, 0),
                              g * a_u * a_v)


def _together_rows(mind, unit: int, a_u: float, ids: torch.Tensor, dst: torch.Tensor, present, slot: int) -> None:
    """The misses of one unit in ``learn_together`` from the assembled rows (when the trust is not finite)."""
    S, cfg = mind.S, mind.cfg
    g = mind.knob["gain"]
    here = torch.zeros(ids.numel(), dtype=F64)     # activation of each partner if present (few are)
    for d, a_d in present.items():
        here[dst == d] = a_d
    w_pos, w_neg = tie_mass(mind, ids)
    e = expectation(w_pos, w_neg, cfg.k)
    weight = e if cfg.neg_weight == "e" else torch.ones_like(e)
    keep = ((e >= cfg.neg_floor) | (here > 0.0)) if cfg.neg_floor > 0.0 else torch.ones_like(here, dtype=torch.bool)
    add_evidence(mind, W_NEG, slot, ids[keep], (g * a_u * weight * (1.0 - here))[keep], all_expect=True)
    S.use_ties(ids[keep])
