# -*- coding: utf-8 -*-
"""The one substance of Zihin (design C2, sections 1 and 10): a unit table and a sparse tie table that GROW,
and the experience log that makes every single moment exactly retractable.

All knowledge is in tensors. The Python dictionaries kept beside them (tie index, adjacency, symbol and tuple
indices, frame keys) are caches that ``rebuild_indices`` reconstructs from the tensors alone; a test compares
the maintained caches with the rebuilt ones.

Tie table (COO): src, dst, form, lag, w+[P], w-[P], checked+[P], checked-[P], use, last, birth, alive.
For PART ties ``lag`` holds the position of the part. ``checked`` is the part of the evidence of a class that
direct experience has already judged (see ``Mind`` on learned trust); it is additive like the evidence.

Experience log. Every change of a persistent tensor made during a moment is recorded as an increment
(table, slot, tie ids, amounts) under the moment id. Increments live on the dyadic grid of ``evidence`` so the
tensors are exact sums of the log. ``retract(m)`` subtracts the increments of moment m; structure created in m
that no other moment gave evidence or use to is removed (tombstoned, or truncated when m is the last moment).
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

import torch

from .evidence import EXACT_LIMIT, quantize, q

EXPECT, PART, ABSTRACT = 0, 1, 2
FORMS = ("EXPECT", "PART", "ABSTRACT")
# Kind flags are consequences of how a unit came to be, not types chosen by hand.
ATOMIC, COMPOSED, SLOTTED, CATEGORY, INDUCED = 1, 2, 4, 8, 16
W_POS, W_NEG, C_POS, C_NEG = 0, 1, 2, 3  # the four additive per-class tables of a tie

F64, I64 = torch.float64, torch.int64


class MomentLog:
    """What one moment changed: the increments, the structure it created, and the last-use times it replaced."""

    __slots__ = ("m", "units0", "ties0", "units1", "ties1", "alpha0", "ev", "tie_use", "unit_use", "last",
                 "undo", "retracted")

    def __init__(self, m: int, n_units: int, n_ties: int, alpha: List[int]) -> None:
        self.m, self.units0, self.ties0, self.alpha0 = m, n_units, n_ties, alpha
        self.units1, self.ties1 = n_units, n_ties
        self.ev: List[Tuple[int, int, torch.Tensor, torch.Tensor]] = []
        self.tie_use: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.unit_use: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.last: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.undo = None
        self.retracted = False

    def nbytes(self) -> int:
        total = 0
        for _, _, ids, amounts in self.ev:
            total += ids.numel() * 8 + amounts.numel() * 8 + 16
        for ids, amounts in self.tie_use + self.unit_use:
            total += ids.numel() * 16
        if self.last is not None:
            total += self.last[0].numel() * 16
        return total


class Substance:
    """Growing unit and tie tables with an exact additive experience log."""

    def __init__(self, n_slots: int = 6, keep_log: bool = True, cap_units: int = 256, cap_ties: int = 1024):
        self.P = n_slots
        self.keep_log = keep_log
        self.n_units = 0
        self.n_ties = 0
        self._cu, self._ct = cap_units, cap_ties
        self.u_kind = torch.zeros(cap_units, dtype=torch.int8)
        self.u_channel = torch.full((cap_units,), -1, dtype=torch.int32)
        self.u_symbol = torch.full((cap_units,), -1, dtype=I64)
        self.u_arity = torch.zeros(cap_units, dtype=torch.int16)
        self.u_birth = torch.zeros(cap_units, dtype=I64)
        self.u_use = torch.zeros(cap_units, dtype=F64)
        self.u_alive = torch.zeros(cap_units, dtype=torch.bool)
        self.t_src = torch.zeros(cap_ties, dtype=I64)
        self.t_dst = torch.zeros(cap_ties, dtype=I64)
        self.t_form = torch.zeros(cap_ties, dtype=torch.int8)
        self.t_lag = torch.zeros(cap_ties, dtype=torch.int16)
        self.t_w = torch.zeros(4, cap_ties, n_slots, dtype=F64)  # W_POS, W_NEG, C_POS, C_NEG
        self.t_use = torch.zeros(cap_ties, dtype=F64)
        self.t_last = torch.zeros(cap_ties, dtype=I64)
        self.t_birth = torch.zeros(cap_ties, dtype=I64)
        self.t_alive = torch.zeros(cap_ties, dtype=torch.bool)
        self.log: Dict[int, MomentLog] = {}
        self.cur: Optional[MomentLog] = None
        self.now = 0
        self._touched: set = set()
        self._clear_indices()

    # ------------------------------------------------------------------ caches
    def _clear_indices(self) -> None:
        self.tie_index: Dict[Tuple[int, int, int, int], int] = {}
        self.out: Dict[Tuple[int, int, int], List[int]] = {}   # (src, form, lag) -> tie ids; PART uses lag -1
        self.inc: Dict[Tuple[int, int], List[int]] = {}        # (dst, form) -> tie ids
        self.atom_index: Dict[Tuple[int, int], int] = {}       # (channel, symbol index) -> unit
        self.tuple_index: Dict[Tuple[int, ...], int] = {}      # parts -> composed unit
        self.frame_first: Dict[Tuple[int, Tuple[int, ...]], int] = {}  # (pos, parts with -1 at pos) -> instance
        self.frame_of: Dict[Tuple[int, Tuple[int, ...]], int] = {}     # same key -> induced frame unit

    def _index_tie(self, tie: int, src: int, dst: int, form: int, lag: int) -> None:
        self.tie_index[(src, dst, form, lag)] = tie
        self.out.setdefault((src, form, -1 if form == PART else lag), []).append(tie)
        self.inc.setdefault((dst, form), []).append(tie)

    def parts(self, unit: int) -> Tuple[int, ...]:
        """Parts of a composed unit in position order, read from its PART ties."""
        ties = self.out.get((unit, PART, -1), [])
        return tuple(int(self.t_dst[t]) for t in sorted(ties, key=lambda t: int(self.t_lag[t])))

    def rebuild_indices(self) -> None:
        """Reconstruct every cache from the tensors alone (knowledge lives in the tensors)."""
        self._clear_indices()
        for tie in range(self.n_ties):
            if self.t_alive[tie]:
                self._index_tie(tie, int(self.t_src[tie]), int(self.t_dst[tie]), int(self.t_form[tie]),
                                int(self.t_lag[tie]))
        for unit in range(self.n_units):
            if not self.u_alive[unit]:
                continue
            kind = int(self.u_kind[unit])
            if kind & ATOMIC:
                self.atom_index[(int(self.u_channel[unit]), int(self.u_symbol[unit]))] = unit
            if kind & COMPOSED:
                self.tuple_index[self.parts(unit)] = unit
        for unit in range(self.n_units):  # frame keys, in unit order (first instance = lowest id)
            if not self.u_alive[unit] or not int(self.u_kind[unit]) & COMPOSED:
                continue
            parts, kind = self.parts(unit), int(self.u_kind[unit])
            for pos in range(len(parts)):
                key = (pos, parts[:pos] + (-1,) + parts[pos + 1:])
                if kind & INDUCED and int(self.u_kind[parts[pos]]) & CATEGORY and \
                        int(self.u_birth[parts[pos]]) == int(self.u_birth[unit]) and parts[pos] == unit - 1:
                    self.frame_of[key] = unit
                elif not kind & SLOTTED:
                    self.frame_first.setdefault(key, unit)

    # ------------------------------------------------------------------ growth
    def _grow_units(self) -> None:
        extra = self._cu
        for name in ("u_kind", "u_arity", "u_birth", "u_use", "u_alive"):
            col = getattr(self, name)
            setattr(self, name, torch.cat([col, torch.zeros(extra, dtype=col.dtype)]))
        self.u_channel = torch.cat([self.u_channel, torch.full((extra,), -1, dtype=torch.int32)])
        self.u_symbol = torch.cat([self.u_symbol, torch.full((extra,), -1, dtype=I64)])
        self._cu += extra

    def _grow_ties(self) -> None:
        extra = self._ct
        for name in ("t_src", "t_dst", "t_form", "t_lag", "t_use", "t_last", "t_birth", "t_alive"):
            col = getattr(self, name)
            setattr(self, name, torch.cat([col, torch.zeros(extra, dtype=col.dtype)]))
        self.t_w = torch.cat([self.t_w, torch.zeros(4, extra, self.P, dtype=F64)], dim=1)
        self._ct += extra

    def new_unit(self, kind: int, channel: int = -1, symbol: int = -1, arity: int = 0) -> int:
        if self.n_units == self._cu:
            self._grow_units()
        unit = self.n_units
        self.u_kind[unit], self.u_channel[unit], self.u_symbol[unit] = kind, channel, symbol
        self.u_arity[unit], self.u_birth[unit], self.u_alive[unit] = arity, self.now, True
        self.u_use[unit] = 0.0
        self.n_units += 1
        if kind & ATOMIC:
            self.atom_index[(channel, symbol)] = unit
        return unit

    def find_tie(self, src: int, dst: int, form: int, lag: int) -> int:
        return self.tie_index.get((src, dst, form, lag), -1)

    def tie(self, src: int, dst: int, form: int, lag: int) -> int:
        """The tie (src, dst, form, lag), created with zero evidence if it does not exist."""
        tie = self.tie_index.get((src, dst, form, lag), -1)
        if tie >= 0:
            return tie
        if self.n_ties == self._ct:
            self._grow_ties()
        tie = self.n_ties
        self.t_src[tie], self.t_dst[tie], self.t_form[tie], self.t_lag[tie] = src, dst, form, lag
        self.t_birth[tie], self.t_alive[tie], self.t_last[tie] = self.now, True, 0
        self.n_ties += 1
        self._index_tie(tie, src, dst, form, lag)
        return tie

    # ------------------------------------------------------------------ additive changes (all logged)
    def add(self, table: int, slot: int, ids: torch.Tensor, amounts: torch.Tensor) -> torch.Tensor:
        """Add grid-snapped amounts to one per-class table of the ties ``ids``; returns the snapped amounts."""
        amounts = quantize(amounts.to(F64))
        keep = amounts != 0
        if not bool(keep.any()):
            return amounts
        live_ids, live = (ids, amounts) if bool(keep.all()) else (ids[keep], amounts[keep])
        self.t_w[table, :, slot].index_add_(0, live_ids, live)
        if table <= W_NEG:
            self._touched.update(live_ids.tolist())
        if self.cur is not None and self.keep_log:
            self.cur.ev.append((table, slot, live_ids.clone(), live.clone()))
        return amounts  # aligned with ``ids``

    def add1(self, table: int, slot: int, tie: int, amount: float) -> float:
        amount = q(amount)
        if amount != 0.0:
            self.add(table, slot, torch.tensor([tie], dtype=I64), torch.tensor([amount], dtype=F64))
        return amount

    def use_ties(self, ids: torch.Tensor, amount: float = 1.0) -> None:
        if ids.numel() == 0:
            return
        amounts = torch.full((ids.numel(),), q(amount), dtype=F64)
        self.t_use.index_add_(0, ids, amounts)
        if self.cur is not None and self.keep_log:
            self.cur.tie_use.append((ids.clone(), amounts))

    def use_units(self, ids: torch.Tensor, amount: float = 1.0) -> None:
        if ids.numel() == 0:
            return
        amounts = torch.full((ids.numel(),), q(amount), dtype=F64)
        self.u_use.index_add_(0, ids, amounts)
        if self.cur is not None and self.keep_log:
            self.cur.unit_use.append((ids.clone(), amounts))

    # ------------------------------------------------------------------ moments
    def begin(self, m: int, alpha: List[int]) -> None:
        self.now = m
        self._touched = set()
        self.cur = MomentLog(m, self.n_units, self.n_ties, alpha)

    def end(self) -> MomentLog:
        cur = self.cur
        cur.units1, cur.ties1 = self.n_units, self.n_ties
        if self._touched:
            ids = torch.tensor(sorted(self._touched), dtype=I64)
            if self.keep_log:
                cur.last = (ids, self.t_last[ids].clone())
            self.t_last[ids] = cur.m
        if self.keep_log:
            self.log[cur.m] = cur
        self.cur = None
        return cur

    def _subtract(self, entry: MomentLog) -> None:
        for table, slot, ids, amounts in entry.ev:
            self.t_w[table, :, slot].index_add_(0, ids, -amounts)
        for ids, amounts in entry.tie_use:
            self.t_use.index_add_(0, ids, -amounts)
        for ids, amounts in entry.unit_use:
            self.u_use.index_add_(0, ids, -amounts)
        if entry.last is not None:
            ids, old = entry.last
            mine = self.t_last[ids] == entry.m
            self.t_last[ids[mine]] = old[mine]

    def retract(self, m: int) -> Dict[str, int]:
        """Remove exactly what moment m added. Structure born in m that nothing else supports is removed."""
        entry = self.log.get(m)
        if entry is None or entry.retracted:
            raise KeyError(f"moment {m} is not in the experience log or was already retracted")
        if self.cur is not None:
            raise RuntimeError("retract inside a moment")
        self._subtract(entry)
        entry.retracted = True
        killed_ties = killed_units = 0
        t0, t1, u0, u1 = entry.ties0, entry.ties1, entry.units0, entry.units1
        if t1 > t0:
            born = torch.arange(t0, t1)
            dead = self.t_alive[born] & (self.t_w[:2, t0:t1].abs().sum(dim=(0, 2)) == 0) & \
                (self.t_use[born] == 0)
            self.t_alive[born[dead]] = False
            killed_ties = int(dead.sum())
        if u1 > u0:
            alive = self.t_alive[:self.n_ties]
            src, dst = self.t_src[:self.n_ties][alive], self.t_dst[:self.n_ties][alive]
            used = torch.zeros(self.n_units, dtype=torch.bool)
            used[src] = True
            used[dst] = True
            for unit in range(u0, u1):
                if self.u_alive[unit] and not used[unit] and self.u_use[unit] == 0:
                    self.u_alive[unit] = False
                    killed_units += 1
        if killed_ties or killed_units:
            self.rebuild_indices()
        return {"ties_removed": killed_ties, "units_removed": killed_units,
                "increments": sum(ids.numel() for _, _, ids, _ in entry.ev)}

    def undo_last(self) -> MomentLog:
        """Retract the most recent moment and give back its rows: every tensor is as before that moment."""
        m = max(self.log)
        entry = self.log[m]
        if entry.retracted:
            raise KeyError("the last moment was already retracted")
        self._subtract(entry)
        self._forget_rows(entry.units0, entry.ties0)
        for name in ("u_kind", "u_arity", "u_birth", "u_use", "u_alive"):
            getattr(self, name)[entry.units0:self.n_units] = 0
        self.u_channel[entry.units0:self.n_units] = -1
        self.u_symbol[entry.units0:self.n_units] = -1
        for name in ("t_src", "t_dst", "t_form", "t_lag", "t_use", "t_last", "t_birth", "t_alive"):
            getattr(self, name)[entry.ties0:self.n_ties] = 0
        self.t_w[:, entry.ties0:self.n_ties] = 0
        self.n_units, self.n_ties = entry.units0, entry.ties0
        del self.log[m]
        self.now = m - 1
        return entry

    def _forget_rows(self, units0: int, ties0: int) -> None:
        """Take the units >= units0 and the ties >= ties0 (the newest rows) out of every cache."""
        for unit in range(units0, self.n_units):
            if not self.u_alive[unit]:
                continue
            kind = int(self.u_kind[unit])
            if kind & ATOMIC:
                self.atom_index.pop((int(self.u_channel[unit]), int(self.u_symbol[unit])), None)
            if kind & COMPOSED:
                parts = self.parts(unit)
                if self.tuple_index.get(parts) == unit:
                    del self.tuple_index[parts]
                for pos in range(len(parts)):
                    key = (pos, parts[:pos] + (-1,) + parts[pos + 1:])
                    if self.frame_first.get(key) == unit:
                        del self.frame_first[key]
                    if kind & INDUCED and parts[pos] == unit - 1 and self.frame_of.get(key) == unit:
                        del self.frame_of[key]
        for tie in range(self.n_ties - 1, ties0 - 1, -1):
            if not self.t_alive[tie]:
                continue
            src, dst = int(self.t_src[tie]), int(self.t_dst[tie])
            form, lag = int(self.t_form[tie]), int(self.t_lag[tie])
            self.tie_index.pop((src, dst, form, lag), None)
            for table, key in ((self.out, (src, form, -1 if form == PART else lag)), (self.inc, (dst, form))):
                ties = table[key]
                ties.remove(tie)
                if not ties:
                    del table[key]

    # ------------------------------------------------------------------ inspection
    def tensors(self) -> Dict[str, torch.Tensor]:
        """The live part of every persistent tensor (views)."""
        n, t = self.n_units, self.n_ties
        return {"u_kind": self.u_kind[:n], "u_channel": self.u_channel[:n], "u_symbol": self.u_symbol[:n],
                "u_arity": self.u_arity[:n], "u_birth": self.u_birth[:n], "u_use": self.u_use[:n],
                "u_alive": self.u_alive[:n], "t_src": self.t_src[:t], "t_dst": self.t_dst[:t],
                "t_form": self.t_form[:t], "t_lag": self.t_lag[:t], "t_w": self.t_w[:, :t],
                "t_use": self.t_use[:t], "t_last": self.t_last[:t], "t_birth": self.t_birth[:t],
                "t_alive": self.t_alive[:t]}

    def digest(self) -> str:
        h = hashlib.sha256()
        for name, tensor in self.tensors().items():
            h.update(name.encode())
            h.update(tensor.contiguous().numpy().tobytes())
        return h.hexdigest()

    def nbytes(self) -> Dict[str, int]:
        core = sum(t.numel() * t.element_size() for t in self.tensors().values())
        return {"core_tensors": core, "log": sum(e.nbytes() for e in self.log.values())}

    def max_evidence(self) -> float:
        return float(self.t_w[:, :self.n_ties].abs().max()) if self.n_ties else 0.0

    def exact(self) -> bool:
        """True while every stored evidence value is inside the range where grid sums are exact."""
        return self.max_evidence() < EXACT_LIMIT
