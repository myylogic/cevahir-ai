# -*- coding: utf-8 -*-
"""The one substance of Zihin (design C2, sections 1 and 10): a unit table and a sparse tie table that GROW,
and the experience log that makes every single moment exactly retractable.

All knowledge is in tensors. The Python dictionaries kept beside them (tie index, adjacency, symbol and tuple
indices, frame keys) are caches that ``rebuild_indices`` reconstructs from the tensors alone; a test compares
the maintained caches with the rebuilt ones.

Tie table (COO): src, dst, form, lag, w+[P], w-[P], checked+[P], checked-[P], use, last, birth, alive.
For PART ties ``lag`` holds the position of the part. ``checked`` is the part of the evidence of a class that
direct experience has already judged (see ``Mind`` on learned trust); it is additive like the evidence.

How the four per-class tables are stored (memory, not meaning). Almost every tie holds evidence of one
provenance class only and nothing judged, so a tie keeps its class (``t_cls``: 0 none yet, 1 + p for class p)
and its two values (``t_ev``: w+, w- of that class). A tie that receives evidence of a second class or a judged
amount is moved, once, to a full row of the four tables (``t_cls`` = WIDE, ``t_row`` = its row in ``ex``). Every
reading assembles exactly the values the four tables hold (``w``, ``w1``, ``cell``; ``t_w`` gives them all in tie
order), so what is stored and what is read do not depend on this layout; the digest is computed over these values.

Experience log. Every change of a persistent tensor made during a moment is recorded as an increment
(table, slot, tie ids, amounts) under the moment id. Increments live on the dyadic grid of ``evidence`` so the
tensors are exact sums of the log. ``retract(m)`` subtracts the increments of moment m; structure created in m
that no other moment gave evidence or use to is removed (tombstoned, or truncated when m is the last moment).
"""
from __future__ import annotations

import hashlib
import os
import pickle
from array import array
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .evidence import EXACT_LIMIT, GRID, expectation, quantize, q

EXPECT, PART, ABSTRACT = 0, 1, 2
RELY = 3  # self-tie g -> g: how well generalisation g has predicted what its instances lived (F9)
FORMS = ("EXPECT", "PART", "ABSTRACT")
# Kind flags are consequences of how a unit came to be, not types chosen by hand.
ATOMIC, COMPOSED, SLOTTED, CATEGORY, INDUCED = 1, 2, 4, 8, 16
W_POS, W_NEG, C_POS, C_NEG = 0, 1, 2, 3  # the four additive per-class tables of a tie
WIDE = 127  # t_cls of a tie whose four tables are kept in a full row (more than one class, or judged amounts)

F64, I64 = torch.float64, torch.int64

IDS = "i"                    # array typecode of tie ids in the adjacency lists: 32-bit signed on every platform
assert array(IDS).itemsize == 4
ID_LIMIT = (1 << 31) - 1     # the first tie id that would not fit


def tie_key(src: int, dst: int, form: int, lag: int) -> int:
    """One integer for a tie's identity (src, dst, form, lag): the tie index's key, a fraction of the memory of a
    4-tuple. Units and ties are below 2^31, a form below 2^3 and a lag below 2^8 - 1 (PART lags are positions), so
    the fields never overlap and the key is unique."""
    return (((src << 31) | dst) << 11) | (form << 8) | (lag + 1)


class DiskLog:
    """The experience log kept on disk (a ledger is an archive, not working memory).

    The most recent ``keep`` moments stay in memory (undo needs the latest one with its working state); older
    entries are appended to one file and found by offset. Retraction reads an entry back; which moments were
    retracted is kept in memory. Behaves like the in-memory dict for get, [], in, del, len, iteration and items.
    """

    def __init__(self, path: str, keep: int = 4) -> None:
        self.path, self.keep = path, keep
        self.ram: Dict[int, "MomentLog"] = {}
        self.index: Dict[int, Tuple[int, int]] = {}
        self.retracted: set = set()

    def __setitem__(self, m: int, entry: "MomentLog") -> None:
        self.ram[m] = entry
        while len(self.ram) > self.keep:
            old = min(self.ram)
            e = self.ram.pop(old)
            e.undo = None
            data = pickle.dumps(self._pack(e), protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.path, "ab") as f:
                offset = f.tell()
                f.write(data)
            self.index[old] = (offset, len(data))

    def get(self, m: int, default=None):
        if m in self.ram:
            return self.ram[m]
        if m in self.index:
            offset, length = self.index[m]
            with open(self.path, "rb") as f:
                f.seek(offset)
                e = self._unpack(pickle.loads(f.read(length)))
            e.retracted = m in self.retracted
            return e
        return default

    def __getitem__(self, m: int) -> "MomentLog":
        e = self.get(m)
        if e is None:
            raise KeyError(m)
        return e

    @staticmethod
    def _pack(e: "MomentLog") -> tuple:
        """Plain numbers and raw array bytes (no per-tensor pickling overhead)."""
        arr = lambda t: (t.numpy().tobytes(), str(t.dtype), tuple(t.shape))
        return (e.m, e.units0, e.ties0, e.units1, e.ties1, e.alpha0,
                [(table, slot, arr(i), arr(a)) for table, slot, i, a in e.ev],
                [(arr(i), arr(a)) for i, a in e.tie_use], [(arr(i), arr(a)) for i, a in e.unit_use],
                None if e.last is None else (arr(e.last[0]), arr(e.last[1])), getattr(e, "rows0", None))

    @staticmethod
    def _unpack(p) -> "MomentLog":
        if isinstance(p, MomentLog):                        # entries written before the raw-array format
            return p
        import numpy as np
        types = {"torch.int64": (np.int64, torch.int64), "torch.float64": (np.float64, torch.float64),
                 "torch.int32": (np.int32, torch.int32)}
        def ten(x):
            raw, dtype, shape = x
            npt, _ = types[dtype]
            return torch.from_numpy(np.frombuffer(raw, dtype=npt).copy().reshape(shape))
        m, u0, t0, u1, t1, alpha, ev, tu, uu, last = p[:10]
        e = MomentLog(m, u0, t0, alpha, p[10] if len(p) > 10 else None)
        e.units1, e.ties1 = u1, t1
        e.ev = [(table, slot, ten(i), ten(a)) for table, slot, i, a in ev]
        e.tie_use = [(ten(i), ten(a)) for i, a in tu]
        e.unit_use = [(ten(i), ten(a)) for i, a in uu]
        e.last = None if last is None else (ten(last[0]), ten(last[1]))
        return e

    def mark_retracted(self, m: int) -> None:
        self.retracted.add(m)
        if m in self.ram:
            self.ram[m].retracted = True

    def __contains__(self, m) -> bool:
        return m in self.ram or m in self.index

    def __delitem__(self, m: int) -> None:
        if m in self.ram:
            del self.ram[m]
        else:
            del self.index[m]

    def __len__(self) -> int:
        return len(self.ram) + len(self.index)

    def __iter__(self):
        return iter(sorted(set(self.ram) | set(self.index)))

    def keys(self):
        return list(iter(self))

    def items(self):
        return [(m, self[m]) for m in self]

    def values(self):
        return [self[m] for m in self]

    def disk_bytes(self) -> int:
        return os.path.getsize(self.path) if os.path.exists(self.path) else 0


class MomentLog:
    """What one moment changed: the increments, the structure it created, and the last-use times it replaced."""

    __slots__ = ("m", "units0", "ties0", "units1", "ties1", "alpha0", "ev", "tie_use", "unit_use", "last",
                 "undo", "retracted", "rows0")

    def __init__(self, m: int, n_units: int, n_ties: int, alpha: List[int], rows0: Optional[int] = None) -> None:
        self.m, self.units0, self.ties0, self.alpha0 = m, n_units, n_ties, alpha
        self.rows0 = rows0                      # full evidence rows before the moment (None: logged before rows)
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


class _Arrays:
    """numpy views (same memory) of the store's arrays. Reading or writing one value, or a few, through numpy costs a
    small fraction of the same operation through torch; the values and their order are the same. Growth replaces the
    tie arrays together, ``ex`` (with the row arrays) alone and the unit arrays together: ``Substance._np`` takes the
    views again when one of them was replaced."""

    __slots__ = ("t_cls_t", "ex_t", "u_kind_t", "five", "form", "cls", "ev", "row", "src", "dst", "lag", "use", "last",
                 "birth", "alive", "ex", "r_tie", "r_was", "kind", "channel", "symbol", "arity", "ubirth", "uuse",
                 "ualive")

    def __init__(self, S: "Substance") -> None:
        self.t_cls_t, self.ex_t, self.u_kind_t = S.t_cls, S.ex, S.u_kind
        self.form, self.cls, self.ev, self.row = S.t_form.numpy(), S.t_cls.numpy(), S.t_ev.numpy(), S.t_row.numpy()
        self.src, self.dst, self.lag, self.use = S.t_src.numpy(), S.t_dst.numpy(), S.t_lag.numpy(), S.t_use.numpy()
        self.last, self.birth, self.alive = S.t_last.numpy(), S.t_birth.numpy(), S.t_alive.numpy()
        self.ex, self.r_tie, self.r_was = S.ex.numpy(), S.r_tie.numpy(), S.r_was.numpy()
        self.kind, self.channel, self.symbol = S.u_kind.numpy(), S.u_channel.numpy(), S.u_symbol.numpy()
        self.arity, self.ubirth, self.uuse, self.ualive = (S.u_arity.numpy(), S.u_birth.numpy(), S.u_use.numpy(),
                                                           S.u_alive.numpy())
        self.five = (self.form, self.cls, self.ev, self.row, self.ex)


class Substance:
    """Growing unit and tie tables with an exact additive experience log."""

    def __init__(self, n_slots: int = 6, keep_log: bool = True, cap_units: int = 256, cap_ties: int = 1024,
                 cap_rows: int = 64):
        assert n_slots < WIDE
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
        self.t_cls = torch.zeros(cap_ties, dtype=torch.int8)    # 0 no evidence, 1 + p one class p, WIDE a full row
        self.t_ev = torch.zeros(2, cap_ties, dtype=F64)         # w+, w- of the one class
        self.t_row = torch.zeros(cap_ties, dtype=torch.int32)   # the full row of a WIDE tie
        self._cr = cap_rows
        self.n_rows = 1                                         # row 0 is never used
        self.ex = torch.zeros(4, cap_rows, n_slots, dtype=F64)  # full rows: W_POS, W_NEG, C_POS, C_NEG
        self.r_tie = torch.zeros(cap_rows, dtype=torch.int32)   # the tie of each full row
        self.r_was = torch.zeros(cap_rows, dtype=torch.int8)    # the t_cls the tie had before its row (for undo)
        self.t_use = torch.zeros(cap_ties, dtype=F64)
        self.t_last = torch.zeros(cap_ties, dtype=I64)
        self.t_birth = torch.zeros(cap_ties, dtype=I64)
        self.t_alive = torch.zeros(cap_ties, dtype=torch.bool)
        self.log: Dict[int, MomentLog] = {}
        self.cur: Optional[MomentLog] = None
        self.now = 0
        self.version = 0                        # counts writes: a reading memoised under one value is still exact
        self._touch: list = []                  # W-table writes of the moment: id arrays (numpy int64) and single ids
        self._touch1: list = []
        self._npv = None
        self._clear_indices()

    # ------------------------------------------------------------------ caches
    def _clear_indices(self) -> None:
        self.tie_index: Dict[int, int] = {}                     # tie_key(src, dst, form, lag) -> tie
        self.out: Dict[Tuple[int, int, int], array] = {}        # (src, form, lag) -> tie ids; PART uses lag -1
        self.inc: Dict[Tuple[int, int], array] = {}             # (dst, form) -> tie ids
        self.atom_index: Dict[Tuple[int, int], int] = {}       # (channel, symbol index) -> unit
        self.tuple_index: Dict[Tuple[int, ...], int] = {}      # parts -> composed unit
        self.frame_first: Dict[Tuple[int, Tuple[int, ...]], int] = {}  # (pos, parts with -1 at pos) -> instance
        self.frame_of: Dict[Tuple[int, Tuple[int, ...]], int] = {}     # same key -> induced frame unit
        self._out_ids: Dict[Tuple[int, int, int], tuple] = {}  # out key -> (count, ids, dst) as tensors
        self._parts: Dict[int, tuple] = {}     # composed unit -> (its PART list, its length, parts), see ``parts``

    def _index_tie(self, tie: int, src: int, dst: int, form: int, lag: int) -> None:
        # Compact indices (exact: the same ids in the same order). A tie's key is one packed integer instead of a
        # 4-tuple, and an adjacency list is an array of 32-bit ids (4 bytes each) instead of a list of Python ints
        # (8 bytes of pointer plus about 32 of int object each): the memory a mind needs per tie falls, and reading a
        # list into numpy needs no conversion. Iteration, len, truth, slicing, remove and append behave as a list's.
        self.tie_index[tie_key(src, dst, form, lag)] = tie
        key = (src, form, -1 if form == PART else lag)
        ties = self.out.get(key)
        if ties is None:
            ties = self.out[key] = array(IDS)
        ties.append(tie)
        key = (dst, form)
        ties = self.inc.get(key)
        if ties is None:
            ties = self.inc[key] = array(IDS)
        ties.append(tie)

    def out_ids(self, key: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """The ties of ``self.out[key]`` and their targets as tensors (the list must not be empty). Kept between
        moments: a list only grows at its end (a longer list extends the tensors); a removal drops the entry."""
        ties = self.out[key]
        hit = self._out_ids.get(key)
        if hit is not None and hit[0] == len(ties):
            return hit[1], hit[2]
        v = self._np()
        if hit is not None and hit[0] < len(ties):
            new = np.frombuffer(ties, dtype=np.int32)[hit[0]:].astype(np.int64)
            ids = torch.from_numpy(np.concatenate((hit[1].numpy(), new)))
            dst = torch.from_numpy(np.concatenate((hit[2].numpy(), v.dst[new])))
        else:
            new = np.frombuffer(ties, dtype=np.int32).astype(np.int64)
            ids, dst = torch.from_numpy(new), torch.from_numpy(v.dst[new])
        self._out_ids[key] = (len(ties), ids, dst)
        return ids, dst

    def _np(self) -> _Arrays:
        """numpy views of every array (see ``_Arrays``), taken again after growth replaced an array."""
        v = self._npv
        if v is None or v.t_cls_t is not self.t_cls or v.ex_t is not self.ex or v.u_kind_t is not self.u_kind:
            v = self._npv = _Arrays(self)
        return v

    def views(self):
        """numpy views (same memory) of t_form, t_cls, t_ev, t_row and ex, for reading and adding single values.
        Growth replaces the tie arrays together and ``ex`` alone: take the views again after any write."""
        return self._np().five

    def arrays(self) -> _Arrays:
        """numpy views (same memory) of the unit and tie columns for reading single values: ``kind``, ``channel``,
        ``symbol``, ``arity``, ``ubirth``, ``uuse``, ``ualive`` (units) and ``src``, ``dst``, ``form``, ``lag``,
        ``use``, ``last``, ``birth``, ``alive`` (ties); ``int(S.arrays().kind[u])`` is ``int(S.u_kind[u])`` at a
        twentieth of the cost. Take them again after a write that can add units or ties (growth replaces arrays)."""
        return self._np()

    _UNIT_ARRAYS = ("u_kind", "u_channel", "u_symbol", "u_arity", "u_birth", "u_use", "u_alive")
    _TIE_ARRAYS = ("t_src", "t_dst", "t_form", "t_lag", "t_cls", "t_row", "t_use", "t_last", "t_birth", "t_alive")
    _ROW_ARRAYS = ("r_tie", "r_was")
    _CACHES = ("_out_ids", "_views", "_class_ids", "_npv", "_parts", "_slot_keys", "_lut")

    def __getstate__(self):
        """Saved and copied without the spare capacity (the rows beyond n_units / n_ties / n_rows are zeros); the
        arrays grow again on demand exactly as they do while living. Caches of tensors are not saved."""
        state = self.__dict__.copy()
        n, t, r = self.n_units, self.n_ties, self.n_rows
        for name in self._UNIT_ARRAYS:
            state[name] = getattr(self, name)[:max(n, 1)].clone()
        for name in self._TIE_ARRAYS:
            state[name] = getattr(self, name)[:max(t, 1)].clone()
        state["t_ev"] = self.t_ev[:, :max(t, 1)].clone()
        for name in self._ROW_ARRAYS:
            state[name] = getattr(self, name)[:r].clone()
        state["ex"] = self.ex[:, :r].clone()
        state["_cu"], state["_ct"], state["_cr"] = max(n, 1), max(t, 1), r
        for name in self._CACHES:
            state.pop(name, None)
        return state

    def __setstate__(self, state) -> None:
        t_w = state.pop("t_w", None)
        touched = state.pop("_touched", None)    # the moment's touched ties as kept before ``_touch`` (a set)
        state.pop("_views", None)
        self.__dict__.update(state)
        self._out_ids, self._npv, self._parts = {}, None, {}
        self.__dict__.setdefault("_touch", [])
        self.__dict__.setdefault("_touch1", sorted(touched) if touched else [])
        self.__dict__.setdefault("version", 0)
        if t_w is not None:                      # saved before the narrow layout: the same values, stored anew
            self._from_tables(t_w)
        self._compact_indices()

    def _compact_indices(self) -> None:
        """Indices saved before the compact form (4-tuple tie keys, lists of ids) are converted once on load: the same
        keys and ids in the same order."""
        index = self.__dict__.get("tie_index")
        if index and isinstance(next(iter(index)), tuple):
            self.tie_index = {tie_key(*k): v for k, v in index.items()}
        for name in ("out", "inc"):
            table = self.__dict__.get(name)
            if table and not isinstance(next(iter(table.values())), array):
                for k, v in table.items():
                    table[k] = array(IDS, v)

    def _from_tables(self, t_w: torch.Tensor) -> None:
        """Store the four tables ``t_w`` [4, cap, P] in the narrow layout (every value kept, bit for bit)."""
        cap, P = t_w.shape[1], self.P
        bits = t_w.view(I64) != 0                                # -0.0 counts as a value
        classes = bits[:2].any(dim=0)                            # [cap, P]: classes with w+ or w-
        n_cls = classes.sum(-1)
        wide = (n_cls > 1) | bits[2:].any(dim=(0, 2))
        cls = torch.where(n_cls == 1, classes.to(torch.int8).argmax(-1).to(torch.int8) + 1,
                          torch.zeros(cap, dtype=torch.int8))
        narrow = (~wide) & (n_cls == 1)
        idx = narrow.nonzero().flatten()
        self.t_ev = torch.zeros(2, cap, dtype=F64)
        col = (cls[idx].long() - 1)
        self.t_ev[0, idx] = t_w[0, idx, col]
        self.t_ev[1, idx] = t_w[1, idx, col]
        rows = wide.nonzero().flatten()
        k = rows.numel()
        self._cr = k + 1
        self.n_rows = k + 1
        self.ex = torch.zeros(4, k + 1, P, dtype=F64)
        self.ex[:, 1:] = t_w[:, rows]
        self.r_tie = torch.zeros(k + 1, dtype=torch.int32)
        self.r_tie[1:] = rows.to(torch.int32)
        self.r_was = torch.zeros(k + 1, dtype=torch.int8)
        cls[rows] = WIDE
        self.t_cls = cls
        self.t_row = torch.zeros(cap, dtype=torch.int32)
        self.t_row[rows] = torch.arange(1, k + 1, dtype=torch.int32)
        self._npv = None

    # ------------------------------------------------------------------ evidence: reading
    def w(self, table: int, ids: torch.Tensor, tables: int = 1) -> torch.Tensor:
        """[n, P]: the per-class values of one table for the ties ``ids`` (a new tensor). With ``tables`` > 1, the
        tables table .. table + tables - 1 as [tables, n, P]."""
        out = torch.from_numpy(self._rows(table, ids.numpy(), tables))
        return out if tables > 1 else out[0]

    def _rows(self, table: int, i: np.ndarray, tables: int) -> np.ndarray:
        """``w`` in numpy: [tables, n, P] float64, assembled by selection only (zeros, then each one-class tie's two
        values in its class column, then the full rows), so every value is a stored value, bit for bit."""
        v = self._np()
        P = self.P
        cm = v.cls[i] - 1                        # one class p: p; none: -1; a full row: WIDE - 1
        n = cm.shape[0]
        out = np.zeros((tables, n, P))
        narrow = n
        if table <= W_NEG:
            r = (cm.view(np.uint8) < P).nonzero()[0]
            narrow = r.shape[0]
            m = min(tables, 2 - table)
            if narrow and n > 64:                # long lists: one flat index per value (the same cells, faster)
                flat = r * P + (cm if narrow == n else cm[r])
                got = v.ev[table:table + m, i if narrow == n else i[r]]
                cells = out.reshape(tables, n * P)
                for t in range(m):
                    cells[t][flat] = got[t]
            elif narrow == n:                    # every tie holds one class (r is 0 .. n-1)
                out[:m, r, cm] = v.ev[table:table + m, i]
            elif narrow:
                out[:m, r, cm[r]] = v.ev[table:table + m, i[r]]
        else:
            narrow = 0
        if narrow < n and self.n_rows > 1:
            wide = (cm == WIDE - 1).nonzero()[0]
            if wide.shape[0]:
                out[:, wide] = v.ex[table:table + tables, v.row[i[wide]]]
        return out

    def _classes(self) -> torch.Tensor:
        c = self.__dict__.get("_class_ids")
        if c is None:
            c = self._class_ids = torch.arange(1, self.P + 1, dtype=torch.int8)
        return c

    def w2(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """W+ and W- rows ([n, P] each) of the ties ``ids``."""
        both = self._rows(W_POS, ids.numpy(), 2)
        return torch.from_numpy(both[0]), torch.from_numpy(both[1])

    def w4(self, ids: torch.Tensor) -> torch.Tensor:
        """[4, n, P]: all four tables for the ties ``ids``."""
        return self.w(W_POS, ids, 4)

    # Fused reads: what callers compute from ``w2`` / ``w1`` rows, bit for bit, without assembling the rows. A one-class
    # tie's row is zero outside its class, so (row * trust).sum(-1) is ev * trust[class] plus zeros; adding 0.0 gives
    # exactly that sum (it turns -0.0 into +0.0, as the zeros do). Full rows keep the row reduction. Valid for a trust
    # vector of finite values without a sign bit (every trust the mind uses); any other trust takes the rows.
    def _fusable(self, trust: torch.Tensor):
        """trust by t_cls (0 and WIDE give 0), or None for a trust the fused reads do not cover. Kept for the last
        trust values: the key is the values themselves, so a change made through .numpy() or .data (which does not
        count in the tensor's _version) can never leave a stale lookup."""
        tr = trust.numpy() if isinstance(trust, torch.Tensor) else np.asarray(trust, dtype=np.float64)
        key = tr.tobytes() if tr.dtype == np.float64 else None
        hit = self.__dict__.get("_lut")
        if key is not None and hit is not None and hit[0] == key:
            return hit[1]
        if tr.shape != (self.P,) or tr.dtype != np.float64 or np.signbit(tr).any() or not np.isfinite(tr).all():
            lut = None
        else:
            lut = np.zeros(WIDE + 1)
            lut[1:self.P + 1] = tr
        if key is not None:
            self._lut = (key, lut)
        return lut

    def _fused(self, ids: torch.Tensor, trust, total: bool):
        i = ids.numpy()
        lut = self._fusable(trust)
        if lut is None:
            return None
        v = self._np()
        c = v.cls[i]
        tr = lut[c]
        ev = v.ev[:, i]
        if total:
            out = [(ev[0] + ev[1]) * tr + 0.0]
        else:
            out = [ev[0] * tr + 0.0, ev[1] * tr + 0.0]
        if self.n_rows > 1:
            wide = (c == WIDE).nonzero()[0]
            if wide.shape[0]:
                rows = v.row[i[wide]]
                t = trust if isinstance(trust, torch.Tensor) else torch.as_tensor(trust, dtype=F64)
                pos, neg = torch.from_numpy(v.ex[W_POS, rows]), torch.from_numpy(v.ex[W_NEG, rows])
                if total:
                    out[0][wide] = ((pos + neg) * t).sum(-1).numpy()
                else:
                    out[0][wide] = (pos * t).sum(-1).numpy()
                    out[1][wide] = (neg * t).sum(-1).numpy()
        return [torch.from_numpy(x) for x in out]

    def mass2(self, ids: torch.Tensor, trust: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trust-weighted (W+, W-) of the ties ``ids``: (w2[0] * trust).sum(-1), (w2[1] * trust).sum(-1)."""
        out = self._fused(ids, trust, False)
        if out is None:
            pos, neg = self.w2(ids)
            return (pos * trust).sum(-1), (neg * trust).sum(-1)
        return out[0], out[1]

    def mass_total(self, ids: torch.Tensor, trust: torch.Tensor) -> torch.Tensor:
        """Trust-weighted W of the ties ``ids``: ((w2[0] + w2[1]) * trust).sum(-1)."""
        out = self._fused(ids, trust, True)
        if out is None:
            pos, neg = self.w2(ids)
            return ((pos + neg) * trust).sum(-1)
        return out[0]

    def raw2(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Raw (not trust-weighted) W+ and W- of the ties ``ids``: w2[0].sum(-1), w2[1].sum(-1)."""
        i = ids.numpy()
        v = self._np()
        c = v.cls[i]
        ev = v.ev[:, i] + 0.0                    # a one-class row sums to its value (+0.0 for -0.0)
        if self.n_rows > 1:
            wide = (c == WIDE).nonzero()[0]
            if wide.shape[0]:
                rows = v.row[i[wide]]
                ev[0][wide] = torch.from_numpy(v.ex[W_POS, rows]).sum(-1).numpy()
                ev[1][wide] = torch.from_numpy(v.ex[W_NEG, rows]).sum(-1).numpy()
        return torch.from_numpy(ev[0]), torch.from_numpy(ev[1])

    def mass1(self, tie: int, trust: torch.Tensor) -> Tuple[float, float]:
        """float((w1(W_POS, tie) * trust).sum()), float((w1(W_NEG, tie) * trust).sum())."""
        v = self._np()
        c = int(v.cls[tie])
        lut = self._fusable(trust) if c != WIDE else None
        if lut is not None:
            if c == 0:                           # a row of zeros
                return 0.0, 0.0
            t = float(lut[c])
            return float(v.ev[W_POS, tie]) * t + 0.0, float(v.ev[W_NEG, tie]) * t + 0.0
        return float((self.w1(W_POS, tie) * trust).sum()), float((self.w1(W_NEG, tie) * trust).sum())

    def expect(self, ids: torch.Tensor, trust: torch.Tensor, k: float = 1.0) -> torch.Tensor:
        """expectation((w2[0] * trust).sum(-1), (w2[1] * trust).sum(-1), k)."""
        w_pos, w_neg = self.mass2(ids, trust)
        return expectation(w_pos, w_neg, k)

    def open_classes(self, ids: torch.Tensor) -> Tuple[List[bool], bool]:
        """What ``learn.judge`` derives from w4(ids): per class, whether any tie holds evidence not yet judged,
        ((W+ != C+) | (W- != C-)).any(0).tolist(), and whether every value of the four tables is finite. A one-class
        tie has nothing judged (C = 0), so its class is open where w+ or w- is not zero; full rows are compared."""
        i = ids.numpy()
        v = self._np()
        P = self.P
        cm = v.cls[i] - 1
        r = (cm.view(np.uint8) < P).nonzero()[0]
        ev = v.ev[:, i[r]]
        found = np.zeros(P, dtype=bool)
        found[cm[r][(ev[0] != 0) | (ev[1] != 0)]] = True
        finite = bool(np.isfinite(ev).all())
        if self.n_rows > 1:
            wide = (cm == WIDE - 1).nonzero()[0]
            if wide.shape[0]:
                ex = v.ex[:, v.row[i[wide]]]
                found |= ((ex[W_POS] != ex[C_POS]) | (ex[W_NEG] != ex[C_NEG])).any(0)
                finite = finite and bool(np.isfinite(ex).all())
        return found.tolist(), finite

    def w1(self, table: int, tie: int) -> torch.Tensor:
        """[P]: the per-class values of one table for one tie (a new tensor)."""
        v = self._np()
        c = int(v.cls[tie])
        if c == WIDE:
            return torch.from_numpy(v.ex[table, v.row[tie]].copy())
        out = np.zeros(self.P)
        if c and table <= W_NEG:
            out[c - 1] = v.ev[table, tie]
        return torch.from_numpy(out)

    def cell(self, table: int, tie: int, slot: int) -> float:
        """The value of one table, one class, one tie."""
        v = self._np()
        c = int(v.cls[tie])
        if c == WIDE:
            return float(v.ex[table, v.row[tie], slot])
        return float(v.ev[table, tie]) if c == slot + 1 and table <= W_NEG else 0.0

    def column(self, table: int, slot: int) -> torch.Tensor:
        """[n_ties]: one table, one class, for every tie in tie order."""
        t = self.n_ties
        if table <= W_NEG:
            out = torch.where(self.t_cls[:t] == slot + 1, self.t_ev[table, :t], torch.zeros((), dtype=F64))
        else:
            out = torch.zeros(t, dtype=F64)
        if self.n_rows > 1:
            out[self.r_tie[1:self.n_rows].long()] = self.ex[table, 1:self.n_rows, slot]
        return out

    @property
    def t_w(self) -> torch.Tensor:
        """The four tables of every tie in tie order, [4, n_ties, P] (assembled: a copy for reading)."""
        return self.w4(torch.arange(self.n_ties))

    def slot_keys(self, parts: Tuple[int, ...]):
        """Slot keys of a tuple: (key, positions). Without ``bind_repeats`` one key per position (as written);
        with it, all positions holding the same unit form one slot (a variable that occurs more than once).
        They depend on the tuple alone: computed once per tuple (and setting), a new list each call."""
        bind = getattr(self, "bind_repeats", False)
        cache = self.__dict__.get("_slot_keys")
        if cache is None or len(cache) > 1 << 16:
            cache = self._slot_keys = {}
        key = (parts if type(parts) is tuple else tuple(parts), bind)
        hit = cache.get(key)
        if hit is None:
            hit = cache[key] = tuple(self._slot_keys_of(parts, bind))
        return list(hit)

    @staticmethod
    def _slot_keys_of(parts: Tuple[int, ...], bind: bool):
        out, seen = [], set()
        for pos, part in enumerate(parts):
            if bind:
                where = tuple(j for j, q in enumerate(parts) if q == part)
                if where in seen:
                    continue
                seen.add(where)
            else:
                where = (pos,)
            masked = tuple(-1 if j in where else q for j, q in enumerate(parts))
            out.append(((where[0] if len(where) == 1 else where, masked), where))
        return out

    def parts(self, unit: int) -> Tuple[int, ...]:
        """Parts of a composed unit in position order, read from its PART ties. Kept per unit with the length of its
        PART list: the list only grows at its end while the unit lives, and the entry is dropped where it can shrink
        (``_forget_rows``, ``rebuild_indices``)."""
        ties = self.out.get((unit, PART, -1))
        if not ties:
            return ()
        hit = self._parts.get(unit)
        if hit is not None and hit[0] == len(ties):
            return hit[1]
        v = self._np()
        lag, dst = v.lag, v.dst
        found = tuple(int(dst[t]) for t in sorted(ties, key=lambda t: int(lag[t])))
        if len(self._parts) > 1 << 17:
            self._parts = {}
        self._parts[unit] = (len(ties), found)
        return found

    def rebuild_indices(self) -> None:
        """Reconstruct every cache from the tensors alone (knowledge lives in the tensors)."""
        self._clear_indices()
        v = self._np()
        for start in range(0, self.n_ties, 1 << 18):       # in tie order, read through numpy in blocks
            stop = min(self.n_ties, start + (1 << 18))
            src, dst = v.src[start:stop].tolist(), v.dst[start:stop].tolist()
            form, lag = v.form[start:stop].tolist(), v.lag[start:stop].tolist()
            for j in np.flatnonzero(v.alive[start:stop]).tolist():
                self._index_tie(start + j, src[j], dst[j], form[j], lag[j])
        alive, kinds = v.ualive[:self.n_units].tolist(), v.kind[:self.n_units].tolist()
        channel, symbol, birth = (v.channel[:self.n_units].tolist(), v.symbol[:self.n_units].tolist(),
                                  v.ubirth[:self.n_units].tolist())
        for unit in range(self.n_units):
            if not alive[unit]:
                continue
            kind = kinds[unit]
            if kind & ATOMIC:
                self.atom_index[(channel[unit], symbol[unit])] = unit
            if kind & COMPOSED:
                self.tuple_index[self.parts(unit)] = unit
        for unit in range(self.n_units):  # frame keys, in unit order (first instance = lowest id)
            if not alive[unit] or not kinds[unit] & COMPOSED:
                continue
            parts, kind = self.parts(unit), kinds[unit]
            for key, where in self.slot_keys(parts):
                pos = where[0]
                if kind & INDUCED and kinds[parts[pos]] & CATEGORY and \
                        birth[parts[pos]] == birth[unit] and parts[pos] == unit - 1:
                    self.frame_of[key] = unit
                elif not kind & SLOTTED:
                    self.frame_first.setdefault(key, unit)

    # ------------------------------------------------------------------ growth
    @staticmethod
    def _grown(col: torch.Tensor, extra: int, fill=0, dim: int = 0) -> torch.Tensor:
        """``col`` followed by ``extra`` rows of ``fill`` along ``dim`` (what torch.cat with a filled block gives,
        written once into new memory)."""
        n = col.shape[dim]
        shape = list(col.shape)
        shape[dim] = n + extra
        out = torch.empty(shape, dtype=col.dtype)
        out.narrow(dim, 0, n).copy_(col)
        out.narrow(dim, n, extra).fill_(fill)
        return out

    def _grow_units(self) -> None:
        extra = self._cu
        for name in ("u_kind", "u_arity", "u_birth", "u_use", "u_alive"):
            setattr(self, name, self._grown(getattr(self, name), extra))
        self.u_channel = self._grown(self.u_channel, extra, -1)
        self.u_symbol = self._grown(self.u_symbol, extra, -1)
        self._cu += extra
        self._npv = None

    def _grow_ties(self) -> None:
        extra = self._ct
        for name in self._TIE_ARRAYS:
            setattr(self, name, self._grown(getattr(self, name), extra))
        self.t_ev = self._grown(self.t_ev, extra, dim=1)
        self._ct += extra
        self._npv = None

    def _grow_rows(self) -> None:
        extra = self._cr
        for name in self._ROW_ARRAYS:
            setattr(self, name, self._grown(getattr(self, name), extra))
        self.ex = self._grown(self.ex, extra, dim=1)
        self._cr += extra
        self._npv = None

    def _widen(self, tie: int) -> int:
        """Give a tie its full row of the four tables (its one class moves there, value for value); the row."""
        if self.n_rows == self._cr:
            self._grow_rows()
        v = self._np()
        row = self.n_rows
        c = int(v.cls[tie])
        if 0 < c < WIDE:
            v.ex[W_POS, row, c - 1] = v.ev[W_POS, tie]
            v.ex[W_NEG, row, c - 1] = v.ev[W_NEG, tie]
            v.ev[:, tie] = 0.0
        v.r_tie[row], v.r_was[row] = tie, c
        v.cls[tie], v.row[tie] = WIDE, row
        self.n_rows += 1
        return row

    def _widen_many(self, ties: np.ndarray) -> None:
        """``_widen`` for distinct ties in ascending order at once: the same rows in the same order, the same values
        moved, the same growth (rows grow while the next one needed does not fit, as one by one)."""
        k = ties.shape[0]
        while self.n_rows + k > self._cr:
            self._grow_rows()
        v = self._np()
        rows = np.arange(self.n_rows, self.n_rows + k)
        c = v.cls[ties]
        one = ((c > 0) & (c < WIDE)).nonzero()[0]
        if one.shape[0]:
            r1, t1, c1 = rows[one], ties[one], c[one].astype(np.intp) - 1
            v.ex[W_POS, r1, c1] = v.ev[W_POS, t1]
            v.ex[W_NEG, r1, c1] = v.ev[W_NEG, t1]
            v.ev[:, t1] = 0.0
        v.r_tie[rows] = ties
        v.r_was[rows] = c
        v.cls[ties] = WIDE
        v.row[ties] = rows
        self.n_rows += k

    def new_unit(self, kind: int, channel: int = -1, symbol: int = -1, arity: int = 0) -> int:
        if self.n_units == self._cu:
            self._grow_units()
        unit = self.n_units
        self.version += 1
        v = self._np()
        v.kind[unit], v.channel[unit], v.symbol[unit] = kind, channel, symbol
        v.arity[unit], v.ubirth[unit], v.ualive[unit] = arity, self.now, True
        v.uuse[unit] = 0.0
        self.n_units += 1
        if kind & ATOMIC:
            self.atom_index[(channel, symbol)] = unit
        return unit

    def find_tie(self, src: int, dst: int, form: int, lag: int) -> int:
        return self.tie_index.get(tie_key(src, dst, form, lag), -1)

    def tie(self, src: int, dst: int, form: int, lag: int) -> int:
        """The tie (src, dst, form, lag), created with zero evidence if it does not exist."""
        tie = self.tie_index.get(tie_key(src, dst, form, lag), -1)
        if tie >= 0:
            return tie
        if self.n_ties == self._ct:
            self._grow_ties()
        tie = self.n_ties
        if tie >= ID_LIMIT:
            raise OverflowError("a mind of more than 2^31 - 1 ties needs 64-bit index arrays")
        self.version += 1
        v = self._np()
        v.src[tie], v.dst[tie], v.form[tie], v.lag[tie] = src, dst, form, lag
        v.birth[tie], v.alive[tie], v.last[tie] = self.now, True, 0
        self.n_ties += 1
        self._index_tie(tie, src, dst, form, lag)
        return tie

    # ------------------------------------------------------------------ additive changes (all logged)
    def add(self, table: int, slot: int, ids: torch.Tensor, amounts: torch.Tensor) -> torch.Tensor:
        """Add grid-snapped amounts to one per-class table of the ties ``ids``; returns the snapped amounts.

        In numpy: the snapping is torch.round(x * GRID) / GRID as np.rint (both round half to even) and the same
        division; the additions are exact grid sums (see ``evidence``), in index order like index_add_."""
        snapped = (amounts if amounts.dtype is F64 else amounts.to(F64)).numpy() * GRID
        np.rint(snapped, out=snapped)
        snapped /= GRID
        out = torch.from_numpy(snapped)          # aligned with ``ids``
        kept = np.count_nonzero(snapped)         # -0.0 counts as zero, like amounts != 0
        if not kept:
            return out
        self.version += 1
        i = ids.numpy()
        if kept == snapped.shape[0]:
            live_i, live = i.copy(), snapped.copy()
        else:
            keep = snapped != 0
            live_i, live = i[keep], snapped[keep]
        self._put_np(table, slot, live_i, live)
        if table <= W_NEG:
            self._touch.append(live_i)
        if self.cur is not None and self.keep_log:
            self.cur.ev.append((table, slot, torch.from_numpy(live_i), torch.from_numpy(live)))
        return out

    def add1(self, table: int, slot: int, tie: int, amount: float) -> float:
        """``add`` for one tie: the same snapped increment (q rounds like torch.round), added and logged alike (the
        log entry is kept as two numbers until the moment ends, see ``_compact``)."""
        amount = q(amount)
        if amount != 0.0:
            self.version += 1
            v = self._np()
            c = v.cls[tie]
            if table <= W_NEG and (c == 0 or c == slot + 1):
                v.cls[tie] = slot + 1
                v.ev[table, tie] += amount
            else:
                r = v.row[tie] if c == WIDE else self._widen(tie)
                self._np().ex[table, r, slot] += amount
            if table <= W_NEG:
                self._touch1.append(tie)
            if self.cur is not None and self.keep_log:
                self.cur.ev.append((table, slot, tie, amount))
        return amount

    def _put(self, table: int, slot: int, ids: torch.Tensor, amounts: torch.Tensor) -> None:
        """Add ``amounts`` to one table and class of the ties ``ids``: in the one class a tie holds (or takes now),
        else in its full row (a tie that cannot hold the value in one class is widened first)."""
        self._put_np(table, slot, ids.numpy(), amounts.numpy())

    def _put_np(self, table: int, slot: int, i: np.ndarray, a: np.ndarray) -> None:
        v = self._np()
        c = v.cls[i]
        if table <= W_NEG:
            mine = c == slot + 1
            if mine.all():
                np.add.at(v.ev[table], i, a)
                return
            fits = mine | (c == 0)
            if fits.all():
                v.cls[i] = slot + 1
                np.add.at(v.ev[table], i, a)
                return
        else:
            fits = np.zeros(i.shape[0], dtype=bool)
        grow = ~fits & (c != WIDE)
        if grow.any():
            self._widen_many(np.unique(i[grow]))      # ascending, as torch.unique: the same rows in the same order
            v = self._np()
            c = v.cls[i]
        wide = c == WIDE
        if not wide.all():
            narrow = ~wide
            v.cls[i[narrow]] = slot + 1
            np.add.at(v.ev[table], i[narrow], a[narrow])
        if wide.any():
            np.add.at(v.ex[table, :, slot], v.row[i[wide]], a[wide])

    def use_ties(self, ids: torch.Tensor, amount: float = 1.0) -> None:
        if ids.numel() == 0:
            return
        a = q(amount)
        self.version += 1
        np.add.at(self._np().use, ids.numpy(), a)
        if self.cur is not None and self.keep_log:
            self.cur.tie_use.append((ids.clone(), torch.full((ids.numel(),), a, dtype=F64)))

    def use_units(self, ids: torch.Tensor, amount: float = 1.0) -> None:
        if ids.numel() == 0:
            return
        a = q(amount)
        self.version += 1
        np.add.at(self._np().uuse, ids.numpy(), a)
        if self.cur is not None and self.keep_log:
            self.cur.unit_use.append((ids.clone(), torch.full((ids.numel(),), a, dtype=F64)))

    # ------------------------------------------------------------------ moments
    def begin(self, m: int, alpha: List[int]) -> None:
        self.now = m
        self._touch, self._touch1 = [], []
        self.cur = MomentLog(m, self.n_units, self.n_ties, alpha, self.n_rows)

    def end(self) -> MomentLog:
        cur = self.cur
        cur.units1, cur.ties1 = self.n_units, self.n_ties
        parts = self._touch
        if self._touch1:
            parts.append(np.array(self._touch1, dtype=np.int64))
        if parts:                                # the ties whose W tables changed in this moment
            touched = (np.concatenate(parts) if len(parts) > 1 else parts[0]).astype(np.int64, copy=False)
            if self.keep_log:
                ids = torch.from_numpy(np.unique(touched))      # ascending and distinct, as sorted(set(...))
                cur.last = (ids, self.t_last[ids].clone())
                self.t_last[ids] = cur.m
            else:
                self._np().last[touched] = cur.m
        self._touch, self._touch1 = [], []
        self.version += 1
        if self.keep_log:
            self._compact(cur)
            self.log[cur.m] = cur
        self.cur = None
        return cur

    @staticmethod
    def _compact(cur: MomentLog) -> None:
        """Merge a finished moment's many small increments into one array per (table, class) and one per use
        table. Amounts are grid-snapped, so sums do not depend on order: retraction stays exact; the per-moment
        memory drops from dozens of small tensors (each with its own overhead) to a handful."""
        if len(cur.ev) > 1 or (cur.ev and not isinstance(cur.ev[0][2], torch.Tensor)):
            groups: Dict[Tuple[int, int], list] = {}
            for table, slot, ids, amounts in cur.ev:
                groups.setdefault((table, slot), []).append((ids, amounts))
            cur.ev = [(table, slot) + Substance._joined(parts) for (table, slot), parts in groups.items()]
        for name in ("tie_use", "unit_use"):
            parts = getattr(cur, name)
            if len(parts) > 1:
                setattr(cur, name, [(torch.cat([i for i, _ in parts]), torch.cat([a for _, a in parts]))])

    @staticmethod
    def _joined(parts: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """The ids and amounts of one (table, class) of a moment in the order they were added; consecutive one-tie
        entries (a tie id and an amount, from ``add1``) become one array, exactly as their one-element tensors
        concatenated would."""
        ids, amounts, run_i, run_a = [], [], [], []
        for i, a in parts:
            if isinstance(i, torch.Tensor):
                if run_i:
                    ids.append(torch.tensor(run_i, dtype=I64))
                    amounts.append(torch.tensor(run_a, dtype=F64))
                    run_i, run_a = [], []
                ids.append(i)
                amounts.append(a)
            else:
                run_i.append(i)
                run_a.append(a)
        if run_i:
            ids.append(torch.tensor(run_i, dtype=I64))
            amounts.append(torch.tensor(run_a, dtype=F64))
        return torch.cat(ids), torch.cat(amounts)

    def _subtract(self, entry: MomentLog) -> None:
        self.version += 1
        for table, slot, ids, amounts in entry.ev:
            self._put(table, slot, ids, -amounts)
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
        if isinstance(self.log, DiskLog):
            self.log.mark_retracted(m)
        killed_ties = killed_units = 0
        t0, t1, u0, u1 = entry.ties0, entry.ties1, entry.units0, entry.units1
        if t1 > t0:
            born = torch.arange(t0, t1)
            held = torch.stack([self.w(W_POS, born), self.w(W_NEG, born)])
            dead = self.t_alive[born] & (held.abs().sum(dim=(0, 2)) == 0) & (self.t_use[born] == 0)
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
        self._give_back_rows(entry)
        for name in ("u_kind", "u_arity", "u_birth", "u_use", "u_alive"):
            getattr(self, name)[entry.units0:self.n_units] = 0
        self.u_channel[entry.units0:self.n_units] = -1
        self.u_symbol[entry.units0:self.n_units] = -1
        for name in self._TIE_ARRAYS:
            getattr(self, name)[entry.ties0:self.n_ties] = 0
        self.t_ev[:, entry.ties0:self.n_ties] = 0
        self.n_units, self.n_ties = entry.units0, entry.ties0
        del self.log[m]
        self.now = m - 1
        return entry

    def _give_back_rows(self, entry: MomentLog) -> None:
        """The full rows given in the moment of ``entry`` go back (after its increments were subtracted): an older
        tie that was widened then holds its one class again, value for value; a younger tie goes with its row."""
        rows0 = getattr(entry, "rows0", None)
        if rows0 is None:                        # logged before full rows: the last rows that belong to its ties
            rows0 = self.n_rows
            while rows0 > 1 and int(self.r_tie[rows0 - 1]) >= entry.ties0:
                rows0 -= 1
        v = self._np()
        for row in range(rows0, self.n_rows):
            tie, was = int(v.r_tie[row]), int(v.r_was[row])
            if tie < entry.ties0:
                v.cls[tie], v.row[tie] = was, 0
                if 0 < was < WIDE:
                    v.ev[W_POS, tie] = v.ex[W_POS, row, was - 1]
                    v.ev[W_NEG, tie] = v.ex[W_NEG, row, was - 1]
        self.ex[:, rows0:self.n_rows] = 0
        self.r_tie[rows0:self.n_rows] = 0
        self.r_was[rows0:self.n_rows] = 0
        self.n_rows = rows0

    def _forget_rows(self, units0: int, ties0: int) -> None:
        """Take the units >= units0 and the ties >= ties0 (the newest rows) out of every cache."""
        v = self._np()
        for unit in range(units0, self.n_units):
            if not v.ualive[unit]:
                continue
            kind = int(v.kind[unit])
            if kind & ATOMIC:
                self.atom_index.pop((int(v.channel[unit]), int(v.symbol[unit])), None)
            if kind & COMPOSED:
                parts = self.parts(unit)
                self._parts.pop(unit, None)
                if self.tuple_index.get(parts) == unit:
                    del self.tuple_index[parts]
                for key, where in self.slot_keys(parts):
                    pos = where[0]
                    if self.frame_first.get(key) == unit:
                        del self.frame_first[key]
                    if kind & INDUCED and parts[pos] == unit - 1 and self.frame_of.get(key) == unit:
                        del self.frame_of[key]
        for tie in range(self.n_ties - 1, ties0 - 1, -1):
            if not v.alive[tie]:
                continue
            src, dst = int(v.src[tie]), int(v.dst[tie])
            form, lag = int(v.form[tie]), int(v.lag[tie])
            self.tie_index.pop(tie_key(src, dst, form, lag), None)
            self._out_ids.pop((src, form, -1 if form == PART else lag), None)
            if form == PART:
                self._parts.pop(src, None)
            for table, key in ((self.out, (src, form, -1 if form == PART else lag)), (self.inc, (dst, form))):
                ties = table[key]
                ties.remove(tie)
                if not ties:
                    del table[key]

    # ------------------------------------------------------------------ inspection
    def tensors(self) -> Dict[str, torch.Tensor]:
        """The live part of every persistent tensor (views), with the four evidence tables of every tie in tie
        order as ``t_w`` (assembled: the same values in any storage layout)."""
        n, t = self.n_units, self.n_ties
        return {"u_kind": self.u_kind[:n], "u_channel": self.u_channel[:n], "u_symbol": self.u_symbol[:n],
                "u_arity": self.u_arity[:n], "u_birth": self.u_birth[:n], "u_use": self.u_use[:n],
                "u_alive": self.u_alive[:n], "t_src": self.t_src[:t], "t_dst": self.t_dst[:t],
                "t_form": self.t_form[:t], "t_lag": self.t_lag[:t], "t_w": self.t_w[:, :t],
                "t_use": self.t_use[:t], "t_last": self.t_last[:t], "t_birth": self.t_birth[:t],
                "t_alive": self.t_alive[:t]}

    def digest(self) -> str:
        """sha256 over the live part of every persistent tensor, the evidence as the four tables [4, n_ties, P] in
        tie order (streamed, never assembled whole): it does not depend on how the evidence is stored."""
        h = hashlib.sha256()
        n, t = self.n_units, self.n_ties
        for name in self._UNIT_ARRAYS + ("t_src", "t_dst", "t_form", "t_lag", "t_w", "t_use", "t_last", "t_birth",
                                          "t_alive"):
            h.update(name.encode())
            if name == "t_w":
                for table in range(4):
                    for start in range(0, t, 1 << 18):
                        ids = torch.arange(start, min(t, start + (1 << 18)))
                        h.update(self.w(table, ids).numpy().tobytes())
            else:
                h.update(getattr(self, name)[:n if name.startswith("u_") else t].contiguous().numpy().tobytes())
        return h.hexdigest()

    def physical(self) -> Dict[str, torch.Tensor]:
        """The live part of every persistent tensor as it is stored."""
        n, t, r = self.n_units, self.n_ties, self.n_rows
        out = {name: getattr(self, name)[:n] for name in self._UNIT_ARRAYS}
        out.update({name: getattr(self, name)[:t] for name in self._TIE_ARRAYS})
        out["t_ev"] = self.t_ev[:, :t]
        out.update({name: getattr(self, name)[:r] for name in self._ROW_ARRAYS})
        out["ex"] = self.ex[:, :r]
        return out

    def nbytes(self) -> Dict[str, int]:
        core = sum(t.numel() * t.element_size() for t in self.physical().values())
        if isinstance(self.log, DiskLog):
            return {"core_tensors": core, "log": sum(e.nbytes() for e in self.log.ram.values()),
                    "log_on_disk": self.log.disk_bytes()}
        return {"core_tensors": core, "log": sum(e.nbytes() for e in self.log.values())}

    def max_evidence(self) -> float:
        if not self.n_ties:
            return 0.0
        return float(torch.maximum(self.t_ev[:, :self.n_ties].abs().max(), self.ex[:, :self.n_rows].abs().max()))

    def exact(self) -> bool:
        """True while every stored evidence value is inside the range where grid sums are exact."""
        return self.max_evidence() < EXACT_LIMIT
