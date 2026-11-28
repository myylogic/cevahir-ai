# -*- coding: utf-8 -*-
"""The substance: tables of ties (docs/architecture/FINAL_FORM.md, 3.2 and 3.3).

A tie is (key, value) -> W: the evidence, counted as lived, that ``value`` stood in the relation ``key`` names. The
total of a key is kept beside its ties. Every store here offers the same four calls, on 64-bit codes:

    add(key, value, w=1, t=0)   W(key, value) += w, and the tie's last moment becomes t (if w > 0)
    bump(key, w=1)              W(key) += w, the key's total
    get(key, value)             W(key, value)
    tot(key)                    W(key)
    vals(key)                   the ties of a key as (value, W, last moment), the most evident first

``Exact`` is the unbounded reference (a dictionary); the bounded forms are compared with it in the kill experiment
(research/final_form_2026_11_28/kill.py). ``Slots`` is the form the final form uses.
"""
from __future__ import annotations

import numpy as np

from cevher.units import MASK, mix, mix_np

_SALT = (0x243F6A8885A308D3, 0x13198A2E03707344, 0xA4093822299F31D0, 0x082EFA98EC4E6C89)


class Exact:
    """The reference: every tie kept exactly, without bound."""

    def __init__(self):
        self.ties = {}                                       # key -> {value: [W, last moment]}
        self.totals = {}                                     # key -> W

    def add(self, key, value, w=1, t=0):
        row = self.ties.get(key)
        if row is None:
            row = self.ties[key] = {}
        cell = row.get(value)
        if cell is None:
            if w > 0:
                row[value] = [w, t]
            elif not row:
                del self.ties[key]
        else:
            cell[0] += w
            if w > 0:
                cell[1] = t
            if cell[0] <= 0:
                del row[value]
                if not row:
                    del self.ties[key]

    def put_back(self, gone) -> None:
        pass

    def bump(self, key, w=1):
        n = self.totals.get(key, 0) + w
        if n <= 0:
            self.totals.pop(key, None)
        else:
            self.totals[key] = n

    def get(self, key, value):
        row = self.ties.get(key)
        cell = row.get(value) if row else None
        return cell[0] if cell else 0

    def tot(self, key):
        return self.totals.get(key, 0)

    def vals(self, key):
        row = self.ties.get(key)
        if not row:
            return []
        return sorted(((v, c[0], c[1]) for v, c in row.items()), key=lambda x: (-x[1], -x[2], x[0]))

    def add_many(self, keys, values, w, t=0):
        for k, v, c in zip(keys.tolist(), values.tolist(), w.tolist()):
            if v:
                self.add(k, v, c, t)
            else:
                self.bump(k, c)
        return None

    def sub_many(self, keys, values, w):
        for k, v, c in zip(keys.tolist(), values.tolist(), w.tolist()):
            if v:
                self.add(k, v, -c, 0)
            else:
                self.bump(k, -c)

    def nbytes(self):
        return 24 * sum(len(r) for r in self.ties.values()) + 12 * len(self.totals)

    def size(self):
        return {"ties": sum(len(r) for r in self.ties.values()), "keys": len(self.totals)}

    def digest(self) -> int:
        """A digest of the substance: every tie and total with its count, wherever it sits (the last moments, which
        depend on the order of life, are not part of it)."""
        d = 0
        for key, row in self.ties.items():
            for v, c in row.items():
                d = (d + mix(mix(key, v), c[0] & MASK)) & MASK
        for key, n in self.totals.items():
            d = (d + mix(mix(key, 0), n & MASK)) & MASK
        return d


class Slots:
    """The bounded form: checked slots (docs/architecture/FINAL_FORM.md, 3.3).

    The table is ``buckets`` x ``WAYS`` slots; a slot holds a key, a value, a count and the tie's last moment (20
    bytes). All the ties of one key sit in the bucket its code names, with its total (the tie whose value is 0), so
    reading a key gives its values without asking about every unit. Two keys never add into one count.

    The table doubles while more than a third of it is used, up to ``max_bytes`` (and to what ``allow`` permits). A tie that finds its bucket full
    waits beside the table (a small overflow, at most one tie for every ``SPILL`` slots) until the table doubles.
    Only when the table can no longer grow and the overflow is full does a tie give way: the one with the least
    evidence in the bucket, the oldest among equals. ``let_go`` counts those; ``add`` returns the tie that gave
    way, so that a ledger can put it back. While nothing has given way the table is exactly the multiset of what
    was written, and its digest equals the exact store's. Collisions cost ties forgotten, never evidence invented.
    """
    WAYS = 8
    SLOT = 20                                                # bytes a slot
    LOAD = 3                                                 # the table doubles when a third of it is used
    SPILL = 32                                               # the overflow holds one tie for every 32 slots

    def __init__(self, max_bytes: int = 256 << 20, buckets: int = 1 << 10):
        # a bucket costs its slots, its share of the filter (a byte a slot) and its share of the overflow
        per_bucket = self.WAYS * (self.SLOT + 1) + 120 * self.WAYS // self.SPILL
        self.max_buckets = 1
        while self.max_buckets * 2 * per_bucket <= max_bytes:
            self.max_buckets *= 2
        self.nb = min(buckets, self.max_buckets)
        self.now_buckets = self.max_buckets                  # how far the table may grow now (``allow``)
        self._alloc(self.nb)
        self.used = 0
        self.let_go = 0
        self.not_kept = 0
        self.spill: dict = {}                                # key -> {value: [W, last moment]}
        self.spilled = 0
        self.filter = bytearray(1)
        self._refilter()

    def _alloc(self, nb: int) -> None:
        n = nb * self.WAYS
        self.key = np.zeros(n, dtype=np.uint64)
        self.val = np.zeros(n, dtype=np.uint64)
        self.cnt = np.zeros(n, dtype=np.int32)
        self.tim = np.zeros(n, dtype=np.uint32)

    # -- a filter of the keys present, so that asking about a key the table does not hold costs almost nothing ----
    def _refilter(self) -> None:
        bits = max(1 << 16, self.nb * self.WAYS * 8)
        self.fmask = bits - 1
        flt = np.zeros(bits // 8, dtype=np.uint8)
        k = self.key[self.key != 0]
        if self.spill:
            k = np.concatenate([k, np.fromiter(self.spill, dtype=np.uint64, count=len(self.spill))])
        if len(k):
            h = (mix_np(k, _SALT[1]) & np.uint64(self.fmask)).astype(np.int64)
            np.bitwise_or.at(flt, h >> 3, (1 << (h & 7)).astype(np.uint8))
        self.filter = bytearray(flt.tobytes())

    def _seen(self, key: int) -> bool:
        h = mix(key, _SALT[1]) & self.fmask
        return bool(self.filter[h >> 3] & (1 << (h & 7)))

    def _mark(self, key: int) -> None:
        h = mix(key, _SALT[1]) & self.fmask
        self.filter[h >> 3] |= 1 << (h & 7)

    def _base(self, key: int) -> int:
        return (mix(key, _SALT[0]) & (self.nb - 1)) * self.WAYS

    def _room_beside(self) -> bool:
        return self.spilled < min(16384, max(64, self.nb * self.WAYS // self.SPILL))

    # -- growth -----------------------------------------------------------------------------------------------------
    def allow(self, nbytes: int) -> None:
        """How large the table may be now (never more than ``max_bytes``): the mind keeps a bounded number of
        bytes for every character it has lived."""
        per_bucket = self.WAYS * (self.SLOT + 1) + 120 * self.WAYS // self.SPILL
        n = 1
        while n * 2 * per_bucket <= nbytes:
            n *= 2
        self.now_buckets = max(self.nb, min(self.max_buckets, n))

    def _grow(self) -> bool:
        if self.nb >= min(self.max_buckets, self.now_buckets):
            return False
        old = (self.key, self.val, self.cnt, self.tim)
        nb = self.nb * 2
        self._alloc(nb)
        at = np.flatnonzero(old[0] != 0)
        if len(at):
            b = (mix_np(old[0][at], _SALT[0]) & np.uint64(nb - 1)).astype(np.int64)
            to = b * self.WAYS + (at % self.WAYS)            # a bucket splits in two; a tie keeps its place in it
            self.key[to], self.val[to], self.cnt[to], self.tim[to] = old[0][at], old[1][at], old[2][at], old[3][at]
        self.nb = nb
        waiting, self.spill, self.spilled = self.spill, {}, 0
        self._refilter()
        for key, row in waiting.items():                     # what waited beside the table goes in where it fits
            for value, (c, t) in row.items():
                self._place(key, value, c, t)
        return True

    def _place(self, key: int, value: int, c: int, t: int):
        """A new tie. A key holds its total and at most ``WAYS - 1`` values: a further value takes the place of the
        least evident of them (the oldest among equals) if it is counted no less, and is not written otherwise.
        Else the tie goes into its bucket; where other keys fill the bucket, beside the table; and where there is
        no room beside it either, in place of the least evident tie of the bucket. Returns the tie that gave way."""
        self._mark(key)
        base = self._base(key)
        keys = self.key[base:base + self.WAYS]
        row = self.spill.get(key) if self.spill else None
        if value:
            own = np.flatnonzero((keys == key) & (self.val[base:base + self.WAYS] != 0))
            beside = [v for v in row if v] if row else []
            if len(own) + len(beside) >= self.WAYS - 1:
                cand = [(int(self.cnt[base + i]), int(self.tim[base + i]), 0, int(i)) for i in own.tolist()]
                cand += [(row[v][0], row[v][1], 1, v) for v in beside]
                wc, wt, where, at = min(cand)
                if wc > c:
                    self.not_kept += 1
                    return None
                self.let_go += 1
                if where:
                    del row[at]
                    row[value] = [c, t]
                    return (key, at, wc, wt)
                i = base + at
                gone = (key, int(self.val[i]), wc, wt)
                self.val[i], self.cnt[i], self.tim[i] = value, c, t
                return gone
        free = np.flatnonzero(keys == 0)
        if len(free):
            i = base + int(free[0])
            self.key[i], self.val[i], self.cnt[i], self.tim[i] = key, value, c, t
            self.used += 1
            return None
        if self._room_beside():
            self.spill.setdefault(key, {})[value] = [c, t]
            self.spilled += 1
            return None
        # no room anywhere: the least evident key of the bucket gives way whole (its total and its values), if it
        # is counted no more than the new tie; the new tie is not written otherwise
        vals_, cnts = self.val[base:base + self.WAYS], self.cnt[base:base + self.WAYS]
        totals = [(int(cnts[i]), int(self.tim[base + i]), int(keys[i])) for i in range(self.WAYS)
                  if int(keys[i]) != key and int(vals_[i]) == 0]
        if not totals:                                       # values whose total waits elsewhere: the least of them
            totals = [(int(cnts[i]), int(self.tim[base + i]), int(keys[i])) for i in range(self.WAYS)
                      if int(keys[i]) != key]
        if not totals or min(totals)[0] > c:
            self.not_kept += 1
            return None
        victim = min(totals)[2]
        gone = []
        for i in range(self.WAYS):
            if int(keys[i]) == victim:
                gone.append((victim, int(vals_[i]), int(cnts[i]), int(self.tim[base + i])))
                self.key[base + i], self.val[base + i], self.cnt[base + i], self.tim[base + i] = 0, 0, 0, 0
                self.used -= 1
        self.let_go += len(gone)
        i = base + int(np.flatnonzero(self.key[base:base + self.WAYS] == 0)[0])
        self.key[i], self.val[i], self.cnt[i], self.tim[i] = key, value, c, t
        self.used += 1
        return gone

    # -- one tie at a time --------------------------------------------------------------------------------------------
    def add(self, key, value, w=1, t=0):
        """W(key, value) += w. Returns the ties that gave way to make room, each as (key, value, W, last moment),
        or None: the ledger keeps them, so that forgetting what was just lived puts them back."""
        if not self._seen(key):
            return self._new(key, value, w, t) if w > 0 else None
        base = self._base(key)
        hit = np.flatnonzero((self.key[base:base + self.WAYS] == key) & (self.val[base:base + self.WAYS] == value))
        if len(hit):
            i = base + int(hit[0])
            c = int(self.cnt[i]) + w
            if c <= 0:
                self.key[i], self.val[i], self.cnt[i], self.tim[i] = 0, 0, 0, 0
                self.used -= 1
            else:
                self.cnt[i] = c
                if w > 0:
                    self.tim[i] = t
            return None
        row = self.spill.get(key) if self.spill else None
        if row is not None and value in row:
            cell = row[value]
            cell[0] += w
            if w > 0:
                cell[1] = t
            if cell[0] <= 0:
                del row[value]
                self.spilled -= 1
                if not row:
                    del self.spill[key]
            return None
        return self._new(key, value, w, t) if w > 0 else None

    def _new(self, key, value, w, t):
        if (self.used + self.spilled) * self.LOAD > self.nb * self.WAYS or not self._room_beside():
            self._grow()
        gone = self._place(key, value, w, t)
        return [gone] if isinstance(gone, tuple) else gone

    def put_back(self, gone) -> None:
        """A tie that gave way is put back as it was (forgetting what made it give way)."""
        key, value, c, t = gone
        if not self.get(key, value):
            before = self.let_go, self.not_kept
            self._place(key, value, c, t)
            self.let_go, self.not_kept = max(0, before[0] - 1), before[1]

    def bump(self, key, w=1):
        return self.add(key, 0, w, 0)

    def get(self, key, value):
        if not self._seen(key):
            return 0
        base = self._base(key)
        hit = np.flatnonzero((self.key[base:base + self.WAYS] == key) & (self.val[base:base + self.WAYS] == value))
        if len(hit):
            return int(self.cnt[base + int(hit[0])])
        row = self.spill.get(key) if self.spill else None
        return row[value][0] if row is not None and value in row else 0

    def tot(self, key):
        return self.get(key, 0)

    def vals(self, key):
        if not self._seen(key):
            return []
        base = self._base(key)
        hit = np.flatnonzero((self.key[base:base + self.WAYS] == key) & (self.val[base:base + self.WAYS] != 0))
        out = [(int(self.val[base + i]), int(self.cnt[base + i]), int(self.tim[base + i])) for i in hit.tolist()]
        row = self.spill.get(key) if self.spill else None
        if row is not None:
            out += [(v, c[0], c[1]) for v, c in row.items() if v != 0]
        out.sort(key=lambda x: (-x[1], -x[2], x[0]))
        return out

    # -- many ties at once (a text read) --------------------------------------------------------------------------------
    def _find(self, keys: np.ndarray, values: np.ndarray):
        ways = np.arange(self.WAYS, dtype=np.int64)
        base = (mix_np(keys, _SALT[0]) & np.uint64(self.nb - 1)).astype(np.int64) * self.WAYS
        at = base[:, None] + ways[None, :]
        match = (self.key[at] == keys[:, None]) & (self.val[at] == values[:, None])
        return base, match.any(axis=1), base + match.argmax(axis=1)

    def add_many(self, keys: np.ndarray, values: np.ndarray, w: np.ndarray, t: int = 0):
        """W(key, value) += w for arrays of distinct (key, value) pairs with w > 0 (totals as value 0). Returns the
        ties that gave way, as four arrays, or None."""
        if not len(keys):
            return None
        while (self.used + self.spilled + len(keys)) * self.LOAD > self.nb * self.WAYS and self._grow():
            pass
        base, found, slot = self._find(keys, values)
        self.cnt[slot[found]] += w[found].astype(np.int32)
        self.tim[slot[found]] = t
        rest = np.flatnonzero(~found)
        gone: list = []
        if len(rest) and self.spill:                         # ties that wait beside the table
            keep = []
            for i in rest.tolist():
                row = self.spill.get(int(keys[i]))
                if row is not None and int(values[i]) in row:
                    cell = row[int(values[i])]
                    cell[0] += int(w[i])
                    cell[1] = t
                else:
                    keep.append(i)
            rest = np.array(keep, dtype=np.int64)
        if len(rest):                                        # new ties: one a bucket at a time, the weightiest first
            ways = np.arange(self.WAYS, dtype=np.int64)
            order = rest[np.lexsort((-w[rest].astype(np.int64), base[rest]))]
            b = base[order]
            first = np.r_[True, b[1:] != b[:-1]]
            rank = np.arange(len(b)) - np.maximum.accumulate(np.where(first, np.arange(len(b)), 0))
            h = (mix_np(keys[rest], _SALT[1]) & np.uint64(self.fmask)).astype(np.int64)
            flt = np.frombuffer(self.filter, dtype=np.uint8)
            np.bitwise_or.at(flt, h >> 3, (1 << (h & 7)).astype(np.uint8))
            for r in range(int(rank.max()) + 1):
                pick = order[rank == r]
                at = base[pick][:, None] + ways[None, :]
                free = self.key[at] == 0
                has = free.any(axis=1)
                target = base[pick[has]] + free[has].argmax(axis=1)
                self.used += len(target)
                self.key[target], self.val[target] = keys[pick[has]], values[pick[has]]
                self.cnt[target], self.tim[target] = w[pick[has]].astype(np.int32), t
                for i in pick[~has].tolist():                # the bucket is full: beside the table, or give way
                    out = self._place(int(keys[i]), int(values[i]), int(w[i]), t)
                    if isinstance(out, tuple):
                        gone.append(out)
                    elif out:
                        gone.extend(out)
        if not gone:
            return None
        # what gave way is kept as it stood before this breath: without what this breath itself added to it (a
        # tie written and let go in one breath was never kept)
        added = dict(zip(zip(keys.tolist(), values.tolist()), w.tolist()))
        gone = [(k, v, c - added.get((k, v), 0), t) for k, v, c, t in gone]
        gone = [x for x in gone if x[2] > 0]
        if not gone:
            return None
        g = np.array(gone, dtype=np.uint64)
        return g[:, 0].copy(), g[:, 1].copy(), g[:, 2].astype(np.int32), g[:, 3].astype(np.uint32)

    def sub_many(self, keys: np.ndarray, values: np.ndarray, w: np.ndarray) -> None:
        """W(key, value) -= w for arrays of distinct pairs; a tie whose count returns to zero is removed."""
        if not len(keys):
            return
        base, found, slot = self._find(keys, values)
        slot = slot[found]
        self.cnt[slot] -= w[found].astype(np.int32)
        dead = slot[self.cnt[slot] <= 0]
        self.key[dead], self.val[dead], self.cnt[dead], self.tim[dead] = 0, 0, 0, 0
        self.used -= len(dead)
        if self.spill:
            for i in np.flatnonzero(~found).tolist():
                self.add(int(keys[i]), int(values[i]), -int(w[i]), 0)

    def put_back_many(self, gone) -> None:
        for k, v, c, t in zip(gone[0].tolist(), gone[1].tolist(), gone[2].tolist(), gone[3].tolist()):
            self.put_back((k, v, c, t))

    def get_many(self, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
        base, found, slot = self._find(keys, values)
        out = np.where(found, self.cnt[slot], 0)
        if self.spill:
            for i in np.flatnonzero(~found).tolist():
                out[i] = self.get(int(keys[i]), int(values[i]))
        return out

    # -- what it holds ----------------------------------------------------------------------------------------------------
    def nbytes(self):
        return self.nb * self.WAYS * self.SLOT + len(self.filter) + 120 * self.spilled

    def size(self):
        ties = int(((self.key != 0) & (self.val != 0)).sum()) + sum(1 for r in self.spill.values() for v in r if v)
        return {"ties": ties, "keys": self.used + self.spilled - ties, "slots": self.nb * self.WAYS,
                "beside": self.spilled, "let_go": self.let_go, "not_kept": self.not_kept}

    def _all(self):
        at = np.flatnonzero(self.key != 0)
        k, v, c, t = self.key[at], self.val[at], self.cnt[at], self.tim[at]
        if self.spill:
            rows = [(key, value, cell[0], cell[1]) for key, row in self.spill.items() for value, cell in row.items()]
            g = np.array(rows, dtype=np.uint64)
            k, v = np.concatenate([k, g[:, 0]]), np.concatenate([v, g[:, 1]])
            c, t = np.concatenate([c, g[:, 2].astype(np.int32)]), np.concatenate([t, g[:, 3].astype(np.uint32)])
        return k, v, c, t

    def digest(self) -> int:
        k, v, c, _ = self._all()
        with np.errstate(over="ignore"):
            d = mix_np(mix_np(k, v), c.astype(np.int64).astype(np.uint64))
            return int(d.sum(dtype=np.uint64))

    def state(self) -> dict:
        k, v, c, t = self._all()
        return {"key": k, "val": v, "cnt": c, "tim": t,
                "nb": np.array([self.nb, self.max_buckets, self.let_go, self.not_kept], dtype=np.int64)}

    @classmethod
    def from_state(cls, st: dict) -> "Slots":
        nb, max_buckets, let_go, not_kept = (int(x) for x in st["nb"])
        self = cls.__new__(cls)
        self.max_buckets, self.nb = max_buckets, nb
        self.now_buckets = max_buckets
        self._alloc(nb)
        self.used, self.let_go, self.not_kept, self.spill, self.spilled = 0, 0, 0, {}, 0
        self.filter = bytearray(1)
        self._refilter()
        self.add_many(st["key"], st["val"], st["cnt"].astype(np.int64), 0)
        base, found, slot = self._find(st["key"], st["val"])
        self.tim[slot[found]] = st["tim"][found]
        if self.spill:
            for i in np.flatnonzero(~found).tolist():
                row = self.spill.get(int(st["key"][i]))
                if row is not None and int(st["val"][i]) in row:
                    row[int(st["val"][i])][1] = int(st["tim"][i])
        self.let_go, self.not_kept = let_go, not_kept
        return self
