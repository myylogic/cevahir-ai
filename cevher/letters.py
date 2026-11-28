"""The letter grain of the one substance (docs/architecture/FINAL_FORM.md, sections 4 and 5).

A text is lived as bytes of UTF-8. For every byte and every context of the last 1 to ``depth`` bytes, two counts
are written: (context -> the byte) and the context's total. Nothing else is written, so what a text writes is a
function of the text alone, and forgetting a text is subtracting the same counts.

The counts live in checked counters (``Counts``): a slot keeps the code of its tie, so that two ties never add
into one count and what is read back is never more than what was written. A count-min sketch was measured in their
place and rejected: its collisions made the probabilities of the 256 bytes sum to between 3 and 17 at 11 bytes a
character (docs/architecture/FINAL_FORM.md, section 10).

Before a byte is written it is foreseen: the longest context held speaks first, and each context passes on to the
shorter one the belief that what follows is new to it (the novelty event of MODEL.md 2.2), judged by the record of
its situation (MODEL.md 2.3: how long the context is and how much it has lived). The bits paid are the measure of
how well the store knows a text.

Everything is done a block of bytes at a time, with NumPy and no compiled code: a block is foreseen from the store
as it stood before the block, then written. That is what is given up for running without a compiler; blocks are
short while the store is young.
"""
from __future__ import annotations

import numpy as np

from cevher.units import code_of, mix_np

BLOCK = 2048
R_LETTER = code_of("\0role letter context")
TOTAL = np.uint64(0x101)                                     # the reserved value of a context's total
N_EDGES = np.array([2, 3, 4, 5, 8, 16, 32, 64], dtype=np.int64)   # how much lived: 1, 2, 3, 4, 5-7, ... 64 or more
_SALT = 0x2545F4914F6CDD1D
_NEVER = 1 << 62                                             # a count no tie has: a slot that does not give way


class Counts:
    """Checked counters for the letters: one slot a tie, the tie's code and its count (12 bytes), eight slots a
    bucket. The table doubles while more than half of it is used, up to ``max_bytes``; after that, in one breath a
    bucket gives one place to what is new whatever is counted there (its least counted tie gives way to the
    weightiest newcomer: a table that kept only what is counted most could learn no new text once it was full);
    the breath's further newcomers there take only the place of older ties counted no more than they are.
    ``let_go`` counts the ties that gave way, ``not_written`` those that found no place."""
    WAYS = 8
    SLOT = 12

    def __init__(self, max_bytes: int = 64 << 20, buckets: int = 1 << 10):
        self.max_buckets = 1
        while self.max_buckets * 2 * self.WAYS * self.SLOT <= max_bytes:
            self.max_buckets *= 2
        self.nb = min(buckets, self.max_buckets)
        self.now_buckets = self.max_buckets
        self.key = np.zeros(self.nb * self.WAYS, dtype=np.uint64)
        self.cnt = np.zeros(self.nb * self.WAYS, dtype=np.uint32)
        self.used = self.let_go = self.not_written = 0

    def allow(self, nbytes: int) -> None:
        """How large the table may be now (never more than ``max_bytes``)."""
        n = 1
        while n * 2 * self.WAYS * self.SLOT <= nbytes:
            n *= 2
        self.now_buckets = max(self.nb, min(self.max_buckets, n))

    def _grow(self) -> bool:
        if self.nb >= min(self.max_buckets, self.now_buckets):
            return False
        key, cnt = self.key, self.cnt
        nb = self.nb * 2
        self.key = np.zeros(nb * self.WAYS, dtype=np.uint64)
        self.cnt = np.zeros(nb * self.WAYS, dtype=np.uint32)
        at = np.flatnonzero(key != 0)
        to = (mix_np(key[at], _SALT) & np.uint64(nb - 1)).astype(np.int64) * self.WAYS + (at % self.WAYS)
        self.key[to], self.cnt[to] = key[at], cnt[at]
        self.nb = nb
        return True

    def _find(self, keys: np.ndarray):
        base = (mix_np(keys, _SALT) & np.uint64(self.nb - 1)).astype(np.int64) * self.WAYS
        at = base[:, None] + np.arange(self.WAYS, dtype=np.int64)[None, :]
        match = self.key[at] == keys[:, None]
        return base, match.any(axis=1), base + match.argmax(axis=1)

    def get_many(self, keys: np.ndarray) -> np.ndarray:
        _, found, slot = self._find(keys)
        return np.where(found, self.cnt[slot], 0).astype(np.int64)

    def add_many(self, keys: np.ndarray, w: np.ndarray, sign: int = 1, room_only: bool = False):
        """count(key) += sign * w for distinct keys. Returns the ties that gave way as (keys, counts), or None.
        With ``room_only`` (ties put back) a new tie is written only where a slot is free."""
        if not len(keys):
            return None
        if sign < 0:
            _, found, slot = self._find(keys)
            slot = slot[found]
            left = self.cnt[slot].astype(np.int64) - w[found]
            dead = slot[left <= 0]
            self.cnt[slot] = np.maximum(left, 0).astype(np.uint32)
            self.key[dead], self.cnt[dead] = 0, 0
            self.used -= len(dead)
            return None
        while (self.used + len(keys)) * 2 > self.nb * self.WAYS and self._grow():
            pass
        base, found, slot = self._find(keys)
        self.cnt[slot[found]] += w[found].astype(np.uint32)
        rest = np.flatnonzero(~found)
        gone = []
        if len(rest):
            ways = np.arange(self.WAYS, dtype=np.int64)
            order = rest[np.lexsort((-w[rest], base[rest]))]
            b = base[order]
            first = np.r_[True, b[1:] != b[:-1]]
            rank = np.arange(len(b)) - np.maximum.accumulate(np.where(first, np.arange(len(b)), 0))
            fresh = np.zeros(0, dtype=np.int64)              # the slots this breath has written
            for r in range(min(self.WAYS, int(rank.max()) + 1)):
                pick = order[rank == r]
                at = base[pick][:, None] + ways[None, :]
                c = self.cnt[at].astype(np.int64)
                c[self.key[at] == 0] = -1
                if r:
                    c[np.isin(at, fresh)] = _NEVER
                col = c.argmin(axis=1)
                least = c[np.arange(len(pick)), col]
                if room_only:
                    ok = least < 0
                elif r:                                      # room, or an older tie counted no more than the new one
                    ok = least <= w[pick]
                else:                                        # room, or the least counted gives way, whatever it is
                    ok = least < _NEVER
                target = base[pick[ok]] + col[ok]
                fresh = np.concatenate([fresh, target])
                taken = least[ok] >= 0
                self.used += int((~taken).sum())
                if taken.any():
                    gone.append((self.key[target[taken]].copy(), self.cnt[target[taken]].copy()))
                    self.let_go += int(taken.sum())
                self.not_written += int((~ok).sum())
                self.key[target], self.cnt[target] = keys[pick[ok]], w[pick[ok]].astype(np.uint32)
            self.not_written += int((rank >= self.WAYS).sum())
        if not gone:
            return None
        # what gave way is kept as it stood before this breath: without what this breath itself added to it
        k = np.concatenate([g[0] for g in gone])
        c = np.concatenate([g[1] for g in gone]).astype(np.int64)
        order = np.argsort(keys)
        at = np.minimum(np.searchsorted(keys[order], k), len(keys) - 1)
        mine = keys[order][at] == k
        c = c - np.where(mine, w[order][at], 0)
        old = c > 0
        return (k[old], c[old].astype(np.uint32)) if old.any() else None

    def put_back(self, gone) -> None:
        """Ties that gave way are put back where there is room (forgetting what made them give way)."""
        keys, counts = gone
        _, found, _ = self._find(keys)
        before = self.let_go, self.not_written
        self.add_many(keys[~found], counts[~found].astype(np.int64), room_only=True)
        self.let_go, self.not_written = before[0] - int((~found).sum()), before[1]

    def nbytes(self) -> int:
        return self.nb * self.WAYS * self.SLOT

    def size(self) -> dict:
        return {"ties": self.used, "slots": self.nb * self.WAYS, "let_go": self.let_go,
                "not_written": self.not_written}

    def digest(self) -> int:
        at = self.key != 0
        with np.errstate(over="ignore"):
            return int(mix_np(self.key[at], self.cnt[at].astype(np.uint64)).sum(dtype=np.uint64))

    def state(self) -> dict:
        """The table as it stands: its two arrays (not copies)."""
        return {"key": self.key, "cnt": self.cnt,
                "nb": np.array([self.nb, self.max_buckets, self.let_go, self.not_written], dtype=np.int64)}

    @classmethod
    def from_state(cls, st: dict, allowed: int = 0) -> "Counts":
        """The table put back as it stood (nothing is placed again, so nothing can give way in loading)."""
        nb, max_buckets, let_go, not_written = (int(x) for x in st["nb"])
        self = cls.__new__(cls)
        self.max_buckets, self.nb = max_buckets, nb
        self.now_buckets = max_buckets
        if allowed:
            self.allow(allowed)
        self.key, self.cnt = st["key"], st["cnt"]
        self.used, self.let_go, self.not_written = int(np.count_nonzero(self.key)), let_go, not_written
        return self


class Letters:
    """The letter grain: contexts of the last 1 to ``depth`` bytes in a table of counters."""

    def __init__(self, table=None, depth: int = 5):
        self.T = table if table is not None else Counts()
        self.depth = depth
        self.zero = np.zeros(256, dtype=np.int64)            # the empty context: what was met at all
        self.sit = np.zeros((depth + 1, 9, 2), dtype=np.int64)   # a situation's record of the novelty event
        self.tail = b""                                      # the last bytes lived: the contexts go on from them
        self.lived = 0

    def _contexts(self, x: np.ndarray, start: int) -> list:
        """The code of the context of each length before every byte of x[start:]."""
        out = []
        g = np.full(len(x) + 1, R_LETTER, dtype=np.uint64)
        for d in range(1, self.depth + 1):
            nxt = np.empty_like(g)
            nxt[0] = R_LETTER
            nxt[1:] = mix_np(g[:-1], x.astype(np.uint64) + np.uint64(d << 16))
            g = nxt
            out.append(g[start:-1])
        return out

    def writes(self, data: bytes, tail: bytes):
        """What living ``data`` after ``tail`` writes: the distinct ties with their counts (a function of the text
        alone)."""
        x = np.frombuffer(tail + data, dtype=np.uint8)
        start = len(tail)
        sym = x[start:].astype(np.uint64)
        keys = []
        for g in self._contexts(x, start):
            keys += [mix_np(g, sym + np.uint64(1)), mix_np(g, TOTAL)]
        keys, counts = np.unique(np.concatenate(keys), return_counts=True)
        return keys, counts.astype(np.int64), sym.astype(np.int64)

    def live(self, text: str, learn: bool = True, gone: list = None) -> float:
        """Foresee and (if ``learn``) write a text. Returns the bits paid. Where a list is given, the ties that
        gave way are appended to it with the place in the text where they did (a ledger keeps them)."""
        data = text.encode("utf-8", "surrogatepass")
        bits, at = 0.0, 0
        while at < len(data):                                # short blocks while the store is young
            size = int(min(BLOCK, max(64, self.lived // 64))) if learn else BLOCK
            b, g = self._block(data[at:at + size], learn)
            bits += b
            if g is not None and gone is not None:
                gone.append((at, at + size, g))
            at += size
        return bits

    def forget(self, text: str, tail: bytes, gone: list = ()) -> None:
        """Take back what living ``text`` after ``tail`` wrote, the last first, and put back each tie that gave way
        at the place where it did."""
        data = text.encode("utf-8", "surrogatepass")

        def take(a: int, b: int) -> None:
            for at in range(a, b, 1 << 16):
                end = min(b, at + (1 << 16))
                keys, counts, sym = self.writes(data[at:end], (tail + data[:at])[-self.depth:])
                self.T.add_many(keys, counts, sign=-1)
                np.subtract.at(self.zero, sym, 1)
                self.lived -= len(sym)
        end = len(data)
        for a, b, g in reversed(list(gone)):
            take(b, end)
            take(a, b)
            self.T.put_back(g)
            end = a
        take(0, end)

    def _block(self, block: bytes, learn: bool):
        x = np.frombuffer(self.tail + block, dtype=np.uint8)
        start = len(self.tail)
        sym = x[start:].astype(np.uint64)
        n_new = len(sym)
        ctxs = self._contexts(x, start)
        mass = np.ones(n_new)
        p = np.zeros(n_new)
        seen, keys = [], []
        for d in range(self.depth, 0, -1):                   # the longest context first
            g = ctxs[d - 1]
            pair, tot = mix_np(g, sym + np.uint64(1)), mix_np(g, TOTAL)
            n_x, n = self.T.get_many(pair), self.T.get_many(tot)
            n = np.maximum(n, n_x)
            held = n > 0
            nb = np.searchsorted(N_EDGES, n, side="right")
            rec = self.sit[d, nb]
            nu = (rec[:, 0] + 0.5) / (rec[:, 1] + 1.0)       # the belief that what follows is new to the context
            p += mass * np.where(held, (1.0 - nu) * n_x / np.maximum(n, 1), 0.0)
            mass = np.where(held, mass * nu, mass)
            seen.append((d, n_x, held, nb))
            keys += [pair, tot]
        total0 = int(self.zero.sum())                        # the empty context, then a byte never met
        n0b = min(8, int(np.searchsorted(N_EDGES, total0, side="right")))
        if total0:
            rec = self.sit[0, n0b]
            nu0 = (rec[0] + 0.5) / (rec[1] + 1.0)
            p += mass * (1.0 - nu0) * self.zero[sym.astype(np.int64)] / total0
            mass = mass * nu0
        p += mass / 256.0
        bits = float(-np.log2(np.maximum(p, 1e-300)).sum())
        gone = None
        if learn:
            for d, n_x, held, nb in seen:                    # each situation's record of the novelty event
                np.add.at(self.sit[d], (nb[held], 1), 1)
                np.add.at(self.sit[d], (nb[held & (n_x == 0)], 0), 1)
            if total0:
                self.sit[0, n0b, 1] += n_new
                self.sit[0, n0b, 0] += int((self.zero[sym.astype(np.int64)] == 0).sum())
            np.add.at(self.zero, sym.astype(np.int64), 1)
            k, c = np.unique(np.concatenate(keys), return_counts=True)
            gone = self.T.add_many(k, c.astype(np.int64))
            self.lived += n_new
        self.tail = (self.tail + block)[-self.depth:]
        return bits, gone

    def size(self) -> dict:
        return dict(self.T.size(), lived=self.lived, bytes=self.T.nbytes())
