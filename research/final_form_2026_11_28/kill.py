# -*- coding: utf-8 -*-
"""The kill experiment (docs/architecture/FINAL_FORM.md, section 9).

    python kill.py scan --scan FOLDER [--fractions 100 1000 10000] [--out results/kill.jsonl]      # K2
    python kill.py text [--corpora tr-lessons en-pydoc en-repo] [--bytes-per-char 11]               # K1

K2, composition.

The frame mechanism of ``cevher.mind`` lives the SCAN battery of the yardstick (six splits, the life of m3_scan.py,
50 seeded test commands each) over each candidate store, held to a fraction of what today's Cevahir holds after
the same life (``results/old_scan_memory.json``, walked by m3_scale.py's walker). The reference is the same
mechanism on the unbounded exact store.

Candidate stores:

- ``slots``    checked slots (``cevher.store.Slots``): a tie that finds no room gives way; nothing is invented;
- ``countmin`` a count-min sketch: every tie adds into two cells chosen by hashes and is read as the lesser;
- ``sketch``   a signed count sketch: three cells with signs, read as the median;
- ``cells``    distributed cells (sparse distributed memory with counters, hashed addressing): a key activates two
               cells of 32 counters each, into which the value's code (+1/-1) is added; read by correlation.

A sketch cannot say which values a key holds. The three sketch forms are given that for nothing: an exact index of
(key -> values, last moments) that is not counted in their bytes. What is measured for them is therefore only what
their collisions do to the counts, in the most favourable setting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import yardstick as Y  # noqa: E402
from cevher.mind import Mind  # noqa: E402
from cevher.store import Exact, Slots  # noqa: E402
from cevher.units import MASK, mix  # noqa: E402


class Indexed:
    """What the three sketches share: an exact index of which values each key holds, given for nothing."""

    def __init__(self):
        self.index = {}                                      # key -> {value: last moment}

    def _note(self, key, value, w, t):
        row = self.index.setdefault(key, {})
        if w > 0:
            row[value] = t
        elif self.get(key, value) <= 0:
            row.pop(value, None)

    def bump(self, key, w=1):
        self._inc(key, 0, w)

    def add(self, key, value, w=1, t=0):
        self._inc(key, value, w)
        self._note(key, value, w, t)

    def tot(self, key):
        return self.get(key, 0)

    def vals(self, key):
        row = self.index.get(key)
        if not row:
            return []
        out = [(v, self.get(key, v), t) for v, t in row.items()]
        out = [x for x in out if x[1] > 0]
        out.sort(key=lambda x: (-x[1], -x[2], x[0]))
        return out

    def put_back(self, gone):
        pass

    def size(self):
        return {"ties": sum(len(r) for r in self.index.values())}


class CountMin(Indexed):
    def __init__(self, nbytes, rows=2):
        super().__init__()
        self.m = max(8, nbytes // 4 // rows)
        self.t = [np.zeros(self.m, dtype=np.int32) for _ in range(rows)]

    def _at(self, key, value):
        h = mix(key, value)
        return [mix(h, 0x1000 + i) % self.m for i in range(len(self.t))]

    def _inc(self, key, value, w):
        for row, i in zip(self.t, self._at(key, value)):
            row[i] += w

    def get(self, key, value):
        return int(min(row[i] for row, i in zip(self.t, self._at(key, value))))

    def nbytes(self):
        return 4 * self.m * len(self.t)


class CountSketch(Indexed):
    def __init__(self, nbytes, rows=3):
        super().__init__()
        self.m = max(8, nbytes // 4 // rows)
        self.t = [np.zeros(self.m, dtype=np.int32) for _ in range(rows)]

    def _at(self, key, value):
        h = mix(key, value)
        hs = [mix(h, 0x2000 + i) for i in range(len(self.t))]
        return [(x % self.m, 1 if (x >> 40) & 1 else -1) for x in hs]

    def _inc(self, key, value, w):
        for row, (i, sgn) in zip(self.t, self._at(key, value)):
            row[i] += sgn * w

    def get(self, key, value):
        return max(0, int(np.median([sgn * int(row[i]) for row, (i, sgn) in zip(self.t, self._at(key, value))])))

    def nbytes(self):
        return 4 * self.m * len(self.t)


class Cells(Indexed):
    """Hashed sparse distributed memory with counters: a key activates ``rows`` cells of ``dim`` counters, the
    value's code is added into each, and a count is read as the least correlation with the value's code."""

    def __init__(self, nbytes, rows=2, dim=32):
        super().__init__()
        self.dim, self.rows = dim, rows
        self.m = max(4, nbytes // 2 // dim)
        self.t = np.zeros((self.m, dim), dtype=np.int16)
        self._codes = {}

    def _code(self, value):
        c = self._codes.get(value)
        if c is None:
            bits = mix(value, 0x3000) | (mix(value, 0x3001) << 64)
            c = np.array([1 if (bits >> i) & 1 else -1 for i in range(self.dim)], dtype=np.int16)
            self._codes[value] = c
        return c

    def _inc(self, key, value, w):
        c = self._code(value) * w
        for i in range(self.rows):
            self.t[mix(key, 0x4000 + i) % self.m] += c

    def get(self, key, value):
        c = self._code(value).astype(np.int64)
        return max(0, int(min(int(self.t[mix(key, 0x4000 + i) % self.m].astype(np.int64) @ c)
                              for i in range(self.rows)) // self.dim))

    def nbytes(self):
        return self.t.nbytes


def run(store, split, folder, seed):
    train, lived, life, test = Y.scan_life(folder, split, seed, 1000, 50)
    m = Mind(store)
    t0 = time.time()
    for c, a in life:
        m.watch(list(c), [Y.lower_tr(x) for x in a], ".")
    life_s = time.time() - t0
    got = Counter()
    for c, a in test:
        out = m.answer(list(c))
        got[Y.verdict(" ".join(out[0]) if out else "", " ".join(a))] += 1
    return got, life_s


def scan(args):
    with open(os.path.join(HERE, "results", "old_scan_memory.json"), encoding="utf-8") as f:
        today = json.loads(f.readline())["walked_bytes"]
    makers = [("exact", None, lambda: Exact())] if "exact" in args.stores else []
    for frac in args.fractions:
        n = today // frac
        makers += [m for m in (("slots", frac, lambda n=n: Slots(max_bytes=n, buckets=4)),
                               ("countmin", frac, lambda n=n: CountMin(n)),
                               ("sketch", frac, lambda n=n: CountSketch(n)),
                               ("cells", frac, lambda n=n: Cells(n))) if m[0] in args.stores]
    for name, frac, make in makers:
        total, held, bytes_, secs = Counter(), [], [], 0.0
        for split in Y.SCAN_FILES:
            store = make()
            got, life_s = run(store, split, args.scan, args.seed)
            total.update(got)
            secs += life_s
            held.append(store.size())
            bytes_.append(store.nbytes())
        row = {"what": "kill K2: SCAN on a candidate store", "code": Y.CODE, "store": name, "seed": args.seed,
               "today_walked_bytes": today, "fraction_of_today": frac, "store_bytes": max(bytes_),
               "right": total["right"], "silent": total["silent"], "wrong": total["wrong"],
               "life_s": round(secs, 1), "held": held[0]}
        print(json.dumps(row), flush=True)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


class ExactCounts:
    """The reference for the letters: every tie kept exactly, without bound (a dictionary)."""

    def __init__(self):
        self.d = {}

    def get_many(self, keys):
        get = self.d.get
        return np.fromiter((get(k, 0) for k in keys.tolist()), dtype=np.int64, count=len(keys))

    def add_many(self, keys, w, sign=1):
        d = self.d
        for k, c in zip(keys.tolist(), w.tolist()):
            d[k] = d.get(k, 0) + sign * c
        return None

    def nbytes(self):
        return 12 * len(self.d)

    def size(self):
        return {"ties": len(self.d)}


class MinCounts:
    """Letters in a count-min sketch: no check of the tie's code, two cells a tie, read as the lesser."""

    def __init__(self, nbytes):
        self.m = max(16, nbytes // 8)
        self.a = np.zeros(self.m, dtype=np.uint32)
        self.b = np.zeros(self.m, dtype=np.uint32)

    def _at(self, keys):
        from cevher.units import mix_np
        return ((mix_np(keys, 0x51) % np.uint64(self.m)).astype(np.int64),
                (mix_np(keys, 0x52) % np.uint64(self.m)).astype(np.int64))

    def get_many(self, keys):
        i, j = self._at(keys)
        return np.minimum(self.a[i], self.b[j]).astype(np.int64)

    def add_many(self, keys, w, sign=1):
        i, j = self._at(keys)
        np.add.at(self.a, i, w.astype(np.uint32))
        np.add.at(self.b, j, w.astype(np.uint32))
        return None

    def nbytes(self):
        return 8 * self.m

    def size(self):
        return {"cells": 2 * self.m}


def every_byte(L, block):
    """The bits of each byte of a block (nothing learned) with the chain's value computed for every one of the 256
    bytes: (the bits as claimed, the bits after the 256 values are made to sum to one, the mean of that sum)."""
    import cevher.letters as LT
    from cevher.units import mix_np
    x = np.frombuffer(L.tail + block, dtype=np.uint8)
    start = len(L.tail)
    sym = x[start:].astype(np.int64)
    n = len(sym)
    ctxs = L._contexts(x, start)
    every = np.arange(256, dtype=np.uint64)
    mass, p = np.ones((n, 1)), np.zeros((n, 256))
    for d in range(L.depth, 0, -1):
        g = ctxs[d - 1]
        n_x = L.T.get_many(mix_np(np.repeat(g, 256), np.tile(every, n) + np.uint64(1))).reshape(n, 256)
        tot = np.maximum(L.T.get_many(mix_np(g, LT.TOTAL)), n_x[np.arange(n), sym])
        held = tot > 0
        rec = L.sit[d, np.searchsorted(LT.N_EDGES, tot, side="right")]
        nu = (rec[:, 0] + 0.5) / (rec[:, 1] + 1.0)
        p += mass * np.where(held, 1.0 - nu, 0.0)[:, None] * n_x / np.maximum(tot, 1)[:, None]
        mass = np.where(held, mass[:, 0] * nu, mass[:, 0])[:, None]
    total0 = int(L.zero.sum())
    if total0:
        rec = L.sit[0, min(8, int(np.searchsorted(LT.N_EDGES, total0, side="right")))]
        nu0 = (rec[0] + 0.5) / (rec[1] + 1.0)
        p += mass * (1.0 - nu0) * L.zero[None, :] / total0
        mass = mass * nu0
    p += mass / 256.0
    own = p[np.arange(n), sym]
    return float(-np.log2(own).sum()), float(-np.log2(own / p.sum(axis=1)).sum()), float(p.sum(axis=1).sum())


def text(args):
    """K1: bits per character of the letter grain, prequential over each corpus: on an exact unbounded table, on
    checked counters and on a count-min sketch, the two bounded forms held to ``--bytes-per-char`` bytes for
    every character of the text. A sketch can read a count too high, so what it claims for the 256 bytes need not
    sum to one: on a sample (the first 48 bytes of every eighth block) the value of every byte is computed, and
    the bits are given as claimed and after the values are made to sum to one."""
    sys.path.insert(0, os.path.join(REPO, "research", "living_intelligence_2026_11_24"))
    import bz2
    import lzma
    import zlib
    import corpora
    import cevher.letters as LT
    from cevher.letters import Counts, Letters
    for name in args.corpora:
        s = corpora.stream(name)
        raw = s.encode("utf-8")
        budget = int(args.bytes_per_char * len(s))
        base = {"what": "kill K1: letters on a candidate store", "code": Y.CODE, "corpus": name, "chars": len(s),
                "bzip2": round(8 * len(bz2.compress(raw, 9)) / len(s), 4),
                "xz": round(8 * len(lzma.compress(raw, preset=9)) / len(s), 4),
                "gzip": round(8 * len(zlib.compress(raw, 9)) / len(s), 4)}
        for store, make in (("exact", lambda: ExactCounts()),
                            ("checked", lambda: Counts(max_bytes=budget, buckets=4)),
                            ("countmin", lambda: MinCounts(budget))):
            table = make()
            L = Letters(table, depth=args.depth)
            t0 = time.time()
            bits, at, k = 0.0, 0, 0
            claimed = summed = total = 0.0
            sample = 0
            while at < len(raw):
                size = int(min(LT.BLOCK, max(64, L.lived // 64)))
                block = raw[at:at + size]
                if k % 8 == 0:
                    c, n, t = every_byte(L, block[:48])
                    claimed, summed, total, sample = claimed + c, summed + n, total + t, sample + len(block[:48])
                bits += L._block(block, True)[0]
                at += size
                k += 1
            ratio = len(raw) / len(s)
            row = dict(base, store=store, depth=args.depth, bytes_per_char_allowed=None if store == "exact"
                       else args.bytes_per_char, store_bytes=table.nbytes(),
                       bits_per_char=round(bits / len(s), 4), seconds=round(time.time() - t0, 1),
                       sample_bytes=sample, sample_bits_per_char_claimed=round(claimed / sample * ratio, 4),
                       sample_bits_per_char_summing_to_one=round(summed / sample * ratio, 4),
                       sample_sum_of_256=round(total / sample, 3), held=table.size())
            print(json.dumps(row), flush=True)
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["scan", "text"])
    ap.add_argument("--scan", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fractions", type=int, nargs="*", default=[100, 1000, 10000])
    ap.add_argument("--stores", nargs="*", default=["exact", "slots", "countmin", "sketch", "cells"])
    ap.add_argument("--corpora", nargs="*", default=["tr-lessons", "en-pydoc", "en-repo"])
    ap.add_argument("--bytes-per-char", type=float, default=11.0)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "kill.jsonl"))
    a = ap.parse_args()
    (scan if a.what == "scan" else text)(a)


if __name__ == "__main__":
    main()
