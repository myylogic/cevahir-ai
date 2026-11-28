# -*- coding: utf-8 -*-
"""Units and their codes (docs/architecture/FINAL_FORM.md, 3.1).

A unit is anything the mind tells apart: a word or mark, a context, a frame, a skeleton, a role, an origin. Every
unit has a code of 64 bits, computed from what the unit is made of and never stored per occurrence: a word's code
from its spelling, a compound unit's code from the codes of its parts, in order. The same functions exist for
single codes (Python integers) and for arrays (NumPy, wrapping 64-bit arithmetic); they give the same numbers.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

MASK = (1 << 64) - 1
TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)          # Zihin's tokens: words and marks
FINAL = (".", "?", "!", "…")                             # closing marks: the tone of an utterance, not its body
LINE = 0.5 + 1.0 / 12.0                                  # the speaking line e* (MODEL.md 2.4)

_C1, _C2, _C3 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB


def lower_tr(text: str) -> str:
    """Turkish lower case (I -> ı, İ -> i), as Zihin's text adapter."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def tokens(text: str) -> list:
    """The units of an utterance: its words and marks, lower-cased."""
    return TOKEN.findall(lower_tr(text))


def body(toks: list) -> tuple:
    """(the tokens without the closing marks, the closing mark or "")."""
    n = len(toks)
    while n and toks[n - 1] in FINAL:
        n -= 1
    return toks[:n], (toks[n] if n < len(toks) else "")


def code_of(spelling: str) -> int:
    """The code of a unit that has a spelling: 64 bits of its BLAKE2b digest (never 0, which no unit has)."""
    return int.from_bytes(hashlib.blake2b(spelling.encode("utf-8"), digest_size=8).digest(), "little") or 1


def mix(a: int, b: int) -> int:
    """The code of the pair (a, b), in this order (the finaliser of SplitMix64 over a combined word)."""
    x = (a ^ ((b + _C1 + ((a << 6) & MASK) + (a >> 2)) & MASK)) & MASK
    x = ((x ^ (x >> 30)) * _C2) & MASK
    x = ((x ^ (x >> 27)) * _C3) & MASK
    return x ^ (x >> 31)


def fold(codes, seed: int) -> int:
    """The code of a sequence of codes under a role."""
    h = seed
    for c in codes:
        h = mix(h, c)
    return h


def mix_np(a: np.ndarray, b) -> np.ndarray:
    """``mix`` over arrays of uint64 (b an array or one code)."""
    a = a.astype(np.uint64, copy=False)
    b = np.asarray(b, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = a ^ (b + np.uint64(_C1) + (a << np.uint64(6)) + (a >> np.uint64(2)))
        x = (x ^ (x >> np.uint64(30))) * np.uint64(_C2)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(_C3)
        return x ^ (x >> np.uint64(31))


def rule(w_plus: float, w: float) -> float:
    """The one rule: e = (W+ + 1/2) / (W + 1)."""
    return (w_plus + 0.5) / (w + 1.0)


class Spellings:
    """What is kept per word type: its spelling, so that a code the store brings back can be said. Packed: the codes
    sorted, a place and a length for each, and the bytes of all the words (about 16 bytes and the word's length a
    type). Words met since the last packing wait in a dictionary and are packed when it has grown."""
    PENDING = 50000

    def __init__(self, most: int = 2_000_000):
        self.codes = np.zeros(0, dtype=np.uint64)
        self.offs = np.zeros(1, dtype=np.int64)
        self.blob = b""
        self.new: dict = {}
        self.most = most

    def __len__(self) -> int:
        return len(self.codes) + len(self.new)

    def get(self, code: int, default=None):
        s = self.new.get(code)
        if s is not None:
            return s
        i = int(np.searchsorted(self.codes, np.uint64(code)))
        if i < len(self.codes) and int(self.codes[i]) == code:
            return self.blob[int(self.offs[i]):int(self.offs[i + 1])].decode("utf-8", "surrogatepass")
        return default

    def __getitem__(self, code: int) -> str:
        s = self.get(code)
        if s is None:
            raise KeyError(code)
        return s

    def setdefault(self, code: int, spelling: str) -> None:
        if code in self.new or len(self) >= self.most:
            return
        i = int(np.searchsorted(self.codes, np.uint64(code)))
        if i < len(self.codes) and int(self.codes[i]) == code:
            return
        self.new[code] = spelling
        if len(self.new) >= self.PENDING:
            self.pack()

    def pack(self) -> None:
        """The words met since the last packing go into the packed table, each where its code belongs. The old
        table is cut where they go in, never unpacked: the work and the memory are those of the new words."""
        if not self.new:
            return
        codes = np.fromiter(self.new, dtype=np.uint64, count=len(self.new))
        order = np.argsort(codes, kind="stable")
        codes = codes[order]
        spelt = list(self.new.values())
        words = [spelt[i].encode("utf-8", "surrogatepass") for i in order.tolist()]
        where = np.searchsorted(self.codes, codes)
        lengths = np.insert(np.diff(self.offs), where, np.array([len(w) for w in words], dtype=np.int64))
        old, parts, at = memoryview(self.blob), [], 0
        for cut, w in zip(self.offs[where].tolist(), words):
            if cut > at:
                parts.append(old[at:cut])
                at = cut
            parts.append(w)
        parts.append(old[at:])
        self.blob = b"".join(parts)
        self.codes = np.insert(self.codes, where, codes)
        self.offs = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        self.new = {}

    def state(self) -> dict:
        self.pack()
        return {"codes": self.codes, "offs": self.offs, "blob": np.frombuffer(self.blob, dtype=np.uint8)}

    @classmethod
    def from_state(cls, st: dict, most: int = 2_000_000) -> "Spellings":
        self = cls(most)
        self.codes, self.offs, self.blob = st["codes"], st["offs"], st["blob"].tobytes()
        return self

    def nbytes(self) -> int:
        return self.codes.nbytes + self.offs.nbytes + len(self.blob) + 200 * len(self.new)
