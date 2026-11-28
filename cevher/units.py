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
