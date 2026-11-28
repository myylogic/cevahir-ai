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
            row[value] = [w, t]
        else:
            cell[0] += w
            if w > 0:
                cell[1] = t
            if cell[0] <= 0:
                del row[value]
                if not row:
                    del self.ties[key]

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

    def set_time(self, key, value, t):
        row = self.ties.get(key)
        if row and value in row:
            row[value][1] = t

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
