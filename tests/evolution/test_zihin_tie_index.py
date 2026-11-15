# -*- coding: utf-8 -*-
"""Zihin store: the tie index as a long-term store (sorted arrays per form and lag) and a day store merged in sleep
or at save. Under any sequence of additions, removals, merges and pickling it answers exactly as a dict would."""
from __future__ import annotations

import pickle
import random

import numpy as np

from zihin.store import TieIndex, tie_key


def test_tie_index_answers_as_a_dict_under_random_life():
    rng = random.Random(0)
    index, ref = TieIndex(), {}
    next_id = 0
    for step in range(6000):
        op = rng.random()
        src, dst = rng.randrange(400), rng.randrange(400)
        form, lag = rng.choice([(0, 0), (0, 1), (0, 2), (1, 0), (1, 3), (2, 0)])
        key = (src, dst, form, lag)
        if op < 0.55 and key not in ref:
            index.add(*key, next_id)
            ref[key] = next_id
            next_id += 1
        elif op < 0.65 and ref:
            gone = rng.choice(list(ref))
            index.remove(*gone)
            del ref[gone]
        elif op < 0.68:
            index.merge()
        elif op < 0.70:
            index = pickle.loads(pickle.dumps(index))
        assert index.find(*key) == ref.get(key, -1)
        if step % 500 == 0:
            for k, tie in ref.items():
                assert index.find(*k) == tie
            assert len(index) == len(ref)
            assert dict(index.items()) == {tie_key(*k): t for k, t in ref.items()}


def test_built_at_once_equals_built_by_adding():
    rng = np.random.default_rng(1)
    n = 3000
    src, dst = rng.integers(0, 500, n), rng.integers(0, 500, n)
    form = rng.integers(0, 3, n)
    lag = np.where(form == 0, rng.integers(0, 3, n), 0)
    seen, alive = set(), np.zeros(n, dtype=bool)
    for i in range(n):
        key = (int(src[i]), int(dst[i]), int(form[i]), int(lag[i]))
        if key not in seen and rng.random() < 0.9:
            seen.add(key)
            alive[i] = True
    whole = TieIndex.from_arrays(src, dst, form, lag, alive)
    added = TieIndex()
    for i in np.flatnonzero(alive).tolist():
        added.add(int(src[i]), int(dst[i]), int(form[i]), int(lag[i]), i)
    assert whole == added and len(whole) == int(alive.sum())
