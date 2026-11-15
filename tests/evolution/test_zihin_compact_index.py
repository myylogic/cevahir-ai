# -*- coding: utf-8 -*-
"""Zihin store: compact indices. A tie's key is one packed integer and an adjacency list is an array of 32-bit ids;
both hold exactly what the 4-tuple keys and lists of ids held, in the same order. Minds saved before are converted
on load to the same indices."""
from __future__ import annotations

import pickle
from array import array

from zihin.store import ABSTRACT, ATOMIC, COMPOSED, EXPECT, PART, W_POS, Substance, tie_key


def small() -> Substance:
    S = Substance()
    S.begin(1, [])
    units = [S.new_unit(ATOMIC, 0, i) for i in range(6)]
    whole = S.new_unit(COMPOSED, arity=2)
    S.tie(whole, units[0], PART, 0)
    S.tie(whole, units[1], PART, 1)
    for a, b in ((0, 2), (0, 3), (1, 2), (4, 5)):
        S.add1(W_POS, 0, S.tie(units[a], units[b], EXPECT, 1), 1.0)
    S.add1(W_POS, 0, S.tie(units[3], units[4], ABSTRACT, 0), 1.0)
    S.end()
    return S


def test_keys_are_unique_and_lists_are_arrays_in_creation_order():
    S = small()
    v = S.arrays()
    for tie in range(S.n_ties):
        assert S.find_tie(int(v.src[tie]), int(v.dst[tie]), int(v.form[tie]), int(v.lag[tie])) == tie
    assert len(set(S.tie_index)) == S.n_ties
    assert tie_key(1, 2, EXPECT, 1) != tie_key(2, 1, EXPECT, 1) != tie_key(1, 2, EXPECT, 0)
    for table in (S.out, S.inc):
        for ties in table.values():
            assert isinstance(ties, array) and list(ties) == sorted(ties)


def test_a_mind_saved_with_the_old_indices_is_converted_to_the_same_ones():
    S = small()
    state = S.__getstate__()
    v = S.arrays()
    state["tie_index"] = {(int(v.src[t]), int(v.dst[t]), int(v.form[t]), int(v.lag[t])): t for t in range(S.n_ties)}
    state["out"] = {k: list(ties) for k, ties in S.out.items()}
    state["inc"] = {k: list(ties) for k, ties in S.inc.items()}
    old = pickle.loads(pickle.dumps(state))
    T = Substance.__new__(Substance)
    T.__setstate__(old)
    assert T.tie_index == S.tie_index
    assert {k: list(x) for k, x in T.out.items()} == {k: list(x) for k, x in S.out.items()}
    assert all(isinstance(x, array) for x in T.out.values()) and all(isinstance(x, array) for x in T.inc.values())
    assert T.digest() == S.digest()
