# -*- coding: utf-8 -*-
"""Zihin, evidence arithmetic and the growing store with its exact experience log.

What is pinned here (hand-checkable numbers)
--------------------------------------------
* One told statement with trust 0.3 and k = 1: W = 0.3, conviction 0.3 / 1.3, expectation 1/2 + c / 2.
* No evidence: frequency 1/2, conviction 0, expectation 1/2.
* Contradiction ADDS negative evidence: positive evidence is untouched, frequency falls, conviction rises.
* Evidence increments live on the grid 2**-30: sums are exact in any order; (x + d) - d == x bit for bit.
* The tables grow past their first capacity and keep what they held.
* Retraction at the level of the ledger: after ``retract(m)`` every tensor equals the sum of the other
  moments' logged increments; structure born in m that nothing else supports is removed.
* The caches kept beside the tensors are rebuilt identically from the tensors alone.

CPU only, tiny, deterministic; nothing is written anywhere.
"""
from __future__ import annotations

import random

import pytest
import torch

from zihin import evidence as ev
from zihin.store import ABSTRACT, ATOMIC, COMPOSED, EXPECT, PART, W_NEG, W_POS, Substance

F64, I64 = torch.float64, torch.int64


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_one_told_statement_with_trust_03():
    trust = torch.tensor([1.0, 0.3, 0.5, 1.0, 0.5, 0.5], dtype=F64)
    w_pos = torch.zeros(6, dtype=F64)
    w_pos[1] = 1.0                                   # one told statement
    W = float(ev.weigh(w_pos, trust))
    assert W == pytest.approx(0.3, abs=1e-15)
    assert ev.conviction(W, 1.0) == pytest.approx(0.3 / 1.3, abs=1e-15)
    assert ev.frequency(W, 0.0) == 1.0
    assert ev.expectation(W, 0.0, 1.0) == pytest.approx(0.5 + 0.5 * 0.3 / 1.3, abs=1e-15)


def test_no_evidence_is_one_half():
    assert ev.frequency(0.0, 0.0) == 0.5
    assert ev.conviction(0.0) == 0.0
    assert ev.expectation(0.0, 0.0) == 0.5
    zeros = torch.zeros(3, dtype=F64)
    assert torch.equal(ev.expectation(zeros, zeros), torch.full((3,), 0.5, dtype=F64))


def test_contradiction_adds_negative_and_never_removes_positive():
    # two sensed confirmations, then one sensed contradiction of weight 1
    pos, neg = 2.0, 0.0
    before = (ev.frequency(pos, neg), ev.conviction(pos + neg), ev.expectation(pos, neg))
    neg += 1.0
    after = (ev.frequency(pos, neg), ev.conviction(pos + neg), ev.expectation(pos, neg))
    assert pos == 2.0                                 # nothing was deleted
    assert before == (1.0, pytest.approx(2 / 3), pytest.approx(0.5 + 0.5 * 2 / 3))
    assert after[0] == pytest.approx(2 / 3)           # frequency moved
    assert after[1] == pytest.approx(3 / 4)           # more has been lived: conviction rose
    assert after[2] == pytest.approx(0.5 + 0.75 * (2 / 3 - 0.5))


def test_grid_makes_sums_exact_in_any_order():
    rng = random.Random(7)
    values = [ev.q(rng.random() * 3.0) for _ in range(2000)]
    forward = 0.0
    for v in values:
        forward += v
    backward = 0.0
    for v in reversed(values):
        backward += v
    shuffled = list(values)
    rng.shuffle(shuffled)
    mixed = 0.0
    for v in shuffled:
        mixed += v
    assert forward == backward == mixed               # bit-identical, not approximately equal
    x, d = ev.q(0.1), ev.q(0.3)
    assert (x + d) - d == x
    assert (0.1 + 0.3) - 0.3 != 0.1                   # the reason the grid is needed
    t = torch.tensor([0.1, 0.7, 2.0 / 3.0], dtype=F64)
    assert torch.equal(ev.quantize(t), torch.tensor([ev.q(0.1), ev.q(0.7), ev.q(2.0 / 3.0)], dtype=F64))
    assert abs(ev.q(2.0 / 3.0) - 2.0 / 3.0) <= 2.0 ** -31


def _fill(S: Substance, moments: int, seed: int = 3):
    """Random moments on a store: new units, new ties, evidence and use; returns the per-moment increments."""
    rng = random.Random(seed)
    for m in range(1, moments + 1):
        S.begin(m, [])
        for _ in range(rng.randint(0, 3)):
            S.new_unit(ATOMIC, 0, S.n_units)
        for _ in range(rng.randint(1, 6)):
            a, b = rng.randrange(S.n_units), rng.randrange(S.n_units)
            tie = S.tie(a, b, EXPECT, rng.randint(0, 1))
            S.add1(rng.choice((W_POS, W_NEG)), rng.randrange(S.P), tie, rng.random())
            S.use_ties(torch.tensor([tie], dtype=I64))
        S.use_units(torch.tensor([rng.randrange(S.n_units)], dtype=I64))
        S.end()


def test_tables_grow_and_keep_their_content():
    S = Substance(cap_units=4, cap_ties=4)
    S.begin(0, [])
    for i in range(3):
        S.new_unit(ATOMIC, 0, i)
    S.end()
    _fill(S, 40)
    assert S.n_units > 4 and S.n_ties > 4
    assert S.u_kind.numel() >= S.n_units and S.t_src.numel() >= S.n_ties
    total = torch.zeros(4, S.n_ties, S.P, dtype=F64)
    for entry in S.log.values():                      # the tensors are the exact sum of the log
        for table, slot, ids, amounts in entry.ev:
            total[table, :, slot].index_add_(0, ids, amounts)
    assert torch.equal(total, S.t_w[:, :S.n_ties])
    assert S.exact()


def test_retract_equals_the_sum_of_the_other_moments():
    S = Substance(cap_units=4, cap_ties=4)
    S.begin(0, [])
    for i in range(3):
        S.new_unit(ATOMIC, 0, i)
    S.end()
    _fill(S, 30)
    before = {name: t.clone() for name, t in S.tensors().items()}
    target = 17
    report = S.retract(target)
    assert report["increments"] > 0
    total = torch.zeros(4, S.n_ties, S.P, dtype=F64)
    use = torch.zeros(S.n_ties, dtype=F64)
    for m, entry in S.log.items():
        if m == target:
            continue
        for table, slot, ids, amounts in entry.ev:
            total[table, :, slot].index_add_(0, ids, amounts)
        for ids, amounts in entry.tie_use:
            use.index_add_(0, ids, amounts)
    assert torch.equal(total, S.t_w[:, :S.n_ties])     # bit-exact, any order of summation
    assert torch.equal(use, S.t_use[:S.n_ties])
    entry = S.log[target]
    untouched = torch.ones(S.n_ties, dtype=torch.bool)
    for _, _, ids, _ in entry.ev:
        untouched[ids] = False
    assert torch.equal(S.t_w[:, :S.n_ties][:, untouched], before["t_w"][:, untouched])
    born = torch.arange(entry.ties0, entry.ties1)
    dead = ~S.t_alive[born]
    assert int(dead.sum()) == report["ties_removed"]
    for tie in born[dead].tolist():                    # removed ties are gone from the caches too
        key = (int(S.t_src[tie]), int(S.t_dst[tie]), int(S.t_form[tie]), int(S.t_lag[tie]))
        assert S.tie_index.get(key, -1) != tie
    with pytest.raises(KeyError):
        S.retract(target)


def test_undo_last_restores_every_tensor_bit_exactly():
    S = Substance(cap_units=4, cap_ties=4)
    S.begin(0, [])
    for i in range(3):
        S.new_unit(ATOMIC, 0, i)
    S.end()
    _fill(S, 25, seed=11)
    before = {name: t.clone() for name, t in S.tensors().items()}
    digest = S.digest()
    S.begin(26, [])
    a = S.new_unit(ATOMIC, 0, 999)
    b = S.new_unit(COMPOSED, arity=2)
    S.tie(b, a, PART, 0)
    tie = S.tie(0, a, EXPECT, 1)
    S.add1(W_POS, 1, tie, 0.3)
    S.add1(W_NEG, 0, 0, 0.7)                           # also touches an old tie
    S.use_ties(torch.tensor([0, tie], dtype=I64))
    S.end()
    assert S.digest() != digest
    S.undo_last()
    after = S.tensors()
    assert set(after) == set(before)
    for name in before:
        assert after[name].shape == before[name].shape, name
        assert torch.equal(after[name], before[name]), name
    assert S.digest() == digest
    assert (0, 999) not in S.atom_index


def test_caches_are_rebuilt_from_the_tensors_alone():
    S = Substance()
    S.begin(1, [])
    units = [S.new_unit(ATOMIC, 0, i) for i in range(5)]
    whole = S.new_unit(COMPOSED, arity=2)
    S.tuple_index[(units[0], units[1])] = whole
    S.tie(whole, units[0], PART, 0)
    S.tie(whole, units[1], PART, 1)
    S.add1(W_POS, 0, S.tie(units[0], units[2], EXPECT, 1), 1.0)
    S.add1(W_POS, 0, S.tie(units[3], units[4], ABSTRACT, 0), 1.0)
    S.end()
    kept = (dict(S.tie_index), {k: list(v) for k, v in S.out.items()}, {k: list(v) for k, v in S.inc.items()},
            dict(S.atom_index), dict(S.tuple_index))
    S.rebuild_indices()
    rebuilt = (S.tie_index, S.out, S.inc, S.atom_index, S.tuple_index)
    assert kept == tuple(dict(r) for r in rebuilt)
    assert S.parts(whole) == (units[0], units[1])
