# -*- coding: utf-8 -*-
"""Zihin, how the evidence is stored: a tie with one provenance class keeps two numbers, a tie with more classes or
with judged amounts a full row of the four tables; what is read never depends on which.

What is pinned here
-------------------
* Most ties hold one class: two values each, no full row. A second class or a judged amount moves a tie, once, to a
  full row, value for value.
* Every reading (``w``, ``w2``, ``w4``, ``w1``, ``cell``, ``column``, ``t_w``) gives the same values as the four
  tables written out in full, which are rebuilt here from the log alone.
* ``undo_last`` gives the rows of the undone moment back: every stored array, not only the values, is as before.
* A mind saved before this layout (the four tables as one array) loads with the same values and the same digest.
* Saving and loading keeps the layout and the digest.

CPU only, tiny, deterministic; files only in pytest's temporary directory.
"""
from __future__ import annotations

import pickle
import random

import pytest
import torch

from zihin.store import ATOMIC, C_NEG, C_POS, EXPECT, PART, W_NEG, W_POS, WIDE, Substance

F64, I64 = torch.float64, torch.int64


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _life(S: Substance, moments: int, seed: int = 5, classes=(0, 1)) -> None:
    """Random moments: new units and ties, evidence mostly of one class per tie, sometimes another class, sometimes
    judged amounts, and PART ties with evidence like a recognised composed unit."""
    rng = random.Random(seed)
    for m in range(1, moments + 1):
        S.begin(m, [])
        for _ in range(rng.randint(0, 3)):
            S.new_unit(ATOMIC, 0, S.n_units)
        for _ in range(rng.randint(1, 6)):
            a, b = rng.randrange(S.n_units), rng.randrange(S.n_units)
            tie = S.tie(a, b, EXPECT, rng.randint(0, 1))
            slot = classes[0] if rng.random() < 0.85 else rng.choice(classes)
            S.add1(rng.choice((W_POS, W_NEG)), slot, tie, rng.random())
            if rng.random() < 0.05:
                S.add1(rng.choice((C_POS, C_NEG)), slot, tie, rng.random() / 4)
        ids = torch.tensor([rng.randrange(S.n_ties) for _ in range(5)], dtype=I64)
        S.add(rng.choice((W_POS, W_NEG)), classes[0], ids, torch.rand(5, dtype=F64,
                                                                     generator=torch.Generator().manual_seed(m)))
        part = S.tie(rng.randrange(S.n_units), rng.randrange(S.n_units), PART, 0)
        S.add1(W_POS, classes[0], part, 1.0)
        S.end()


def _start(cap_ties: int = 4, cap_rows: int = 2) -> Substance:
    S = Substance(cap_units=4, cap_ties=cap_ties, cap_rows=cap_rows)
    S.begin(0, [])
    for i in range(3):
        S.new_unit(ATOMIC, 0, i)
    S.end()
    return S


def _tables_from_log(S: Substance) -> torch.Tensor:
    total = torch.zeros(4, S.n_ties, S.P, dtype=F64)
    for entry in S.log.values():
        if entry.retracted:
            continue
        for table, slot, ids, amounts in entry.ev:
            total[table, :, slot].index_add_(0, ids, amounts)
    return total


def test_one_class_holds_two_numbers_and_more_classes_a_full_row():
    S = _start()
    _life(S, 60, classes=(1, 4))
    cls = S.t_cls[:S.n_ties]
    assert int(((cls >= 1) & (cls <= S.P)).sum()) > int((cls == WIDE).sum()) > 0
    assert S.n_rows - 1 == int((cls == WIDE).sum())
    full = _tables_from_log(S)
    classes = (full[:2] != 0).any(dim=0).sum(-1)
    judged = (full[2:] != 0).any(dim=(0, 2))
    assert torch.equal(cls == WIDE, (classes > 1) | judged)   # a full row exactly where one class is not enough
    assert torch.equal(S.t_w, full)


def test_every_reading_gives_the_values_of_the_full_tables():
    S = _start()
    _life(S, 50, seed=9, classes=(2, 0, 5))
    full = _tables_from_log(S)
    ids = torch.tensor(list(range(S.n_ties))[::-1] + [0, 0, 3], dtype=I64)
    for table in range(4):
        assert torch.equal(S.w(table, ids), full[table, ids])
        assert torch.equal(S.column(table, 2), full[table, :, 2])
    pos, neg = S.w2(ids)
    assert torch.equal(pos, full[W_POS, ids]) and torch.equal(neg, full[W_NEG, ids])
    assert torch.equal(S.w4(ids), full[:, ids])
    for tie in range(S.n_ties):
        for table in range(4):
            assert torch.equal(S.w1(table, tie), full[table, tie])
            for slot in range(S.P):
                assert S.cell(table, tie, slot) == float(full[table, tie, slot])
    assert S.max_evidence() == float(full.abs().max())


def test_undo_last_gives_back_the_rows_of_the_moment():
    S = _start()
    _life(S, 30, seed=2, classes=(1, 3))
    before = {name: t.clone() for name, t in S.physical().items()}
    digest = S.digest()
    narrow = next(t for t in range(S.n_ties) if 0 < int(S.t_cls[t]) < WIDE)
    c = int(S.t_cls[narrow]) - 1
    S.begin(31, [])
    S.add1(W_POS, (c + 1) % S.P, narrow, 0.25)           # an old tie widened by a second class
    S.add1(C_POS, c, narrow, 0.125)
    a = S.new_unit(ATOMIC, 0, 999)
    tie = S.tie(0, a, EXPECT, 1)
    S.add1(W_POS, 1, tie, 0.5)
    S.add1(W_NEG, 2, tie, 0.5)                           # a new tie widened in its first moment
    S.end()
    assert int(S.t_cls[narrow]) == WIDE and S.digest() != digest
    S.undo_last()
    after = S.physical()
    assert set(after) == set(before)
    for name in before:
        assert torch.equal(after[name], before[name]), name
    assert S.digest() == digest


def test_retraction_and_growth_keep_the_values_exact():
    S = _start(cap_ties=8, cap_rows=1)
    _life(S, 45, seed=4, classes=(0, 3))
    S.retract(20)
    S.retract(33)
    assert torch.equal(S.t_w, _tables_from_log(S))
    assert S.exact()


def test_a_mind_saved_with_the_full_tables_loads_with_the_same_values():
    S = _start()
    _life(S, 40, seed=8, classes=(1, 2))
    values, digest = S.t_w.clone(), S.digest()
    state = pickle.loads(pickle.dumps(S.__getstate__()))   # a copy, written as the layout before rows was
    for name in ("t_cls", "t_ev", "t_row", "ex", "r_tie", "r_was", "n_rows", "_cr"):
        state.pop(name)
    state["t_w"] = torch.cat([values, torch.zeros(4, 3, S.P, dtype=F64)], dim=1)   # with spare capacity
    for name in ("t_src", "t_dst", "t_form", "t_lag", "t_use", "t_last", "t_birth", "t_alive"):
        state[name] = torch.cat([state[name], torch.zeros(3, dtype=state[name].dtype)])
    state["_ct"] = S.n_ties + 3
    old = Substance.__new__(Substance)
    old.__setstate__(state)
    assert torch.equal(old.t_w, values) and old.digest() == digest
    assert old.n_rows - 1 == int((old.t_cls[:old.n_ties] == WIDE).sum())
    _life(old, 5, seed=1, classes=(1, 2))                # and lives on like any mind
    _life(S, 5, seed=1, classes=(1, 2))
    assert old.digest() == S.digest()


def test_saved_and_loaded_with_the_layout_and_the_digest(tmp_path):
    S = _start()
    _life(S, 35, seed=6, classes=(0, 4))
    path = tmp_path / "s.pkl"
    path.write_bytes(pickle.dumps(S, protocol=pickle.HIGHEST_PROTOCOL))
    T = pickle.loads(path.read_bytes())
    assert T.digest() == S.digest()
    for name, tensor in S.physical().items():
        assert torch.equal(T.physical()[name], tensor), name
    _life(S, 5, seed=3, classes=(0, 4))
    _life(T, 5, seed=3, classes=(0, 4))
    assert T.digest() == S.digest()
