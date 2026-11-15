# -*- coding: utf-8 -*-
"""Zihin F25: one-shot evidence. A lag-1 tie born now starts with the failures a full record would hold.

What is pinned here (hand-checkable numbers, raw evidence unless said otherwise)
--------------------------------------------------------------------------------
* Off by default, and off is the old rule: "x y" told three times, then "x z" told once gives x -> z no negative
  evidence at all, e = 1/2 + 1/2 * (0.5 / 1.5) = 2/3 with trust 1/2 in told.
* On: x -> z is born with W-[told] = 3 (what x's other successor on that channel received, in its class), marked as
  judged (C-[told] = 3), so e = c (f - 1/2) + 1/2 with W+ = 0.5, W- = 1.5 (trust-weighted): 1/3. x -> y is untouched.
* Per class: two sensed "x y" and one told "x w" give the new x -> z W-[sensed] = 2 and W-[told] = 1; only the told
  part is marked as judged (sensed is direct experience and is never judged).
* The count is the rivals' evidence, not the unit's use: with F13 (only what was present learns) a word lingers in the
  focus longer than it learns, and the born tie still holds exactly what was learned.
* Only targets on sense and act channels (the world) are born with failures; the mind's own feelings are not.
* The induce record of the newborn shows the failures after the moment and nothing before it, so F9 / F24 still see
  a first time.
* Learned trust is not judged by them: a later sensed "x z" judges the told source exactly as with the setting off.
* Exactly retractable: retracting the birth moment leaves the tie without evidence and removes it, and every tensor
  equals the sum of the other moments' logged increments; undoing it restores the digest bit for bit.
* (b) The rivals reading: the negative mass of a world target is the positive mass of its rivals.

CPU only, tiny, deterministic; nothing is written anywhere.
"""
from __future__ import annotations

import pytest
import torch

import zihin.learn as learn
from zihin import IMITATED, SENSED, TOLD, Config, Event, Mind
from zihin.evidence import expectation
from zihin.read import own_mass
from zihin.store import C_NEG, EXPECT, W_NEG, W_POS


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _mind(**over) -> Mind:
    mind = Mind(Config.changed(support="members", **over))
    mind.add_channel("text", "act")
    return mind


def _utter(mind: Mind, symbols, prov=TOLD) -> list:
    """One symbol per moment, then a silent moment; returns the moment of each symbol."""
    moments = []
    for symbol in symbols:
        mind.step([Event("text", symbol, prov)])
        moments.append(mind.t)
    mind.step([])
    return moments


def _tie(mind: Mind, a: str, b: str) -> int:
    return mind.S.find_tie(mind.unit_of("text", a), mind.unit_of("text", b), EXPECT, 1)


def _echo_toy(**over) -> Mind:
    mind = _mind(**over)
    for _ in range(3):
        _utter(mind, ["x", "y"])
    _utter(mind, ["x", "z"])
    return mind


def test_off_by_default_and_the_old_rule():
    assert Config().born_failures is False and Config().rivals is False
    assert Config.changed().born_failures is False
    mind = _echo_toy()
    tie = _tie(mind, "x", "z")
    S = mind.S
    assert S.t_w[W_NEG, tie].abs().sum() == 0.0                   # a new tie: no failure at all
    assert float(S.t_w[W_POS, tie, TOLD]) == 1.0
    p, n = own_mass(mind, mind.unit_of("text", "x"))[mind.unit_of("text", "z")]
    assert (p, n) == (0.5, 0.0)
    assert expectation(p, n) == pytest.approx(2.0 / 3.0, abs=1e-15)


def test_born_tie_starts_with_the_failures_of_the_full_record():
    off, on = _echo_toy(), _echo_toy(born_failures=True)
    S = on.S
    tie = _tie(on, "x", "z")
    assert float(S.t_w[W_POS, tie, TOLD]) == 1.0
    assert float(S.t_w[W_NEG, tie, TOLD]) == 3.0                   # x was followed by y three times before
    assert float(S.t_w[C_NEG, tie, TOLD]) == 3.0                   # marked as judged
    assert float(S.t_w[W_NEG, tie].sum()) == 3.0                   # nothing in any other class
    p, n = own_mass(on, on.unit_of("text", "x"))[on.unit_of("text", "z")]
    assert (p, n) == (0.5, 1.5)
    assert expectation(p, n) == pytest.approx(1.0 / 3.0, abs=1e-15)
    # the old tie x -> y is exactly as with the setting off (F14 governs it)
    assert torch.equal(S.t_w[:, _tie(on, "x", "y")], off.S.t_w[:, _tie(off, "x", "y")])
    y = on.unit_of("text", "y")
    assert own_mass(on, on.unit_of("text", "x"))[y] == own_mass(off, off.unit_of("text", "x"))[y]


def test_failures_keep_the_class_of_the_moments_that_taught_the_rivals():
    mind = _mind(born_failures=True)
    _utter(mind, ["x", "y"], SENSED)
    _utter(mind, ["x", "y"], SENSED)
    _utter(mind, ["x", "w"], TOLD)
    _utter(mind, ["x", "z"], IMITATED)
    S, tie = mind.S, _tie(mind, "x", "z")
    assert float(S.t_w[W_POS, tie, IMITATED]) == 1.0
    assert float(S.t_w[W_NEG, tie, SENSED]) == 2.0
    assert float(S.t_w[W_NEG, tie, TOLD]) == 1.0
    assert float(S.t_w[W_NEG, tie, IMITATED]) == 0.0
    assert float(S.t_w[C_NEG, tie, TOLD]) == 1.0                   # a source whose trust is learned: judged
    assert float(S.t_w[C_NEG, tie, SENSED]) == 0.0                 # direct experience is never judged


def test_the_count_is_what_was_learned_not_the_use_of_the_unit():
    mind = _mind(born_failures=True, next_only_present=True)       # F13: only what was present learns
    _utter(mind, ["a", "x", "y"])
    _utter(mind, ["b", "x", "y"])
    _utter(mind, ["c", "x", "z"])
    S, x = mind.S, mind.unit_of("text", "x")
    tie = _tie(mind, "x", "z")
    assert float(S.t_w[W_NEG, tie, TOLD]) == 2.0                   # x learned twice before, both times y
    assert float(S.u_use[x]) > 3.0                                 # it lingered in the focus longer than that


def test_only_targets_of_the_world_are_born_with_failures():
    mind = _echo_toy(born_failures=True)
    S = mind.S
    n = S.n_ties
    marked = (S.t_w[C_NEG, :n].abs().sum(-1) > 0).nonzero().flatten().tolist()
    assert marked                                                  # (nothing here is judged by direct experience)
    text = mind.ch.id("text")
    for tie in marked:
        assert int(S.u_channel[int(S.t_dst[tie])]) == text
        assert int(S.t_lag[tie]) == 1 and int(S.t_form[tie]) == EXPECT
    off = _echo_toy()
    assert off.S.t_w[C_NEG, :off.S.n_ties].abs().sum() == 0.0


def test_induce_record_holds_the_failures_after_the_moment(monkeypatch):
    seen = []
    real = learn.induce

    def spy(mind, changes, occurred, slot_of_channel):
        seen.append({u: {v: list(r) for v, r in rec.items()} for u, rec in changes.items()})
        return real(mind, changes, occurred, slot_of_channel)

    monkeypatch.setattr(learn, "induce", spy)
    mind = _mind(born_failures=True)
    for _ in range(3):
        _utter(mind, ["x", "y"])
    seen.clear()
    mind.step([Event("text", "x", TOLD)])
    mind.step([Event("text", "z", TOLD)])
    x, z = mind.unit_of("text", "x"), mind.unit_of("text", "z")
    record = seen[-1][x][z]
    assert record[0] == 0.0 and record[2] == 0.0                   # nothing before: a first time for F9 / F24
    assert record[1] == 1.0                                        # what it received now
    assert record[3] == 3.0 and record[5] == TOLD                  # the failures, in the class that holds them


def test_learned_trust_is_not_judged_by_born_failures():
    rows = []
    for flag in (False, True):
        mind = _echo_toy(born_failures=flag)
        _utter(mind, ["x", "z"], SENSED)                           # direct experience lands on x -> z
        S = mind.S
        rows.append((S.t_w[:2, mind.source_tie[TOLD]].clone(), mind.trust.clone()))
    assert torch.equal(rows[0][0], rows[1][0])
    assert torch.equal(rows[0][1], rows[1][1])


def test_birth_moment_is_retracted_exactly():
    mind = _mind(born_failures=True)
    for _ in range(3):
        _utter(mind, ["x", "y"])
    moments = _utter(mind, ["x", "z"])
    born_in = moments[1]
    S = mind.S
    tie = _tie(mind, "x", "z")
    assert float(S.t_w[W_NEG, tie, TOLD]) == 3.0
    mind.retract(born_in)
    assert S.t_w[:, tie].abs().sum() == 0.0                        # positive, failures and their marks: all gone
    assert not bool(S.t_alive[tie])                                # nothing else supports it: removed
    ledger = torch.zeros_like(S.t_w[:, :S.n_ties])
    for _, entry in S.log.items():
        if not entry.retracted:
            for table, slot, ids, amounts in entry.ev:
                ledger[table, :, slot].index_add_(0, ids, amounts)
    assert torch.equal(S.t_w[:, :S.n_ties], ledger)


def test_undo_of_a_birth_restores_every_tensor():
    mind = _mind(born_failures=True)
    for _ in range(3):
        _utter(mind, ["x", "y"])
    mind.step([Event("text", "x", TOLD)])
    before = mind.S.digest()
    mind.step([Event("text", "z", TOLD)])
    assert float(mind.S.t_w[W_NEG, _tie(mind, "x", "z"), TOLD]) == 3.0
    mind.undo_last()
    assert mind.S.digest() == before


def test_rivals_reading_shares_one_mass():
    mind = _echo_toy(rivals=True)
    x, y, z = (mind.unit_of("text", w) for w in ("x", "y", "z"))
    mass = own_mass(mind, x)
    assert mass[y] == (1.5, 0.5) and mass[z] == (0.5, 1.5)          # f(z | x) = 1 / 4, f(y | x) = 3 / 4
    off = _echo_toy()
    for target, (p, n) in own_mass(off, x).items():                 # targets that are not the world keep their own
        if mind.ch.kind(int(mind.S.u_channel[target])) not in ("sense", "act"):
            assert mass[target] == (p, n)
