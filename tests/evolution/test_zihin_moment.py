# -*- coding: utf-8 -*-
"""Zihin, one moment of experience: learning by the one rule, growth, determinism, exact retraction.

What is pinned here
-------------------
* A told statement "A B" with trust 0.3: the tie A -> B holds w+[told] = 1 and conviction 0.3 / 1.3.
* Contradiction: after "A B" then "A C" the tie A -> B gets negative evidence e = 2/3 (trust 1/2) and keeps
  its positive evidence bit for bit: frequency 0.6, conviction 5/11, expectation 0.5 + 0.1 * 5/11.
* A silent channel is not a failed prediction.
* Growth per moment is bounded by the focus (explicit bound in F), independent of the size of the mind.
* Determinism: the same stream gives the identical state; keeping the experience log or not changes nothing.
* ``undo_last`` restores every tensor, the alphabets and the working state bit-exactly.
* ``retract(m)`` of a moment in the middle removes exactly what that moment added and nothing else.
* Probing a clone leaves the life untouched.

CPU only, tiny, deterministic; nothing is written anywhere.
"""
from __future__ import annotations

import random

import pytest
import torch

from zihin import SENSED, TOLD, Config, Event, Mind
from zihin import read
from zihin.store import EXPECT, W_NEG, W_POS


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def say(mind, words, prov=TOLD, pause=1, channel="text"):
    for word in words:
        mind.step([Event(channel, word, prov)])
    for _ in range(pause):
        mind.step([])


def new_mind(cfg):
    mind = Mind(cfg)
    mind.add_channel("text", "act")
    return mind


def stream(seed, n, vocab=12):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if rng.random() < 0.25:
            out.append([])
        else:
            out.append([Event("text", "w%d" % rng.randrange(vocab), rng.choice((TOLD, SENSED)))])
    return out


CONFIGS = {"as_written": lambda **kw: Config(**kw), "changed": lambda **kw: Config.changed(**kw)}


@pytest.mark.parametrize("variant", sorted(CONFIGS))
def test_first_hearing_is_weak(variant):
    mind = new_mind(CONFIGS[variant](trust_fixed={TOLD: 0.3}))
    say(mind, ["A", "B"], pause=0)
    a, b = mind.unit_of("text", "A"), mind.unit_of("text", "B")
    tie = mind.S.find_tie(a, b, EXPECT, 1)
    assert float(mind.S.t_w[W_POS, tie, TOLD]) == 1.0          # one event = one unit of raw evidence
    assert float(mind.S.t_w[W_POS, tie].sum()) == 1.0 and float(mind.S.t_w[W_NEG, tie].sum()) == 0.0
    belief = read.belief(mind, a, b)
    assert belief["own_c"] == pytest.approx(0.3 / 1.3, abs=1e-12)
    assert belief["own_f"] == 1.0
    assert belief["own_e"] == pytest.approx(0.5 + 0.5 * 0.3 / 1.3, abs=1e-12)


@pytest.mark.parametrize("variant", sorted(CONFIGS))
def test_contradiction_refines_without_deleting(variant):
    mind = new_mind(CONFIGS[variant](learn_trust=False))        # told trust stays at the no-evidence value 1/2
    say(mind, ["A", "B"])
    a, b = mind.unit_of("text", "A"), mind.unit_of("text", "B")
    tie = mind.S.find_tie(a, b, EXPECT, 1)
    first = read.belief(mind, a, b)
    assert first["own_c"] == pytest.approx(1 / 3) and first["own_e"] == pytest.approx(2 / 3)
    say(mind, ["A", "C"])
    assert float(mind.S.t_w[W_POS, tie, TOLD]) == 1.0          # positive evidence untouched
    after = read.belief(mind, a, b)
    if variant == "changed":                                    # a silent moment released the old context
        assert float(mind.S.t_w[W_NEG, tie, TOLD]) == pytest.approx(2 / 3, abs=1e-9)   # g * a * e, e = 2/3
        assert after["own_f"] == pytest.approx(0.6, abs=1e-9)
        assert after["own_c"] == pytest.approx(5 / 11, abs=1e-9)   # more was lived: conviction rose
        assert after["own_e"] == pytest.approx(0.5 + 0.1 * 5 / 11, abs=1e-9)
    else:                                                       # as written the decayed "A" of the first
        assert float(mind.S.t_w[W_NEG, tie, TOLD]) > 2 / 3      # utterance also expected B when "A" came again
        assert after["own_c"] > first["own_c"] and after["own_f"] < 1.0
    c = mind.unit_of("text", "C")
    assert read.belief(mind, a, c)["own_e"] == pytest.approx(2 / 3)
    say(mind, ["A", "C"])                                       # the refinement goes on, still nothing deleted
    assert float(mind.S.t_w[W_POS, tie, TOLD]) == 1.0
    assert read.belief(mind, a, b)["own_e"] < after["own_e"] < first["own_e"]
    if variant == "changed":
        assert read.belief(mind, a, c)["own_e"] > 2 / 3
    else:                                                       # finding F5 of NOTES.md: as written, the "A" still
        assert read.belief(mind, a, c)["own_e"] < 2 / 3         # carried from the last utterance expects C, sees
                                                                # "A" again and so weakens the very fact repeated


def test_silence_is_not_a_failed_prediction():
    mind = new_mind(Config.changed(learn_trust=False))
    say(mind, ["A", "B"])
    a, b = mind.unit_of("text", "A"), mind.unit_of("text", "B")
    tie = mind.S.find_tie(a, b, EXPECT, 1)
    say(mind, ["A"], pause=3)                                   # B expected, nothing said on the channel
    assert float(mind.S.t_w[W_NEG, tie].sum()) == 0.0
    assert read.belief(mind, a, b)["own_e"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("variant,F", [("as_written", 7), ("changed", 7), ("changed", 3)])
def test_growth_per_moment_is_bounded_by_the_focus(variant, F):
    mind = new_mind(CONFIGS[variant](F=F))
    n_events, feelings = 1, 4
    unit_bound = 2 * n_events + 1 + 3 * F
    tie_bound = F + 2 * F * F + 2 * F + 2 * F * (n_events + feelings) + F * (F - 1) + 1
    growth_units, growth_ties = [], []
    for events in stream(5, 400, vocab=30):
        u, t = mind.S.n_units, mind.S.n_ties
        mind.step(events)
        growth_units.append(mind.S.n_units - u)
        growth_ties.append(mind.S.n_ties - t)
        assert len(mind.focus) <= F
    assert max(growth_units) <= unit_bound
    assert max(growth_ties) <= tie_bound
    late_u, late_t = max(growth_units[300:]), max(growth_ties[300:])
    assert late_u <= unit_bound and late_t <= tie_bound          # no growth of the growth with the size


@pytest.mark.parametrize("variant", sorted(CONFIGS))
def test_same_stream_gives_identical_state(variant):
    events = stream(9, 300)
    minds = [new_mind(CONFIGS[variant]()), new_mind(CONFIGS[variant]()), new_mind(CONFIGS[variant](keep_log=False))]
    for mind in minds:
        for moment in events:
            mind.step(moment)
    digests = {mind.S.digest() for mind in minds}
    assert len(digests) == 1                                    # also without the log: nothing is read from it
    assert minds[0].focus == minds[1].focus == minds[2].focus
    assert minds[0].predicted == minds[2].predicted
    assert minds[0].S.exact()
    assert not minds[2].S.log


@pytest.mark.parametrize("variant", sorted(CONFIGS))
def test_undo_last_restores_tensors_alphabet_and_working_state(variant):
    mind = new_mind(CONFIGS[variant]())
    events = stream(21, 200)
    for moment in events:
        mind.step(moment)
    tensors = {name: t.clone() for name, t in mind.S.tensors().items()}
    working = mind._snapshot()
    alphabet = mind.ch.sizes()
    mind.step([Event("text", "never-seen", TOLD), Event("other-sense", 42, SENSED)])
    assert mind.S.n_units > tensors["u_kind"].numel()
    mind.undo_last()
    after = mind.S.tensors()
    for name, before in tensors.items():
        assert after[name].shape == before.shape, name
        assert torch.equal(after[name], before), name
    assert mind._snapshot() == working
    assert mind.ch.sizes()[:len(alphabet)] == alphabet
    assert mind.unit_of("text", "never-seen") == -1
    twin = new_mind(CONFIGS[variant]())
    for moment in events:
        twin.step(moment)
    follow = [Event("text", "w3", TOLD)]
    mind.step(follow)
    twin.step(follow)
    assert mind.S.digest() == twin.S.digest()                   # the undone moment left no trace at all


def test_retract_a_middle_moment_removes_exactly_its_evidence():
    mind = new_mind(Config.changed(learn_trust=False))
    say(mind, ["A", "B"])                                       # moments 1, 2 and a pause (3)
    say(mind, ["C", "D"])                                       # moments 4, 5 and a pause (6)
    say(mind, ["A", "B"])                                       # moments 7, 8: the fact is heard a second time
    a, b = mind.unit_of("text", "A"), mind.unit_of("text", "B")
    c, d = mind.unit_of("text", "C"), mind.unit_of("text", "D")
    tie_ab, tie_cd = mind.S.find_tie(a, b, EXPECT, 1), mind.S.find_tie(c, d, EXPECT, 1)
    assert float(mind.S.t_w[W_POS, tie_ab, TOLD]) == 2.0
    kept = mind.S.t_w[:, tie_cd].clone()
    mind.retract(2)                                             # the first hearing of B after A
    assert float(mind.S.t_w[W_POS, tie_ab, TOLD]) == 1.0        # exactly one hearing left
    assert read.belief(mind, a, b)["own_e"] == pytest.approx(2 / 3)
    assert torch.equal(mind.S.t_w[:, tie_cd], kept)             # other knowledge untouched, bit for bit
    mind.retract(8)                                             # the second hearing: the belief is gone
    assert read.belief(mind, a, b)["own_e"] == 0.5
    assert mind.S.find_tie(a, b, EXPECT, 1) in (-1, tie_ab)
    assert float(mind.S.t_w[:2, tie_ab].abs().sum()) == 0.0
    assert read.belief(mind, c, d)["own_e"] == pytest.approx(2 / 3)


def test_probing_a_clone_leaves_the_life_untouched():
    mind = new_mind(Config.changed())
    for moment in stream(2, 120):
        mind.step(moment)
    digest, focus = mind.S.digest(), list(mind.focus)
    twin = mind.clone()
    say(twin, ["w1", "w2", "w3"])
    assert twin.S.digest() != digest
    assert mind.S.digest() == digest and mind.focus == focus
    assert len(mind.S.log) == mind.t + 1 and not twin.S.log or len(twin.S.log) <= 5
