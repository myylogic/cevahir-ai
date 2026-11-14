# -*- coding: utf-8 -*-
"""Zihin, concepts are formed: recruitment, slots and frames, emergent categories, induction up, support down,
learned trust, chunking.

What is pinned here
-------------------
* Recruitment: an utterance becomes composed units (its prefixes) with PART ties in order; meeting it again
  adds evidence and no structure.
* Slots: two composed units that differ in exactly one position give a category for that position and a frame;
  the fillers hold ABSTRACT ties to the category with one unit of evidence per encounter.
* A category stands in for its instance (typed frame) and acquires, by induction, what the instance acquired:
  one told instance with ABSTRACT expectation 2/3 gives the frame tie 2/3 of a unit.
* Support down as written (W_eff = W_own + lam e W_g) lets twelve confirming instances override two sensed
  contradictions of an exception; the capped and the nested form keep the exception (finding F3).
* Nested support: general to specific, own evidence last; equals e = c (f - 1/2) + 1/2 without generalisations.
* Induction as written never gives a category negative evidence for the values other members have (F4).
* Trust is learned: a told claim confirmed by direct experience raises the trust in telling to 3/4; a told
  claim refuted lowers it; every claim is judged once.
* Chunking fuses a reliable succession into one item.

CPU only, tiny, deterministic; nothing is written anywhere.
"""
from __future__ import annotations

import pytest
import torch

from zihin import SENSED, TOLD, Config, Event, Mind
from zihin import read
from zihin.evidence import conviction, expectation, frequency, q
from zihin.store import ABSTRACT, CATEGORY, COMPOSED, EXPECT, INDUCED, PART, SLOTTED, W_NEG, W_POS


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def say(mind, words, prov=TOLD, pause=1):
    for word in words:
        mind.step([Event("text", word, prov)])
    for _ in range(pause):
        mind.step([])


def new_mind(**over):
    over.setdefault("learn_trust", False)
    over.setdefault("lag0", False)
    mind = Mind(Config.changed(**over))
    mind.add_channel("text", "act")
    return mind


def unit(mind, *words):
    ids = tuple(mind.unit_of("text", w) for w in words)
    return ids[0] if len(ids) == 1 else mind.S.tuple_index.get(ids, -1)


def inject(mind, tie, table, slot, amount):
    """Put evidence on a tie directly (as one more moment of the log), to get hand-checkable masses."""
    mind.t += 1
    mind.S.begin(mind.t, mind.ch.sizes())
    mind.S.add1(table, slot, tie, amount)
    mind.S.end()


def fillers(mind, category):
    S = mind.S
    return {mind.symbol_of(int(S.t_src[t]))[1] for t in S.inc.get((category, ABSTRACT), [])}


def test_recruitment_makes_composed_units_with_ordered_parts():
    mind = new_mind(slots=False)
    say(mind, ["a", "b", "c"])
    a, b, c = (unit(mind, w) for w in "abc")
    ab, abc = unit(mind, "a", "b"), unit(mind, "a", "b", "c")
    S = mind.S
    assert ab >= 0 and abc >= 0 and int(S.u_kind[ab]) & COMPOSED and int(S.u_arity[abc]) == 3
    assert S.parts(ab) == (a, b) and S.parts(abc) == (a, b, c)
    part = S.find_tie(abc, c, PART, 2)
    assert float(S.t_w[W_POS, part, TOLD]) == 1.0
    units, ties = S.n_units, S.n_ties
    say(mind, ["a", "b", "c"])
    assert S.n_units == units                               # met again: no new structure, more evidence
    assert float(S.t_w[W_POS, part, TOLD]) == 2.0
    assert S.find_tie(ab, c, EXPECT, 1) >= 0                # the prefix expects what followed it
    assert read.belief(mind, ab, c)["own_c"] == pytest.approx(0.5)   # two told hearings, trust 1/2: W = 1


def test_slots_and_categories_emerge_from_instances_in_the_same_place():
    mind = new_mind()
    say(mind, ["a", "x", "p"])
    S = mind.S
    assert not (S.u_kind[:S.n_units] & CATEGORY).any()      # one instance: no category yet
    say(mind, ["b", "x", "p"])
    cats = [u for u in range(S.n_units) if int(S.u_kind[u]) & CATEGORY]
    assert len(cats) == 2                                   # the place before "x" and the place before "x p"
    for category in cats:
        frame = category + 1
        assert int(S.u_kind[frame]) & SLOTTED and int(S.u_kind[frame]) & INDUCED
        assert category in S.parts(frame)
        assert fillers(mind, category) == {"a", "b"}
    a, b, x = unit(mind, "a"), unit(mind, "b"), unit(mind, "x")
    first = cats[0]
    assert S.parts(first + 1) == (first, x)
    tie_a = S.find_tie(a, first, ABSTRACT, 0)
    assert float(S.t_w[W_POS, tie_a, TOLD]) == 1.0 and float(S.t_w[W_POS, S.find_tie(b, first, ABSTRACT, 0), TOLD]) == 1.0
    say(mind, ["a", "x", "p"])                              # the filler is met in that place again
    assert float(S.t_w[W_POS, tie_a, TOLD]) == 2.0
    say(mind, ["c", "y", "q"])                              # differs in more than one position: no category
    assert len([u for u in range(S.n_units) if int(S.u_kind[u]) & CATEGORY]) == 2
    say(mind, ["c", "x", "p"])                              # a third instance joins the existing categories
    assert fillers(mind, first) == {"a", "b", "c"}
    assert len([u for u in range(S.n_units) if int(S.u_kind[u]) & CATEGORY]) == 3   # plus "c" before x / y


def test_category_stands_in_and_acquires_what_its_instance_acquires():
    mind = new_mind()
    say(mind, ["m1", "isa", "C"])
    say(mind, ["m2", "isa", "C"])
    say(mind, ["m1", "d", "v"])
    S = mind.S
    m1, d, v = unit(mind, "m1"), unit(mind, "d"), unit(mind, "v")
    cats = [int(S.t_dst[t]) for t in S.out[(m1, ABSTRACT, 0)]]
    assert len(cats) == 2
    for category in cats:
        e_abs = expectation(0.5 * float(S.t_w[W_POS, S.find_tie(m1, category, ABSTRACT, 0), TOLD]), 0.0)
        assert e_abs == pytest.approx(2 / 3)                 # one told encounter, trust 1/2
        frame = S.tuple_index.get((category, d), -1)
        assert frame >= 0                                    # the category stood in for m1: (category, d)
        kind = int(S.u_kind[frame])
        assert kind & SLOTTED and not kind & INDUCED
        tie = S.find_tie(frame, v, EXPECT, 1)
        assert float(S.t_w[W_POS, tie, TOLD]) == q(2 / 3)    # induction: min(1, 1) * e(m1 -> category)
        assert float(S.t_w[W_NEG, tie].sum()) == 0.0
    # m2 was never told its d: what it expects comes from the categories only
    twin = mind.clone()
    for word in ("m2", "d"):
        twin.step([Event("text", word, TOLD)])
    expected = twin.expected("text")
    assert expected["v"] > 0.5
    assert read.own_mass(twin, twin.context) == {}           # no own evidence at all


def _exception_setup(support):
    """m1 and m2 are followed by "d v"; then the category tie gets N sensed confirmations and the instance
    tie (m1, d) -> v two sensed contradictions."""
    mind = new_mind(support=support, typed_frames=False)
    say(mind, ["m1", "d", "v"])
    say(mind, ["m2", "d", "v"])
    S = mind.S
    instance, v = unit(mind, "m1", "d"), unit(mind, "v")
    gens = read.generalisations(mind, instance)
    assert len(gens) == 1
    frame, e_abs = gens[0]
    assert e_abs == pytest.approx(2 / 3)
    return mind, instance, v, frame, S.find_tie(instance, v, EXPECT, 1), S.tie(frame, v, EXPECT, 1)


@pytest.mark.parametrize("confirmations", [3, 12])
def test_support_down_as_written_lets_a_strong_rule_override_an_exception(confirmations):
    mind, instance, v, frame, own_tie, rule_tie = _exception_setup("raw")
    inject(mind, own_tie, W_NEG, SENSED, 2.0)                # the exception: sensed twice that it is not v
    inject(mind, rule_tie, W_POS, SENSED, float(confirmations))
    S = mind.S
    own_pos = float((S.t_w[W_POS, own_tie] * mind.trust).sum())
    rule_pos = float((S.t_w[W_POS, rule_tie] * mind.trust).sum())
    assert own_pos == 0.5 and rule_pos == pytest.approx(confirmations + 1 / 3, abs=1e-8)
    w_pos, w_neg = read.effective(mind, instance)[v]
    assert w_pos == pytest.approx(own_pos + 0.5 * (2 / 3) * rule_pos, abs=1e-8)   # W_own + lam e W_g
    assert w_neg == 2.0
    e = read.expect_map(mind, instance)[v]
    if confirmations == 3:
        assert e < 0.5                                       # the exception still holds
    else:
        assert e > 0.5                                       # finding F3: twelve instances override it


@pytest.mark.parametrize("support", ["capped", "nested"])
@pytest.mark.parametrize("confirmations", [3, 12, 300])
def test_changed_support_keeps_the_exception_however_strong_the_rule(support, confirmations):
    mind, instance, v, frame, own_tie, rule_tie = _exception_setup(support)
    before = read.expect_map(mind, instance)[v]
    inject(mind, rule_tie, W_POS, SENSED, float(confirmations))
    supported = read.expect_map(mind, instance)[v]
    assert supported > before                                # the rule does raise the instance belief
    inject(mind, own_tie, W_NEG, SENSED, 2.0)
    assert read.expect_map(mind, instance)[v] < 0.5          # own experience outweighs inherited support
    if support == "capped":
        w_pos, w_neg = read.effective(mind, instance)[v]
        rule_pos = confirmations + 1 / 3
        assert w_pos == pytest.approx(0.5 + 0.5 * (2 / 3) * rule_pos / max(1.0, rule_pos), abs=1e-8)
        assert w_neg == 2.0


def test_nested_support_is_general_to_specific_then_own():
    mind, instance, v, frame, own_tie, rule_tie = _exception_setup("nested")
    S = mind.S
    rule = (float((S.t_w[W_POS, rule_tie] * mind.trust).sum()), float((S.t_w[W_NEG, rule_tie] * mind.trust).sum()))
    own = (float((S.t_w[W_POS, own_tie] * mind.trust).sum()), float((S.t_w[W_NEG, own_tie] * mind.trust).sum()))
    c_rule = 0.5 * (2 / 3) * conviction(sum(rule))           # lam * e(i -> g) * conviction of the rule
    inherited = c_rule * frequency(*rule) + (1 - c_rule) * 0.5
    c_own = conviction(sum(own))
    by_hand = c_own * frequency(*own) + (1 - c_own) * inherited
    assert read.expect_map(mind, instance)[v] == pytest.approx(by_hand, abs=1e-12)
    lonely = new_mind(support="nested", slots=False)         # no generalisation: the plain formula
    say(lonely, ["m1", "d", "v"])
    i2, v2 = unit(lonely, "m1", "d"), unit(lonely, "v")
    assert read.expect_map(lonely, i2)[v2] == pytest.approx(expectation(0.5, 0.0), abs=1e-12)


@pytest.mark.parametrize("general_miss", [False, True])
def test_induction_as_written_counts_confirming_instances_only(general_miss):
    mind = new_mind(general_miss=general_miss, typed_frames=False)
    for who, value in (("m1", "v"), ("m2", "v"), ("m3", "w"), ("m4", "w"), ("m5", "v"), ("m6", "w")):
        say(mind, [who, "d", value])
    S = mind.S
    frame = S.frame_of[(0, (-1, unit(mind, "d")))]
    masses = read.own_mass(mind, frame)
    v, w = unit(mind, "v"), unit(mind, "w")
    assert masses[v][0] > 0 and masses[w][0] > 0
    if general_miss:                                         # changed: the category also learns what failed
        assert masses[v][1] > 0 and masses[w][1] > 0
        assert frequency(*masses[v]) < 1.0 and frequency(*masses[w]) < 1.0
    else:                                                    # as written: both values look like a rule (F4)
        assert masses[v][1] == 0.0 and masses[w][1] == 0.0
        assert frequency(*masses[v]) == 1.0 and frequency(*masses[w]) == 1.0


def test_trust_is_learned_from_direct_experience_and_every_claim_is_judged_once():
    mind = new_mind(learn_trust=True, slots=False)
    assert float(mind.trust[TOLD]) == 0.5                    # nothing lived: expectation 1/2
    say(mind, ["a", "b"], prov=TOLD)
    say(mind, ["a", "b"], prov=SENSED)                       # the world confirms what was told
    mind.step([])
    assert float(mind.trust[TOLD]) == pytest.approx(0.75)    # one claim held: e = 1/2 + 1/2 * 1/2
    say(mind, ["a", "b"], prov=SENSED)                       # sensed again: the same claim is not judged twice
    mind.step([])
    assert float(mind.trust[TOLD]) == pytest.approx(0.75)
    say(mind, ["a", "c"], prov=TOLD)                         # a second claim ...
    a, c = unit(mind, "a"), unit(mind, "c")
    e_claim = read.belief(mind, a, c)["own_e"]
    say(mind, ["a", "b"], prov=SENSED)                       # ... refuted: after "a" the world shows "b"
    mind.step([])
    assert e_claim == pytest.approx(expectation(0.75, 0.0))  # the claim "a c" at trust 3/4
    denied = expectation(0.75 + 2.0, 0.0)                    # telling "a c" also denied "a b" with this weight
    # refuted twice: c did not come (judged share = the sensed amount e_claim), and b, which it denied, did
    assert float(mind.trust[TOLD]) == pytest.approx(expectation(1.0, e_claim + denied), abs=1e-6)
    assert float(mind.trust[TOLD]) < 0.75
    assert float(mind.trust[SENSED]) == 1.0                  # direct experience is the judge


def test_chunking_fuses_a_reliable_succession_into_one_item():
    mind = new_mind(chunking=True, slots=False)
    plain = new_mind(chunking=False, slots=False)
    for i in range(7):
        for m in (mind, plain):
            say(m, ["x%d" % i, "p", "q"], prov=SENSED)
    assert not plain.chunked
    p, q_unit = unit(mind, "p"), unit(mind, "q")
    chunk = mind.S.tuple_index.get((p, q_unit), -1)
    assert chunk >= 0 and mind.chunked and set(mind.chunked) == {chunk}
    tie = mind.S.find_tie(p, q_unit, EXPECT, 1)
    assert float(mind.S.t_use[tie]) >= mind.cfg.m_chunk
    assert read.belief(mind, p, q_unit)["own_e"] >= mind.cfg.e_chunk
    x6 = unit(mind, "x6")
    assert mind.S.tuple_index.get((x6, chunk), -1) >= 0      # the chunk counts as ONE item of the episode
    assert plain.S.tuple_index.get((unit(plain, "x6"), unit(plain, "p"), unit(plain, "q")), -1) >= 0
