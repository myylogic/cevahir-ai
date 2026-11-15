# -*- coding: utf-8 -*-
"""Zihin, F24: learning how far a voice can be believed, and carrying it into a new domain (T6, learning to learn).

What is pinned here
-------------------
* The kind of a voice is two magnitudes the mind perceives: how many members speak (doubling bands 1, 2, 3-4, 5-8,
  ...) and how much they agree (quarters, so that one half is a band edge; all agreeing is a band of its own).
* Credibility is read general -> specific: a kind never relied on is worth what voices in general have been worth;
  with no experience of voices at all it is 1/2.
* With no experience of voices the choice is exactly F11's: the setting on and off read the same.
* What comes next judges every voice that could have spoken (the one that spoke among them): once, the first time the
  context lives a value, one unit per claim on its kind and on every voice; the same fact heard again adds nothing.
* The voice that speaks is the one with the highest F11 score times credibility.
* An outcome is logged evidence: retracting its moment removes exactly it, nothing else moves.
* The lesson holds no word: in a new vocabulary the same kind is read with the same credibility, a fresh mind
  reads 1/2, and the new domain's outcomes add to the old ones.
* Off by default: a life without the setting has no channel for voices and notes no voice.

CPU only, tiny, deterministic; nothing is written anywhere.
"""
from __future__ import annotations

import pytest
import torch

from zihin import TOLD, Config, Event, Mind
from zihin import read
from zihin.read import VOICE_ALL, VOICE_CHANNEL, member_votes, voice_credibility, voice_kind
from zihin.store import EXPECT, W_NEG, W_POS
from zihin.evidence import conviction, frequency
from zihin.events import SENSED


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def say(mind, words, pause=1):
    for word in words:
        mind.step([Event("text", word, TOLD)])
    for _ in range(pause):
        mind.step([])


def new_mind(voices=True):
    mind = Mind(Config.changed(support="members", voices=voices))
    mind.add_channel("text", "act")
    return mind


def world(p):
    """A toy domain whose words all begin with ``p``: a class k of four members (x is never told d) and a value r of
    the dimension d that the three others hold."""
    w = lambda s: p + s                                                     # noqa: E731
    return [[w("a"), w("isa"), w("k")], [w("b"), w("isa"), w("k")], [w("c"), w("isa"), w("k")],
            [w("x"), w("isa"), w("k")], [w("a"), w("d"), w("r")], [w("b"), w("d"), w("r")], [w("c"), w("d"), w("r")]]


def live(mind, lines, times=2):
    for _ in range(times):
        for line in lines:
            say(mind, line)


def voice_rows(mind):
    """(symbol, W+, W-) of every tie on the channel of voices."""
    if not mind.ch.has(VOICE_CHANNEL):
        return {}
    S, cid, out = mind.S, mind.ch.id(VOICE_CHANNEL), {}
    for symbol in mind.ch.symbols[cid]:
        tie = S.find_tie(mind.unit_of(VOICE_CHANNEL, symbol), mind.holds, EXPECT, 0)
        out[symbol] = (float(S.t_w[W_POS, tie].sum()), float(S.t_w[W_NEG, tie].sum()))
    return out


def write(mind, symbol, held, failed):
    """Evidence on (voice -> holds) as one moment of the mind's life (to set up a reading)."""
    S = mind.S
    S.begin(mind.t + 1, mind.ch.sizes())
    mind.t += 1
    channel = mind.add_channel(VOICE_CHANNEL, "knob")
    tie = S.tie(mind._atom(channel, symbol), mind.holds, EXPECT, 0)
    S.add1(W_POS, SENSED, tie, held)
    S.add1(W_NEG, SENSED, tie, failed)
    S.end()


def test_kind_of_a_voice_is_numerosity_and_agreement_in_bands():
    assert voice_kind(1, 1.0) == (1, 4)
    assert voice_kind(2, 0.5) == (2, 2)
    assert voice_kind(2, 0.49) == (2, 1)
    assert voice_kind(3, 0.74) == (3, 2)
    assert voice_kind(4, 0.75) == (3, 3)
    assert voice_kind(5, 0.99) == (4, 3)
    assert voice_kind(8, 0.2) == (4, 0)
    assert voice_kind(9, 1.0) == (5, 4)
    assert voice_kind(17, 0.3) == (6, 1)


def test_credibility_is_read_general_to_specific():
    mind = new_mind()
    assert voice_credibility(mind, (1, 4)) == 0.5                     # nothing lived about voices
    write(mind, VOICE_ALL, 1.0, 3.0)
    e_all = (1.0 + 0.5) / (4.0 + 1.0)
    assert voice_credibility(mind, (1, 4)) == pytest.approx(e_all, abs=1e-12)     # never relied on: as voices go
    write(mind, (1, 4), 0.0, 3.0)
    c = conviction(3.0)
    assert voice_credibility(mind, (1, 4)) == pytest.approx(c * 0.0 + (1 - c) * e_all, abs=1e-12)
    write(mind, (3, 4), 4.0, 0.0)
    c = conviction(4.0)
    assert voice_credibility(mind, (3, 4)) == pytest.approx(c * 1.0 + (1 - c) * e_all, abs=1e-12)


def test_without_experience_of_voices_the_choice_is_f11():
    off = new_mind(voices=False)
    live(off, world("a_"))
    say(off, ["a_x", "a_d"], pause=0)
    on = off.clone()
    on.cfg.voices = True                                               # same substance, nothing lived about voices
    context = off.context
    assert read.members_map(off, context) == read.members_map(on, context)
    spoke, heard = on.voice[context]                                   # the voice that spoke and all that could
    assert spoke in heard and spoke[0] is not None


def test_what_comes_next_judges_the_voice_once():
    mind = new_mind()
    live(mind, world("a_"))
    before = voice_rows(mind)
    say(mind, ["a_x", "a_d"], pause=0)
    (kind, claim, f), heard = mind.voice[mind.context]
    assert mind.symbol_of(claim)[1] == "a_r" and f > 0.5                # the class speaks: x, like a, b, c, holds r
    r = mind.unit_of("text", "a_r")
    expected = {VOICE_ALL: [0.0, 0.0]}
    for k, c, f_k in heard:                                             # every voice is judged by what came
        held = c == r and f_k > 0.5
        for symbol in (VOICE_ALL, k):
            expected.setdefault(symbol, [0.0, 0.0])[0 if held else 1] += 1.0
    say(mind, ["a_r"])
    after = voice_rows(mind)
    gained = {s: [after[s][0] - before.get(s, (0.0, 0.0))[0], after[s][1] - before.get(s, (0.0, 0.0))[1]]
              for s in after}
    assert {s: g for s, g in gained.items() if g != [0.0, 0.0]} == expected
    assert expected[VOICE_ALL][0] >= 1.0                                # the voice that spoke held
    say(mind, ["a_x", "a_d", "a_r"])                                   # the same fact again: its own past speaks
    assert voice_rows(mind) == after


def test_the_voice_that_speaks_has_the_highest_f11_score_times_credibility():
    mind = new_mind()
    live(mind, world("a_") + [["a_p", "a_f", "a_u"], ["a_x", "a_f", "a_u"], ["a_p", "a_d", "a_s"]], times=3)
    say(mind, ["a_x", "a_d"], pause=0)
    context = mind.context
    scored = {}
    for general, e_abs in read.generalisations(mind, context):
        mass, members, where, fixed, voters = member_votes(mind, general, context)
        if mass:
            top = max(frequency(p, n) for p, n in mass.values())
            scored[general] = (e_abs * conviction(members) * top, voice_kind(voters, top))
    kinds = {kind for _, kind in scored.values()}
    assert len(kinds) >= 2                                             # voices of different kinds compete
    for favoured in sorted(kinds):
        twin = mind.clone()
        for kind in kinds:                                              # one kind proven, the others failed
            write(twin, kind, 9.0, 0.0) if kind == favoured else write(twin, kind, 0.0, 9.0)
        read.members_map(twin, context)
        best = max(scored.values(), key=lambda sk: sk[0] * voice_credibility(twin, sk[1]))
        assert twin.voice[context][0][0] == best[1] == favoured


def test_an_outcome_is_logged_evidence_and_retracts_exactly():
    mind = new_mind()
    live(mind, world("a_"))
    say(mind, ["a_x", "a_d", "a_r"])
    S = mind.S
    tie_all = S.find_tie(mind.unit_of(VOICE_CHANNEL, VOICE_ALL), mind.holds, EXPECT, 0)
    moment = max(m for m in S.log if any(tie_all in ids.tolist() for _, _, ids, _ in S.log[m].ev))
    before = S.t_w[:, :S.n_ties].clone()
    touched = torch.zeros(S.n_ties, dtype=torch.bool)
    delta = torch.zeros_like(before[:, tie_all])
    for table, slot, ids, amounts in S.log[moment].ev:
        touched[ids] = True
        delta[table, slot] += float(amounts[ids == tie_all].sum())
    assert float(delta[:2].sum()) >= 1.0                                # outcomes of voices lived in that moment
    mind.retract(moment)
    after = S.t_w[:, :S.n_ties]
    assert torch.equal(after[:, tie_all], before[:, tie_all] - delta)   # exactly that outcome is gone
    assert torch.equal(after[:, ~touched], before[:, ~touched])        # nothing else moved
    assert S.exact()


def test_the_lesson_holds_no_word_and_carries_into_a_new_vocabulary():
    old = new_mind()
    live(old, world("a_"))
    for x in ("a_x", "a_y", "a_z"):                                    # three held-out members, each told once
        say(old, [x, "a_isa", "a_k"])
        say(old, [x, "a_d", "a_r"])
    lesson = voice_rows(old)
    assert lesson[VOICE_ALL][0] >= 1.0                                  # it has lived outcomes of voices
    assert all(isinstance(s, tuple) or s == VOICE_ALL for s in lesson)  # kinds are magnitudes, not words
    fresh = new_mind()
    first = None
    for line in world("b_"):                                            # a new domain, new words, word by word
        for word in line + [None]:
            for mind in (old, fresh):
                mind.step([Event("text", word, TOLD)] if word else [])
            if first is None and old.voice:                             # the first voice that speaks in B
                first = next(iter(old.voice.values()))[0][0]
                assert voice_rows(old) == lesson                        # no outcome in B yet
                assert voice_credibility(old, first) != 0.5             # read with what A taught
                assert voice_credibility(fresh, first) == 0.5           # a fresh mind has no such experience
    assert first is not None
    grown = voice_rows(old)
    assert set(lesson) <= set(grown)
    assert sum(p + n for p, n in grown.values()) > sum(p + n for p, n in lesson.values())   # B adds to A's lesson


def test_off_by_default_no_voice_is_noted():
    assert Config().voices is False and Config.changed().voices is False
    mind = new_mind(voices=False)
    live(mind, world("a_"))
    say(mind, ["a_x", "a_d", "a_r"])
    assert not mind.ch.has(VOICE_CHANNEL)
    assert mind.voice == {}
