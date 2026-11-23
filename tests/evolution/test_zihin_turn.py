# -*- coding: utf-8 -*-
"""Zihin, F69 Its Own Turn Comes Last (``Config.own_turn``, with silent claims).

What is pinned here
-------------------
* Off by default, a studied mechanism, in use in the full configuration; it needs silent claims.
* The two conditions a claim is made in are booked apart: when it was the mind's turn (said, held back, tried) as
  its own act, while it only listened as what it inferred and did not act on. A way's record is read general ->
  specific as everything is: what its claims came to while listening, then what they came to on its own turn.
  One failure of its own after seven successes while listening brings a claim under the line the mind speaks at;
  pooled, the same record leaves it far above.
* ``hata.proven`` is the own turn's record first; ``hata.failed`` counts an experience against a way in either
  condition; the vote of the ways like it takes each way's own turn by its conviction and the rest from listening.
* What the mind said or held back at a context stays kept when it then listens to that context (the teacher shows
  the dialogue it has just answered); without F69 the silent claim of that moment replaced it.
* A life: a frame whose watched things only happen to share a value, and a lawful one, asked in the teacher's
  loop. With silent claims and no F69 the mind is wrong five times before it stops asserting; with F69 twice;
  each part of F69 alone changes nothing; the lawful frame is answered from its first question in every case
  (what silent claims are for), and without silent claims it is not.
* A mind that only watches is never asked: nothing of its own turn exists and F69 changes nothing it says.
"""
from __future__ import annotations

import pytest
import torch

import test_zihin_own_race as R
from zihin import Config, hata
from zihin.events import ACTED, INFERRED
from zihin.__main__ import NEEDS, STUDIED, Session, VARIANTS
import zihin.__main__ as chat
from zihin.store import EXPECT, W_NEG, W_POS

THINGS = ["inek", "koyun", "keçi", "at", "eşek", "deve", "aslan", "kurt", "fok", "serçe", "kaz", "yunus", "tilki",
          "ayı", "balina", "kartal", "panter", "somon", "leylek", "geyik"]
MEMBERS, POOL, FRESH = THINGS[:6], THINGS[6:14], THINGS[14:20]
VALUES = ["ot", "et", "balık", "tohum"]


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def misleading(x: str):
    """What a thing eats: the watched things share one food; every other thing eats another."""
    value = VALUES[0] if x in MEMBERS else VALUES[1 + THINGS.index(x) % 3]
    return "%s ne yer ?" % x, "%s %s yer ." % (x, value)


def lawful(x: str):
    return "%s canlı mı ?" % x, "evet %s canlıdır ." % x


def mark(said: str, answer: str) -> str:
    return "R" if said + " ." == answer else ("-" if said == "(sessiz)" or len(said.split()) < 2 else "W")


def life(part=None, asked: bool = True, **switches):
    """Six things are watched being asked in both frames; then eight new things are asked in both in the teacher's
    loop (whatever the mind says, the true dialogue is shown), or, not ``asked``, only watched. Returns (the
    session, the misleading frame's marks, the lawful frame's, the marks of six fresh things on copies)."""
    s = R.born("full", **switches)
    if part:
        s.mind.__dict__["_own_turn"] = part                 # (research: the one part of F69 that is on)
    for _ in range(2):
        for x in MEMBERS:
            s.watch(*misleading(x))
            s.watch(*lawful(x))
    s.close_turn()
    marks = {misleading: "", lawful: ""}
    for x in POOL:
        for frame in (misleading, lawful):
            q, a = frame(x)
            if asked:
                said = s.reply(q, limit=20)
                s.close_turn()
                marks[frame] += mark(said, a)
            s.watch(q, a)
            s.close_turn()
    fresh = "".join(mark(Session(s.mind.clone()).reply(misleading(x)[0], limit=20), misleading(x)[1]) for x in FRESH)
    return s, marks[misleading], marks[lawful], fresh


def test_off_by_default_studied_and_needs_silent_claims():
    assert Config().own_turn is False and "own_turn" in STUDIED
    assert ("own_turn", ("silent_claims",)) in NEEDS
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        assert VARIANTS["full"]().own_turn is True and VARIANTS["speaking"]().own_turn is False
    finally:
        chat._log_path = made


def test_the_two_conditions_are_booked_apart_and_its_own_turn_is_read_last():
    apart, pooled = R.born("full").mind, R.born("full", own_turn=False).mind
    way = hata.reading_ways("bir yol")
    for mind in (apart, pooled):
        for _ in range(7):
            hata.settle(mind, way, True, 0.6, turn=False)   # seven claims held while it only listened
        hata.settle(mind, way, False, 0.6)                  # one failed when it was its turn
    k = apart.cfg.k
    unit = apart.unit_of(hata.CHANNEL, way[-1])
    tie = apart.S.find_tie(unit, apart.holds, EXPECT, 0)
    cell = apart.S.cell
    assert (cell(W_POS, tie, INFERRED), cell(W_NEG, tie, INFERRED)) == (7.0, 0.0)
    assert (cell(W_POS, tie, ACTED), cell(W_NEG, tie, ACTED)) == (0.0, 1.0)
    assert hata._cells(apart, way[-1]) == ((0.0, 1.0), (7.0, 0.0))
    listening, turn = hata._levels(apart, way[-1], apart.unit_of(hata.CHANNEL, hata.EXPECTED))
    assert listening == pytest.approx((7.0 / (7.0 + k), 1.0, 0.6)) and turn == pytest.approx((1.0 / (1.0 + k), 0.0, 0.6))
    line = 0.5 + apart.cfg.theta / (2.0 * apart.cfg.T)
    e_listening = listening[0] * 1.0 + (1.0 - listening[0]) * 0.6          # held more often than claimed: up to 1
    assert hata.read_as([listening, turn], 0.6) == pytest.approx(turn[0] * 0.0 + (1.0 - turn[0]) * e_listening)
    assert hata.read_as(hata.record(apart, way), 0.6) < 0.5 < line          # its own failure is read last
    assert hata.read_as(hata.record(pooled, way), 0.6) > 0.8 > line         # pooled: seven to one
    c = turn[0] + (1.0 - turn[0]) * listening[0]                           # the way's vote among the ways like it
    assert hata._own(apart, way[-1], apart.unit_of(hata.CHANNEL, hata.EXPECTED)) == pytest.approx(
        (c, (1.0 - turn[0]) * listening[0] / c, 0.6))
    for mind in (apart, pooled):                           # it then holds five times on its own turn: its voice
        for _ in range(5):                                 # is back
            hata.settle(mind, way, True, 0.6)
    assert hata.read_as(hata.record(apart, way), 0.6) > line


def test_proven_is_its_own_turn_first_and_failed_counts_both_conditions():
    mind = R.born("full").mind
    k = mind.cfg.k
    shown, tried, both = (hata.reading_ways(name) for name in ("gösterilen", "denenen", "ikisi"))
    for _ in range(7):
        hata.settle(mind, shown, True, 0.6, turn=False)
        hata.settle(mind, both, True, 0.6, turn=False)
    for _ in range(3):
        hata.settle(mind, tried, True, 0.6)
    hata.settle(mind, both, False, 0.6)
    hata.settle(mind, both, False, 0.6, turn=False)
    assert hata.proven(mind, shown) == pytest.approx(7.0 / (7.0 + k))       # nothing of its own turn: as shown
    assert hata.proven(mind, tried) == pytest.approx(3.0 / (3.0 + k))
    assert hata.proven(mind, both) == pytest.approx((1.0 - 1.0 / (1.0 + k)) * 7.0 / (8.0 + k))
    assert hata.proven(mind, both) < 0.5 < hata.proven(mind, shown)
    assert hata.failed(mind, both) and not hata.failed(mind, shown)         # (one against it in each condition)


def test_what_it_said_on_its_own_turn_stays_kept_when_it_listens_to_that_context():
    for own_turn in (True, False):
        s = R.born("full", own_turn=own_turn)
        mind = s.mind
        for _ in range(2):
            for x in MEMBERS:
                s.watch(*misleading(x))
        s.close_turn()
        q, a = misleading(POOL[0])
        said = s.reply(q, limit=20)
        s.close_turn()
        assert mark(said, a) == "W"                          # it asserts what the watched six share
        s.listen(q, chat.IMITATED)                           # the teacher shows the dialogue: the same question
        s.end_of_turn(chat.IMITATED, tone="?")
        claims = mind.claimed.get(mind.context)              # the context it answered at, lived again
        assert claims
        if own_turn:
            assert any(claim[2] is True for claim in claims)     # what it said there is what is on trial there
        else:
            assert all(claim[2] is None for claim in claims)     # the silent claim of this moment replaced it


def test_a_life_of_a_misleading_and_a_lawful_frame():
    _, pooled, law_pooled, fresh_pooled = life(own_turn=False)
    s, own, law_own, fresh_own = life()
    assert pooled == "WWWWW---"                              # five wrong answers before it stops asserting
    assert own == "WW------"                                 # two
    assert law_pooled == law_own == "R" * len(POOL)          # the lawful frame: right from its first question
    assert fresh_pooled == fresh_own == "-" * len(FRESH)     # and silent on fresh things in the misleading one
    line = [ln for ln in hata.report(s.mind, 40) if "X ne yer ⏎ X)" in ln and "kendi sırasında" in ln][0]
    assert "kendi sırasında 0.0 / 2.0, dinlerken 4.0 / 0.0" in line
    for part in ("held", "record"):                          # each part alone: as without F69
        assert life(part)[1] == pooled


def test_without_silent_claims_the_lawful_frame_is_not_answered_from_the_start():
    _, marks, law, _ = life(silent_claims=False, relied_on=False, own_turn=False)
    assert marks == "W-------" and law == "--WRRRRR"


def test_a_mind_that_only_watches_is_never_asked_and_nothing_changes():
    _, _, _, fresh_pooled = life(asked=False, own_turn=False)
    s, _, _, fresh_own = life(asked=False)
    assert fresh_pooled == fresh_own
    for symbol in [hata.ALL]:
        assert hata.votes(s.mind)                            # (its ways were judged while it watched)
    channel = s.mind.ch.id(hata.CHANNEL)
    turns = 0.0
    for index in range(s.mind.ch.sizes()[channel]):
        symbol = s.mind.ch.symbol(channel, index)
        if symbol not in (hata.EXPECTED, hata.HELD):
            (t_pos, t_neg), _ = hata._cells(s.mind, symbol)
            turns += t_pos + t_neg
    assert turns == 0.0                                      # nothing was ever its turn
