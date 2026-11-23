# -*- coding: utf-8 -*-
"""Zihin, F72 The Subject Carries (``Config.subject_carries``).

What is pinned here
-------------------
* Off by default, a studied mechanism, in use in the full configuration.
* After an answer that told what the mind knew of a thing (F41's topic), an utterance it has nothing to say to is
  heard again inwardly with that thing in the place one of its own frames has there, and it answers from there:
  "aslan nerede yaşar ?" - "aslan ormanda yaşar" - "ne yer ?" is answered "aslan balık yer".
* Nothing is carried where there is nothing to carry: the first utterance of a conversation, a turn after an answer
  that held to no thing, an utterance that names the thing itself, or one that names another thing (that one is
  answered).
* Only the turn right after the answer: two turns later the subject is no longer held.
* While the mind only listens (``Config.subject_heard``, with silent claims): a sentence read right after one in
  which a thought held, that leaves open a place the mind has lived such items with, is heard again inwardly with
  that thought's subject in the place: "fok denizde yaşar ." then "balık yer ." teaches "fok balık yer"; with the
  switch off the mind answers what the others eat. A sentence that names its own subject is not touched.
"""
from __future__ import annotations

import pytest
import torch

import test_zihin_own_race as R
from zihin import Config
from zihin.__main__ import NEEDS, STUDIED, Session, VARIANTS
import zihin.__main__ as chat

THINGS = ["inek", "koyun", "keçi", "at", "eşek", "deve", "aslan", "kurt", "fok", "serçe"]
FOOD = {"inek": "ot", "koyun": "ot", "keçi": "et", "at": "tohum", "eşek": "ot", "deve": "balık", "aslan": "et",
        "kurt": "balık", "fok": "balık", "serçe": "tohum"}
HOME = {"inek": "dağda", "koyun": "ormanda", "keçi": "çölde", "at": "ormanda", "eşek": "dağda", "deve": "çölde",
        "aslan": "ormanda", "kurt": "denizde", "fok": "denizde", "serçe": "dağda"}


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def minds():
    out = {}
    for on in (True, False):
        s = R.born("full", subject_carries=on)
        for _ in range(2):
            for x in THINGS:
                R.tell(s, "%s %s yer ." % (x, FOOD[x]))
                R.tell(s, "%s %s yaşar ." % (x, HOME[x]))
            for x in THINGS[:5]:
                s.watch("%s ne yer ?" % x, "%s %s yer ." % (x, FOOD[x]))
                s.watch("%s nerede yaşar ?" % x, "%s %s yaşar ." % (x, HOME[x]))
        s.close_turn()
        out[on] = s
    return out


def copy(s):
    return Session(s.mind.clone())


def test_off_by_default_and_studied():
    assert Config().subject_heard is False and "subject_heard" in STUDIED
    assert ("subject_heard", ("silent_claims",)) in NEEDS
    assert Config().subject_carries is False and "subject_carries" in STUDIED
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        assert VARIANTS["full"]().subject_carries is True and VARIANTS["speaking"]().subject_carries is False
        assert VARIANTS["full"]().subject_heard is True and VARIANTS["speaking"]().subject_heard is False
    finally:
        chat._log_path = made


def test_the_subject_of_the_answer_carries_into_the_next_question(minds):
    for x in THINGS[5:]:
        on, off = copy(minds[True]), copy(minds[False])
        for c in (on, off):
            assert c.reply("%s nerede yaşar ?" % x, limit=20) == "%s %s yaşar" % (x, HOME[x])
        assert on.held_subject == on.mind.unit_of(R.CHANNEL, x)
        assert on.reply("ne yer ?", limit=20) == "%s %s yer" % (x, FOOD[x])
        assert off.reply("ne yer ?", limit=20) == "(sessiz)"
        c = copy(minds[True])                            # the other order
        c.reply("%s ne yer ?" % x, limit=20)
        assert c.reply("nerede yaşar ?", limit=20) == "%s %s yaşar" % (x, HOME[x])


def test_nothing_is_carried_where_there_is_nothing_to_carry(minds):
    x, y = THINGS[6], THINGS[7]
    assert copy(minds[True]).reply("ne yer ?", limit=20) == "(sessiz)"         # the first utterance
    c = copy(minds[True])
    assert c.reply("%s ne içer ?" % x, limit=20) == "(sessiz)"                 # an answer that held to nothing
    assert c.held_subject is None and c.reply("ne yer ?", limit=20) == "(sessiz)"
    c = copy(minds[True])
    c.reply("%s nerede yaşar ?" % x, limit=20)
    assert c.reply("%s ne yer ?" % y, limit=20) == "%s %s yer" % (y, FOOD[y])  # another thing named: it is answered


def test_only_the_turn_right_after_the_answer(minds):
    x = THINGS[8]
    c = copy(minds[True])
    c.reply("%s nerede yaşar ?" % x, limit=20)
    assert c.reply("merhaba", limit=20) in ("(sessiz)", "merhaba")              # a turn between, holding no thing
    assert c.held_subject is None
    assert c.reply("ne yer ?", limit=20) == "(sessiz)"


def test_a_sentence_heard_with_its_subject_dropped_is_heard_again_with_it():
    """Every food is read whole of some animals; four others are read with the subject of the second sentence
    dropped."""
    answers = {}
    for on in (True, False):
        s = R.born("full", subject_heard=on)
        for _ in range(2):
            for x in THINGS[:8]:
                R.tell(s, "%s %s yaşar ." % (x, HOME[x]))
                R.tell(s, "%s %s yer ." % (x, FOOD[x]))
            for x in THINGS[:5]:
                s.watch("%s ne yer ?" % x, "%s %s yer ." % (x, FOOD[x]))
        for x in ("panter", "tilki"):
            R.tell(s, "%s ormanda yaşar ." % x)
            R.tell(s, "et yer .")
        s.close_turn()
        answers[on] = [Session(s.mind.clone()).reply("%s ne yer ?" % x, limit=20) for x in ("panter", "tilki")]
        answers[on].append(Session(s.mind.clone()).reply("%s ne yer ?" % THINGS[7], limit=20))
    assert answers[True][:2] == ["panter et yer", "tilki et yer"]
    assert all("et" not in a.split() for a in answers[False][:2])         # without it: not what it read
    assert answers[True][2] == answers[False][2] == "%s %s yer" % (THINGS[7], FOOD[THINGS[7]])   # read whole
