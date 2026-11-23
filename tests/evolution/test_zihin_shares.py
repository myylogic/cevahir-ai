# -*- coding: utf-8 -*-
"""Zihin, F64 A Share Is Tested Before It Is Carried (``Config.shares_tested``, with told_apart).

What is pinned here
-------------------
* Off by default, a studied mechanism, in use in the full configuration; it needs F45 (told_apart).
* F45 tells apart what a word never said alone answers with, between the words it stood between. Each time a word
  is told apart in a place new for it, what it answered with before was a prediction for that place: it held or it
  did not, one experience on the word's own record and on the record of any share.
* Where shares were found to carry (directions that answer alike after every move), a word told apart stands as a
  whole in an utterance never answered, exactly as with F45 alone: "a round back" is answered.
* Where a share was found in one place only and nothing says that shares carry, it is not carried: asked a kind of
  question it never watched of a thing, the mind does not answer with what that thing answered another kind with
  (which F45 alone does), and says what it read of the thing.
* Shares that differ from place to place count against carrying.
* The places and the record are the mind's own: kept when it is saved and loaded, and taken back with a moment
  that is undone.
"""
from __future__ import annotations

import pytest
import torch

import test_zihin_wholes as Wh
import zihin.__main__ as chat
from zihin import IMITATED, Config, persist
from zihin import butun as B
from zihin.__main__ import NEEDS, STUDIED, Session, VARIANTS


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_off_by_default_studied_and_needs_told_apart():
    assert Config().shares_tested is False and "shares_tested" in STUDIED
    assert ("shares_tested", ("told_apart",)) in NEEDS
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        assert VARIANTS["full"]().shares_tested is True and VARIANTS["speaking"]().shares_tested is False
    finally:
        chat._log_path = made


# ----------------------------------------------------------------------------- shares that carry are carried
def test_a_share_found_to_carry_is_carried_as_before():
    plain = Wh.directions(told_apart=True)
    tested = Wh.directions(told_apart=True, shares_tested=True)
    mind = tested.mind
    for word in ("left", "right", "back"):                   # each was told apart after every move and after "go"
        assert len(mind.told_in[word]) >= 4
        assert B.share_carries(mind, word)
    held, failed = B._share_record(mind, B.ANY_SHARE)
    assert held >= 6.0 and failed == 0.0                     # in every new place the share was the one found before
    assert plain.mind.told_in == {}                          # (F45 alone keeps no places and no record)
    for command in ("a round back", "c round back", "c twice", "b round left"):
        assert Wh.ask(tested, command) == Wh.ask(plain, command), command
    assert Wh.ask(tested, "a round back") == "tk ia tk ia"


def test_the_first_place_is_no_test():
    s = Wh.fresh(window=9, wholes=True, told_apart=True, shares_tested=True)
    for m in "ab":
        s.watch(m, "i%s ." % m)
    s.watch("a left", "tl ia .")
    s.watch("a right", "tr ia .")                            # told apart: left -> tl, right -> tr, in one place
    mind = s.mind
    assert B.lived_reply(mind, ("left",)) == ("tl",) and mind.told_in["left"] == (((("a",), ())),)
    assert B._share_record(mind, B.ANY_SHARE) == (0.0, 0.0) and not B.share_carries(mind, "left")
    s.watch("b left", "tl ib .")
    s.watch("b right", "tr ib .")                            # a second place: what they answered with held there
    assert B._share_record(mind, ("w", "left")) == (1.0, 0.0)
    assert B._share_record(mind, B.ANY_SHARE) == (2.0, 0.0)
    assert B.share_carries(mind, "left")                     # (by the record of any share: two for, none against)
    s.watch("b left", "tl ib .")                             # the same place again is no new test
    assert B._share_record(mind, B.ANY_SHARE) == (2.0, 0.0)


# ----------------------------------------------------------------------------- a share found in one place
THINGS = ["inek", "koyun", "keçi", "at", "eşek", "tavuk", "deve", "geyik"]
FACTS = {"yer": ["ot", "et", "balık", "tohum"], "içer": ["su", "süt", "çay", "şerbet"]}


def fact(x: str, q: str) -> str:
    return FACTS[q][(THINGS.index(x) * 3 + len(q)) % 4]


def asked_once(same: bool = False, **switches) -> Session:
    """Both facts read of every thing. Watched: what the first four eat and what the last four drink (``same``: what
    the first four eat and drink), answered with the value alone."""
    s = Wh.fresh(window=9, wholes=True, told_apart=True, known=True, families=True, lived_apart=True,
                 asked_place=True, **switches)
    for i, x in enumerate(THINGS):
        sentences = ["%s %s %s ." % (x, fact(x, q), q) for q in FACTS]
        for sentence in sentences[i % 2:] + sentences[:i % 2]:
            s.company(None)
            s.listen(sentence, IMITATED)
            s.end_of_turn(IMITATED, tone=".")
            s.mind.step([])
    for _ in range(2):
        for q, of in (("yer", THINGS[:4]), ("içer", THINGS[:4] if same else THINGS[4:])):
            for x in of:
                s.watch("%s ne %s ?" % (x, q), "%s ." % fact(x, q))
    s.close_turn()
    return s


def test_a_share_found_in_one_place_is_not_carried():
    """Each thing was asked one kind of question: what it answered with was told apart between "ne yer" (or between
    "ne içer") only. Asked the other kind, F45 alone answers with that share; with F64 it is not carried, and the
    mind says what it read of the thing."""
    alone, tested = asked_once(), asked_once(shares_tested=True)
    x = THINGS[0]
    assert B.lived_reply(alone.mind, (x,)) == (fact(x, "yer"),)          # its share, found between "ne yer"
    assert not B.share_carries(tested.mind, x) and B._share_record(tested.mind, B.ANY_SHARE) == (0.0, 0.0)
    wrong = [Wh.ask(alone, "%s ne içer ?" % t) for t in THINGS[:4]]
    assert wrong == [fact(t, "yer") for t in THINGS[:4]]                 # F45 alone: what it eats, for what it drinks
    assert [Wh.ask(tested, "%s ne içer ?" % t) for t in THINGS[:4]] == [fact(t, "içer") for t in THINGS[:4]]
    assert [Wh.ask(tested, "%s ne yer ?" % t) for t in THINGS[4:]] == [fact(t, "yer") for t in THINGS[4:]]
    assert Wh.ask(tested, "%s ne yer ?" % x) == fact(x, "yer")           # (what was answered before is remembered)


def test_shares_that_differ_from_place_to_place_count_against():
    s = asked_once(same=True, shares_tested=True)            # the same four things asked both kinds
    mind = s.mind
    held, failed = B._share_record(mind, B.ANY_SHARE)
    assert failed >= 2.0 and held == 0.0                     # what a thing eats is not what it drinks
    assert not any(B.share_carries(mind, x) for x in THINGS[:4])
    for x in THINGS[:4]:
        assert B._share_record(mind, ("w", x)) == (0.0, 1.0) and len(mind.told_in[x]) == 2


# ----------------------------------------------------------------------------- the mind's own
def test_kept_with_the_mind_and_taken_back_with_a_moment(tmp_path):
    s = Wh.directions(told_apart=True, shares_tested=True)
    path = str(tmp_path / "paylar")
    persist.save(s.mind, path)
    loaded = persist.load(path, ledger=False)
    assert loaded.told_in == s.mind.told_in and loaded.cfg.shares_tested is True
    assert B._share_record(loaded, B.ANY_SHARE) == B._share_record(s.mind, B.ANY_SHARE)
    assert Wh.ask(Session(loaded), "a round back") == "tk ia tk ia"
    old = persist.load(path, ledger=False)
    del old.told_in                                          # a mind saved before F64
    persist.save(old, path)
    assert persist.load(path, ledger=False).told_in == {}
