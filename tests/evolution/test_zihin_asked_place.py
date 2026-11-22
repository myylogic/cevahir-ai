# -*- coding: utf-8 -*-
"""Zihin, F55 The Asked Place: a question is a statement the mind knows with one item replaced, and the answer is
that item.

What is pinned here
-------------------
* The switch is off by default and is one of the studied mechanisms (``--ac asked_place``).
* ``read.asked_places``: the run of items a thing begins in a context, read against what is known of the thing
  (the sequences that began with it, without what was only ever asked of it). A known sequence that is the run in
  every item but one names what stands at that place. Another question about the same thing is not what is known
  of it; a fact of another kind differs in two places and names nothing.
* The error this was built for: the answer is the value alone ("X ne yer ?" -> "V ."), and two facts were read of
  every thing. With F41 alone a fresh thing asked the mastered kind is not answered (F41 goes on from the thing's
  name, and the name is not in the answer); with the switch it is, with the fact of the asked kind.
* Nothing is made up: of a thing nothing was read of, nothing is said.
* Two things at the asked place (two facts of the asked kind read of a thing): each is an answer. For a context
  being read the most lived is named, among equals the one learned first; for a member of a frame the relation
  holds where its value is any of them. (Before this, none was named, and a value two watched things happened to
  share was said of a thing that had neither: 4 of 40.)
* Where the answer names the thing first ("X V yer .") the same relation names the value there too.
* With F53 and F54: a new kind of question, answered with the value alone, is answered at first sight. Where the
  answers name the thing first, the whole answer still comes: the value is named by two kinds of family claim (the
  asked place, and the thing known that goes on to the asked word), both are kept, and the second makes the thing
  the answer's topic (found with this switch on in every test: the answer had stopped at the value).
* The claim is a way of reading with a record of its own (F49), named in the mind's chain.
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Config, Mind, read
from zihin.__main__ import CHANNEL, STUDIED, TURN, TURN_END, Session, VARIANTS

THINGS = ["inek", "koyun", "keçi", "at", "eşek", "tavuk", "deve", "geyik", "aslan", "kurt", "fok", "serçe",
          "kaz", "arı", "tilki", "ayı"]
FACTS = {"yer": ["ot", "et", "balık", "tohum"], "içer": ["su", "süt", "çay", "şerbet"],
         "taşır": ["dal", "taş", "yaprak", "kum"]}


def fact(x: str, q: str) -> str:
    return FACTS[q][(THINGS.index(x) * 3 + len(q)) % 4]


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def raised(asked=("yer",), echo=False, read_facts=True, **switches) -> Session:
    """Every fact read once of every thing; the kinds of ``asked`` watched twice about six things."""
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    for name, value in switches.items():
        setattr(cfg, name, value)
    torch.manual_seed(0)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    for x in THINGS if read_facts else THINGS[:6]:
        for q in FACTS:
            s.company(None)
            s.listen("%s %s %s ." % (x, fact(x, q), q), IMITATED)
            s.end_of_turn(IMITATED, tone=".")
            mind.step([])
    for _ in range(2):
        for q in asked:
            for x in THINGS[:6]:
                s.watch("%s ne %s ?" % (x, q), ("%s %s %s ." % (x, fact(x, q), q)) if echo else "%s ." % fact(x, q))
    s.close_turn()
    return s


def asks(s: Session, question: str) -> str:
    return Session(s.mind.clone()).reply(question, limit=30)


def unit(m, w):
    return m.unit_of(TURN, TURN_END) if w == TURN_END else m.unit_of(CHANNEL, w)


def test_off_by_default_and_studied():
    assert Config().asked_place is False and "asked_place" in STUDIED


def test_the_place_where_what_was_asked_and_what_is_known_differ():
    s = raised(asked=("yer", "içer"), known=True, asked_place=True)
    m = s.mind
    x = unit(m, "inek")                                    # a thing that was asked two kinds of question
    instance = m.S.tuple_index[(x, unit(m, "ne"), unit(m, "yer"), unit(m, TURN_END))]
    found = read.asked_places(m, x, instance, 0)
    assert found == {unit(m, fact("inek", "yer")): pytest.approx(0.5)}     # the sentence read, once; not "içer"
    assert read.asked_place(m, x, instance, 0) == unit(m, fact("inek", "yer"))
    copy = Session(m.clone())                              # a thing never asked about: the question as it is heard
    copy.listen("aslan ne taşır ?", IMITATED)
    mind = copy.mind
    a = unit(mind, "aslan")
    assert read.asked_place(mind, a, mind.context, 0) == unit(mind, fact("aslan", "taşır"))


def test_the_value_alone_and_two_facts_read_of_every_thing():
    off, on = raised(known=True), raised(known=True, asked_place=True)
    for x in ("aslan", "kurt", "fok"):                     # things it was never asked about
        assert asks(off, "%s ne yer ?" % x) == "(sessiz)"  # the error: F41 goes on from a name that is not said
        assert asks(on, "%s ne yer ?" % x) == fact(x, "yer")
    assert asks(on, "inek ne yer ?") == asks(off, "inek ne yer ?") == fact("inek", "yer")


def test_of_a_thing_nothing_was_read_of_nothing_is_said():
    on = raised(read_facts=False, known=True, asked_place=True)
    assert asks(on, "aslan ne yer ?") == "(sessiz)"
    assert asks(on, "inek ne yer ?") == fact("inek", "yer")


def test_two_things_at_the_asked_place():
    s = raised(known=True, asked_place=True)
    s.company(None)
    other = next(v for v in FACTS["yer"] if v != fact("aslan", "yer"))
    s.listen("aslan %s yer ." % other, IMITATED)           # a second fact of the asked kind about one thing
    s.end_of_turn(IMITATED, tone=".")
    s.mind.step([])
    copy = Session(s.mind.clone())
    copy.listen("aslan ne yer ?", IMITATED)
    m = copy.mind
    a, first, second = unit(m, "aslan"), unit(m, fact("aslan", "yer")), unit(m, other)
    found = read.asked_places(m, a, m.context, 0)
    assert found == {first: pytest.approx(0.5), second: pytest.approx(0.5)}
    assert read.asked_place(m, a, m.context, 0) == min(first, second)      # lived equally: the one learned first
    assert asks(s, "aslan ne yer ?") in (fact("aslan", "yer"), other)


def test_where_the_answer_names_the_thing_first():
    on = raised(echo=True, known=True, asked_place=True, lived_apart=True)
    for x in ("aslan", "kurt"):
        assert asks(on, "%s ne yer ?" % x) == "%s %s yer" % (x, fact(x, "yer"))


def test_a_new_kind_answered_with_the_value_alone_at_first_sight():
    switches = dict(known=True, families=True, lived_apart=True)
    off, on = raised(asked=("yer", "içer"), **switches), raised(asked=("yer", "içer"), asked_place=True, **switches)
    for x in ("aslan", "kurt", "fok"):
        assert asks(on, "%s ne taşır ?" % x) == fact(x, "taşır")           # never asked of anything
        assert asks(off, "%s ne taşır ?" % x) == "(sessiz)"
        assert asks(on, "%s ne içer ?" % x) == fact(x, "içer")


def test_with_the_family_the_whole_answer_still_comes():
    s = raised(asked=("yer", "içer"), echo=True, known=True, families=True, lived_apart=True, asked_place=True,
               faults=True)
    for x in ("aslan", "kurt", "fok"):
        assert asks(s, "%s ne taşır ?" % x) == "%s %s taşır" % (x, fact(x, "taşır"))
    copy = Session(s.mind.clone())
    copy.reply("aslan ne taşır ?", limit=30)
    ways = {claim[0][-1] for claims in copy.mind.claimed.values() for claim in claims}
    assert {("aile", read.ASKED), ("aile", -8)} <= ways    # both kinds that named the value (-8: aile.LINK)


def test_the_claim_is_a_way_with_a_record_of_its_own():
    s = raised(known=True, asked_place=True, faults=True)
    copy = Session(s.mind.clone())
    copy.reply("aslan ne yer ?", limit=30)
    ways = {claim[0][1] for claims in copy.mind.claimed.values() for claim in claims}
    assert ("g", read.ASKED) in ways
