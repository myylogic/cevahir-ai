# -*- coding: utf-8 -*-
"""Zihin, F53 Frames of Frames: what the mind has generalised is itself generalised, so what it learned about kinds
of question carries over to a new kind.

What is pinned here
-------------------
* The switch is off by default and is one of the studied mechanisms (``--ac families``).
* A small world: things with three kinds of fact, each read once as a sentence ("X V Q ."). Two kinds of question
  were watched being asked and answered ("X ne Q ?" -> "X V Q ."); the third kind was never asked of anything. A
  mind with the switch answers the third kind at first sight, for a thing it was never asked anything about, with
  the fact it read; without the switch it says nothing. A mind that watched no kind of question says nothing
  either way: there is no family to read from.
* Of a thing whose fact of that kind was never read it says the asked word and no value.
* After one watched dialogue of the new kind the answer stands (a frame with one member is nested over its
  family's claims; without the switch one dialogue is not enough).
* A kind of question that looks like the others and is answered otherwise is learned from its own dialogues, and no
  value is made up for it before.
* The family's claims are ways of reading with records of their own (F49), named in the mind's chain.
* The index of families is not kept when the mind is copied or saved.
* A place holds items of one kind: what the mind feels while it speaks (the inner event of looking, held in its
  context where a word then stood) is not read as a word. In F50's world, with a library, the answers after a look
  stay whole and right with the switch on (before this rule the family said the word that follows the value while
  the mind was looking: "arı yer ot yer").
"""
from __future__ import annotations

import pickle

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Config, Mind
from zihin import aile
from zihin.__main__ import CHANNEL, STUDIED, Session, VARIANTS

THINGS = ["inek", "koyun", "keçi", "at", "eşek", "tavuk", "deve", "geyik", "aslan", "kurt", "fok", "serçe"]
FACTS = {"yer": ["ot", "et", "balık", "tohum"], "içer": ["su", "süt", "çay", "şerbet"],
         "taşır": ["dal", "taş", "yaprak", "kum"]}
ASKED = THINGS[:6]


def fact(x: str, q: str) -> str:
    return FACTS[q][(THINGS.index(x) + len(q)) % 4]


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def raised(kinds=("yer", "içer"), unread=(), **switches) -> Session:
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
    for x in THINGS:
        for q in FACTS:
            if q == "taşır" and x in unread:
                continue
            s.company(None)
            s.listen("%s %s %s ." % (x, fact(x, q), q), IMITATED)
            s.end_of_turn(IMITATED, tone=".")
            mind.step([])
    for _ in range(2):
        for q in kinds:
            for x in ASKED:
                s.watch("%s ne %s ?" % (x, q), "%s %s %s ." % (x, fact(x, q), q))
    s.close_turn()
    return s


def asks(s: Session, question: str) -> str:
    return Session(s.mind.clone()).reply(question, limit=30)


ON = dict(known=True, families=True)
OFF = dict(known=True)


def test_off_by_default_and_studied():
    assert Config().families is False and "families" in STUDIED


def test_a_kind_of_question_never_asked_is_answered_at_first_sight():
    on, off = raised(**ON), raised(**OFF)
    for x in ("aslan", "kurt", "fok"):                    # things it was never asked anything about
        assert asks(on, "%s ne taşır ?" % x) == "%s %s taşır" % (x, fact(x, "taşır"))
        assert asks(off, "%s ne taşır ?" % x) == "(sessiz)"
    for x in ("aslan", "inek"):                           # the kinds it mastered are answered as before
        assert asks(on, "%s ne yer ?" % x) == asks(off, "%s ne yer ?" % x) == "%s %s yer" % (x, fact(x, "yer"))


def test_with_no_kind_of_question_mastered_there_is_no_family():
    on = raised(kinds=(), **ON)
    assert asks(on, "aslan ne taşır ?") == "(sessiz)"
    one = raised(kinds=("yer",), **ON)                    # one frame is not a family
    assert asks(one, "aslan ne taşır ?") == "(sessiz)"


def test_of_a_thing_whose_fact_was_never_read_it_makes_no_value_up():
    on = raised(unread=("serçe",), **ON)
    assert asks(on, "serçe ne taşır ?") == "serçe"        # the asked word, and nothing known of it leads to the verb
    assert asks(on, "fok ne taşır ?") == "fok %s taşır" % fact("fok", "taşır")


def test_after_one_dialogue_of_the_new_kind_the_answer_stands():
    on, off = raised(**ON), raised(**OFF)
    for s in (on, off):
        s.watch("inek ne taşır ?", "inek %s taşır ." % fact("inek", "taşır"))
        s.close_turn()
    assert asks(on, "aslan ne taşır ?") == "aslan %s taşır" % fact("aslan", "taşır")
    assert asks(off, "aslan ne taşır ?") == "(sessiz)"    # one instance is no frame


def test_a_kind_that_is_answered_otherwise_is_learned_from_its_own_dialogues():
    on = raised(**ON)
    first = asks(on, "aslan ne bilir ?")
    assert first in ("aslan", "(sessiz)")                 # no value made up at first sight
    for x in ASKED[:4]:
        on.watch("%s ne bilir ?" % x, "hiçbir şey .")
    on.close_turn()
    assert asks(on, "aslan ne bilir ?") == "hiçbir şey"


def test_the_family_claims_are_ways_with_records_of_their_own():
    s = raised(faults=True, **ON)
    copy = Session(s.mind.clone())
    copy.reply("aslan ne taşır ?", limit=30)
    ways = {claim[0][-1] for claims in copy.mind.claimed.values() for claim in claims}
    assert ways & {("aile", -1), ("aile", aile.SECOND)}   # the asked word: the frames' filler, or their varying word
    assert ("aile", aile.LINK) in ways                    # the thing known of it that leads to the other asked word


def test_the_index_of_families_is_not_kept_in_a_copy():
    s = raised(**ON)
    asks(s, "aslan ne taşır ?")
    s.reply("kurt ne taşır ?", limit=30)
    assert "_aileler" in s.mind.__dict__
    assert "_aileler" not in s.mind.clone().__dict__
    assert "_aileler" not in pickle.loads(pickle.dumps(s.mind)).__dict__


class Library:
    """A source that stands where the web stands: one sentence about each thing it holds."""

    def __init__(self, facts: dict) -> None:
        self.facts = dict(facts)

    def search(self, topic: str, limit: int = 2):
        return [topic] if topic in self.facts else []

    def extract(self, identifier: str) -> str:
        return "%s %s yer ." % (identifier, self.facts[identifier])

    def address(self, identifier: str) -> str:
        return "library:" + identifier


def test_what_the_mind_feels_while_it_looks_is_not_read_as_a_word():
    new = {"aslan": "et", "kurt": "et", "fok": "balık", "serçe": "tohum", "geyik": "ot", "deve": "tohum"}
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    for name in ("open_claims", "look", "faults", "find_out", "known", "families"):
        setattr(cfg, name, True)
    torch.manual_seed(0)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    for _ in range(2):
        for x in THINGS[:6]:
            s.company(None)
            s.listen("%s ot yer ." % x, IMITATED)
            s.end_of_turn(IMITATED, tone=".")
            mind.step([])
            s.watch("%s ne yer ?" % x, "%s ot yer ." % x)
        s.watch("merhaba", "merhaba .")
    s.close_turn()
    s.look_gap, s.look_source, s.may_look = 0.0, Library(new), True
    for x, v in new.items():                              # asked, it looks, answers; then the true dialogue is shown
        assert s.reply("%s ne yer ?" % x, limit=30) == "%s %s yer" % (x, v)
        s.close_turn()
        s.watch("%s ne yer ?" % x, "%s %s yer ." % (x, v))
        s.close_turn()
    assert len(s.looks) == len(new)
