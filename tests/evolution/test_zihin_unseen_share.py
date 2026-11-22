# -*- coding: utf-8 -*-
"""Zihin, F60 The Unseen Share: the frame that speaks counts against a word its members never went on with only as
far as its members went on with one another's words.

What is pinned here
-------------------
* The switch is off by default and is a studied mechanism (``--ac families,unseen_share``).
* ``aile.repeated``: the share of a frame's members whose word another member went on with too: 0 where every
  member went on with a word of its own, 1 where all went on with the same.
* The error it was built for: a thing that was asked two kinds of question is asked a third kind that the mind has
  never met. The frame made of that thing's own earlier questions has members (the new kind's frame has none), it
  speaks, and what its members never went on with it counts against: the family's claim of the right word (F53,
  with the asked place, F55) is lowered under 1/2 and not said. With the switch the claim stands and is said.
  A thing never asked about is answered with or without it (no such frame speaks there).
* With the switch off the reading is as before.
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Config, Mind, aile
from zihin.__main__ import CHANNEL, STUDIED, TURN, TURN_END, Session, VARIANTS

THINGS = ["inek", "koyun", "keçi", "at", "eşek", "tavuk", "deve", "geyik", "aslan", "kurt", "fok", "serçe"]
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


def raised(**switches) -> Session:
    """Every fact read once of every thing (in an order of its own); two kinds of question watched twice about the
    same six things."""
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    for name, value in dict({"families": True, "lived_apart": True, "asked_place": True}, **switches).items():
        setattr(cfg, name, value)
    torch.manual_seed(0)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    for i, x in enumerate(THINGS):
        sentences = ["%s %s %s ." % (x, fact(x, q), q) for q in FACTS]
        for sentence in sentences[i % 3:] + sentences[:i % 3]:
            s.company(None)
            s.listen(sentence, IMITATED)
            s.end_of_turn(IMITATED, tone=".")
            mind.step([])
    for _ in range(2):
        for q in ("yer", "içer"):
            for x in THINGS[:6]:
                s.watch("%s ne %s ?" % (x, q), "%s ." % fact(x, q))
    s.close_turn()
    return s


def asks(s: Session, question: str) -> str:
    return Session(s.mind.clone()).reply(question, limit=30)


def test_off_by_default_and_studied():
    assert Config().unseen_share is False and "unseen_share" in STUDIED


def test_a_new_kind_asked_of_a_thing_that_was_asked_other_kinds():
    off, on = raised(), raised(unseen_share=True)
    for x in THINGS[6:9]:                                  # never asked about: answered either way
        assert asks(off, "%s ne taşır ?" % x) == fact(x, "taşır")
        assert asks(on, "%s ne taşır ?" % x) == fact(x, "taşır")
    for x in THINGS[:6]:                                   # asked two kinds before
        assert asks(off, "%s ne taşır ?" % x) == "(sessiz)"     # the error: the claim is lowered and not said
        assert asks(on, "%s ne taşır ?" % x) == fact(x, "taşır")
        assert asks(on, "%s ne yer ?" % x) == fact(x, "yer")    # what it could answer it still does


def test_how_far_the_members_of_a_place_repeat_one_another():
    from zihin import read
    from zihin.store import CATEGORY
    m = raised(unseen_share=True).mind
    end = m.unit_of(TURN, TURN_END)
    kinds = read.mv_view(m, "u_kind")
    inek, ne, yer = (m.unit_of(CHANNEL, w) for w in ("inek", "ne", "yer"))
    asked = m.S.tuple_index[(inek, ne, yer, end)]          # a question that was asked of this thing
    frames = [frame for frame, _ in read.generalisations(m, asked)]
    own = next(f for f in frames if read.parts_of(m, f)[0] == inek and kinds[read.parts_of(m, f)[2]] & CATEGORY)
    assert aile.repeated(m, own, None) == 0.0              # "inek ne X ⏎": each question went on with its own word
    kind = next(f for f in frames if read.parts_of(m, f)[2] == yer and kinds[read.parts_of(m, f)[0]] & CATEGORY)
    assert 0.0 < aile.repeated(m, kind, None) <= 1.0       # "X ne yer ⏎": six things, four values: some repeat
