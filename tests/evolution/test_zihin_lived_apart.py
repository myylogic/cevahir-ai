# -*- coding: utf-8 -*-
"""Zihin, F54 Relations Lived Apart: a frame's relation "what the filler goes on with" (NEXT) counts for a member
only as far as the member lived it apart from its own instance in the frame.

What is pinned here
-------------------
* The switch is off by default and is one of the studied mechanisms (``--ac lived_apart``).
* Where an answer repeats the asked word and goes on ("X ne yer ?" -> "X V yer ."), the instance's continuation is
  the member's continuation: one event, written on both. ``read.lived_apart`` takes the instance's own evidence out
  of the member's: what is left is what it lived elsewhere (the sentence it read).
* The error this was found by: two facts read of every thing, one kind of question mastered. With F41 alone a
  fresh thing asked the mastered kind gets its name and nothing more (NEXT held for every member by the dialogue
  itself, and names the other fact of the fresh thing, which ties with what is known of it); with the switch it is
  answered.
* A relation that was lived apart still holds: one fact read of every thing, and the fresh thing is answered with
  and without the switch.
* Where the answer does not repeat the asked word, the instance's continuation is not the member's: nothing
  changes.
* With F53: new kinds of question learned one after another stay answered (without the switch the family comes to
  vote for NEXT, and the kinds learned earlier go silent).
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
         "sever": ["oyunu", "uykuyu", "güneşi", "yağmuru"], "giyer": ["kürk", "tüy", "pul", "kabuk"],
         "taşır": ["dal", "taş", "yaprak", "kum"]}


def fact(x: str, q: str) -> str:
    return FACTS[q][(THINGS.index(x) * 3 + len(q)) % 4]


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def session(**switches) -> Session:
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
    return Session(mind)


def read_facts(s: Session, kinds) -> None:
    for x in THINGS:
        for q in kinds:
            s.company(None)
            s.listen("%s %s %s ." % (x, fact(x, q), q), IMITATED)
            s.end_of_turn(IMITATED, tone=".")
            s.mind.step([])


def raised(kinds=("yer", "taşır"), echo=True, **switches) -> Session:
    """Every fact of ``kinds`` read once of every thing; "X ne yer ?" watched twice about six things."""
    s = session(**switches)
    read_facts(s, kinds)
    for _ in range(2):
        for x in THINGS[:6]:
            s.watch("%s ne yer ?" % x, ("%s %s yer ." % (x, fact(x, "yer"))) if echo else "%s ." % fact(x, "yer"))
    s.close_turn()
    return s


def asks(s: Session, question: str) -> str:
    return Session(s.mind.clone()).reply(question, limit=30)


def test_off_by_default_and_studied():
    assert Config().lived_apart is False and "lived_apart" in STUDIED


def test_what_a_member_lived_apart_is_what_it_lived_outside_its_own_instance():
    s = raised(known=True, lived_apart=True)
    m = s.mind

    def unit(w):
        return m.unit_of(TURN, TURN_END) if w == TURN_END else m.unit_of(CHANNEL, w)
    x, eats, carries = unit("inek"), unit(fact("inek", "yer")), unit(fact("inek", "taşır"))
    fixed = (unit("ne"), unit("yer"))
    instance = m.S.tuple_index[(x, unit("ne"), unit("yer"), unit(TURN_END), x)]
    own = dict(zip(*read._world_pos(m, instance)))
    alone = {v: p for v, (p, _) in read.own_mass(m, x, 1).items()}
    assert own == {eats: pytest.approx(1.0)}                   # the two watched answers
    assert alone[eats] == pytest.approx(1.5) and alone[carries] == pytest.approx(0.5)   # and the sentence read
    apart = read.lived_apart(m, x, instance, fixed)
    assert apart == {eats: pytest.approx(0.5), carries: pytest.approx(0.5)}             # the two sentences read
    assert read.resolve(m, read.NEXT, x, fixed) == eats        # by its whole record: the dialogue's own answer
    assert read.next_apart(m, x, fixed, instance) is None      # apart: two it went on with equally, none strongest


def test_one_kind_mastered_and_two_facts_read_of_every_thing():
    off, on = raised(known=True), raised(known=True, lived_apart=True)
    for x in ("aslan", "kurt", "fok"):                         # things it was never asked about
        assert asks(off, "%s ne yer ?" % x) == x               # the error: its name and nothing more
        assert asks(on, "%s ne yer ?" % x) == "%s %s yer" % (x, fact(x, "yer"))
    assert asks(on, "inek ne yer ?") == asks(off, "inek ne yer ?") == "inek %s yer" % fact("inek", "yer")


def test_a_relation_that_was_lived_apart_still_holds():
    for switches in ({}, {"known": True}):
        off, on = raised(kinds=("yer",), **switches), raised(kinds=("yer",), lived_apart=True, **switches)
        for x in ("aslan", "kurt"):
            assert asks(on, "%s ne yer ?" % x) == asks(off, "%s ne yer ?" % x) == "%s %s yer" % (x, fact(x, "yer"))


def test_where_the_answer_does_not_repeat_the_asked_word_nothing_changes():
    off, on = raised(echo=False), raised(echo=False, lived_apart=True)
    for x in ("aslan", "inek"):
        a, b = off.mind.clone(), on.mind.clone()
        for mind in (a, b):
            Session(mind).reply("%s ne yer ?" % x, limit=1)
        assert a.expected(CHANNEL) == b.expected(CHANNEL)
        assert asks(off, "%s ne yer ?" % x) == asks(on, "%s ne yer ?" % x) == fact(x, "yer")


def test_new_kinds_learned_one_after_another_stay_answered():
    kinds = list(FACTS)

    def life(**switches):
        s = session(known=True, families=True, **switches)
        read_facts(s, kinds)
        for _ in range(2):
            for q in kinds[:2]:
                for x in THINGS[:5]:
                    s.watch("%s ne %s ?" % (x, q), "%s %s %s ." % (x, fact(x, q), q))
        s.close_turn()
        for j, q in enumerate(kinds[2:]):                      # the teacher's loop: asked, then shown
            for x in THINGS[5 + 2 * j:7 + 2 * j]:
                s.reply("%s ne %s ?" % (x, q), limit=30)
                s.close_turn()
                s.watch("%s ne %s ?" % (x, q), "%s %s %s ." % (x, fact(x, q), q))
                s.close_turn()
        return s
    off, on = life(), life(lived_apart=True)
    for q in kinds:
        assert asks(on, "kaz ne %s ?" % q) == "kaz %s %s" % (fact("kaz", q), q)
    assert asks(off, "kaz ne sever ?") == "kaz"                # the error: a kind learned earlier went silent
