# -*- coding: utf-8 -*-
"""Zihin, F34 Reply Relation: a frame's value may be what its filler, said on its own, was answered with; several
places after a prefix are bound to the filler; a frame's claim is corroborated by the lived suffixes of its context.

What is pinned here
-------------------
* The reply of a word is read from the lived context "word, end of turn": watched "jump" -> "i_jump ." it is i_jump.
* Compositional generalisation in the manner of SCAN's "add primitive" split: commands of walk, run and look in a few
  templates ("X", "X twice", "X and Y") and "jump" only alone are watched; "jump twice" is then answered "i_jump
  i_jump", "walk and jump" "i_walk i_jump", "jump and run" "i_jump i_run". The members of each template's frame agree
  that each action is what their primitive alone was answered with, and every action already said is bound to the
  primitive. Without the switch none of them is answered so.
* Corroboration reads the lived suffixes of the context that reach every fixed word of the frame: "peki adın ne ?" is
  still answered (the lived "adın ne ?" is such a suffix), while "walk ⏎", a suffix of "jump left and walk ⏎" that
  leaves out "left" and "and", is another command and does not override the frame ("jump left and walk" is answered
  "i_lturn i_jump i_walk").
* Reading writes nothing, and the switch is off by default.
"""
from __future__ import annotations

import random

import pytest
import torch

import zihin.__main__ as chat
from zihin import Config, Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS
from zihin.read import members_map, reply_of


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fresh(replies: bool) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path, cfg.replies = False, None, replies
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    return Session(mind)


ACT = {"walk": "i_walk", "run": "i_run", "look": "i_look", "jump": "i_jump"}


def answer(command: str) -> str:
    w = command.split()
    if "and" in w:
        j = w.index("and")
        return answer(" ".join(w[:j])) + " " + answer(" ".join(w[j + 1:]))
    if w[1:] == ["left"]:
        return "i_lturn " + ACT[w[0]]
    return " ".join([ACT[w[0]]] * (2 if w[1:] == ["twice"] else 1))


def commanded(replies: bool) -> Session:
    s = fresh(replies)
    known = ["walk", "run", "look"]
    train = known + ["%s twice" % x for x in known] + ["%s and %s" % (x, y) for x in known for y in known if x != y]
    train += ["%s left" % x for x in known] + ["%s left and %s" % (x, y) for x in known for y in known]
    rng = random.Random(0)
    for _ in range(3):
        block = train + ["jump"] * 3
        rng.shuffle(block)
        for c in block:
            s.watch(c, answer(c) + " .")
    return s


def ask(s: Session, text: str) -> str:
    return Session(s.mind.clone()).reply(text)


def test_the_switch_is_off_by_default():
    assert Config().replies is False and Config.changed().replies is False


def test_the_reply_of_a_word_is_what_it_alone_was_answered_with():
    s = fresh(True)
    for _ in range(2):
        s.watch("jump", "i_jump .")
    mind = s.mind
    end = mind.unit_of(chat.TURN, chat.TURN_END)
    assert reply_of(mind, mind.unit_of(CHANNEL, "jump"), (end,)) == mind.unit_of(CHANNEL, "i_jump")


@pytest.mark.parametrize("command", ["jump twice", "walk and jump", "jump and run", "jump left and walk"])
def test_a_primitive_seen_alone_is_used_in_every_template(command):
    assert ask(commanded(True), command) == answer(command)
    assert ask(commanded(False), command) != answer(command)


def test_corroboration_reads_the_lived_suffixes():
    s = fresh(True)
    for _ in range(3):
        s.watch("adın ne ?", "adım cevahir .")
    s.watch("senin adın ne ?", "adım cevahir .")
    for _ in range(2):
        s.watch("bugün ne yedin ?", "makarna yedim .")
        s.watch("en sevdiğin renk ne ?", "en sevdiğim renk mavi .")
    assert ask(s, "peki adın ne ?") == "adım cevahir"


def test_reading_writes_nothing():
    s = commanded(True)
    s.reply("jump twice")
    mind = s.mind
    end = mind.unit_of(chat.TURN, chat.TURN_END)
    words = [mind.unit_of(CHANNEL, w) for w in ("jump", "twice")]
    context = mind.S.tuple_index[tuple(words) + (end,)]
    digest = mind.S.digest()
    read = members_map(mind, context)
    assert mind.S.digest() == digest
    assert read[mind.unit_of(CHANNEL, "i_jump")] > 0.5 + 1.0 / 12
