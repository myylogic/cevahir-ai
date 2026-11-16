# -*- coding: utf-8 -*-
"""Zihin, F38 Witness Reliability: a lived suffix that leaves out a fixed item of a frame's claim is a witness of it, by
the deduction of NARS through how far the lived contexts that end with that item and the suffix went on as it did.

What is pinned here
-------------------
* A transparent item: after "merhaba , adın ne ?" and "selam , nasılsın ?" were answered as "adın ne ?" and "nasılsın ?"
  were, "günaydın , adın ne ?" (a greeting lived once, alone) is answered "adım cevahir": the lived "adın ne ?" leaves
  out the frame's comma, and what ended with ", adın ne ?" went on as "adın ne ?" did. Without the switch (F34's strict
  rule: such a suffix never counts) it is not answered.
* An opaque item: in the commands of the reply test, "walk ⏎" of "jump left and walk ⏎" leaves out "left and"; what
  ended with "and walk ⏎" went on otherwise, so it is no witness, and "jump left and walk" is still answered
  "i_lturn i_jump i_walk" (test_reading_an_opaque_item below).
* The evidence of the reliability is the lived contexts that end with the left-out item and the suffix: none, (0, 0);
  one that went on as the suffix did, (1, 1).
* Reading writes nothing; the switch is off by default (also in the speaking configuration: lived with, it changed
  what a SCAN life learned, see the notes).
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import Config, Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS
from zihin.read import witness_reliability


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fresh(witness: bool) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path, cfg.witness = False, None, witness
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    return Session(mind)


def greeted(witness: bool) -> Session:
    s = fresh(witness)
    for _ in range(3):
        s.watch("adın ne ?", "adım cevahir .")
        s.watch("nasılsın ?", "iyiyim , teşekkür ederim .")
        s.watch("merhaba", "merhaba .")
    s.watch("merhaba , nasılsın ?", "iyiyim , teşekkür ederim .")
    s.watch("selam , nasılsın ?", "iyiyim , teşekkür ederim .")
    s.watch("merhaba , adın ne ?", "adım cevahir .")
    s.watch("selam", "selam .")
    s.watch("günaydın", "günaydın .")
    return s


def ask(s: Session, text: str) -> str:
    return Session(s.mind.clone()).reply(text)


def test_the_switch_is_off_by_default():
    assert Config().witness is False and Config.changed().witness is False
    assert VARIANTS["speaking"]().witness is False


@pytest.mark.parametrize("question", ["günaydın , adın ne ?", "selam , adın ne ?"])
def test_a_transparent_item_lets_the_shorter_context_witness(question):
    assert ask(greeted(True), question) == "adım cevahir"
    assert ask(greeted(False), question) == "(sessiz)"


def _tuple(mind, words):            # the words of an utterance and its end (the closing "?" is its tone)
    end = mind.unit_of(chat.TURN, chat.TURN_END)
    return tuple(mind.unit_of(CHANNEL, w) for w in words) + (end,)


def test_the_reliability_is_measured_on_lived_contexts():
    s = greeted(True)
    mind = s.mind
    comma = mind.unit_of(CHANNEL, ",")
    t = _tuple(mind, ["adın", "ne"])
    suffix = mind.S.tuple_index[t]
    present = mind.S.tuple_index[_tuple(mind, ["merhaba", ",", "adın", "ne"])]
    assert witness_reliability(mind, comma, t, suffix, present) == (0.0, 0.0)   # the only one is the present one
    other = mind.S.tuple_index[_tuple(mind, ["merhaba", ",", "nasılsın"])]
    t2 = _tuple(mind, ["nasılsın"])
    agree, pairs = witness_reliability(mind, comma, t2, mind.S.tuple_index[t2], other)
    assert pairs == 1.0 and agree == pytest.approx(1.0)   # "selam , nasılsın ?" went on as "nasılsın ?" did


def test_reading_writes_nothing():
    s = greeted(True)
    mind = s.mind
    digest = mind.S.digest()
    assert ask(s, "günaydın , adın ne ?") == "adım cevahir"
    comma = mind.unit_of(CHANNEL, ",")
    t = _tuple(mind, ["nasılsın"])
    witness_reliability(mind, comma, t, mind.S.tuple_index[t], -1)
    assert mind.S.digest() == digest


ACT = {"walk": "i_walk", "run": "i_run", "look": "i_look", "jump": "i_jump"}


def _answer(command: str) -> str:
    w = command.split()
    if "and" in w:
        j = w.index("and")
        return _answer(" ".join(w[:j])) + " " + _answer(" ".join(w[j + 1:]))
    if w[1:] == ["left"]:
        return "i_lturn " + ACT[w[0]]
    return " ".join([ACT[w[0]]] * (2 if w[1:] == ["twice"] else 1))


def test_reading_an_opaque_item():
    import random
    s = fresh(True)
    known = ["walk", "run", "look"]
    train = known + ["%s twice" % x for x in known] + ["%s and %s" % (x, y) for x in known for y in known if x != y]
    train += ["%s left" % x for x in known] + ["%s left and %s" % (x, y) for x in known for y in known]
    rng = random.Random(0)
    for _ in range(3):
        block = train + ["jump"] * 3
        rng.shuffle(block)
        for c in block:
            s.watch(c, _answer(c) + " .")
    assert ask(s, "jump left and walk") == "i_lturn i_jump i_walk"
