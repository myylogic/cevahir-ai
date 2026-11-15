# -*- coding: utf-8 -*-
"""Zihin, F26 Sequence Starts: how sequences that began with a word went on, apart from what followed it anywhere.

What is pinned here
-------------------
* ``start_mass`` counts beginnings exactly: a composed unit of two items (x, v) is recognised once in every episode that
  began x, v, each recognition is evidence on its PART ties, and their trust-weighted mass is what began so; what began
  with x and went on otherwise counts against v (its rivals' share). Hand-checkable: read sentences (imitated, trust 1/2)
  "kedi süt içer" once and "kedi su içer" twice give W(süt) = (1/2, 1) and W(su) = (1, 1/2); "bu kedi uyur" adds nothing
  (kedi did not begin it).
* A word brought to mind after a pause (the inner voice, think) goes on as a sequence that began with it went on, not
  as the word went on anywhere: "kedi" read three times inside "bu kedi uyur" and once at the start of "kedi süt içer"
  is thought on with "süt içer" (with the switch off the mind's race is silent there).
* An answer that repeats the asked word goes on as a sequence that began with that word: watched "elma nedir ?" /
  "elma kırmızı bir meyvedir ." and two more such pairs, then read the sentences and "muz uzun bir meyvedir ." once and
  "bu muz tatlı ." three times, "muz nedir ?" is answered "muz uzun bir meyvedir" (the members agree on START); with the
  switch off the answer goes on with what followed muz most often ("muz tatlı").
* With the switch off nothing of this is read: ``starts`` defaults to False, a chat mind (the speaking configuration)
  is raised with it, and a saved mind keeps its own configuration (the program's --ac turns it on for a mind raised
  before it).
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Config, Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS
from zihin.dusunce import think
from zihin.read import start_continuations, start_mass


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fresh(starts: bool) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path, cfg.starts = False, None, starts
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    return Session(mind)


def test_the_switch_is_off_by_default_and_on_for_a_chat_mind():
    assert Config().starts is False and Config.changed().starts is False
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        speaking = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    assert speaking.starts is True and speaking.exclusive is True and speaking.exact_relations is True
    assert speaking.analogy is True and speaking.corroborate is True
    assert {"starts", "exclusive", "exact_relations", "analogy", "corroborate"} <= set(chat.NEWER)   # a mind raised
                                                                                     # before them lives without them


def test_beginnings_are_counted_exactly():
    s = fresh(True)
    s.live("kedi süt içer .", IMITATED)
    for _ in range(2):
        s.live("kedi su içer .", IMITATED)
    s.live("bu kedi uyur .", IMITATED)
    mind = s.mind
    kedi, sut, su = (mind.unit_of(CHANNEL, w) for w in ("kedi", "süt", "su"))
    began = start_mass(mind, kedi)
    assert set(began) == {sut, su}
    assert began[sut] == pytest.approx((0.5, 1.0), abs=1e-12)
    assert began[su] == pytest.approx((1.0, 0.5), abs=1e-12)
    e = start_continuations(mind, kedi)
    assert e[su] > 0.5 > e[sut]
    assert start_mass(mind, mind.unit_of(CHANNEL, "uyur")) == {}     # nothing began with uyur


@pytest.mark.parametrize("starts, thought", [(True, ["süt", "içer"]), (False, [])])
def test_a_word_brought_to_mind_goes_on_as_a_sequence_that_began_with_it(starts, thought):
    s = fresh(starts)
    for _ in range(3):
        s.live("bu kedi uyur .", IMITATED)
    s.live("kedi süt içer .", IMITATED)
    assert think(s, "kedi")[0] == thought


@pytest.mark.parametrize("starts, answer", [(True, "muz uzun bir meyvedir"), (False, "muz tatlı")])
def test_an_answer_goes_on_as_the_sequence_that_began_with_the_asked_word(starts, answer):
    s = fresh(starts)
    for q, a in (("elma nedir ?", "elma kırmızı bir meyvedir ."), ("armut nedir ?", "armut sarı bir meyvedir ."),
                 ("kiraz nedir ?", "kiraz küçük bir meyvedir .")):
        s.watch(q, a)
    for t in ("elma kırmızı bir meyvedir .", "armut sarı bir meyvedir .", "kiraz küçük bir meyvedir .",
              "muz uzun bir meyvedir .", "bu muz tatlı .", "bu muz tatlı .", "bu muz tatlı ."):
        s.live(t, IMITATED)
    assert s.reply("muz nedir ?") == answer
