# -*- coding: utf-8 -*-
"""Zihin, F40 Urge to Go On: a leading word whose rival is itself expected races against silence as far as the two are
things known about the same item (their share of that item's lived continuations on the words' channel).

What is pinned here
-------------------
* Two facts read about one thing ("baykuş uçar .", "baykuş fare yer .") are things known about "baykuş": after
  "... baykuş" the urge of "uçar" and "fare" is their whole share of what "baykuş" went on with, 1. Two words that no
  item of the context went on with together have no urge.
* In the purpose world (T-purpose's raising), the thing never handed that the mind met in silence is answered: its two
  polite words were both expected and both are what "teşekkür" goes on with.
* The switch is off by default, and a rival that is not itself over the speaking line is never weighed so.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Config, Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS
from zihin.act import _urge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "research", "living_mind_2026_11_15"))


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fresh(urge: bool) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path, cfg.urge = False, None, urge
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    return Session(mind)


def read(s: Session, sentence: str) -> None:
    s.company(None)
    s.listen(sentence, IMITATED)
    s.end_of_turn(IMITATED, tone=s.tone(sentence))
    s.mind.step([])


def test_the_switch_is_off_by_default():
    assert Config().urge is False and Config.changed().urge is False
    assert VARIANTS["speaking"]().urge is False


def test_two_things_known_about_one_item_have_its_whole_urge():
    s = fresh(True)
    for _ in range(2):
        read(s, "baykuş uçar .")
        read(s, "baykuş fare yer .")
        read(s, "hamsi yüzer .")
        read(s, "levrek yüzer .")
    mind = s.mind
    s.listen("baykuş", IMITATED)                         # the present context ends with "baykuş"
    w = lambda x: mind.unit_of(CHANNEL, x)
    assert _urge(mind, w("uçar"), w("fare")) == pytest.approx(1.0)
    assert _urge(mind, w("hamsi"), w("levrek")) == 0.0


def test_a_thing_met_in_silence_is_answered():
    import t_purpose as P
    with_urge = P.run(0, {"urge": True})
    without = P.run(0, {})
    assert "(sessiz)" in without["new"] and "(sessiz)" not in with_urge["new"]
    assert all(r.split()[:1] == ["teşekkür"] for r in with_urge["new"])
