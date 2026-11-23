# -*- coding: utf-8 -*-
"""Zihin, F75 The Reach of a Continuation (``Config.continuation_reach``).

What is pinned here
-------------------
* Off by default, and not in the whole mind: it was measured and not kept there (notes of 2026-11-22, section 41):
  the study's small mind said 10 of 41 untaught definitions instead of 4, the battery changed one line for the better,
  and the small world below goes silent with it: the ways it opens are claimed with while the mind listens (F49's
  silent claims), each is one voice in the vote of every way where the one way "devam" was one, and where they fail
  there every reading is believed less (the answer is not even begun).
* With the switch on, a continuation read from the lived sequence its last items match is a way of its own for each
  length of that match ("devam2", "devam3", ...), so F49 keeps their records apart; off, every continuation is the one
  way "devam".
* Off, the mind answers as it did before the switch existed.
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Config, Mind, read
from zihin.__main__ import CHANNEL, STUDIED, Session, VARIANTS

FACTS = ["güneş bir yıldızdır ve ısı verir .", "ay dünyanın uydusudur ve ışık yansıtır .",
         "mars kızıl bir gezegendir ve soğuktur .", "venüs sıcak bir gezegendir ve parlaktır ."]


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _spying():
    """Count the continuation ways read (``read._take``), {name: times}; returns (seen, undo)."""
    seen = {}
    original = read._take

    def spy(mind, out, src, reading, name):
        if name.startswith("devam") and reading:
            seen[name] = seen.get(name, 0) + 1
        return original(mind, out, src, reading, name)
    read._take = spy

    def undo():
        read._take = original
    return seen, undo


def raised(reach: bool):
    """Four facts read twice; two of them asked and answered ("X nedir ?"), the other two never asked. Returns (the
    session, the continuation ways read while it lived)."""
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["full"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    cfg.continuation_reach = reach
    torch.manual_seed(0)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    seen, undo = _spying()
    try:
        with torch.no_grad():
            for _ in range(2):
                for fact in FACTS:
                    s.company(None)
                    s.listen(fact, IMITATED)
                    s.end_of_turn(IMITATED, tone=".")
                    mind.step([])
            s.watch("güneş nedir ?", "güneş bir yıldızdır ve ısı verir .")
            s.watch("ay nedir ?", "ay dünyanın uydusudur ve ışık yansıtır .")
    finally:
        undo()
    return s, dict(seen)


def answer(s: Session, question: str) -> str:
    with torch.no_grad():
        return Session(s.mind.clone()).reply(question, limit=30)


@pytest.fixture(scope="module")
def minds():
    return {reach: raised(reach) for reach in (False, True)}


def test_off_by_default_and_not_in_the_whole_mind():
    assert Config().continuation_reach is False
    assert "continuation_reach" not in STUDIED
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        assert VARIANTS["full"]().continuation_reach is False
    finally:
        chat._log_path = made


def test_off_every_continuation_is_one_way(minds):
    s, seen = minds[False]
    assert answer(s, "mars nedir ?") == "mars kızıl bir gezegendir ve soğuktur"
    assert set(seen) == {"devam"}


def test_on_a_continuation_is_a_way_of_its_own_for_each_reach(minds):
    _, seen = minds[True]
    assert "devam" in seen
    assert any(name[5:].isdigit() and int(name[5:]) >= 2 for name in seen)


def test_on_the_answer_is_not_begun_here(minds):
    """The failure that kept it out of the whole mind, pinned so that a change to it is seen."""
    s, _ = minds[True]
    assert answer(s, "mars nedir ?") == "(sessiz)"
