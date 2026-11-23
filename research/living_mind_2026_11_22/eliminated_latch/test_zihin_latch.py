# -*- coding: utf-8 -*-
"""Zihin, F71 Latching (``Config.latch``, with silent claims).

What is pinned here
-------------------
* Off by default, a studied mechanism, in use in the full configuration; it needs silent claims.
* While the mind only listens a run of items is one thought: sought from the run's beginning until it claims
  something, followed while what it claimed is what was then heard; where a claim of it fails it is let go and the
  mind only listens until the next run begins.
* A thought that held reaches further: the window of F70 is one step wider for every claim of it that held.
* What the mind says when it is its turn is untouched: latching is about the moments it only listens.
"""
from __future__ import annotations

import pytest
import torch

import test_zihin_own_race as R
from zihin import Config, aile, hata
from zihin.__main__ import NEEDS, STUDIED, VARIANTS
import zihin.__main__ as chat

THINGS = ["inek", "koyun", "keçi", "at", "eşek", "deve"]


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def readings(s, sentence: str):
    """Listen to a sentence; returns, for each of its moments, whether the full reading was made while listening."""
    from zihin import read
    made = []
    whole = read.members_map

    def counted(*a, **kw):
        made[-1] = True
        return whole(*a, **kw)
    read.members_map = counted
    real = s.mind.step

    def step(*a, **kw):
        made.append(False)
        return real(*a, **kw)
    s.mind.step = step
    try:
        R.tell(s, sentence)
    finally:
        read.members_map = whole
        del s.mind.step
    return made


def raised(**switches):
    s = R.born("full", **switches)
    for _ in range(2):
        for x in THINGS:
            R.tell(s, "%s ot yer ve süt verir ." % x)
    return s


def test_off_by_default_studied_and_needs_silent_claims():
    assert Config().latch is False and "latch" in STUDIED
    assert ("latch", ("silent_claims",)) in NEEDS
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        assert VARIANTS["full"]().latch is True and VARIANTS["speaking"]().latch is False
    finally:
        chat._log_path = made


def test_a_thought_that_holds_is_followed_and_one_that_fails_is_let_go():
    held = readings(raised(), "kuzu ot yer ve süt verir .")
    failed = readings(raised(), "kuzu taş atar ve koşar gider .")
    without = readings(raised(latch=False), "kuzu taş atar ve koşar gider .")
    assert held[0] and failed[0]                             # a thought begins where the sentence begins
    assert sum(held) > sum(failed)                           # what it claimed was heard: it follows the thought
    assert sum(failed) < sum(without)                        # what it claimed was not heard: it lets go and listens


def test_a_thought_that_held_reaches_further():
    s = raised()
    mind = s.mind
    mind.__dict__["_reach"] = 0
    unit = mind.context if mind.context is not None else 1
    base = aile.reach(mind, unit)
    mind.__dict__["_reach"] = 3
    assert aile.reach(mind, unit) == base + 3                # three claims of the thought held: three steps wider
    mind.__dict__.pop("_reach")
    assert aile.reach(mind, unit) == base


def test_the_record_of_what_it_heard_says_whether_the_thought_held():
    s = raised()
    mind = s.mind
    R.tell(s, "kuzu ot yer ve süt verir .")
    thought = mind.__dict__.get("_thought")
    assert thought is not None and thought[1] >= 1 and not thought[2]   # its claims were heard: never let go
    R.tell(s, "kuzu taş atar ve koşar gider .")
    thought = mind.__dict__.get("_thought")
    assert thought[2]                                        # a claim of this run was not heard: let go