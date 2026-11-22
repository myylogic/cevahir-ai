# -*- coding: utf-8 -*-
"""Zihin: undoing a moment that was lived while listening does not leave the mind mute.

While someone speaks, the session raises the mind's act threshold for each moment and sets it back after (it does
not speak into the utterance). A moment's working state is kept from its start, so the state kept for such a moment
held the raised threshold, and ``Mind.undo_last`` put it back: after undoing a heard moment the mind decided nothing
any more, until its threshold was set again. Now the threshold the mind has when the undo is asked is kept.

What is pinned here
-------------------
* After a heard moment is undone, the act threshold is what it was, and the mind answers as before.
* A threshold the mind's own knob act changed in the undone moment still goes back to what it was before it.
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def raised() -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = True, None
    torch.manual_seed(0)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    for _ in range(4):
        s.watch("merhaba", "merhaba , hoş geldin")
        s.watch("nasılsın ?", "iyiyim")
    return s


def test_undoing_a_heard_moment_keeps_the_threshold_and_the_mind_still_answers():
    s = raised()
    mind = s.mind
    theta = mind.knob["theta"]
    assert theta < 1e8
    before = Session(mind.clone()).reply("merhaba", limit=20)
    assert before and before != "(sessiz)"
    t = mind.t
    s.listen("güzel", IMITATED)                         # heard: one moment, under the raised threshold
    assert mind.t == t + 1 and mind.knob["theta"] == theta   # (the session set the threshold back)
    mind.undo_last()                                    # the heard moment is undone: everything as before it
    assert mind.t == t and mind.knob["theta"] == theta
    assert Session(mind.clone()).reply("merhaba", limit=20) == before


def test_a_threshold_the_mind_changed_itself_still_goes_back():
    s = raised()
    mind = s.mind
    theta = mind.knob["theta"]
    mind.step([])
    mind.knob["theta"] = theta * 2.0                    # as its own knob act would leave it after a moment
    mind.undo_last()
    assert mind.knob["theta"] == theta
