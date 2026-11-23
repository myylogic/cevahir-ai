# -*- coding: utf-8 -*-
"""Zihin, F68 Deduction Stands Against the Co-members' Vote (``Config.deduced_first``, with F37 and F49).

What is pinned here
-------------------
* Off by default, a studied mechanism, in use in the full configuration; it needs F49 (faults) and F37 (inherit).
* Two of a frame's claims read that the thing asked about is of a kind: F37 reads what was said of its classes,
  its own record last (deduction, with the exception where there is one); F49's narrower vote reads what its
  co-members went on with (induction, which knows no exception). In the whole mind both come to be believed, and
  for an exception two steps away ("penguen bir kuştur", "kuşlar uçar", "penguen yüzer") they tie: the mind
  stops after the thing's name. With F68 the deduction stands against the co-members' vote: "penguen yüzer".
* What follows from the chain for the other things is answered alike with and without it.
"""
from __future__ import annotations

import pytest
import torch

import test_zihin_chain as Ch
import zihin.__main__ as chat
from zihin import Config
from zihin.__main__ import NEEDS, STUDIED, Session, VARIANTS


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def whole(**switches) -> Session:
    """The world of ``test_zihin_chain`` (facts read apart, two members of each class asked), lived by the whole
    mind with the given changes."""
    speaking = chat.VARIANTS["speaking"]

    def made(*args):
        cfg = VARIANTS["full"](*args)
        for name, value in switches.items():
            setattr(cfg, name, value)
        return cfg
    chat.VARIANTS["speaking"] = made
    try:
        return Ch.raised(True)
    finally:
        chat.VARIANTS["speaking"] = speaking


def ask(s: Session, thing: str) -> str:
    return Session(s.mind.clone()).reply("%s ne yapar ?" % thing)


def test_off_by_default_studied_and_needs_faults_and_inherit():
    assert Config().deduced_first is False and "deduced_first" in STUDIED
    assert ("deduced_first", ("faults", "inherit")) in NEEDS
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        assert VARIANTS["full"]().deduced_first is True and VARIANTS["speaking"]().deduced_first is False
    finally:
        chat._log_path = made


def test_the_exception_is_answered_by_its_own_record():
    without, deduced = whole(deduced_first=False), whole()
    assert ask(without, "penguen") == "penguen"              # the two claims tie: it stops after the name
    assert ask(deduced, "penguen") == "penguen yüzer"        # the deduction (its own record last) stands
    for thing, right in (("levrek", "levrek solur"), ("kartal", "kartal uçar"), ("papatya", "papatya filizlenir")):
        assert ask(without, thing) == ask(deduced, thing) == right     # the chain, as before


def test_without_the_ways_judged_while_it_listens_nothing_tied():
    s = whole(silent_claims=False, relied_on=False, deduced_first=False)
    assert ask(s, "penguen") == "penguen yüzer"              # (what the whole mind lost in section 27, and why)
