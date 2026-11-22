# -*- coding: utf-8 -*-
"""Zihin, F56 The Kind of a Whole: the form of a whole says which wholes stand in it also where its answer reads as
their replies in more than one order.

What is pinned here
-------------------
* The switch is off by default and is one of the studied mechanisms (``--ac whole_kinds``; with wholes).
* The error it was found by, in a small world (five words said alone, twice and thrice; every two of them joined
  by "and"; three commands of the form "p and q thrice"). "a and a" was lived and answered "ia ia", which reads as
  the replies of its two wholes in either order: its alignment concludes nothing, so it had no form and stood as a
  bare whole. Asked "a and a thrice", the mind read "[a and a] thrice" (a whole before "thrice" was lived more
  often than a whole, "and", and a whole with "thrice") and said six actions; the command means "a and [a thrice]",
  four. With the switch the form of "a and a" says that two wholes joined by "and" stand in it; such a form never
  stood before "thrice", that reading is not taken, and the answer is right.
* The summary of the lived dialogue itself is still not concluded (its answer has no one order): only the form is
  known. What the mind answered before is answered as before.
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import Config, Mind
from zihin import butun as B
from zihin.__main__ import CHANNEL, STUDIED, Session, VARIANTS

PRIMITIVES = {"a": "ia", "b": "ib", "c": "ic", "d": "id", "e": "ie"}


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def lived(**switches) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    cfg.wholes = True
    for name, value in switches.items():
        setattr(cfg, name, value)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    mind.knob["F"] = 7
    s = Session(mind)
    parts = [d for x in PRIMITIVES for d in (((x,), (PRIMITIVES[x],)), ((x, "twice"), (PRIMITIVES[x],) * 2),
                                             ((x, "thrice"), (PRIMITIVES[x],) * 3))]
    train = parts + [((p, "and", q), (PRIMITIVES[p], PRIMITIVES[q])) for p in PRIMITIVES for q in PRIMITIVES]
    train += [((p, "and", q, "thrice"), (PRIMITIVES[p],) + (PRIMITIVES[q],) * 3) for p, q in ("bc", "cb", "de")]
    for _ in range(2):
        for command, answer in sorted(train, key=lambda d: len(d[0])):
            s.watch(" ".join(command), " ".join(answer) + " .")
    return s


def ask(s: Session, command: str) -> str:
    return Session(s.mind.clone()).reply(command, limit=200)


def test_off_by_default_and_studied():
    assert Config().whole_kinds is False and "whole_kinds" in STUDIED


def test_a_lived_command_whose_answer_reads_in_two_orders_is_not_an_action_phrase():
    off, on = lived(), lived(whole_kinds=True)
    assert ask(off, "a and a thrice") == "ia ia ia ia ia ia"       # the error: "[a and a] thrice"
    assert ask(on, "a and a thrice") == "ia ia ia ia"              # "a and [a thrice]"
    for command, answer in (("a and a", "ia ia"), ("b and a thrice", "ib ia ia ia"), ("c thrice", "ic ic ic"),
                            ("e and d thrice", "ie id id id")):
        assert ask(on, command) == ask(off, command) == answer


def test_the_form_is_known_and_the_summary_is_still_not_concluded():
    on, off = lived(whole_kinds=True), lived()
    for s in (on, off):
        mind, F = s.mind, s.mind.knob["F"]
        assert B.align(mind, ("a", "and", "a"), ("ia", "ia"), F, {}) is None       # no one order of its answer
    assert B.form(off.mind, ("a", "and", "a"), ("ia", "ia"), off.mind.knob["F"], {}) == B.WHOLE
    assert B.form(on.mind, ("a", "and", "a"), ("ia", "ia"), on.mind.knob["F"], {}) == ("whole", B.WHOLE, "and", B.WHOLE)
