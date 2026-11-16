# -*- coding: utf-8 -*-
"""Zihin, F36 Word Forms: a word never met is read as the lived word it is a form of.

What is pinned here
-------------------
* The base of a word never met shares its longest stem with a lived word, the stem being a boundary of the lived words
  (a lived word itself, or a beginning lived words go on from in two ways): "nasılsınız" is a form of "nasılsın"
  (f = 8/10, c = conviction(1) = 1/2). A beginning one lived word has by chance ("mas" of "masa" for "maske") is not
  a stem, and a word with no lived stem ("keyfin") has no base.
* Watched "nasılsın ?" answered "iyiyim , teşekkür ederim .", the mind answers "nasılsınız ?" the same way with the
  switch, and stays silent without it; "merhabalar" is greeted as "merhaba" is.
* The reading is used only when nothing would be said otherwise, and the switch is off by default.
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import Config, Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS
from zihin.evidence import conviction
from zihin.read import form_base


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fresh(forms: bool) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path, cfg.forms = False, None, forms
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    return Session(mind)


def lived(forms: bool) -> Session:
    s = fresh(forms)
    for _ in range(3):
        s.watch("nasılsın ?", "iyiyim , teşekkür ederim .")
        s.watch("merhaba", "merhaba , hoş geldin .")
    s.watch("masa nerede ?", "masa odada .")
    return s


def ask(s: Session, text: str) -> str:
    return Session(s.mind.clone()).reply(text)


def test_the_switch_is_off_by_default():
    assert Config().forms is False and Config.changed().forms is False


def test_the_base_is_a_lived_stem():
    s = lived(True)
    mind = s.mind
    since = mind.t + 1
    unknown = []
    for w in ("nasılsınız", "maske", "keyfin"):           # new words, met after everything lived
        mind.step(s.events(w, 1))
        unknown.append(mind.unit_of(CHANNEL, w))
    base = form_base(mind, unknown[0], since)
    assert base is not None and base[0] == mind.unit_of(CHANNEL, "nasılsın")
    assert base[1] == pytest.approx(8 / 10) and base[2] == pytest.approx(conviction(1.0, mind.cfg.k))
    assert form_base(mind, unknown[1], since) is None    # "mas" is not a stem of the lived words
    assert form_base(mind, unknown[2], since) is None    # nothing lived begins as "keyfin" does


@pytest.mark.parametrize("forms, answer", [(True, "iyiyim , teşekkür ederim"), (False, "(sessiz)")])
def test_a_form_never_met_is_answered_as_its_base(forms, answer):
    assert ask(lived(forms), "nasılsınız ?") == answer


def test_a_greeting_form_never_met_is_greeted():
    assert ask(lived(True), "merhabalar").startswith("merhaba")
