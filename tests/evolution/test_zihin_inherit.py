# -*- coding: utf-8 -*-
"""Zihin, F37 Inheritance: what a member's classes go on with, its own record last (deduction with exceptions).

What is pinned here
-------------------
* Facts read apart: "kartal bir kuştur ." and "kuşlar uçar ." (and of fish), never together. Members asked what they
  do were answered with their class's property ("serçe ne yapar ?" -> "serçe uçar ."). A member never asked and never
  stated with the property is answered by putting the two facts together: "kartal ne yapar ?" -> "kartal uçar",
  "levrek ne yapar ?" -> "levrek yüzer". Without the switch it is not.
* An exception stated of itself wins over its class: "penguen bir kuştur ." with "penguen yüzer ." gives "penguen
  yüzer" (what its class-mates mostly did not go on with is its own; its own record comes last).
* The class is named by a stem two lived words share ("kuştur", "kuşlar": cover 3/6); reading writes nothing; the
  switch is off by default.
"""
from __future__ import annotations

import random

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Config, Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS
from zihin.read import stem_mates

CLASSES = {"kuş": ("kuştur", "kuşlar", "uçar", ["serçe", "güvercin", "kartal"]),
           "balık": ("balıktır", "balıklar", "yüzer", ["hamsi", "sazan", "levrek"])}


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def raised(inherit: bool) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path, cfg.inherit = False, None, inherit
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    rng = random.Random(0)
    for _ in range(3):
        sentences = ["penguen bir kuştur .", "penguen yüzer ."]
        for copula, plural, prop, members in CLASSES.values():
            sentences += ["%s bir %s ." % (m, copula) for m in members] + ["%s %s ." % (plural, prop)] * 2
        rng.shuffle(sentences)
        for sentence in sentences:
            s.company(None)
            s.listen(sentence, IMITATED)
            s.end_of_turn(IMITATED, tone=s.tone(sentence))
            mind.step([])
        for copula, plural, prop, members in CLASSES.values():
            for m in members[:2]:                        # the first two are asked; the third never is
                s.watch("%s ne yapar ?" % m, "%s %s ." % (m, prop))
    return s


def ask(s: Session, text: str) -> str:
    return Session(s.mind.clone()).reply(text)


def test_the_switch_is_off_by_default():
    assert Config().inherit is False and Config.changed().inherit is False


@pytest.mark.parametrize("member, answer", [("kartal", "kartal uçar"), ("levrek", "levrek yüzer")])
def test_a_conclusion_never_taught_from_facts_read_apart(member, answer):
    assert ask(raised(True), "%s ne yapar ?" % member) == answer
    assert ask(raised(False), "%s ne yapar ?" % member) != answer


def test_an_exception_stated_of_itself_wins_over_its_class():
    assert ask(raised(True), "penguen ne yapar ?") == "penguen yüzer"


def test_the_class_is_named_by_a_shared_stem():
    s = raised(True)
    mind = s.mind
    mates = dict(stem_mates(mind, mind.unit_of(CHANNEL, "kuştur"), mind.t + 1))
    assert mates.get(mind.unit_of(CHANNEL, "kuşlar")) == pytest.approx(3 / 6)


def test_reading_writes_nothing():
    s = raised(True)
    mind = s.mind
    digest = mind.S.digest()
    twin_answer = ask(s, "kartal ne yapar ?")
    assert twin_answer == "kartal uçar" and mind.S.digest() == digest
