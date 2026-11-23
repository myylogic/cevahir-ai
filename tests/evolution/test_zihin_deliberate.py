# -*- coding: utf-8 -*-
"""Zihin, F74 Recall or Reason (``Config.deliberate``).

What is pinned here
-------------------
* Off by default, a studied mechanism, in use in the full configuration, where every other studied mechanism is
  what only a reasoned moment reads (``Config.deep_only``).
* A moment is recalled when the speaking configuration's reading says what this very context was lived to go on
  with, within the first half of the race: "serçe ne yapar ?", asked and answered before, is answered "serçe uçar"
  word by word from recall, as the mind answers it without the switch.
* Every other moment is reasoned with everything: "kartal ne yapar ?", "levrek ne yapar ?" and "penguen ne yapar
  ?" were never asked; their answers are only deduced (F37) or are the thing's own exception, and they are answered
  as the mind answers them without the switch ("kartal uçar", "levrek yüzer", "penguen yüzer"), not as recall alone
  would ("levrek bir balıktır": what the frames of the question suggest is already reasoning, never recall).
* With no deep mechanism switched on, the switch changes nothing.
* What the first reading set is put back before the moment is reasoned: the word it hoped for.
"""
from __future__ import annotations

import random

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Config, Mind
from zihin.__main__ import CHANNEL, STUDIED, Session, VARIANTS

CLASSES = {"kuş": ("kuştur", "kuşlar", "uçar", ["serçe", "güvercin", "kartal"]),
           "balık": ("balıktır", "balıklar", "yüzer", ["hamsi", "sazan", "levrek"])}
ANSWERS = {"serçe": "serçe uçar", "kartal": "kartal uçar", "levrek": "levrek yüzer", "penguen": "penguen yüzer"}


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def raised(inherit: bool, deliberate: bool) -> Session:
    """The world of test_zihin_inherit: facts read apart, two members of each class asked what they do."""
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    cfg.inherit, cfg.deliberate, cfg.deep_only = inherit, deliberate, ("inherit",)
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
            for m in members[:2]:
                s.watch("%s ne yapar ?" % m, "%s %s ." % (m, prop))
    return s


@pytest.fixture(scope="module")
def minds():
    with torch.no_grad():
        return {(inherit, deliberate): raised(inherit, deliberate)
                for inherit in (True, False) for deliberate in (True, False)}


def ask(s: Session, text: str):
    """(what the mind says, [moments recalled, moments reasoned]) on a copy."""
    c = Session(s.mind.clone())
    c.mind.__dict__.pop("_recall_counts", None)
    said = c.reply(text)
    return said, c.mind.__dict__.get("_recall_counts", [0, 0])


def test_off_by_default_and_studied():
    assert Config().deliberate is False and Config().deep_only == () and "deliberate" in STUDIED
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        full = VARIANTS["full"]()
        assert full.deliberate is True and set(full.deep_only) == set(STUDIED)
        assert VARIANTS["speaking"]().deliberate is False
    finally:
        chat._log_path = made


def test_what_was_asked_before_is_recalled(minds):
    said, (recalled, reasoned) = ask(minds[True, True], "serçe ne yapar ?")
    assert said == ANSWERS["serçe"] == ask(minds[True, False], "serçe ne yapar ?")[0]
    assert recalled == len(said.split())                   # every word from recall; the silence that ends it is
    assert reasoned >= 1                                   # a moment recall did not settle


@pytest.mark.parametrize("thing", ["kartal", "levrek", "penguen"])
def test_what_was_never_asked_is_reasoned(minds, thing):
    said, (recalled, reasoned) = ask(minds[True, True], "%s ne yapar ?" % thing)
    assert said == ANSWERS[thing] == ask(minds[True, False], "%s ne yapar ?" % thing)[0]
    assert recalled == 0 and reasoned >= len(said.split())
    assert ask(minds[False, True], "%s ne yapar ?" % thing)[0] != ANSWERS[thing]   # recall alone does not say it


def test_with_nothing_deep_the_switch_changes_nothing(minds):
    for thing in ANSWERS:
        assert ask(minds[False, True], "%s ne yapar ?" % thing)[0] == ask(minds[False, False], "%s ne yapar ?" % thing)[0]


def test_what_the_first_reading_hoped_for_is_put_back():
    s = raised(True, True)
    mind = s.mind
    s.listen("levrek ne yapar ?", IMITATED)
    s.end_of_turn(IMITATED, tone="?")
    before = mind.hoping
    mind.__dict__.pop("_recall_counts", None)
    acts = mind.step([])
    assert mind.__dict__["_recall_counts"][1] == 1         # reasoned
    word = mind.unit_of(acts[0].channel, acts[0].symbol) if acts else None
    assert (mind.hoping == before) if not acts else (mind.hoping[0] == word)
