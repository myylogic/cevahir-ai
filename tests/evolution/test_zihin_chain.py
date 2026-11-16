# -*- coding: utf-8 -*-
"""Zihin, F37 Inheritance two steps away: a member's class inherits from its own class, and what a class word goes on
with where it is said to be of a class is the form of that saying, not a property.

What is pinned here
-------------------
* Facts read apart: "levrek bir balıktır .", "balıklar bir hayvandır .", "hayvanlar solur ." (never two together). Members
  asked were answered with what follows from the chain; a member never asked is answered by putting the three facts
  together: "levrek ne yapar ?" -> "levrek solur". The nearer property wins where a class has one ("kuşlar uçar ."):
  "kartal uçar". An exception stated of itself wins over both ("penguen yüzer"). Without the switch none of them.
* "bir" and "hayvandır" of "balıklar bir hayvandır ." are the form of being of a class for "balıklar"; a word its place
  merely shares with another member ("yüzer" of "balıklar yüzer ." and "penguen yüzer .") is not.
"""
from __future__ import annotations

import random

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS
from zihin.read import _forms_of

KINDS = {"hayvan": ("hayvandır", "hayvanlar", "solur"), "bitki": ("bitkidir", "bitkiler", "filizlenir")}
CLASSES = {"kuş": ("kuştur", "kuşlar", "hayvan", "uçar", ["serçe", "güvercin", "kartal"]),
           "balık": ("balıktır", "balıklar", "hayvan", None, ["hamsi", "sazan", "levrek"]),
           "çiçek": ("çiçektir", "çiçekler", "bitki", None, ["gül", "lale", "papatya"]),
           "ağaç": ("ağaçtır", "ağaçlar", "bitki", None, ["meşe", "çam", "kavak"])}   # a place needs two fillers


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def holds(name: str) -> str:
    copula, plural, kind, own, members = CLASSES[name]
    return own or KINDS[kind][2]


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
        for copula, plural, prop in KINDS.values():
            sentences += ["%s %s ." % (plural, prop)] * 2
        for name, (copula, plural, kind, own, members) in CLASSES.items():
            sentences += ["%s bir %s ." % (m, copula) for m in members]
            sentences += ["%s bir %s ." % (plural, KINDS[kind][0])] * 2
            if own:
                sentences += ["%s %s ." % (plural, own)] * 2
        rng.shuffle(sentences)
        for sentence in sentences:
            s.company(None)
            s.listen(sentence, IMITATED)
            s.end_of_turn(IMITATED, tone=s.tone(sentence))
            mind.step([])
        for name, (copula, plural, kind, own, members) in CLASSES.items():
            for m in members[:2]:                        # the first two are asked; the third never is
                s.watch("%s ne yapar ?" % m, "%s %s ." % (m, holds(name)))
    return s


def ask(s: Session, text: str) -> str:
    return Session(s.mind.clone()).reply(text)


@pytest.fixture(scope="module")
def minds():
    return {True: raised(True), False: raised(False)}


@pytest.mark.parametrize("member, answer", [("levrek", "levrek solur"), ("papatya", "papatya filizlenir"),
                                            ("kartal", "kartal uçar"), ("penguen", "penguen yüzer")])
def test_a_conclusion_two_steps_away(minds, member, answer):
    assert ask(minds[True], "%s ne yapar ?" % member) == answer
    if member != "penguen":
        assert ask(minds[False], "%s ne yapar ?" % member) != answer


def test_the_form_of_being_of_a_class_is_not_a_property(minds):
    mind = minds[True].mind
    word = lambda w: mind.unit_of(CHANNEL, w)
    forms = _forms_of(mind, word("balıklar"))
    assert word("bir") in forms and word("hayvandır") in forms
    assert word("yüzer") not in _forms_of(mind, word("penguen"))
