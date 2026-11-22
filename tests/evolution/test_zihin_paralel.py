# -*- coding: utf-8 -*-
"""Zihin, at the same time with nothing lost (zihin/paralel.py).

What is pinned here
-------------------
* Many questions answered at the same time, each in a process that is the mind as it stands: the answers are, word
  for word, what the mind gives when it answers each on a copy of itself one after another.
* The same for thoughts (the inner words and their reality score).
* The mind that asked is not changed: the same moment, the same substance digest, nothing in its ledger.
* It holds for answers in parts too (F43), and with one worker (no process) the results are the same.
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Mind
from zihin import paralel
from zihin.__main__ import CHANNEL, Session, VARIANTS
from zihin.dusunce import think

FACTS = {"gül": "kırmızı bir çiçektir", "papatya": "beyaz bir çiçektir", "çam": "yeşil bir ağaçtır",
         "meşe": "yaşlı bir ağaçtır", "çekiç": "ağır bir alettir", "testere": "keskin bir alettir",
         "tren": "uzun bir taşıttır", "gemi": "büyük bir taşıttır"}
ASK = "%s nedir ?"


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def raised(keep_log: bool = True, **switches) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = keep_log, None
    for name, value in switches.items():
        setattr(cfg, name, value)
    torch.manual_seed(0)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    for _ in range(3):
        for thing, what in FACTS.items():
            s.company(None)
            s.listen("%s %s ." % (thing, what), IMITATED)
            s.end_of_turn(IMITATED, tone=".")
            mind.step([])
        for thing in list(FACTS)[:6]:
            s.watch(ASK % thing, "%s %s ." % (thing, FACTS[thing]))
    s.close_turn()
    return s


def one_after_another(s: Session, question: str) -> str:
    copy = Session(s.mind.clone(), s.pieces)
    copy.partner, copy.dil = s.partner, s.dil
    return copy.reply(question, limit=80)


def test_many_answers_at_once_are_the_answers_given_one_after_another_and_the_mind_is_not_changed():
    s = raised()
    mind = s.mind
    questions = [ASK % thing for thing in FACTS] + ["zambak nedir ?", "merhaba"]
    before = (mind.t, mind.S.digest(), len(mind.S.log), mind.S.n_units, mind.S.n_ties)
    reference = [one_after_another(s, q) for q in questions]
    assert reference[0] == "gül kırmızı bir çiçektir" and reference[-2] == "zambak"
    at_once = paralel.answers(s, questions, workers=3)
    assert at_once == reference
    assert paralel.answers(s, questions, workers=1) == reference       # no process: the same
    assert (mind.t, mind.S.digest(), len(mind.S.log), mind.S.n_units, mind.S.n_ties) == before
    assert paralel.answers(s, []) == []


def test_many_thoughts_at_once_are_the_thoughts_thought_one_after_another():
    s = raised()
    mind = s.mind
    topics = ["gül", "tren uzun", "çam yeşil bir"]
    before = (mind.t, mind.S.digest())
    reference = []
    for topic in topics:
        copy = Session(mind.clone(), s.pieces)
        words, reality = think(copy, topic, limit=15)
        reference.append((list(words), float(reality)))
    assert paralel.thoughts(s, topics, workers=3) == reference
    assert any(words for words, _ in reference)
    assert (mind.t, mind.S.digest()) == before


def test_answers_in_parts_at_once_are_the_same():
    primitives = {"a": "ia", "b": "ib", "c": "ic", "d": "id"}
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path, cfg.wholes, cfg.exclusive_world = False, None, True, True
    torch.manual_seed(0)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    mind.knob["F"] = 7
    s = Session(mind)
    parts = [((x,) + tail, (primitives[x],) * n) for x in "abc" for tail, n in (((), 1), (("twice",), 2),
                                                                                (("thrice",), 3))]
    train = parts + [(p + ("and",) + q, a + b) for p, a in parts for q, b in parts] + [(("d",), ("id",))]
    for _ in range(2):
        for command, answer in sorted(train, key=lambda d: len(d[0])):
            s.watch(" ".join(command), " ".join(answer) + " .")
    s.close_turn()
    questions = ["d twice", "d thrice and a twice", "b thrice and d thrice", "d and d thrice"]
    reference = [one_after_another(s, q) for q in questions]
    assert reference == ["id id", "id id id ia ia", "ib ib ib id id id", "id id id id"]
    digest = mind.S.digest()
    assert paralel.answers(s, questions, workers=4) == reference
    assert mind.S.digest() == digest
