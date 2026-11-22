# -*- coding: utf-8 -*-
"""Zihin, F46 Open Claims: the mind notes what it found it did not know, wants that first, and the want ends when it
is met.

What is pinned here
-------------------
* The switch is off by default and is one of the studied mechanisms (``--ac open_claims``).
* A small world: things read about, one sentence each; for some the question "X nedir ?" is watched being answered.
  Asked about a thing it never met, the mind says the name and has nothing to go on with: the frame that speaks
  claims "what follows is how the sequence that begins with the filler went on", and for this filler the relation
  names nothing. With F46 that is noted as evidence on the mind's own channel: ``read.open_claim`` of the item rises
  over 1/2; of a thing it can answer about it stays 1/2 (no unit is made for it).
* ``web.wanted_topics`` puts such items first; without the switch it does not (it names what it is unsure of).
* Looked up (``web.research`` from a source that stands where the web stands) and asked again, the mind answers; and
  asked for real once more, the claim is met, the evidence turns, and the item is no longer wanted.
* The note is evidence of its moment like any other: retracting the moment takes it back.
* ``web.research`` ends an answered turn that is still open before it reads (the chat does so for every command).
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Config, Mind
from zihin.__main__ import CHANNEL, STUDIED, Session, VARIANTS
from zihin.read import OPEN_CHANNEL, open_claim
from zihin.web import research, wanted_topics

KNOWN = {"gül": "kırmızı bir çiçektir", "papatya": "beyaz bir çiçektir", "çam": "yeşil bir ağaçtır",
         "meşe": "yaşlı bir ağaçtır", "çekiç": "ağır bir alettir", "testere": "keskin bir alettir",
         "tren": "uzun bir taşıttır", "gemi": "büyük bir taşıttır"}
WATCHED = ("gül", "çam", "çekiç", "tren", "papatya", "testere")
LIBRARY = {"zambak": "mor bir çiçektir", "vapur": "yavaş bir taşıttır"}
ASK = "%s nedir ?"


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fact(thing: str) -> str:
    return "%s %s ." % (thing, {**KNOWN, **LIBRARY}[thing])


class Library:
    def __init__(self) -> None:
        self.asked = []

    def search(self, topic: str, limit: int = 2):
        self.asked.append(topic)
        return [topic] if topic in LIBRARY else []

    def extract(self, identifier: str) -> str:
        return fact(identifier)

    def address(self, identifier: str) -> str:
        return "library:" + identifier


def raised(keep_log: bool = False, **switches) -> Session:
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
        for thing in KNOWN:
            s.company(None)
            s.listen(fact(thing), IMITATED)
            s.end_of_turn(IMITATED, tone=s.tone(fact(thing)))
            mind.step([])
        for thing in WATCHED:
            s.watch(ASK % thing, fact(thing))
    return s


def ask(s: Session, thing: str) -> str:
    return Session(s.mind.clone()).reply(ASK % thing, limit=40)


def test_the_switch_is_off_by_default_and_studied():
    assert Config().open_claims is False
    assert "open_claims" in STUDIED
    assert raised().mind.cfg.open_claims is False


def test_what_it_could_not_answer_is_what_it_wants_and_the_want_ends_when_met():
    s = raised(open_claims=True)
    mind = s.mind
    assert ask(s, "meşe") == "meşe yaşlı bir ağaçtır"                 # read, never asked: the form carries over
    assert not mind.ch.has(OPEN_CHANNEL)                              # nothing was left open so far
    assert [s.reply(ASK % thing, limit=40) for thing in LIBRARY] == ["zambak", "vapur"]   # the name, and no more
    s.close_turn()
    assert open_claim(mind, "zambak") > 0.5 and open_claim(mind, "vapur") > 0.5
    assert open_claim(mind, "meşe") == 0.5 and mind.unit_of(OPEN_CHANNEL, "meşe") < 0
    assert set(wanted_topics(mind, CHANNEL, 2)) == set(LIBRARY)
    library = Library()
    for topic in wanted_topics(mind, CHANNEL, 2):
        assert research(s, topic, source=library, say=lambda line: None)["read"] == [topic]
    assert sorted(library.asked) == sorted(LIBRARY)
    assert ask(s, "zambak") == "zambak mor bir çiçektir" and ask(s, "vapur") == "vapur yavaş bir taşıttır"
    for thing in LIBRARY:                                             # asked for real: the claim is met
        assert s.reply(ASK % thing, limit=40) == fact(thing).rstrip(" .")
    s.close_turn()
    assert open_claim(mind, "zambak") <= 0.5 and open_claim(mind, "vapur") <= 0.5
    assert not set(wanted_topics(mind, CHANNEL, 2)) & set(LIBRARY)


def test_without_the_switch_nothing_is_noted_and_the_gaps_are_not_what_it_wants():
    s = raised()
    assert [s.reply(ASK % thing, limit=40) for thing in LIBRARY] == ["zambak", "vapur"]
    s.close_turn()
    assert not s.mind.ch.has(OPEN_CHANNEL)
    assert open_claim(s.mind, "zambak") == 0.5
    assert not set(wanted_topics(s.mind, CHANNEL, 2)) & set(LIBRARY)


def test_an_open_claim_is_evidence_of_its_moment_and_is_retracted_with_it():
    s = raised(keep_log=True, open_claims=True)
    mind = s.mind
    before = mind.t
    assert s.reply(ASK % "zambak", limit=40) == "zambak"
    s.close_turn()
    assert open_claim(mind, "zambak") > 0.5
    for m in range(mind.t, before, -1):                               # the moments of the question, taken back
        mind.retract(m)
    assert open_claim(mind, "zambak") == 0.5


def test_research_ends_an_answered_turn_before_it_reads():
    s = raised(open_claims=True)
    assert s.reply(ASK % "vapur", limit=40) == "vapur"
    assert s.open_turn                                                # (the turn stays open for feedback)
    research(s, "zambak", source=Library(), say=lambda line: None)
    assert not s.open_turn
    assert ask(s, "vapur") == "vapur"                                 # nothing read was taken as vapur's own
    assert ask(s, "zambak") == "zambak mor bir çiçektir"
