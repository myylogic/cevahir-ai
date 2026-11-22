# -*- coding: utf-8 -*-
"""Zihin, F48 Thinking by Itself, and the stricter settling of a look (F47).

What is pinned here
-------------------
* The switch is off by default and is one of the studied mechanisms; without it nothing is decided about thinking.
* What the mind took in leaves a feeling of its own: its surprise over those moments, under 1/2 when what it
  expected came, 1/2 when it expected nothing (words it never met), over 1/2 when what it expected did not come.
* Left in silence it decides by itself to think about what went against its expectations, once (the feeling is
  spent by the thought), and not about what it expected, nor about what it could expect nothing of.
* What a thought came to is evidence for thinking (its reality score) or against it (a thought in which nothing was
  said), and that moves what it takes to think again.
* A thought of its own is an entry of the history, taken back like one.
* F47, stricter: a claim counts as met only when the mind, brought back to the situation, can say what follows to
  the end (it stops where the end of a turn is expected). A first word is not knowing: the claim stays open, the
  look counts against looking, and the way of reading is judged by whether the claim closed.
* Before anything is fetched the mind is brought back to the situation: when it has learned it meanwhile, nothing
  is fetched.
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, TOLD, Config, Mind
from zihin.__main__ import CHANNEL, STUDIED, Session, VARIANTS
from zihin.dusunce import goes_to_an_end
from zihin.learn import claim_state
from zihin.read import open_claim, think_expectation
from zihin.web import WayChooser

KNOWN = {"gül": "kırmızı bir çiçektir", "papatya": "beyaz bir çiçektir", "çam": "yeşil bir ağaçtır",
         "meşe": "yaşlı bir ağaçtır", "çekiç": "ağır bir alettir", "testere": "keskin bir alettir",
         "tren": "uzun bir taşıttır", "gemi": "büyük bir taşıttır"}
WATCHED = ("gül", "çam", "çekiç", "tren", "papatya", "testere")
ASK = "%s nedir ?"


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


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
        for thing, what in KNOWN.items():
            s.company(None)
            s.listen("%s %s ." % (thing, what), IMITATED)
            s.end_of_turn(IMITATED, tone=".")
            mind.step([])
        for thing in WATCHED:
            s.watch(ASK % thing, "%s %s ." % (thing, KNOWN[thing]))
    s.close_turn()
    s.look_gap = 0.0
    return s


def quiet(line: str) -> None:
    pass


def told(s: Session, text: str) -> None:
    """Someone says something to the mind (no question: nothing is asked of it)."""
    s.live(text, TOLD)


def silence(s: Session, moments: int = 6) -> None:
    for _ in range(moments):
        s.silent_moment(say=quiet)


def test_the_switch_is_off_by_default_and_studied():
    assert Config().think is False and "think" in STUDIED
    s = raised()
    told(s, "gül sarı bir çiçektir .")
    silence(s)
    assert s.mind.heard is None and s.mind.thinking is None and s.thoughts == []


def test_what_it_took_in_leaves_its_surprise():
    felt = {}
    for name, text in (("expected", "gül kırmızı bir çiçektir ."), ("against", "gül sarı bir çiçektir ."),
                       ("nothing", "leylek göçmen bir kuştur .")):
        s = raised(think=True)
        told(s, text)
        s.mind.step([])
        felt[name] = s.mind.heard
    assert felt["expected"] < 0.5 < felt["against"] and felt["nothing"] == 0.5


def test_it_thinks_by_itself_about_what_went_against_its_expectations_and_only_once():
    s = raised(keep_log=True, think=True)
    mind = s.mind
    assert think_expectation(mind) == 0.5
    told(s, "gül kırmızı bir çiçektir .")                  # what it expected: nothing to think about
    silence(s)
    assert s.thoughts == []
    told(s, "leylek göçmen bir kuştur .")                  # it expected nothing: nothing to think with
    silence(s)
    assert s.thoughts == []
    told(s, "gül sarı bir çiçektir .")                     # against what it expected
    silence(s, 10)
    assert len(s.thoughts) == 1 and mind.heard is None     # one thought; the feeling is spent
    topic, words, reality = s.thoughts[0]
    assert topic and think_expectation(mind) != 0.5        # what it came to is evidence about thinking
    entry = s.history[-1]
    assert entry.text.startswith("[kendi kararıyla düşündü]")
    before = mind.t
    s.forget(entry.number)                                 # a thought is taken back like any entry
    assert think_expectation(mind) == 0.5 and mind.t == before


def test_what_thoughts_come_to_moves_what_it_takes_to_think():
    from zihin.act import think_now
    from zihin.learn import settle_thought
    s = raised(think=True)
    mind = s.mind
    mind.heard = 0.55                                      # a little against its expectations: not enough at first
    assert think_now(mind) is None
    for _ in range(3):
        settle_thought(mind, True, 0.9)                    # thoughts that came to something real
        mind.heard = 0.55
    assert think_expectation(mind) > 0.5 and think_now(mind) is not None
    for _ in range(12):
        settle_thought(mind, False, 0.0)                   # thoughts in which nothing was said
        mind.heard = 0.7
    assert think_expectation(mind) < 0.5 and think_now(mind) is None


class Pages:
    """A source whose page about a thing is given text."""

    def __init__(self, pages: dict) -> None:
        self.pages, self.asked = pages, []

    def search(self, topic: str, limit: int = 2):
        self.asked.append(topic)
        return [topic] if topic in self.pages else []

    def extract(self, identifier: str) -> str:
        return self.pages[identifier]

    def address(self, identifier: str) -> str:
        return "pages:" + identifier


def test_a_first_word_is_not_knowing_the_claim_stays_open():
    s = raised(open_claims=True, look=True)
    mind = s.mind
    fork = "zambak mor bir çiçektir .\nzambak , güzel kokar .\nzambak , suyu sever .\nzambak ile lale benzer ."
    pages = Pages({"zambak": fork, "vapur": "vapur yavaş bir taşıttır ."})
    s.may_look, s.look_source = True, pages
    for thing in ("vapur", "zambak"):                      # (the one it can learn first: a first look that came to
        assert s.reply(ASK % thing, limit=40) == thing     # nothing would have stopped it looking, test_zihin_look)
    s.close_turn()
    silence(s, 12)
    by_topic = {r["topic"]: r for r in s.looks}
    assert by_topic["vapur"]["met"] is True                # it can say it to the end
    assert open_claim(mind, "vapur") <= 0.5
    said = Session(mind.clone()).reply(ASK % "zambak", limit=40)
    assert said.startswith("zambak") and said != "zambak mor bir çiçektir"   # many sentences begin with it: it stops
    assert all(r["met"] is False for r in s.looks if r["topic"] == "zambak")  # ... and it knows that it cannot say it
    assert open_claim(mind, "zambak") > 0.5
    assert len([r for r in s.looks if r["topic"] == "zambak"]) <= 3           # it gives up by its own evidence
    ways = WayChooser(mind)
    assert any(ways.expectation(way)[1] > 0.0 for way in ("ozet", "derin"))   # the ways were judged by the claims


def test_what_it_learned_meanwhile_is_not_looked_up():
    s = raised(open_claims=True, look=True)
    mind = s.mind
    pages = Pages({"vapur": "vapur yavaş bir taşıttır ."})
    s.look_source = pages                                  # (the owner has not let it look yet)
    assert s.reply(ASK % "vapur", limit=40) == "vapur"
    s.close_turn()
    filler = mind.unit_of(CHANNEL, "vapur")
    assert claim_state(mind, filler)[1] is True
    assert goes_to_an_end(mind, mind.open_where[filler]) is False
    s.company(None)                                        # someone tells it meanwhile (a sentence it reads)
    s.listen("vapur yavaş bir taşıttır .", IMITATED)
    s.end_of_turn(IMITATED, tone=".")
    mind.step([])
    assert goes_to_an_end(mind, mind.open_where[filler]) is True
    s.may_look = True
    silence(s, 4)
    assert pages.asked == []                               # nothing was fetched
    assert [r.get("note") for r in s.looks] == ["kendi bildiğiyle"] and open_claim(mind, "vapur") <= 0.5
