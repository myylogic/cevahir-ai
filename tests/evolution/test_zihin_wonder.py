# -*- coding: utf-8 -*-
"""Zihin, F51 Asking Itself: what is new in what the mind was told it asks itself about, in its own time; what it
cannot answer it looks up; what it reads brings the next new thing.

What is pinned here
-------------------
* The switch is off by default and is one of the studied mechanisms (``--ac wonder``; with open_claims and look).
* New is what was met in one run of items only, in what the mind was told or read: an item lived again in another
  run is no longer new (the words every sentence holds never stay), and what the mind only said or imagined itself
  does not count.
* A small world: eight things read of ("X V yer .") and watched being asked ("X ne yer ?"), so the mind knows one
  question. A library holds a chain: each thing eats the next. Told one sentence about the first, and then left in
  silence, the mind asks itself the question it knows about the new thing, cannot answer, looks it up by itself,
  reads, and goes on with the new thing it read of, to the end of the chain. Nothing is said aloud; every inner
  question is an entry of the history. Afterwards it answers about every thing of the chain; of the last, of which
  nothing could be found, it says the asked word and holds.
* Without the switch nothing of this happens: no question, no look, and it knows only what it was told.
* With nowhere to look it still asks itself and then knows what it does not know: the item is an open claim.
* What the look then shows is what the asking was worth: met counts for asking itself, not met against.
* Patience: an act of its own time that has rarely paid is not given up; it is taken again after a longer silence.
* What it keeps for this survives saving and loading.
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, TOLD, Config, Mind
from zihin import act
from zihin.__main__ import CHANNEL, STUDIED, Session, VARIANTS
from zihin.read import open_claim, wonder_expectation

THINGS = ["inek", "koyun", "keçi", "at", "eşek", "tavuk", "deve", "geyik"]
FOOD = dict(zip(THINGS, ["ot", "yaprak", "arpa", "saman", "ot", "yem", "diken", "yaprak"]))
CHAIN = ["kartal", "yılan", "kurbağa", "sinek", "şeker"]
ASK, SAY = "%s ne yer ?", "%s %s yer ."
ALL = dict(open_claims=True, look=True, faults=True, find_out=True, known=True, wonder=True)
QUIET = lambda *_: None  # noqa: E731


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


class Library:
    def __init__(self, facts: dict) -> None:
        self.facts, self.asked = dict(facts), []

    def search(self, topic: str, limit: int = 2):
        self.asked.append(topic)
        return [topic] if topic in self.facts else []

    def extract(self, identifier: str) -> str:
        return self.facts[identifier]

    def address(self, identifier: str) -> str:
        return "library:" + identifier


def raised(source=None, **switches) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    for name, value in switches.items():
        setattr(cfg, name, value)
    torch.manual_seed(0)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    for _ in range(2):
        for x in THINGS:
            s.company(None)
            s.listen(SAY % (x, FOOD[x]), IMITATED)
            s.end_of_turn(IMITATED, tone=".")
            mind.step([])
            s.watch(ASK % x, SAY % (x, FOOD[x]))
        s.watch("merhaba", "merhaba .")
    s.close_turn()
    s.look_gap = 0.0
    s.look_source = source
    s.may_look = source is not None
    return s


def chain_library() -> Library:
    return Library({CHAIN[i]: SAY % (CHAIN[i], CHAIN[i + 1]) for i in range(1, len(CHAIN) - 1)})


def told(s: Session, sentence: str) -> None:
    s.listen(sentence, TOLD, speaker=s.partner)
    s.end_of_turn(TOLD, tone=".")


def silence(s: Session, moments: int) -> list:
    said = []
    for _ in range(moments):
        acts = s.silent_moment(say=QUIET)
        if acts:
            said += s.speak(acts, None, 30)
            s.close_turn()
    return said


def names(mind, units) -> list:
    return [mind.ch.symbol(int(mind.S.u_channel[u]), int(mind.S.u_symbol[u])) for u in units]


def asks(s: Session, question: str) -> str:
    copy = Session(s.mind.clone())
    copy.may_look = False
    return copy.reply(question, limit=30)


def test_off_by_default_and_studied():
    assert Config().wonder is False and "wonder" in STUDIED


def test_new_is_what_was_met_in_one_run_of_items_only():
    s = raised(None, **ALL)
    mind = s.mind
    assert not mind.met_once                              # every word of its lessons was lived in more than one run
    told(s, SAY % ("kartal", "yılan"))
    assert names(mind, mind.met_once) == ["kartal", "yılan"]   # new, and "yer" is not
    mind.step([])
    told(s, "yılan uzun .")                               # one of them is lived again in another run: no longer new
    assert names(mind, mind.met_once) == ["kartal", "uzun"]
    mind.step([])
    before = dict(mind.met_once)
    s.reply("merhaba", limit=10)                          # what the mind says itself brings nothing new
    assert mind.met_once == before


def test_told_one_sentence_it_follows_the_chain_by_itself():
    s = raised(chain_library(), **ALL)
    mind = s.mind
    told(s, SAY % (CHAIN[0], CHAIN[1]))
    said = silence(s, 30)
    assert said == []                                     # nothing is said aloud: it is its own time
    questions = [q for q, _, _ in s.thoughts]
    assert questions == ["%s ne yer" % x for x in CHAIN]  # the question it knows, of each new thing, in order
    assert s.thoughts[0][1] == [CHAIN[0], CHAIN[1], "yer"] and s.thoughts[0][2] == 1.0   # the first it can answer
    assert all(reality == 0.0 for _, _, reality in s.thoughts[1:])                        # the others it cannot
    assert [r["topic"] for r in s.looks][:3] == CHAIN[1:4] and all(r["met"] is True for r in s.looks[:3])
    assert s.look_source.asked[:4] == CHAIN[1:5]          # the last thing was looked for too, and nothing found
    assert any(r["topic"] == CHAIN[4] and r["met"] is False for r in s.looks)
    assert sum(1 for entry in s.history if "[kendi kendine sordu]" in entry.text) == len(CHAIN)
    for i in range(len(CHAIN) - 1):                       # it has learned the whole chain
        assert asks(s, ASK % CHAIN[i]) == "%s %s yer" % (CHAIN[i], CHAIN[i + 1])
    assert asks(s, ASK % CHAIN[4]) == CHAIN[4]            # of the last it knows that it does not know: it holds
    assert wonder_expectation(mind) > 0.5                 # asking itself has paid
    assert not mind.met_once


def test_without_the_switch_nothing_of_this_happens():
    s = raised(chain_library(), **dict(ALL, wonder=False))
    told(s, SAY % (CHAIN[0], CHAIN[1]))
    assert silence(s, 30) == [] and not s.thoughts and not s.looks and s.look_source.asked == []
    assert asks(s, ASK % CHAIN[0]) == "%s %s yer" % (CHAIN[0], CHAIN[1])
    assert asks(s, ASK % CHAIN[1]) == CHAIN[1]


def test_with_nowhere_to_look_it_still_knows_what_it_does_not_know():
    s = raised(None, **ALL)
    mind = s.mind
    told(s, SAY % (CHAIN[0], CHAIN[1]))
    silence(s, 12)
    assert [q for q, _, _ in s.thoughts] == ["%s ne yer" % x for x in CHAIN[:2]] and not s.looks
    unit = mind.unit_of(CHANNEL, CHAIN[1])
    assert open_claim(mind, CHAIN[1]) > 0.5 and unit in mind.open_where and unit in mind.wondered
    assert open_claim(mind, CHAIN[0]) <= 0.5              # what it could answer is not open


def test_what_the_look_shows_is_what_the_asking_was_worth():
    met, unmet = raised(chain_library(), **ALL), raised(Library({}), **ALL)
    for s in (met, unmet):
        assert wonder_expectation(s.mind) == 0.5
        told(s, SAY % (CHAIN[0], CHAIN[1]))
        silence(s, 6)
    assert met.looks[0]["met"] is True and wonder_expectation(met.mind) > 0.5
    assert unmet.looks[0]["met"] is False and wonder_expectation(unmet.mind) < 0.5
    assert not unmet.mind.wondered                        # settled: the item no longer waits


def test_an_act_of_its_own_time_that_rarely_paid_is_taken_again_after_a_longer_silence():
    from zihin import ACTED
    from zihin.learn import EXPECT, W_NEG, add_evidence1
    from zihin.read import THINK_CHANNEL, WONDER
    s = raised(None, **ALL)
    mind = s.mind
    told(s, SAY % (CHAIN[0], CHAIN[1]))
    mind.step([])
    assert act.wonder(mind) is not None                   # with no record asking is driven by the unknown alone
    S = mind.S                                            # (as if) eight askings that led nowhere
    S.begin(mind.t + 1, mind.ch.sizes())
    mind.t += 1
    try:
        tie = S.tie(mind._atom(mind.add_channel(THINK_CHANNEL, "knob"), WONDER), mind.feel_unit["relief"], EXPECT, 1)
        add_evidence1(mind, W_NEG, ACTED, tie, 8.0, relief=False)
    finally:
        S.end()
    rate = 2.0 * wonder_expectation(mind)
    assert 0.0 < rate < mind.knob["theta"] / mind.cfg.T   # too weak to reach the line within a moment
    mind.idle = 1
    assert act.wonder(mind) is None
    mind.idle = 1000                                      # ... but not given up: after a long silence it asks again
    assert act.wonder(mind) is not None and act.wonder(mind)[1] == CHAIN[0]


def test_what_it_keeps_survives_saving_and_loading(tmp_path):
    from zihin import persist
    s = raised(None, **ALL)
    told(s, SAY % (CHAIN[0], CHAIN[1]))
    silence(s, 4)
    told(s, SAY % ("panda", "bambu"))
    path = str(tmp_path / "m.zihin")
    persist.save(s.mind, path)
    loaded = persist.load(path)
    assert loaded.met_once == s.mind.met_once and loaded.met_once
    assert loaded.wondered == s.mind.wondered and loaded.wondered
