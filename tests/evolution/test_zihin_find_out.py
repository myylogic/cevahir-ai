# -*- coding: utf-8 -*-
"""Zihin, F50 Finding Out: what the mind does not know it acts on, in the turn it is asked in, and it goes on living
while it does.

What is pinned here
-------------------
* The switch is off by default and is one of the studied mechanisms (``--ac find_out``; with open_claims, look and
  faults).
* A small world: eight things read of ("X V yer .") and watched being asked ("X ne yer ?"); they share one value.
  A library holds one sentence about other things. Asked about a thing it has read nothing of, the mind is about to
  say the word the eight share: that word rests on what others share, not on what it lived of the thing asked about
  (``act.grounded``). With somewhere to look it does not say it: it looks the thing up within the turn, is brought
  back to what it was asked, and says what it read. Without F50 it says the shared word in that turn.
* With nowhere to look it holds the word (it knows that of this thing it has no record: the others answered
  from theirs) where before it asserted it; what it lived it says as before.
* Holding back is itself judged by what follows: while what the mind holds back is then shown to it, it goes on
  holding what rests on what others share; where no one ever answers, it says it (that is the only way left to
  find out). A shared word whose way is proven by its own record (it held more often than it failed by more than
  one experience: one lucky hit is not knowledge) is said: it is no longer a guess.
* The rest of the sentence it read: with F41 (what is known of the thing and not said yet) or with F52 (the echo
  rule in its exact form) the answer after a look is the whole sentence; without either, the value alone.
* One look is decided for a thing until the mind is next called on: a second word it would have said in the same
  turn does not move what the first look was for.
* What was lived of the thing itself stands against what the others share: a mind that has read about the thing
  says what it read; without the rule the two claims tie and it says nothing after the asked word.
* That it is looking is an inner event; how one says so is never given: a mind that has not lived it says nothing
  then, and one that heard a word said in that state says that word itself the next time.
* The mind goes on living while what it sent for is on its way: the turn ends at once, it answers something else
  meanwhile, one look at a time; when it has come it reads it and says what it can say now.
* A look that finds nothing: the claim stays open, looking is worth less, and the mind does not assert the shared
  word in that turn.
* A thought that comes to nothing is an unknown about its topic: a plain open claim that the mind may look up.
* What is open in this way survives saving and loading.
"""
from __future__ import annotations

import threading

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, TOLD, Config, Mind
from zihin.__main__ import CHANNEL, STUDIED, Session, VARIANTS
from zihin.mind import INNER_CHANNEL, LOOKING
from zihin.read import look_expectation, open_claim

THINGS = ["inek", "koyun", "keçi", "at", "eşek", "tavuk", "deve", "geyik"]
FACTS = {"aslan": "et", "kurt": "et", "fok": "balık", "tavşan": "ot", "serçe": "tohum", "kaplan": "et"}
ASK, SAY = "%s ne yer ?", "%s %s yer ."


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


class Library:
    """A source that stands where the web stands: one sentence about each thing it holds."""

    def __init__(self, facts: dict, gate: threading.Event = None) -> None:
        self.facts, self.asked, self.gate = dict(facts), [], gate

    def search(self, topic: str, limit: int = 2):
        if self.gate is not None:
            self.gate.wait(10.0)                           # a service that takes its time
        self.asked.append(topic)
        return [topic] if topic in self.facts else []

    def extract(self, identifier: str) -> str:
        return SAY % (identifier, self.facts[identifier])

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
            s.listen(SAY % (x, "ot"), IMITATED)
            s.end_of_turn(IMITATED, tone=".")
            mind.step([])
            s.watch(ASK % x, SAY % (x, "ot"))
        s.watch("merhaba", "merhaba .")
    s.close_turn()
    s.look_gap = 0.0
    s.look_source = source
    s.may_look = source is not None
    return s


ALL = dict(open_claims=True, look=True, faults=True, find_out=True)
BEFORE = dict(open_claims=True, look=True, faults=True)


def test_off_by_default_and_studied():
    assert Config().find_out is False and "find_out" in STUDIED


def test_asked_about_a_thing_it_never_read_of_it_looks_and_answers_in_the_same_turn():
    s, before = raised(Library(FACTS), **ALL), raised(Library(FACTS), **BEFORE)
    assert before.reply(ASK % "aslan", limit=30) == "aslan ot yer"       # without F50: the word the eight share
    assert not before.looks
    answer = s.reply(ASK % "aslan", limit=30)
    assert answer.split()[:2] == ["aslan", "et"] and "ot" not in answer.split()   # what it read, never the shared word
    assert [r["topic"] for r in s.looks] == ["aslan"] and s.looks[0]["met"] is True
    assert s.look_source.asked == ["aslan"]
    assert any("bakıyor: aslan" in note for note in s.look_notes)
    s.close_turn()
    assert s.reply(ASK % "tavşan", limit=30) == "tavşan ot yer"          # a thing that does eat what the eight eat:
    assert len(s.looks) == 2 and s.looks[1]["met"] is True               # looked up too, and said whole
    s.close_turn()
    assert s.reply(ASK % "inek", limit=30) == "inek ot yer" and len(s.looks) == 2   # what it lived it does not look up


def test_with_nowhere_to_look_it_holds_what_it_knows_it_does_not_know():
    s, before = raised(None, **ALL), raised(None, **BEFORE)
    assert s.mind.__dict__.get("can_look") is False
    assert before.reply(ASK % "aslan", limit=30) == "aslan ot yer"       # before: asserted, and wrong
    assert s.reply(ASK % "aslan", limit=30) == "aslan"      # the eight answered from their own record; of this thing
    assert not s.looks                                      # it has none, and it cannot look: it holds the word
    s.close_turn()
    before.close_turn()
    assert s.reply(ASK % "inek", limit=30) == before.reply(ASK % "inek", limit=30) == "inek ot yer"   # what it lived


def test_holding_back_is_judged_by_whether_it_is_then_answered():
    from zihin import hata
    never = raised(None, **ALL)                             # no one shows it anything, and it cannot look
    said = []
    for x in ("aslan", "kurt", "fok"):
        said.append(never.reply(ASK % x, limit=30))
        never.close_turn()
    assert said[0] == "aslan" and hata.answered(never.mind) < 0.5   # held once; called on again, still unanswered
    assert said[1:] == ["kurt ot yer", "fok ot yer"]        # then it says it: the only way left to find out
    shown = raised(None, **ALL)                             # someone shows the true dialogue after every question
    said = []
    for x in ("aslan", "kurt", "fok"):
        said.append(shown.reply(ASK % x, limit=30))
        shown.close_turn()
        shown.watch(ASK % x, SAY % (x, FACTS[x]))
        shown.close_turn()
    assert said == ["aslan", "kurt", "fok"] and hata.answered(shown.mind) > 0.5   # it goes on holding: 0 wrong


def test_a_shared_word_whose_way_is_proven_is_said():
    s = raised(None, **ALL)
    said = []
    for x in ("tavşan", "ceylan", "zebra", "manda", "bizon", "lama"):   # things that do eat what the eight eat
        said.append(s.reply(ASK % x, limit=30))
        s.close_turn()
        s.watch(ASK % x, SAY % (x, "ot"))
        s.close_turn()
    assert said[:2] == ["tavşan", "ceylan"]                 # untested, then one lucky hit: held, and then shown
    assert said[2:4] == ["zebra ot", "manda ot"]            # the value's way has held twice: said; the next word's
    assert said[4:] == ["bizon ot yer", "lama ot yer"]      # way then has too: the whole answer


def test_the_rest_of_the_sentence_it_read():
    answers = {}
    for name, more in (("alone", {}), ("exact echo", {"echo_exact": True}), ("known", {"known": True})):
        s = raised(Library(FACTS), **dict(ALL, **more))
        answers[name] = [s.reply(ASK % "aslan", limit=30)]
        s.close_turn()
        answers[name].append(s.reply(ASK % "fok", limit=30))
    assert answers["alone"] == ["aslan et", "fok balık"]    # the value it read, and no wrong word
    assert answers["exact echo"] == answers["known"] == ["aslan et yer", "fok balık yer"]
    assert Config().echo_exact is False and "echo_exact" in STUDIED


def test_one_look_is_decided_for_a_thing_until_it_is_next_called_on():
    gate = threading.Event()
    s = raised(Library(FACTS, gate), **ALL)
    s.background = True
    quiet = lambda *_: None  # noqa: E731
    for x in ("aslan", "kurt"):                             # the word one says while looking, lived twice ...
        gate.clear()
        s.reply(ASK % x, limit=30)
        s.listen("bakalım", TOLD, speaker=s.partner)        # ... each time followed by the partner's own "bakalım"
        s.end_of_turn(TOLD, tone=".")
        gate.set()
        s.fetching[2].join(5.0)
        s.speak(s.silent_moment(say=quiet), None, 30)
        s.close_turn()
    gate.clear()
    first = s.reply(ASK % "fok", limit=30)                  # it says the word, would say it again, and does not
    where = s.mind.open_where[s.mind.unit_of(CHANNEL, "fok")]
    assert first.split() == ["fok", "bakalım"] and s.fetching[0][1] == "fok"
    assert names_of(s.mind, where)[-1] == "fok"             # the look is still for what it was asked about
    gate.set()
    s.fetching[2].join(5.0)
    assert s.speak(s.silent_moment(say=quiet), None, 30)[:1] == ["balık"] and s.looks[-1]["met"] is True


def names_of(mind, unit) -> list:
    from zihin.form import parts_of
    return [str(mind.ch.symbol(int(mind.S.u_channel[p]), int(mind.S.u_symbol[p]))) for p in parts_of(mind, unit)]


def test_what_was_lived_of_the_thing_itself_stands_against_what_the_others_share():
    answers = {}
    for name, switches in (("rule", ALL), ("no rule", BEFORE)):
        s = raised(None, **switches)
        s.company(None)
        s.listen(SAY % ("serçe", "tohum"), IMITATED)        # it has read about the thing itself
        s.end_of_turn(IMITATED, tone=".")
        s.mind.step([])
        answers[name] = Session(s.mind.clone()).reply(ASK % "serçe", limit=30)
    assert answers["no rule"] == "serçe"                    # the two claims tie: nothing after the asked word
    assert answers["rule"].split()[:2] == ["serçe", "tohum"]


def test_that_it_is_looking_is_an_inner_event_and_how_one_says_so_is_learned():
    gate = threading.Event()
    s = raised(Library(FACTS, gate), **ALL)
    s.background = True
    mind = s.mind
    seen = []
    live = mind.step

    def spy(events=()):
        seen.extend(e for e in list(mind.pending) if e.channel == INNER_CHANNEL)
        return live(events)
    mind.step = spy
    quiet = lambda *_: None  # noqa: E731
    firsts = []
    for i, x in enumerate(("aslan", "kurt", "fok", "serçe")):
        gate.clear()
        firsts.append(s.reply(ASK % x, limit=30))
        if i < 2:                                           # someone says a word right after the mind began to look
            s.listen("bakalım", TOLD, speaker=s.partner)
            s.end_of_turn(TOLD, tone=".")
        gate.set()
        s.fetching[2].join(5.0)
        s.speak(s.silent_moment(say=quiet), None, 30)
        s.open_turn = True                                  # (its own utterance is a turn: the online life marks it)
        s.close_turn()
    assert any(e.symbol == LOOKING and 0.0 < e.intensity <= 1.0 for e in seen)
    assert "bakalım" not in firsts[0].split() and "bakalım" not in firsts[1].split()   # never lived: not said
    assert firsts[2].split() == ["fok", "bakalım"] and firsts[3].split() == ["serçe", "bakalım"]


def test_the_mind_goes_on_living_while_what_it_sent_for_is_on_its_way():
    gate = threading.Event()
    s = raised(Library(FACTS, gate), **ALL)
    s.background = True
    quiet = lambda *_: None  # noqa: E731
    first = s.reply(ASK % "aslan", limit=30)
    assert first == "aslan" and s.fetching is not None and s.fetching[2].is_alive()   # the turn has ended; it is fetching
    assert not s.looks and s.look_source.asked == []
    assert s.reply("merhaba", limit=10) == "merhaba"        # it answers something else meanwhile
    moment = s.mind.t
    assert s.silent_moment(say=quiet) == [] and s.mind.t > moment and s.fetching is not None   # it lives on; nothing yet
    again = s.reply(ASK % "kurt", limit=30)                 # another unknown while one look is on its way: it waits
    assert again == "kurt" and s.fetching[0][1] == "aslan"
    gate.set()
    s.fetching[2].join(5.0)
    late = s.speak(s.silent_moment(say=quiet), None, 30)    # it has come: it reads, and says what it was asked
    assert late[:1] == ["et"] and s.fetching is None and [r["topic"] for r in s.looks] == ["aslan"]
    assert s.looks[0]["met"] is True


def test_a_look_that_finds_nothing():
    s = raised(Library({}), **ALL)
    mind = s.mind
    before = look_expectation(mind, "aslan")
    answer = s.reply(ASK % "aslan", limit=30)
    assert "ot" not in answer.split() and answer.split()[:1] == ["aslan"]   # it does not assert the shared word
    assert len(s.looks) == 1 and s.looks[0]["met"] is False and s.looks[0].get("note") == "bulunamadı"
    assert look_expectation(mind, "aslan") < before == 0.5              # looking is worth less
    assert open_claim(mind, "aslan") > 0.5 and mind.unit_of(CHANNEL, "aslan") in mind.open_plain   # still open


def test_a_thought_that_comes_to_nothing_is_an_unknown_about_its_topic():
    from zihin import act
    from zihin.learn import note_unknown
    s = raised(Library({"zebra": "ot"}), **ALL)
    mind = s.mind
    s.listen("zebra", TOLD, speaker=s.partner)              # a word it has met and can say nothing about
    s.end_of_turn(TOLD, tone=".")
    mind.step([])
    unit = mind.unit_of(CHANNEL, "zebra")
    assert open_claim(mind, "zebra") == 0.5 and unit not in mind.open_plain
    note_unknown(mind, unit, unit)                          # (Session.silent_moment does this for an empty thought)
    assert open_claim(mind, "zebra") > 0.5 and mind.open_plain[unit] is False and mind.open_where[unit] == unit
    mind.step([])                                           # a silent moment: it decides to look it up
    assert mind.looking is not None and mind.looking[1] == "zebra"
    assert act.look(mind)[1] == "zebra"


def test_what_is_open_in_this_way_survives_saving_and_loading(tmp_path):
    from zihin import persist
    s = raised(Library({}), **ALL)
    s.reply(ASK % "aslan", limit=30)
    s.close_turn()
    path = str(tmp_path / "m.zihin")
    persist.save(s.mind, path)
    loaded = persist.load(path)
    assert loaded.open_plain == s.mind.open_plain and loaded.open_plain
    assert loaded.open_where == s.mind.open_where
