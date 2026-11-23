# -*- coding: utf-8 -*-
"""Zihin, F61 One Race for Its Own Time and F62 The Sentence Being Told: what was found when everything the mind
has was turned on together, and the two mechanisms that came of it.

What is pinned here
-------------------
F61 (``Config.own_race``)
* Off by default, a studied mechanism, in use in the full configuration.
* The acts of the mind's own time (looking something up, asking itself, thinking a thing over, thinking about what
  it took in) are candidates each by its own rule and rate; in a silent moment the one that reaches the line first
  is decided, among equals the one with the higher rate, then the one not taken for longest. At most one is
  decided; the wants of the others stay.
* An act whose rate reaches the line only after a wait is ready when it has itself waited that long (its own count
  of the silent moments in which it wanted and was not taken), is taken once, and waits again; an act that wants
  nothing waits for nothing.
* A look the mind has nowhere to carry out takes no part.
* Without it the acts are taken in the order they were built and each gives way to those before it: a mind that
  always has something new to ask itself about never thinks anything over (no conjecture is made in a life in
  which the same mind without F51 makes them). With the race it comes to the same conjectures.

* A look waits its own time too: one whose rate is over nothing but reaches the line only after a wait (looking
  has rarely paid) is not left out as before: it is looked up when it has waited that long.
* While an act it decided is carried out (the pauses of an inner question, a thought, a reading) nothing is decided
  about its own time: one decision a silent moment.
* What it asked itself and could not answer, and can say when it comes to look (it learned it meanwhile), is no
  longer wondered about: nothing is fetched and the look is not decided again.

F65 (``Config.unknown_outside``, with L7)
* Off by default, a studied mechanism, in use in the full configuration.
* The amount the mind does not know is kept when the world has just done something and it decided nothing; a
  silent moment later it perceives that as an inner event (L7). The end of a question it asked itself is imagined:
  not the world doing something. Without F65 the inner event follows its own unanswered question and is lived as
  the beginning of the next thing it hears: a young mind then never learns the question it is watching as the
  question it is, and asks itself malformed questions about every new thing. With F65 the event is never
  perceived in such a life, and it asks itself the question it watched.
* What the world left it without an answer to is felt as before.

F62 (``Config.echo_told``, with ``echo_exact``)
* Off by default, a studied mechanism, in use in the full configuration; it needs the exact echo (F52).
* F52 alone carries on whatever once followed the mind's last word, though the partner just said it: asked a kind
  of question it never watched, of a thing it read a sentence of, it says the value and then the question's own
  verb. With F62 the partner's word is carried on only where the mind's utterance, from its beginning, is how a
  sequence it lived began: the answer ends with the value, and a sentence it read is still told to its end.
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Config, Mind, act, dusunce, varsayim, web
from zihin.__main__ import CHANNEL, NEEDS, STUDIED, Session, VARIANTS
from zihin.mind import INNER_CHANNEL, NOT_KNOWING


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def born(variant: str = "speaking", **switches) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS[variant]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    for name, value in switches.items():
        setattr(cfg, name, value)
    torch.manual_seed(0)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    return Session(mind)


def tell(s: Session, sentence: str, silent: int = 0) -> None:
    s.company(None)
    s.listen(sentence, IMITATED)
    s.end_of_turn(IMITATED, tone=".")
    if not silent:
        s.mind.step([])
    for _ in range(silent):
        s.silent_moment(say=lambda *_: None)


# ------------------------------------------------------------------------------------------------ F61
def test_both_are_off_by_default_studied_and_in_the_full_configuration():
    plain = Config()
    assert plain.own_race is False and plain.echo_told is False
    assert "own_race" in STUDIED and "echo_told" in STUDIED
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        full, speaking = VARIANTS["full"](), VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    assert full.own_race and full.echo_told and not speaking.own_race and not speaking.echo_told
    assert ("echo_told", ("echo_exact",)) in NEEDS


def wants(monkeypatch, mind, look=None, wonder=None, suppose=None, think=None) -> None:
    """The four acts as candidates with the given rates (None: not a candidate now)."""
    monkeypatch.setattr(act, "look", lambda m: None if look is None else (7, "x", look))
    monkeypatch.setattr(act, "wonder", lambda m: None if wonder is None else (8, "y", wonder))
    monkeypatch.setattr(act, "suppose", lambda m: None if suppose is None else 9)
    monkeypatch.setattr(varsayim, "worth", lambda m: 0.5 if suppose is None else suppose / 2.0)
    monkeypatch.setattr(act, "think_now", lambda m: think)
    mind.looking = mind.wondering = mind.supposing = mind.thinking = None


def decided(mind) -> str:
    slots = {"look": mind.looking, "wonder": mind.wondering, "suppose": mind.supposing, "think": mind.thinking}
    taken = [name for name, value in slots.items() if value is not None]
    assert len(taken) <= 1                                   # one act a moment
    return taken[0] if taken else ""


def test_the_one_that_reaches_the_line_first_is_decided(monkeypatch):
    mind = born("full").mind
    theta = mind.knob["theta"]
    wants(monkeypatch, mind, look=theta / 3.0, wonder=theta, suppose=theta / 2.0, think=theta / 4.0)
    act.own_time(mind)                                       # latencies 3, 1, 2, 4
    assert decided(mind) == "wonder" and mind.wondering == (8, "y", theta)
    wants(monkeypatch, mind, look=theta / 3.0, suppose=theta / 2.0, think=theta / 4.0)
    act.own_time(mind)
    assert decided(mind) == "suppose" and mind.supposing == 9
    wants(monkeypatch, mind)
    act.own_time(mind)
    assert decided(mind) == ""                               # nothing wants the moment: nothing is decided


def test_among_equals_the_higher_rate_then_the_one_not_taken_for_longest(monkeypatch):
    mind = born("full").mind
    theta = mind.knob["theta"]
    wants(monkeypatch, mind, wonder=theta * 0.6, suppose=theta * 0.9)     # both reach the line in two moments
    act.own_time(mind)
    assert decided(mind) == "suppose"                        # the firmer of the two
    order = []
    for moment in range(6):                                  # equal in everything: they take turns
        mind.t += 1
        wants(monkeypatch, mind, wonder=theta, suppose=theta, think=theta)
        act.own_time(mind)
        order.append(decided(mind))
    assert order[:3] == ["wonder", "think", "suppose"]       # (suppose was taken last; then the order they were built)
    assert order[3:] == order[:3]                            # each in its turn, none left out
    assert set(mind.own_taken) == {"wonder", "suppose", "think"}


def test_an_act_that_rarely_paid_waits_its_own_time_and_is_never_given_up(monkeypatch):
    """A rate that reaches the line only in twenty moments: the act is ready when it has itself waited twenty silent
    moments, whatever the others did meanwhile; it is taken once and waits again. An act that wants nothing waits
    for nothing."""
    mind = born("full").mind
    theta, T = mind.knob["theta"], mind.cfg.T
    slow, order = theta / 20.0, []
    assert 20 > T
    for moment in range(45):
        mind.t += 1
        wants(monkeypatch, mind, look=None, wonder=slow, suppose=theta)
        act.own_time(mind)
        order.append(decided(mind))
    assert order.count("wonder") == 2 and order.index("wonder") == 20      # after twenty moments of its own waiting
    assert [i for i, name in enumerate(order) if name == "wonder"] == [20, 41]   # ... once, and again after as long
    assert all(name == "suppose" for name in order if name != "wonder")    # the other took the rest of the moments
    assert mind.own_waited["wonder"] == 3
    wants(monkeypatch, mind, suppose=theta)                                # it wants nothing now: its count is gone
    act.own_time(mind)
    assert "wonder" not in mind.own_waited
    wants(monkeypatch, mind, wonder=slow)                                  # nothing else wants the moment, and still
    act.own_time(mind)                                                     # it is not ready before its time
    assert decided(mind) == "" and mind.own_waited["wonder"] == 1


def test_a_look_it_has_nowhere_to_carry_out_takes_no_part(monkeypatch):
    s = born("full")
    mind = s.mind
    theta = mind.knob["theta"]
    assert mind.__dict__["can_look"] is False                # the session gave it nowhere to look
    wants(monkeypatch, mind, look=theta, suppose=theta / 3.0)
    act.own_time(mind)
    assert decided(mind) == "suppose"
    s.may_look = True
    wants(monkeypatch, mind, look=theta, suppose=theta / 3.0)
    act.own_time(mind)
    assert decided(mind) == "look"


ANIMALS = ["aslan", "kedi", "yılan", "kertenkele", "serçe", "kartal", "kurt", "timsah"]
CLASS = {"aslan": "memeli", "kedi": "memeli", "kurt": "memeli", "yılan": "sürüngen", "kertenkele": "sürüngen",
         "timsah": "sürüngen", "serçe": "kanatlı", "kartal": "kanatlı"}
BEARS = {"memeli": "yavru", "sürüngen": "yumurta", "kanatlı": "civciv"}
EATS = {"aslan": "et", "kedi": "balık", "yılan": "tohum", "kertenkele": "ot", "serçe": "et", "kartal": "balık",
        "kurt": "ot", "timsah": "tohum"}


def life(variant: str = "speaking", **switches) -> Session:
    """Of every class what it bears; of every animal its class and what it eats; of three animals also what their
    class bears. A silent moment after every sentence, and silence at the end."""
    s = born(variant, **switches)
    for k in BEARS:
        tell(s, "%s %s doğurur ." % (k, BEARS[k]), 1)
    for x in ANIMALS:
        tell(s, "%s %s sınıfındandır ." % (x, CLASS[x]), 1)
        tell(s, "%s %s yer ." % (x, EATS[x]), 1)
    for x in ("aslan", "yılan", "serçe"):
        tell(s, "%s %s doğurur ." % (x, BEARS[CLASS[x]]), 1)
    for _ in range(120):
        s.silent_moment(say=lambda *_: None)
    return s


def conjectured(s: Session) -> set:
    return {" ".join(str(s.mind.symbol_of(u)[1]) for u in entry["statement"]) for entry in s.mind.conjectures.values()}


def test_without_the_race_the_whole_mind_never_thinks_anything_over():
    alone = conjectured(life(conjectures=True))                          # F59 as it was measured: by itself
    assert alone == {"%s %s doğurur" % (x, BEARS[CLASS[x]]) for x in ANIMALS if x not in ("aslan", "yılan", "serçe")}
    starved = life("full", own_race=False)                               # everything, in the order it was built
    assert conjectured(starved) == set() and starved.mind.pondering      # things wait to be thought over, for good
    raced = life("full")
    assert conjectured(raced) == alone and not raced.mind.pondering
    assert "suppose" in raced.mind.own_taken and len(raced.mind.own_taken) > 1   # more than one act got its time


def test_what_it_lives_in_summary_is_not_something_new_it_was_told():
    """F43 lives a watched dialogue once more in summary, on a channel of its own. Those items are the mind's own
    wholes and pointers: nothing of them is new to ask itself about (F51) or told to think over (F59). Found in the
    online life of the whole mind: it asked itself its question of an item that is no word ("ne yer -> bilmiyor")."""
    s = born("full")
    mind = s.mind
    for x, v in (("kedi", "balık"), ("köpek", "et"), ("kuş", "tohum")):
        s.watch("%s ne yer ?" % x, "%s %s yer ." % (x, v))
    tell(s, "serçe bir kuştur .")
    text = mind.ch.id(CHANNEL)
    assert mind.ch.has("ozet")                               # summaries were lived ...
    channel = mind.S.arrays().channel
    assert mind.met_once and all(int(channel[u]) == text for u in mind.met_once)     # ... and none of their items
    assert all(int(channel[u]) == text for u in mind.pondering)                      # is new, or to be thought over
    asked = []
    for _ in range(40):
        s.silent_moment(say=asked.append)
    inner = [line for line in asked if "kendi kendine sordu" in line]
    assert inner and all(len(line.split("):")[1].split("->")[0].split()) >= 3 for line in inner)   # "<word> ne yer"


# ------------------------------------------------------------------------------------------------ F62
THINGS = ["inek", "koyun", "keçi", "at", "eşek", "tavuk", "deve", "geyik", "aslan", "kurt", "fok", "serçe"]
FACTS = {"yer": ["ot", "et", "balık", "tohum"], "içer": ["su", "süt", "çay", "şerbet"],
         "taşır": ["dal", "taş", "yaprak", "kum"]}


def fact(x: str, q: str) -> str:
    return FACTS[q][(THINGS.index(x) * 3 + len(q)) % 4]


def asked_world(**switches) -> Session:
    """Every fact read of every thing; two kinds of question watched of the first eight things, answered with the
    value alone; the third kind (taşır) never watched."""
    s = born("speaking", known=True, families=True, lived_apart=True, asked_place=True, **switches)
    for i, x in enumerate(THINGS):
        sentences = ["%s %s %s ." % (x, fact(x, q), q) for q in FACTS]
        for sentence in sentences[i % 3:] + sentences[:i % 3]:
            tell(s, sentence)
    for _ in range(2):
        for q in ("yer", "içer"):
            for x in THINGS[:8]:
                s.watch("%s ne %s ?" % (x, q), "%s ." % fact(x, q))
    s.close_turn()
    return s


def answers(s: Session, q: str) -> list:
    return [Session(s.mind.clone()).reply("%s ne %s ?" % (x, q), limit=20) for x in THINGS[8:]]


def test_the_exact_echo_alone_goes_on_past_the_answer_and_the_sentence_being_told_does_not():
    right = [fact(x, "taşır") for x in THINGS[8:]]
    assert answers(asked_world(), "taşır") == right                           # the echo rule as it was
    assert answers(asked_world(echo_exact=True), "taşır") == ["%s taşır" % v for v in right]   # F52: the verb too
    told = asked_world(echo_exact=True, echo_told=True)
    assert answers(told, "taşır") == right                                   # F62: it ends with the value
    assert answers(told, "yer") == [fact(x, "yer") for x in THINGS[8:]]      # (a kind it watched, as before)


class Library:
    """A source that stands where the web stands: one sentence about a thing."""

    def __init__(self, facts: dict) -> None:
        self.facts = facts

    def search(self, topic: str, limit: int = 2):
        return [topic] if topic in self.facts else []

    def extract(self, identifier: str) -> str:
        return self.facts[identifier]

    def address(self, identifier: str) -> str:
        return "library:" + identifier


FOOD = ["ot", "et", "balık", "tohum"]
LOOKS = []                   # how many looks of its own the last lives of ``looked_up`` made


def looked_up(**switches) -> list:
    """F50's world (what F52 was built for): eight things read of and watched being asked, answered with the whole
    sentence; then the mind is asked about things it has read nothing of, looks them up and answers in that turn."""
    s = born("speaking", faults=True, open_claims=True, look=True, find_out=True, **switches)
    food = {x: FOOD[0] if i < 8 else FOOD[1 + i % 3] for i, x in enumerate(THINGS)}
    s.look_gap = 0.0
    s.look_source = Library({x: "%s %s yer ." % (x, food[x]) for x in THINGS})
    s.may_look = True
    for _ in range(2):
        for x in THINGS[:8]:
            tell(s, "%s %s yer ." % (x, food[x]))
            s.watch("%s ne yer ?" % x, "%s %s yer ." % (x, food[x]))
        s.watch("merhaba", "merhaba .")
    s.close_turn()
    said = []
    for x in THINGS[8:]:
        said.append((s.reply("%s ne yer ?" % x, limit=30), "%s %s yer" % (x, food[x])))
        s.close_turn()
        s.watch("%s ne yer ?" % x, "%s %s yer ." % (x, food[x]))
        s.close_turn()
    LOOKS.append(len(s.looks))
    return said


def test_a_sentence_it_read_is_still_told_to_its_end():
    """What F52 was built for: after it has looked a thing up, the verb the question holds follows the value,
    because the sentence the mind read of the thing goes on so. F62 keeps exactly that."""
    plain, exact, told = looked_up(), looked_up(echo_exact=True), looked_up(echo_exact=True, echo_told=True)
    whole = [sum(1 for answer, right in said if answer == right) for said in (plain, exact, told)]
    assert whole[1] > whole[0]                               # F52 completes answers the echo rule left at the value
    assert told == exact                                     # ... and F62 answers word for word as F52 does here


# ------------------------------------------------------------------------------------------------ F63
def test_relied_on_is_off_by_default_studied_and_needs_silent_claims():
    assert Config().relied_on is False and "relied_on" in STUDIED
    assert ("relied_on", ("silent_claims",)) in NEEDS


def test_agreeing_with_what_it_already_knew_does_not_prove_what_others_share():
    """F50's world. The mind read a sentence of each of eight things, which share what they eat, and watched them
    asked. With silent claims alone its generalisation counts each of those dialogues as a claim that held (it
    agreed with what the mind had read of that very thing), is proven, and of things it has read nothing of the
    mind asserts what the eight share instead of looking it up. With F63 it is not proven by that, and the mind
    looks first, as it does without silent claims."""
    right = 4
    careful = looked_up(known=True)
    assert sum(1 for answer, truth in careful if answer == truth) == right and LOOKS[-1] == 4
    bold = looked_up(known=True, silent_claims=True)
    assert sum(1 for answer, truth in bold if answer == truth) < right and LOOKS[-1] < 4
    assert any(answer.split()[1:2] == [FOOD[0]] for answer, _ in bold)       # what the eight share, said of another
    relied = looked_up(known=True, silent_claims=True, relied_on=True)
    assert relied == careful and LOOKS[-1] == 4


def test_where_nothing_was_lived_of_the_thing_silent_claims_are_kept():
    """What silent claims were built for stays: of things it has read nothing of, what it watches is a test of what
    others share, with F63 as without it (the same record, tie for tie)."""
    from zihin import hata
    records = []
    for switches in ({"silent_claims": True}, {"silent_claims": True, "relied_on": True}):
        s = born("speaking", faults=True, **switches)
        for x in THINGS[:8]:                                 # nothing read: only watched
            s.watch("%s ne yer ?" % x, "%s ot yer ." % x)
        s.close_turn()
        records.append(hata.report(s.mind))
    assert any("tuttu" in line for line in records[0])       # ways have a record of their own from watching alone
    assert records[0] == records[1]


def test_without_the_exact_echo_the_switch_changes_nothing():
    assert answers(asked_world(echo_told=True), "taşır") == answers(asked_world(), "taşır")


# ------------------------------------------------------------------------------------------------ F65, its own time
YOUNG = ["koyun", "kurbağa", "fok", "inek", "maymun", "baykuş"]
QUIET = lambda *_: None      # noqa: E731


def young(silent: int = 3, **switches) -> Session:
    """A young mind: of each thing a sentence is read and a question about it is watched being answered, with
    ``silent`` moments of its own after each (it asks itself, looks, thinks things over while it knows little)."""
    s = born("full", **switches)
    s.look_gap = 0.0
    s.look_source = Library({})
    s.may_look = True
    for i, x in enumerate(YOUNG):
        tell(s, "%s %s yer ." % (x, FOOD[i % 4]), silent)
        s.watch("%s ne yer ?" % x, "%s %s yer ." % (x, FOOD[i % 4]))
        s.close_turn()
        for _ in range(silent):
            s.silent_moment(say=QUIET)
    return s


def test_unknown_outside_is_off_by_default_studied_and_in_the_full_configuration():
    assert Config().unknown_outside is False and "unknown_outside" in STUDIED
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        assert VARIANTS["full"]().unknown_outside is True and VARIANTS["speaking"]().unknown_outside is False
    finally:
        chat._log_path = made


def inner_event(s: Session) -> int:
    return s.mind.unit_of(INNER_CHANNEL, NOT_KNOWING)


def test_a_question_it_asked_itself_is_not_the_world_asking():
    whole, without = young(), young(unknown_outside=False)
    assert inner_event(whole) < 0            # the world left nothing unanswered: not knowing was never an event
    assert inner_event(without) >= 0         # the end of its own unanswered question was taken for the world's
    later = ["%s ne yer" % x for x in YOUNG[2:]]
    asked = [entry[0] for entry in whole.thoughts]
    assert all(question in asked for question in later)      # it asks itself the question it watched, of each thing
    asked = [entry[0] for entry in without.thoughts]
    assert not any(question in asked for question in later)  # the watched questions were lived as what follows the
    assert "koyun %s yer" % YOUNG[4] in asked                # inner event: the first sentence it read stays the
    assert len(whole.looks) < len(without.looks)             # question it knows best, and it looks for what it


def test_what_the_world_left_unanswered_is_still_felt():     # was told a moment ago
    s = young()
    mind = s.mind
    assert inner_event(s) < 0
    said = s.reply("zebra nerede ?", limit=10)               # asked from outside what it has no answer to
    for _ in range(2):
        mind.step([])                                        # a silent moment: the state is perceived (L7)
    assert said == "(sessiz)" and inner_event(s) >= 0


def test_what_it_can_now_say_is_no_longer_wondered_about():
    s = young()
    mind = s.mind
    tell(s, "zebra .")
    zebra = mind.unit_of(CHANNEL, "zebra")
    asked, said, unknown = dusunce.ask_itself(s, zebra)
    assert asked == ["zebra", "ne", "yer"] and unknown and zebra in mind.wondered
    tell(s, "zebra ot yer .")                                # it learns it before it comes to look
    s.watch("zebra ne yer ?", "zebra ot yer .")
    s.close_turn()
    mind.idle = 1 << 30
    assert act.look(mind)[:2] == (zebra, "zebra")            # (the want is still there: it asked and did not look)
    report = web.look_up(s, (zebra, "zebra", 1.0), source=Library({}), say=QUIET)
    assert report["note"] == "kendi bildiğiyle" and report["met"] is True
    assert zebra not in mind.wondered and act.look(mind) is None   # ... and ends there: not decided again


def test_a_look_that_rarely_paid_waits_its_own_time_in_the_race(monkeypatch):
    import zihin.read as read
    s = young()
    mind = s.mind
    tell(s, "zebra .")
    zebra = mind.unit_of(CHANNEL, "zebra")
    mind.open_where, mind.open_plain, mind.wondered = {zebra: mind.unit_of(CHANNEL, "zebra")}, {zebra: False}, {}
    monkeypatch.setattr(read, "open_claim", lambda m, symbol: 0.75)
    monkeypatch.setattr(read, "look_expectation", lambda m, symbol: 0.27)   # rate 0.5 - 0.46: the line in 25 moments
    theta, T = mind.knob["theta"], mind.cfg.T
    assert theta == 1.0 and 25 > T
    mind.cfg.own_race = False
    assert act.look(mind) is None                            # without the race: over T, left out
    mind.cfg.own_race = True
    monkeypatch.setattr(act, "wonder", lambda m: None)
    monkeypatch.setattr(act, "suppose", lambda m: None)
    monkeypatch.setattr(act, "think_now", lambda m: None)
    order = []
    for moment in range(30):
        mind.t += 1
        mind.looking = mind.wondering = mind.supposing = mind.thinking = None
        act.own_time(mind)
        order.append(decided(mind))
    assert [i for i, name in enumerate(order) if name] == [25] and order[25] == "look"


def test_nothing_is_decided_while_an_act_is_carried_out(monkeypatch):
    calls = []
    real = act.own_time
    monkeypatch.setattr(act, "own_time", lambda m: (calls.append(m.t), real(m))[1])
    s = young()
    asked = len(s.thoughts)
    assert asked >= 6                                        # (inner questions and thoughts: each has pauses)
    before, moments = len(calls), 0
    for _ in range(6):
        t = s.mind.t
        s.silent_moment(say=QUIET)
        moments += 1
        assert len([c for c in calls[before:] if c > t]) <= 1    # one decision a silent moment
        before = len(calls)
    assert "_carrying" not in s.mind.__dict__
