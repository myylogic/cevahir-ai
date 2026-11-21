# -*- coding: utf-8 -*-
"""Zihin, F43 Wholes (an utterance held as parts and answered part by part), F44 Lived Exclusion across channels and
F42 Episode Echo.

What is pinned here
-------------------
* A small world of commands (three primitives and one seen only alone; "twice", "thrice", "and") lived with a window
  of 7 items. Commands with the primitive seen alone whose dialogue does not fit the window ("d thrice and d twice":
  five words, the end and five actions) are answered exactly with F43 and F44, and not without them.
* What the summary of a lived dialogue is (``align``): the wholes that stand in the utterance as their forms, the
  replies as pointers to their places; nothing when two readings tie ("a and a").
* Considering a reading never lived (``consider``) leaves nothing: the substance, its sizes, the clock and the knobs
  are what they were.
* The working state set aside for a part is taken up again as it was.
* F44: an instance that was lived to go on with the end of the turn counts against a word it never went on with; without
  the switch F27 says nothing across channels.
* F42: what followed the last word earlier in the present episode is not read as its lived continuation.
* The switches are off by default; the speaking configuration has F42 and F44, not F43.
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import ACTED, IMITATED, Config, Mind
from zihin.__main__ import CHANNEL, TURN, TURN_END, Session, VARIANTS
from zihin import butun as B
from zihin.events import Event
from zihin.read import continuation_backoff, excluded

PRIMITIVES = {"a": "ia", "b": "ib", "c": "ic", "d": "id"}    # (the text adapter lower-cases: actions are other words)


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fresh(window: int = 7, **switches) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    for name, value in switches.items():
        setattr(cfg, name, value)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    mind.knob["F"] = window
    return Session(mind)


def simple(x: str):
    act = PRIMITIVES[x]
    return [((x,), (act,)), ((x, "twice"), (act,) * 2), ((x, "thrice"), (act,) * 3)]


def lived(**switches) -> Session:
    """The life: every command of a, b and c (alone, twice, thrice, and two of them joined by "and"), and "d" alone;
    two epochs, the shorter commands first."""
    s = fresh(**switches)
    parts = [d for x in "abc" for d in simple(x)]
    train = parts + [(p + ("and",) + q, a + b) for p, a in parts for q, b in parts] + [(("d",), ("id",))]
    for _ in range(2):
        for command, answer in sorted(train, key=lambda d: len(d[0])):
            s.watch(" ".join(command), " ".join(answer) + " .")
    return s


@pytest.fixture(scope="module")
def with_wholes() -> Session:
    return lived(wholes=True, exclusive_world=True)


def ask(s: Session, command: str) -> str:
    return Session(s.mind.clone()).reply(command, limit=200)


def test_switches_are_off_by_default():
    cfg = Config()
    assert cfg.wholes is False and cfg.exclusive_world is False and cfg.echo is False
    s = fresh()                                                       # the speaking configuration: F42 and F44 are
    assert s.mind.cfg.wholes is False                                 # in it, F43 is not
    assert s.mind.cfg.echo is True and s.mind.cfg.exclusive_world is True


def test_commands_longer_than_the_window_are_answered_in_parts(with_wholes):
    cases = {"d twice": "id id", "d thrice and a twice": "id id id ia ia", "b thrice and d thrice": "ib ib ib id id id",
             "d thrice and d twice": "id id id id id", "d and d thrice": "id id id id"}
    for command, answer in cases.items():
        assert ask(with_wholes, command) == answer, command


def test_without_wholes_a_dialogue_that_does_not_fit_the_window_is_not_answered():
    s = lived(wholes=False)
    assert ask(s, "d twice") == "id id"                                # fits the window: answered as before
    assert ask(s, "d thrice and d twice") != "id id id id id"          # five words, the end, five actions: eleven items


def test_summary_of_a_lived_dialogue(with_wholes):
    mind, F = with_wholes.mind, with_wholes.mind.knob["F"]
    whole = B.WHOLE
    assert B.lived_reply(mind, ("a", "twice")) == ("ia", "ia")
    assert B.lived_reply(mind, ("d", "twice")) is None
    assert B.align(mind, ("a", "twice"), ("ia", "ia"), F, {}) == ((whole, "twice"), (("reply", 0),) * 2, {0: (0, 1)})
    joined = B.align(mind, ("a", "thrice", "and", "b", "twice"), ("ia", "ia", "ia", "ib", "ib"), F, {})
    assert joined == ((("whole", whole, "thrice"), "and", ("whole", whole, "twice")), (("reply", 0), ("reply", 2)),
                      {0: (0, 2), 2: (3, 5)})
    assert B.align(mind, ("a", "and", "a"), ("ia", "ia"), F, {}) is None    # two readings tie: nothing is concluded
    assert B.align(mind, ("a", "twice"), ("ib", "ic"), F, {}) is None       # not made of the reply of a whole in it


def test_reading_an_utterance_never_answered(with_wholes):
    mind, F = with_wholes.mind, with_wholes.mind.knob["F"]
    found = B.reading(mind, ("d", "thrice", "and", "d", "twice"), F, {})
    assert found == ((("whole", B.WHOLE, "thrice"), "and", ("whole", B.WHOLE, "twice")), {0: (0, 2), 2: (3, 5)})
    assert B.reading(mind, ("thrice", "and", "twice"), F, {}) is None   # no whole stands in it


def test_considering_a_reading_leaves_nothing(with_wholes):
    twin = Session(with_wholes.mind.clone())
    mind = twin.mind
    before = (mind.S.digest(), mind.t, mind.S.n_units, mind.S.n_ties, len(mind.S.tuple_index), dict(mind.knob),
              mind.S.keep_log, mind.cfg.undo_depth, list(mind.episode), mind.context)
    firm = B.consider(twin, (B.WHOLE, "thrice", "and", ("whole", "never", "lived")))
    assert 0.0 <= firm <= 1.0
    after = (mind.S.digest(), mind.t, mind.S.n_units, mind.S.n_ties, len(mind.S.tuple_index), dict(mind.knob),
             mind.S.keep_log, mind.cfg.undo_depth, list(mind.episode), mind.context)
    assert before == after
    assert mind.unit_of(B.LEVEL, ("whole", "never", "lived")) == -1


def test_working_state_set_aside_is_taken_up_again(with_wholes):
    twin = Session(with_wholes.mind.clone())
    mind = twin.mind
    twin.listen("a twice", IMITATED)
    held_before = {name: getattr(mind, name) for name in B.WORKING}
    held = B.set_aside(mind)
    assert mind.episode == [] and mind.context is None and mind.pending == []
    twin.listen("b", IMITATED)                                         # another episode is lived meanwhile
    B.take_up(mind, held)
    for name, value in held_before.items():
        assert getattr(mind, name) == value, name


def test_exclusion_across_channels():
    for across in (False, True):
        s = fresh(exclusive_world=across)
        mind = s.mind
        s.watch("a", "ia .")
        end, word = mind.unit_of(TURN, TURN_END), mind.unit_of(CHANNEL, "ia")
        out = {word: 0.8}
        excluded(mind, out, {end: (3.0, 0.0)})                         # the instance always went on with the end
        if across:
            assert out[word] == pytest.approx(0.8 * (1.0 - 3.0 / (3.0 + mind.cfg.k)))
        else:
            assert out[word] == 0.8                                    # F27 reads one channel at a time


def test_echo_is_not_read_as_what_follows():
    for echo in (False, True):
        s = fresh(echo=echo)
        mind = s.mind
        for _ in range(3):
            s.watch("q", "x y .")                                      # x is lived to go on with y
        s.listen("q", IMITATED)
        s.end_of_turn(IMITATED, tone=".")
        for word in ("x", "y", "x"):                                   # the mind's own words: x y x
            mind.step([Event(CHANNEL, word, ACTED)])
        y = mind.unit_of(CHANNEL, "y")
        went = continuation_backoff(mind, mind.context)
        if echo:
            assert y not in went                                       # y followed x earlier in this episode
        else:
            assert went.get(y, 0.0) > 0.5


def test_undo_drops_the_reading_of_situations():
    """The situations' reading is kept by unit and moment; both come again after an undo, so an undo drops it."""
    mind = Mind(Config(log_path=None))
    mind.add_channel(CHANNEL, "act")
    mind.step([Event(CHANNEL, "a", IMITATED)])
    mind.step([Event(CHANNEL, "b", IMITATED)])
    mind._situation_cache = ((0, mind.t, None), {})
    mind.undo_last()
    assert "_situation_cache" not in mind.__dict__


def test_a_summary_is_also_lived_as_its_construction(with_wholes):
    """The construction of a summary: its words as they are, every whole bare. A summary whose forms never met in it
    is heard as its construction when that was lived and answered."""
    mind = with_wholes.mind
    whole = B.WHOLE
    summary = (("whole", whole, "thrice"), "and", ("whole", whole, "twice"))
    assert B.construction(summary) == (whole, "and", whole)
    assert B.went_on(mind, B.construction(summary))                      # lived as a construction too
    assert B._heard_as(mind, summary) == summary                         # lived as it is: heard as it is
    never = (("whole", whole, "never"), "and", whole)
    assert not B.went_on(mind, never) and B._heard_as(mind, never) == (whole, "and", whole)
