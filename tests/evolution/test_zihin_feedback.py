# -*- coding: utf-8 -*-
"""Zihin, feedback on an answer (!dogru / !yanlis) while the mind lives online.

What is pinned here
-------------------
* The path that led to an answer stays available for its feedback until the feedback or the next utterance: after
  the silent moments the life clock lives in between, and after a thought, the value reaches the same path with the
  same evidence as feedback typed at once: the lag-2 ties of every step of the path to the value (L5). Only that
  path matches. The value's lag-1 ties differ, because time passed: typed at once the answer's last focus is tied to
  the value that follows it; after silent moments the moment before the value is a silent one and nothing is.
* Typed at once, feedback is what it always was: the kept path is the episode's trace itself.
* "not edildi" is said only when the value evidence grew. A silence of hesitation leaves no path (its silent moments
  ended the episode), and after the mind spoke on its own the answer is no longer reached: the program says that
  nothing was learned. A silence after not knowing leaves the episode of its not knowing (L7) as its path, and the
  feedback reaches it: no coded question closes the reply any more.
* Through the real life clock: "!dogru" typed after several silent beats is learned.
"""
from __future__ import annotations

import threading
import time

from zihin import Config, Mind
from zihin.__main__ import CHANNEL, VALUE, Life, Session, _one_line
from zihin.dusunce import think
from zihin.store import EXPECT, W_POS

DIALOGUES = [("Merhaba!", "Merhaba, nasılsın?"), ("Adın ne?", "Benim adım Cevahir."),
             ("Kedi ne yer?", "Kedi balık yer."), ("Köpek ne yer?", "Köpek kemik yer."),
             ("Gökyüzü ne renk?", "Gökyüzü mavidir."), ("Ankara nerede?", "Ankara Türkiye'nin başkentidir.")]


class Args:
    his = False


def taught() -> Session:
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True, situations=True, explore=True))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    for question, answer in DIALOGUES:
        session.watch(question, answer)
    return session


def value_paths(mind, label: str = "iyi", lag: int = 2) -> dict:
    """The units tied to a value at a lag (2: the steps of the path, "later in the episode"; 1: what came just before
    it) and their positive evidence."""
    value = mind.unit_of(VALUE, label)
    if value < 0:
        return {}
    S = mind.S
    return {int(S.t_src[t]): float(S.w1(W_POS, t).sum()) for t in S.inc.get((value, EXPECT), [])
            if int(S.t_lag[t]) == lag}


def answered_then(silent: int, thought: bool = False):
    session = taught()
    answer = session.reply("Kedi ne yer?")
    for _ in range(silent):                              # what the life clock does every beat while you are silent
        session.mind.step([])
    if thought:
        think(session, "kedi")                           # ... and after a short silence it thinks
    return answer, session.feedback(True), value_paths(session.mind), value_paths(session.mind, lag=1)


def test_feedback_after_silent_moments_reaches_the_same_path_as_feedback_at_once():
    at_once = answered_then(0)
    assert at_once[0] == "kedi balık yer" and "not edildi (iyi)" in at_once[1]
    assert at_once[2] and all(mass > 0 for mass in at_once[2].values())
    assert at_once[3]                                    # typed at once: the answer's last focus is tied to the value
    for later in (answered_then(1), answered_then(3), answered_then(3, thought=True)):
        assert later[:3] == at_once[:3]                  # the same answer, message and lag-2 path ...
        assert later[3] == {} != at_once[3]              # ... but not the same lag-1 ties: time passed


def test_typed_at_once_the_kept_path_is_the_trace_itself():
    session = taught()
    session.reply("Kedi ne yer?")
    assert session.answered and session.answered == session.mind.trace


def test_an_answer_that_left_no_path_is_not_reported_as_learned():
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True, situations=True))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    for _ in range(2):                                   # two answers equally supported: it hesitates, it does not
        session.watch("hangi renk ?", "mavi .")          # feel not knowing, and its silent moments end the episode
        session.watch("hangi renk ?", "yeşil .")
    assert session.reply("hangi renk ?") == "(sessiz)" and session.answered == []
    message = session.feedback(True)
    assert "not edildi" not in message and "öğrenilmedi" in message
    assert value_paths(session.mind) == {}


def test_feedback_on_a_silence_after_not_knowing_reaches_that_episode():
    from zihin.mind import INNER_CHANNEL, NOT_KNOWING
    session = taught()
    assert session.reply("kuantum nedir ?") == "(sessiz)"   # nothing is asked back for it (no curiosity reflex)
    felt = session.mind.unit_of(INNER_CHANNEL, NOT_KNOWING)
    assert session.answered == [felt]                    # the path is the episode it lived: its not knowing
    assert "not edildi (iyi)" in session.feedback(True)  # the owner's feedback reaches it (nothing cut it off)
    assert list(value_paths(session.mind)) == [felt]


def living(**clock):
    session = taught()

    def handle(line):
        out = _one_line(session, line, True, Args(), lambda reason: None, 0)
        return out is not None
    life = Life(session, handle, threading.RLock(), pace=0.0, **clock)
    said = []
    life.say = lambda w: said.append(w)
    session.voice = life.say
    return session, life, said


def test_through_the_life_clock_feedback_after_silent_beats_is_learned(capsys):
    session, life, _ = living(beat=0.02, think_after=60, speak_after=60, sleep_after=0)
    life.start()
    life.put("Kedi ne yer?")
    assert life.handled.wait(10)
    answered_at = session.mind.t
    deadline = time.time() + 20
    while session.mind.t < answered_at + 3 and time.time() < deadline:   # three silent beats at least
        time.sleep(0.02)
    assert session.mind.t >= answered_at + 3 and session.mind.trace == []
    life.put("!dogru")
    assert life.handled.wait(10)
    life.stop()
    out = capsys.readouterr().out
    assert "not edildi (iyi)" in out
    assert value_paths(session.mind) == answered_then(0)[2]      # the lag-2 path, as typed at once


def test_after_the_mind_spoke_on_its_own_feedback_no_longer_reaches_the_answer(capsys):
    _, life, said = living(beat=0.02, think_after=0.05, speak_after=0.1, sleep_after=0, real_enough=0.0)
    life.start()
    life.put("Kedi ne yer?")
    assert life.handled.wait(10)
    answer_words = len(said)
    deadline = time.time() + 20
    while not life.spoke and time.time() < deadline:
        time.sleep(0.02)
    assert life.spoke and len(said) > answer_words      # it said something of its own after its answer
    life.put("!dogru")
    assert life.handled.wait(10)
    life.stop()
    out = capsys.readouterr().out
    assert "kendiliğinden konuştu" in out and "not edildi" not in out
