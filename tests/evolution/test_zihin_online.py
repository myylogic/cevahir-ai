# -*- coding: utf-8 -*-
"""Zihin online life: the mind lives in its own time from the moment the session starts.

What is pinned here
-------------------
* A line arrives as an event and is heard; what the mind says comes out word by word as it says it.
* Between lines the mind lives silent moments: its age grows while nobody types.
* An innate rhythm, declared as such (Life.THINK_AFTER, Life.SLEEP_AFTER): after a short silence it thinks inwardly,
  after a long one it sleeps and dreams; each once per silence.
* Nothing is said for it in silence: no first speech after a while (its thought, or a word it wants to know with a
  question mark, were written into it as its own speech before). What its own decision emits in a silent moment is
  said, word by word until it decides nothing, lived once as its own echo, and is an entry of the history.
* Quitting ends its life loop.
"""
from __future__ import annotations

import time

from zihin import ACTED, TOLD, Config, Event, Mind
from zihin.__main__ import CHANNEL, Life, Session, _one_line


class Args:
    his = False


def living(**clock):
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True, situations=True))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    for _ in range(3):
        session.watch("kahvaltıda ne yedin ?", "simit ve peynir yedim .")
        session.live("güneş dünyayı ısıtır .", TOLD)
    heard = []

    def handle(line):
        heard.append(line)
        out = _one_line(session, line, True, Args(), lambda reason: None, 0)
        return out is not None
    import threading
    life = Life(session, handle, threading.RLock(), pace=0.0, **clock)
    words = []
    life.say = lambda w: words.append(w)
    session.voice = life.say
    return session, life, heard, words


def test_a_line_is_heard_and_the_answer_comes_out_word_by_word():
    session, life, heard, words = living(beat=0.05, think_after=60, sleep_after=0)
    life.start()
    life.put("kahvaltıda ne yedin ?")
    assert life.handled.wait(10)
    life.stop()
    assert heard == ["kahvaltıda ne yedin ?"]
    assert words[:3] == ["simit", "ve", "peynir"]


def test_between_lines_it_lives_thinks_and_sleeps_and_nothing_is_said_for_it(capsys):
    assert (Life.THINK_AFTER, Life.SLEEP_AFTER) == (10.0, 600.0)   # the innate rhythm, declared
    session, life, heard, words = living(beat=0.02, think_after=0.05, sleep_after=0.3)
    life.start()
    life.put("güneş")
    assert life.handled.wait(10)
    age, answer = session.mind.t, list(words)
    deadline = time.time() + 20
    while not life.slept and time.time() < deadline:
        time.sleep(0.05)
    life.stop()
    assert session.mind.t > age                          # silent moments were lived
    assert life.thought is not None and life.slept
    assert words == answer                               # nothing was said in the silence
    out = capsys.readouterr().out
    assert "kendiliğinden" not in out and "uyudum" in out
    assert not hasattr(life, "spoke") and not hasattr(life, "speak_after")


def test_quitting_ends_its_life():
    session, life, heard, words = living(beat=0.05, think_after=60, sleep_after=0)
    life.start()
    life.put("!cik")
    assert life.handled.wait(10)
    assert life.done.is_set()
    life.thread.join(5)
    assert not life.thread.is_alive()


def deciding_once(session, word: str):
    """The mind's step, which at its first silent moment from now on decides ``word`` (the act is pending for the next
    moment, as decide leaves it): a decision of the mind in a silent moment of its life. Returns the echoes lived as
    its own speech, recorded moment by moment."""
    mind, step, state, spoken = session.mind, session.mind.step, {"armed": True}, []

    def stepping(events=()):
        spoken.append([str(e.symbol) for e in list(mind.pending) + list(events)
                       if e.channel == CHANNEL and e.prov == ACTED])
        acts = step(events)
        if state["armed"] and not list(events) and not acts:
            state["armed"] = False
            act = Event(CHANNEL, word, ACTED)
            mind.pending.append(act)
            acts = [act]
        return acts
    mind.step = stepping
    return spoken


def test_what_its_own_decision_says_in_a_silent_moment_is_said(capsys):
    session, life, heard, words = living(beat=0.02, think_after=60, sleep_after=0)
    life.start()
    life.put("kahvaltıda ne yedin ?")
    assert life.handled.wait(10)
    answer = list(words)
    spoken = deciding_once(session, "merhaba")
    deadline = time.time() + 20
    while len(words) == len(answer) and time.time() < deadline:
        time.sleep(0.02)
    time.sleep(0.1)
    life.stop()
    assert words[len(answer):][:1] == ["merhaba"]        # said, as decided
    assert [w for m in spoken for w in m] == words[len(answer):]   # each word said lived once as its own echo
    assert "kendiliğinden" in capsys.readouterr().out
    entry = session.history[-1]
    assert entry.kind == "spoke" and entry.counts.get("merhaba") == 1 and entry.moments
