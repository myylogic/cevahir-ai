# -*- coding: utf-8 -*-
"""Zihin online life: the mind lives in its own time from the moment the session starts.

What is pinned here
-------------------
* A line arrives as an event and is heard; what the mind says comes out word by word as it says it.
* Between lines the mind lives silent moments: its age grows while nobody types.
* After a short silence it thinks, after a longer one it may speak first (when its thought was real enough), after
  a long one it sleeps and dreams; each once per silence.
* Quitting ends its life loop.
"""
from __future__ import annotations

import time

from zihin import TOLD, Config, Mind
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
    session, life, heard, words = living(beat=0.05, think_after=60, speak_after=60, sleep_after=0)
    life.start()
    life.put("kahvaltıda ne yedin ?")
    assert life.handled.wait(10)
    life.stop()
    assert heard == ["kahvaltıda ne yedin ?"]
    assert words[:3] == ["simit", "ve", "peynir"]


def test_between_lines_it_lives_thinks_speaks_first_and_sleeps(capsys):
    session, life, heard, words = living(beat=0.02, think_after=0.05, speak_after=0.1, sleep_after=0.3,
                                         real_enough=0.0)
    life.start()
    life.put("güneş")
    assert life.handled.wait(10)
    age = session.mind.t
    deadline = time.time() + 20
    while not life.slept and time.time() < deadline:
        time.sleep(0.05)
    life.stop()
    assert session.mind.t > age                          # silent moments were lived
    assert life.thought is not None and life.spoke and life.slept
    out = capsys.readouterr().out
    assert "kendiliğinden" in out and "uyudum" in out


def test_quitting_ends_its_life():
    session, life, heard, words = living(beat=0.05, think_after=60, speak_after=60, sleep_after=0)
    life.start()
    life.put("!cik")
    assert life.handled.wait(10)
    assert life.done.is_set()
    life.thread.join(5)
    assert not life.thread.is_alive()


def test_with_no_real_thought_it_asks_about_what_it_wants_to_know(capsys):
    session, life, heard, words = living(beat=0.02, think_after=0.05, speak_after=0.1, sleep_after=0,
                                         real_enough=1.01)   # no thought is real enough
    life.start()
    life.put("güneş")
    assert life.handled.wait(10)
    deadline = time.time() + 20
    while not life.spoke and time.time() < deadline:
        time.sleep(0.05)
    life.stop()
    assert life.spoke and words[-1] == "?"
