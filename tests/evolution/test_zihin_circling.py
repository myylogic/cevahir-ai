# -*- coding: utf-8 -*-
"""Zihin: how speech and the inner voice end.

What is pinned here
-------------------
* Speech ends when the mind decides nothing, or when the echo of its last word brings back a composed context this
  episode already lived (it has come round to a situation it was in): that echo is heard and the mind does not
  decide again there. No echo is dropped and nothing it decided is left unsaid; no moment perceives the text channel
  twice. (Before, a hand rule stopped speech when its last 1-4 words repeated the 1-4 before them, and the pending
  echo of the word it had decided was deleted.)
* The length limit is only a safety cap, ended the same way.
* The inner voice ends the same way, every word lived as imagined; a topic of several words is taken in without
  deciding until its last word, so no decision of the mind is perceived beside the next topic word.
"""
from __future__ import annotations

from zihin import ACTED, IMAGINED, Config, Mind
from zihin.__main__ import CHANNEL, Session
from zihin.dusunce import think

SONG = " ".join(["la"] * 12)


def raised() -> Session:
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True, situations=True))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    for _ in range(3):
        session.watch("şarkı söyle", SONG)
        session.watch("kahvaltıda ne yedin ?", "simit ve peynir yedim .")
    return session


def moments(session, run):
    """(what ``run`` returned, the text events each moment perceived: what was pending, then the new events)."""
    mind, seen, step = session.mind, [], session.mind.step

    def recorded(events=()):
        seen.append([e for e in list(mind.pending) + list(events) if e.channel == CHANNEL])
        return step(events)
    mind.step = recorded
    try:
        out = run()
    finally:
        mind.step = step
    return out, seen


def of(seen, prov) -> list:
    return [str(e.symbol) for events in seen for e in events if e.prov == prov]


def test_speech_ends_where_its_situation_comes_round():
    session = raised()
    answer, seen = moments(session, lambda: session.reply("şarkı söyle"))
    said = answer.split()
    assert set(said) == {"la"} and 2 < len(said) < 12     # more than the old rule let through, and ended by
    assert of(seen, ACTED) == said                        # coming round, not by the length it watched
    assert all(len(events) <= 1 for events in seen)
    assert not [e for e in session.mind.pending if e.channel == CHANNEL]   # nothing decided is left unsaid


def test_the_length_limit_is_only_a_safety_cap_and_drops_no_echo():
    session = raised()
    answer, seen = moments(session, lambda: session.reply("şarkı söyle", limit=3))
    assert answer == "la la la" and of(seen, ACTED) == ["la", "la", "la"]
    assert not [e for e in session.mind.pending if e.channel == CHANNEL]


def test_the_inner_voice_ends_the_same_way():
    session = raised()
    (thought, reality), seen = moments(session, lambda: think(session, "la"))
    assert set(thought) == {"la"} and 2 < len(thought) < 15
    assert of(seen, IMAGINED) == ["la"] + thought and of(seen, ACTED) == []
    assert all(len(events) <= 1 for events in seen)


def test_a_topic_of_several_words_is_taken_in_without_deciding_until_its_last_word():
    session = raised()
    (thought, _), seen = moments(session, lambda: think(session, "kahvaltıda ne yedin"))
    assert of(seen, IMAGINED)[:3] == ["kahvaltıda", "ne", "yedin"] and of(seen, ACTED) == []
    assert all(len(events) <= 1 for events in seen)
