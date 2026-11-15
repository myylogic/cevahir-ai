# -*- coding: utf-8 -*-
"""Zihin: thoughts stay inner voice; what the mind says is only what its own decisions emit.

What is pinned here
-------------------
* Asked something it has not lived how to answer, the mind stays silent, although it can think about the topic: the
  thought is its inner voice (lived as imagined, nothing of it spoken), and no thought is said for it. The session
  used to think about the question's words before hearing them and write the first thought that was real enough
  into the mind as its own speech, with the topic word put in front of it by code (removed).
* Nothing is thought about an utterance before it is heard: the first moment of a reply perceives the first word
  said to it, and no imagined event is lived during a reply.
* The words said are exactly the words its decisions emitted, each lived once as the echo of its own act, in order;
  no moment perceives the text channel twice.
"""
from __future__ import annotations

from zihin import ACTED, IMAGINED, TOLD, Config, Mind
from zihin.__main__ import CHANNEL, Session
from zihin.dusunce import think


def raised() -> Session:
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True, situations=True))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    for _ in range(3):
        session.live("güneş dünyayı ısıtır ve bitkileri büyütür .", TOLD)
        session.watch("kahvaltıda ne yedin ?", "simit ve peynir yedim .")
    return session


def moments(session, run):
    """The events each moment perceives while ``run`` happens (what was pending from the last moment, then the new
    events)."""
    mind, seen, step = session.mind, [], session.mind.step

    def recorded(events=()):
        seen.append(list(mind.pending) + list(events))
        return step(events)
    mind.step = recorded
    try:
        out = run()
    finally:
        mind.step = step
    return out, seen


def spoken(seen) -> list:
    """The words lived as the mind's own speech (its acts on the text channel) in the recorded moments."""
    return [str(e.symbol) for events in seen for e in events if e.channel == CHANNEL and e.prov == ACTED]


def test_a_thought_is_never_said_for_the_mind():
    session = raised()
    (thought, reality), inner = moments(session, lambda: think(session, "güneş"))
    assert thought[:2] == ["dünyayı", "ısıtır"] and reality > 0.5      # it can think about the topic: its inner
    assert not spoken(inner)                             # voice, nothing of it spoken
    answer, seen = moments(session, lambda: session.reply("güneş hakkında ne düşünüyorsun ?"))
    assert answer == "(sessiz)"                          # it has not lived how to answer this question ...
    assert not spoken(seen)                              # ... and no thought is said for it


def test_nothing_is_thought_before_the_utterance_is_heard():
    assert not hasattr(Session, "associations") and not hasattr(Session, "think_aloud")
    session = raised()
    _, seen = moments(session, lambda: session.reply("güneş hakkında ne düşünüyorsun ?"))
    first = [e for e in seen[0] if e.channel == CHANNEL]
    assert [(e.symbol, e.prov) for e in first] == [("güneş", TOLD)]
    assert not [e for events in seen for e in events if e.prov == IMAGINED]


def test_the_words_said_are_exactly_its_decisions():
    session = raised()
    answer, seen = moments(session, lambda: session.reply("kahvaltıda ne yedin ?"))
    assert answer == "simit ve peynir yedim"
    assert spoken(seen) == answer.split()                # each word it said, lived once as its own act, in order
    assert all(len([e for e in events if e.channel == CHANNEL]) <= 1 for events in seen)
