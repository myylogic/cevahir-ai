# -*- coding: utf-8 -*-
"""Zihin L7: not knowing, felt, and how it is said, learned.

What is pinned here
-------------------
* When the world calls on the mind (someone's turn ends) and it decides nothing, the amount it does not know is
  measured; if the next moment passes in silence it perceives that state as an inner event, alone in its moment.
  It answers what it knows without feeling it; watching others never makes it feel it.
* Before it has lived how one says it, the mind does not say "bilmiyorum" (it only asks back the unknown word).
* After lessons lived at the moment it felt not knowing, it says "bilmiyorum" about something it does not know,
  asks about the unknown word, and still answers what it knows.
"""
from __future__ import annotations

from zihin import Config, Mind
from zihin.__main__ import CHANNEL, Session
from zihin.mind import INNER_CHANNEL, NOT_KNOWING


def raised() -> Session:
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True, situations=True))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    for _ in range(3):
        session.watch("kahvaltıda ne yedin ?", "simit ve peynir yedim .")
        session.watch("hava nasıl ?", "hava serin .")
    return session


def felt_events(session, run):
    """Inner not-knowing events perceived while ``run`` happens."""
    mind, seen, step = session.mind, [], session.mind.step

    def recorded(events=()):
        seen.extend(e for e in list(mind.pending) + list(events) if e.channel == INNER_CHANNEL)
        return step(events)
    mind.step = recorded
    try:
        run()
    finally:
        mind.step = step
    return seen


def test_not_knowing_is_felt_only_when_called_on_and_without_an_answer():
    session = raised()
    assert felt_events(session, lambda: session.reply("kahvaltıda ne yedin ?")) == []
    assert felt_events(session, lambda: session.watch("kedi nedir ?", "kedi bir hayvandır .")) == []
    felt = felt_events(session, lambda: session.reply("zorbalak nedir ?"))
    assert len(felt) == 1 and felt[0].symbol == NOT_KNOWING and 0.0 < felt[0].intensity <= 1.0


def test_before_lessons_it_does_not_say_bilmiyorum():
    session = raised()
    answer = session.reply("kuantum nedir ?")
    assert "bilmiyorum" not in answer and answer.endswith("kuantum ?")


def test_after_lessons_it_says_it_does_not_know_asks_and_still_answers_what_it_knows():
    session = raised()
    session.lessons_about_itself(2)
    unknown = session.reply("fotosentez ne demek ?")
    assert unknown.startswith("bilmiyorum") and unknown.endswith("fotosentez ?")
    assert session.reply("kahvaltıda ne yedin ?").startswith("simit ve peynir")
