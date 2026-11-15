# -*- coding: utf-8 -*-
"""Zihin L7: not knowing is felt; how it is said is never given by the core.

What is pinned here
-------------------
* When the world calls on the mind (someone's turn ends) and it decides nothing, the amount it does not know is
  measured; if the next moment passes in silence it perceives that state as an inner event, alone in its moment.
  It answers what it knows without feeling it; watching others never makes it feel it.
* The core holds no sentences to say: a mind that has not lived how one says it does not say "bilmiyorum".
* There are no scripted lessons about the mind or about not knowing in the core (removed at the owner's decision).
* Nothing in the session asks back for the mind: the curiosity reflex that wrote the unknown word and '?' into the mind
  as its own acts is gone. A mind that has not lived how to ask stays silent; nothing is lived as its speech.
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


def test_a_mind_that_never_lived_it_does_not_say_bilmiyorum():
    session = raised()
    answer = session.reply("kuantum nedir ?")
    assert "bilmiyorum" not in answer


def test_the_core_holds_no_scripted_lessons():
    import zihin.kendilik as kendilik
    for name in ("LESSONS", "not_knowing_lessons", "_SYLLABLES", "_ASKING", "_NOT_KNOWING"):
        assert not hasattr(kendilik, name)
    for name in ("lessons_about_itself", "not_knowing_lesson"):
        assert not hasattr(Session, name)


def test_nothing_in_the_session_asks_back_for_the_mind():
    from zihin import ACTED
    assert not hasattr(Session, "wonder") and not hasattr(Session, "known_before")
    empty = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                                backoff=True, resemblance=True, race=True, situations=True))
    empty.add_channel(CHANNEL, "act")
    for session in (Session(empty), raised()):
        assert not hasattr(session, "curious")
        mind, acted, step = session.mind, [], session.mind.step

        def recorded(events=()):
            acted.extend(e for e in list(mind.pending) + list(events) if e.channel == CHANNEL and e.prov == ACTED)
            return step(events)
        mind.step = recorded
        assert session.reply("zorbalak nedir ?") == "(sessiz)"
        assert acted == []                               # nothing was lived as its own speech
