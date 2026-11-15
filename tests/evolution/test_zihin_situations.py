# -*- coding: utf-8 -*-
"""Zihin F23: a situation never lived as a whole is met by the lived situations most like it.

What is pinned here
-------------------
* A question never watched as a whole, whose words and ending were lived in watched questions, is answered with
  what followed those situations, and the answer it starts is carried on along the answers that were watched.
* A question with a word nothing lived (an unknown topic) is not answered from resemblance: the unknown word weighs
  most, the resemblance is weak and the mind stays silent. Nothing in the session asks back for it (the curiosity
  reflex that said the unknown word back with a question tone is gone: asking is learned like any speech).
* A question watched as a whole keeps its own answer.
* Without the setting the mind behaves as before (silent on the unwatched question).
"""
from __future__ import annotations

from zihin import Config, Mind
from zihin.__main__ import CHANNEL, Session

WATCHED = [
    ("naber , ne yapıyorsun ?", "arkadaşlarla sinemaya gideceğim ."),
    ("selam , ne yapıyorsun ?", "arkadaşlarla sinemaya gideceğim ."),
    ("hey , ne yapıyorsun ?", "arkadaşlarla sinemaya gideceğim ."),
    ("günaydın , ne yapıyorsun ?", "arkadaşlarla sinemaya gideceğim ."),
    ("kahvaltıda ne yedin ?", "simit ve peynir yedim ."),
    ("akşam ne yedin ?", "çorba içtim ."),
    ("hava nasıl ?", "hava serin ."),
    ("kedi nedir ?", "kedi bir hayvandır ."),
]


def raised(situations: bool) -> Session:
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True, situations=situations))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    for _ in range(3):
        for question, answer in WATCHED:
            session.watch(question, answer)
    return session


def test_an_unwatched_question_is_met_by_the_situations_most_like_it():
    session = raised(True)
    assert session.reply("ne yapıyorsun ?").startswith("arkadaşlarla sinemaya gideceğim")


def test_an_unknown_topic_is_not_answered_from_resemblance_and_nothing_is_asked_back():
    session = raised(True)
    assert session.reply("kuantum nedir ?") == "(sessiz)"


def test_a_watched_question_keeps_its_own_answer():
    session = raised(True)
    assert session.reply("kahvaltıda ne yedin ?").startswith("simit ve peynir")


def test_without_the_setting_the_unwatched_question_gets_no_answer():
    session = raised(False)
    assert not session.reply("ne yapıyorsun ?").startswith("arkadaşlarla")
