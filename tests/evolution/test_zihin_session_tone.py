# -*- coding: utf-8 -*-
"""Zihin session: the closing punctuation of an utterance is heard the way the mind lived it.

What is pinned here
-------------------
* A new mind hears the closing punctuation (? ! .) as the tone of the turn's end: it is not a word of the utterance.
* A mind that lived turn ends with the punctuation as a word (minds raised before tones existed) keeps hearing it as a
  word and gets no tone: a question it watched is the same sequence when it is asked, so it is answered. Hearing it
  the new way would make every watched question a sequence the mind never lived (it fell silent).
* The convention follows the mind across sessions: a session never switches a mind between the two.
"""
from __future__ import annotations

from zihin import TOLD, Config, Mind
from zihin.__main__ import CHANNEL, TONE, Session


def speaking_mind() -> Mind:
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True))
    mind.add_channel(CHANNEL, "act")
    return mind


def tone_symbols(mind) -> int:
    return len([t for t in "?!." if mind.ch.has(TONE) and
                mind.ch.symbol_index(mind.ch.id(TONE), t, create=False) >= 0])


def test_a_new_mind_hears_closing_punctuation_as_tone():
    session = Session(speaking_mind())
    assert session.turns and session.tones
    assert session.body("adın ne ?") == ["adın", "ne"]
    session.watch("adın ne ?", "adım cevahir .")
    assert tone_symbols(session.mind) == 2


def test_a_mind_that_lived_punctuation_as_words_keeps_it_and_answers_what_it_watched():
    session = Session(speaking_mind())
    session.tones = False                                   # raised the older way: "?" was a word before the turn end
    for _ in range(3):
        session.watch("kahvaltıda ne yedin ?", "simit ve peynir .")
        session.watch("hava nasıl ?", "biraz bulutlu .")
    mind = session.mind
    assert tone_symbols(mind) == 0
    again = Session(mind)                                   # a later session with the same mind
    assert again.turns and not again.tones
    assert again.body("hava nasıl ?") == ["hava", "nasıl", "?"]
    assert again.reply("kahvaltıda ne yedin ?").startswith("simit ve peynir")
    assert tone_symbols(mind) == 0                          # the session did not switch it to tones


def test_a_mind_that_lived_tones_keeps_them():
    session = Session(speaking_mind())
    for _ in range(3):
        session.watch("kahvaltıda ne yedin ?", "simit ve peynir .")
    again = Session(session.mind)
    assert again.tones
    assert again.reply("kahvaltıda ne yedin ?").startswith("simit ve peynir")
