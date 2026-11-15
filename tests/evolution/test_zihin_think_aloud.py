# -*- coding: utf-8 -*-
"""Zihin: think, then speak (T8 in conversation).

What is pinned here
-------------------
* Asked something it has no answer to at once, the mind thinks about the words of the question it is least sure of
  and says a thought that is real enough (what it lived about that word), not a copy of the question.
* The thoughts are called to mind by the words as they are heard, before the utterance is taken in: what the mind
  knew, not an echo of the question. A word it never met calls nothing to mind.
"""
from __future__ import annotations

from zihin import TOLD, Config, Mind
from zihin.__main__ import CHANNEL, Session


def raised() -> Session:
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True, situations=True))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    for _ in range(3):
        session.live("güneş dünyayı ısıtır ve bitkileri büyütür .", TOLD)
        session.watch("kahvaltıda ne yedin ?", "simit ve peynir yedim .")
    return session


def test_it_thinks_about_the_question_and_says_what_it_lived():
    session = raised()
    answer = session.reply("güneş hakkında ne düşünüyorsun ?")
    assert answer.startswith("güneş dünyayı ısıtır")
    assert "hakkında ne düşünüyorsun" not in answer


def test_thoughts_are_what_it_knew_before_the_question_not_an_echo():
    session = raised()
    thoughts = session.associations("güneş hakkında ne düşünüyorsun ?")
    assert thoughts[0][0] == "güneş" and thoughts[0][1][:2] == ["dünyayı", "ısıtır"]
    assert all("hakkında" not in thought for _, thought, _ in thoughts)


def test_a_word_it_never_met_calls_nothing_to_mind():
    session = raised()
    assert session.associations("masa ne işe yarar ?") == []
