# -*- coding: utf-8 -*-
"""Zihin: hesitation ends in a try, and consequences teach which alternative to keep (learning code by its tests).

What is pinned here
-------------------
* With two equally supported answers the race leaves nothing leading; with exploration the mind tries one instead of
  falling silent, and the same alternative is chosen again under the same knowledge (deterministic).
* An alternative that led to "kötü" is not tried again while untried alternatives remain; after "iyi" the mind keeps
  the one that led there.
* Nothing well supported is not hesitation: a question about something unknown still meets not knowing (the mind
  feels it, L7) and no try is made; nothing is said for it (the curiosity reflex is gone).
* Writing tiny functions: after watching mostly wrong completions, with the test results as consequences the mind
  writes passing code for every task within a few rounds; without the consequences it does not.
"""
from __future__ import annotations

import os
import sys

from zihin import Config, Mind
from zihin.__main__ import CHANNEL, Session


def raised(explore: bool = True) -> Session:
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True, situations=True, explore=explore))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    for _ in range(2):
        session.watch("hangi renk ?", "mavi .")
        session.watch("hangi renk ?", "yeşil .")
    return session


def test_hesitation_ends_in_a_try_instead_of_silence():
    silent = raised(explore=False).reply("hangi renk ?")
    first = raised().reply("hangi renk ?")
    again = raised().reply("hangi renk ?")
    assert silent == "(sessiz)"
    assert first in ("mavi", "yeşil") and again == first


def test_a_bad_try_is_not_repeated_and_a_good_one_is_kept():
    session = raised()
    first = session.reply("hangi renk ?")
    session.feedback(False)
    second = session.reply("hangi renk ?")
    assert second in ("mavi", "yeşil") and second != first
    session.feedback(True)
    assert session.reply("hangi renk ?") == second


def test_nothing_well_supported_is_still_not_knowing():
    from zihin.mind import INNER_CHANNEL, NOT_KNOWING
    session = raised()
    mind, felt, step = session.mind, [], session.mind.step

    def recorded(events=()):
        felt.extend(e.symbol for e in list(mind.pending) + list(events) if e.channel == INNER_CHANNEL)
        return step(events)
    mind.step = recorded
    assert session.reply("kuantum nedir ?") == "(sessiz)"
    assert felt == [NOT_KNOWING]


def test_code_is_learned_from_its_tests():
    here = os.path.join(os.path.dirname(__file__), "..", "..", "research", "living_mind_2026_11_15")
    sys.path.insert(0, os.path.abspath(here))
    import t_code
    learned = t_code.run(0, 4, consequences=True, explore=True)["per_round"]
    control = t_code.run(0, 4, consequences=False, explore=True)["per_round"]
    assert learned[-1] == 1.0 and learned[0] < 1.0
    assert control[-1] < 1.0
