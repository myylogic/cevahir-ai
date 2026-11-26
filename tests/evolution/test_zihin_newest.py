# -*- coding: utf-8 -*-
"""F76 (The Newest Holds): each tie's last confirmed moment, kept exactly, and the way of reading built on it.

What is pinned here
-------------------
* A mind that tracks its confirmations (``Config.newest``) writes a tie's last confirmed moment when its positive
  evidence grows, not when it fails; ``undo_last`` and ``retract`` give back the moment it had, bit for bit, and a
  saved mind keeps it. A mind that does not track them keeps nothing and its substance is as before.
* Where people's places change, Zihin with the newest says the place told last, and Cevahir with it; without it
  the counts.
"""
from __future__ import annotations

import pickle
import random

import pytest
import torch

from zihin import TOLD, Config, Event, Mind
from zihin.store import EXPECT


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _mind(**kw):
    torch.manual_seed(0)
    mind = Mind(Config.changed(**kw))
    mind.add_channel("text", "sense")
    return mind


def _say(mind, words):
    for word in words:
        mind.step([Event("text", word, TOLD)])
    mind.step([])


def test_a_tie_keeps_its_last_confirmed_moment_exactly():
    mind = _mind(newest=True)
    _say(mind, ["ali", "evde"])
    S = mind.S
    a, b = mind.unit_of("text", "ali"), mind.unit_of("text", "evde")
    tie = S.find_tie(a, b, EXPECT, 1)
    first = int(S.t_won[tie])
    assert first > 0
    _say(mind, ["ali", "okulda"])                      # "evde" expected after "ali" and missed: not a confirmation
    assert int(S.t_won[tie]) == first and int(S.t_last[tie]) > first
    mind.step([Event("text", "ali", TOLD)])
    before = S.digest()
    mind.step([Event("text", "evde", TOLD)])           # confirmed again
    assert int(S.t_won[tie]) > first
    mind.undo_last()
    assert int(S.t_won[tie]) == first and S.digest() == before   # given back bit for bit
    copy = pickle.loads(pickle.dumps(mind))
    assert torch.equal(copy.S.t_won[:S.n_ties], S.t_won[:S.n_ties]) and copy.S.digest() == S.digest()
    plain = _mind()
    _say(plain, ["ali", "evde"])
    assert plain.S.t_won is None                        # nothing kept without the switch


def _moving_world():
    """Six people told where they are and asked, three rounds, moving between them; then two others, only told:
    one three times in one place and once elsewhere, one once in a place and once elsewhere."""
    rng = random.Random(0)
    people = ["Veli", "Ayşe", "Can", "Ece", "Deniz", "Selin"]
    places = ["evde", "okulda", "parkta", "bahçede"]
    life = []
    for _ in range(3):
        for name in people:
            place = rng.choice(places)
            life += [("tell", "%s %s ." % (name, place)), ("watch", "%s nerede ?" % name, "%s %s ." % (name, place))]
    life += [("tell", "Mert evde .")] * 3 + [("tell", "Mert okulda ."), ("tell", "Zeynep parkta ."),
                                            ("tell", "Zeynep bahçede .")]
    return life


def _answers(newest):
    from living.cevahir import Cevahir
    c = Cevahir(variant="speaking", switches={"newest": newest})
    for item in _moving_world():
        c.tell(item[1]) if item[0] == "tell" else c.watch(item[1], item[2])
    c.session.close_turn()
    said = c.answers(["Mert nerede ?", "Zeynep nerede ?"], way="every", workers=1)
    return [c.session.words(s["zihin"]) for s in said], [c.session.words(s["cevahir"]) for s in said]


def test_where_places_change_the_newest_says_the_place_told_last():
    zihin, cevahir = _answers(True)
    assert zihin == [["mert", "okulda"], ["zeynep", "bahçede"]]
    assert cevahir == zihin                       # the whole of Cevahir says it too (its plan is not held back)
    assert _answers(False)[0] != [["mert", "okulda"], ["zeynep", "bahçede"]]   # the counts: the old place, or none
