# -*- coding: utf-8 -*-
"""Zihin, F70 The Open Window (``Config.open_window``).

What is pinned here
-------------------
* Off by default, a studied mechanism, in use in the full configuration. It needs nothing: it bounds what
  frames of frames (F53) and F37 read, and a way the mind's own trials put away is not brought back by it.
* A context read for the first time is at depth 1; read again in the same moment it is the same; every later
  moment in which it is read again takes it one deeper (``aile.depth``).
* A family with more frames than a moment has steps (T) is read through T of them, T more at every depth, and
  whole once the window holds it: then the family's vote is the vote made without the window, number for number.
* Which frames come first: those whose word stands most where the context's word stands (F31), of equally near
  ones the one with the most members, then the one made first; they are read in the family's own order.
* A family the window holds whole is read exactly as without the window.
* F37 reads the places of an item that come to mind: T at depth 1; first the places that say what the item is
  (their frame names a class), then the surest memberships, of equally sure ones the one lived first; without the
  window all of them. (The first form took the surest first and lost the place that names the class: the world
  of deduction two steps away fell from 27 of 27 to 24.)
"""
from __future__ import annotations

import pytest
import torch

import test_zihin_own_race as R
from zihin import Config, aile, read
from zihin.__main__ import NEEDS, STUDIED, VARIANTS
import zihin.__main__ as chat

VERBS = ["yer", "içer", "sever", "giyer", "okur", "çizer", "taşır", "sayar", "arar"]
THINGS = ["inek", "koyun", "keçi", "at"]


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def raised(kinds: int = len(VERBS), **switches):
    """Nine kinds of question (frames that differ in one word), each watched of two to four things."""
    s = R.born("full", **switches)
    for _ in range(2):
        for i, verb in enumerate(VERBS[:kinds]):
            for x in THINGS[:2 + (i % 3)]:
                s.watch("%s ne %s ?" % (x, verb), "%s v%d_%s %s ." % (x, i, x, verb))
    s.close_turn()
    return s


def asked(s, question: str = "deve ne yer ?"):
    """The mind listens to a question about a thing it never met; returns (the context, its family by the kind
    of question: (the filler's positions, the varying word's, the frames))."""
    s.listen(question, R.IMITATED)
    s.end_of_turn(R.IMITATED, tone="?")
    unit = s.mind.context
    family = [f for f in aile.families(s.mind, unit) if f[0] == (0,)][0]
    return unit, family


def test_off_by_default_studied_and_needs_nothing():
    assert Config().open_window is False and "open_window" in STUDIED
    assert all(name != "open_window" for name, _ in NEEDS)
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        assert VARIANTS["full"]().open_window is True and VARIANTS["speaking"]().open_window is False
    finally:
        chat._log_path = made


def test_a_context_is_one_deeper_every_later_moment_it_is_read_in():
    s = raised(3)
    mind = s.mind
    unit, _ = asked(s)
    assert aile.depth(mind, unit) == 1 and aile.depth(mind, unit) == 1     # the same moment: the same depth
    mind.step([])
    assert aile.depth(mind, unit) == 2 and aile.depth(mind, unit) == 2
    mind.step([])
    mind.step([])                                            # (a moment in which it was not read does not count)
    assert aile.depth(mind, unit) == 3
    assert aile.depth(mind, unit + 1) == 1                   # another context: its own depth


def test_a_large_family_is_read_t_frames_a_moment_and_whole_in_time():
    s = raised()
    mind = s.mind
    unit, (first, second, frames) = asked(s)
    T = mind.cfg.T
    assert len(frames) == len(VERBS) - 1 > T                 # the eight other kinds of question
    word = read.parts_of(mind, unit)[second[0]]
    window = aile.in_window(mind, unit, word, frames)
    assert len(window) == T and set(window) < set(frames)
    assert window == [f for f in frames if f in window]      # read in the family's own order
    sizes = mind.S.abstract_sizes()

    def members(frame):
        return max(int(sizes[p]) for p in read.parts_of(mind, frame) if read.mv_view(mind, "u_kind")[p] & read.CATEGORY)
    left = [f for f in frames if f not in window]
    assert max(members(f) for f, _ in left) <= min(members(f) for f, _ in window)   # the most familiar first
    narrow = aile.votes(mind, unit)
    mind.step([])                                            # a later moment with the same context in mind
    assert aile.in_window(mind, unit, word, frames) == frames               # 2 T frames hold the whole family
    whole = aile.votes(mind, unit)
    without = raised(open_window=False)
    unit2, _ = asked(without)
    without.mind.step([])
    assert unit2 == unit and whole == aile.votes(without.mind, unit2)       # number for number
    assert narrow[2:5] == whole[2:5] and narrow[1].keys() == whole[1].keys()   # (the same family and kinds)
    assert narrow[0] < whole[0]                              # the conviction of the frames that were read


def test_the_frames_whose_word_stands_where_the_contexts_word_stands_come_first():
    s = raised()
    mind = s.mind
    for x in ("kedi", "kuzu", "tay"):                        # "giyer" and "yer" fill the same lived place
        for verb in ("yer", "giyer"):
            R.tell(s, "%s çok %s ." % (x, verb))
    unit, (first, second, frames) = asked(s)
    word = read.parts_of(mind, unit)[second[0]]
    giyer = mind.unit_of(R.CHANNEL, "giyer")
    near = read.stand_ins(mind, word, mind.episode_start)
    assert giyer in near                                     # (F31 holds evidence that the two stand alike)
    window = aile.in_window(mind, unit, word, frames)
    assert any(w == giyer for _, w in window)                # ... so its frame is in the window,
    plain = raised()
    unit2, (_, second2, frames2) = asked(plain)
    word2 = read.parts_of(plain.mind, unit2)[second2[0]]
    giyer2 = plain.mind.unit_of(R.CHANNEL, "giyer")
    assert all(w != giyer2 for _, w in aile.in_window(plain.mind, unit2, word2, frames2))   # where by familiarity
                                                                                            # alone it is not


def test_a_family_the_window_holds_is_read_as_without_it():
    for kinds in (3, 6):
        with_window, without = raised(kinds), raised(kinds, open_window=False)
        unit, (_, _, frames) = asked(with_window)
        unit2, _ = asked(without)
        assert len(frames) <= with_window.mind.cfg.T
        assert aile.votes(with_window.mind, unit) == aile.votes(without.mind, unit2)
        assert R.Session(with_window.mind.clone()).reply("deve ne içer ?", limit=12) == \
            R.Session(without.mind.clone()).reply("deve ne içer ?", limit=12)


def test_the_places_of_an_item_that_come_to_mind():
    s = raised()
    mind = s.mind
    x = mind.unit_of(R.CHANNEL, "inek")
    since = mind.t + 1
    every = read.memberships_before(mind, x, since)
    T = mind.cfg.T
    assert len(every) > T                                    # it filled the place of every kind of question
    places, budget = read._places_in_mind(mind, x, since)
    assert budget == T and len(places) == T
    sure = [(place, e) for place, e in every.items() if e > 0.5]
    order = sorted(range(len(sure)), key=lambda i: (-sure[i][1], i))[:T]
    assert places == [sure[i] for i in sorted(order)]        # the surest first, of equals the one lived first
    mind.cfg.open_window = False
    places, budget = read._places_in_mind(mind, x, since)
    assert budget == 0 and dict(places) == every             # without the window: all of them


def test_the_place_that_says_what_an_item_is_comes_to_mind_first():
    s = R.born("full")
    mind = s.mind
    for _ in range(2):
        R.tell(s, "kuşlar uçar .")
        for i in range(7):                                   # seven places the two things filled first
            for x in ("serçe", "martı"):
                R.tell(s, "%s bugün iş%d yapar ." % (x, i))
        for x in ("serçe", "martı"):                         # ... and, lived last, the one that names their class
            R.tell(s, "%s bir kuştur ." % x)
    x = mind.unit_of(R.CHANNEL, "serçe")
    since = mind.t + 1
    every = read.memberships_before(mind, x, since)
    naming = [place for place in every if read._names_a_class(mind, place, since)]
    assert len(every) > mind.cfg.T and naming               # more places than the window; some name a class
    places, budget = read._places_in_mind(mind, x, since)
    assert len(places) == budget == mind.cfg.T
    assert set(naming) <= {place for place, _ in places}     # the places that say what it is are in the window
    kept = [place for place, _ in places]
    assert kept == [place for place in every if place in kept]   # read in the order they were lived
