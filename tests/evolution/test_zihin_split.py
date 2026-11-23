# -*- coding: utf-8 -*-
"""Zihin, F67 The Category Is Split Where It Failed (``Config.split_failed``, with faults F49).

What is pinned here
-------------------
* Off by default, a studied mechanism, in use in the full configuration; it needs F49 (faults).
* F49 reads a frame's claim through a narrower place of its filler (the members that share another lived place
  with it) when the frame's own vote was found wanting: its claims held less often than it claimed, on the whole.
  ``hata.failed``: the way's own record holds more than one experience against it, whatever it also got right.
* A life in which the whole record hides the fault: what a thing eats follows its kind (read in a sentence), the
  watched dialogues are all of one kind, and the mind's ways are judged while it watches too (silent claims), so
  the frame's vote comes to the first question with a record of having held. Asked about things of every kind it
  is right for one kind and wrong for the others, calibrated on the whole. With F67 the category is split rounds
  sooner (in the measured world of the notes, where without it it is not split at all in 24 rounds).
* Where nothing ever failed the switch changes nothing.
"""
from __future__ import annotations

import pytest
import torch

import test_zihin_own_race as R
from zihin import Config, hata
from zihin.__main__ import CHANNEL, NEEDS, STUDIED, Session, VARIANTS
import zihin.__main__ as chat

THINGS = ["inek", "koyun", "keçi", "at", "eşek", "deve", "aslan", "kurt", "fok", "serçe", "kaz", "yunus", "tilki",
          "ayı", "balina", "kartal", "panter", "somon", "leylek", "geyik", "vaşak", "levrek", "martı", "manda",
          "çakal", "turna", "sazan", "öküz"]
FOODS = {"evcil": "ot", "yırtıcı": "et", "sucul": "balık", "kanatlı": "tohum"}
NAMES = list(FOODS)
MEMBERS = THINGS[:8]
KIND = {x: NAMES[0] if x in MEMBERS else NAMES[i % 4] for i, x in enumerate(THINGS)}
POOL, FRESH = THINGS[8:16], THINGS[16:24]


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def qa(x: str):
    return "%s ne yer ?" % x, "%s %s yer ." % (x, FOODS[KIND[x]])


def life(regular: bool = False, **switches):
    """The kind of every thing is read; eight things of one kind are watched being asked; then the things of the pool
    are asked in the teacher's loop (whatever the mind says it is shown the true dialogue). ``regular``: every
    thing eats what the watched six eat. Returns (the session, the loop's marks, the fresh things' marks)."""
    s = R.born("full", **switches)

    def truth(x: str) -> str:
        return "%s %s yer ." % (x, FOODS[NAMES[0]]) if regular else qa(x)[1]
    for _ in range(2):
        for x in THINGS:
            R.tell(s, "%s %s bir hayvandır ." % (x, KIND[x]))
        for x in MEMBERS:
            s.watch(qa(x)[0], truth(x))
    s.close_turn()
    marks = ""
    for x in POOL:
        said = s.reply(qa(x)[0], limit=20)
        s.close_turn()
        s.watch(qa(x)[0], truth(x))
        s.close_turn()
        marks += "R" if said + " ." == truth(x) else "W"
    fresh = "".join("R" if Session(s.mind.clone()).reply(qa(x)[0], limit=20) + " ." == truth(x) else "W"
                    for x in FRESH)
    return s, marks, fresh


def test_off_by_default_studied_and_needs_faults():
    assert Config().split_failed is False and "split_failed" in STUDIED
    assert ("split_failed", ("faults",)) in NEEDS
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        assert VARIANTS["full"]().split_failed is True and VARIANTS["speaking"]().split_failed is False
    finally:
        chat._log_path = made


def test_failed_is_not_wanting():
    """A way that claimed 0.6 and held 14 times of 23 is calibrated (not wanting) and has failed; one failure is
    the weight of no record (k), not yet a failure to split by; a way that never failed has not failed."""
    mind = R.born("full").mind
    mixed, once, clean = (hata.reading_ways(name) for name in ("karışık", "bir kez", "temiz"))
    for held in [True] * 14 + [False] * 9:
        hata.settle(mind, mixed, held, 0.6)
    hata.settle(mind, once, False, 0.6)
    for _ in range(5):
        hata.settle(mind, once, True, 0.6)
        hata.settle(mind, clean, True, 0.6)
    assert not hata.wanting(mind, mixed) and hata.failed(mind, mixed)
    assert not hata.failed(mind, once) and not hata.failed(mind, clean)


def test_a_vote_calibrated_on_the_whole_and_wrong_in_each_part_is_split_sooner():
    without, marks_without, fresh_without = life(split_failed=False)
    split, marks_split, fresh_split = life()
    assert marks_without == "RWWWRWWW"                       # right for the watched kind, wrong for the others,
    assert marks_split == "RWWWRWRR"                         # to the end of the loop; split after the sixth round
    assert fresh_split == "R" * len(FRESH)                   # every kind answered by its kind
    frame = [line for line in hata.report(without.mind) if "X ne yer" in line and "/" in line][0]
    assert "üyelerin aynı kelimesi" in frame                 # (the frame's own vote: it held and it failed)


def test_where_nothing_failed_the_switch_changes_nothing():
    _, marks_without, fresh_without = life(regular=True, split_failed=False)
    _, marks_split, fresh_split = life(regular=True)
    assert marks_without == marks_split == "R" * len(POOL)
    assert fresh_without == fresh_split == "R" * len(FRESH)
