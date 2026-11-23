# -*- coding: utf-8 -*-
"""Zihin, the cost of the whole mind (section 30): readings shared where they are provably the same.

What is pinned here
-------------------
Nothing the mind does changes (the gates and the store digests of the probe worlds say so for whole lives); here
each shared reading is compared with the reading it replaces, made the long way, on a small mind:

* ``read.start_mass`` reads the two-item units that begin with a unit from a list kept as they are made
  (``read._start_ties``): the same as looking through every unit that holds it; undoing a moment drops the list.
* ``read.inherited`` (F37) reads, for any fixed words of a frame, a summary made once a run
  (``read._inherited_of``): the same as going through the item's own record again for each frame.
* ``read.resolve`` for NEXT and START reads what the member goes on with once, whatever the frame, and leaves a
  frame's fixed words out only where one of them is among them.
* ``read._place_words``: the words of a place's frame that can name a class, kept while the units live.
* Readings of the moment's memo are kept between moments too, under the store's count of writes: never inside a
  moment, never where evidence is recalled in present company, and a write makes them be read again.
"""
from __future__ import annotations

import random

import pytest
import torch

import test_zihin_own_race as R
from zihin import read
from zihin.evidence import conviction, frequency
from zihin.form import mass1, mv_view, parts_of
from zihin.read import INHERIT, NEXT, START
from zihin.store import ATOMIC, COMPOSED, EXPECT, PART, SLOTTED

BIRDS = ["serçe", "martı", "kartal", "leylek", "karga", "şahin"]
FISH = ["levrek", "sazan", "somon", "hamsi"]


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def raised(**switches):
    """A small mind with classes (F37), sequences that begin with a word (F26) and questions watched."""
    s = R.born("full", **switches)
    for _ in range(2):
        R.tell(s, "kuşlar uçar .")
        R.tell(s, "balıklar yüzer .")
        for x in BIRDS:
            R.tell(s, "%s bir kuştur ." % x)
            R.tell(s, "%s tohum yer ." % x)
        for x in FISH:
            R.tell(s, "%s bir balıktır ." % x)
            R.tell(s, "%s yosun yer ." % x)
        R.tell(s, "penguen bir kuştur .")
        R.tell(s, "penguen yüzer .")
        for x in BIRDS[:4]:
            s.watch("%s ne yapar ?" % x, "%s uçar ." % x)
        for x in FISH[:3]:
            s.watch("%s ne yapar ?" % x, "%s yüzer ." % x)
    s.close_turn()
    return s


def words(mind) -> list:
    """Every word (an atom of the text channel) the mind lived."""
    kinds, channels = mv_view(mind, "u_kind"), mv_view(mind, "u_channel")
    act = read.kind_masks(mind)[5]
    return [u for u in range(mind.S.n_units) if kinds[u] & ATOMIC and act[channels[u]]]


def start_mass_the_long_way(mind, unit: int) -> dict:
    """``read.start_mass`` as it was read before: every unit that holds ``unit`` is looked at."""
    S = mind.S
    kinds, arity = mv_view(mind, "u_kind"), mv_view(mind, "u_arity")
    src, lag = mv_view(mind, "t_src"), mv_view(mind, "t_lag")
    world, channel = read.kind_masks(mind)[3], mv_view(mind, "u_channel")
    began = {}
    for tie in S.inc.get((unit, PART), ()):
        c = src[tie]
        if lag[tie] != 0 or arity[c] != 2 or kinds[c] & SLOTTED or not kinds[c] & COMPOSED:
            continue
        v = parts_of(mind, c)[1]
        if v == unit or not world[channel[v]]:
            continue
        w_pos, _ = mass1(mind, tie)
        if w_pos > 0.0:
            began[v] = began.get(v, 0.0) + w_pos
    total = sum(began.values())
    return {v: (b, max(0.0, total - b)) for v, b in began.items()}


def inherited_the_long_way(mind, x: int, fixed=()):
    """``read.inherited`` as it was read before: x's own record is gone through for every frame."""
    since = getattr(mind, "episode_start", mind.t)
    order, values, e_deduced, lived, k = read._inheritance(mind, x, since)[:5]
    fixed_set = set(fixed)
    lived_mass = sum(p for v, (p, _) in lived.items() if v not in fixed_set)
    factor = 1.0 - conviction(lived_mass, k) if lived_mass > 0.0 else 1.0
    best, best_e = None, 0.5
    for i in order:
        v = values[i]
        if v in fixed_set or v in lived:
            continue
        if factor * e_deduced[i] > best_e:
            best, best_e = v, factor * e_deduced[i]
        break
    position = {v: i for i, v in enumerate(values)} if lived else {}
    for v, (p, n) in lived.items():
        if v in fixed_set:
            continue
        c = conviction(p + n, k)
        i = position.get(v)
        e = c * frequency(p, n) + (1.0 - c) * (e_deduced[i] if i is not None else 0.5)
        if e > best_e:
            best, best_e = v, e
    return best


def strongest_the_long_way(mind, target: int, filler: int, fixed):
    """``read.resolve`` for NEXT and START as it was read before: the continuations without the fixed words."""
    goes_on = read.start_continuations(mind, filler, fixed) if target == START else \
        read.continuations(mind, filler, fixed)
    found = max(goes_on, key=goes_on.get) if goes_on else None
    if found is not None and getattr(mind.cfg, "known", False):
        top = goes_on[found]
        if sum(1 for e in goes_on.values() if e >= top - 1e-12) > 1:
            found = None
    return found


def test_start_mass_reads_the_kept_list_as_the_long_way_does():
    s = raised()
    mind = s.mind
    some = 0
    for unit in words(mind):
        found = read.start_mass(mind, unit)
        assert found == start_mass_the_long_way(mind, unit)
        assert list(found) == list(start_mass_the_long_way(mind, unit))      # (the same order too)
        some += bool(found)
    assert some >= len(BIRDS)                                # (sequences began with the things)
    R.tell(s, "serçe ötmek ister .")                         # the lists grow: only what was added is looked at
    R.tell(s, "serçe yuva yapar .")
    for unit in words(mind):
        assert read.start_mass(mind, unit) == start_mass_the_long_way(mind, unit)


def test_undoing_a_moment_drops_what_was_kept_with_the_store():
    s = raised(keep_log=True)                                # (a moment can be undone where the ledger is kept)
    mind = s.mind
    for unit in words(mind):
        read.start_mass(mind, unit)
        read._forms_of(mind, unit)
    assert mind.S.__dict__.get("_starts") and mind.S.__dict__.get("_place_words")
    R.tell(s, "karga parlak şeyleri sever .")
    mind.undo_last()
    assert "_starts" not in mind.S.__dict__ and "_place_words" not in mind.S.__dict__
    for unit in words(mind):
        assert read.start_mass(mind, unit) == start_mass_the_long_way(mind, unit)


def test_inherited_reads_the_runs_summary_as_the_long_way_does():
    s = raised()
    mind = s.mind
    atoms = words(mind)
    rng = random.Random(0)
    named = 0
    for x in atoms:
        lived = read._inheritance(mind, x, getattr(mind, "episode_start", mind.t))[3]
        tries = [(), tuple(rng.sample(atoms, 2)), tuple(rng.sample(atoms, 3))]
        if lived:                                            # a fixed word that is one of its own lived values
            tries += [(next(iter(lived)),), tuple(lived)[:2]]
        for fixed in tries:
            mind.__dict__.pop("_between_memo", None)         # (the moment's memo holds the answer of each frame)
            found = read.inherited(mind, x, fixed)
            assert found == inherited_the_long_way(mind, x, fixed)
            named += found is not None
    assert named                                             # (something is deduced in this world)
    unit = mind.unit_of(R.CHANNEL, "penguen")
    mind.__dict__.pop("_between_memo", None)
    assert mind.symbol_of(read.inherited(mind, unit))[1] == "yüzer"   # its own record, over what birds do


def test_the_strongest_continuation_is_read_once_for_every_frame():
    s = raised()
    mind = s.mind
    atoms = words(mind)
    rng = random.Random(1)
    some = 0
    for filler in atoms:
        goes_on = read.continuations(mind, filler)
        tries = [(), tuple(rng.sample(atoms, 2))]
        if goes_on:
            tries += [(max(goes_on, key=goes_on.get),), tuple(goes_on)[:2]]   # the strongest itself is fixed
        for fixed in tries:
            for target in (NEXT, START):
                found = read.resolve(mind, target, filler, fixed)
                assert found == strongest_the_long_way(mind, target, filler, fixed)
                some += found is not None
    assert some


def test_the_words_of_a_places_frame_are_kept_as_they_are_read():
    s = raised()
    mind = s.mind
    kinds, channels = mv_view(mind, "u_kind"), mv_view(mind, "u_channel")
    act = read.kind_masks(mind)[5]
    places = 0
    for place in range(mind.S.n_units - 1):
        frame = place + 1
        long_way = None
        if kinds[frame] & SLOTTED:
            long_way = [q for q in parts_of(mind, frame) if q != place and kinds[q] & ATOMIC and act[channels[q]]]
        assert read._place_words(mind, place) == long_way
        assert read._place_words(mind, place) == long_way    # (and when it is read from what was kept)
        places += bool(long_way)
    assert places
    assert read._place_words(mind, mind.S.n_units - 1) is None   # (no unit after it yet: not kept)
    assert mind.S.n_units - 1 not in mind.S.__dict__["_place_words"]


def test_readings_are_kept_between_moments_only_while_nothing_is_written():
    s = raised()
    mind = s.mind
    unit = mind.unit_of(R.CHANNEL, "serçe")
    assert mind.S.cur is None and mind.__dict__.get("_read_memo") is None      # between moments
    memo, key = read._memo_key(mind, "own", unit, 1, None)
    assert memo is read._moment_memo(mind) and key[4] == mind.S.version
    first = read._unit_arrays(mind, unit, 1)
    assert read._unit_arrays(mind, unit, 1) is first         # the very reading, kept
    tie = mind.S.out[(unit, EXPECT, 1)][0]
    from zihin.events import SENSED
    from zihin.store import W_POS
    mind.S.add1(W_POS, SENSED, tie, 1.0)                     # a write (no moment was lived): it is read again
    again = read._unit_arrays(mind, unit, 1)
    assert again is not first and again[1][0] != first[1][0]
    mind.S.begin(mind.t + 1, mind.ch.sizes())                # inside a moment (its writers are at work): not kept
    try:
        assert read._memo_key(mind, "own", unit, 1, None) == (None, None)
    finally:
        mind.S.end()
    mind.__dict__["personal"] = ({}, {})                     # recalled in present company (F33): not kept
    assert read._memo_key(mind, "own", unit, 1, None) == (None, None)
