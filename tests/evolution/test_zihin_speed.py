# -*- coding: utf-8 -*-
"""Zihin, speed that changes nothing: the lazy ranking of generalisations, the count of a category's extension and
the per-moment memos of two readers.

What is pinned here
-------------------
* ``Mind._rank_first`` (the focus filled in rounds from candidates ordered as arrays, only those whose turn comes
  built) gives, at every moment of a life, exactly what ``Mind._rank_all`` (every candidate built and sorted) gives:
  the same ranking with the same values, and for every ranked key the same entries of what the settling reads. It
  is used at most moments, and it steps back (None) rather than answer when it cannot know.
* ``Substance.abstract_sizes`` is ``len(inc[(unit, ABSTRACT)])`` of every unit: while living (brought up to date
  with the ties born since), after a moment is undone, after a moment is retracted, in a clone and in a mind saved
  and loaded.
* The memos of ``read._value_keys`` and ``read.bound_frames`` (open for a moment, while nothing is written) change
  no answer, and the substance is read fewer times with them.
"""
from __future__ import annotations

import collections

import pytest
import torch

import zihin.__main__ as chat
import zihin.read as read
from zihin import Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS
from zihin.persist import load, save
from zihin.store import ABSTRACT

PRIMITIVES = {"a": "ia", "b": "ib", "c": "ic", "d": "id"}


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fresh(window: int = 7, keep_log: bool = False, **switches) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = keep_log, None
    for name, value in switches.items():
        setattr(cfg, name, value)
    torch.manual_seed(0)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    mind.knob["F"] = window
    return Session(mind)


def the_life():
    """Every command of a, b and c (alone, twice, thrice, and two of them joined by "and"), and "d" alone; two
    epochs, the shorter commands first (the life of ``test_zihin_wholes``)."""
    parts = [((x,) + tail, (PRIMITIVES[x],) * n) for x in "abc" for tail, n in (((), 1), (("twice",), 2),
                                                                                   (("thrice",), 3))]
    train = parts + [(p + ("and",) + q, a + b) for p, a in parts for q, b in parts] + [(("d",), ("id",))]
    return [d for _ in range(2) for d in sorted(train, key=lambda d: len(d[0]))]


def live(s: Session, dialogues) -> None:
    for command, answer in dialogues:
        s.watch(" ".join(command), " ".join(answer) + " .")


def ask(s: Session, command: str) -> str:
    return Session(s.mind.clone()).reply(command, limit=200)


@pytest.mark.parametrize("switches", [{}, {"wholes": True, "told_apart": True}], ids=["plain", "wholes"])
def test_the_lazy_ranking_is_the_full_ranking_at_every_moment(monkeypatch, switches):
    seen = collections.Counter()

    def rank(self, base, parts, context, fresh_units, F):
        full = self._rank_all(dict(base), parts, context, set(fresh_units), F)
        lazy = self._rank_first(dict(base), parts, context, set(fresh_units), F)
        seen["moments"] += 1
        if lazy is None:
            seen["stepped back"] += 1
            return full
        seen["lazy"] += 1
        (r0, g0, v0, gr0), (r1, g1, v1, gr1) = full, lazy
        assert r1 == r0, "the ranking differs at moment %d" % self.t
        assert all(type(a) is type(b) for (_, a), (_, b) in zip(r0, r1))
        for key, _ in r0:
            assert g1.get(key) == g0.get(key) and (key in gr1) == (key in gr0)
            if key in gr0:
                assert gr1[key] == gr0[key]
            if isinstance(key, tuple):
                assert v1.get(key) == v0.get(key)
        return full

    monkeypatch.setattr(Mind, "_rank", rank)
    s = fresh(**switches)
    live(s, the_life())
    for command in ("d twice", "b thrice and d thrice", "c and a twice"):
        ask(s, command)
    assert seen["moments"] > 500
    assert seen["lazy"] > seen["moments"] // 2            # the lazy ranking is what a moment usually runs


def sizes_agree(S) -> int:
    count = S.abstract_sizes()
    assert count.size >= S.n_units
    for unit in range(S.n_units):
        assert int(count[unit]) == len(S.inc.get((unit, ABSTRACT), ())), unit
    assert not count[S.n_units:].any()
    return int(count[:S.n_units].sum())


def test_the_size_of_a_category_follows_the_index(tmp_path):
    s = fresh(keep_log=True)
    dialogues = the_life()
    live(s, dialogues[:40])
    S = s.mind.S
    first = sizes_agree(S)
    assert first > 0
    live(s, dialogues[40:80])                              # kept between moments: the ties born since are added
    assert sizes_agree(S) > first
    for k in range(3):                                     # a moment undone (only the last one can be): its ties
        s.mind.undo_last()                                 # leave the index, and the life goes on
        sizes_agree(S)
        live(s, dialogues[80 + 3 * k:83 + 3 * k])
        sizes_agree(S)
    grown = sizes_agree(S)
    born = [m for m, entry in S.log.items() if entry.ties1 > entry.ties0 and not entry.retracted]
    s.mind.retract(born[len(born) // 2])                   # a moment retracted (what it alone supported is removed)
    sizes_agree(S)
    twin = s.mind.clone()
    assert sizes_agree(twin.S) == sizes_agree(S)
    live(Session(twin), dialogues[90:95])                  # the clone lives on by itself
    sizes_agree(twin.S)
    sizes_agree(S)
    save(s.mind, str(tmp_path / "mind"))
    again = load(str(tmp_path / "mind"), ledger=False)
    assert sizes_agree(again.S) == sizes_agree(S)
    assert grown >= first


def test_the_memos_of_a_moment_change_no_answer(monkeypatch):
    s = fresh(wholes=True, told_apart=True)
    live(s, the_life())
    asked = ["d twice", "d thrice and a twice", "b thrice and d thrice", "c and d thrice", "a twice and b"]
    reads = collections.Counter()
    inner = read._value_keys_read

    def counted(*args, **kwargs):
        reads["now"] += 1
        return inner(*args, **kwargs)

    monkeypatch.setattr(read, "_value_keys_read", counted)
    with_memos = [ask(s, command) for command in asked]
    assert with_memos == ["id id", "id id id ia ia", "ib ib ib id id id", "ic id id id", "ia ia ib"]
    read_with = reads.pop("now")

    def unmemoised(reader):
        def call(mind, *args, **kwargs):
            memo = mind.__dict__.get("_read_memo")
            mind.__dict__["_read_memo"] = None
            try:
                return reader(mind, *args, **kwargs)
            finally:
                mind.__dict__["_read_memo"] = memo
        return call

    monkeypatch.setattr(read, "_value_keys", unmemoised(read._value_keys))
    monkeypatch.setattr(read, "bound_frames", unmemoised(read.bound_frames))
    assert [ask(s, command) for command in asked] == with_memos
    assert 0 < read_with < reads["now"]
