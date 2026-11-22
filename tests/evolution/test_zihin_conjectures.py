# -*- coding: utf-8 -*-
"""Zihin, F59 Conjectures: what a kind of link has been found to carry, the mind carries further by itself, in its
own time, and keeps what it comes to apart from what it was told.

What is pinned here
-------------------
* The switch is off by default and is a studied mechanism (``--ac conjectures``); while it is off nothing of it
  exists in the mind.
* A link: a thing told to go on with an item that is itself told about ("x y <tail>"). A candidate: a statement
  told of that item, with the thing in its place. Where the candidate was told of the thing too the link carried it;
  where the thing was told it with one item different it did not. A candidate nobody told, whose kind of link has
  been found to carry by more than one experience, is a conjecture: the mind thinks it (imagined, as every thought
  of its inner voice) and keeps it with its source.
* The conjectures are made in another field than the one the evidence came from: the things of a field of which
  nothing was told but their classes get their classes' statements, through the same word for the link.
* Nothing is conjectured without evidence, on one experience, where the evidence is against, or through a kind of
  link that was found not to carry (the mind is not told which word makes a class).
* In its own time: nothing is conjectured while it listens; a silent moment in which it decides to think a thing
  over, and the session letting that happen, make it.
* Settled by what it is told later: held, or failed; a statement that failed of two things of a class is not
  conjectured of a third, the class's other statement is.
* What it thought is imagined, not told: read as told it is not there; asked, the mind answers from it (with F53,
  F54 and F55 it answers what was told of nothing in that field), and without the switch it says nothing.
* The record and the conjectures are saved with the mind; ``!varsayim`` reports them.
"""
from __future__ import annotations

import os

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Config, Mind, persist, varsayim
from zihin.__main__ import CHANNEL, STUDIED, Session, VARIANTS

ANIMALS = ["aslan", "kedi", "yılan", "kertenkele", "serçe", "kartal", "kurt", "timsah"]
CLASS = {"aslan": "memeli", "kedi": "memeli", "kurt": "memeli", "yılan": "sürüngen", "kertenkele": "sürüngen",
         "timsah": "sürüngen", "serçe": "kanatlı", "kartal": "kanatlı"}
BEARS = {"memeli": "yavru", "sürüngen": "yumurta", "kanatlı": "civciv"}
WEARS = {"memeli": "kıl", "sürüngen": "zırh", "kanatlı": "telek"}
NUMBERS = {"altı": "çift", "sekiz": "çift", "yedi": "tek", "dokuz": "tek"}
SPLITS = {"çift": "ikiye", "tek": "kendine"}
EATS = {"aslan": "et", "kedi": "balık", "yılan": "tohum", "kertenkele": "ot", "serçe": "et", "kartal": "balık",
        "kurt": "ot", "timsah": "tohum"}


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def born(**switches) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    for name, value in switches.items():
        setattr(cfg, name, value)
    torch.manual_seed(0)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    return Session(mind)


def tell(s: Session, sentence: str, silent: int = 1) -> None:
    s.company(None)
    s.listen(sentence, IMITATED)
    s.end_of_turn(IMITATED, tone=".")
    for _ in range(silent):
        s.silent_moment(say=lambda *_: None)


def rest(s: Session, most: int = 200) -> None:
    for _ in range(most):
        s.silent_moment(say=lambda *_: None)
        if not s.mind.pondering:
            break


def raised(told=("aslan", "yılan", "serçe"), otherwise=False, numbers_link="sınıfındandır", **switches) -> Session:
    """The classes' statements; of every animal its class, what it eats and whom it chases; of every number its
    class; of the animals in ``told`` also what their class bears and wears (or, ``otherwise``, another value)."""
    s = born(**dict({"conjectures": True}, **switches))
    for k in BEARS:
        tell(s, "%s %s doğurur ." % (k, BEARS[k]))
        tell(s, "%s %s örtünür ." % (k, WEARS[k]))
    for k, v in SPLITS.items():
        tell(s, "%s %s bölünür ." % (k, v))
    for i, x in enumerate(ANIMALS):
        tell(s, "%s %s sınıfındandır ." % (x, CLASS[x]))
        tell(s, "%s %s yer ." % (x, EATS[x]))
        tell(s, "%s %s kovalar ." % (x, ANIMALS[(i + 3) % len(ANIMALS)]))
    for n, k in NUMBERS.items():
        tell(s, "%s %s %s ." % (n, k, numbers_link))
    for x in told:
        other = next(v for v in BEARS.values() if v != BEARS[CLASS[x]])
        tell(s, "%s %s doğurur ." % (x, other if otherwise else BEARS[CLASS[x]]))
        other = next(v for v in WEARS.values() if v != WEARS[CLASS[x]])
        tell(s, "%s %s örtünür ." % (x, other if otherwise else WEARS[CLASS[x]]))
    rest(s)
    return s


def made(s: Session) -> set:
    return {varsayim.shown(s.mind, e["statement"]) for e in s.mind.conjectures.values()}


def holds() -> set:
    out = {"%s %s doğurur" % (x, BEARS[k]) for x, k in CLASS.items()} | {"%s %s örtünür" % (x, WEARS[k])
                                                                        for x, k in CLASS.items()}
    return out | {"%s %s bölünür" % (n, SPLITS[k]) for n, k in NUMBERS.items()}


def test_off_by_default_and_studied():
    assert Config().conjectures is False and "conjectures" in STUDIED


def test_while_off_nothing_of_it_exists():
    s = raised(conjectures=False)
    m = s.mind
    assert not m.conjectures and not m.pondering and not m.linked and not m.waiting and m.supposing is None
    assert not m.ch.has(varsayim.CHANNEL)


def test_what_a_link_was_found_to_carry_is_carried_further():
    s = raised()
    mine = made(s)
    told = {"%s %s doğurur" % (x, BEARS[CLASS[x]]) for x in ("aslan", "yılan", "serçe")} | {
        "%s %s örtünür" % (x, WEARS[CLASS[x]]) for x in ("aslan", "yılan", "serçe")}
    assert mine <= holds()                                 # nothing that does not hold
    assert holds() - told <= mine                          # and everything that holds and nobody told:
    assert "kedi yavru doğurur" in mine and "timsah zırh örtünür" in mine      # the other animals,
    assert {"altı ikiye bölünür", "yedi kendine bölünür"} <= mine              # and the numbers: another field
    entry = next(e for e in s.mind.conjectures.values() if varsayim.shown(s.mind, e["statement"]) == "altı ikiye bölünür")
    assert varsayim.shown(s.mind, entry["from"]) == "çift ikiye bölünür"       # kept with its source
    assert varsayim.shown(s.mind, (entry["x"], entry["y"]) + entry["tail"]) == "altı çift sınıfındandır"
    assert entry["state"] == "open" and entry["for"] > entry["against"] + s.mind.cfg.k


def test_nothing_without_evidence_on_one_experience_or_against_it():
    assert made(raised(told=())) == set()                  # no statement of any thing: nothing shows a link carries
    one = born(conjectures=True)
    for sentence in ("memeli yavru doğurur .", "aslan memeli sınıfındandır .", "kedi memeli sınıfındandır .",
                     "aslan yavru doğurur ."):
        tell(one, sentence)
    rest(one)
    assert made(one) == set()                              # one experience is not knowledge
    assert made(raised(otherwise=True)) == set()           # told otherwise: the link was found not to carry


def test_a_kind_of_link_that_does_not_carry_is_not_used():
    s = raised()
    m = s.mind
    chases = (m.unit_of(CHANNEL, "kovalar"),)
    n_for, n_against = varsayim.votes(m)[("v", chases)]
    assert n_against > n_for + m.cfg.k                     # "x y kovalar": what is told of y is not told of x
    assert not any(" kovalar" in c or " yer" in c or "sınıfındandır" in c for c in made(s))
    other = raised(numbers_link="kümesindendir")           # a word for the link the mind has no record of
    assert not any("bölünür" in c for c in made(other))    # is not believed because another word carried
    assert "kedi yavru doğurur" in made(other)


def test_in_its_own_time():
    s = born(conjectures=True)
    m = s.mind
    sentences = ["memeli yavru doğurur .", "memeli kıl örtünür .", "sürüngen yumurta doğurur .",
                 "aslan memeli sınıfındandır .", "aslan yavru doğurur .", "aslan kıl örtünür .",
                 "kedi memeli sınıfındandır ."]
    for sentence in sentences:                             # told, a pause after each; nothing lets its decisions
        s.company(None)                                    # happen
        s.listen(sentence, IMITATED)
        assert m.supposing is None                         # while it listens it decides no such thing
        s.end_of_turn(IMITATED, tone=".")
        m.step([])
    assert m.supposing is not None                         # in a silent moment it decided to think a thing over,
    assert not m.conjectures and m.pondering               # and nothing was conjectured: no one let it happen
    before = m.t
    rest(s)                                                # silent moments of a session: its decisions happen
    assert made(s) == {"kedi yavru doğurur", "kedi kıl örtünür"}
    assert m.t > before and not m.pondering


def test_settled_by_what_it_is_told_later():
    s = raised()
    m = s.mind
    assert varsayim.worth(m) == 0.5                        # no conjecture of its own has met anything yet
    tell(s, "kedi yavru doğurur .")                        # as it conjectured
    tell(s, "kurt zırh örtünür .")                         # otherwise than it conjectured
    tell(s, "kedi zırh örtünür .")                         # and of a second thing of the class
    rest(s)
    state = {varsayim.shown(m, e["statement"]): e["state"] for e in m.conjectures.values()}
    assert state["kedi yavru doğurur"] == "held"
    assert state["kurt kıl örtünür"] == "failed" and state["kedi kıl örtünür"] == "failed"
    assert state["kurt yavru doğurur"] == "open"
    tell(s, "vaşak memeli sınıfındandır .")                # a thing of that class it never heard of
    rest(s)
    mine = made(s)
    assert "vaşak yavru doğurur" in mine                   # the class's other statement
    assert "vaşak kıl örtünür" not in mine                 # not the one that failed of two of its things


def test_what_it_thought_is_imagined_and_it_answers_from_it():
    base = dict(faults=True, families=True, lived_apart=True, asked_place=True)
    answers = {}
    for on in (False, True):
        s = raised(conjectures=on, **base)
        for _ in range(2):
            for x in ("aslan", "yılan", "serçe"):
                s.watch("%s ne doğurur ?" % x, "%s ." % BEARS[CLASS[x]])
                s.watch("%s ne örtünür ?" % x, "%s ." % WEARS[CLASS[x]])
        s.close_turn()
        rest(s)
        answers[on] = [Session(s.mind.clone()).reply(q, limit=30) for q in (
            "altı ne bölünür ?", "yedi ne bölünür ?", "kedi ne doğurur ?", "timsah ne örtünür ?")]
        if on:
            m = s.mind
            six = m.unit_of(CHANNEL, "altı")
            thought = {varsayim.shown(m, t) for t in varsayim.statements(m, six, varsayim.imagined(m))}
            known = {varsayim.shown(m, t) for t in varsayim.statements(m, six, varsayim.told(m))}
            assert "altı ikiye bölünür" in thought and "altı ikiye bölünür" not in known
    right = ["ikiye", "kendine", "yavru", "zırh"]
    assert not any(a == b for a, b in zip(answers[False], right))   # told of nothing there: nothing right to say
    assert answers[False][:2] == ["(sessiz)", "(sessiz)"]
    assert answers[True] == right


def test_saved_with_the_mind_and_reported(tmp_path):
    s = raised()
    m = s.mind
    path = os.path.join(str(tmp_path), "mind.zihin")
    persist.save(m, path)
    loaded = persist.load(path, ledger=False)
    assert loaded.conjectures == m.conjectures and loaded.linked == m.linked and loaded.waiting == m.waiting
    assert varsayim.votes(loaded) == varsayim.votes(m)
    text = "\n".join(varsayim.report(loaded))
    assert "sınanmadı" in text and "kaynak:" in text and "sınıfındandır" in text
    assert "henüz bir varsayımım yok" in "\n".join(varsayim.report(born(conjectures=True).mind))
