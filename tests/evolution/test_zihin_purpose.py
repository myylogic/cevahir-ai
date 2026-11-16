# -*- coding: utf-8 -*-
"""Zihin, F35 Generalised Consequences: what a path never taken would lead to is read from the paths its frames'
members lived; with F33, what a path led to with one person is an expectation of that relationship.

What is pinned here
-------------------
* Watched people being handed things answer rudely twice as often as politely ("al bu X" -> "ver bakayım ." /
  "teşekkür ederim ."); handed two things itself, its answers are received (polite well, rude badly). Handed a thing
  it was never handed, it answers politely with the switch and rudely (the habit) without it: what its own answers led
  to carries over to a situation it never met.
* Two people with opposite expectations (one receives the polite answer well, the other the banter): handed a new
  thing, the mind answers each the way their relationship has gone (with F33; the value they gave is noted with them).
* How a person received the mind's act is noted with that person, while the world sensed directly is not; the switch
  is off by default.
"""
from __future__ import annotations

import random

import pytest
import torch

import zihin.__main__ as chat
from zihin import Config, Mind
from zihin.__main__ import CHANNEL, VALUE, Session, VARIANTS


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fresh(purpose: bool, company: bool = True) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path, cfg.purpose, cfg.company = False, None, purpose, company
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    return Session(mind)


def watched(s: Session, rude: int, polite: int) -> None:
    rng = random.Random(0)
    lines = [(x, "ver bakayım .") for x in ("elma", "kalem", "kitap") for _ in range(rude)] + \
            [(x, "teşekkür ederim .") for x in ("elma", "kalem", "kitap") for _ in range(polite)]
    rng.shuffle(lines)
    for x, a in lines:
        s.watch("al bu %s" % x, a)


def first(s: Session, partner: str, text: str) -> str:
    twin = Session(s.mind.clone())
    twin.partner = partner
    said = twin.reply(text)
    return said.split()[0] if said != "(sessiz)" else said


def test_the_switch_is_off_by_default():
    assert Config().purpose is False and Config.changed().purpose is False


@pytest.mark.parametrize("purpose, answer", [(True, "teşekkür"), (False, "ver")])
def test_what_its_answers_led_to_carries_over(purpose, answer):
    s = fresh(purpose)
    watched(s, 2, 1)
    for _ in range(4):
        for x in ("elma", "kalem"):
            said = s.reply("al bu %s" % x)
            s.feedback(said.split()[:1] == ["teşekkür"])
    assert first(s, chat.SOMEONE, "al bu kitap") == answer


def test_each_relationship_has_its_own_expectations():
    s = fresh(True)
    watched(s, 2, 2)
    likes = {"ayşe": "teşekkür", "mehmet": "ver"}
    for _ in range(4):
        for person, liked in likes.items():
            s.partner = person
            for x in ("elma", "kalem"):
                said = s.reply("al bu %s" % x)
                s.feedback(said.split()[:1] == [liked])
    assert first(s, "ayşe", "al bu kitap") == "teşekkür"
    assert first(s, "mehmet", "al bu kitap") == "ver"


def test_how_a_person_received_an_act_is_noted_with_them():
    s = fresh(True)
    watched(s, 1, 1)
    s.partner = "ayşe"
    s.reply("al bu elma")
    s.feedback(True)
    mind = s.mind
    good = mind.unit_of(VALUE, "iyi")
    rows = [rows["ayşe"] for tie, rows in mind.personal[0].items()
            if int(mind.S.t_dst[tie]) == good and "ayşe" in rows]
    assert rows and all(row.sum() > 0.0 for row in rows)
