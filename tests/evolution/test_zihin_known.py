# -*- coding: utf-8 -*-
"""Zihin, F41 Unsaid Known and the urge to go on: an answer that tells what is known of a thing never asked about.

What is pinned here
-------------------
* T-long's world with the facts told in any order (three facts of each animal read apart; six animals asked, the
  watched answers their three facts shuffled). Asked about an animal never asked: without F41 the mind says its name
  and stops; with it, it says the animal's three facts, each once and whole, and nothing else, and then stops.
* A world F41 was not developed on (T-tell: twelve things with two to four facts, three kinds of question lived by
  one mind). Asked to tell about things never in a dialogue, and about things asked other questions before, the mind
  says exactly their facts (two, three or four) and stops; asked where a thing is, it says that fact alone; and no
  sentence it says anywhere is not a fact of the thing asked about.
* What was only ever asked of a thing is not what is known of it: the word that followed the thing in a question that
  was answered is ``only_asked``; the word that followed it in a sentence that was read and left at that is not.
* The unsaid share of what is known of a thing (the urge) is 1 before anything is said of it and 0 when everything
  has been said in the present episode.
* The switch is off by default.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Config, Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS
from zihin.read import only_asked, unsaid_share

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "research", "living_mind_2026_11_15"))


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fresh(known: bool) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path, cfg.known = False, None, known
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    return Session(mind)


def read(s: Session, sentence: str) -> None:
    s.company(None)
    s.listen(sentence, IMITATED)
    s.end_of_turn(IMITATED, tone=s.tone(sentence))
    s.mind.step([])


def test_the_switch_is_off_by_default():
    assert Config().known is False and Config.changed().known is False
    assert VARIANTS["speaking"]().known is False


def test_what_is_known_of_a_thing_never_asked_is_told_whole_and_no_more():
    import t_long as L
    without = L.run(0, {}, varied=True)
    with_f41 = L.run(0, {"known": True}, varied=True)
    assert sum(r["right"] for r in without["rows"]) == 0
    for r in with_f41["rows"]:
        assert (r["right"], r["unfinished"], r["wrong"], r["sentences"]) == (3, 0, 0, 3), r["said"]
        assert sorted(p.strip() for p in r["said"].split(" . ")) == sorted(L.facts(r["animal"]))


def test_a_world_it_was_not_developed_on():
    import t_tell as T
    without = T.run(0, {})
    assert not any(r["whole"] for g in ("B", "C") for r in without["tell"][g])
    told = T.run(0, {"known": True})
    for g in ("B", "C"):
        for r in told["tell"][g]:
            assert r["whole"] and r["unfinished"] == 0 and r["wrong"] == 0, r["said"]
    assert sorted(r["facts"] for g in ("B", "C") for r in told["tell"][g]) == [2, 2, 3, 3, 3, 3, 4, 4]
    assert all(r["verdict"] == "right" for r in told["one"]["where"]), told["one"]["where"]
    for kind in ("where", "does"):                       # whatever is said is a fact of the thing asked about
        for r in told["one"][kind]:
            assert all(p == r["thing"] or p in T.facts(r["thing"]) for p in T.sentences(r["said"])), r["said"]


def test_what_was_only_asked_is_not_known():
    s = fresh(True)
    for _ in range(2):
        read(s, "kartal uçar .")
        s.watch("kartal nerede yaşar ?", "kartal dağda yaşar .")
    w = lambda x: s.mind.unit_of(CHANNEL, x)
    assert only_asked(s.mind, w("kartal"), w("nerede")) is True       # the question was answered
    assert only_asked(s.mind, w("kartal"), w("uçar")) is False        # the sentence was read and left at that
    assert only_asked(s.mind, w("kartal"), w("dağda")) is False       # never the beginning of a sequence


def test_the_urge_is_the_unsaid_share_of_what_is_known():
    s = fresh(True)
    for _ in range(2):
        read(s, "kartal uçar .")
        read(s, "kartal et yer .")
    mind = s.mind
    w = lambda x: mind.unit_of(CHANNEL, x)
    s.listen("kartal", IMITATED)
    assert unsaid_share(mind, w("kartal")) == pytest.approx(1.0)
    s.listen("uçar", IMITATED)                           # heard now, it is evidence too: uçar 1.5, et 1.0 unsaid
    assert unsaid_share(mind, w("kartal")) == pytest.approx(0.4)
    s.listen("kartal et", IMITATED)
    assert unsaid_share(mind, w("kartal")) == pytest.approx(0.0)


def test_the_program_turns_it_on_by_the_owners_choice(tmp_path, monkeypatch, capsys):
    from zihin.__main__ import STUDIED, main
    from zihin.persist import load, save
    assert "known" in STUDIED and "known" not in chat.NEWER
    path = str(tmp_path / "bir.zihin")
    s = fresh(False)
    read(s, "kartal uçar .")
    save(s.mind, path)
    monkeypatch.setenv("CEVAHIR_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("!cik\n"))
    main(["--load", path, "--kaydetme", "--yasamsiz", "--ac", "known", "--kim", "test"])
    assert "açıldı: known" in capsys.readouterr().out
    assert load(path, ledger=False).cfg.known is False          # the saved mind keeps its own settings
