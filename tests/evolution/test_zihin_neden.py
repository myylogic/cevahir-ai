# -*- coding: utf-8 -*-
"""Zihin, ``!neden``: where each word of the last answer came from, read at the moment it is decided.

What is pinned here
-------------------
* A recalled answer is attributed to lived evidence word by word ("adın ne ?" watched with "adım cevahir ." is
  answered "adım cevahir", both words "yaşanmış").
* A word said through a frame is attributed to the frame: "jump twice", never watched, is answered through the members
  of "X twice" (F34), and its words are "genelleme".
* Attribution only reads: the same question answered with and without it gives the same answer and the same core.
* ``!neden`` alone prints the record of the last answer; ``!neden <metin> => <kelime>`` still prints the white-box
  belief (the record does not shadow it).
"""
from __future__ import annotations

import re

import pytest
import torch

import zihin.__main__ as chat
from zihin import Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS, _one_line


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


class Args:
    yasamsiz = True
    defter_yok = True


def fresh() -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    return Session(mind)


def lived() -> Session:
    s = fresh()
    for _ in range(3):
        s.watch("adın ne ?", "adım cevahir .")
        s.watch("nasılsın ?", "iyiyim , teşekkür ederim .")
    return s


def commanded() -> Session:
    s = fresh()
    act = {"walk": "i_walk", "run": "i_run", "look": "i_look", "jump": "i_jump"}
    for _ in range(3):
        for x in ("walk", "run", "look"):
            s.watch(x, act[x] + " .")
            s.watch("%s twice" % x, "%s %s ." % (act[x], act[x]))
        s.watch("jump", "i_jump .")
    return s


def explained(s: Session, text: str):
    twin = Session(s.mind.clone())
    twin.explain = True
    said = twin.reply(text)
    return said, twin


def test_a_recalled_answer_is_lived_evidence_word_by_word():
    said, twin = explained(lived(), "adın ne ?")
    assert said == "adım cevahir"
    assert [r["word"] for r in twin.sources] == ["adım", "cevahir"]
    assert {r["source"] for r in twin.sources} == {"yaşanmış"}


def test_a_word_said_through_a_frame_is_attributed_to_it():
    said, twin = explained(commanded(), "jump twice")
    assert said == "i_jump i_jump"
    assert [r["source"] for r in twin.sources] == ["genelleme", "genelleme"]
    assert all("üye" in str(r["detail"]) for r in twin.sources)


def test_attribution_only_reads():
    s = lived()
    plain = Session(s.mind.clone())
    said_plain = plain.reply("adın ne ?")
    said, twin = explained(s, "adın ne ?")
    assert said == said_plain
    assert twin.mind.S.digest() == plain.mind.S.digest()


def test_the_command_prints_the_record_and_keeps_the_belief(capsys):
    s = lived()
    s.explain = True
    s.reply("adın ne ?")
    capsys.readouterr()
    _one_line(s, "!neden", False, Args(), lambda reason: None, 0)
    out = capsys.readouterr().out
    assert re.search(r"adım\s+yaşanmış", out) and "özet: yaşanmış 2" in out
    _one_line(s, "!neden adın ne ? => adım", False, Args(), lambda reason: None, 0)
    out = capsys.readouterr().out
    assert "kendi kanıtı" in out and "hata" not in out
