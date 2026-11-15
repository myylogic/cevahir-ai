# -*- coding: utf-8 -*-
"""Zihin, the English command aliases.

What is pinned here
-------------------
* Each English alias does what its Turkish command does, with the same output and the same mind afterwards: !read
  (!oku), !teach (!ogret), !watch (!izle), !good (!dogru), !bad (!yanlis), !sleep (!uyu), !think (!dusun),
  !introspect (!kendin). !quit (!cik) ends the session.
"""
from __future__ import annotations

import re

import pytest
import torch

from zihin import Config, Mind
from zihin.__main__ import CHANNEL, Session, _one_line

DIALOGUES = [("Merhaba!", "Merhaba, nasılsın?"), ("Kedi ne yer?", "Kedi balık yer."),
             ("Köpek ne yer?", "Köpek kemik yer."), ("Ankara nerede?", "Ankara Türkiye'nin başkentidir.")]


class Args:
    his = False


def answered() -> Session:
    torch.manual_seed(0)
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True, situations=True, explore=True))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    for question, answer in DIALOGUES:
        session.watch(question, answer)
    session.reply("Kedi ne yer?")                        # an answer, so !good / !bad have something to judge
    return session


def run(session, line: str, capsys):
    capsys.readouterr()
    result = _one_line(session, line, False, Args(), lambda reason: None, 0)
    return result, re.sub(r"\d+ sn \([\d.]+ ms/an\)", "", capsys.readouterr().out)   # the clock is not the mind


@pytest.mark.parametrize("turkish, english", [
    ("!oku {doc}", "!read {doc}"), ("!ogret {tsv}", "!teach {tsv}"),
    ("!izle Zürafa ne yer? || Zürafa yaprak yer.", "!watch Zürafa ne yer? || Zürafa yaprak yer."),
    ("!dogru", "!good"), ("!yanlis", "!bad"), ("!uyu 1", "!sleep 1"), ("!dusun kedi", "!think kedi"),
    ("!kendin", "!introspect")])
def test_an_english_alias_does_what_its_turkish_command_does(tmp_path, capsys, turkish, english):
    (tmp_path / "hayvanlar.txt").write_text("Kedi evcil bir hayvandır. Zürafa uzun boyunlu bir hayvandır.",
                                            encoding="utf-8")
    (tmp_path / "ders.tsv").write_text("Penguen nerede yaşar?\tPenguen soğuk yerlerde yaşar.\n", encoding="utf-8")
    done = []
    for line in (turkish, english):
        session = answered()
        result, out = run(session, line.format(doc=tmp_path / "hayvanlar.txt", tsv=tmp_path / "ders.tsv"), capsys)
        assert "bilinmeyen komut" not in out and "hata" not in out
        done.append((result, out, session.mind.t, session.mind.S.digest()))
    assert done[0] == done[1]


def test_quit_ends_the_session_like_cik(capsys):
    assert run(answered(), "!cik", capsys)[0] is None
    assert run(answered(), "!quit", capsys)[0] is None
