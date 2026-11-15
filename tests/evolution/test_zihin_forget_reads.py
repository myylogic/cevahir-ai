# -*- coding: utf-8 -*-
"""Zihin, taking back what was read, taught or watched (!unut).

What is pinned here
-------------------
* A document read with !oku (one entry per file), a file taught with !ogret and a dialogue watched with !izle are
  entries of the history with the moments they lived; "!unut <no>" retracts every one of those moments through the
  ledger, and a probe that the reading changed changes back exactly.
* Taking back an entry leaves the mind as retracting each of its moments one by one, the latest first, leaves it: the
  same tensors and the same indices (the indices are rebuilt once at the end, not once per moment).
* Nothing is reported as taken back with 0 records: an entry with no moments, or a mind that keeps no ledger, is
  refused with the reason, and the mind is unchanged.
* History numbers are never given twice.
"""
from __future__ import annotations

import re

import pytest

from zihin import TOLD, Config, Mind
from zihin.__main__ import CHANNEL, Session, _one_line

ANIMALS = ("Kedi evcil bir hayvandır. Kediler balık sever. Zürafa uzun boyunlu bir hayvandır. Zürafalar yaprak "
           "yer. Penguen soğuk yerlerde yaşar.")
TAUGHT = "Kedi ne yer?\tKedi balık yer.\nPenguen nerede yaşar?\tPenguen soğuk yerlerde yaşar.\n"


class Args:
    his = False


def taught() -> Session:
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True, situations=True, explore=True))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    for question, answer in (("Merhaba!", "Merhaba, nasılsın?"), ("Adın ne?", "Benim adım Cevahir."),
                             ("Köpek ne yer?", "Köpek kemik yer.")):
        session.watch(question, answer)
    return session


def run(session, line: str, capsys) -> str:
    capsys.readouterr()
    _one_line(session, line, False, Args(), lambda reason: None, 0)
    return capsys.readouterr().out


def records(out: str) -> int:
    found = re.search(r"; (\d+) kayıt\)", out)
    return int(found.group(1)) if found else 0


@pytest.mark.parametrize("command, probe", [
    ("!oku {doc}", "kedi evcil bir"),
    ("!ogret {tsv}", "penguen nerede"),
    ("!izle Zürafa ne yer? || Zürafa yaprak yer.", "zürafa ne"),
])
def test_what_was_read_taught_or_watched_is_taken_back_and_a_probe_changes_back(tmp_path, capsys, command, probe):
    (tmp_path / "hayvanlar.txt").write_text(ANIMALS, encoding="utf-8")
    (tmp_path / "ders.tsv").write_text(TAUGHT, encoding="utf-8")
    session = taught()
    session.live("Bugün hava güzel.", TOLD)
    before = session.ask(probe)
    run(session, command.format(doc=tmp_path / "hayvanlar.txt", tsv=tmp_path / "ders.tsv"), capsys)
    number, text, moments = session.history[-1]
    assert text.startswith(("[okudu] hayvanlar.txt", "[öğretti] ders.tsv", "[izledi] Zürafa ne yer?"))
    assert moments == list(range(moments[0], session.mind.t + 1)) and len(moments) > 5
    assert session.ask(probe) != before                  # what was lived changed what it expects
    out = run(session, "!unut %d" % number, capsys)
    assert "geri alındı" in out and records(out) > 0
    assert session.ask(probe) == before                  # ... and taking it back changes it back
    assert [n for n, _, _ in session.history] == [1]     # the typed line stays


def indices(S) -> tuple:
    return S.tie_index, S.out, S.inc, S.atom_index, S.tuple_index, S.frame_first, S.frame_of


def test_taking_back_a_read_equals_retracting_each_of_its_moments(tmp_path):
    (tmp_path / "hayvanlar.txt").write_text(ANIMALS, encoding="utf-8")
    one, other = taught(), taught()
    for session in (one, other):
        session.read_path(str(tmp_path / "hayvanlar.txt"))
        session.live("Kedi evcil bir hayvandır.", TOLD)  # later life keeps some of what the read built
    assert one.mind.S.digest() == other.mind.S.digest()
    number = one.history[-2][0]
    alive = lambda S: int(S.t_alive[:S.n_ties].sum())
    held = alive(one.mind.S)
    assert "geri alındı" in one.forget(number)
    assert alive(one.mind.S) < held                      # ties the read built were removed: the indices changed
    for m in reversed(other.history[-2][2]):
        other.mind.retract(m)
    assert one.mind.S.digest() == other.mind.S.digest()
    assert indices(one.mind.S) == indices(other.mind.S)
    assert "rebuild_indices" not in vars(one.mind.S)    # the substance's own method again


def test_nothing_is_reported_as_taken_back_with_no_records(tmp_path, capsys):
    (tmp_path / "hayvanlar.txt").write_text(ANIMALS, encoding="utf-8")
    session = taught()
    session.mind.S.log, session.mind.S.keep_log = {}, False      # a mind that keeps no ledger (--defter-yok)
    session.mind.cfg.keep_log = False
    session.read_path(str(tmp_path / "hayvanlar.txt"))
    digest = session.mind.S.digest()
    out = run(session, "!unut 1", capsys)
    assert "geri alınamaz" in out and "defter" in out and "geri alındı" not in out
    assert session.mind.S.digest() == digest and len(session.history) == 1
    session.history.append((99, "[okudu] boş.txt", []))         # an entry that lived no moment
    out = run(session, "!unut 99", capsys)
    assert "geri alınamaz" in out and "geri alındı" not in out


def test_history_numbers_are_never_given_twice():
    session = taught()
    for text in ("Bir.", "İki.", "Üç."):
        session.live(text, TOLD)
    session.forget(2)
    session.live("Dört.", TOLD)
    assert [n for n, _, _ in session.history] == [1, 3, 4]
    assert "Dört." in session.forget(4) and [n for n, _, _ in session.history] == [1, 3]
