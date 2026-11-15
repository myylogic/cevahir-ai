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
* An interrupted !unut is resumed by the next one: the moments it already took back are skipped (a retry failed with
  KeyError before), and in the end the mind is exactly as after one uninterrupted !unut.
* "!unut" with something that is not a number says what it needs; a number that is not in the history says so.
* !yas and !boyut count the units and ties the mind holds now, so what !unut took back is no longer counted (the dead
  rows stay in the tables).
"""
from __future__ import annotations

import re

import pytest

from zihin import TOLD, Config, Mind
from zihin.__main__ import CHANNEL, READ, Entry, Session, _one_line

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
    entry = session.history[-1]
    number, text, moments = entry.number, entry.text, entry.moments
    assert text.startswith(("[okudu] hayvanlar.txt", "[öğretti] ders.tsv", "[izledi] Zürafa ne yer?"))
    assert moments == list(range(moments[0], session.mind.t + 1)) and len(moments) > 5
    assert session.ask(probe) != before                  # what was lived changed what it expects
    out = run(session, "!unut %d" % number, capsys)
    assert "geri alındı" in out and records(out) > 0
    assert session.ask(probe) == before                  # ... and taking it back changes it back
    assert [e.number for e in session.history] == [1]   # the typed line stays


def indices(S) -> tuple:
    return S.tie_index, S.out, S.inc, S.atom_index, S.tuple_index, S.frame_first, S.frame_of


def test_taking_back_a_read_equals_retracting_each_of_its_moments(tmp_path):
    (tmp_path / "hayvanlar.txt").write_text(ANIMALS, encoding="utf-8")
    one, other = taught(), taught()
    for session in (one, other):
        session.read_path(str(tmp_path / "hayvanlar.txt"))
        session.live("Kedi evcil bir hayvandır.", TOLD)  # later life keeps some of what the read built
    assert one.mind.S.digest() == other.mind.S.digest()
    number = one.history[-2].number
    alive = lambda S: int(S.t_alive[:S.n_ties].sum())
    held = alive(one.mind.S)
    assert "geri alındı" in one.forget(number)
    assert alive(one.mind.S) < held                      # ties the read built were removed: the indices changed
    for m in reversed(other.history[-2].moments):
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
    session.history.append(Entry(99, READ, "[okudu] boş.txt", []))   # an entry that lived no moment
    out = run(session, "!unut 99", capsys)
    assert "geri alınamaz" in out and "geri alındı" not in out


def test_history_numbers_are_never_given_twice():
    session = taught()
    for text in ("Bir.", "İki.", "Üç."):
        session.live(text, TOLD)
    session.forget(2)
    session.live("Dört.", TOLD)
    assert [e.number for e in session.history] == [1, 3, 4]
    assert "Dört." in session.forget(4) and [e.number for e in session.history] == [1, 3]


def test_an_interrupted_forget_is_resumed_and_takes_every_moment_back_once(tmp_path, monkeypatch):
    (tmp_path / "hayvanlar.txt").write_text(ANIMALS, encoding="utf-8")
    one, other = taught(), taught()
    for session in (one, other):
        session.read_path(str(tmp_path / "hayvanlar.txt"))
    number = one.history[-1].number
    real, taken = one.mind.retract, []

    def interrupted(m):
        if len(taken) == 3:
            raise KeyboardInterrupt                      # Ctrl+C between two moments of the retraction
        taken.append(m)
        return real(m)
    monkeypatch.setattr(one.mind, "retract", interrupted)
    with pytest.raises(KeyboardInterrupt):
        one.forget(number)
    assert taken == one.history[-1].moments[-3:][::-1]  # three moments, the latest first, are taken back ...
    assert [e.number for e in one.history] == [number]  # ... and the entry is still there
    monkeypatch.setattr(one.mind, "retract", real)
    out = one.forget(number)
    assert "geri alındı" in out and "3 anı önceki !unut geri almıştı" in out and one.history == []
    other.forget(other.history[-1].number)
    assert one.mind.S.digest() == other.mind.S.digest()
    assert indices(one.mind.S) == indices(other.mind.S)


def test_forget_needs_a_number(capsys):
    session = taught()
    for line in ("!unut x", "!unut", "!unut -1", "!unut 2.5"):
        out = run(session, line, capsys)
        assert "bir numara ister" in out and "hata" not in out
    assert "böyle bir numara yok" in run(session, "!unut 99", capsys)


def counted(session, capsys) -> dict:
    out = run(session, "!yas", capsys) + run(session, "!boyut", capsys)
    found = {name: int(re.search(pattern, out).group(1)) for name, pattern in (
        ("words", r"karşılaştığı kelime: (\d+)"), ("phrases", r"birleşik birim \(dizi\): (\d+)"),
        ("ties", r"bağ: (\d+)"), ("units", r"'units': (\d+)"), ("ties2", r"'ties': (\d+)"))}
    assert found["ties"] == found["ties2"]
    return found


def test_age_and_size_count_what_the_mind_holds_now(tmp_path, capsys):
    (tmp_path / "hayvanlar.txt").write_text(ANIMALS, encoding="utf-8")
    session = taught()
    before = counted(session, capsys)
    session.read_path(str(tmp_path / "hayvanlar.txt"))
    read = counted(session, capsys)
    assert all(read[name] > before[name] for name in before)
    assert "geri alındı" in session.forget(session.history[-1].number)
    assert counted(session, capsys) == before           # what was taken back is not counted any more
    assert session.mind.S.n_ties > before["ties"]       # (its rows are still in the tables, dead)
