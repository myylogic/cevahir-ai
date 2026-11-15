# -*- coding: utf-8 -*-
"""Zihin, where a session's experience ledger is written and when it is deleted.

What is pinned here
-------------------
* The ledger is written next to the kept mind (the home folder $CEVAHIR_HOME/zihin, or the folder of --kayit), never
  into the working folder, with or without --kaydetme.
* At a clean exit it is deleted once the mind has been saved without it, and at exit with --kaydetme. A mind loaded
  with its saved ledger lives on a copy of it, made in the same folder: the copy is deleted, the saved ledger kept.
* --defter-yok keeps no ledger at all.
"""
from __future__ import annotations

import io
import os

import pytest

import zihin.__main__ as program

LINES = "Kedi balık yer.\nKöpek kemik yer.\n!boyut\n"


def session_run(monkeypatch, tmp_path, argv, text=LINES + "!cik\n"):
    """Run the program on piped lines; returns the working folder and the ledger seen at each line (path, on disk)."""
    work = tmp_path / "calisma"
    work.mkdir(exist_ok=True)
    monkeypatch.setenv("CEVAHIR_HOME", str(tmp_path / "ev"))
    monkeypatch.chdir(work)
    monkeypatch.setattr("sys.stdin", io.StringIO(text))
    seen = []
    real = program._one_line

    def watching(session, line, *rest):
        path = program.ledger_of(session.mind)
        seen.append((path, bool(path) and os.path.exists(path)))
        return real(session, line, *rest)
    monkeypatch.setattr(program, "_one_line", watching)
    program.main(argv)
    monkeypatch.setattr(program, "_one_line", real)
    return work, seen


@pytest.mark.parametrize("argv", [["--yeni", "--kaydetme"], ["--yeni"]])
def test_the_ledger_lives_next_to_the_kept_mind_and_is_deleted_at_a_clean_exit(monkeypatch, tmp_path, argv):
    work, seen = session_run(monkeypatch, tmp_path, argv)
    ledger, written = seen[-1]
    home = tmp_path / "ev" / "zihin"
    assert written and os.path.dirname(ledger) == str(home)       # it was there during the session
    assert not os.path.exists(ledger)                             # ... and is gone after it
    assert os.listdir(work) == []                                 # nothing in the working folder
    assert [n for n in os.listdir(home) if n.startswith("defter-")] == []
    assert (home / "cevahir.zihin").exists() == ("--kaydetme" not in argv)


def test_with_kayit_the_ledger_is_next_to_that_mind(monkeypatch, tmp_path):
    work, seen = session_run(monkeypatch, tmp_path, ["--yeni", "--kayit", str(tmp_path / "baska" / "benim")])
    ledger, written = seen[-1]
    assert written and os.path.dirname(ledger) == str(tmp_path / "baska") and not os.path.exists(ledger)
    assert (tmp_path / "baska" / "benim.zihin").exists() and os.listdir(work) == []


def test_a_loaded_minds_copy_of_its_ledger_is_deleted_and_the_saved_ledger_kept(monkeypatch, tmp_path):
    saved = tmp_path / "kayitli"
    session_run(monkeypatch, tmp_path, ["--yeni", "--kaydetme"], LINES + "!kaydet %s\n!cik\n" % saved)
    assert os.path.exists(str(saved) + ".zihin.log")
    _, seen = session_run(monkeypatch, tmp_path, ["--load", str(saved), "--kaydetme"])
    ledger, written = seen[-1]
    assert written and os.path.dirname(ledger) == str(tmp_path / "ev" / "zihin") and not os.path.exists(ledger)
    assert os.path.exists(str(saved) + ".zihin.log")               # the saved ledger is never written or deleted
    assert sorted(os.listdir(tmp_path)) == ["calisma", "ev", "kayitli.zihin", "kayitli.zihin.json",
                                            "kayitli.zihin.log"]   # no copy left beside it


def test_without_a_ledger_nothing_is_written(monkeypatch, tmp_path):
    work, seen = session_run(monkeypatch, tmp_path, ["--yeni", "--kaydetme", "--defter-yok"])
    assert all(path is None for path, _ in seen)
    home = tmp_path / "ev" / "zihin"
    assert not home.exists() or [n for n in os.listdir(home) if n.startswith("defter-")] == []
    assert os.listdir(work) == []
