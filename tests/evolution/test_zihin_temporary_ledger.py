# -*- coding: utf-8 -*-
"""Zihin, the ledger of a mind built in code (tests, research scripts), and what git leaves out.

What is pinned here
-------------------
* A mind built with the chat's configuration outside the program (``VARIANTS["speaking"]()``) keeps its ledger in a
  temporary folder of its process, removed with the ledger when the process exits: nothing is written into the
  user's home ($CEVAHIR_HOME or ~). The program passes its own folder (the folder of the kept mind) explicitly.
* A research script that builds the configuration with ``_log_path`` replaced by a function of no arguments (as
  research/living_mind_2026_11_15/t_meaning.py does, to make no ledger file) still gets it.
* git ignores the session ledgers (defter-*.log) and the live copies a load makes of a saved ledger
  (*.zihin.log.live-*).
"""
from __future__ import annotations

import fnmatch
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import zihin.__main__ as program

REPO = Path(__file__).resolve().parents[2]

LIVES = textwrap.dedent("""
    import os
    from zihin import TOLD, Mind
    from zihin.__main__ import CHANNEL, VARIANTS, Session
    mind = Mind(VARIANTS["speaking"]())
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    for i in range(6):
        session.live("Kedi balık yer, köpek kemik yer.", TOLD)
    print(mind.S.log.path)
    print(os.path.getsize(mind.S.log.path))
""")


def test_a_mind_built_in_code_keeps_its_ledger_in_a_temporary_folder_removed_at_exit(tmp_path):
    env = dict(os.environ, CEVAHIR_HOME=str(tmp_path / "ev"), HOME=str(tmp_path / "kullanici"),
               USERPROFILE=str(tmp_path / "kullanici"), PYTHONPATH=str(REPO))
    done = subprocess.run([sys.executable, "-c", LIVES], cwd=str(tmp_path), env=env, capture_output=True, text=True,
                          timeout=300)
    assert done.returncode == 0, done.stderr
    ledger, size = done.stdout.split()
    assert int(size) > 0                                    # the ledger was written while the mind lived ...
    assert os.path.basename(os.path.dirname(ledger)).startswith("zihin-defter-")
    assert not os.path.exists(ledger) and not os.path.exists(os.path.dirname(ledger))   # ... and is gone with it
    assert sorted(os.listdir(tmp_path)) == []               # nothing in the home folders or the working folder


def test_the_program_passes_its_own_folder(tmp_path):
    config = program.VARIANTS["speaking"](str(tmp_path / "zihin"))
    assert os.path.dirname(config.log_path) == str(tmp_path / "zihin")
    assert os.path.basename(config.log_path).startswith("defter-")


def test_a_script_that_replaces_the_ledger_path_still_gets_the_configuration(monkeypatch):
    monkeypatch.setattr(program, "_log_path", lambda: None)
    config = program.VARIANTS["speaking"]()
    assert config.log_path is None and config.support == "members" and config.explore


def test_git_ignores_session_ledgers_and_live_copies_of_saved_ledgers():
    patterns = [line.strip() for line in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")]
    for name in ("defter-1763200000000-4242.log", "cevahir.zihin.log.live-4242-1763200000000000000",
                 "cevahir.zihin.log.live", "cevahir.zihin.tmp"):
        assert any(fnmatch.fnmatch(name, pattern) for pattern in patterns), name
