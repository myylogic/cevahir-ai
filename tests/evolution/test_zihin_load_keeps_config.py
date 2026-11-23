# -*- coding: utf-8 -*-
"""Zihin persist: a saved mind keeps the configuration it was raised with: ``persist.load`` switches nothing on.
Which mechanisms a mind then lives with is the program's to settle (``configure``): with --variant speaking a newer
mechanism is turned on only by an explicit choice of the owner (--ac); with the full configuration (the program's
default) a loaded mind is given every mechanism it was raised without (tests/evolution/test_zihin_full.py)."""
from __future__ import annotations

from zihin import Config, Mind
from zihin.__main__ import CHANNEL, Session, main
from zihin.persist import load, save


def raised(tmp_path, **flags) -> str:
    mind = Mind(Config.changed(support="members", bind_repeats=True, keep_log=False, **flags))
    mind.add_channel(CHANNEL, "act")
    Session(mind).watch("kedi ne yer ?", "kedi balık yer .")
    path = str(tmp_path / "eski")
    save(mind, path)
    return path


def test_load_switches_nothing_on(tmp_path):
    path = raised(tmp_path, backoff=False, resemblance=False, race=False, situations=False, explore=False,
                  starts=False, exclusive=False, exact_relations=False, analogy=False, corroborate=False,
                  paradigmatic=False, bind_relations=False, company=False, replies=False, purpose=False,
                  forms=False)
    mind = load(path, ledger=False)
    for name in ("backoff", "resemblance", "race", "situations", "explore", "starts", "exclusive", "exact_relations",
                 "analogy", "corroborate", "paradigmatic", "bind_relations", "company", "replies",
                 "purpose", "forms"):
        assert getattr(mind.cfg, name) is False


def test_the_owner_can_switch_on_a_newer_mechanism(tmp_path, monkeypatch, capsys):
    path = raised(tmp_path, situations=False, explore=False)
    monkeypatch.setenv("CEVAHIR_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("!cik\n"))
    main(["--load", path, "--kaydetme", "--yasamsiz", "--variant", "speaking", "--ac", "situations", "--kim", "test"])
    out = capsys.readouterr().out
    assert "açıldı: situations" in out
    line = [x for x in out.splitlines() if "onlarsız yaşar" in x][0]
    assert "explore" in line and "situations" not in line.split("açmak için")[0]   # named as one it lives without
    assert "tam yapı" not in out
