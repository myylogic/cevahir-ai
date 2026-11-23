# -*- coding: utf-8 -*-
"""Zihin, the whole mind: everything that was built is in use by default.

What is pinned here
-------------------
* ``VARIANTS["full"]`` is the speaking configuration with every studied mechanism on, and it is the program's
  default. ``speaking`` and ``Config()`` are as they were (every published row names its switches from them).
* A new mind of the program lives with all of it; a loaded mind is given every mechanism it was raised without,
  said in one line, and kept so (even when nothing was lived in the session).
* What stands against it: what the owner keeps off (``--kapat``, remembered with the mind until ``--ac`` names it;
  what cannot work without one of them goes with it and comes back with it), and a way of reading the mind's own
  trials put away (F57: that decision is the mind's; ``--ac`` overrules).
* The owner's change of a way that is on trial is an entry of the trials' record: after a moment is undone the
  configuration is rebuilt from that record and stays as the owner set it; trials that slept begin again.
* ``--variant speaking``: a loaded mind keeps its own settings and only ``--ac`` turns something on (as before).
* The web stays the owner's choice of each run: with everything on the mind decides by itself to look, and has
  nowhere to look until it is given a folder (``--kaynak``) or the web (``--ag``).
"""
from __future__ import annotations

import dataclasses
import io

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Config, Mind, deney
from zihin.__main__ import CHANNEL, NEEDS, NEWER, STUDIED, Session, VARIANTS, configure, main
from zihin.persist import load, save


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def config(variant: str, **switches):
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS[variant]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    for name, value in switches.items():
        setattr(cfg, name, value)
    return cfg


def raised(variant: str = "speaking", **switches) -> Session:
    torch.manual_seed(0)
    mind = Mind(config(variant, **switches))
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    s.watch("kedi ne yer ?", "kedi balık yer .")
    return s


def kept(tmp_path, variant: str = "speaking", name: str = "eski", **switches) -> str:
    path = str(tmp_path / name)
    save(raised(variant, **switches).mind, path)
    return path


def run(monkeypatch, capsys, args, lines="!cik\n") -> str:
    monkeypatch.setattr("sys.stdin", io.StringIO(lines))
    main(args + ["--yasamsiz", "--kim", "test"])
    return capsys.readouterr().out


def said(lines) -> str:
    return "\n".join(lines)


# ---------------------------------------------------------------------------------- the configuration itself
def test_full_is_speaking_with_everything_that_was_built():
    full, speaking = config("full"), config("speaking")
    for f in dataclasses.fields(Config):
        if f.name in STUDIED:
            assert getattr(full, f.name) is True and getattr(speaking, f.name) is False, f.name
        elif f.name == "deep_only":                      # F74: what only a reasoned moment reads
            assert full.deep_only == STUDIED and speaking.deep_only == (), f.name
        else:
            assert getattr(full, f.name) == getattr(speaking, f.name), f.name
    assert all(getattr(full, name) is True for name in NEWER + STUDIED)


def test_what_the_published_rows_are_named_from_is_as_it_was():
    plain = Config()
    assert not any(getattr(plain, name) for name in NEWER + STUDIED)
    assert all(getattr(config("speaking"), name) is True for name in NEWER)


def test_full_is_the_programs_default_and_what_it_needs_is_in_it():
    import argparse
    seen = {}
    monkey = argparse.ArgumentParser.add_argument

    def note(self, *names, **kw):
        if "--variant" in names:
            seen.update(kw)
        return monkey(self, *names, **kw)
    argparse.ArgumentParser.add_argument = note
    try:
        with pytest.raises(SystemExit):
            main(["--help"])
    finally:
        argparse.ArgumentParser.add_argument = monkey
    assert seen["default"] == "full" and "speaking" in seen["choices"]
    full = config("full")
    for need, wants in NEEDS:
        assert getattr(full, need) and all(getattr(full, k) for k in wants)


# ---------------------------------------------------------------------------------- a new mind of the program
def test_a_new_mind_lives_with_all_of_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CEVAHIR_HOME", str(tmp_path / "home"))
    out = run(monkeypatch, capsys, ["--yeni"], "!izle kedi ne yer ? || kedi balık yer .\n!deney\n!varsayim\n!cik\n")
    assert "deney kapalı" not in out and "varsayım kapalı" not in out   # both reports are those of a mind that has them
    assert "kullanımda" in out and "uykuda" not in out       # every way of reading is in use
    assert "tam yapı" not in out                             # nothing to say: it was raised so
    mind = load(str(tmp_path / "home" / "zihin" / "cevahir"), ledger=False)
    assert all(getattr(mind.cfg, name) is True for name in NEWER + STUDIED)
    assert mind.owner_off == []


def test_the_web_stays_the_owners_choice(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CEVAHIR_HOME", str(tmp_path / "home"))
    out = run(monkeypatch, capsys, ["--yeni", "--kaydetme"])
    assert "baktığı yer: yok" in out and "ağ:" not in out    # it decides to look and has nowhere to, until given


# ---------------------------------------------------------------------------------- a loaded mind
def test_a_loaded_mind_is_given_what_it_was_raised_without(tmp_path, monkeypatch, capsys):
    path = kept(tmp_path, situations=False, explore=False)
    monkeypatch.setenv("CEVAHIR_HOME", str(tmp_path / "home"))
    at = str(tmp_path / "yeni")
    out = run(monkeypatch, capsys, ["--load", path, "--kayit", at])
    line = [x for x in out.splitlines() if "tam yapı" in x][0]
    for name in ("situations", "explore") + STUDIED:
        assert name in line
    assert "zihin kaydedildi" in out                         # kept though nothing was lived
    mind = load(at, ledger=False)
    assert all(getattr(mind.cfg, name) is True for name in NEWER + STUDIED)
    assert load(path, ledger=False).cfg.known is False       # (the file it was loaded from is as it was)
    monkeypatch.setenv("CEVAHIR_HOME", str(tmp_path / "home"))
    again = run(monkeypatch, capsys, ["--load", at, "--kayit", at])
    assert "tam yapı" not in again and "zihin kaydedildi" not in again


def test_speaking_variant_keeps_a_loaded_minds_settings(tmp_path, monkeypatch, capsys):
    path = kept(tmp_path, situations=False, explore=False)
    monkeypatch.setenv("CEVAHIR_HOME", str(tmp_path / "home"))
    at = str(tmp_path / "yeni")
    out = run(monkeypatch, capsys, ["--load", path, "--kayit", at, "--variant", "speaking", "--ac", "situations,known"])
    assert "açıldı: situations" in out and "açıldı: known" in out and "tam yapı" not in out
    assert "onlarsız yaşar: explore" in out
    mind = load(at, ledger=False)
    assert mind.cfg.situations and mind.cfg.known and not mind.cfg.explore
    assert not any(getattr(mind.cfg, name) for name in STUDIED if name != "known")


def test_what_the_owner_keeps_off_stays_off_and_is_remembered(tmp_path, monkeypatch, capsys):
    path = kept(tmp_path)
    monkeypatch.setenv("CEVAHIR_HOME", str(tmp_path / "home"))
    at = str(tmp_path / "yeni")
    out = run(monkeypatch, capsys, ["--load", path, "--kayit", at, "--kapat", "faults,known"])
    assert "senin isteğinle kapalı: faults, known" in out
    mind = load(at, ledger=False)
    assert mind.owner_off == ["faults", "known"]
    off = {name for name in NEWER + STUDIED if not getattr(mind.cfg, name)}
    assert off == {"faults", "known", "find_out", "silent_claims", "relied_on", "split_failed",
                   "deduced_first", "own_turn", "subject_heard"}               # ... and what cannot work
                                                                                  # without faults
    out = run(monkeypatch, capsys, ["--load", at, "--kayit", at])     # remembered: the default does not turn it on
    mind = load(at, ledger=False)
    assert not mind.cfg.faults and not mind.cfg.known and not mind.cfg.find_out and "tam yapı" not in out
    out = run(monkeypatch, capsys, ["--load", at, "--kayit", at, "--ac", "faults"])
    mind = load(at, ledger=False)
    assert mind.owner_off == ["known"]                       # named again: on, and what needed it comes back
    assert mind.cfg.faults and mind.cfg.find_out and mind.cfg.silent_claims and mind.cfg.relied_on
    assert not mind.cfg.known


def test_what_was_asked_for_keeps_what_it_needs():
    s = raised("full")
    lines = []
    configure(s.mind, "full", on=["find_out"], off=["faults", "look"], say=lines.append)
    assert s.mind.cfg.find_out and s.mind.cfg.faults and s.mind.cfg.look and s.mind.owner_off == []
    assert "faults kapatılmadı" in said(lines) and "look kapatılmadı" in said(lines)


def test_an_unknown_name_is_said_and_changes_nothing():
    s = raised("full")
    lines = []
    assert configure(s.mind, "full", on=["yok"], off=["olmayan"], say=lines.append) is False
    assert "bilinmeyen mekanizma 'yok'" in said(lines) and "bilinmeyen mekanizma 'olmayan'" in said(lines)
    assert s.mind.owner_off == [] and all(getattr(s.mind.cfg, name) for name in NEWER + STUDIED)


def test_the_clock_of_the_program_takes_thinking_for_a_session(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CEVAHIR_HOME", str(tmp_path / "home"))
    out = run(monkeypatch, capsys, ["--yeni", "--kaydetme", "--saatli"])
    assert "--saatli" in out and "programın saati" in out


# ---------------------------------------------------------------------------------- the mind's own trials stand
def test_a_way_its_own_trials_put_away_is_not_turned_on_again():
    s = raised("full")
    mind = s.mind
    mind.trial_base = {way: True for way in deney.WAYS}      # its trials began with everything in use ...
    mind.cfg.known = False                                   # ... and came to put one way away
    mind.trial_changes.append({"t": mind.t, "way": "known", "on": False, "better": 0.0, "worse": 2.0})
    assert deney.own_choice(mind, "known") is False and deney.own_choice(mind, "families") is None
    lines = []
    configure(mind, "full", loaded=True, say=lines.append)
    assert mind.cfg.known is False and "kendi denemeleriyle bıraktığı yollar kapalı kaldı: known" in said(lines)
    configure(mind, "full", on=["known"], loaded=True, say=lines.append)     # the owner overrules
    assert mind.cfg.known is True
    assert mind.trial_changes[-1] == {"t": mind.t, "way": "known", "on": True, "better": 0.0, "worse": 0.0,
                                      "owner": True}
    assert deney.own_choice(mind, "known") is None           # how it stands now was set from outside
    deney.sync(mind)                                         # (what an undone moment rebuilds it from)
    assert mind.cfg.known is True
    assert "sahibi açtı" in said(deney.report(mind))


def test_a_pair_put_away_counts_for_both_of_its_ways():
    s = raised("full")
    mind = s.mind
    mind.trial_base = {way: True for way in deney.WAYS}
    pair = (deney.PAIR, "families", "unseen_share")
    mind.cfg.families = mind.cfg.unseen_share = False
    mind.trial_changes.append({"t": mind.t, "way": pair, "on": False, "better": 0.0, "worse": 2.0})
    assert deney.own_choice(mind, "families") is False and deney.own_choice(mind, "unseen_share") is False
    configure(mind, "full", loaded=True, say=lambda line: None)
    assert not mind.cfg.families and not mind.cfg.unseen_share


def test_the_owners_change_restarts_the_trials_that_slept():
    s = raised("full", known=False, families=False)
    mind = s.mind
    mind.trial_base = {way: deney.in_use(mind, way) for way in deney.WAYS}
    deney._settle(mind, "known", True)                       # one case in which the sleeping way would have done better
    deney._settle(mind, "families", False)
    assert deney.standing(mind, "known") == (1.0, 0.0) and deney.standing(mind, "families") == (0.0, 1.0)
    deney.set_from_outside(mind, {"families": True})
    assert mind.cfg.families is True and mind.trial_base["families"] is False   # the base is what the trials began with
    assert deney.standing(mind, "known") == (0.0, 0.0)       # it slept against a configuration that is no more
    assert deney.standing(mind, "families") == (0.0, 0.0)    # in use now: asked anew whether the mind does better
    assert deney.record(mind, "known") == (1.0, 0.0)         # (the evidence stays in its tie)
    deney.sync(mind)
    assert mind.cfg.families is True and mind.cfg.known is False


def test_before_the_trials_began_nothing_is_recorded():
    s = raised("speaking")
    mind = s.mind
    assert mind.trial_base == {}
    deney.set_from_outside(mind, {"known": True})
    assert mind.cfg.known is True and mind.trial_changes == [] and mind.trial_mark == {}


def test_a_full_mind_answers_and_its_trials_begin_with_everything_in_use():
    s = raised("full")
    for x, y in (("köpek", "et"), ("kuş", "tohum"), ("at", "ot")):
        s.watch("%s ne yer ?" % x, "%s %s yer ." % (x, y))
    assert s.reply("kedi ne yer ?", limit=20) == "kedi balık yer"
    s.close_turn()
    s.watch("kedi ne yer ?", "kedi balık yer .")              # what came is shown: the trial of that moment
    assert s.mind.trial_base == {way: True for way in deney.WAYS}
    assert s.mind.trial_changes == []
    s.company(None)
    s.listen("serçe kanatlı sınıfındandır .", IMITATED)       # ... and it goes on living as any mind
    s.end_of_turn(IMITATED, tone=".")
    s.mind.step([])
    assert Session(s.mind.clone()).reply("köpek ne yer ?", limit=20) == "köpek et yer"
