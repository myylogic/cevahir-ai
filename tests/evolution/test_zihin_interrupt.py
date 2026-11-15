# -*- coding: utf-8 -*-
"""Zihin, Ctrl+C during a long command, and the end of a session.

What is pinned here
-------------------
* Online (the terminal's default): Ctrl+C during a long !oku stops it at its next sentence (Session.check), the life
  thread ends, and only then is the mind saved: it loads, it holds exactly the moments lived, and no ledger
  (defter-*.log) or .tmp file is left. With --kaydetme the ledger is deleted after the thread has ended and is not
  made again.
* A life that cannot be stopped in time (a command that does not check) is not saved: the program says so, the mind
  saved before is untouched, and its ledger is deleted and not made again by the thread that goes on.
* Without a life thread (--yasamsiz, or lines from a pipe): the first Ctrl+C during a command asks it to stop between
  two moments; the session ends after it and the mind is saved whole. A second Ctrl+C does not wait: the mind may be
  cut inside a moment and is not saved; the mind saved before stays byte for byte.
* !unut, research and sleep stop at their next moment, sentence or dream when asked; a stopped !unut is resumed.
* A save stopped before its files were whole leaves the mind saved before and no file of its own; a save stopped
  while its files were being put in place is finished by the next load.
"""
from __future__ import annotations

import builtins
import io
import os
import signal
import threading
import time

import pytest

import zihin.__main__ as program
from zihin import TOLD, Config, Mind
from zihin.__main__ import CHANNEL, Interrupted, Session
from zihin.persist import export_portable, load, save

DOCUMENT = "\n".join("Penguen %d buzun üstünde yavaşça yürür." % i for i in range(60))


class Terminal(io.StringIO):
    """Standard input as a terminal (the program then lives online and reads lines with input())."""

    def isatty(self) -> bool:
        return True


def prepare(monkeypatch, tmp_path):
    home = tmp_path / "ev"
    work = tmp_path / "calisma"
    work.mkdir(exist_ok=True)
    monkeypatch.setenv("CEVAHIR_HOME", str(home))
    monkeypatch.chdir(work)
    (tmp_path / "uzun.txt").write_text(DOCUMENT, encoding="utf-8")
    return home / "zihin", tmp_path / "uzun.txt"


def online(monkeypatch, lines):
    """The program at a terminal: the given lines typed, then the end of input. Returns the lives it started."""
    monkeypatch.setattr("sys.stdin", Terminal())
    typed = iter(lines)

    def fake_input(prompt=""):
        try:
            return next(typed)
        except StopIteration:
            raise EOFError
    monkeypatch.setattr(builtins, "input", fake_input)
    lives = []

    class Recorded(program.Life):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            lives.append(self)
    monkeypatch.setattr(program, "Life", Recorded)
    return lives


def ctrl_c_at(monkeypatch, sentence: int, times: int = 1, wait_for_stop: bool = True):
    """Ctrl+C (SIGINT) while the mind hears its ``sentence``-th sentence; from the life thread Python runs the
    handler in the main thread. With ``wait_for_stop`` the hearing goes on once the session was asked to stop."""
    real, heard = Session.listen, []

    def listen(self, text, prov, speaker=None):
        real(self, text, prov, speaker)
        heard.append(text)
        if len(heard) == sentence:
            for _ in range(times):
                signal.raise_signal(signal.SIGINT)
            if wait_for_stop:
                assert self.stop.wait(10)
    monkeypatch.setattr(Session, "listen", listen)
    return heard


def files(folder, pattern):
    return sorted(p.name for p in folder.glob(pattern)) if folder.exists() else []


@pytest.mark.parametrize("kaydetme", [False, True])
def test_ctrl_c_during_a_long_read_online_stops_it_and_keeps_a_whole_mind(monkeypatch, tmp_path, capsys, kaydetme):
    folder, document = prepare(monkeypatch, tmp_path)
    lives = online(monkeypatch, ["!oku %s" % document])
    heard = ctrl_c_at(monkeypatch, 5)
    program.main(["--yeni", "--kim", "ben"] + (["--kaydetme"] if kaydetme else []))
    life = lives[0]
    assert not life.thread.is_alive()                    # its life ended before anything was saved
    out = capsys.readouterr().out
    assert "yarıda kesildi" in out and "durdurulamadı" not in out
    session = life.session
    entry = session.history[-1]
    assert entry.kind == "read" and len(heard) == 5 < 60
    assert entry.moments == list(range(entry.moments[0], entry.moments[-1] + 1))
    time.sleep(0.3)                                      # nothing is written after the end
    assert files(folder, "defter-*") == [] and files(folder, "*.tmp") == []
    if kaydetme:
        assert files(folder, "cevahir.zihin*") == [] and "kaydedildi" not in out
    else:
        assert "kaydedildi" in out
        kept = load(str(folder / "cevahir"), ledger=False)
        assert kept.t == session.mind.t and kept.S.cur is None
        assert kept.S.digest() == session.mind.S.digest()


def test_a_life_that_cannot_be_stopped_is_not_saved_and_leaves_nothing(monkeypatch, tmp_path, capsys):
    folder, document = prepare(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO("Kedi balık yer.\n"))
    program.main(["--yeni"])                             # a mind saved before
    before = (folder / "cevahir.zihin").read_bytes()
    capsys.readouterr()
    lives = online(monkeypatch, ["!oku %s" % document])
    monkeypatch.setattr(program.Life, "WAIT", 0.5)
    release, lived = threading.Event(), []

    def deaf_read(self, path, cap=0):                    # a command that never checks whether to stop
        for i in range(100000):
            self.mind.step([program.Event(CHANNEL, "kedi", TOLD)])
            lived.append(i)
            if i == 3:
                signal.raise_signal(signal.SIGINT)
            if release.is_set():
                break
            time.sleep(0.005)
        return "  okudum"
    monkeypatch.setattr(Session, "read_path", deaf_read)
    program.main(["--kim", "ben"])
    out = capsys.readouterr().out
    assert "durdurulamadı" in out and "kaydedildi" not in out
    life = lives[0]
    assert life.thread.is_alive()                        # it goes on living ...
    going = len(lived)
    deadline = time.time() + 10
    while len(lived) < going + 20 and time.time() < deadline:
        time.sleep(0.01)
    assert len(lived) >= going + 20                      # ... and writing its ledger, which goes nowhere now
    assert files(folder, "defter-*") == [] and files(folder, "*.tmp") == []
    assert (folder / "cevahir.zihin").read_bytes() == before        # the mind saved before is untouched
    release.set()
    life.thread.join(10)
    assert not life.thread.is_alive() and files(folder, "defter-*") == []


def test_without_a_life_ctrl_c_stops_the_command_between_moments_and_the_mind_is_kept(monkeypatch, tmp_path, capsys):
    folder, document = prepare(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO("!oku %s\nKöpek kemik yer.\n" % document))
    heard = ctrl_c_at(monkeypatch, 5, wait_for_stop=False)
    program.main(["--yeni"])
    out = capsys.readouterr().out
    assert "yarıda kesildi" in out and "kaydedildi" in out and len(heard) == 5
    kept = load(str(folder / "cevahir"), ledger=False)
    assert kept.S.cur is None and kept.unit_of(CHANNEL, "penguen") >= 0
    assert kept.unit_of(CHANNEL, "köpek") < 0            # the session ended with the stopped command
    assert files(folder, "defter-*") == [] and files(folder, "*.tmp") == []
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler   # Ctrl+C is as before after the session


def test_a_second_ctrl_c_does_not_wait_and_the_mind_saved_before_stays(monkeypatch, tmp_path, capsys):
    folder, document = prepare(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO("Kedi balık yer.\n"))
    program.main(["--yeni"])
    before = (folder / "cevahir.zihin").read_bytes(), (folder / "cevahir.zihin.json").read_bytes()
    capsys.readouterr()
    monkeypatch.setattr("sys.stdin", io.StringIO("!oku %s\n" % document))
    ctrl_c_at(monkeypatch, 5, times=2, wait_for_stop=False)
    program.main([])
    out = capsys.readouterr().out
    assert "kaydedilmedi" in out and "kaydedildi" not in out
    assert ((folder / "cevahir.zihin").read_bytes(), (folder / "cevahir.zihin.json").read_bytes()) == before
    assert files(folder, "defter-*") == [] and files(folder, "*.tmp") == []
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def taught() -> Session:
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True, situations=True, explore=True))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    for question, answer in (("Merhaba!", "Merhaba, nasılsın?"), ("Kedi ne yer?", "Kedi balık yer."),
                             ("Köpek ne yer?", "Köpek kemik yer.")):
        session.watch(question, answer)
    return session


def test_a_stopped_unut_says_how_far_it_came_and_is_resumed(tmp_path, monkeypatch):
    (tmp_path / "uzun.txt").write_text(DOCUMENT[:400], encoding="utf-8")
    one, other = taught(), taught()
    for session in (one, other):
        session.read_path(str(tmp_path / "uzun.txt"))
    number, n = one.history[-1].number, len(one.history[-1].moments)
    real, taken = one.mind.retract, []

    def retract(m):
        taken.append(m)
        if len(taken) == 3:
            one.stop.set()                               # Ctrl+C while the third moment is taken back
        return real(m)
    monkeypatch.setattr(one.mind, "retract", retract)
    out = one.forget(number)
    assert "yarıda kesildi" in out and "3/%d" % n in out and "!unut %d" % number in out
    one.stop.clear()
    assert "geri alındı" in one.forget(number) and len(taken) == n   # every moment once
    other.forget(other.history[-1].number)
    assert one.mind.S.digest() == other.mind.S.digest()


def test_research_and_sleep_stop_when_asked(monkeypatch):
    from zihin.dusunce import sleep
    from zihin.web import research

    class Pages:
        def search(self, topic, limit=2):
            return ["Kedi"]

        def extract(self, title):
            return " ".join("Kedi %d balık yer." % i for i in range(20))

        def address(self, title):
            return "test://kedi"

    session = taught()
    real, heard = Session.listen, []

    def listen(self, text, prov, speaker=None):
        real(self, text, prov, speaker)
        heard.append(text)
        if len(heard) == 2:
            self.stop.set()
    monkeypatch.setattr(Session, "listen", listen)
    with pytest.raises(Interrupted):
        research(session, "kedi", source=Pages(), way="derin", say=lambda line: None)
    assert len(heard) == 2
    nights = getattr(session.mind, "nights", 0)
    with pytest.raises(Interrupted):                     # asked to stop: no dream, no sorting, no night
        sleep(session)
    assert getattr(session.mind, "nights", 0) == nights


def small_mind(text: str) -> Mind:
    mind = Mind(Config.changed(support="members"))
    mind.add_channel(CHANNEL, "act")
    Session(mind).live(text, TOLD)
    return mind


def test_a_save_stopped_before_its_files_were_whole_leaves_the_mind_saved_before(tmp_path, monkeypatch):
    path = str(tmp_path / "zihin")
    export_portable(small_mind("Kedi balık yer."), path)
    before = sorted(os.listdir(tmp_path)), (tmp_path / "zihin.zihin").read_bytes()
    import pickle

    def stopped(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(pickle, "dump", stopped)
    with pytest.raises(KeyboardInterrupt):
        export_portable(small_mind("Köpek kemik yer."), path)
    assert (sorted(os.listdir(tmp_path)), (tmp_path / "zihin.zihin").read_bytes()) == before
    monkeypatch.undo()
    assert load(path, ledger=False).unit_of(CHANNEL, "kedi") >= 0


@pytest.mark.parametrize("with_ledger, stop_after", [(False, 1), (True, 1), (True, 2)])
def test_a_save_stopped_while_its_files_were_put_in_place_is_finished_by_load(tmp_path, monkeypatch, with_ledger,
                                                                              stop_after):
    path = str(tmp_path / "zihin")
    first = small_mind("Kedi balık yer.")
    second = Mind(program.speaking(str(tmp_path / "defterler"))) if with_ledger else Mind(Config.changed())
    second.add_channel(CHANNEL, "act")
    for _ in range(3):
        Session(second).live("Köpek kemik yer.", TOLD)
    (save if with_ledger else export_portable)(first, path)
    import zihin.persist as persist
    real, done = os.replace, []

    def replace(src, dst):
        if len(done) == stop_after:
            raise KeyboardInterrupt                      # stopped between two of its files
        done.append(dst)
        return real(src, dst)
    monkeypatch.setattr(persist.os, "replace", replace)
    with pytest.raises(KeyboardInterrupt):
        (save if with_ledger else export_portable)(second, path)
    monkeypatch.undo()
    loaded = load(path, ledger=with_ledger)
    assert loaded.unit_of(CHANNEL, "köpek") >= 0 and loaded.S.digest() == second.S.digest()
    assert [n for n in os.listdir(tmp_path) if n.endswith(".tmp")] == []


def test_what_an_unfinished_save_left_is_removed_by_the_next_load(tmp_path):
    import json
    path = str(tmp_path / "zihin")
    export_portable(small_mind("Kedi balık yer."), path)
    (tmp_path / "zihin.zihin.tmp").write_bytes(b"half a pickle")      # a save killed while it was pickling
    assert load(path, ledger=False).unit_of(CHANNEL, "kedi") >= 0
    assert sorted(os.listdir(tmp_path)) == ["zihin.zihin", "zihin.zihin.json"]
    manifest = json.loads((tmp_path / "zihin.zihin.json").read_text(encoding="utf-8"))
    manifest["sha256"] = "0" * 64                                      # a manifest whose mind file is not there
    (tmp_path / "zihin.zihin.json.tmp").write_text(json.dumps(manifest), encoding="utf-8")
    assert load(path, ledger=False).unit_of(CHANNEL, "kedi") >= 0    # the mind saved before, not that manifest
    assert sorted(os.listdir(tmp_path)) == ["zihin.zihin", "zihin.zihin.json"]
