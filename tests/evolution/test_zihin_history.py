# -*- coding: utf-8 -*-
"""Zihin, the session's history: structured entries, and what a thought after a silence is about.

What is pinned here
-------------------
* Every entry of the history says what it was in a field of its own (``kind``: a line "told" or "sensed", a "chat"
  line with the mind's answer, a file "read" or "taught", a dialogue "watched"), with its number, how !gecmis shows
  it, its moments, the last utterance heard in it and the words lived in it. Nothing is recognised by the text it is
  shown with.
* What to think about after a silence (topic_of, the online life's thought) is the last utterance the mind heard,
  whatever the entry: a line of the owner that starts with "[" is a line like any other, and after a file was read
  it is the file's last sentence (no topic at all before). Its word is the best topic by the mind's evidence
  (tau x D: telling and not yet well known), so a word whose followers were never tried is none.
"""
from __future__ import annotations

import threading
import time

from zihin import TOLD, Config, Mind
from zihin.__main__ import CHANNEL, Life, Session, _one_line
from zihin.dusunce import topic_of


class Args:
    his = False


def session() -> Session:
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True, situations=True, explore=True))
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    for question, answer in (("Merhaba!", "Merhaba, nasılsın?"), ("Kedi ne yer?", "Kedi balık yer."),
                             ("Köpek ne yer?", "Köpek kemik yer.")):
        s.watch(question, answer)
    return s


def run(s, line: str, chat: bool = False) -> None:
    _one_line(s, line, chat, Args(), lambda reason: None, 0)


def test_every_entry_says_what_it_was_in_a_field_of_its_own(tmp_path, capsys):
    (tmp_path / "gezi.txt").write_text("Penguenler buzulların kenarında yaşar. Fokbalıkları kayaların üstünde uyur.",
                                       encoding="utf-8")
    (tmp_path / "ders.tsv").write_text("Zürafa ne yer?\tZürafa yaprak yer.\n", encoding="utf-8")
    s = session()
    run(s, "[okudu] bu bir satır")                       # an owner's line that looks like an entry's label
    run(s, "!gordum Bir kuş uçtu.")
    run(s, "Kedi ne yer?", chat=True)
    run(s, "!oku %s" % (tmp_path / "gezi.txt"))
    run(s, "!ogret %s" % (tmp_path / "ders.tsv"))
    run(s, "!izle Köpek ne yer? || Köpek kemik yer.")
    assert [e.kind for e in s.history] == ["told", "sensed", "chat", "read", "taught", "watched"]
    assert [e.number for e in s.history] == [1, 2, 3, 4, 5, 6]
    assert [e.text for e in s.history] == ["[okudu] bu bir satır", "Bir kuş uçtu.", "Kedi ne yer?", "[okudu] gezi.txt",
                                           "[öğretti] ders.tsv", "[izledi] Köpek ne yer? || Köpek kemik yer."]
    assert [e.heard for e in s.history] == ["[okudu] bu bir satır", "Bir kuş uçtu.", "Kedi ne yer?",
                                            "Fokbalıkları kayaların üstünde uyur.", "Zürafa yaprak yer.",
                                            "Köpek kemik yer."]
    assert s.history[0].counts == {"[": 1, "okudu": 1, "]": 1, "bu": 1, "bir": 1, "satır": 1}
    assert s.history[2].counts["kedi"] >= 2                # heard, and said in its answer ("kedi balık yer")
    capsys.readouterr()
    run(s, "!gecmis")
    assert capsys.readouterr().out.splitlines()[0] == "    1  [okudu] bu bir satır"


LINE = "[not] Penguenler soğuk sularda yüzer."


def test_a_line_that_starts_with_a_bracket_is_a_line():
    s = session()
    s.live(LINE, TOLD)
    s.live(LINE, TOLD)                                     # lived twice: what follows its words has been tried once
    assert topic_of(s) in set(s.words(LINE))             # a word of the line (it was "" before: the line was skipped)
    assert topic_of(s) == topic_of(s, LINE) != ""


def test_after_a_file_was_read_the_thought_is_about_its_last_sentence(tmp_path, capsys):
    (tmp_path / "gezi.txt").write_text("Penguenler buzulların kenarında yaşar. Fokbalıkları kayaların üstünde uyur.",
                                       encoding="utf-8")
    s = session()
    s.read_path(str(tmp_path / "gezi.txt"))
    s.read_path(str(tmp_path / "gezi.txt"))                # read twice: what follows its words has been tried once
    assert topic_of(s) in {"fokbalıkları", "kayaların", "üstünde"}   # its last sentence ("uyur" ends it: nothing

    def handle(line):
        return _one_line(s, line, True, Args(), lambda reason: None, 0) is not None
    life = Life(s, handle, threading.RLock(), beat=0.02, think_after=0.05, sleep_after=0, pace=0.0)
    life.start()
    deadline = time.time() + 20
    while life.thought is None and time.time() < deadline:
        time.sleep(0.02)
    life.stop()
    assert life.thought is not None and life.thought[0] in {"fokbalıkları", "kayaların", "üstünde"}   # followed it)
