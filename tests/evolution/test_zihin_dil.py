# -*- coding: utf-8 -*-
"""Zihin, a mind's language convention (--dil): how its text is lower-cased.

What is pinned here
-------------------
* "tr" (the default) lower-cases by Turkish rules (I -> ı, İ -> i); "diger" is plain lower case (I -> i), for
  languages that are not Turkish.
* The session uses the mind's convention for everything it reads and hears: documents, lines, the speaker's name,
  what it thinks about, and what it researches (the topic and the pages read).
* The convention is kept by the mind: saved and loaded with it; a mind saved before it existed is "tr".
* The program raises a new mind with --dil; a loaded mind keeps the convention it was raised with and says so.
"""
from __future__ import annotations

import io

from zihin import TOLD, Config, Mind
from zihin.__main__ import CHANNEL, PERSON, Session, _one_line, main
from zihin.dusunce import topic_of
from zihin.persist import load, save

ENGLISH = "How are you?\tI am fine.\nWhere is Istanbul?\tIstanbul is in Turkey.\n"


class Args:
    his = False


def new_mind(dil: str = None) -> Mind:
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True, situations=True))
    mind.add_channel(CHANNEL, "act")
    if dil:
        mind.dil = dil
    return mind


def test_tr_is_the_default_and_diger_is_plain_lower_case():
    tr = Session(new_mind())
    assert tr.dil == "tr" and tr.words("I am in Istanbul, İzmir.") == ["ı", "am", "in", "ıstanbul", ",", "izmir", "."]
    other = Session(new_mind("diger"))
    assert other.words("I am in Istanbul.") == ["i", "am", "in", "istanbul", "."]


def test_the_session_reads_and_hears_by_the_minds_convention(tmp_path, capsys):
    (tmp_path / "en.tsv").write_text(ENGLISH, encoding="utf-8")
    for dil, i, not_i, city in (("tr", "ı", "i", "ıstanbul"), ("diger", "i", "ı", "istanbul")):
        mind = new_mind(dil)
        session = Session(mind)
        session.read_path(str(tmp_path / "en.tsv"))       # read
        session.live("I see.", TOLD)                     # heard
        _one_line(session, "!kim Ian", False, Args(), lambda reason: None, 0)
        session.live("Istanbul is big.", TOLD)
        assert mind.unit_of(CHANNEL, i) >= 0 and mind.unit_of(CHANNEL, not_i) < 0
        assert mind.unit_of(CHANNEL, city) >= 0 and session.partner == ("ıan" if dil == "tr" else "ian")
        assert mind.unit_of(PERSON, session.partner) >= 0
        assert topic_of(session, "Istanbul") == city      # what it thinks about is the word it heard, as it heard it


def test_a_mind_keeps_its_convention_saved_and_loaded(tmp_path):
    english = new_mind("diger")
    Session(english).live("I am here.", TOLD)
    save(english, str(tmp_path / "en"))
    assert load(str(tmp_path / "en"), ledger=False).dil == "diger"
    older = new_mind()                                   # raised before the convention was kept
    Session(older).live("Merhaba.", TOLD)
    assert not hasattr(older, "dil")
    save(older, str(tmp_path / "eski"))
    assert load(str(tmp_path / "eski"), ledger=False).dil == "tr"


def test_the_program_raises_a_new_mind_with_dil_and_a_loaded_mind_keeps_its_own(monkeypatch, tmp_path, capsys):
    home = tmp_path / "ev"
    monkeypatch.setenv("CEVAHIR_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO("I am here.\n!cik\n"))
    main(["--yeni", "--dil", "diger", "--kim", "Ian"])
    mind = load(str(home / "zihin" / "cevahir"), ledger=False)
    assert mind.dil == "diger" and mind.unit_of(CHANNEL, "i") >= 0 and mind.unit_of(CHANNEL, "ı") < 0
    assert mind.unit_of(PERSON, "ian") >= 0
    capsys.readouterr()
    monkeypatch.setattr("sys.stdin", io.StringIO("I see.\n!cik\n"))
    main(["--dil", "tr"])                                # the home mind again: it keeps how it was raised
    assert "'diger' harf kuralıyla büyüdü" in capsys.readouterr().out
    mind = load(str(home / "zihin" / "cevahir"), ledger=False)
    assert mind.dil == "diger" and mind.unit_of(CHANNEL, "ı") < 0


class Pages:
    """A source of one page (no network): what research reads."""

    def search(self, topic, limit=2):
        return ["Istanbul"]

    def extract(self, title):
        return " ".join(["Istanbul is a big city."] * 5 + ["It lies on two continents."])

    def address(self, title):
        return "test://%s" % title


def test_research_reads_and_measures_by_the_minds_convention():
    from zihin.web import firm_knowledge, research
    for dil, city, not_city in (("tr", "ıstanbul", "istanbul"), ("diger", "istanbul", "ıstanbul")):
        mind = new_mind(dil)
        report = research(Session(mind), "Istanbul", source=Pages(), way="ozet", say=lambda line: None)
        assert report["sentences"] == 6
        assert mind.unit_of(CHANNEL, city) >= 0 and mind.unit_of(CHANNEL, not_city) < 0
        assert report["firm_after"] == firm_knowledge(mind, mind.unit_of(CHANNEL, city)) > 0   # the gain is measured
