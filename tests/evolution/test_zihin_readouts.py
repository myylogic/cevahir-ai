# -*- coding: utf-8 -*-
"""Zihin: the program prints what it reads about the mind as labelled readouts of the evidence, never as sentences
in the mind's mouth (no first-person self report or night report), and the self report ranks words by lift over
their base rate, excluding none by shape or frequency."""
from __future__ import annotations

from zihin import Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS, night_report
from zihin.dusunce import sleep
from zihin.kendilik import self_report

FIRST_PERSON = ("kendimi", "sesimle", "içimde", "kanıtım", "rüyamda", "gördüm", "uyandım", "güvenim", "hatırladım")


def lived() -> Session:
    mind = Mind(VARIANTS["speaking"]())
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    for _ in range(4):
        session.watch("kedi ne yer ?", "kedi balık yer .")
        session.watch("köpek ne yer ?", "köpek kemik yer .")
    session.reply("kedi ne yer ?")
    return session


def test_self_report_is_a_labelled_readout():
    session = lived()
    text = self_report(session.mind, CHANNEL)
    assert text.lstrip().startswith("öz-birim:") and not any(w in text for w in FIRST_PERSON)


def test_night_report_is_a_labelled_readout():
    session = lived()
    lines = night_report(sleep(session, 1))
    assert any(line.lstrip().startswith("uyku raporu:") for line in lines)
    assert not any(w in line for line in lines for w in FIRST_PERSON)
