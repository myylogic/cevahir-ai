# -*- coding: utf-8 -*-
"""Zihin, F66 An End Is an Outcome (``Config.trial_ends``, with trials F57).

What is pinned here
-------------------
* Off by default, a studied mechanism, in use in the full configuration; it needs the trials (F57).
* F57 tries the mind's ways of reading where it decided and the world then went on from outside with a word.
  Where the other ended there, nothing was an outcome: a way that makes the mind say a word more than the answer
  could not be found to do harm. With F66 the end is what came: a configuration that would have said nothing there
  did better, one that would have said a word did worse (``deney._worth``).
* A life in which a way does that harm (the exact echo F52 without F62, answers that are the value alone): asked a
  kind of question it never watched, in the teacher's loop, the mind puts that way away by itself after two cases,
  and another kind it never watched is then answered right at first sight. Without F66 nothing is put away, and
  every new kind is wrong at first sight again.
* Where the same way is what completes the answer (answers that end with the question's verb, without F41 and
  F62) it is not put away, and the answers stay whole.
* The whole mind (with F62) is answered alike with and without it: nothing is put away.
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Config, Mind, deney
from zihin.__main__ import CHANNEL, NEEDS, STUDIED, Session, VARIANTS

THINGS = ["inek", "koyun", "keçi", "at", "eşek", "tavuk", "deve", "geyik", "aslan", "kurt", "fok", "serçe", "kaz",
          "arı", "tilki", "ayı", "fil", "kedi", "köpek", "yılan", "balina", "kartal", "tavşan", "sincap"]
FACTS = {"yer": ["ot", "et", "balık", "tohum"], "içer": ["su", "süt", "çay", "şerbet"],
         "taşır": ["dal", "taş", "yaprak", "kum"], "sever": ["güneş", "gölge", "yağmur", "rüzgar"]}
KINDS = list(FACTS)


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fact(x: str, q: str) -> str:
    return FACTS[q][(THINGS.index(x) * 3 + len(q)) % 4]


def answer(x: str, q: str, names: bool) -> str:
    return "%s %s %s" % (x, fact(x, q), q) if names else fact(x, q)


def life(names: bool = False, **switches):
    """Four kinds of fact read of every thing; two kinds of question watched twice of eight things; then the third
    kind in the teacher's loop of eight other things. Returns (the session, the loop's marks)."""
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["full"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    for name, value in switches.items():
        setattr(cfg, name, value)
    torch.manual_seed(0)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    for i, x in enumerate(THINGS):
        sentences = ["%s %s %s ." % (x, fact(x, q), q) for q in KINDS]
        for sentence in sentences[i % 4:] + sentences[:i % 4]:
            s.company(None)
            s.listen(sentence, IMITATED)
            s.end_of_turn(IMITATED, tone=".")
            mind.step([])
    for _ in range(2):
        for q in KINDS[:2]:
            for x in THINGS[:8]:
                s.watch("%s ne %s ?" % (x, q), answer(x, q, names) + " .")
    s.close_turn()
    marks = ""
    for x in THINGS[8:16]:
        said = s.reply("%s ne %s ?" % (x, KINDS[2]), limit=20)
        s.close_turn()
        s.watch("%s ne %s ?" % (x, KINDS[2]), answer(x, KINDS[2], names) + " .")
        s.close_turn()
        marks += "R" if said == answer(x, KINDS[2], names) else "W"
    return s, marks


def first_sight(s: Session, names: bool = False) -> list:
    """The fourth kind, never watched and never in a loop, of eight fresh things (on copies)."""
    return [Session(s.mind.clone()).reply("%s ne %s ?" % (x, KINDS[3]), limit=20) for x in THINGS[16:]]


def put_away(s: Session) -> list:
    return [str(c["way"]) for c in s.mind.trial_changes if not c["on"] and not c.get("owner")]


def test_off_by_default_studied_and_needs_the_trials():
    assert Config().trial_ends is False and "trial_ends" in STUDIED
    assert ("trial_ends", ("trials",)) in NEEDS
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        assert VARIANTS["full"]().trial_ends is True and VARIANTS["speaking"]().trial_ends is False
    finally:
        chat._log_path = made


def test_what_an_answer_is_worth_where_nothing_came():
    assert deney._worth(None, None) == 1 and deney._worth(7, None) == -1      # the other ended there
    assert deney._worth(7, 7) == 1 and deney._worth(None, 7) == 0 and deney._worth(8, 7) == -1   # (as before)


def test_a_way_that_goes_on_past_the_answer_is_put_away_by_the_mind_itself():
    right = [fact(x, KINDS[3]) for x in THINGS[16:]]
    without, marks = life(echo_told=False, trial_ends=False)
    assert marks.count("W") >= 4 and marks.endswith("RR")    # it learns this kind in the loop, by being shown
    assert put_away(without) == []                           # ... and no way was found to be at fault
    assert first_sight(without) == ["%s %s" % (v, KINDS[3]) for v in right]   # the next kind: the verb again
    ends, marks = life(echo_told=False, trial_ends=True)
    assert marks == "WWRRRRRR"                               # two cases: the answer ended where it went on
    assert put_away(ends) == ["echo_exact"] and ends.mind.cfg.echo_exact is False
    change = [c for c in ends.mind.trial_changes if not c["on"]][0]
    assert (change["better"], change["worse"]) == (0.0, 2.0)
    assert first_sight(ends) == right                        # the next kind is right at first sight


def test_where_the_same_way_completes_the_answer_it_is_kept():
    s, marks = life(names=True, known=False, echo_told=False, trial_ends=True)
    assert marks == "RRRRRRRR" and put_away(s) == [] and s.mind.cfg.echo_exact is True
    assert first_sight(s, True) == [answer(x, KINDS[3], True) for x in THINGS[16:]]


def test_the_whole_mind_is_answered_alike_with_it():
    plain, marks_plain = life(trial_ends=False)
    ends, marks_ends = life(trial_ends=True)
    assert marks_plain == marks_ends == "RRRRRRRR"
    assert put_away(plain) == put_away(ends) == []
    assert first_sight(plain) == first_sight(ends) == [fact(x, KINDS[3]) for x in THINGS[16:]]
