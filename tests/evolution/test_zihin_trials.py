# -*- coding: utf-8 -*-
"""Zihin, F57 Trials and F58 Built Relations: the mind's ways of reading are on trial against its own configuration,
the configuration follows what the trials show, and where it fails it builds relations of its own and tries them.

What is pinned here
-------------------
* Both switches are off by default and are studied mechanisms (``--ac trials``, ``--ac trials,built_relations``).
  While they are off nothing of them exists in the mind.
* The experiment: a context at which the mind decided, and which then goes on from outside, is read again dry with
  the configuration as it is and with one way changed. A way that did better twice and never worse is taken into
  use (the asked place, in a world where the answer is the value alone); the mind then answers things it was never
  asked about, which it could not before. Every case and every change is kept with what each configuration would
  have said.
* Nothing is found where nothing can be known: of things nothing was read of, no way is taken.
* One change at a time: two ways that each mend the same failure are both proven by the same cases; one is taken
  (the first in the mind's list), and the other's trial begins again against the new configuration, where it makes
  no difference any more.
* A way in use is put away by the same rule the other way round (two cases against it, none for).
* F58: where the configuration did not say what came, a chain of two of the mind's relations that names it for the
  filler is proposed, tried like any way and taken into use ("what is known of the thing, then what stands at the
  asked place of that": a fact read of the thing's class). A relation it built is a step of the next chain: a fact
  read of the class's order is reached by a chain of three, built from the chain of two.
* Two together: where no single way mends a failure, a sleeping way that alone changed what was read there is
  tried with each other sleeping way; a pair that says what came is kept and tried like any way, and when its
  record proves it the two are taken into use together, as one change.
* The configuration the mind came to is its own: saved and loaded with it; a moment undone takes the change back.
* The chat's ``!deney`` reports the record.
"""
from __future__ import annotations

import os

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Config, Mind, deney, persist, read
from zihin.__main__ import CHANNEL, STUDIED, Session, VARIANTS

THINGS = ["inek", "koyun", "keçi", "at", "eşek", "tavuk", "deve", "geyik", "aslan", "kurt", "fok", "serçe",
          "kaz", "arı", "tilki", "ayı", "fil", "kedi", "köpek", "yılan"]
FACTS = {"yer": ["ot", "et", "balık", "tohum"], "içer": ["su", "süt", "çay", "şerbet"],
         "taşır": ["dal", "taş", "yaprak", "kum"]}
CLASSES = ["memeli", "sürüngen", "kanatlı", "sucul"]
BEARS = {"memeli": "yavru", "sürüngen": "yumurta", "kanatlı": "civciv", "sucul": "larva"}
ORDER = {"memeli": "sıcakkanlılar", "sürüngen": "pullular", "kanatlı": "tüylüler", "sucul": "solungaçlılar"}
MAKES = {"sıcakkanlılar": "ısı", "pullular": "zehir", "tüylüler": "ötüş", "solungaçlılar": "kabarcık"}


def fact(x: str, q: str) -> str:
    return FACTS[q][(THINGS.index(x) * 3 + len(q)) % 4]


def kind(x: str) -> str:
    return CLASSES[(THINGS.index(x) * 7 + 1) % 4]


def truth(x: str, q: str) -> str:
    if q == "doğurur":
        return BEARS[kind(x)]
    if q == "üretir":
        return MAKES[ORDER[kind(x)]]
    return fact(x, q)


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def raised(read_facts=True, keep_log=False, **switches) -> Session:
    """Every fact and its class read once of every thing (in an order that differs from thing to thing); of every
    class what it bears and its order, of every order what it makes. Nothing is asked yet."""
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = keep_log, None
    for name, value in switches.items():
        setattr(cfg, name, value)
    torch.manual_seed(0)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)

    def told(sentence: str) -> None:
        s.company(None)
        s.listen(sentence, IMITATED)
        s.end_of_turn(IMITATED, tone=".")
        mind.step([])
    for k in CLASSES:
        told("%s %s doğurur ." % (k, BEARS[k]))
        told("%s %s takımındandır ." % (k, ORDER[k]))
        told("%s %s üretir ." % (ORDER[k], MAKES[ORDER[k]]))
    for i, x in enumerate(THINGS if read_facts else ()):
        sentences = ["%s %s %s ." % (x, fact(x, q), q) for q in FACTS] + ["%s %s sınıfındandır ." % (x, kind(x))]
        for sentence in sentences[i % 4:] + sentences[:i % 4]:      # in an order of its own: no fact is "the first"
            told(sentence)
    return s


def teacher(s: Session, x: str, q: str) -> str:
    """The mind is asked; whatever it says, the true dialogue is then shown. What it said."""
    said = s.reply("%s ne %s ?" % (x, q), limit=30)
    s.close_turn()
    s.watch("%s ne %s ?" % (x, q), "%s ." % truth(x, q))
    s.close_turn()
    return said


def asks(s: Session, x: str, q: str) -> str:
    return Session(s.mind.clone()).reply("%s ne %s ?" % (x, q), limit=30)


def test_off_by_default_and_studied():
    cfg = Config()
    assert cfg.trials is False and cfg.built_relations is False
    assert "trials" in STUDIED and "built_relations" in STUDIED


def test_while_off_nothing_of_it_exists():
    s = raised(faults=True)
    for x in THINGS[:4]:
        teacher(s, x, "yer")
    m = s.mind
    assert not m.called and not m.trial_changes and not m.trial_cases and not m.ch.has(deney.CHANNEL)
    assert m.cfg.asked_place is False and m.relations == () and m.relation_paths == []


def test_a_way_is_taken_into_use_by_the_minds_own_trials():
    s = raised(faults=True, trials=True)
    m = s.mind
    assert m.cfg.asked_place is False
    assert all(asks(s, x, "yer") == "(sessiz)" for x in THINGS[12:16])     # before: the value alone is not found
    said = [teacher(s, x, "yer") for x in THINGS[:6]]
    assert m.cfg.asked_place is True                                       # its own doing: nothing set it
    change = m.trial_changes[0]
    assert (change["way"], change["on"], change["better"], change["worse"]) == ("asked_place", True, 2.0, 0.0)
    assert [c["way"] for c in m.trial_changes] == ["asked_place"]          # and nothing else was changed
    assert all(asks(s, x, "yer") == fact(x, "yer") for x in THINGS[12:16])  # things it was never asked about
    assert said[-1] == fact(THINGS[5], "yer")                              # and in its own life from then on
    w_pos, w_neg = deney.record(m, "asked_place")
    assert w_pos >= 2.0 and w_neg == 0.0
    case = next(c for c in m.trial_cases if c["way"] == "asked_place")     # the case is kept as it was read
    assert case["better"] and case["with"] == case["came"] and case["without"] != case["came"]
    assert m.trial_base["asked_place"] is False                            # as it stood when the trials began


def test_nothing_is_taken_where_nothing_can_be_known():
    s = raised(read_facts=False, faults=True, trials=True)
    for x in THINGS[:6]:
        teacher(s, x, "yer")
    m = s.mind
    assert m.cfg.asked_place is False and m.trial_changes == []
    assert deney.record(m, "asked_place") == (0.0, 0.0)


def scripted(monkeypatch, says, reads=None):
    """The dry readings replaced by a script: ``says(mind)`` -> the word it would say, or None; ``reads(mind)`` ->
    something that stands for what it read (ways that change it are told apart from ways that change nothing)."""
    def would_say(mind, context, chain=None, read=None):
        if read is not None and reads is not None:
            read["read"] = reads(mind)
        word = says(mind)
        if word is None:
            return {}
        unit = mind.unit_of(CHANNEL, word)
        return {int(globals()["read"].mv_view(mind, "u_channel")[unit]): unit}
    monkeypatch.setattr(deney, "would_say", would_say)


def test_one_change_at_a_time_and_a_sleeping_trial_begins_again(monkeypatch):
    s = raised(trials=True)
    m = s.mind
    now = {}
    scripted(monkeypatch, lambda mind: now["right"] if (mind.cfg.known or mind.cfg.asked_place) else None)
    for x in THINGS[:2]:                                   # either of two ways alone mends the same two failures
        now["right"] = fact(x, "yer")
        teacher(s, x, "yer")
    assert m.cfg.known is True and m.cfg.asked_place is False              # one is taken: the first in its list
    assert [(c["way"], c["on"]) for c in m.trial_changes] == [("known", True)]
    assert deney.record(m, "asked_place") == (2.0, 0.0)                    # the evidence stays in its tie
    assert deney.standing(m, "asked_place") == (0.0, 0.0)                  # its trial begins again
    for x in THINGS[2:6]:                                  # against the new configuration it makes no difference
        now["right"] = fact(x, "yer")
        teacher(s, x, "yer")
    assert m.cfg.asked_place is False and len(m.trial_changes) == 1
    assert deney.standing(m, "known")[0] >= 2.0 and deney.standing(m, "known")[1] == 0.0


def test_a_way_in_use_that_does_harm_is_put_away(monkeypatch):
    s = raised(trials=True, families=True)                 # raised with it on by hand
    m = s.mind
    scripted(monkeypatch, lambda mind: "taş" if mind.cfg.families else None)   # with it: a wrong word; without: none
    teacher(s, THINGS[0], "yer")
    assert m.cfg.families is True and deney.record(m, "families") == (0.0, 1.0)   # one case is not knowledge
    teacher(s, THINGS[1], "yer")
    assert m.cfg.families is False
    change = m.trial_changes[-1]
    assert (change["way"], change["on"], change["better"], change["worse"]) == ("families", False, 0.0, 2.0)
    assert m.trial_base["families"] is True


def test_two_ways_that_only_work_together_are_found_and_taken_as_one_change(monkeypatch):
    s = raised(trials=True)
    m = s.mind
    now = {}
    scripted(monkeypatch,                                  # the answer comes only with both; one of them alone
             lambda mind: now["right"] if (mind.cfg.families and mind.cfg.unseen_share) else None,   # changes what
             lambda mind: bool(mind.cfg.families))         # is read, the other alone changes nothing
    now["right"] = fact(THINGS[0], "yer")
    teacher(s, THINGS[0], "yer")
    pair = (deney.PAIR, "families", "unseen_share")
    assert m.pairs_tried == [pair] and deney.record(m, pair) == (1.0, 0.0)
    assert deney.record(m, "families") == (0.0, 0.0) and deney.record(m, "unseen_share") == (0.0, 0.0)
    assert m.cfg.families is False and m.cfg.unseen_share is False          # one case is not knowledge
    now["right"] = fact(THINGS[1], "yer")
    teacher(s, THINGS[1], "yer")
    assert m.cfg.families is True and m.cfg.unseen_share is True            # taken together
    change = m.trial_changes[-1]
    assert (change["way"], change["on"], change["better"], change["worse"]) == (pair, True, 2.0, 0.0)
    assert len(m.trial_changes) == 1 and pair not in deney.ways(m)          # one change; the pair is not a way any more
    assert "birlikte" in "\n".join(deney.report(m))


def test_no_pair_is_tried_where_one_way_mends_it(monkeypatch):
    s = raised(trials=True)
    m = s.mind
    now = {}
    scripted(monkeypatch, lambda mind: now["right"] if mind.cfg.asked_place else None,
             lambda mind: (bool(mind.cfg.asked_place), bool(mind.cfg.families)))
    for x in THINGS[:2]:
        now["right"] = fact(x, "yer")
        teacher(s, x, "yer")
    assert m.pairs_tried == [] and m.cfg.asked_place is True


def test_it_builds_a_relation_of_its_own_and_then_one_out_of_that():
    s = raised(faults=True, trials=True, built_relations=True)
    m = s.mind
    assert all(asks(s, x, "doğurur") == "(sessiz)" for x in THINGS[12:16])
    for x in THINGS[:6]:                                   # never read of the thing: read of its class
        teacher(s, x, "doğurur")
    two = (read.KNOWN, read.ASKED)                         # what is known of it, then the asked place of that
    assert two in m.relations and two in m.relation_paths
    assert read.built_code(m, two) <= read.BUILT and read.built_path(m, read.built_code(m, two)) == two
    assert all(asks(s, x, "doğurur") == BEARS[kind(x)] for x in THINGS[12:16])
    assert any(c["way"] == two and c["on"] for c in m.trial_changes)
    assert all(asks(s, x, "üretir") == "(sessiz)" for x in THINGS[12:16])  # read of the class's order: not yet
    for x in THINGS[6:12]:
        teacher(s, x, "üretir")
    three = (read.KNOWN, read.KNOWN, read.ASKED)           # a chain of the chain it took into use
    assert three in m.relations
    assert all(asks(s, x, "üretir") == MAKES[ORDER[kind(x)]] for x in THINGS[12:16])
    assert all(asks(s, x, "doğurur") == BEARS[kind(x)] for x in THINGS[12:16])    # and what it could do is kept
    text = "\n".join(deney.report(m))
    assert "kendi kurduğu ilişki" in text and "kullanıma alındı" in text


def test_the_configuration_it_came_to_is_saved_with_it(tmp_path):
    s = raised(faults=True, trials=True, built_relations=True)
    for x in THINGS[:6]:
        teacher(s, x, "doğurur")
    m = s.mind
    assert m.relations
    path = os.path.join(str(tmp_path), "mind.zihin")
    persist.save(m, path)
    loaded = persist.load(path, ledger=False)
    assert loaded.relations == m.relations and loaded.relation_paths == m.relation_paths
    assert loaded.trial_changes == m.trial_changes and loaded.trial_mark == m.trial_mark
    assert loaded.cfg.trials is True and loaded.cfg.asked_place == m.cfg.asked_place
    for way in deney.ways(m):
        assert deney.record(loaded, way) == deney.record(m, way)
    twin = Session(loaded)
    assert all(asks(twin, x, "doğurur") == BEARS[kind(x)] for x in THINGS[12:16])


def test_a_moment_undone_takes_the_change_back():
    s = raised(keep_log=True, faults=True, trials=True)
    m = s.mind
    undone = False
    for x in THINGS[:6]:
        s.reply("%s ne yer ?" % x, limit=30)
        s.close_turn()
        s.company(chat.SOMEONE)
        s.listen("%s ne yer ?" % x, IMITATED, speaker=chat.SOMEONE)
        s.end_of_turn(IMITATED, tone="?")
        before = len(m.trial_changes)
        s.listen(fact(x, "yer"), IMITATED)                 # the moment the answer comes: the trial, and the change
        if len(m.trial_changes) > before and not undone:
            assert m.cfg.asked_place is True
            m.undo_last()
            assert m.cfg.asked_place is False and len(m.trial_changes) == before
            assert deney.record(m, "asked_place") == (1.0, 0.0)
            s.listen(fact(x, "yer"), IMITATED)             # lived again, it comes to the same
            assert m.cfg.asked_place is True and len(m.trial_changes) == before + 1
            undone = True
        s.end_of_turn(IMITATED, tone=".")
        m.step([])
        s.close_turn()
    assert undone


def test_the_chat_reports_the_record():
    s = raised(faults=True, trials=True)
    assert "henüz hiçbir yol bir fark yaratmadı" in "\n".join(deney.report(s.mind))
    for x in THINGS[:6]:
        teacher(s, x, "yer")
    text = "\n".join(deney.report(s.mind))
    assert "sorulan yer (F55)" in text and "kullanıma alındı" in text and "gelen:" in text
