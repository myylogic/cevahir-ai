# -*- coding: utf-8 -*-
"""Zihin, the ground (``Config.grounded``, zihin/zemin.py): what is perceived and what is said meet.

What is pinned here
-------------------
* Off by default and not in the whole mind; off, perceiving changes nothing (the hook does not run).
* On, everything the ground learns is written in the moments of the mind: retracting the moments of a situation
  takes its evidence back exactly (the store's digest as before the situation).
* In a small world of people, things, orders and their descriptions (opaque features; the same words for the same
  situations): a proper name comes to name its person's very identity, a kind word the kind's features; an order never
  heard, of a person and a thing never together in an order, is understood with only the things in view and answered
  from the meaning ("ali topa bak" -> "ali topa bakıyor"), as the description goes on with what is seen; a word that
  expects what a thing is not ("topa" expects roundness; a cat is not round) does not name it.
"""
from __future__ import annotations

import itertools

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Config, Mind
from zihin.__main__ import CHANNEL, STUDIED, SOMEONE, Session, VARIANTS
from zihin.zemin import Zemin, answer, contradicts, understand, _features, _word_profile

PEOPLE = {"ali": 1, "ayşe": 2, "can": 3}
KINDS = {  # word in the dative (the target of "bak"/"koş"), its features
    "kediye": {"d0": "v1", "d1": "v2", "d2": "v3", "d3": "v9"},
    "topa": {"d0": "v4", "d1": "v5", "d2": "v6", "d3": "v9"},
    "eve": {"d0": "v4", "d1": "v7", "d2": "v8", "d3": "v9"},
}
PERSON = {"d0": "v1", "d1": "v10", "d2": "v11", "d3": "v12"}
VERBS = {"bak": ("bakıyor", {"a0": "u1", "a1": "u0"}), "koş": ("koşuyor", {"a0": "u0", "a1": "u1"})}
DOER = {"p0": 1, "p1": 0}
OTHER = {"p0": 0, "p1": 1}
HELD = ("ali", "bak", "topa")                      # never ordered together


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def mind_of(grounded: bool) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    cfg.grounded = grounded
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    mind.zemin = Zemin(CHANNEL)
    return Session(mind)


def scenes(person: str, kind: str, verb: str, serial: int):
    """(what is in view while the order is heard, the same with the event going on)."""
    thing = 100 + serial
    things = [{"index": PEOPLE[person], "feats": dict(PERSON, d9="w%d" % PEOPLE[person])},
              {"index": thing, "feats": dict(KINDS[kind])}]
    before = {"things": things, "event": None, "pointed": None}
    event = {"action": dict(VERBS[verb][1]), "parts": [{"index": PEOPLE[person], "part": dict(DOER)},
                                                       {"index": thing, "part": dict(OTHER)}]}
    return before, {"things": things, "event": event, "pointed": None}


def live(s: Session, person: str, kind: str, verb: str, serial: int) -> None:
    """An order, then the doing, described as it happens."""
    z, mind = s.mind.zemin, s.mind
    before, during = scenes(person, kind, verb, serial)
    order, said = "%s %s %s" % (person, kind, verb), "%s %s %s" % (person, kind, VERBS[verb][0])
    s.company(SOMEONE)
    z.see(before)
    s.listen(order, IMITATED, speaker=SOMEONE)
    s.end_of_turn(IMITATED, tone=".")
    z.heard(s.body(order), "soran")
    z.see(during)
    s.listen(said, IMITATED)
    s.end_of_turn(IMITATED, tone=".")
    z.heard(s.body(said), "cevaplayan")
    mind.step([])


def life(s: Session, rounds: int = 3) -> None:
    serial = 0
    with torch.no_grad():
        for _ in range(rounds):
            for person, kind, verb in itertools.product(PEOPLE, KINDS, VERBS):
                if (person, verb, kind) == HELD:
                    continue
                serial += 1
                live(s, person, kind, verb, serial)
        s.mind.zemin.see(None)
        s.mind.step([])


@pytest.fixture(scope="module")
def grown():
    s = mind_of(True)
    life(s)
    return s


def test_off_by_default_and_not_in_the_whole_mind():
    assert Config().grounded is False and "grounded" not in STUDIED
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        assert VARIANTS["full"]().grounded is False
    finally:
        chat._log_path = made


def test_off_perceiving_changes_nothing():
    a, b = mind_of(False), mind_of(False)
    with torch.no_grad():
        live(a, "ali", "kediye", "koş", 1)
        b.mind.zemin.pending = []                    # b never perceives: its core lives the same words
        before, during = scenes("ali", "kediye", "koş", 1)
        for text in ("ali kediye koş", "ali kediye koşuyor"):
            b.company(SOMEONE if text.endswith("koş") else SOMEONE)
        b.company(SOMEONE)
        b.listen("ali kediye koş", IMITATED, speaker=SOMEONE)
        b.end_of_turn(IMITATED, tone=".")
        b.listen("ali kediye koşuyor", IMITATED)
        b.end_of_turn(IMITATED, tone=".")
        b.mind.step([])
    assert a.mind.S.digest() == b.mind.S.digest()


def _evidence(S, n: int) -> torch.Tensor:
    """The four evidence tables of the first n ties, [4, n, P]."""
    ids = torch.arange(n)
    return torch.stack([S.w(table, ids, tables=1) for table in range(4)])


def test_on_a_situation_is_retracted_exactly():
    """Exact Retraction is exact for the evidence (the structure a moment made may stay, as in the core): every tie
    that was there holds what it held, and every tie born in the situation holds nothing."""
    s = mind_of(True)
    s.mind.S.keep_log = True
    with torch.no_grad():
        live(s, "ali", "kediye", "koş", 1)
        live(s, "ayşe", "eve", "bak", 2)
        S = s.mind.S
        n, first = S.n_ties, s.mind.t + 1
        held = _evidence(S, n).clone()
        live(s, "can", "topa", "koş", 3)
        grounded = [t for t in range(n, S.n_ties) if int(S.t_form[t]) >= 4]
        assert grounded                                      # the ground wrote in the situation
        for m in range(s.mind.t, first - 1, -1):
            s.mind.retract(m)
    assert torch.equal(_evidence(S, n), held)
    assert float(_evidence(S, S.n_ties)[:, n:].abs().sum()) == 0.0


def test_a_proper_name_names_its_person_a_kind_word_its_kind(grown):
    mind, z = grown.mind, grown.mind.zemin
    wc = mind.ch.id(CHANNEL)
    ali = mind._atom(wc, "ali", create=False)
    topa = mind._atom(wc, "topa", create=False)
    person = mind.unit_of("z_nesne", PEOPLE["ali"])
    ident = mind.ch.id("z_kimlik")
    own = next(f for f in _features(mind, person) if int(mind.S.u_channel[f]) == ident)
    assert _word_profile(mind, z, ali)[ident][own] > 0.5            # "ali" expects Ali's very identity
    ball = {mind.unit_of("z:" + d, v): 1.0 for d, v in KINDS["topa"].items()}
    cat = {mind.unit_of("z:" + d, v): 1.0 for d, v in KINDS["kediye"].items()}
    assert not contradicts(mind, z, topa, ball) and contradicts(mind, z, topa, cat)


def test_an_order_never_heard_is_understood_and_answered_from_the_meaning(grown):
    person, verb, kind = HELD
    before, during = scenes(person, kind, verb, 999)
    order = "%s %s %s" % (person, kind, verb)
    for scene, when in ((before, None), (during, "once")):           # with only the things in view; and while the
        c = grown.mind.clone()                                        # person does it (heard before it began)
        z = c.zemin
        z.pending, z.waiting, z.turn, z.event = [], None, None, None
        z.see(before)
        c.step([])
        q = understand(c, z, Session(c).body(order))
        assert q is not None and len(q["heads"]) == 2
        if scene is during:
            z.see(during)
            c.step([])
        assert answer(c, z, Session(c).body(order), when=when) == ["ali", "topa", "bakıyor"]
