# -*- coding: utf-8 -*-
"""Zihin, F27 Lived Exclusion: what an instance was lived to be followed by on a channel counts against what a
generalisation proposes there that it never went on with.

What is pinned here (hand-checkable numbers)
--------------------------------------------
Read (imitated, trust 1/2) "şu mutasyon : yeni .", "o mutasyon : eski ." and "bu mutasyon , zararlı .". The two-item
contexts (şu, mutasyon), (o, mutasyon), (bu, mutasyon) differ in their first place, so a category of {şu, o, bu} and the
frame (C, mutasyon) are induced. Reading the context (bu, mutasyon):

* the frame proposes ":" with what its other members lived: each member's membership is one imitated experience
  (e = 2/3), so conviction(members) = (4/3) / (4/3 + 1) = 4/7 and c = lam * e(bu -> C) * 4/7 = 1/2 * 2/3 * 4/7 = 4/21,
  e(":") = 4/21 + (1 - 4/21) / 2 = 25/42;
* the context's own lived evidence gives "," one imitated experience: c = 1/3, e(",") = 1/3 + (2/3)(1/2) = 2/3;
* off, the two race and the inner voice is silent (2 (2/3 - 25/42) * 6 < 1);
* with ``exclusive`` the context's own positive mass on the text channel (1/2) is evidence against ":", which it was
  never followed by: e(":") = (1 - 1/3) * 25/42 = 25/63, "," keeps 2/3, and the inner voice goes on ", zararlı".
* A target the instance has its own evidence on keeps that evidence (no double count), targets off the world channels
  (feelings) are never touched, and nothing is written by the reading (the digest is unchanged).
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Config, Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS
from zihin.dusunce import think
from zihin.read import excluded, members_map, own_mass


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def lived(exclusive: bool) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path, cfg.exclusive = False, None, exclusive
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    for text in ("şu mutasyon : yeni .", "o mutasyon : eski .", "bu mutasyon , zararlı ."):
        s.live(text, IMITATED)
    return s


def words_of(mind, out):
    return {mind.symbol_of(u)[1]: e for u, e in out.items() if mind.symbol_of(u)[0] == CHANNEL}


def context(mind):
    return mind.S.tuple_index[(mind.unit_of(CHANNEL, "bu"), mind.unit_of(CHANNEL, "mutasyon"))]


def test_the_switch_is_off_by_default():
    assert Config().exclusive is False and Config.changed().exclusive is False


@pytest.mark.parametrize("exclusive, colon, thought", [(False, 25 / 42, []), (True, 25 / 63, [",", "zararlı"])])
def test_what_the_instance_lived_counts_against_what_it_never_went_on_with(exclusive, colon, thought):
    s = lived(exclusive)
    mind = s.mind
    digest = mind.S.digest()
    read = words_of(mind, members_map(mind, context(mind)))
    assert mind.S.digest() == digest                                  # reading writes nothing
    assert read == {":": pytest.approx(colon, abs=1e-12), ",": pytest.approx(2 / 3, abs=1e-12)}
    assert think(s, "bu mutasyon")[0] == thought


def test_own_evidence_is_not_counted_twice_and_feelings_are_not_touched():
    s = lived(True)
    mind = s.mind
    ctx = context(mind)
    own = own_mass(mind, ctx, 1)
    comma = mind.unit_of(CHANNEL, ",")
    relief = mind.feel_unit["relief"]
    out = {comma: 0.6, relief: 0.7}
    excluded(mind, out, own)
    assert out == {comma: 0.6, relief: 0.7}                           # own target and a feeling: unchanged
    colon = mind.unit_of(CHANNEL, ":")
    out = {colon: 0.6}
    excluded(mind, out, own)
    assert out[colon] == pytest.approx(0.6 * (1 - 1 / 3), abs=1e-12)
