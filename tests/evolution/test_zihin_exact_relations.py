# -*- coding: utf-8 -*-
"""Zihin, F28 Exact Relations: a frame's relation (SELF, NEXT, START) counts for a member only where it names the very
value the member lived, as it would be resolved for the read instance; the value itself always counts.

What is pinned here (hand-checkable numbers)
--------------------------------------------
Told (trust 1/2) "renk kedi siyah .", "renk köpek siyah .", "boy kedi kısa ." twice, "boy köpek uzun ." twice and
"boy kuş kısa ."; then "renk kuş" is presented and the next word is read. The two-item contexts (renk, kedi) and
(renk, köpek) make a category {kedi, köpek} and the frame (renk, C); kuş joins it when (renk, kuş) is recognised.

* Both members lived "siyah" after their instance, and "siyah" is also one of the words they went on with. F12 alone
  counted that as NEXT ("what the member goes on with"), never as the value, so the frame said NEXT and NEXT resolved
  to kuş's own strongest continuation: e("kısa") = 25/42, e("siyah") = 1/2. That is how the relational world of the
  first lives was failed by the chat configuration (held-out 0.025).
* With ``exact_relations`` the literal value counts (both members: f = 1) and NEXT counts for a member only if "siyah"
  is that member's own strongest continuation, which it is not (kedi went on with kısa twice, köpek with uzun twice):
  e("siyah") = 25/42, e("kısa") = 1/2. The arithmetic: members kedi and köpek, each a member by one told experience
  (e = 2/3), conviction(4/3) = 4/7; kuş's membership 2/3; c = 1/2 * 2/3 * 4/7 = 4/21; e = 4/21 + (17/21) / 2 = 25/42.
* A relation the members do agree on still speaks: "muz nedir ?" after three watched "X nedir ?" / "X ..." pairs is
  answered with the asked word (SELF) and the sentence that began with it (START), with the switch on as off.
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, TOLD, Config, Event, Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fresh(exact: bool) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path, cfg.exact_relations = False, None, exact
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    return Session(mind)


def test_the_switch_is_off_by_default():
    assert Config().exact_relations is False and Config.changed().exact_relations is False


@pytest.mark.parametrize("exact, black, short", [(False, 0.5, 25 / 42), (True, 25 / 42, 0.5)])
def test_a_relation_counts_only_where_it_names_the_value_the_member_lived(exact, black, short):
    s = fresh(exact)
    for text in ("renk kedi siyah .", "renk köpek siyah .", "boy kedi kısa .", "boy kedi kısa .", "boy köpek uzun .",
                 "boy köpek uzun .", "boy kuş kısa ."):
        s.live(text, TOLD)
    probe = s.mind.clone()
    for word in ("renk", "kuş"):
        probe.step([Event(CHANNEL, word, TOLD)])
    expected = probe.expected(CHANNEL)
    assert expected.get("siyah", 0.5) == pytest.approx(black, abs=1e-12)
    assert expected.get("kısa", 0.5) == pytest.approx(short, abs=1e-12)


@pytest.mark.parametrize("exact", [False, True])
def test_a_relation_the_members_agree_on_still_speaks(exact):
    s = fresh(exact)
    for q, a in (("elma nedir ?", "elma kırmızı bir meyvedir ."), ("armut nedir ?", "armut sarı bir meyvedir ."),
                 ("kiraz nedir ?", "kiraz küçük bir meyvedir .")):
        s.watch(q, a)
    for t in ("elma kırmızı bir meyvedir .", "armut sarı bir meyvedir .", "kiraz küçük bir meyvedir .",
              "muz uzun bir meyvedir .", "bu muz tatlı .", "bu muz tatlı .", "bu muz tatlı ."):
        s.live(t, IMITATED)
    assert s.reply("muz nedir ?") == "muz uzun bir meyvedir"
