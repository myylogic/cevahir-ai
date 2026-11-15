# -*- coding: utf-8 -*-
"""Zihin, F29 Analogy by Interchangeability: a context never lived as a whole is read through the lived contexts one
interchangeable item away, the interchangeability of two items read from the lived places both have filled.

What is pinned here
-------------------
* Interchangeability is counted from shared places (hand-checkable): read "kedi uyur ." and "köpek uyur ." (imitated,
  trust 1/2). The contexts (kedi, uyur) and (köpek, uyur), and (kedi, uyur, end) and (köpek, uyur, end), differ only in
  their first place, so two places of two members each are born, each membership one imitated experience (e = 2/3,
  m = 2e - 1 = 1/3). Each place gives m_kedi m_köpek / (n - 1) = 1/9: W+ = 2/9, W- = 0.
* What is being read cannot vouch for itself: counted from before the moment köpek's sentence began, the same pair has
  no evidence at all (both memberships were born in that run).
* The analogy speaks where the evidence holds and nowhere else: watched "merhaba" / "selam" answered alike ("X , hoş
  geldin .", "X , nasılsın ?") and "merhaba , adın ne ?" answered "adım cevahir ." three times; then "selam , adın ne
  ?" is answered "adım cevahir" with the switch on and not at all with it off; "keyfin , adın ne ?" (a word never met)
  is not answered either way. The analogy works inside the frame the members' reading chose: it carries the members'
  own claims over to the instance and proposes nothing they did not live.
* Reading writes nothing (the digest is unchanged), and the switch is off by default.
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Config, Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS
from zihin.read import interchangeability, members_map


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fresh(analogy: bool) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path, cfg.analogy = False, None, analogy
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    return Session(mind)


def test_the_switch_is_off_by_default():
    assert Config().analogy is False and Config.changed().analogy is False


def test_interchangeability_is_counted_from_shared_places_lived_before():
    s = fresh(True)
    s.live("kedi uyur .", IMITATED)
    began = s.mind.t + 1                                  # the first moment of köpek's sentence
    s.live("köpek uyur .", IMITATED)
    mind = s.mind
    kedi, kopek = mind.unit_of(CHANNEL, "kedi"), mind.unit_of(CHANNEL, "köpek")
    w_pos, w_neg = interchangeability(mind, kedi, kopek, mind.t + 1)
    assert w_pos == pytest.approx(2 / 9, abs=1e-12) and w_neg == 0.0
    assert interchangeability(mind, kedi, kopek, began) == (0.0, 0.0)


def greeted(analogy: bool) -> Session:
    s = fresh(analogy)
    for _ in range(2):
        s.watch("merhaba", "merhaba , hoş geldin .")
        s.watch("selam", "selam , hoş geldin .")
        s.watch("merhaba", "merhaba , nasılsın ?")
        s.watch("selam", "selam , nasılsın ?")
    for _ in range(3):
        s.watch("merhaba , adın ne ?", "adım cevahir .")
    return s


@pytest.mark.parametrize("analogy, answer", [(True, "adım cevahir"), (False, "(sessiz)")])
def test_an_interchangeable_word_carries_the_lived_answer_over(analogy, answer):
    assert greeted(analogy).reply("selam , adın ne ?") == answer


@pytest.mark.parametrize("analogy", [True, False])
def test_a_word_never_met_carries_nothing(analogy):
    assert greeted(analogy).reply("keyfin , adın ne ?") == "(sessiz)"


def test_reading_writes_nothing():
    s = greeted(True)
    s.reply("selam , adın ne ?")                         # the question's context and its frame now exist
    mind = s.mind
    end = mind.unit_of(chat.TURN, chat.TURN_END)
    words = [mind.unit_of(CHANNEL, w) for w in ("selam", ",", "adın", "ne")]
    context = mind.S.tuple_index[tuple(words) + (end,)]
    digest = mind.S.digest()
    read = members_map(mind, context)
    assert mind.S.digest() == digest
    assert read[mind.unit_of(CHANNEL, "adım")] > 0.5 + 1.0 / 12   # carried over past the speaking line
