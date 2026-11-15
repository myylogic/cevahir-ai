# -*- coding: utf-8 -*-
"""Zihin, F30 Corroboration: independent lived evidence about the same claim adds up (the revision rule of NARS).

What is pinned here
-------------------
* Watched "adın ne ?" -> "adım cevahir ." three times and "senin adın ne ?" -> "adım cevahir ." once. Asked "peki adın
  ne ?" (never lived), the frame [X] adın ne <end> speaks with its one member (senin) and stays under the speaking
  line; the lived situation most like the question ("adın ne <end>", lived three times) also stays under it alone. With
  ``corroborate`` the two independent readings revise each other and the mind answers "adım cevahir"; without it, it is
  silent.
* No lived context is counted twice: the situations' reading leaves out the frame's member instances (the lived
  "senin adın ne <end>" is the frame's evidence, not the situations').
* Where other "... ne ?" questions were answered otherwise ("bugün ne yedin ?", "en sevdiğin renk ne ?", "ne
  yapıyorsun ?"), "keyfin ne ?" (a word never met) is not answered: the situations disagree, so nothing is corroborated.
  In a narrower life, where every "... ne ?" it ever heard was the name question, the same question is answered with
  the name: that is the induction from what it lived, not a slip of the reading.
* The switch is off by default.
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import Config, Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS
from zihin.read import _member_instances, generalisations, situation_evidence


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fresh(corroborate: bool, others: bool = True) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path, cfg.corroborate = False, None, corroborate
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    for _ in range(3):
        s.watch("adın ne ?", "adım cevahir .")
    s.watch("senin adın ne ?", "adım cevahir .")
    if others:                                            # other questions ending in "ne ?", answered otherwise
        for _ in range(2):
            s.watch("bugün ne yedin ?", "makarna yedim .")
            s.watch("en sevdiğin renk ne ?", "en sevdiğim renk mavi .")
            s.watch("ne yapıyorsun ?", "kitap okuyorum .")
    return s


def test_the_switch_is_off_by_default():
    assert Config().corroborate is False and Config.changed().corroborate is False


@pytest.mark.parametrize("corroborate, answer", [(True, "adım cevahir"), (False, "(sessiz)")])
def test_independent_readings_that_agree_add_up(corroborate, answer):
    assert fresh(corroborate).reply("peki adın ne ?") == answer


@pytest.mark.parametrize("corroborate", [True, False])
def test_a_word_never_met_is_not_answered(corroborate):
    assert fresh(corroborate).reply("keyfin ne ?") == "(sessiz)"


def test_a_narrow_life_generalises_what_it_lived():
    assert fresh(True, others=False).reply("keyfin ne ?") == "adım cevahir"
    assert fresh(False, others=False).reply("keyfin ne ?") == "(sessiz)"


def test_the_situations_leave_out_what_the_frame_counted():
    s = fresh(True)
    s.reply("peki adın ne ?")
    mind = s.mind
    end = mind.unit_of(chat.TURN, chat.TURN_END)
    context = mind.S.tuple_index[tuple(mind.unit_of(CHANNEL, w) for w in ("peki", "adın", "ne")) + (end,)]
    frames = [g for g, _ in generalisations(mind, context) if _member_instances(mind, g, context)]
    assert frames
    used = _member_instances(mind, frames[0], context)
    senin = mind.S.tuple_index[tuple(mind.unit_of(CHANNEL, w) for w in ("senin", "adın", "ne")) + (end,)]
    assert senin in used
    whole = situation_evidence(mind, context)
    left = situation_evidence(mind, context, leave_out=frozenset(used))
    assert left[0] > 0.0 and left != whole or senin not in used
