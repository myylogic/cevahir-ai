# -*- coding: utf-8 -*-
"""Zihin, F39 Replacement Witness (Mill's method of difference): when nothing would be said, a lived situation that
differs from the present one by one replacement is a witness of it as far as the lived pairs that differ by the same
replacement went on alike.

What is pinned here
-------------------
* In T-cause's world (clouds the common cause of the umbrella advice and of the rain), a sunny day with the umbrella
  advice was never lived. Without F39 nothing expects the fine weather there (0.5); with it the fine weather is
  expected over 1/2 and leads the rain the words alone lead to, still under the speaking line (it leans, it does not
  assert). The probes the mind already answers from what it lived are left as they were.
* The switch is off by default.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

from zihin import Config
from zihin.__main__ import VARIANTS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "research", "living_mind_2026_11_15"))


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_the_switch_is_off_by_default():
    assert Config().difference is False and Config.changed().difference is False
    assert VARIANTS["speaking"]().difference is False


def test_the_cause_leads_where_the_words_led_to_the_spurious_answer():
    import t_cause_context as C
    line = 0.5 + Config().theta / (2.0 * Config().T)
    without = C.run(0, {})
    with_f39 = C.run(0, {"difference": True})
    probe = "güneşli / şemsiye al ."
    assert without["observed"][probe]["hava"] == 0.5
    seen = with_f39["observed"][probe]
    assert seen["hava"] > 0.5 and seen["hava"] > seen["yağmur"] and seen["hava"] < line
    assert seen["yağmur"] == without["observed"][probe]["yağmur"]
    for lived in ("bulutlu / şemsiye al .", "güneşli / iyi eğlenceler ."):     # answered from what it lived
        assert with_f39["observed"][lived] == without["observed"][lived]
