# -*- coding: utf-8 -*-
"""Zihin, sleep: dreams kept with what they claim, sorted against what is real.

What is pinned here
-------------------
* A dream claims what a lived sequence would teach between words: each word followed by the next. Its new part is
  the claims nothing real had touched when it was dreamt.
* A dream that only recalls what was lived has no new claim ("biliniyordu").
* A dream whose new claims life later backs is found true; one whose new claim life later contradicts is found
  false; one life never touches stays open.
* Sorting weighs imagined against real evidence: it moves the trust in imagination, judges each imagined amount
  once (a second sorting with nothing newly imagined changes nothing), and deletes nothing.
* A night keeps its dreams in the mind, counts itself, and reports an earlier dream once when life has judged it;
  the dreams survive saving and loading.
"""
from __future__ import annotations

import torch

from zihin import IMAGINED, TOLD, Config, Event, Mind
from zihin.__main__ import CHANNEL, Session, words
from zihin.dusunce import remember_dream, sleep, sort_dreams, verdict
from zihin.store import W_NEG, W_POS

TEXT = "research/living_mind_2026_11_14/data/imagination_text.txt"


def new_session() -> Session:
    torch.manual_seed(0)
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    session.read_path(TEXT)
    session.read_path(TEXT)
    return session


def dream(session: Session, text: str) -> dict:
    """Imagine a sentence (lived as imagined) and keep it as a dream."""
    mind = session.mind
    ws = words(text)
    for w in ws:
        mind.step([Event(CHANNEL, w, IMAGINED)])
    mind.step([])
    return remember_dream(mind, ws[0], "", ws, None)


def test_a_recalled_sentence_is_no_new_claim():
    session = new_session()
    record = dream(session, "su ateşi söndürür")
    assert all(wp + wn > 0 for _, _, _, wp, wn in record["claims"])
    assert verdict(session.mind, record)[0] == "biliniyordu"


def test_life_backs_one_dream_contradicts_another_and_leaves_a_third_open():
    session = new_session()
    backed = dream(session, "yağmur ateşi söndürür")
    contradicted = dream(session, "güneş ateşi söndürür")
    untouched = dream(session, "soba bitkileri büyütür")
    for record in (backed, contradicted, untouched):
        assert verdict(session.mind, record)[0] == "acik"   # nothing real has met the new claims yet
    session.live("yağmur ateşi söndürür .", TOLD)
    session.live("güneş evi ısıtır .", TOLD)
    session.live("güneş evi ısıtır .", TOLD)
    assert verdict(session.mind, backed)[0] == "dogru"
    assert verdict(session.mind, contradicted)[0] == "yanlis"
    assert verdict(session.mind, untouched)[0] == "acik"


def test_sorting_judges_each_imagined_amount_once_and_deletes_nothing():
    session = new_session()
    mind = session.mind
    dream(session, "yağmur evi ısıtır")
    imagined = (mind.S.t_w[W_POS, :mind.S.n_ties, IMAGINED].sum(), mind.S.t_w[W_NEG, :mind.S.n_ties, IMAGINED].sum())
    first = sort_dreams(mind)
    assert first["true"] + first["false"] > 0
    assert first["trust"] != first.get("trust_before", None)
    second = sort_dreams(mind)
    assert second["true"] == 0 and second["false"] == 0 and second["trust"] == first["trust"]
    after = (mind.S.t_w[W_POS, :mind.S.n_ties, IMAGINED].sum(), mind.S.t_w[W_NEG, :mind.S.n_ties, IMAGINED].sum())
    assert torch.equal(after[0], imagined[0]) and torch.equal(after[1], imagined[1])


def test_a_night_keeps_dreams_counts_itself_and_reports_a_judged_dream_once(tmp_path):
    from zihin.persist import load, save
    session = new_session()
    mind = session.mind
    record = dream(session, "güneş ateşi söndürür")
    first = sleep(session, 2)
    assert mind.nights == 1 and len(mind.dreams) >= 1 + len(first["dreams"])
    assert 0.0 <= first["sorting"]["trust"] <= 1.0
    session.live("güneş evi ısıtır .", TOLD)
    session.live("güneş evi ısıtır .", TOLD)
    second = sleep(session, 1)
    assert any(v == "yanlis" and d is record for v, d in second["judged"])
    third = sleep(session, 1)
    assert not any(d is record for _, d in third["judged"])          # told once
    save(mind, str(tmp_path / "uyuyan.zihin"))
    again = load(str(tmp_path / "uyuyan.zihin"), ledger=False)
    assert again.nights == 3 and [d["words"] for d in again.dreams] == [d["words"] for d in mind.dreams]
