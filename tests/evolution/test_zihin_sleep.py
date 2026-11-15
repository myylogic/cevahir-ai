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
* A dream is a chain of the mind's own inner voice: only the topic is brought to mind, every other word is what the
  mind decided (lived as imagined); the next topic is the best topic of what it thought, and the chain ends when the
  mind's race falls silent (a count of 3 links decided it before; the cap is only a safety limit). What it saw with a
  word is seen again, alone in its moment and taken in without a decision, then the word.
* No word is related to another by a shared root: the inference by hand-made Turkish morphology is gone from dreams,
  and so is the analogy that replayed a thought about X on its structural counterpart (both wrote words into the
  mind's inner voice and discarded what it decided meanwhile).
"""
from __future__ import annotations

import torch

from zihin import ACTED, IMAGINED, TOLD, Config, Event, Mind
from zihin.__main__ import CHANNEL, Session, words
from zihin.dusunce import remember_dream, sleep, sort_dreams, verdict
from zihin.store import W_NEG, W_POS

TEXT = "research/living_mind_2026_11_14/data/imagination_text.txt"   # the F20 imagination text


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


def test_a_dream_is_a_chain_and_sees_what_was_seen_with_its_words(tmp_path):
    import pytest
    Image = pytest.importorskip("PIL.Image")             # seeing needs Pillow (optional dependency)
    from zihin.gorme import EYE, features
    from zihin.dusunce import dream as dream_chain, seen_with
    session = new_session()
    picture = tmp_path / "ben.png"
    img = Image.new("RGB", (64, 64), (240, 240, 240))
    for x in range(16, 48):
        for y in range(8, 56):
            img.putpixel((x, y), (40, 60, 160))
    img.save(picture)
    for _ in range(2):
        session.view = features(str(picture))
        session.live("güneş evi ısıtır .", TOLD)
    gist = features(str(picture))[0]
    assert seen_with(session.mind, "güneş") == [gist]
    chain, seen = moments(session, lambda: dream_chain(session, "güneş"))
    assert chain and chain[0]["seen"] == [gist]
    first = next(i for i, events in enumerate(seen) if any(e.channel == EYE for e in events))
    assert seen[first] == [Event(EYE, gist, IMAGINED)]            # the image first, alone in its moment ...
    assert [(e.symbol, e.prov) for e in seen[first + 1] if e.channel == CHANNEL] == [("güneş", IMAGINED)]   # then
    assert not [e for events in seen for e in events if e.channel == CHANNEL and e.prov == ACTED]   # the word
    eye = session.mind.ch.id(EYE)
    unit = session.mind.S.atom_index[(eye, session.mind.ch.symbol_index(eye, gist, create=False))]
    assert float(session.mind.S.t_w[W_POS, :session.mind.S.n_ties, IMAGINED].sum()) > 0.0
    assert unit >= 0


def moments(session, run):
    """(what ``run`` returned, the events each moment perceived: what was pending from the last moment (feelings left
    out), then the new events)."""
    mind, seen, step = session.mind, [], session.mind.step

    def recorded(events=()):
        seen.append([e for e in list(mind.pending) + list(events) if e.channel != "feel"])
        return step(events)
    mind.step = recorded
    try:
        out = run()
    finally:
        mind.step = step
    return out, seen


def test_a_dream_is_the_minds_own_inner_voice_and_ends_when_its_race_falls_silent():
    from zihin.dusunce import DREAM_CAP, dream as dream_chain, think, topic_of
    session = new_session()
    chain, seen = moments(session, lambda: dream_chain(session, "güneş"))
    assert 1 <= len(chain) < DREAM_CAP                   # ended by the mind, not by a count
    imagined = [str(e.symbol) for events in seen for e in events if e.channel == CHANNEL and e.prov == IMAGINED]
    kept = [w for d in chain for w in d["words"]]        # each imagining kept: the topic brought to mind, then what
    assert imagined[:len(kept)] == kept                  # it decided; the chain ends when its race falls silent: the
    silent = imagined[len(kept):]                        # last topic was brought to mind and nothing came after it
    assert len(silent) <= 1 and (not silent or not think(session, silent[0])[0])
    assert not [e for events in seen for e in events if e.channel == CHANNEL and e.prov != IMAGINED]
    assert all(len([e for e in events if e.channel == CHANNEL]) <= 1 for events in seen)
    dreamt = {d["about"] for d in chain}
    following = topic_of(session, " ".join(w for w in chain[-1]["words"][1:] if w not in dreamt))
    assert not following or not think(session, following)[0]    # what would come next: nothing, or a silent race
    assert len(dream_chain(session, "güneş", cap=1)) == 1        # the cap is only a safety limit


def test_no_word_is_related_to_another_by_a_shared_root():
    """The inference by a shared Turkish root (hand-made morphology facets or the tokenizer's morphology run over the
    vocabulary: 'hayvandır' and 'hayvanlar' made mates, and what followed one carried over to the topic) is gone."""
    import zihin.dusunce as dusunce
    from zihin.dusunce import dream as dream_chain
    for name in ("root_mates", "_stems", "infer"):
        assert not hasattr(dusunce, name)
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    for _ in range(3):
        session.live("kedi bir hayvandır .", TOLD)
        session.live("hayvanlar nefes alır ve büyür .", TOLD)
    mind.step([])
    chain = dream_chain(session, "kedi")
    assert chain and all(d.get("kind") != "çıkarım" for d in chain)
    assert not any(d["about"] == "kedi" and "nefes" in d["words"] for d in chain)   # nothing carried over by a root
