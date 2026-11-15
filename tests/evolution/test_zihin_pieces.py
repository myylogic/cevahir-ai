# -*- coding: utf-8 -*-
"""Zihin word pieces from the repository's own tokenizer (tokenizer_management).

What is pinned here
-------------------
* A word's pieces are its root (when it differs from the word), the suffixes the morphology splits off and its last
  syllable; words built alike share pieces ("yapıyorsun" / "gidiyorsun": "ek:sun"; "yapıyorsun" / "yaptım": the root).
* Short words and punctuation have no pieces.
* Pieces are opt-in for a new mind (raised with them it grew about five times as many ties per moment); asked for,
  they accompany each word as facets; a mind raised with them keeps them; a mind raised without them keeps living
  without them.
"""
from __future__ import annotations

from zihin import Config, Mind
from zihin.__main__ import CHANNEL, FACETS, Session, pieces_for
from zihin.parca import own_pieces


def speaking_mind() -> Mind:
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True, situations=True))
    mind.add_channel(CHANNEL, "act")
    return mind


def test_words_built_alike_share_pieces():
    pieces = own_pieces()
    a, b, c = pieces.tokenize("yapıyorsun"), pieces.tokenize("gidiyorsun"), pieces.tokenize("yaptım")
    assert "ek:sun" in a and "ek:sun" in b
    assert "kök:yap" in a and "kök:yap" in c
    assert pieces.tokenize("ve") == [] and pieces.tokenize("?") == []


def test_pieces_are_opt_in_for_a_new_mind_and_kept_once_lived():
    mind = speaking_mind()
    assert pieces_for(mind) is None                        # opt-in: no pieces unless asked
    pieces = pieces_for(mind, "cevahir")
    assert pieces is not None
    session = Session(mind, pieces)
    session.watch("ne yapıyorsun ?", "kitap okuyorum .")
    assert mind.ch.sizes()[mind.ch.id(FACETS)] > 0
    assert pieces_for(mind) is not None                    # raised with pieces: keeps them


def test_a_mind_raised_without_pieces_keeps_living_without_them():
    mind = speaking_mind()
    session = Session(mind, None)
    session.watch("ne yapıyorsun ?", "kitap okuyorum .")
    assert pieces_for(mind) is None
    assert pieces_for(mind, "cevahir") is not None          # unless asked for
