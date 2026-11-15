# -*- coding: utf-8 -*-
"""Zihin, dreams of a small mind.

What is pinned here
-------------------
* Sleep leaves out the frequent words (they carry the language, not a topic), cut to the vocabulary: the most used
  tenth of it, at most 200. A mind taught a dozen dialogues dreams; with the absolute 200 every word it knew was
  frequent and no topic was left.
* From 2000 words on the cut is the absolute 200 with the same words, so larger minds dream as before.
* What was read, taught or watched in the session is the day: its last lines are the day's utterances where the day
  residue looks for topics. A session of typed lines has the same day as before.
"""
from __future__ import annotations

import torch

from zihin import TOLD, Config, Mind
from zihin.__main__ import CHANNEL, Session
from zihin.dusunce import day_residue, frequent_cut, frequent_words, sleep

DIALOGUES = ("Merhaba!\tMerhaba, nasılsın?\nNasılsın?\tİyiyim, teşekkür ederim. Sen nasılsın?\n"
             "Adın ne?\tBenim adım Cevahir.\nKedi ne yer?\tKedi balık yer.\nKöpek ne yer?\tKöpek kemik yer.\n"
             "Gökyüzü ne renk?\tGökyüzü mavidir.\nÇimen ne renk?\tÇimen yeşildir.\n"
             "Ankara nerede?\tAnkara Türkiye'nin başkentidir.\nİstanbul nerede?\tİstanbul Türkiye'nin en büyük "
             "şehridir.\nSu neden önemli?\tSu yaşam için gereklidir.\nKitap okumayı sever misin?\tEvet, kitap "
             "okumayı severim.\nGünaydın!\tGünaydın, iyi sabahlar.\nGörüşürüz!\tHoşça kal, görüşürüz.\n")


def small() -> Session:
    torch.manual_seed(0)
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True, situations=True, explore=True))
    mind.add_channel(CHANNEL, "act")
    return Session(mind)


def test_a_small_taught_mind_dreams(tmp_path):
    (tmp_path / "diyaloglar.tsv").write_text(DIALOGUES, encoding="utf-8")
    session = small()
    session.teach(str(tmp_path / "diyaloglar.tsv"))
    everything = frequent_words(session.mind)            # the absolute 200: every word this mind knows
    assert not [w for w in day_residue(session, 3) if w not in everything]
    assert len(frequent_words(session.mind, relative=True)) < len(everything) // 5
    night = sleep(session)
    assert night["dreams"] and all(chain for chain in night["dreams"])


def with_vocabulary(n: int) -> Mind:
    """A mind that knows n words, each used a (seeded) random number of times."""
    mind = Mind(Config.changed(support="members"))
    cid = mind.add_channel(CHANNEL, "act")
    mind.S.begin(1, mind.ch.sizes())
    units = [mind._atom(cid, "k%d" % i) for i in range(n)]
    mind.S.end()
    g = torch.Generator().manual_seed(n)
    mind.S.u_use[torch.tensor(units)] = torch.randint(1, 60, (n,), generator=g).to(mind.S.u_use.dtype)
    return mind


def test_from_two_thousand_words_on_the_cut_is_the_absolute_two_hundred():
    assert [frequent_cut(v) for v in (0, 51, 104, 737, 1990, 1991, 2000, 4000, 50_000)] == \
        [0, 6, 11, 74, 199, 200, 200, 200, 200]
    for n in (2000, 2600):
        mind = with_vocabulary(n)
        assert frequent_words(mind, relative=True) == frequent_words(mind)
    mind = with_vocabulary(1500)
    cut = frequent_words(mind, relative=True)
    assert len(cut) == 150 and cut < frequent_words(mind)      # the most used of the same ranking


def test_what_was_read_is_the_day_and_typed_lines_are_the_day_as_before(tmp_path):
    (tmp_path / "gezi.txt").write_text("Penguenler buzulların kenarında yaşar. Fokbalıkları kayaların üstünde "
                                       "uyur.", encoding="utf-8")
    session = small()
    session.read_path(str(tmp_path / "gezi.txt"))
    assert [text for _, text, _ in session.history[-12:]] == ["[okudu] gezi.txt"]   # what the residue saw before
    residue = day_residue(session, 3)
    assert residue and set(residue) <= {"penguenler", "buzulların", "kenarında", "yaşar", "fokbalıkları",
                                        "kayaların", "üstünde", "uyur"}
    typed = small()
    for i in range(15):
        typed.live("cümle numara %d burada ." % i, TOLD)
    assert typed.recent_utterances(12) == [text for _, text, _ in typed.history[-12:]]
