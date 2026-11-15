# -*- coding: utf-8 -*-
"""Zihin, dreams of a small mind.

What is pinned here
-------------------
* Sleep leaves out the frequent words (they carry the language, not a topic), cut to the vocabulary: the most used
  tenth of it, at most 200. A mind taught a dozen dialogues dreams; with the absolute 200 every word it knew was
  frequent and no topic was left.
* From 2000 words on the cut is the absolute 200 with the same words, so larger minds dream as before.
* The day is the session's last 12 history entries (a line, a chat, a dialogue watched, a whole file read or taught),
  and its words are chosen by the mind's own evidence: how often a word was lived in them times
  tau(u) = 1 / (1 + the number of composed contexts holding it). The same measure for every kind of entry, with no
  word-shape rule: a two-letter word can be what the day was about. A document's references at its end are not its
  day (the day residue took a read file's last lines before).
* A topic the mind can think nothing after gives no dream and the next one is tried, so a small taught mind still
  dreams when the words most characteristic of its day are the last words of sentences.
"""
from __future__ import annotations

import torch

from zihin import TOLD, Config, Mind
from zihin.__main__ import CHANNEL, Session
from zihin.dusunce import day_residue, frequent_cut, frequent_words, sleep
from zihin.store import PART

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


def ranking(session, entries: int = 12) -> list:
    """The day's words by count in the last entries times tau, computed here from the entries and the substance."""
    counts = {}
    for entry in reversed(session.history[-entries:]):
        for w, c in entry.counts.items():
            counts[w] = counts.get(w, 0) + c
    S = session.mind.S

    def weight(w):
        return counts[w] / (1 + len(S.inc.get((session.mind.unit_of(CHANNEL, w), PART), ())))
    return sorted(counts, key=lambda w: -weight(w))


def test_the_day_is_the_words_most_characteristic_of_the_recent_entries(tmp_path):
    (tmp_path / "gezi.txt").write_text("Penguenler buzulların kenarında yaşar. Fokbalıkları kayaların üstünde "
                                       "uyur.", encoding="utf-8")
    session = small()
    session.read_path(str(tmp_path / "gezi.txt"))
    entry = session.history[-1]
    assert entry.kind == "read" and entry.counts["penguenler"] == 1 and entry.heard.startswith("Fokbalıkları")
    residue = day_residue(session, 3)
    assert residue == ranking(session)[:3]
    assert set(residue) <= {"penguenler", "buzulların", "kenarında", "yaşar", "fokbalıkları", "kayaların", "üstünde",
                            "uyur"}
    typed = small()
    for i in range(15):
        typed.live("cümle numara %d burada ." % i, TOLD)
    assert day_residue(typed, 40) == ranking(typed)        # every word of the day, by the same measure ...
    assert not {"0", "1", "2"} & set(day_residue(typed, 40))   # ... and only of the last 12 entries


REFERENCES = ("Penguenler soğuk denizlerin kıyısında yaşar. İmparator penguenler buzun üstünde yaşar. Penguenler "
              "uçamaz ama suda çok hızlı yüzer. Penguen yavruları gri tüylerle yüzer. Penguenler balık ve kril yer. "
              "Kaynakça\n[1] Smith, J. (2019). Antarctic birds. Oxford: Polar Press.\n[2] Yılmaz, A. (2020). Kutup "
              "hayvanları. İstanbul: Doğa Yayınları.\n")


def test_a_documents_references_are_not_its_day(tmp_path):
    (tmp_path / "penguen.txt").write_text(REFERENCES, encoding="utf-8")
    session = small()
    session.read_path(str(tmp_path / "penguen.txt"))
    assert session.history[-1].heard.startswith("İstanbul")            # its last line is a reference ...
    body = set(session.words(REFERENCES[:REFERENCES.index("Kaynakça")]))
    residue = day_residue(session, 3)
    assert residue and set(residue) <= body                          # ... its day is what the text was about
    assert residue[:2] == ["yaşar", "yüzer"]                         # said twice, each time where few contexts are


def test_no_word_shape_rule_decides_what_the_day_was_about(tmp_path):
    (tmp_path / "su.txt").write_text("Bu soğuk su. Bu tuzlu su. Bu temiz su. Kediler balık yer.", encoding="utf-8")
    session = small()
    session.read_path(str(tmp_path / "su.txt"))
    assert day_residue(session, 1) == ["su"]                         # two letters: never a topic by the old rule
