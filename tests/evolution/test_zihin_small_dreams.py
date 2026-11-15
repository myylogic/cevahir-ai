# -*- coding: utf-8 -*-
"""Zihin, dreams of a small mind, and the one measure by which the program chooses topics.

What is pinned here
-------------------
* Every topic the program chooses (what to think about, what the day was about, what to dream, what to look up) is
  chosen by one measure of the mind's own evidence: topic_score(u) = tau(u) x D(u), tau = 1 / (1 + the number of
  composed contexts holding u) (how telling it is) and D(u) = use x (1 - conviction) over its lag-1 ties (how little
  is known of what follows it, read.drive_of). No frequency cut-off decides it (the program had five that disagreed:
  the most used 200 words, a tenth of the vocabulary, a fifth of it, ...) and no word-shape rule (alphabetic, at
  least 4 letters). A word nothing ever followed scores 0 and is never offered.
* A mind taught a dozen dialogues dreams (under the absolute cut of 200 frequent words it had no topic at all).
* What the mind most wants to know (wanted_topics) is every word it holds ranked by that measure; the most used words
  are not left out by rule.
* The day is the session's last 12 history entries (a line, a chat, a dialogue watched, a whole file read or taught):
  how often each word was lived in them times its topic_score. What a document's day was about is its topic, not the
  references at its end (the day residue took a read file's last lines before); a two-letter word can be what the day
  was about.
"""
from __future__ import annotations

import torch

from zihin import TOLD, Config, Mind
from zihin.__main__ import CHANNEL, Session
from zihin.dusunce import day_residue, sleep, topic_score
from zihin.read import drive_of
from zihin.store import ATOMIC, PART
from zihin.web import wanted_topics

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


def score(mind, w: str) -> float:
    """tau x D computed here from the substance (the measure, independently of dusunce)."""
    unit = mind.unit_of(CHANNEL, w)
    if unit < 0:
        return 0.0
    return (1.0 / (1 + len(mind.S.inc.get((unit, PART), ())))) * drive_of(mind, unit)


def taught(tmp_path) -> Session:
    (tmp_path / "diyaloglar.tsv").write_text(DIALOGUES, encoding="utf-8")
    session = small()
    session.teach(str(tmp_path / "diyaloglar.tsv"))
    return session


def test_a_small_taught_mind_dreams(tmp_path):
    session = taught(tmp_path)
    residue = day_residue(session, 3)
    assert residue and all(score(session.mind, w) > 0 for w in residue)
    night = sleep(session)
    assert night["dreams"] and all(chain for chain in night["dreams"])


def test_what_it_wants_to_know_is_every_word_it_holds_by_the_one_measure(tmp_path):
    session = taught(tmp_path)
    mind, S = session.mind, session.mind.S
    cid = mind.ch.id(CHANNEL)
    held = [str(mind.ch.symbol(cid, int(S.u_symbol[u]))) for u in range(S.n_units)
            if int(S.u_kind[u]) & ATOMIC and int(S.u_channel[u]) == cid and S.u_alive[u]]
    ranked = sorted((w for w in held if score(mind, w) > 0), key=lambda w: -score(mind, w))
    assert wanted_topics(mind, CHANNEL, len(held)) == ranked
    assert all(topic_score(mind, mind.unit_of(CHANNEL, w)) == score(mind, w) for w in held)
    most_used = max(held, key=lambda w: float(S.u_use[mind.unit_of(CHANNEL, w)]))
    assert (most_used in ranked) == (score(mind, most_used) > 0)   # not left out for being used most


def ranking(session, entries: int = 12) -> list:
    """The day's words by count in the last entries times tau x D, computed here from the entries and the substance."""
    counts = {}
    for entry in reversed(session.history[-entries:]):
        for w, c in entry.counts.items():
            counts[w] = counts.get(w, 0) + c
    weights = {w: c * score(session.mind, w) for w, c in counts.items()}
    return sorted((w for w in counts if weights[w] > 0), key=lambda w: -weights[w])


def test_the_day_is_the_words_most_characteristic_of_the_recent_entries(tmp_path):
    (tmp_path / "gezi.txt").write_text("Penguenler buzulların kenarında yaşar. Fokbalıkları kayaların üstünde "
                                       "uyur. Penguenler suda yüzer.", encoding="utf-8")
    session = small()
    session.read_path(str(tmp_path / "gezi.txt"))
    entry = session.history[-1]
    assert entry.kind == "read" and entry.counts["penguenler"] == 2 and entry.heard.startswith("Penguenler suda")
    residue = day_residue(session, 3)
    assert residue and residue == ranking(session)[:3]
    assert "penguenler" in residue                       # what the text is about: telling, and followed by
    typed = small()                                      # different words each time
    for _ in range(3):
        typed.live("kayık suda yüzer .", TOLD)
    for i in range(12):
        typed.live("cümle numara %d burada ." % i, TOLD)
    assert day_residue(typed, 40) == ranking(typed)        # every word of the day, by the same measure ...
    assert score(typed.mind, "kayık") > 0 and "kayık" not in day_residue(typed, 40)   # ... of the last 12 entries


REFERENCES = ("Penguenler soğuk denizlerin kıyısında yaşar. İmparator penguenler buzun üstünde yaşar. Penguenler "
              "uçamaz ama suda çok hızlı yüzer. Penguen yavruları gri tüylerle yüzer. Penguenler balık ve kril yer. "
              "Kaynakça\n[1] Smith, J. (2019). Antarctic birds. Oxford: Polar Press.\n[2] Yılmaz, A. (2020). Kutup "
              "hayvanları. İstanbul: Doğa Yayınları.\n")


def test_a_documents_references_are_not_its_day(tmp_path):
    (tmp_path / "penguen.txt").write_text(REFERENCES, encoding="utf-8")
    session = small()
    session.read_path(str(tmp_path / "penguen.txt"))
    assert session.history[-1].heard.startswith("İstanbul")            # its last line is a reference ...
    assert day_residue(session, 3) == ranking(session)[:3]
    assert day_residue(session, 1) == ["penguenler"]                 # ... what its day was about is its topic


def test_no_word_shape_rule_decides_what_the_day_was_about(tmp_path):
    (tmp_path / "su.txt").write_text("Su soğuk. Su akar. Su içilir. Kediler balık yer.", encoding="utf-8")
    session = small()
    session.read_path(str(tmp_path / "su.txt"))
    assert day_residue(session, 1) == ["su"]                         # two letters: never a topic by the old rule
