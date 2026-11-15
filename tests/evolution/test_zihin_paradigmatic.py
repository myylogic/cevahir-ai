# -*- coding: utf-8 -*-
"""Zihin, F31 Paradigmatic Relation and F32 Relational Binding.

What is pinned here
-------------------
* F31, hand-checkable: read "kedi uyur ." and "köpek uyur ." (imitated): the two fill two lived places of two members
  each, so köpek stands where kedi stands with W+ = 2/9 (as F29 counts it), and köpek is kedi's stand-in. The rival
  fillers of a frame's own place are never what the relation names: with the place kedi and köpek fill as the frame's
  category, kedi has no stand-in. When "kuş uyur ." is read too, köpek and kuş stand where kedi stands alike (1/9 each)
  and no single item is the stand-in.
* F31 + F32 in a small life: synonyms are read filling the same places ("kırmızı araba yolda gider ." / "kırmızı
  otomobil yolda gider ." ...), greetings are watched answered alike, and three definitions are watched ("otomobil ne
  demek ?" -> "otomobil , araba demektir ."). Asked "selam ne demek ?", the mind answers "selam , merhaba demektir":
  the members of the definition frame agree that the word after the comma stands where the asked word stands (F31),
  and the word after that is read through each member's own pair (F32). With F31 alone it says "selam , merhaba" and
  goes on as merhaba goes on; without it, "selam ," and silence. A word never met has no stand-in.
* Reading writes nothing, and both switches are off by default.
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Config, Mind
from zihin.__main__ import CHANNEL, Session, VARIANTS
from zihin.form import categories_of
from zihin.read import bound_frames, members_map, stand_in, stand_ins


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fresh(paradigmatic: bool, bind: bool) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path, cfg.paradigmatic, cfg.bind_relations = False, None, paradigmatic, bind
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    return Session(mind)


def test_the_switches_are_off_by_default():
    assert Config().paradigmatic is False and Config.changed().paradigmatic is False
    assert Config().bind_relations is False and Config.changed().bind_relations is False


def test_the_stand_in_is_read_from_shared_places():
    s = fresh(True, True)
    for t in ("kedi uyur .", "köpek uyur .", "kuş uçar ."):
        s.live(t, IMITATED)
    mind = s.mind
    kedi, kopek = mind.unit_of(CHANNEL, "kedi"), mind.unit_of(CHANNEL, "köpek")
    mind.episode_start = mind.t + 1                      # read as if a new utterance began now
    found = stand_ins(mind, kedi, mind.episode_start)
    assert set(found) == {kopek}
    assert found[kopek] == pytest.approx((2 / 9, 0.0), abs=1e-12)
    assert stand_in(mind, kedi) == kopek
    place = next(g for g, _ in categories_of(mind, kopek) if g in {h for h, _ in categories_of(mind, kedi)})
    assert stand_in(mind, kedi, category=place) is None  # a rival filler of the frame's place is not its relation
    s.live("kuş uyur .", IMITATED)
    mind.episode_start = mind.t + 1
    found = stand_ins(mind, kedi, mind.episode_start)
    kus = mind.unit_of(CHANNEL, "kuş")
    assert found[kopek] == pytest.approx((1 / 9, 0.0), abs=1e-12) and found[kus] == pytest.approx((1 / 9, 0.0), abs=1e-12)
    assert stand_in(mind, kedi) is None                  # two items stand there alike: no single stand-in


def defined(paradigmatic: bool, bind: bool) -> Session:
    s = fresh(paradigmatic, bind)
    for a, b, frames in (("araba", "otomobil", ("kırmızı {} yolda gider .", "babam yeni bir {} aldı .")),
                         ("doktor", "hekim", ("{} ilaç yazdı .", "annem {} oldu .")),
                         ("cevap", "yanıt", ("bu soruya {} ver .", "{} doğru çıktı ."))):
        for f in frames:
            s.live(f.format(a), IMITATED)
            s.live(f.format(b), IMITATED)
    for _ in range(2):
        for g in ("merhaba", "selam"):
            s.watch(g, g + " , hoş geldin .")
            s.watch(g, g + " , nasılsın ?")
    for q, a in (("otomobil ne demek ?", "otomobil , araba demektir ."), ("hekim ne demek ?", "hekim , doktor demektir ."),
                 ("yanıt ne demek ?", "yanıt , cevap demektir .")):
        s.watch(q, a)
    return s


@pytest.mark.parametrize("paradigmatic, bind, answer", [(True, True, "selam , merhaba demektir"),
                                                       (True, False, "selam , merhaba ,"),
                                                       (False, True, "selam ,"), (False, False, "selam ,")])
def test_the_answer_stands_where_the_asked_word_stands(paradigmatic, bind, answer):
    assert defined(paradigmatic, bind).reply("selam ne demek ?") == answer


def test_the_relation_is_read_for_each_word():
    assert defined(True, True).reply("merhaba ne demek ?") == "merhaba , selam demektir"
    assert defined(True, True).reply("otomobil ne demek ?") == "otomobil , araba demektir"   # lived as a whole


def test_a_word_never_met_has_no_stand_in():
    assert defined(True, True).reply("keyif ne demek ?") == "keyif ,"


def test_reading_writes_nothing():
    s = defined(True, True)
    s.reply("selam ne demek ?")                          # the contexts of the question and the answer now exist
    mind = s.mind
    end = mind.unit_of(chat.TURN, chat.TURN_END)
    words = [mind.unit_of(CHANNEL, w) for w in ("selam", "ne", "demek")]
    said = [mind.unit_of(CHANNEL, w) for w in ("selam", ",", "merhaba")]
    context = mind.S.tuple_index[tuple(words) + (end,) + tuple(said)]
    digest = mind.S.digest()
    assert bound_frames(mind, context)                   # the prefix's frame, bound to merhaba by ALIKE
    read = members_map(mind, context)
    assert mind.S.digest() == digest
    assert read[mind.unit_of(CHANNEL, "demektir")] > 0.5 + 1.0 / 12
