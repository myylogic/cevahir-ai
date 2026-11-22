# -*- coding: utf-8 -*-
"""Zihin, F49 Fault Tracing: the mind finds where its own computation broke and learns how far each of its ways of
reading deserves to be believed (zihin/hata.py).

What is pinned here
-------------------
* The switch is off by default and is one of the studied mechanisms (``--ac faults``). On, and with nothing settled
  yet, the mind answers as it does with the switch off and nothing is written into its substance: the claims of
  its answers are only kept, with the contexts they were made in.
* A small world: eight things watched being asked two questions. In one (V) the eight happen to share the answer;
  in the other (L) every thing has the same answer. Asked about a new thing the mind generalises in both.
* The chain of an answer: each word, the way that claimed it (the frame and the kind of its claim), the context.
  When the true dialogue is then shown, the words before the break are confirmed (the asked word said back: the
  relation held), the word at the break is settled as failed on the frame that claimed it, and the fault is kept:
  where, what it said, what was lived, which way.
* The record is read back with the core's rule (``hata.read_as``): nothing settled leaves a claim as it is; a way
  that held more often than it claimed is believed more; one that held less often than not falls under 1/2.
* Local repair of the structure: after its failures in V the mind says nothing of the next new thing there (with
  the switch off it goes on asserting), while it still says the asked word back.
* A principle: a frame never tried has only the vote of the frames like it (each frame one vote). Where those
  failed, the mind weighs holding its claim back against saying it once to find out, by how far what it held back
  was then answered (``hata.answered``): shown what it did not say, it holds an untested frame of that kind back
  (also a lawful one, which speaks again once it has been shown to hold); only told "good" or "bad", it tries it.
  The research setting "vote" always holds back.
* A value signal alone ("good" / "bad"): good settles every claim the mind said as held; bad blames the link the
  mind was least sure of most.
* A settled claim is evidence of its moment: undoing the moment takes it back and the claim is kept again.
* What is kept survives saving and loading; a mind saved before F49 loads with nothing kept.
* ``hata.report`` names the ways and the last faults.
* Restructuring: where what a thing eats follows a kind it was read to be of, and the frame's own vote was found
  wanting, the frame is read through the narrower place the thing shares with some of the members (the things read
  to be of its kind): a new thing of each kind is then answered right, where the mind with that step left out says
  nothing after the asked word and the mind with the switch off answers with something else it read. The chat's
  ``!neden`` names that way.
* Silent claims (``Config.silent_claims``, off by default, studied): a mind that is never asked, only reads and
  watches, settles what its ways claimed of what it was about to hear. Without the switch nothing is ever settled
  in such a life; with it the ways have records of their own, none of them counts as a claim held back, and in
  the world of kinds the mind that only watched answers a new thing of each kind right, where the mind without it
  answers with something else it read.
"""
from __future__ import annotations

import pickle

import pytest
import torch

import zihin.__main__ as chat
from zihin import Config, Mind, hata
from zihin.__main__ import CHANNEL, STUDIED, Session, VARIANTS

THINGS = ["inek", "koyun", "keçi", "at", "eşek", "tavuk", "deve", "geyik"]
NEW = ["aslan", "kurt", "kaplan", "çakal", "vaşak", "tilki", "ayı", "fok"]
V = ("%s ne yer ?", "%s %s yer .")
V2 = ("%s nerede yaşar ?", "%s %s yaşar .")
L = ("%s canlı mı ?", "evet %s canlıdır .")


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def raised(keep_log: bool = False, **switches) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = keep_log, None
    for name, value in switches.items():
        setattr(cfg, name, value)
    torch.manual_seed(0)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    for _ in range(2):
        for x in THINGS:
            s.watch(V[0] % x, V[1] % (x, "ot"))
            s.watch(V2[0] % x, V2[1] % (x, "ormanda"))
            s.watch(L[0] % x, L[1] % x)
    s.close_turn()
    return s


@pytest.fixture(scope="module")
def base():
    torch.set_num_threads(1)
    return pickle.dumps(raised(faults=True).mind), pickle.dumps(raised().mind)


def session_of(blob: bytes) -> Session:
    return Session(pickle.loads(blob))


def taught(s: Session, question: str, answer: str) -> str:
    """The teacher's loop: asked, then shown the true dialogue whatever it said."""
    said = s.reply(question, limit=30)
    s.close_turn()
    s.watch(question, answer)
    s.close_turn()
    return said


def ask(s: Session, question: str) -> str:
    return Session(s.mind.clone()).reply(question, limit=30)


def test_off_by_default_and_studied():
    assert Config().faults is False and "faults" in STUDIED


def test_on_with_nothing_settled_answers_as_off_and_writes_nothing(base):
    on, off = session_of(base[0]), session_of(base[1])
    assert on.mind.S.digest() == off.mind.S.digest()
    for question in [V[0] % x for x in NEW[:3]] + [L[0] % x for x in NEW[3:6]]:
        assert on.reply(question, limit=30) == off.reply(question, limit=30)
        on.close_turn()
        off.close_turn()
    assert on.mind.S.digest() == off.mind.S.digest()      # the claims are only kept
    assert on.mind.claimed and not off.mind.claimed and not on.mind.ch.has(hata.CHANNEL)
    kept = [claim for claims in on.mind.claimed.values() for claim in claims]
    assert all(said for _, _, said, _, _ in kept)          # each is a word it said, with the way that claimed it
    assert {ways[1] for ways, _, _, _, _ in kept} == {("g", 0), ("g", -1)}   # the members' word; the asked word


def test_the_break_is_located_and_the_words_before_it_are_confirmed(base):
    s = session_of(base[0])
    mind = s.mind
    assert taught(s, V[0] % "aslan", V[1] % ("aslan", "et")) == "aslan ot yer"
    assert len(mind.faults) == 1
    fault = mind.faults[0]
    assert hata._shown(mind, fault["context"]).split()[-1] == "aslan"       # after the asked word said back
    assert hata._shown(mind, fault["claimed"]) == "ot" and hata._shown(mind, fault["lived"]) == "et"
    assert fault["said"] and fault["ways"][1] == ("g", 0)                   # the members' own word, by a frame
    assert fault["after"] < 0.5 < fault["before"]
    found = hata.votes(mind)
    echo = [key for key in found if len(key) == 3 and key[1] == -1]
    value = [key for key in found if len(key) == 3 and key[1] == 0]
    assert len(echo) == 1 and found[echo[0]][1] == 1.0                      # the asked word said back: it held
    assert len(value) == 1 and found[value[0]][1] == 0.0                    # the word at the break: it failed
    # what it said after the break ("yer", after "ot") was never lived, so it is still kept, not settled
    assert any(hata._shown(mind, context).split()[-1] == "ot" for context in mind.claimed)


def test_read_as_is_the_nested_rule():
    assert hata.read_as([], 0.66) == 0.66
    assert hata.read_as([(0.5, 0.66, 0.66)], 0.66) == pytest.approx(0.66)   # held as often as it claimed
    assert hata.read_as([(0.5, 1.0, 0.66)], 0.66) == pytest.approx(0.5 * 1.0 + 0.5 * 0.66)
    assert hata.read_as([(0.75, 0.0, 0.66)], 0.66) == pytest.approx(0.25 * 0.66)   # never held: under 1/2
    weak, strong = hata.read_as([(0.9, 0.8, 0.7)], 0.55), hata.read_as([(0.9, 0.8, 0.7)], 0.65)
    assert 0.5 < weak < strong                              # the order of a way's claims stays the reading's
    general_then_own = hata.read_as([(0.5, 0.0, 0.66), (0.5, 1.0, 0.66)], 0.66)
    assert general_then_own == pytest.approx(0.5 * 1.0 + 0.5 * (0.5 * 0.0 + 0.5 * 0.66))


def test_after_its_failures_the_frame_is_silent_for_the_next_new_thing(base):
    on, off = session_of(base[0]), session_of(base[1])
    for s in (on, off):
        for x, food in zip(NEW[:4], ("et", "balık", "et", "tohum")):
            taught(s, V[0] % x, V[1] % (x, food))
    assert ask(off, V[0] % "fok") == "fok ot yer"           # off: it goes on asserting what the eight shared
    assert ask(on, V[0] % "fok") == "fok"                   # on: the asked word (the relation held), then nothing
    assert len(on.mind.faults) >= 2


def test_an_untested_frame_is_held_back_or_tried_by_whether_holding_back_was_answered(base):
    shown, told, fresh = session_of(base[0]), session_of(base[0]), session_of(base[0])
    foods = dict(zip(NEW[:4], ("et", "balık", "et", "tohum")))
    for x, food in foods.items():
        taught(shown, V[0] % x, V[1] % (x, food))           # what it held back was then shown to it
    for x, food in list(foods.items()) + [(x, "et") for x in NEW[4:7]]:
        said = told.reply(V[0] % x, limit=30)               # ... and here it is only told good or bad
        if said not in (x, "(sessiz)"):
            told.feedback(said == "%s %s yer" % (x, food))
        told.close_turn()
    assert hata.answered(fresh.mind) == 0.5 and hata.answered(shown.mind) > 0.5 > hata.answered(told.mind)
    assert ask(fresh, V2[0] % "fok") == "fok ormanda yaşar"  # nothing settled: as the reading made it
    assert ask(shown, V2[0] % "fok") == "fok"               # frames like it failed and its silences are answered:
    assert ask(shown, L[0] % "fok") == "(sessiz)"           # held back before the first try (also a lawful frame,
    said = [taught(shown, L[0] % x, L[1] % x) for x in NEW[4:7]]   # which speaks again once shown to hold)
    assert said[0] == "(sessiz)" and said[-1] == L[1].replace(" .", "") % NEW[6]
    assert ask(told, V[0] % "fok") in ("fok", "(sessiz)")   # the frame that failed is silent here too (told only
                                                            # "bad", the asked word said back took blame as well)
    assert ask(told, V2[0] % "fok") == "fok ormanda yaşar"  # but no one answers its silences: the untested frame
    assert ask(told, L[0] % "fok") == "evet fok canlıdır"   # of that kind is tried, and so is the lawful one


def test_the_stricter_form_reads_an_untested_frame_by_the_others_vote_and_showing_gives_it_back(base):
    s = session_of(base[0])
    s.mind.__dict__["_fault_untested"] = "vote"
    for x, food in zip(NEW[:4], ("et", "balık", "et", "tohum")):
        taught(s, V[0] % x, V[1] % (x, food))
    assert ask(s, V2[0] % "fok") == "fok"                   # never asked about, yet its claim is not said
    assert ask(s, L[0] % "fok") == "(sessiz)"               # nor the lawful frame's
    said = [taught(s, L[0] % x, L[1] % x) for x in NEW[4:7]]
    assert said[0] == "(sessiz)" and said[-1] == L[1].replace(" .", "") % NEW[6]   # shown to hold: it speaks again
    assert ask(s, L[0] % "fok") == "evet fok canlıdır"
    assert ask(s, V[0] % "fok") == "fok"                    # the frame that failed stays silent


def test_a_value_signal_settles_what_was_said_and_blames_the_weakest_link(base):
    s = session_of(base[0])
    mind = s.mind
    for x in NEW[:3]:                                       # three good answers in the lawful frame
        assert s.reply(L[0] % x, limit=30) == "evet %s canlıdır" % x
        s.feedback(True)
        s.close_turn()
    found = hata.votes(mind)
    own = {key: found[key] for key in found if len(key) == 3}
    assert len(own) == 3 and all(held == 1.0 for _, held, _ in own.values()) and not mind.faults
    assert not mind.claimed                                 # every claim it said was settled as held
    assert s.reply(V[0] % "kaplan", limit=30) == "kaplan ot yer"
    sure = {"kaplan": 0.95, "ot": 0.60, "yer": 0.80}        # how sure it was of each link (set here, to pin the rule)
    for context, claims in list(mind.claimed.items()):
        mind.claimed[context] = tuple((ways, unit, said, e0, sure[hata._shown(mind, unit)])
                                      for ways, unit, said, e0, _ in claims)
    kept = {hata._shown(mind, unit): ways[-1] for claims in mind.claimed.values() for ways, unit, _, _, _ in claims}
    s.feedback(False)                                       # bad, and nothing says which word
    s.close_turn()
    whole = 0.95 * 0.60 * 0.80
    for word, e in sure.items():                            # (1 - e_i) / (1 - prod e_j): the weakest link most
        assert hata._mass(mind, kept[word])[1] == pytest.approx((1.0 - e) / (1.0 - whole))
    assert all(f["lived"] is None and f["said"] for f in mind.faults) and len(mind.faults) == 3
    assert not any(c[2] for claims in mind.claimed.values() for c in claims)


def test_undoing_the_moment_takes_a_settled_claim_back():
    s = raised(keep_log=True, faults=True)
    mind = s.mind
    assert s.reply(V[0] % "aslan", limit=30) == "aslan ot yer"
    s.close_turn()
    kept = dict(mind.claimed)
    digest = mind.S.digest()
    s.company("biri")
    s.listen("aslan ne yer ?", 2, speaker="biri")
    s.end_of_turn(2, tone="?")
    s.listen("aslan", 2)
    before = (mind.S.digest(), dict(mind.claimed), list(mind.faults))
    s.listen("et", 2)                                       # the break: the kept claim "ot" is settled as failed
    assert mind.faults and mind.ch.has(hata.CHANNEL) and len(mind.claimed) < len(before[1])
    mind.undo_last()
    assert (mind.S.digest(), dict(mind.claimed), list(mind.faults)) == before
    assert kept.keys() >= before[1].keys() and digest != before[0]


def test_what_is_kept_survives_saving_and_loading(tmp_path, base):
    from zihin import persist
    s = session_of(base[0])
    mind = s.mind
    taught(s, V[0] % "aslan", V[1] % ("aslan", "et"))
    s.reply(V[0] % "kurt", limit=30)
    s.close_turn()
    path = str(tmp_path / "m.zihin")
    persist.save(mind, path)
    loaded = persist.load(path)
    assert loaded.claimed == mind.claimed and loaded.faults == mind.faults
    assert hata.report(loaded) == hata.report(mind)
    old = pickle.loads(base[1])
    del old.__dict__["claimed"], old.__dict__["faults"]    # a mind saved before F49
    older = str(tmp_path / "old.zihin")
    persist.save(old, older)
    again = persist.load(older)
    assert again.claimed == {} and again.faults == []


def test_the_report_names_the_ways_and_the_last_faults(base):
    s = session_of(base[0])
    assert "karşılaşmadı" in hata.report(s.mind)[0]
    taught(s, V[0] % "aslan", V[1] % ("aslan", "et"))
    text = "\n".join(hata.report(s.mind))
    assert "üyelerin aynı kelimesi" in text and "sorulan kelimenin kendisi" in text
    assert "'ot' dedi" in text and "yaşanan 'et'" in text and "X ne yer" in text


KIND = {"aslan": "yırtıcı", "kurt": "yırtıcı", "kaplan": "yırtıcı", "çakal": "yırtıcı",
        "fok": "sucul", "yunus": "sucul", "balina": "sucul", "mors": "sucul"}
FOOD = {"evcil": "ot", "yırtıcı": "et", "sucul": "balık"}


def raised_with_kinds(**switches) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    for name, value in switches.items():
        setattr(cfg, name, value)
    torch.manual_seed(0)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    kinds = {**{x: "evcil" for x in THINGS}, **KIND}
    for _ in range(2):
        for x, kind in kinds.items():                      # the kind of every thing, read in a sentence
            s.company(None)
            s.listen("%s %s bir hayvandır ." % (x, kind), 2)
            s.end_of_turn(2, tone=".")
            mind.step([])
        for x in THINGS:                                    # the watched dialogues are all of one kind
            s.watch(V[0] % x, V[1] % (x, "ot"))
    s.close_turn()
    return s


def test_a_frame_found_wanting_is_read_through_a_narrower_place_of_its_filler():
    answers = {}
    for name, switches, narrower in (("off", {}, True), ("doubt", {"faults": True}, False),
                                     ("on", {"faults": True}, True)):
        s = raised_with_kinds(**switches)
        if not narrower:
            s.mind.__dict__["_fault_narrower"] = False
        for x in ("aslan", "fok", "kurt", "yunus", "kaplan", "balina"):
            taught(s, V[0] % x, V[1] % (x, FOOD[KIND[x]]))
        answers[name] = (ask(s, V[0] % "çakal"), ask(s, V[0] % "mors"))
        if name == "on":
            mind = s.mind
            assert hata.wanting(mind, hata.frame_ways(0, next(key[2] for key in hata.votes(mind)
                                                              if len(key) == 3 and key[1] == 0)))
            own = [key for key in hata.votes(mind) if len(key) == 3 and key[1] == hata.NARROWER]
            assert len(own) == 1 and hata.votes(mind)[own[0]][1] == 1.0       # that way's claims all held
            assert "başka bir yeri de paylaşan üyelerin kelimesi" in "\n".join(hata.report(mind))
            c = Session(mind.clone())
            c.explain = True
            assert c.reply(V[0] % "çakal", limit=30) == "çakal et yer"
            assert any("başka bir yeri de paylaşan" in str(r.get("detail", "")) for r in c.sources)
    assert answers["on"] == ("çakal et yer", "mors balık yer")   # the category split by what the things were read to be
    assert answers["doubt"] == ("çakal", "mors")                 # doubt alone: the asked word, then nothing
    assert "et" not in answers["off"][0].split() and "balık" not in answers["off"][1].split()


def test_a_mind_that_only_watches_settles_its_silent_claims():
    assert Config().silent_claims is False and "silent_claims" in STUDIED
    answers = {}
    for name, switches in (("on", {"faults": True}), ("silent", {"faults": True, "silent_claims": True})):
        s = raised_with_kinds(**switches)
        for x in ("aslan", "fok", "kurt", "yunus", "kaplan", "balina"):
            s.watch(V[0] % x, V[1] % (x, FOOD[KIND[x]]))    # watched; the mind is never asked
        s.close_turn()
        mind = s.mind
        answers[name] = (ask(s, V[0] % "çakal"), ask(s, V[0] % "mors"))
        if name == "on":
            assert not mind.ch.has(hata.CHANNEL) and not mind.claimed and not mind.faults   # nothing was settled
        else:
            assert any(len(key) == 3 for key in hata.votes(mind))            # its ways have records of their own
            assert mind.faults and all(f["said"] is None for f in mind.faults)   # neither said nor held back
            assert hata.answered(mind) == 0.5                # ... so it says nothing of whether holding back is answered
            assert "bekliyordu (dinlerken)" in "\n".join(hata.report(mind))
    assert answers["silent"] == ("çakal et yer", "mors balık yer")
    assert "et" not in answers["on"][0].split() and "balık" not in answers["on"][1].split()
