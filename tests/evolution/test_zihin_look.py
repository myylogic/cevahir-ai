# -*- coding: utf-8 -*-
"""Zihin, F47 Looking Up by Itself: the mind decides, in a silent moment of its life, to look up what it found it did
not know; the session carries the look out when its owner lets it; the mind settles it by reading the claim again.

What is pinned here
-------------------
* The switch is off by default and is one of the studied mechanisms (``--ac look``); without it no moment decides
  to look.
* Asked about a thing it never met, the mind says the name and no more (F46 notes the open claim and where it was
  left open). Left in silence, nobody telling it to, it decides to look that thing up, and nothing else; before it
  was asked it decided nothing.
* The decision is the mind's; the reading is the session's and happens only when the owner lets it (``may_look``),
  from the source the owner gave, and not more often than the session's gap.
* After reading, the mind reads the claim again where it was left open: met, the want is over, it answers when
  asked, and it looks no more. That a look closed its claim is evidence for looking (the item's own record and the
  record of every look).
* When what it reads says nothing of the thing, the claim stays open and that counts against looking: with no
  earlier success it stops; the want stays, and it is not looked up again in the silence that follows.
* A look is an entry of the history and is taken back like one.
* A folder of the owner's documents is a source (the sentences that hold the word are what is read), asked before
  the next source.
* A page read while the mind is talking with someone is no one's in particular (as a document is): what it said is
  told to anyone who asks, not only to the person who was there.
* The state is kept through saving and loading; a mind saved before F47 gets it empty.
"""
from __future__ import annotations

import pytest
import torch

import zihin.__main__ as chat
from zihin import IMITATED, Config, Mind
from zihin.__main__ import CHANNEL, STUDIED, Session, VARIANTS
from zihin.persist import load, save
from zihin.read import look_expectation, open_claim
from zihin.web import Folder, Sources

KNOWN = {"gül": "kırmızı bir çiçektir", "papatya": "beyaz bir çiçektir", "çam": "yeşil bir ağaçtır",
         "meşe": "yaşlı bir ağaçtır", "çekiç": "ağır bir alettir", "testere": "keskin bir alettir",
         "tren": "uzun bir taşıttır", "gemi": "büyük bir taşıttır"}
WATCHED = ("gül", "çam", "çekiç", "tren", "papatya", "testere")
LIBRARY = {"zambak": "mor bir çiçektir", "vapur": "yavaş bir taşıttır"}
ASK = "%s nedir ?"


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fact(thing: str) -> str:
    return "%s %s ." % (thing, {**KNOWN, **LIBRARY}[thing])


class Library:
    def __init__(self, things=LIBRARY) -> None:
        self.things, self.asked = set(things), []

    def search(self, topic: str, limit: int = 2):
        self.asked.append(topic)
        return [topic] if topic in self.things else []

    def extract(self, identifier: str) -> str:
        return fact(identifier)

    def address(self, identifier: str) -> str:
        return "library:" + identifier


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
    for _ in range(3):
        for thing in KNOWN:
            s.company(None)
            s.listen(fact(thing), IMITATED)
            s.end_of_turn(IMITATED, tone=s.tone(fact(thing)))
            mind.step([])
        for thing in WATCHED:
            s.watch(ASK % thing, fact(thing))
    s.look_gap = 0.0
    return s


def quiet(line: str) -> None:
    pass


def silence(s: Session, moments: int = 6) -> None:
    for _ in range(moments):
        s.silent_moment(say=quiet)


def ask(s: Session, thing: str) -> str:
    return Session(s.mind.clone()).reply(ASK % thing, limit=40)


def test_the_switch_is_off_by_default_and_studied():
    assert Config().look is False and "look" in STUDIED
    s = raised(open_claims=True)                                      # the want alone decides nothing
    s.may_look, s.look_source = True, Library()
    assert s.reply(ASK % "zambak", limit=40) == "zambak"
    s.close_turn()
    silence(s)
    assert s.mind.looking is None and s.look_source.asked == [] and s.looks == []


def test_left_to_itself_it_looks_up_what_it_could_not_answer_and_then_stops():
    s = raised(open_claims=True, look=True)
    mind, library = s.mind, Library()
    s.may_look, s.look_source = True, library
    silence(s)
    assert library.asked == []                                        # nobody asked anything: nothing to look for
    assert s.reply(ASK % "zambak", limit=40) == "zambak"
    s.close_turn()
    assert open_claim(mind, "zambak") > 0.5 and mind.unit_of(CHANNEL, "zambak") in mind.open_where
    assert look_expectation(mind, "zambak") == 0.5                    # nothing was ever looked up
    silence(s)
    assert library.asked == ["zambak"]                                # that thing, once, and nothing else
    assert [r["met"] for r in s.looks] == [True]                      # it read the claim again itself: met
    assert open_claim(mind, "zambak") <= 0.5
    assert look_expectation(mind, "vapur") > 0.5                      # looking has led to something
    assert ask(s, "zambak") == "zambak mor bir çiçektir"
    assert s.history[-1].text == "[kendi kararıyla baktı] zambak"
    silence(s, 10)
    assert library.asked == ["zambak"]


def test_the_decision_is_the_minds_and_the_reading_waits_for_the_owner():
    s = raised(open_claims=True, look=True)
    library = Library()
    s.look_source = library                                           # the owner has not let it look
    assert s.reply(ASK % "vapur", limit=40) == "vapur"
    s.close_turn()
    silence(s, 3)
    assert s.mind.looking is not None and s.mind.looking[1] == "vapur"   # it decided to
    assert library.asked == [] and ask(s, "vapur") == "vapur"            # and nothing was read
    s.may_look, s.look_gap = True, 3600.0
    s.looked_at = __import__("time").time()                           # a look was just made: not again so soon
    silence(s, 3)
    assert library.asked == []
    s.look_gap = 0.0
    silence(s, 1)
    assert library.asked == ["vapur"] and ask(s, "vapur") == "vapur yavaş bir taşıttır"


def test_a_look_that_leads_to_nothing_counts_against_looking():
    s = raised(open_claims=True, look=True)
    mind, library = s.mind, Library(things=())
    s.may_look, s.look_source = True, library
    for thing in LIBRARY:
        assert s.reply(ASK % thing, limit=40) == thing
    s.close_turn()
    silence(s, 12)
    assert len(library.asked) == 1                                    # one look, nothing found, and it stops
    assert [r["met"] for r in s.looks] == [False]
    assert look_expectation(mind, "\0 any other item") < 0.5
    assert all(open_claim(mind, thing) > 0.5 for thing in LIBRARY)    # the wants are still there
    assert all(ask(s, thing) == thing for thing in LIBRARY)


def test_a_look_is_an_entry_of_the_history_and_is_taken_back():
    s = raised(keep_log=True, open_claims=True, look=True)
    s.may_look, s.look_source = True, Library()
    assert s.reply(ASK % "zambak", limit=40) == "zambak"
    s.close_turn()
    silence(s, 2)
    assert ask(s, "zambak") == "zambak mor bir çiçektir"
    entry = s.history[-1]
    assert entry.text == "[kendi kararıyla baktı] zambak"
    s.forget(entry.number)
    assert ask(s, "zambak") == "zambak"
    assert open_claim(s.mind, "zambak") > 0.5                         # the claim is open again, as before the look


def test_a_folder_of_the_owners_documents_is_a_source(tmp_path):
    (tmp_path / "bitkiler.txt").write_text(
        "bahçede çok şey vardır .\nzambak mor bir çiçektir .\nyaz sıcak geçer .\n", encoding="utf-8")
    (tmp_path / "baska.txt").write_text("deniz mavidir .\n", encoding="utf-8")
    folder = Folder(str(tmp_path))
    assert folder.search("zambak") == [str(tmp_path / "bitkiler.txt")]
    assert folder.extract(str(tmp_path / "bitkiler.txt")) == "zambak mor bir çiçektir ."
    assert folder.search("vapur") == []
    s = raised(open_claims=True, look=True)
    later = Library()
    s.may_look, s.look_source = True, Sources([folder, later])
    for thing in LIBRARY:
        assert s.reply(ASK % thing, limit=40) == thing
    s.close_turn()
    silence(s, 8)
    assert ask(s, "zambak") == "zambak mor bir çiçektir" and ask(s, "vapur") == "vapur yavaş bir taşıttır"
    assert later.asked == ["vapur"]                                   # the folder held zambak: nothing else was asked


def test_a_page_looked_up_while_someone_is_present_is_no_ones_in_particular():
    from zihin.web import research
    s = raised(open_claims=True)
    s.partner = "ali"
    assert s.reply(ASK % "zambak", limit=40) == "zambak"              # ali asks; the mind does not know
    research(s, "zambak", source=Library(), say=quiet)                # and the page is read while ali is there
    other = Session(s.mind.clone())
    other.partner = "veli"
    assert other.reply(ASK % "zambak", limit=40) == "zambak mor bir çiçektir"   # anyone is told what a page said


def test_the_want_is_kept_through_saving_and_a_mind_saved_before_gets_it_empty(tmp_path):
    s = raised(open_claims=True, look=True)
    assert s.reply(ASK % "zambak", limit=40) == "zambak"
    s.close_turn()
    save(s.mind, str(tmp_path / "mind"))
    again = load(str(tmp_path / "mind"), ledger=False)
    assert again.open_where == s.mind.open_where and open_claim(again, "zambak") > 0.5
    woken = Session(again)
    woken.may_look, woken.look_source, woken.look_gap = True, Library(), 0.0
    silence(woken, 4)
    assert woken.look_source.asked == ["zambak"]
    del s.mind.open_where, s.mind.looking                             # as a mind saved before F47 is
    save(s.mind, str(tmp_path / "older"))
    older = load(str(tmp_path / "older"), ledger=False)
    assert older.open_where == {} and older.looking is None
    Session(older).silent_moment(say=quiet)                           # and it lives on
