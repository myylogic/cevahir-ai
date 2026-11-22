# -*- coding: utf-8 -*-
"""Zihin, F43 Wholes (an utterance held as parts and answered part by part), F44 Lived Exclusion across channels and
F42 Episode Echo.

What is pinned here
-------------------
* A small world of commands (three primitives and one seen only alone; "twice", "thrice", "and") lived with a window
  of 7 items. Commands with the primitive seen alone whose dialogue does not fit the window ("d thrice and d twice":
  five words, the end and five actions) are answered exactly with F43 and F44, and not without them.
* What the summary of a lived dialogue is (``align``): the wholes that stand in the utterance as their forms, the
  replies as pointers to their places; nothing when two readings tie ("a and a").
* The working state set aside for a part is taken up again as it was.
* F44: an instance that was lived to go on with the end of the turn counts against a word it never went on with; without
  the switch F27 says nothing across channels.
* F42: what followed the last word earlier in the present episode is not read as its lived continuation.
* The switches are off by default; the speaking configuration has F42 and F44, not F43.
* A part never met is a whole when its own summary was lived as its construction (every item of it lived before):
  "b down twice and a" is answered though "x down twice" was never watched.
* F45 (Told Apart): a word never said alone answers with what two lived dialogues that differ only in it differ in;
  it is then a whole, and a construction never lived with it ("a round back") is answered. The residue: what another
  word of the utterance answers with is not the word's share, and nothing is told apart before what the two answers
  share is accounted for. A word told apart in several ways has no reply.
* The program raises a new mind with the window it is asked for (--pencere, within the range of the mind's own knob);
  a loaded mind keeps its window and says so.
"""
from __future__ import annotations

import io

import pytest
import torch

import zihin.__main__ as chat
from zihin import ACTED, IMITATED, INFERRED, Config, Mind
from zihin.__main__ import CHANNEL, TURN, TURN_END, Session, VARIANTS
from zihin import butun as B
from zihin.events import Event
from zihin.read import continuation_backoff, excluded

PRIMITIVES = {"a": "ia", "b": "ib", "c": "ic", "d": "id"}    # (the text adapter lower-cases: actions are other words)


@pytest.fixture(autouse=True)
def _single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def fresh(window: int = 7, **switches) -> Session:
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    for name, value in switches.items():
        setattr(cfg, name, value)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    mind.knob["F"] = window
    return Session(mind)


def simple(x: str):
    act = PRIMITIVES[x]
    return [((x,), (act,)), ((x, "twice"), (act,) * 2), ((x, "thrice"), (act,) * 3)]


def lived(**switches) -> Session:
    """The life: every command of a, b and c (alone, twice, thrice, and two of them joined by "and"), and "d" alone;
    two epochs, the shorter commands first."""
    s = fresh(**switches)
    parts = [d for x in "abc" for d in simple(x)]
    train = parts + [(p + ("and",) + q, a + b) for p, a in parts for q, b in parts] + [(("d",), ("id",))]
    for _ in range(2):
        for command, answer in sorted(train, key=lambda d: len(d[0])):
            s.watch(" ".join(command), " ".join(answer) + " .")
    return s


@pytest.fixture(scope="module")
def with_wholes() -> Session:
    return lived(wholes=True, exclusive_world=True)


def ask(s: Session, command: str) -> str:
    return Session(s.mind.clone()).reply(command, limit=200)


def test_switches_are_off_by_default():
    cfg = Config()
    assert cfg.wholes is False and cfg.exclusive_world is False and cfg.echo is False and cfg.told_apart is False
    s = fresh()                                                       # the speaking configuration: F42 and F44 are
    assert s.mind.cfg.wholes is False                                 # in it, F43 is not
    assert s.mind.cfg.echo is True and s.mind.cfg.exclusive_world is True and s.mind.cfg.told_apart is False


def test_commands_longer_than_the_window_are_answered_in_parts(with_wholes):
    cases = {"d twice": "id id", "d thrice and a twice": "id id id ia ia", "b thrice and d thrice": "ib ib ib id id id",
             "d thrice and d twice": "id id id id id", "d and d thrice": "id id id id"}
    for command, answer in cases.items():
        assert ask(with_wholes, command) == answer, command


def test_without_wholes_a_dialogue_that_does_not_fit_the_window_is_not_answered():
    s = lived(wholes=False)
    assert ask(s, "d twice") == "id id"                                # fits the window: answered as before
    assert ask(s, "d thrice and d twice") != "id id id id id"          # five words, the end, five actions: eleven items


def test_summary_of_a_lived_dialogue(with_wholes):
    mind, F = with_wholes.mind, with_wholes.mind.knob["F"]
    whole = B.WHOLE
    assert B.lived_reply(mind, ("a", "twice")) == ("ia", "ia")
    assert B.lived_reply(mind, ("d", "twice")) is None
    assert B.align(mind, ("a", "twice"), ("ia", "ia"), F, {}) == ((whole, "twice"), (("reply", 0),) * 2, {0: (0, 1)})
    joined = B.align(mind, ("a", "thrice", "and", "b", "twice"), ("ia", "ia", "ia", "ib", "ib"), F, {})
    assert joined == ((("whole", whole, "thrice"), "and", ("whole", whole, "twice")), (("reply", 0), ("reply", 2)),
                      {0: (0, 2), 2: (3, 5)})
    assert B.align(mind, ("a", "and", "a"), ("ia", "ia"), F, {}) is None    # two readings tie: nothing is concluded
    assert B.align(mind, ("a", "twice"), ("ib", "ic"), F, {}) is None       # not made of the reply of a whole in it


def test_reading_an_utterance_never_answered(with_wholes):
    mind, F = with_wholes.mind, with_wholes.mind.knob["F"]
    found = B.reading(mind, ("d", "thrice", "and", "d", "twice"), F, {})
    assert found == ((("whole", B.WHOLE, "thrice"), "and", ("whole", B.WHOLE, "twice")), {0: (0, 2), 2: (3, 5)})
    assert B.reading(mind, ("thrice", "and", "twice"), F, {}) is None   # no whole stands in it


def test_working_state_set_aside_is_taken_up_again(with_wholes):
    twin = Session(with_wholes.mind.clone())
    mind = twin.mind
    twin.listen("a twice", IMITATED)
    held_before = {name: getattr(mind, name) for name in B.WORKING}
    held = B.set_aside(mind)
    assert mind.episode == [] and mind.context is None and mind.pending == []
    twin.listen("b", IMITATED)                                         # another episode is lived meanwhile
    B.take_up(mind, held)
    for name, value in held_before.items():
        assert getattr(mind, name) == value, name


def test_exclusion_across_channels():
    for across in (False, True):
        s = fresh(exclusive_world=across)
        mind = s.mind
        s.watch("a", "ia .")
        end, word = mind.unit_of(TURN, TURN_END), mind.unit_of(CHANNEL, "ia")
        out = {word: 0.8}
        excluded(mind, out, {end: (3.0, 0.0)})                         # the instance always went on with the end
        if across:
            assert out[word] == pytest.approx(0.8 * (1.0 - 3.0 / (3.0 + mind.cfg.k)))
        else:
            assert out[word] == 0.8                                    # F27 reads one channel at a time


def test_echo_is_not_read_as_what_follows():
    for echo in (False, True):
        s = fresh(echo=echo)
        mind = s.mind
        for _ in range(3):
            s.watch("q", "x y .")                                      # x is lived to go on with y
        s.listen("q", IMITATED)
        s.end_of_turn(IMITATED, tone=".")
        for word in ("x", "y", "x"):                                   # the mind's own words: x y x
            mind.step([Event(CHANNEL, word, ACTED)])
        y = mind.unit_of(CHANNEL, "y")
        went = continuation_backoff(mind, mind.context)
        if echo:
            assert y not in went                                       # y followed x earlier in this episode
        else:
            assert went.get(y, 0.0) > 0.5


def test_undo_drops_the_reading_of_situations():
    """The situations' reading is kept by unit and moment; both come again after an undo, so an undo drops it."""
    mind = Mind(Config(log_path=None))
    mind.add_channel(CHANNEL, "act")
    mind.step([Event(CHANNEL, "a", IMITATED)])
    mind.step([Event(CHANNEL, "b", IMITATED)])
    mind._situation_cache = ((0, mind.t, None), {})
    mind.undo_last()
    assert "_situation_cache" not in mind.__dict__


def test_a_summary_is_also_lived_as_its_construction(with_wholes):
    """The construction of a summary: its words as they are, every whole bare. A summary whose forms never met in it
    is heard as its construction when that was lived and answered."""
    mind = with_wholes.mind
    whole = B.WHOLE
    summary = (("whole", whole, "thrice"), "and", ("whole", whole, "twice"))
    assert B.construction(summary) == (whole, "and", whole)
    assert B.went_on(mind, B.construction(summary))                      # lived as a construction too
    assert B._heard_as(mind, summary) == summary                         # lived as it is: heard as it is
    never = (("whole", whole, "never"), "and", whole)
    assert not B.went_on(mind, never) and B._heard_as(mind, never) == (whole, "and", whole)


def test_the_summaries_followed_hold_every_lived_one(with_wholes):
    """Choices of wholes are followed only while their construction so far was lived: every summary of the utterance
    that was lived, as it is or as its construction, is among them, and each of them is a choice of its wholes."""
    mind, F = with_wholes.mind, with_wholes.mind.knob["F"]
    for words in (("d", "thrice", "and", "d", "twice"), ("a", "twice", "and", "b"), ("c", "and", "d", "thrice")):
        memo, forms = {}, {}
        for span in B._spans(len(words)):
            part = words[span[0]:span[1]]
            reply = B._reply(mind, part, memo)
            if reply is not None:
                forms[span] = B.form(mind, part, reply, F, memo)
            elif len(part) > 1 and B.reading(mind, part, F, memo) is not None:
                forms[span] = B.form(mind, part, None, F, memo)
        followed = {items for items, _ in B._summaries(mind, words, forms, F)}
        every = {B._summary(words, chosen, forms)[0] for chosen in B._disjoint(forms)}
        lived = {items for items in every if len(items) + 2 <= F
                 and (B.went_on(mind, items) or B.went_on(mind, B.construction(items)))}
        assert lived and lived <= followed <= every, words


def watch_all(s: Session, train, longer_first: bool = False) -> Session:
    for _ in range(2):
        for command, answer in sorted(train, key=lambda d: len(d[0]), reverse=longer_first):
            s.watch(" ".join(command), " ".join(answer) + " .")
    return s


def test_a_part_is_read_by_its_lived_construction():
    """ "x up twice" is watched, "x down twice" never; "x down" is, alone and joined. The part "b down twice" has no
    lived summary of its own forms; its construction (a whole, twice) was lived, and so was each of its items."""
    parts = []
    for x, act in (("a", "ia"), ("b", "ib")):
        parts += [((x,), (act,)), ((x, "up"), (act, "iu")), ((x, "down"), (act, "id")), ((x, "twice"), (act, act)),
                  ((x, "up", "twice"), (act, "iu", act, "iu"))]
    train = parts + [(p + ("and",) + q, a + b) for p, a in parts for q, b in parts]
    s = watch_all(fresh(window=9, wholes=True), train)
    mind, F = s.mind, s.mind.knob["F"]
    assert B.lived_reply(mind, ("b", "down", "twice")) is None
    summary = (("whole", B.WHOLE, "down"), "twice")
    assert not B.went_on(mind, summary) and B.went_on(mind, B.construction(summary))
    assert B.reading(mind, ("b", "down", "twice"), F, {}) == (summary, {0: (0, 2)})
    assert ask(s, "b down twice") == "ib id ib id"
    assert ask(s, "b down twice and a") == "ib id ib id ia"
    assert ask(s, "a up and b down twice") == "ia iu ib id ib id"


def directions(longer_first: bool = False, **switches) -> Session:
    """Moves said alone (a, b, c), and twice ("a twice"); directions never said alone (left, right, back), each one
    step wherever it stands; "round" with left and right only; "go" has no step of its own."""
    turns = {"left": "tl", "right": "tr", "back": "tk"}
    train = [((m,), ("i" + m,)) for m in "abc"] + [((m, "twice"), ("i" + m,) * 2) for m in "abc"]
    for m in "abc":
        for d, t in turns.items():
            train.append(((m, d), (t, "i" + m)))
            if d != "back":
                train.append(((m, "round", d), (t, "i" + m, t, "i" + m)))
    train += [(("go", d), (t,)) for d, t in turns.items()] + [(("go", "left", "twice"), ("tl", "tl"))]
    return watch_all(fresh(window=9, wholes=True, **switches), train, longer_first)


def test_a_word_never_said_alone_is_told_apart():
    s = directions(told_apart=True)
    mind, F = s.mind, s.mind.knob["F"]
    assert B.said_alone(mind, "a") and not B.said_alone(mind, "back")
    alone = [B.said(("a",))]                                 # a word said alone keeps the reply it was given: nothing
    assert B.went_on(mind, alone) == B.went_on(mind, alone, without=(INFERRED,))      # was inferred of it
    assert B.went_on(mind, [B.said(("back",))]) and not B.went_on(mind, [B.said(("back",))], without=(INFERRED,))
    assert B.lived_reply(mind, ("back",)) == ("tk",) and B.lived_reply(mind, ("left",)) == ("tl",)
    assert B.lived_reply(mind, ("go",)) is None and B.lived_reply(mind, ("round",)) is None
    assert B.align(mind, ("a", "round", "left"), ("tl", "ia", "tl", "ia"), F, {}) == (
        (B.WHOLE, "round", B.WHOLE), (("reply", 2), ("reply", 0)) * 2, {0: (0, 1), 2: (2, 3)})
    assert ask(s, "a round back") == "tk ia tk ia"          # "round" was never met with "back"
    assert ask(s, "c round back") == "tk ic tk ic"
    # two readings of "tl tl" are equally cheap ("go left" twice; "left" twice), the one inside the other: they are
    # one reading at two grains, and the larger whole is taken
    assert B.align(mind, ("go", "left", "twice"), ("tl", "tl"), F, {}) == (
        (("whole", "go", B.WHOLE), "twice"), (("reply", 0),) * 2, {0: (0, 2)})


def test_the_residue_is_not_a_words_share():
    """ "a twice" and "a left" differ in one word, and their answers ("ia ia", "tl ia") in one pair of words: but "ia"
    is what "a" answers with, so nothing is told of "twice". When the commands of two words are watched before the
    moves are said alone, what the answers share is not accounted for yet and nothing is told apart until it is."""
    for longer_first in (False, True):
        s = directions(longer_first, told_apart=True)
        mind = s.mind
        assert B.lived_reply(mind, ("twice",)) is None and not B.went_on(mind, [B.said(("twice",))])
        assert B.lived_reply(mind, ("back",)) == ("tk",)
        assert ask(s, "a round back") == "tk ia tk ia"
        assert ask(s, "c twice") == "ic ic"


def test_readings_that_cross_conclude_nothing():
    """ "a then b left" answered "tl ib ia" is "a then (b left)" and, where "a then b" was said alone, also "(a then
    b) left": the two readings are equally cheap and their wholes cross, so no summary is concluded (the larger
    whole would make a conjunction stand before a direction)."""
    turns = {"left": "tl", "right": "tr"}
    train = [((m,), ("i" + m,)) for m in "ab"]
    train += [((m, d), (t, "i" + m)) for m in "ab" for d, t in turns.items()]
    train += [(("a", "then", "b"), ("ib", "ia")), (("b", "then", "a"), ("ia", "ib"))]
    s = watch_all(fresh(window=9, wholes=True, told_apart=True), train)
    mind, F = s.mind, s.mind.knob["F"]
    assert B.lived_reply(mind, ("left",)) == ("tl",) and B.lived_reply(mind, ("a", "then", "b")) == ("ib", "ia")
    assert B.align(mind, ("a", "then", "b", "left"), ("tl", "ib", "ia"), F, {}) is None
    assert B.align(mind, ("b", "then", "b", "left"), ("tl", "ib", "ib"), F, {}) == (       # "b then b" never said
        (B.WHOLE, "then", ("whole", B.WHOLE, B.WHOLE)), (("reply", 2), ("reply", 0)), {0: (0, 1), 2: (2, 4)})


def test_a_word_told_apart_in_several_ways_has_no_reply():
    s = directions(told_apart=True)
    mind = s.mind
    for reply in ("ia", "ib"):                              # inferred of "twice", in two ways, as often
        B._live(s, (B.said(("twice",)),), (B.said((reply,)),), INFERRED, None, (".", "."))
    assert B.went_on(mind, [B.said(("twice",))]) and B.lived_reply(mind, ("twice",)) is None
    B._live(s, (B.said(("twice",)),), (B.said(("ia",)),), INFERRED, None, (".", "."))
    assert B.lived_reply(mind, ("twice",)) == ("ia",)       # one of them is most of the evidence now


def test_without_told_apart_a_word_never_said_alone_has_no_reply():
    s = directions()
    assert B.lived_reply(s.mind, ("back",)) is None
    assert ask(s, "a round back") != "tk ia tk ia"


def test_the_program_raises_a_new_mind_with_its_window(monkeypatch, tmp_path, capsys):
    from zihin.__main__ import main
    from zihin.persist import load
    home = tmp_path / "ev"
    monkeypatch.setenv("CEVAHIR_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO("merhaba .\n!cik\n"))
    main(["--yeni", "--pencere", "13", "--ac", "wholes,told_apart", "--kim", "deneme", "--yasamsiz"])
    mind = load(str(home / "zihin" / "cevahir"), ledger=False)
    assert mind.knob["F"] == 13 and mind.cfg.wholes is True and mind.cfg.told_apart is True
    capsys.readouterr()
    monkeypatch.setattr("sys.stdin", io.StringIO("!cik\n"))
    main(["--pencere", "5", "--yasamsiz", "--kim", "deneme"])       # the home mind again: it keeps its window
    assert "penceresi 13 öğe" in capsys.readouterr().out
    assert load(str(home / "zihin" / "cevahir"), ledger=False).knob["F"] == 13
    monkeypatch.setattr("sys.stdin", io.StringIO("merhaba .\n!cik\n"))
    main(["--yeni", "--pencere", "40", "--yasamsiz", "--kim", "deneme"])      # the range of the mind's own knob
    assert load(str(home / "zihin" / "cevahir"), ledger=False).knob["F"] == 15
