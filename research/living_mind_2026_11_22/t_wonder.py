# -*- coding: utf-8 -*-
"""t_wonder.py: F51 Asking Itself, measured. What is new in what the mind was told it asks itself about, in its own
time; what it cannot answer it looks up; what it reads brings the next new thing.

    python research/living_mind_2026_11_22/t_wonder.py chain   [--seeds 5]
    python research/living_mind_2026_11_22/t_wonder.py chains  [--seeds 5]
    python research/living_mind_2026_11_22/t_wonder.py dead    [--seeds 5]
    python research/living_mind_2026_11_22/t_wonder.py young   [--seeds 5]

The mind of every part: 8 things read of ("X V yer .") and watched being asked ("X ne yer ?" -> "X V yer ."), with
four values among them, twice; so it knows one question and how it is answered. Nothing else is given to it: no
list of what to ask about, no order to look.

``chain``: a chain of 7 things, each of which eats the next. A library holds one sentence about each of the middle
ones ("c_i c_{i+1} yer ."); about the last it holds nothing. The partner tells the mind one sentence, about the
first ("c_0 c_1 yer ."), and then says nothing: 40 silent moments pass. Counted: the questions the mind asked
itself, its looks and how many met what it was looking for, the silent moment of its last look; then, on copies
that may not look, "c_i ne yer ?" for the six things that eat something (right / wrong / silent), and what it
says of the last one.
Conditions (the same life in each): F46 + F47 + F49 + F50 + F41; the same with F51; with F51 and nowhere to look.

``chains``: two such chains of 4 things, their first sentences told one after the other; 40 silent moments. Whether
both are followed (the things asked about are taken oldest first).

``dead``: the partner tells 10 sentences about 10 pairs of new things, 6 silent moments after each; the library
holds nothing at all. Counted sentence by sentence: the questions it asked itself and its looks, and what asking
and looking are then expected to lead to (e_wonder, e_look). Then the library is given the sentences and 60 more
silent moments pass: whether the mind comes back to what it could not find.

``young``: the mind is raised with F51 on from its first moment and 3 silent moments after every lesson, the
library holding only the chain: what it asks itself and looks up while it is raised (about the words of its
lessons), and whether it still follows the chain afterwards.

Appends to ``results/wonder.jsonl``.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import t_fault as T  # noqa: E402

OUT = os.path.join(HERE, "results", "wonder.jsonl")
BASE = {"open_claims": True, "look": True, "find_out": True, "known": True}
QUIET = lambda *_: None  # noqa: E731


def raise_mind(seed: int, more: dict, library: dict, silent: int = 0):
    """The mind and its session; returns (session, the things not used in raising, the questions asked itself and
    the looks made while it was raised)."""
    from zihin import IMITATED
    rng = random.Random(seed)
    things = list(T.THINGS)
    rng.shuffle(things)
    members = things[:8]
    values = T.FRAMES["V1"][2]
    food = {x: values[i % 4] for i, x in enumerate(members)}
    mind, s = T.make_mind(seed, True, None, dict(more))
    s.look_gap = 0.0
    s.look_source = T.Library(library) if library is not None else None
    s.may_look = library is not None
    for _ in range(2):
        for x in members:
            s.company(None)
            s.listen("%s %s yer ." % (x, food[x]), IMITATED)
            s.end_of_turn(IMITATED, tone=".")
            mind.step([])
            for _ in range(silent):
                s.silent_moment(say=QUIET)
            s.watch("%s ne yer ?" % x, "%s %s yer ." % (x, food[x]))
            s.close_turn()
            for _ in range(silent):
                s.silent_moment(say=QUIET)
        s.watch("merhaba", "merhaba .")
    s.close_turn()
    before = (len(s.thoughts), len(s.looks))
    return s, things[8:], before


def told(s, sentence: str) -> None:
    from zihin import TOLD
    s.listen(sentence, TOLD, speaker=s.partner)
    s.end_of_turn(TOLD, tone=".")
    s.mind.step([])                                        # the telling has ended: what comes next is another run


def silence(s, moments: int):
    """``moments`` silent moments; returns (what it said aloud, the moment of its last look)."""
    said, last = [], 0
    for i in range(moments):
        looks = len(s.looks)
        acts = s.silent_moment(say=QUIET)
        if acts:
            said += s.speak(acts, None, 30)
            s.close_turn()
        if len(s.looks) > looks:
            last = i + 1
    return said, last


def asked(s, since: int = 0):
    """(the questions it asked itself, how many of them it could not answer)."""
    mine = s.thoughts[since:]
    return len(mine), sum(1 for _, _, reality in mine if reality == 0.0)


def probe(s, question: str) -> str:
    from zihin.__main__ import Session
    copy = Session(s.mind.clone(), s.pieces)
    copy.partner, copy.dil = s.partner, s.dil
    copy.may_look = False
    return copy.reply(question, limit=30)


CONDITIONS = [("F46 + F47 + F49 + F50 + F41", dict(BASE), True),
              ("the same with F51", dict(BASE, wonder=True), True),
              ("with F51, nowhere to look", dict(BASE, wonder=True), False)]


def chain(args, lengths=(7,)) -> None:
    rows = []
    for label, more, can in CONDITIONS:
        total = {"asked": 0, "unknown": 0, "looks": 0, "met": 0, "right": 0, "wrong": 0, "silent": 0,
                 "last_right": 0, "aloud": 0, "open": 0}
        last_look, first = [], None
        for seed in range(args.seeds):
            rng = random.Random(1000 + seed)
            pool = None
            s, rest, before = raise_mind(seed, more, None)
            pool = list(rest)
            rng.shuffle(pool)
            chains_, at = [], 0
            for n in lengths:
                chains_.append(pool[at:at + n])
                at += n
            library = {c[i]: "%s %s yer ." % (c[i], c[i + 1]) for c in chains_ for i in range(1, len(c) - 1)}
            if can:
                s.look_source, s.may_look = T.Library(library), True
            for c in chains_:
                told(s, "%s %s yer ." % (c[0], c[1]))
            aloud, last = silence(s, 40)
            n_asked, n_unknown = asked(s, before[0])
            marks = ""
            for c in chains_:
                for i in range(len(c) - 1):
                    marks += T.verdict(s, probe(s, "%s ne yer ?" % c[i]), "%s %s yer ." % (c[i], c[i + 1]))
                total["last_right"] += probe(s, "%s ne yer ?" % c[-1]) == c[-1]   # of the last it says the asked
            for key, n in T.count(marks).items():                                 # word and holds
                total[key] += n
            total["asked"] += n_asked
            total["unknown"] += n_unknown
            total["looks"] += len(s.looks) - before[1]
            total["met"] += sum(1 for r in s.looks[before[1]:] if r.get("met") is True)
            total["aloud"] += len(aloud)
            total["open"] += sum(1 for item in s.mind.open_where if item in s.mind.open_plain)
            last_look.append(last)
            first = first or {"told": ["%s %s yer ." % (c[0], c[1]) for c in chains_],
                              "asked_itself": ["%s -> %s" % (q, " ".join(a) or "(nothing)") for q, a, _ in
                                               s.thoughts[before[0]:]],
                              "looked_up": [(r["topic"], r.get("met")) for r in s.looks[before[1]:]],
                              "answers": marks}
        rows.append({"test": "wonder_chain" if len(lengths) == 1 else "wonder_chains", "condition": label,
                     "seeds": args.seeds, "lengths": list(lengths), "counts": total,
                     "last_look_at": last_look, "first": first})
    links = sum(n - 1 for n in lengths)
    lines = ["== told %s, then 40 silent moments: %d seeds; %d things that eat something are asked about "
             "afterwards on copies that may not look (right / wrong / silent) ==" % (
                 "one sentence about the first of a chain of %d things" % lengths[0] if len(lengths) == 1 else
                 "the first sentences of %d chains of %d things" % (len(lengths), lengths[0]),
                 args.seeds, links * args.seeds)]
    for row in rows:
        c = row["counts"]
        lines.append("  %-30s asked itself %d (could not answer %d) ; looks %d (met %d) ; afterwards %d/%d/%d ; "
                     "of the last thing it says the asked word and holds: %d of %d ; words said aloud in the "
                     "silence %d ; its last look at silent moment %s" % (
                         row["condition"], c["asked"], c["unknown"], c["looks"], c["met"], c["right"], c["wrong"],
                         c["silent"], c["last_right"], len(lengths) * args.seeds, c["aloud"],
                         "/".join(str(n) for n in row["last_look_at"])))
    shown = rows[1]["first"]
    lines.append("  the first seed with F51: told %s" % " ".join(shown["told"]))
    lines.append("    asked itself: %s" % " | ".join(shown["asked_itself"]))
    lines.append("    looked up: %s" % ", ".join("%s (%s)" % (t, {True: "met", False: "not met", None: "-"}[m])
                                                 for t, m in shown["looked_up"]))
    T.finish(args, rows, lines)


def dead(args) -> None:
    from zihin.read import LOOK_ALL, look_expectation, wonder_expectation
    rows, lines = [], []
    per = [[0, 0] for _ in range(10)]
    after = {"looks": 0, "met": 0, "right": 0, "wrong": 0, "silent": 0, "asked": 0}
    e_w, e_l, first = [0.0] * 10, [0.0] * 10, None
    for seed in range(args.seeds):
        rng = random.Random(2000 + seed)
        s, rest, before = raise_mind(seed, dict(BASE, wonder=True), {})
        pool = list(rest)
        rng.shuffle(pool)
        pairs = [(pool[2 * i], pool[2 * i + 1]) for i in range(10)]
        more = list(pool[20:30])
        shown = []
        for i, (x, y) in enumerate(pairs):
            thoughts, looks = len(s.thoughts), len(s.looks)
            told(s, "%s %s yer ." % (x, y))
            silence(s, 6)
            per[i][0] += len(s.thoughts) - thoughts
            per[i][1] += len(s.looks) - looks
            e_w[i] += wonder_expectation(s.mind)
            e_l[i] += look_expectation(s.mind, LOOK_ALL)
            shown.append("%d/%d" % (len(s.thoughts) - thoughts, len(s.looks) - looks))
        # the library now holds a sentence about every thing it could not find
        s.look_source = T.Library({y: "%s %s yer ." % (y, more[i]) for i, (x, y) in enumerate(pairs)})
        thoughts, looks = len(s.thoughts), len(s.looks)
        silence(s, 60)
        after["asked"] += len(s.thoughts) - thoughts
        after["looks"] += len(s.looks) - looks
        after["met"] += sum(1 for r in s.looks[looks:] if r.get("met") is True)
        marks = "".join(T.verdict(s, probe(s, "%s ne yer ?" % y), "%s %s yer ." % (y, more[i]))
                        for i, (x, y) in enumerate(pairs))
        for key, n in T.count(marks).items():
            after[key] += n
        first = first or " ".join(shown)
    n = float(args.seeds)
    rows.append({"test": "wonder_dead", "seeds": args.seeds, "asked_itself_per_sentence": [p[0] for p in per],
                 "looks_per_sentence": [p[1] for p in per], "e_wonder": [round(e / n, 3) for e in e_w],
                 "e_look": [round(e / n, 3) for e in e_l], "after_the_library_was_filled": after, "first": first})
    lines.append("== a library that holds nothing: %d seeds; 10 sentences told, each about two new things, 6 silent "
                 "moments after each ==" % args.seeds)
    lines.append("  questions it asked itself, sentence by sentence: %s" % " ".join(str(p[0]) for p in per))
    lines.append("  looks, sentence by sentence:                     %s" % " ".join(str(p[1]) for p in per))
    lines.append("  what asking is then expected to lead to (mean):  %s" % " ".join("%.2f" % (e / n) for e in e_w))
    lines.append("  what looking is then expected to lead to (mean): %s" % " ".join("%.2f" % (e / n) for e in e_l))
    lines.append("  the first seed (asked itself/looks): %s" % first)
    lines.append("  then the library is given a sentence about each of the 10 things it could not find, and 60 silent "
                 "moments pass: asked itself %d, looks %d (met %d); the %d things asked about afterwards on copies: "
                 "%d/%d/%d (right / wrong / silent)" % (after["asked"], after["looks"], after["met"], 10 * args.seeds,
                                                        after["right"], after["wrong"], after["silent"]))
    T.finish(args, rows, lines)


def young(args) -> None:
    rows, lines = [], []
    for label, silent in (("no silent moments while it is raised", 0), ("3 silent moments after every lesson", 3)):
        total = {"raised_asked": 0, "raised_unknown": 0, "raised_looks": 0, "raised_met": 0, "asked": 0, "looks": 0,
                 "met": 0, "right": 0, "wrong": 0, "silent": 0}
        first = None
        for seed in range(args.seeds):
            rng = random.Random(1000 + seed)
            things = list(T.THINGS)
            random.Random(seed).shuffle(things)
            pool = things[8:]
            rng.shuffle(pool)
            c = pool[:7]
            library = {c[i]: "%s %s yer ." % (c[i], c[i + 1]) for i in range(1, len(c) - 1)}
            s, _, before = raise_mind(seed, dict(BASE, wonder=True), library, silent)
            total["raised_asked"] += before[0]
            total["raised_unknown"] += sum(1 for _, _, reality in s.thoughts[:before[0]] if reality == 0.0)
            total["raised_looks"] += before[1]
            total["raised_met"] += sum(1 for r in s.looks[:before[1]] if r.get("met") is True)
            told(s, "%s %s yer ." % (c[0], c[1]))
            silence(s, 40)
            n_asked, _ = asked(s, before[0])
            total["asked"] += n_asked
            total["looks"] += len(s.looks) - before[1]
            total["met"] += sum(1 for r in s.looks[before[1]:] if r.get("met") is True)
            marks = "".join(T.verdict(s, probe(s, "%s ne yer ?" % c[i]), "%s %s yer ." % (c[i], c[i + 1]))
                            for i in range(len(c) - 1))
            for key, n in T.count(marks).items():
                total[key] += n
            first = first or ["%s -> %s" % (q, " ".join(a) or "(nothing)") for q, a, _ in s.thoughts[:before[0]]][:12]
        rows.append({"test": "wonder_young", "condition": label, "seeds": args.seeds, "counts": total, "first": first})
    lines.append("== raised with F51 on from the first moment; the library holds only the chain: %d seeds; then one "
                 "sentence about the first of a chain of 7 things, 40 silent moments, and the 6 things asked about on "
                 "copies ==" % args.seeds)
    for row in rows:
        c = row["counts"]
        lines.append("  %-38s while raised: asked itself %d (could not answer %d), looks %d (met %d) ; then: asked "
                     "itself %d, looks %d (met %d), afterwards %d/%d/%d" % (
                         row["condition"], c["raised_asked"], c["raised_unknown"], c["raised_looks"], c["raised_met"],
                         c["asked"], c["looks"], c["met"], c["right"], c["wrong"], c["silent"]))
        if row["first"]:
            lines.append("      the first seed, what it asked itself while raised: %s" % " | ".join(row["first"]))
    T.finish(args, rows, lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["chain", "chains", "dead", "young"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
    {"chain": chain, "chains": lambda a: chain(a, (4, 4)), "dead": dead, "young": young}[args.what](args)


if __name__ == "__main__":
    main()
