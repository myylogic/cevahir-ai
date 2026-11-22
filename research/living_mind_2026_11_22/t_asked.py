# -*- coding: utf-8 -*-
"""t_asked.py: F55 The Asked Place, measured. A question read against what is known of the thing it is about.

    python research/living_mind_2026_11_22/t_asked.py forms     [--seeds 5]
    python research/living_mind_2026_11_22/t_asked.py echo      [--seeds 5]
    python research/living_mind_2026_11_22/t_asked.py unknown   [--seeds 5]
    python research/living_mind_2026_11_22/t_asked.py twice     [--seeds 5]

The world of ``t_grow.py``: 48 things, seven kinds of fact ("X V Q ."), every fact of every thing read once as a
sentence. A kind of question is mastered when the mind watched it asked and answered about 8 things, twice. Here
the answer is **the value alone** ("X ne Q ?" -> "V ."): the thing asked about is not named in the answer, which
is where F41 and F53's ``LINK`` do not read (they go on from the thing's name).

``forms``: the mind has mastered k kinds (k = 0, 1, 2, 4); then a new kind of question appears. After n = 0, 1, 2, 4
watched dialogues of the new kind, 8 fresh things are asked it on copies (right / wrong / silent); at the end 8
fresh things are asked one mastered kind. With F41; with F41, F53 and F54; with F41 and F55; with all four; and
with all four and F49 and F50 (with F46 and F47, which F50 needs; there is nowhere to look), where what was lived
of the thing itself stands against what the watched things merely share.

``echo``: the same with answers that name the thing ("X ne Q ?" -> "X V Q ."), the world of ``t_grow.py forms``:
what F55 changes there.

``unknown``: k = 4, no dialogue of the new kind, and for half of the fresh things the fact of the new kind was
never read: what the mind says of those.

``twice``: k = 2 and two facts of the asked kind were read of every fresh thing ("X V1 Q ." and "X V2 Q ."): the
place that was asked holds two things. What the mind says.

Appends to ``results/asked.jsonl``.
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
import t_grow as G  # noqa: E402

OUT = os.path.join(HERE, "results", "asked.jsonl")
NEW = G.NEW
CONDITIONS = [("F41", {"known": True}),
              ("F41 + F53 + F54", {"known": True, "families": True, "lived_apart": True}),
              ("F41 + F55", {"known": True, "asked_place": True}),
              ("F41 + F53 + F54 + F55", {"known": True, "families": True, "lived_apart": True, "asked_place": True}),
              ("the four with F49 and F50", {"known": True, "families": True, "lived_apart": True, "asked_place": True,
                                             "faults": True, "open_claims": True, "look": True, "find_out": True})]
DIALOGUES = [0, 1, 2, 4]


def answer(x: str, value: str, q: str, echo: bool) -> str:
    return "%s %s %s ." % (x, value, q) if echo else "%s ." % value


def raised(seed: int, k: int, more: dict, echo: bool, unread=(), twice=()):
    """The mind that has read every fact and mastered ``k`` kinds of question; (session, things, fact, kinds)."""
    from zihin import IMITATED
    rng = random.Random(seed)
    things = list(T.THINGS)
    rng.shuffle(things)
    fact = {(x, q): rng.choice(G.VALUES[q]) for x in things for q in G.VERBS}
    other = {x: rng.choice([v for v in G.VALUES[NEW] if v != fact[(x, NEW)]]) for x in things}
    mind, s = T.make_mind(seed, False, None, dict(more))
    known = [q for q in G.VERBS if q != NEW][:k]

    def read(sentence):
        s.company(None)
        s.listen(sentence, IMITATED)
        s.end_of_turn(IMITATED, tone=".")
        mind.step([])
    for x in things:
        for q in known + [NEW]:
            if q == NEW and x in unread:
                continue
            read("%s %s %s ." % (x, fact[(x, q)], q))
        if x in twice:
            read("%s %s %s ." % (x, other[x], NEW))
    for _ in range(2):
        for q in known:
            for x in things[:8]:
                s.watch("%s ne %s ?" % (x, q), answer(x, fact[(x, q)], q, echo))
    s.close_turn()
    return s, things, fact, known, other


def forms(args, echo: bool = False) -> None:
    rows = []
    for label, more in CONDITIONS:
        for k in (0, 1, 2, 4):
            at = {n: {"right": 0, "wrong": 0, "silent": 0} for n in DIALOGUES}
            old = {"right": 0, "wrong": 0, "silent": 0}
            first = None
            for seed in range(args.seeds):
                s, things, fact, known, _ = raised(seed, k, more, echo)
                shown, marks_by_n = 0, {}
                for n in DIALOGUES:
                    while shown < n:
                        x = things[8 + shown]
                        s.watch("%s ne %s ?" % (x, NEW), answer(x, fact[(x, NEW)], NEW, echo))
                        s.close_turn()
                        shown += 1
                    marks = "".join(T.probe(s, "%s ne %s ?" % (x, NEW), answer(x, fact[(x, NEW)], NEW, echo))
                                    for x in things[24:32])
                    marks_by_n[n] = marks
                    for key, c in T.count(marks).items():
                        at[n][key] += c
                if known:
                    q = known[0]
                    marks = "".join(T.probe(s, "%s ne %s ?" % (x, q), answer(x, fact[(x, q)], q, echo))
                                    for x in things[32:40])
                    for key, c in T.count(marks).items():
                        old[key] += c
                first = first or " ".join("%d:%s" % (n, marks_by_n[n]) for n in DIALOGUES)
            rows.append({"test": "asked_echo" if echo else "asked_forms", "condition": label, "mastered": k,
                         "seeds": args.seeds, "fresh_after": at, "a_mastered_kind": old, "first": first})
    lines = ["== a new kind of question, answered with %s, after the mind has mastered k kinds: %d seeds; 8 fresh "
             "things asked it on copies after n watched dialogues of the new kind (right / wrong / silent) ==" % (
                 "the thing's name and the value (\"X V Q .\")" if echo else "the value alone (\"V .\")", args.seeds)]
    for row in rows:
        o = row["a_mastered_kind"]
        lines.append("  %-25s k=%d  %s ; a mastered kind, 8 fresh things %d/%d/%d ; the first seed %s" % (
            row["condition"], row["mastered"],
            "  ".join("n=%d %d/%d/%d" % (n, row["fresh_after"][n]["right"], row["fresh_after"][n]["wrong"],
                                         row["fresh_after"][n]["silent"]) for n in DIALOGUES),
            o["right"], o["wrong"], o["silent"], row["first"]))
    T.finish(args, rows, lines)


def unknown(args) -> None:
    from zihin.__main__ import Session
    rows = []
    for label, more in CONDITIONS:
        read = {"right": 0, "wrong": 0, "silent": 0}
        unread = {"a value": 0, "nothing": 0}
        first = []
        for seed in range(args.seeds):
            rng = random.Random(seed)
            things = list(T.THINGS)
            rng.shuffle(things)
            missing = set(things[28:32])
            s, things, fact, _, _ = raised(seed, 4, more, False, unread=missing)
            for x in things[24:32]:
                copy = Session(s.mind.clone(), s.pieces)
                said = copy.reply("%s ne %s ?" % (x, NEW), limit=30)
                if x in missing:
                    unread["nothing" if said == "(sessiz)" else "a value"] += 1
                else:
                    read[{"R": "right", "W": "wrong", "-": "silent"}[T.verdict(s, said, fact[(x, NEW)] + " .")]] += 1
                if seed == 0:
                    first.append("%s%s -> %s" % (x, " (never read)" if x in missing else "", said))
        rows.append({"test": "asked_unknown", "condition": label, "seeds": args.seeds, "fact_read": read,
                     "fact_never_read": unread, "first": first})
    lines = ["== a new kind of question at first sight (the value alone), the mind having mastered 4 kinds; of half "
             "of the fresh things the fact was never read: %d seeds ==" % args.seeds]
    for row in rows:
        r, u = row["fact_read"], row["fact_never_read"]
        lines.append("  %-25s the fact was read: %d/%d/%d (right / wrong / silent) ; never read: a value %d, nothing %d"
                     % (row["condition"], r["right"], r["wrong"], r["silent"], u["a value"], u["nothing"]))
    lines.append("  the first seed with all four: %s" % " | ".join(rows[-2]["first"]))
    T.finish(args, rows, lines)


def twice(args) -> None:
    from zihin.__main__ import Session
    rows = []
    for label, more in CONDITIONS:
        said_as = {"one of the two": 0, "both": 0, "another word": 0, "nothing": 0}
        first = []
        for seed in range(args.seeds):
            rng = random.Random(seed)
            things = list(T.THINGS)
            rng.shuffle(things)
            two = set(things[24:32])
            s, things, fact, _, other = raised(seed, 2, more, False, twice=two)
            for x in things[8:10]:                          # the new kind, watched about two things with one fact
                s.watch("%s ne %s ?" % (x, NEW), "%s ." % fact[(x, NEW)])
                s.close_turn()
            for x in things[24:32]:
                copy = Session(s.mind.clone(), s.pieces)
                said = copy.reply("%s ne %s ?" % (x, NEW), limit=30)
                words = [] if said == "(sessiz)" else said.split()
                pair = {fact[(x, NEW)], other[x]}
                key = "nothing" if not words else ("both" if pair <= set(words) else (
                    "one of the two" if set(words) & pair and not set(words) - pair else "another word"))
                said_as[key] += 1
                if seed == 0:
                    first.append("%s (%s, %s) -> %s" % (x, fact[(x, NEW)], other[x], said))
        rows.append({"test": "asked_twice", "condition": label, "seeds": args.seeds, "said": said_as, "first": first})
    lines = ["== two facts of the asked kind read of every fresh thing (\"X V1 Q .\", \"X V2 Q .\"); the kind was "
             "watched about two things with one fact: %d seeds, 8 fresh things ==" % args.seeds]
    for row in rows:
        d = row["said"]
        lines.append("  %-25s one of the two %d ; both %d ; another word %d ; nothing %d" % (
            row["condition"], d["one of the two"], d["both"], d["another word"], d["nothing"]))
    lines.append("  the first seed with all four: %s" % " | ".join(rows[-2]["first"]))
    T.finish(args, rows, lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["forms", "echo", "unknown", "twice"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
    {"forms": forms, "echo": lambda a: forms(a, True), "unknown": unknown, "twice": twice}[args.what](args)


if __name__ == "__main__":
    main()
