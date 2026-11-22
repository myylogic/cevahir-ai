# -*- coding: utf-8 -*-
"""t_grow.py: does the mind learn better as it learns? The same new thing given to the same mind at different
stages of what it has learned. F53 Frames of Frames, measured.

    python research/living_mind_2026_11_22/t_grow.py forms      [--seeds 5]
    python research/living_mind_2026_11_22/t_grow.py exception  [--seeds 5]
    python research/living_mind_2026_11_22/t_grow.py unknown    [--seeds 5]
    python research/living_mind_2026_11_22/t_grow.py cost       [--seeds 3]
    python research/living_mind_2026_11_22/t_grow.py judged     [--seeds 5]

The world: 48 things, seven kinds of fact ("X V Q .": what a thing eats, drinks, likes, wears, reads, draws,
carries; for every kind four values, one at random for each thing). Every fact of every thing is read once, as a
sentence. A kind of question is **mastered** when the mind watched it asked and answered about 8 things, twice
("X ne Q ?" -> "X V Q ."). The mind has mastered k kinds; then a new kind of question appears, about the seventh
kind of fact.

``forms``: after n = 0, 1, 2, 4 watched dialogues of the new kind (about things of their own), 8 fresh things are
asked it on copies (right / wrong / silent), for k = 0, 1, 2, 3, 4, 6. With F41 alone, with F41 and F53, and each
with F54 (a relation counts for a member only as far as it was lived apart from the member's own instance). At the
end 8 fresh things are asked one of the mastered kinds too (nothing may be lost there).

``exception``: the new kind of question looks like the others and is answered otherwise: with the same two words,
whatever the thing ("X ne bilir ?" -> "hiçbir şey ."). What a mind that mastered 4 kinds says at first sight, and
how many dialogues it takes to say the true answer, against a mind that mastered none.

``unknown``: as ``forms`` with k = 4 and no dialogue of the new kind, but for half of the fresh things the fact of
the new kind was never read: what the mind says of those.

``cost``: the seconds a life and its answers take with and without F53 (k = 6).

``judged``: does the mind learn how far its own generalising deserves to be believed? It has mastered 3 kinds of
question; then four new kinds appear one after another, each in the teacher's loop (asked about a thing of its own,
then shown the true dialogue whatever it said; 4 rounds a kind). Three worlds: every new kind is answered as the
mastered ones are (``lawful``); every new kind is answered otherwise although its facts were read, each with two
words of its own whatever the thing (``otherwise``); the two in turn (``mixed``). With F41 and F53, with F54 too,
and each with F49 on top (the family's claims are ways of reading with records of their own). Counted: the first
answer of every new
kind (right / wrong / silent), every answer of the loops, and at the end 8 fresh things asked every new kind and
one mastered kind on copies.

Appends to ``results/grow.jsonl``.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import t_fault as T  # noqa: E402

OUT = os.path.join(HERE, "results", "grow.jsonl")
VERBS = ["yer", "içer", "sever", "giyer", "okur", "çizer", "taşır"]
VALUES = {"yer": ["ot", "et", "balık", "tohum"], "içer": ["su", "süt", "çay", "şerbet"],
          "sever": ["oyunu", "uykuyu", "güneşi", "yağmuru"], "giyer": ["kürk", "tüy", "pul", "kabuk"],
          "okur": ["masal", "şiir", "roman", "mektup"], "çizer": ["daire", "kare", "üçgen", "yıldız"],
          "taşır": ["dal", "taş", "yaprak", "kum"]}
NEW = "taşır"
CONDITIONS = [("F41", {"known": True}), ("F41 + F53", {"known": True, "families": True}),
              ("F41 + F54", {"known": True, "lived_apart": True}),
              ("F41 + F53 + F54", {"known": True, "families": True, "lived_apart": True})]
DIALOGUES = [0, 1, 2, 4]


def raised(seed: int, k: int, more: dict, unread=()):
    """The mind that has read every fact and mastered ``k`` kinds of question; (session, things, fact)."""
    from zihin import IMITATED
    rng = random.Random(seed)
    things = list(T.THINGS)
    rng.shuffle(things)
    fact = {(x, q): rng.choice(VALUES[q]) for x in things for q in VERBS}
    mind, s = T.make_mind(seed, False, None, dict(more))
    known = [q for q in VERBS if q != NEW][:k]
    for x in things:
        for q in known + [NEW]:
            if q == NEW and x in unread:
                continue
            s.company(None)
            s.listen("%s %s %s ." % (x, fact[(x, q)], q), IMITATED)
            s.end_of_turn(IMITATED, tone=".")
            mind.step([])
    for _ in range(2):
        for q in known:
            for x in things[:8]:
                s.watch("%s ne %s ?" % (x, q), "%s %s %s ." % (x, fact[(x, q)], q))
    s.close_turn()
    return s, things, fact, known


def forms(args) -> None:
    rows = []
    for label, more in CONDITIONS:
        for k in (0, 1, 2, 3, 4, 6):
            at = {n: {"right": 0, "wrong": 0, "silent": 0} for n in DIALOGUES}
            old = {"right": 0, "wrong": 0, "silent": 0}
            first = None
            for seed in range(args.seeds):
                s, things, fact, known = raised(seed, k, more)
                shown, marks_by_n = 0, {}
                for n in DIALOGUES:
                    while shown < n:
                        x = things[8 + shown]
                        s.watch("%s ne %s ?" % (x, NEW), "%s %s %s ." % (x, fact[(x, NEW)], NEW))
                        s.close_turn()
                        shown += 1
                    marks = "".join(T.probe(s, "%s ne %s ?" % (x, NEW), "%s %s %s ." % (x, fact[(x, NEW)], NEW))
                                    for x in things[24:32])
                    marks_by_n[n] = marks
                    for key, c in T.count(marks).items():
                        at[n][key] += c
                if known:
                    q = known[0]
                    marks = "".join(T.probe(s, "%s ne %s ?" % (x, q), "%s %s %s ." % (x, fact[(x, q)], q))
                                    for x in things[32:40])
                    for key, c in T.count(marks).items():
                        old[key] += c
                first = first or " ".join("%d:%s" % (n, marks_by_n[n]) for n in DIALOGUES)
            rows.append({"test": "grow_forms", "condition": label, "mastered": k, "seeds": args.seeds,
                         "fresh_after": at, "a_mastered_kind": old, "first": first})
    lines = ["== a new kind of question, after the mind has mastered k kinds: %d seeds; 8 fresh things asked it on "
             "copies after n watched dialogues of the new kind (right / wrong / silent) ==" % args.seeds]
    for row in rows:
        o = row["a_mastered_kind"]
        lines.append("  %-15s k=%d  %s ; a mastered kind, 8 fresh things %d/%d/%d ; the first seed %s" % (
            row["condition"], row["mastered"],
            "  ".join("n=%d %d/%d/%d" % (n, row["fresh_after"][n]["right"], row["fresh_after"][n]["wrong"],
                                         row["fresh_after"][n]["silent"]) for n in DIALOGUES),
            o["right"], o["wrong"], o["silent"], row["first"]))
    T.finish(args, rows, lines)


def exception(args) -> None:
    rounds = [0, 1, 2, 4, 8]
    rows = []
    for label, more in CONDITIONS:
        for k in (0, 4):
            at = {n: {"right": 0, "wrong": 0, "silent": 0} for n in rounds}
            for seed in range(args.seeds):
                s, things, fact, _ = raised(seed, k, more)
                shown = 0
                for n in rounds:
                    while shown < n:
                        s.watch("%s ne bilir ?" % things[8 + shown], "hiçbir şey .")
                        s.close_turn()
                        shown += 1
                    for x in things[24:32]:
                        mark = T.probe(s, "%s ne bilir ?" % x, "hiçbir şey .")
                        at[n]["right" if mark == "R" else ("wrong" if mark == "W" else "silent")] += 1
            rows.append({"test": "grow_exception", "condition": label, "mastered": k, "seeds": args.seeds,
                         "fresh_after": at})
    lines = ["== a kind of question that looks like the others and is answered otherwise (\"X ne bilir ?\" -> "
             "\"hiçbir şey .\"): %d seeds; 8 fresh things after n watched dialogues (right / wrong / silent) ==" %
             args.seeds]
    for row in rows:
        lines.append("  %-15s k=%d  %s" % (row["condition"], row["mastered"], "  ".join(
            "n=%d %d/%d/%d" % (n, row["fresh_after"][n]["right"], row["fresh_after"][n]["wrong"],
                               row["fresh_after"][n]["silent"]) for n in rounds)))
    T.finish(args, rows, lines)


def unknown(args) -> None:
    from zihin.__main__ import Session
    rows = []
    for label, more in CONDITIONS:
        read = {"right": 0, "wrong": 0, "silent": 0}
        unread = {"only the asked word": 0, "a value": 0, "nothing": 0}
        first = []
        for seed in range(args.seeds):
            rng = random.Random(seed)
            things = list(T.THINGS)
            rng.shuffle(things)
            missing = set(things[28:32])
            s, things, fact, _ = raised(seed, 4, more, missing)
            for x in things[24:32]:
                copy = Session(s.mind.clone(), s.pieces)
                said = copy.reply("%s ne %s ?" % (x, NEW), limit=30)
                if x in missing:
                    words = [] if said == "(sessiz)" else said.split()
                    key = "nothing" if not words else ("only the asked word" if words == [x] else "a value")
                    unread[key] += 1
                else:
                    read[{"R": "right", "W": "wrong", "-": "silent"}[
                        T.verdict(s, said, "%s %s %s ." % (x, fact[(x, NEW)], NEW))]] += 1
                if seed == 0:
                    first.append("%s%s -> %s" % (x, " (never read)" if x in missing else "", said))
        rows.append({"test": "grow_unknown", "condition": label, "seeds": args.seeds, "fact_read": read,
                     "fact_never_read": unread, "first": first})
    lines = ["== a new kind of question at first sight, the mind having mastered 4 kinds; of half of the fresh "
             "things the fact was never read: %d seeds ==" % args.seeds]
    for row in rows:
        r, u = row["fact_read"], row["fact_never_read"]
        lines.append("  %-15s the fact was read: %d/%d/%d (right / wrong / silent) ; never read: only the asked word "
                     "%d, a value %d, nothing %d" % (row["condition"], r["right"], r["wrong"], r["silent"],
                                                     u["only the asked word"], u["a value"], u["nothing"]))
    lines.append("  the first seed with F53: %s" % " | ".join(rows[-1]["first"]))
    T.finish(args, rows, lines)


def cost(args) -> None:
    rows = []
    for label, more in CONDITIONS:
        life = answers = 0.0
        moments = 0
        for seed in range(args.seeds):
            t0 = time.time()
            s, things, fact, known = raised(seed, 6, more)
            life += time.time() - t0
            moments += s.mind.t
            t0 = time.time()
            for q in known + [NEW]:
                for x in things[24:32]:
                    T.probe(s, "%s ne %s ?" % (x, q), "%s %s %s ." % (x, fact[(x, q)], q))
            answers += time.time() - t0
        rows.append({"test": "grow_cost", "condition": label, "seeds": args.seeds, "life_s": round(life, 1),
                     "moments": moments, "answers_s": round(answers, 1), "answers": 56 * args.seeds})
    lines = ["== what it costs (6 kinds mastered): %d seeds ==" % args.seeds]
    for row in rows:
        lines.append("  %-15s the life %.1f s (%d moments) ; %d answers on copies %.1f s (%.2f s an answer)" % (
            row["condition"], row["life_s"], row["moments"], row["answers"], row["answers_s"],
            row["answers_s"] / row["answers"]))
    T.finish(args, rows, lines)


OTHERWISE = {"giyer": "hiçbir şey .", "okur": "kim bilir .", "çizer": "belli olmaz .", "taşır": "orası karışık ."}
JUDGED = [("F41 + F53", {"known": True, "families": True}),
          ("F41 + F53 + F54", {"known": True, "families": True, "lived_apart": True}),
          ("F41 + F53 + F49", {"known": True, "families": True, "faults": True}),
          ("F41 + F53 + F54 + F49", {"known": True, "families": True, "lived_apart": True, "faults": True})]
WORLDS = [("lawful", ()), ("otherwise", ("giyer", "okur", "çizer", "taşır")), ("mixed", ("giyer", "çizer"))]


def judged(args) -> None:
    from zihin import IMITATED
    mastered, new = VERBS[:3], VERBS[3:]
    rounds = 4
    rows = []
    for world, otherwise in WORLDS:
        for label, more in JUDGED:
            first = {q: {"right": 0, "wrong": 0, "silent": 0} for q in new}
            loops = {q: {"right": 0, "wrong": 0, "silent": 0} for q in new}
            end = {q: {"right": 0, "wrong": 0, "silent": 0} for q in new + [mastered[0]]}
            marks_first = None
            for seed in range(args.seeds):
                rng = random.Random(seed)
                things = list(T.THINGS)
                rng.shuffle(things)
                fact = {(x, q): rng.choice(VALUES[q]) for x in things for q in VERBS}
                mind, s = T.make_mind(seed, False, None, dict(more))

                def qa(x, q):
                    return "%s ne %s ?" % (x, q), (OTHERWISE[q] if q in otherwise else
                                                   "%s %s %s ." % (x, fact[(x, q)], q))
                for x in things:
                    for q in VERBS:
                        s.company(None)
                        s.listen("%s %s %s ." % (x, fact[(x, q)], q), IMITATED)
                        s.end_of_turn(IMITATED, tone=".")
                        mind.step([])
                for _ in range(2):
                    for q in mastered:
                        for x in things[:8]:
                            s.watch(*qa(x, q))
                s.close_turn()
                marks = {}
                for j, q in enumerate(new):
                    marks[q] = "".join(T.teacher(s, *qa(x, q)) for x in things[8 + rounds * j:8 + rounds * (j + 1)])
                    first[q][{"R": "right", "W": "wrong", "-": "silent"}[marks[q][0]]] += 1
                    for key, n in T.count(marks[q]).items():
                        loops[q][key] += n
                for q in new + [mastered[0]]:
                    for key, n in T.count("".join(T.probe(s, *qa(x, q)) for x in things[24:32])).items():
                        end[q][key] += n
                marks_first = marks_first or " ".join("%s:%s" % (q, marks[q]) for q in new)
            rows.append({"test": "grow_judged", "world": world, "answered_otherwise": list(otherwise),
                         "condition": label, "seeds": args.seeds, "rounds": rounds, "first_answer": first,
                         "loops": loops, "fresh_at_the_end": end, "first": marks_first})
    lines = ["== new kinds of question one after another, the mind having mastered 3 kinds: %d seeds; each new kind "
             "in the teacher's loop for %d rounds (right / wrong / silent) ==" % (args.seeds, rounds)]
    for row in rows:
        lines.append("  %-9s %-21s the first answer of each new kind: %s" % (
            row["world"], row["condition"], "  ".join("%s %d/%d/%d" % (
                q, row["first_answer"][q]["right"], row["first_answer"][q]["wrong"], row["first_answer"][q]["silent"])
                for q in new)))
        lines.append("  %-9s %-21s all rounds: %s ; fresh at the end: %s ; the first seed %s" % (
            "", "", "  ".join("%s %d/%d/%d" % (q, row["loops"][q]["right"], row["loops"][q]["wrong"],
                                               row["loops"][q]["silent"]) for q in new),
            "  ".join("%s %d/%d/%d" % (q, r["right"], r["wrong"], r["silent"])
                      for q, r in row["fresh_at_the_end"].items()), row["first"]))
    T.finish(args, rows, lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["forms", "exception", "unknown", "cost", "judged"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
    {"forms": forms, "exception": exception, "unknown": unknown, "cost": cost, "judged": judged}[args.what](args)


if __name__ == "__main__":
    main()
