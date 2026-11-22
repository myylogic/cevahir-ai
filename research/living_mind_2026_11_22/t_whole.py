# -*- coding: utf-8 -*-
"""t_whole.py: the three mechanisms of this round in one mind, nothing set by hand.

    python research/living_mind_2026_11_22/t_whole.py life [--seeds 5]

One life in the three-field world of ``t_conj.py`` (the classes' statements, every thing's class, own fact and
other link; the class's two statements told of six animals only), with two more facts of its own read of every
animal (what it drinks, what it likes). The mind reads everything, a silent moment after each sentence and silent
moments at the end until it has nothing left to think over. Then it is asked, in the teacher's loop (whatever it
says, the true dialogue is shown; every answer the value alone), five kinds of question about the twelve animals,
each animal one question: what four of them eat, what two drink, what two like, and, of the six that were told
their class's statements, what two bear and what two are covered with. Nothing is asked about any body or number,
and nothing was told of them but their classes, their own facts and their other links.

(``--same``: the first form of this life, kept as it was measured: the five kinds are all asked of the same six
animals, six rounds each.)

Then, on copies: every body and number is asked its class's two statements ("demir ne iletir ?"), and the six
other animals theirs: kinds of question that were never asked in those fields, about facts nobody told.

Conditions (the same life):

* ``as it was``: the speaking configuration with F49;
* ``trials alone``: F57 and F58 (the mind sets its ways of reading itself), no conjectures;
* ``conjectures alone``: F59, no trials;
* ``the three together``: F57, F58 and F59; no way of reading is turned on by hand in any condition.

Appends to ``results/whole.jsonl``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import t_conj as C  # noqa: E402
import t_fault as T  # noqa: E402
import t_self as S  # noqa: E402

OUT = os.path.join(HERE, "results", "whole.jsonl")
MORE = {"içer": ["su", "süt", "çay", "şerbet"], "sever": ["güneşi", "yağmuru", "oyunu", "uykuyu"]}
ASKED = ("yer", "içer", "sever", "doğurur", "örtünür")


def the_world(seed: int) -> C.World:
    """The world of ``t_conj.py`` with two more facts read of every animal."""
    import random
    world = C.World(seed)
    rng = random.Random(seed + 7)
    for x in world.things["animals"]:
        for q, values in MORE.items():
            world.sentences.append((x, rng.choice(values), q))
    rng.shuffle(world.sentences)
    return world
CONDITIONS = [("as it was", {"faults": True}),
              ("trials alone", {"faults": True, "trials": True, "built_relations": True}),
              ("conjectures alone", {"faults": True, "conjectures": True}),
              ("the three together", {"faults": True, "trials": True, "built_relations": True, "conjectures": True})]


def life(args) -> None:
    from zihin.__main__ import Session
    rows, lines = [], ["== one life, the three fields: read, then asked about animals in the teacher's loop (%s); then "
                       "asked, on copies, what was told of no thing (right / wrong / silent): %d seeds ==" % (
                           "every kind of the same six" if args.same else "each animal one question", args.seeds)]
    for label, switches in CONDITIONS:
        asked = {field: {"right": 0, "wrong": 0, "silent": 0} for field in C.FIELDS}
        loops, ways, made, hold, took = {}, {}, 0, 0, 0.0
        first = None
        for seed in range(args.seeds):
            world = the_world(seed)
            t0 = time.time()
            s, _, _ = C.live(world, dict(switches))
            mind = s.mind
            animals = [x for x in world.things["animals"] if any(fact[0] == x for fact in world.told)]
            told = {}
            for words in world.sentences:                    # the animal's own facts, as read
                if words[0] in animals and words[2] in ("yer",) + tuple(MORE):
                    told[(words[0], words[2])] = words[1]
            for fact in world.told:
                told[(fact[0], fact[2])] = fact[1]
            others = [x for x in world.things["animals"] if x not in animals]
            for words in world.sentences:
                if words[0] in others and words[2] in ("yer",) + tuple(MORE):
                    told[(words[0], words[2])] = words[1]
            plan = ([(q, animals) for q in ASKED] if args.same else
                    [("yer", others[:4]), ("içer", others[4:6]), ("sever", animals[:2]), ("doğurur", animals[2:4]),
                     ("örtünür", animals[4:6])])
            marks_of = []
            for q, of in plan:
                marks = ""
                for x in of:
                    marks += S.teacher(s, "%s ne %s ?" % (x, q), "%s ." % told[(x, q)])
                    s.quiet(1)
                S.add(loops.setdefault(q, {}), marks)
                marks_of.append("%s:%s" % (q, marks))
            s.quiet(400, until_done=True)
            took += time.time() - t0
            for fact in sorted(world.untold()):
                copy = Session(mind.clone(), s.pieces)
                copy.partner, copy.dil = s.partner, s.dil
                said = copy.reply("%s ne %s ?" % (fact[0], fact[2]), limit=30)
                key = "right" if said == fact[1] else ("silent" if said == "(sessiz)" else "wrong")
                asked[world.field_of[fact[0]]][key] += 1
            state = S.in_use(mind)
            for way in state["ways"] + [S.name_of(p) for p in state["relations"]]:
                ways[way] = ways.get(way, 0) + 1
            mine = C.conjectured(world, mind)
            made += len(mine)
            hold += len(mine & world.holds)
            first = first or " ".join(marks_of)
        rows.append({"test": "whole_life", "condition": label, "seeds": args.seeds,
                     "asked_about": "the same six animals" if args.same else "each animal once",
                     "asked": asked, "loops": loops,
                     "in_use_at_the_end": ways, "conjectures": made, "conjectures_that_hold": hold,
                     "life_s": round(took, 1), "first": first})
        lines.append("  %-20s %s" % (label, "  ".join("%s %d/%d/%d" % (f, c["right"], c["wrong"], c["silent"])
                                                     for f, c in asked.items())))
        lines.append("  %-20s conjectures %d (hold %d) ; in use at the end (lives of %d): %s ; the rounds it was asked: "
                     "%s ; %.0f s" % ("", made, hold, args.seeds, ", ".join("%s %d" % kv for kv in ways.items())
                                      or "nothing", "  ".join("%s %s" % (q, S.rws(v)) for q, v in loops.items()),
                                      took))
        lines.append("  %-20s the first seed: %s" % ("", first))
    T.finish(args, rows, lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["life"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--same", action="store_true", help="every kind asked of the same six animals")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
    used = {w for f in C.FIELDS.values() for w in f["things"] + f["classes"] + f["own"][1] + [f["own"][0]]
            + [v for vs in f["of"].values() for v in vs] + list(f["of"])}
    assert not used & ({v for vs in MORE.values() for v in vs} | set(MORE))
    life(args)


if __name__ == "__main__":
    main()
