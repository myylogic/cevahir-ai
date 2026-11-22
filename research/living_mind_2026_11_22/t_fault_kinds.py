# -*- coding: utf-8 -*-
"""t_fault_kinds.py: F49's restructuring step, measured. A frame found wanting is read through a narrower place of
the thing asked about.

    python research/living_mind_2026_11_22/t_fault_kinds.py [--seeds 5]
    python research/living_mind_2026_11_22/t_fault_kinds.py again        (the parts of t_fault.py, run again)

What a thing eats follows its kind (four kinds, four foods), and the kind of every thing was read once in a sentence
("X K bir hayvandır ."). The 8 watched dialogues ("X ne yer ?" -> "X V yer .") are all of one kind, so the mind's
frame generalises one food to everything. Then 24 rounds of the teacher's loop (asked, then shown the true dialogue
whatever it said: ``t_fault.teacher``) with new things of all four kinds; 8 fresh things, two of each kind, are
probed on copies after 0, 4, 8, 16 and 24 rounds. Nothing tells the mind that the kind matters, or that anything
does: it has what it read, its own answers' chains, and what it then lives.

Conditions (the same life in each): F49 off; F49 on with the narrower reading left out (doubt only); F49 on.

``again``: the narrower reading was added after the parts of ``t_fault.py`` were measured and committed
(``results/fault.jsonl``). This runs those parts once more with the code as it is now and says, part by part,
whether any count differs from the committed rows.

Appends to ``results/fault_kinds.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import t_fault as T  # noqa: E402

OUT = os.path.join(HERE, "results", "fault_kinds.jsonl")
FOODS = {"evcil": "ot", "yırtıcı": "et", "sucul": "balık", "kanatlı": "tohum"}
ROUNDS = [0, 4, 8, 16, 24]
CONDITIONS = [("off", False, True), ("on, doubt only (the narrower reading left out)", True, False), ("on", True, True)]


def kinds(args) -> None:
    from zihin import IMITATED
    names = list(FOODS)
    rows = []
    for label, on, narrower in CONDITIONS:
        at = {r: {"right": 0, "wrong": 0, "silent": 0} for r in ROUNDS}
        loop = {"right": 0, "wrong": 0, "silent": 0}
        first = None
        for seed in range(args.seeds):
            rng = random.Random(seed)
            things = list(T.THINGS)
            rng.shuffle(things)
            members = things[:8]
            kind = {x: names[0] if x in members else names[i % 4] for i, x in enumerate(things)}
            mind, s = T.make_mind(seed, on)
            if not narrower:
                mind.__dict__["_fault_narrower"] = False

            def qa(x):
                return "%s ne yer ?" % x, "%s %s yer ." % (x, FOODS[kind[x]])
            for _ in range(2):
                for x in things:                         # the kind of every thing, read once a round
                    s.company(None)
                    s.listen("%s %s bir hayvandır ." % (x, kind[x]), IMITATED)
                    s.end_of_turn(IMITATED, tone=".")
                    mind.step([])
                for x in members:
                    s.watch(*qa(x))
            s.close_turn()
            pool, fresh = things[8:40], things[40:]
            marks = ""
            for i in range(ROUNDS[-1] + 1):
                if i in at:
                    for key, n in T.count("".join(T.probe(s, *qa(x)) for x in fresh)).items():
                        at[i][key] += n
                if i < ROUNDS[-1]:
                    marks += T.teacher(s, *qa(pool[i]))
            for key, n in T.count(marks).items():
                loop[key] += n
            first = first or marks
        rows.append({"test": "fault_kinds", "condition": label, "seeds": args.seeds, "fresh_after": at, "loop": loop,
                     "first": first})
    lines = ["== what a thing eats follows its kind, read in a sentence; the watched dialogues are all of one kind: "
             "%d seeds, 8 fresh things (two of each kind) after n rounds of the teacher's loop "
             "(right / wrong / silent) ==" % args.seeds]
    for row in rows:
        lines.append("  %s: %s ; in the 24 rounds %d/%d/%d ; the first seed: %s" % (
            row["condition"], "  ".join("n=%d %d/%d/%d" % (r, row["fresh_after"][r]["right"],
                                                            row["fresh_after"][r]["wrong"],
                                                            row["fresh_after"][r]["silent"]) for r in ROUNDS),
            row["loop"]["right"], row["loop"]["wrong"], row["loop"]["silent"], row["first"]))
    T.finish(args, rows, lines)


def again(args) -> None:
    """The parts of t_fault.py once more with the code as it is now, compared with the committed rows."""
    committed = {}
    with open(os.path.join(HERE, "results", "fault.jsonl"), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            committed.setdefault(row["test"], []).append(row)
    import tempfile
    handle, scratch = tempfile.mkstemp(suffix=".jsonl")
    os.close(handle)
    now = argparse.Namespace(seeds=5, out=scratch)
    for part in (T.misled, lambda a: T.misled(a, True), T.many, T.valued, T.two, T.loop):
        part(now)
    fresh = {}
    with open(scratch, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            fresh.setdefault(row["test"], []).append(row)
    os.remove(scratch)
    lines = ["== the parts of t_fault.py run again with the narrower reading in the code (5 seeds each), against the "
             "committed rows of results/fault.jsonl =="]
    rows = []
    for test, old_rows in committed.items():
        if test == "fault_report":
            continue
        new_rows = fresh.get(test, [])
        differ = []
        for old, new in zip(old_rows, new_rows):
            keys = [k for k in old if k not in ("summary", "loop_s")]
            if any(old.get(k) != new.get(k) for k in keys):
                differ.append((old.get("order", ""), old["condition"],
                               {k: new.get(k) for k in keys if old.get(k) != new.get(k) and k not in ("marks", "first")}))
        lines.append("  %-14s %d rows: %s" % (test, len(old_rows), "the same counts" if not differ and len(new_rows) ==
                                             len(old_rows) else "%d rows differ" % len(differ)))
        for order, condition, changed in differ:
            lines.append("      %s %s: now %s" % (order, condition, json.dumps(changed, ensure_ascii=False)[:600]))
        rows.append({"test": "fault_again", "part": test, "rows": len(old_rows), "differ": len(differ),
                     "changed": [[o, c, ch] for o, c, ch in differ]})
    T.finish(args, rows, lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="kinds", choices=["kinds", "again"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    (kinds if args.what == "kinds" else again)(args)


if __name__ == "__main__":
    main()
