# -*- coding: utf-8 -*-
"""t_turn.py: F69 Its Own Turn Comes Last (``Config.own_turn``), measured on the whole mind.

    python research/living_mind_2026_11_22/t_turn.py misled    [--seeds 5]
    python research/living_mind_2026_11_22/t_turn.py half      [--seeds 5]
    python research/living_mind_2026_11_22/t_turn.py exception [--seeds 5]
    python research/living_mind_2026_11_22/t_turn.py watched   [--seeds 5]

With silent claims (F49 while the mind only listens) a way of reading earns a record from everything the mind
watches. Section 28 left two lines where the whole mind did worse than the configurations before it, both with
silent claims as the cause: it kept asserting in frames that had misled it (66 wrong answers against 29), and it
asserted a value it could not look up (9 against 1). Two things were found by following those lives:

* When the teacher shows the true dialogue after the mind has answered, the mind listens to the same question
  again, and the silent claim of that moment *replaced* the claim it had kept from its own turn at that context.
  So what it said or held back there was settled as if it had only been listening: a claim it held back was never
  seen to be answered (and each such moment counted as being called on again with the claim still waiting), until
  the mind concluded that holding back is never answered and asserted instead.
* A way's record pooled what its claims came to while the mind listened (what it was shown) with what they came
  to when it was its turn (what it was asked). Eight things shown to it shared a value; the things it was then
  asked about did not. Seven successes while listening outweighed every failure of its own.

F69: what the mind kept from its own turn at a context stays when it listens to that context again, and a moment
of listening is not being called on; the two conditions are booked apart, and the record of its own turn is read
last, over the listening record, as the more specific evidence is everywhere (an instance over its frame, the
person present over the others).

``misled`` (the world of ``t_fault.py misled``, 6 new things a frame): right / wrong / silent in the teacher's
loop and on 8 things never asked about. ``half`` (``t_find_out.py half``): the library holds every second thing.
``exception``: what the rule costs where a frame holds but for one thing: the 8 watched things share a value and
so does every thing asked about but the first. ``watched`` (``t_fault_watch.py``): the mind is never asked, it only
watches; nothing of its own turn exists, so nothing may change.

Every mind is the whole mind (``zihin.__main__.VARIANTS["full"]``) with the changes its condition names. Appends to
``results/turn.jsonl``.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import torch  # noqa: E402

import t_fault as T  # noqa: E402

OUT = os.path.join(HERE, "results", "turn.jsonl")
CONDITIONS = [("the whole mind of section 28 (own_turn off)", {"own_turn": False}, None),
              ("F69: its own turn comes last", {}, None),
              ("F69, only: what it kept from its own turn stays", {}, "held"),
              ("F69, only: the two records apart", {}, "record"),
              ("without silent claims (and what needs them)", {"silent_claims": False, "relied_on": False,
                                                              "own_turn": False}, None)]


class whole_mind:
    """Every mind made inside is the whole mind with ``switches`` changed (as ``t_full.py`` makes them)."""

    def __init__(self, switches: dict) -> None:
        self.switches = switches

    def __enter__(self):
        import zihin.__main__ as chat
        self.chat, self.speaking, switches = chat, chat.VARIANTS["speaking"], self.switches

        def whole(folder=None):
            cfg = chat.full(folder) if folder else chat.full()
            for name, value in switches.items():
                if not hasattr(cfg, name):
                    raise SystemExit("no such switch: %s" % name)
                setattr(cfg, name, value)
            return cfg
        chat.VARIANTS["speaking"] = whole
        return self

    def __exit__(self, *exc) -> None:
        self.chat.VARIANTS["speaking"] = self.speaking


def part_of(mind, part) -> None:
    if part is not None:
        mind.__dict__["_own_turn"] = part                  # (research: the one part of F69 that is on)


def rws(c: dict) -> str:
    return "%d/%d/%d" % (c["right"], c["wrong"], c["silent"])


def misled(args) -> None:
    order = ["V1", "V2", "L1", "V3", "L2"]
    rows = []
    for label, switches, part in CONDITIONS:
        loops = {f: {"right": 0, "wrong": 0, "silent": 0} for f in order}
        fresh_end = {f: {"right": 0, "wrong": 0, "silent": 0} for f in order}
        first = None
        for seed in range(args.seeds):
            things, members, truth = T.world(seed, False)
            with whole_mind(switches):
                mind, s = T.make_mind(seed, True, None)
            part_of(mind, part)

            def qa(f, x):
                q, a, _ = T.FRAMES[f]
                return q % x, a % (x, truth[f, x])
            for _ in range(2):
                for x in members:
                    for f in T.FRAMES:
                        s.watch(*qa(f, x))
            s.close_turn()
            pool, fresh = things[8:], things[40:]
            marks, k = {}, 0
            for f in order:
                marks[f] = "".join(T.teacher(s, *qa(f, x)) for x in pool[k:k + 6])
                k += 6
                for key, n in T.count(marks[f]).items():
                    loops[f][key] += n
            for f in order:
                end = "".join(T.probe(s, *qa(f, x)) for x in fresh)
                for key, n in T.count(end).items():
                    fresh_end[f][key] += n
            first = first or " ".join("%s %s" % (f, marks[f]) for f in order)
        rows.append({"test": "turn_misled", "condition": label, "switches": switches, "part": part,
                     "seeds": args.seeds, "loops": loops, "fresh_end": fresh_end, "first": first})
    vs, ls = ["V1", "V2", "V3"], ["L1", "L2"]
    lines = ["== misled (three frames whose 8 watched things only happen to share a value, two lawful frames): the "
             "teacher's loop, 6 new things a frame, then 8 things never asked about; right / wrong / silent; %d seeds =="
             % args.seeds]
    for row in rows:
        def total(side, frames):
            return {key: sum(row[side][f][key] for f in frames) for key in ("right", "wrong", "silent")}
        lines.append("  %-50s loops: misleading %-9s lawful %-8s ; fresh: misleading %-10s lawful %-8s ; the first "
                     "seed: %s" % (row["condition"], rws(total("loops", vs)), rws(total("loops", ls)),
                                   rws(total("fresh_end", vs)), rws(total("fresh_end", ls)), row["first"]))
    T.finish(args, rows, lines)


def half(args) -> None:
    import t_find_out as F
    rows = []
    for label, switches, part in CONDITIONS:
        total = {"right": 0, "key": 0, "wrong": 0, "silent": 0}
        there = {"right": 0, "key": 0, "wrong": 0, "silent": 0}
        looks, first = 0, None
        for seed in range(args.seeds):
            with whole_mind(switches):
                s, things, food = F.raise_mind(seed, True, dict(F.BASE, find_out=True, known=True),
                                               lambda i: i % 2 == 0)
            part_of(s.mind, part)
            marks = ""
            for x in things[8:24]:
                q, a = "%s ne yer ?" % x, "%s %s yer ." % (x, food[x])
                mark = T.key_verdict(s, s.reply(q, limit=30), a)
                marks += mark
                if things.index(x) % 2 == 0:
                    there[{"R": "right", "K": "key", "W": "wrong", "-": "silent"}[mark]] += 1
                s.close_turn()
                s.watch(q, a)
                s.close_turn()
            looks += len(s.looks)
            for key, n in F.tally(marks).items():
                total[key] += n
            first = first or marks
        rows.append({"test": "turn_half", "condition": label, "switches": switches, "part": part,
                     "seeds": args.seeds, "answers": total, "held_by_the_library": there, "looks": looks,
                     "first": first})
    lines = ["== asked once about each of 16 things it has read nothing of; the library holds every second thing; "
             "the whole answer right / the right value but not the rest / another value / nothing; %d seeds =="
             % args.seeds]
    for row in rows:
        a, h = row["answers"], row["held_by_the_library"]
        lines.append("  %-50s %d/%d/%d/%d ; of the things the library holds %d/%d/%d/%d ; looks of its own %d ; the "
                     "first seed: %s" % (row["condition"], a["right"], a["key"], a["wrong"], a["silent"], h["right"],
                                         h["key"], h["wrong"], h["silent"], row["looks"], row["first"]))
    T.finish(args, rows, lines)


def exception(args) -> None:
    rows = []
    for label, switches, part in CONDITIONS:
        loop = {"right": 0, "wrong": 0, "silent": 0}
        fresh = {"right": 0, "wrong": 0, "silent": 0}
        first = None
        for seed in range(args.seeds):
            things, members, _ = T.world(seed, True)
            q, a, values = T.FRAMES["V1"]
            odd = things[8]                                # the first thing asked about is the one exception
            with whole_mind(switches):
                mind, s = T.make_mind(seed, True, None)
            part_of(mind, part)

            def qa(x):
                return q % x, a % (x, values[1] if x == odd else values[0])
            for _ in range(2):
                for x in members:
                    s.watch(*qa(x))
            s.close_turn()
            marks = "".join(T.teacher(s, *qa(x)) for x in things[8:16])
            end = "".join(T.probe(s, *qa(x)) for x in things[40:])
            for total, m in ((loop, marks), (fresh, end)):
                for key, n in T.count(m).items():
                    total[key] += n
            first = first or "%s ; fresh %s" % (marks, end)
        rows.append({"test": "turn_exception", "condition": label, "switches": switches, "part": part,
                     "seeds": args.seeds, "loop": loop, "fresh": fresh, "first": first})
    lines = ["== a frame that holds but for one thing (the first asked about): the teacher's loop, 8 new things, then "
             "8 things never asked about; right / wrong / silent; %d seeds ==" % args.seeds]
    for row in rows:
        lines.append("  %-50s loop %-8s fresh %-8s ; the first seed: %s" % (
            row["condition"], rws(row["loop"]), rws(row["fresh"]), row["first"]))
    T.finish(args, rows, lines)


def watched(args) -> None:
    import t_fault_watch as W
    rows = []
    lines = ["== never asked, only watching (``t_fault_watch.py``): right / wrong / silent on 8 fresh things after "
             "watching n more dialogues; %d seeds ==" % args.seeds]
    for world in ("kinds", "shared"):
        lines.append("  %s" % world)
        for label, switches, part in CONDITIONS[:2]:
            by_n = {n: {"right": 0, "wrong": 0, "silent": 0} for n in W.WATCHED}
            for seed in range(args.seeds):
                with whole_mind(switches):
                    out, _, _ = W.life(seed, {"faults": True}, world)
                for n, marks in out:
                    for key, v in T.count(marks).items():
                        by_n[n][key] += v
            rows.append({"test": "turn_watched", "world": world, "condition": label, "switches": switches,
                         "seeds": args.seeds, "by_watched": {str(n): c for n, c in by_n.items()}})
            lines.append("    %-50s %s" % (label, "  ".join("after %d: %s" % (n, rws(by_n[n])) for n in W.WATCHED)))
    T.finish(args, rows, lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["misled", "half", "exception", "watched"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    {"misled": misled, "half": half, "exception": exception, "watched": watched}[args.what](args)


if __name__ == "__main__":
    main()
