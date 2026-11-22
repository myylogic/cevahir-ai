# -*- coding: utf-8 -*-
"""t_find_out.py: F50 Finding Out, measured. What the mind does not know it acts on, in the turn it is asked in.

    python research/living_mind_2026_11_22/t_find_out.py asked   [--seeds 5]
    python research/living_mind_2026_11_22/t_find_out.py half    [--seeds 5]
    python research/living_mind_2026_11_22/t_find_out.py living  [--seeds 5]
    python research/living_mind_2026_11_22/t_find_out.py again   (the parts of t_fault.py and t_fault_kinds.py with F50 on)

The world of ``t_fault.py loop``: 8 things were read of ("X V yer .") and watched being asked ("X ne yer ?" ->
"X V yer ."); they share one value. Of the other things the mind has read nothing. A library holds one sentence
about every thing (``half``: about every second thing only), and the mind may look into it by its own decision.

``asked``: each of 16 new things is asked **once**. Whatever the mind says in that turn (its own look and what it
says after it included) is its answer; then the true dialogue is shown. Counted: whole answer right / the right
value but not the rest / another value / nothing; and the looks it made.
Conditions (the same life in each): off; F46 + F47 + F49 (looks only in later silent moments, as before);
F46 + F47 + F49 + F50; the same with F52 (the echo rule in its exact form); the same with F41 (the mind goes on with
what is known of the thing and not said yet: the rest of the sentence it read).

``half``: the same with a library that holds nothing about every second thing: what the mind does when its look
finds nothing.

``living``: the turn ends at once and the look is fetched while the mind goes on living (``Session.background``):
between the question and the arrival of what it sent for, the mind is asked something else and answers it; then it
reads and says its late answer. In the first three turns the partner says a word right after the mind began to look
("bakalım"); counted afterwards: how often the mind says that word by itself in that state, the answers given
meanwhile, the late answers. With F50, and with F50 and F41.

``again``: F50 changes what the mind says also where it has nowhere to look (it holds a word that rests on what
others share when its own record says the thing is unknown to it, and what it lived of the thing stands against
what the others share). The parts of ``t_fault.py`` and ``t_fault_kinds.py`` are run once more with F50 on in every
condition that has F49 on, and compared count by count with the committed rows (``results/fault.jsonl``,
``results/fault_kinds.jsonl``). Only the loop's world has a library.

Appends to ``results/find_out.jsonl``.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import t_fault as T  # noqa: E402

OUT = os.path.join(HERE, "results", "find_out.jsonl")
BASE = {"open_claims": True, "look": True}
CONDITIONS = [("off", False, {}),
              ("F46 + F47 + F49 (looks in later silences only)", True, dict(BASE)),
              ("F46 + F47 + F49 + F50", True, dict(BASE, find_out=True)),
              ("F46 + F47 + F49 + F50 + F52 (exact echo)", True, dict(BASE, find_out=True, echo_exact=True)),
              ("F46 + F47 + F49 + F50 + F41 (known, unsaid)", True, dict(BASE, find_out=True, known=True))]


def raise_mind(seed: int, faults: bool, more: dict, held=None):
    """The mind of the loop world and its session, with a library it may look into; returns (session, things, food)."""
    from zihin import IMITATED
    rng = random.Random(seed)
    things = list(T.THINGS)
    rng.shuffle(things)
    members = things[:8]
    values = T.FRAMES["V1"][2]
    food = {x: values[0] if x in members else rng.choice(values) for x in things}
    mind, s = T.make_mind(seed, faults, None, dict(more))
    library = {x: "%s %s yer ." % (x, food[x]) for x in things if held is None or held(things.index(x))}
    s.look_gap = 0.0
    s.look_source = T.Library(library)
    s.may_look = bool(more.get("look"))
    for _ in range(2):
        for x in members:
            s.company(None)
            s.listen("%s %s yer ." % (x, food[x]), IMITATED)
            s.end_of_turn(IMITATED, tone=".")
            mind.step([])
            s.watch("%s ne yer ?" % x, "%s %s yer ." % (x, food[x]))
        s.watch("merhaba", "merhaba .")
    s.close_turn()
    return s, things, food


def tally(marks: str) -> dict:
    return {"right": marks.count("R"), "key": marks.count("K"), "wrong": marks.count("W"), "silent": marks.count("-")}


def asked(args, half: bool = False) -> None:
    rows = []
    for label, faults, more in CONDITIONS:
        total = {"right": 0, "key": 0, "wrong": 0, "silent": 0}
        there = {"right": 0, "key": 0, "wrong": 0, "silent": 0}       # half: the things the library holds
        looks, first = 0, None
        for seed in range(args.seeds):
            s, things, food = raise_mind(seed, faults, more, (lambda i: i % 2 == 0) if half else None)
            marks = ""
            for x in things[8:24]:
                q, a = "%s ne yer ?" % x, "%s %s yer ." % (x, food[x])
                mark = T.key_verdict(s, s.reply(q, limit=30), a)
                marks += mark
                if half and things.index(x) % 2 == 0:
                    there[{"R": "right", "K": "key", "W": "wrong", "-": "silent"}[mark]] += 1
                s.close_turn()
                s.watch(q, a)
                s.close_turn()
            looks += len(s.looks)
            for key, n in tally(marks).items():
                total[key] += n
            first = first or marks
        rows.append({"test": "find_out_half" if half else "find_out_asked", "condition": label, "seeds": args.seeds,
                     "answers": total, "held_by_the_library": there if half else None, "looks": looks,
                     "first": first})
    lines = ["== asked once about each of 16 things it has read nothing of%s: %d seeds; the answer of that one turn "
             "(whole answer right / the right value but not the rest / another value / nothing) ==" % (
                 "; the library holds every second thing" if half else "", args.seeds)]
    for row in rows:
        a = row["answers"]
        extra = ""
        if half:
            h = row["held_by_the_library"]
            extra = " ; of the things the library holds %d/%d/%d/%d" % (h["right"], h["key"], h["wrong"], h["silent"])
        lines.append("  %-48s %d/%d/%d/%d ; looks of its own %d%s ; the first seed: %s" % (
            row["condition"], a["right"], a["key"], a["wrong"], a["silent"], row["looks"], extra, row["first"]))
    T.finish(args, rows, lines)


class Slow(T.Library):
    """A library that answers only when it is let (a service that takes its time)."""
    gate = threading.Event()

    def search(self, topic: str, limit: int = 2):
        self.gate.wait(10.0)
        return super().search(topic, limit)


def living(args) -> None:
    for condition in (CONDITIONS[2], CONDITIONS[4]):
        living_in(args, condition)


def living_in(args, condition) -> None:
    from zihin import TOLD
    label, faults, more = condition
    at_once = said_it = meanwhile = 0
    late = {"right": 0, "key": 0, "wrong": 0, "silent": 0}
    turns, first = 0, None
    for seed in range(args.seeds):
        s, things, food = raise_mind(seed, faults, more)
        s.look_source = Slow(s.look_source.facts)
        s.background = True
        quiet = lambda *_: None  # noqa: E731
        shown = []
        for i, x in enumerate(things[8:20]):
            q, a = "%s ne yer ?" % x, "%s %s yer ." % (x, food[x])
            Slow.gate.clear()
            first_words = s.reply(q, limit=30)                # the turn ends at once: what it sent for is on its way
            in_flight = s.fetching is not None and s.fetching[2].is_alive()
            other = ""
            if i < 3:                                         # the partner says what one says then
                s.listen("bakalım", TOLD, speaker=s.partner)
                s.end_of_turn(TOLD, tone=".")
            else:
                turns += 1
                at_once += in_flight
                said_it += "bakalım" in first_words.split()
                other = s.reply("merhaba", limit=10)          # it goes on living: something else is asked meanwhile
                meanwhile += other == "merhaba"
            for _ in range(2):
                s.silent_moment(say=quiet)                    # silent moments pass; nothing has come yet
            Slow.gate.set()
            if s.fetching is not None:
                s.fetching[2].join(5.0)
            words = s.speak(s.silent_moment(say=quiet), None, 30)   # it has come: it reads, and answers
            whole = " ".join(first_words.replace("bakalım", "").split() + words)
            mark = T.key_verdict(s, whole, a)
            if i >= 3:
                late[{"R": "right", "K": "key", "W": "wrong", "-": "silent"}[mark]] += 1
            shown.append("%s | %s | %s" % (first_words, other, " ".join(words)))
            s.close_turn()
            s.watch(q, a)
            s.close_turn()
        first = first or shown
    rows = [{"test": "find_out_living", "condition": label, "seeds": args.seeds, "turns": turns,
             "turn_ended_while_fetching": at_once, "said_the_word_it_lived_then": said_it,
             "answered_something_else_meanwhile": meanwhile, "late_answers": late, "first": first}]
    lines = ["== the look fetched while the mind goes on living (%s): %d seeds, 12 things each; in the first 3 turns "
             "the partner says \"bakalım\" right after the mind began to look; counted over the other %d turns ==" % (
                 label, args.seeds, turns),
             "  the turn ended while what it sent for was still on its way: %d of %d" % (at_once, turns),
             "  it said \"bakalım\" by itself as it began to look: %d of %d" % (said_it, turns),
             "  asked \"merhaba\" meanwhile, it answered \"merhaba\": %d of %d" % (meanwhile, turns),
             "  its answer after what it sent for had come (whole / right value / another value / nothing): "
             "%d/%d/%d/%d" % (late["right"], late["key"], late["wrong"], late["silent"]),
             "  the first seed, turn by turn (said at once | answered meanwhile | said after reading):"]
    lines += ["    " + line for line in first]
    T.finish(args, rows, lines)


def again(args) -> None:
    """The parts of t_fault.py and t_fault_kinds.py once more, F50 on wherever F49 is on, against the committed rows."""
    import json
    import tempfile
    import t_fault_kinds as K
    committed = {}
    for name in ("fault.jsonl", "fault_kinds.jsonl"):
        with open(os.path.join(HERE, "results", name), encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["test"] not in ("fault_report", "fault_again"):
                    committed.setdefault(row["test"], []).append(row)
    made = T.make_mind

    def with_f50(seed, on, levels=None, more=None):
        more = dict(more or {})
        if on:
            more.update(open_claims=True, look=True, find_out=True)
        return made(seed, on, levels, more)
    handle, scratch = tempfile.mkstemp(suffix=".jsonl")
    os.close(handle)
    now = argparse.Namespace(seeds=5, out=scratch)
    T.make_mind = with_f50
    try:
        for part in (T.misled, lambda a: T.misled(a, True), T.many, T.valued, T.two, T.loop, K.kinds):
            part(now)
    finally:
        T.make_mind = made
    fresh = {}
    with open(scratch, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            fresh.setdefault(row["test"], []).append(row)
    os.remove(scratch)
    skip = ("summary", "loop_s", "marks", "first")
    lines = ["== the parts of t_fault.py and t_fault_kinds.py run again with F50 on wherever F49 is on (5 seeds each; "
             "nowhere to look except in the loop), against the committed rows =="]
    rows = []
    for test, old_rows in committed.items():
        new_rows = fresh.get(test, [])
        differ = []
        for old, new in zip(old_rows, new_rows):
            changed = {k: [old.get(k), new.get(k)] for k in old if k not in skip and old.get(k) != new.get(k)}
            if changed:
                differ.append((old.get("order", ""), old["condition"], changed))
        lines.append("  %-14s %d rows: %s" % (test, len(old_rows), "the same counts" if not differ and
                                             len(new_rows) == len(old_rows) else "%d rows differ" % len(differ)))
        for order, condition, changed in differ:
            lines.append("      %s %s: committed -> now %s" % (order, condition,
                                                               json.dumps(changed, ensure_ascii=False)[:900]))
        rows.append({"test": "find_out_again", "part": test, "rows": len(old_rows), "differ": len(differ),
                     "changed": [[o, c, ch] for o, c, ch in differ]})
    T.finish(args, rows, lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["asked", "half", "living", "again"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    {"asked": asked, "half": lambda a: asked(a, True), "living": living, "again": again}[args.what](args)


if __name__ == "__main__":
    main()
