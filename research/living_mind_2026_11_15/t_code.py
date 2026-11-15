# -*- coding: utf-8 -*-
"""T-code: learning to write code by its consequences (a first, small test).

Each task is a function head ("topla a b") and three completions ("a + b", "a * b", "a - b"), one correct. The
mind first watches others complete the head: the most common completion it sees is a WRONG one (three times), the
correct one once, the other wrong one once, so imitation alone writes wrong code. Then, in rounds, the head is said
to it, it writes a completion (its own acts), the completed function is actually run against a test, and the result
comes back as a value signal: "iyi" if the test passes, "kötü" if not (the session's feedback, F15/L5). Measured: the
share of passing completions per round, with the consequence signal and without it (control: the same rounds, the
code is run but nothing comes back). Nothing about code is built into the core: tokens are words, the test result is
a value signal like any other.

Usage: python research/living_mind_2026_11_15/t_code.py [--seeds 3] [--rounds 8]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import torch  # noqa: E402

from zihin import Config, Mind  # noqa: E402
from zihin.__main__ import CHANNEL, Session  # noqa: E402

TASKS = [
    ("topla", "a b", ["a + b", "a * b", "a - b"], lambda f: f(2, 3) == 5),
    ("carp", "a b", ["a * b", "a + b", "a - b"], lambda f: f(2, 3) == 6),
    ("cikar", "a b", ["a - b", "a + b", "a * b"], lambda f: f(5, 3) == 2),
    ("kare", "x", ["x * x", "x + x", "x - x"], lambda f: f(3) == 9),
    ("iki_kati", "x", ["x + x", "x * x", "x - x"], lambda f: f(3) == 6),
    ("sifir", "x", ["x - x", "x + x", "x * x"], lambda f: f(3) == 0),
]


def passes(name: str, args: str, written: str, test) -> bool:
    """Run the written completion as the body of the function and test it (our own six tasks only)."""
    if not written or written == "(sessiz)":
        return False
    source = "def %s(%s):\n    return %s\n" % (name, ", ".join(args.split()), written)
    space = {}
    try:
        exec(compile(source, "<zihin>", "exec"), {"__builtins__": {}}, space)
        return bool(test(space[name]))
    except Exception:
        return False


def run(seed: int, rounds: int, consequences: bool, explore: bool = True) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    mind = Mind(Config.changed(support="members", bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14,
                               backoff=True, resemblance=True, race=True, situations=True, explore=explore))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    session.curious = False                              # this test is about writing code, not asking
    session.thinking = False
    watched = []
    for name, args, completions, _ in TASKS:
        right, wrong = completions[0], completions[1:]
        common = wrong[seed % 2]                         # the most imitated completion is a wrong one
        other = wrong[1 - seed % 2]
        watched += [(name, args, common)] * 3 + [(name, args, right), (name, args, other)]
    rng.shuffle(watched)
    for name, args, completion in watched:
        session.watch("%s %s" % (name, args), completion)
    per_round = []
    for _ in range(rounds):
        order = list(TASKS)
        rng.shuffle(order)
        ok = 0
        for name, args, _, test in order:
            written = session.reply("%s %s" % (name, args))
            good = passes(name, args, written, test)
            ok += good
            if consequences:
                session.feedback(good)
            else:
                session.close_turn()
        per_round.append(ok / len(TASKS))
    return {"seed": seed, "consequences": consequences, "explore": explore, "per_round": per_round}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "t_code.jsonl"))
    args = ap.parse_args()
    torch.set_num_threads(1)
    rows = []
    for seed in range(args.seeds):
        for consequences, explore in ((True, True), (False, True), (True, False)):
            row = run(seed, args.rounds, consequences, explore)
            rows.append(row)
            print(json.dumps(row), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
