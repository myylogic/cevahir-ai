# -*- coding: utf-8 -*-
"""T-scan-splits: SCAN's other splits, with commands and answers of every length. Public tests, generated here from
SCAN's own grammar (``t_compose.scan_commands``: all 20,910 commands with their meanings). F43 (Wholes) was developed
on the add-jump split (``t_scan.py``) only; this script was written before it was run on any of these.

    python research/living_mind_2026_11_15/t_scan_splits.py --split length [--train 1000] [--test 500] [--set ...]
    python research/living_mind_2026_11_15/t_scan_splits.py --split length --who transformer [--steps 6000]

``--split length`` (Lake & Baroni 2018). Training: the commands whose answer has at most 22 actions (16,990); test:
those whose answer has 24 to 48 actions (3,920; none has 23). The sizes generated here are the published ones. No
answer of the test's length is ever seen: the only command without a connective in the test is "V around D thrice"
(eight of them, 24 actions each), and every other test command joins two parts by "and" / "after".

``--split around_right`` (after the hardest template split of Loula, Baroni & Lake 2018). Training: the commands in
which "around right" does not stand (15,225); test: every command in which it stands (5,685). "around" is met with
"left" only, "right" in every other construction. The split is generated here: the published one holds out the
template with the four primitive verbs, and its files were not downloaded, so its sizes may differ from these (here
"turn around right" is held out too).

The other template splits of the same paper, generated here in the same way (``V`` is walk, look, run or jump):
``--split right`` holds out every command in which "V right" stands, ``--split opposite_right`` those with
"V opposite right", ``--split jump_around_right`` those with "jump around right". The sizes are written in the result.

The mind watches a part of the training set, written in the result: every training command without a connective,
and ``--train`` training commands with "and" / "after" drawn at random (0: all). ``--test``: that many test commands
drawn at random (0: all). The life and the test are those of ``t_scan.py``: each training dialogue watched once an
epoch, the shorter commands first in the first epoch (``--shuffled``: every epoch shuffled); each test command told
to a copy of the mind as it was at the end of its life (a forked process); exact match of the whole answer.

Reported: all; commands without a connective, and with one; by how many of a command's parts (the command itself
when it has no connective) are themselves outside the training set (for ``length``: a part of 24 actions, "V around
D thrice"; for a template split: a part in which the template stands); by the length of the answer; silent (nothing
said) and wrong apart. Also: up to a hundred of the watched commands whose dialogue does not fit 12 items, asked
again (what was lived, not the split's test).

``--who transformer``: the decoder-only Transformer of ``t_scan.py`` trained from scratch on the very dialogues the
mind watches.

Appends to ``results/scan_splits.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import sys
import time
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "scan_splits.jsonl")
MIND = None                                             # the mind at the end of its life (read by forked copies)
SPLIT = "length"
SHUFFLED = False
Dialogue = Tuple[tuple, tuple]


def joined(command: tuple) -> str:
    return "and" if "and" in command else ("after" if "after" in command else "-")


def parts(command: tuple) -> List[tuple]:
    """The parts a connective joins (the command itself when it has none)."""
    for word in ("and", "after"):
        if word in command:
            i = command.index(word)
            return [command[:i], command[i + 1:]]
    return [command]


VERBS = ("walk", "look", "run", "jump")
TEMPLATES = {"around_right": [("around", "right")],
             "right": [(v, "right") for v in VERBS],
             "opposite_right": [(v, "opposite", "right") for v in VERBS],
             "jump_around_right": [("jump", "around", "right")]}


def in_training(command: tuple, answer: tuple) -> bool:
    if SPLIT == "length":
        return len(answer) <= 22
    return not any(command[i:i + len(t)] == t for t in TEMPLATES[SPLIT] for i in range(len(command) - len(t) + 1))


def everything() -> List[Dialogue]:
    import t_compose as T
    return [(tuple(c), tuple(a)) for c, a in T.scan_commands()]


def split(seed: int, n_train: int, n_test: int) -> Tuple[List[Dialogue], List[Dialogue]]:
    every = everything()
    rng = random.Random(seed)
    train = [d for d in every if in_training(*d)]
    test = [d for d in every if not in_training(*d)]
    simple = [d for d in train if joined(d[0]) == "-"]
    longer = [d for d in train if joined(d[0]) != "-"]
    if n_train:
        longer = rng.sample(longer, min(n_train, len(longer)))
    if n_test:
        test = rng.sample(test, min(n_test, len(test)))
    return simple + longer, test


def life(seed: int, train, epochs: int) -> List[Dialogue]:
    rng = random.Random(seed + 1)
    out: List[Dialogue] = []
    for epoch in range(epochs):
        block = list(train)
        rng.shuffle(block)
        if epoch == 0 and not SHUFFLED:
            block.sort(key=lambda d: len(d[0]))          # stable: the shorter commands first, shuffled among equals
        out += block
    return out


def _answer(command: tuple) -> str:
    from zihin.__main__ import Session
    return Session(MIND).reply(" ".join(command), limit=400)


def rows_of(test, said) -> List[dict]:
    answers = dict(everything())
    rows = []
    for (command, answer), got in zip(test, said):
        got = got if isinstance(got, str) else " ".join(got)
        rows.append({"command": " ".join(command), "said": got, "right": got == " ".join(answer),
                     "silent": got.strip() in ("", "(sessiz)"), "answer": len(answer), "joined": joined(command),
                     "new_parts": sum(not in_training(p, answers[p]) for p in parts(command))})
    return rows


def run(seed: int, switches: Dict[str, bool], n_train: int, n_test: int, window: int, epochs: int, workers: int) -> dict:
    import t_relations as R
    from zihin import Mind
    from zihin.__main__ import CHANNEL, Session
    global MIND
    torch.manual_seed(seed)
    mind = Mind(R.speaking_config(switches))
    mind.add_channel(CHANNEL, "act")
    mind.knob["F"] = window
    s = Session(mind)
    train, test = split(seed, n_train, n_test)
    t0 = time.time()
    for command, answer in life(seed, train, epochs):
        s.watch(" ".join(command), " ".join(answer) + " .")
    lived = time.time() - t0
    MIND = mind
    sizes = [sum(in_training(*d) for d in everything()), sum(not in_training(*d) for d in everything())]
    long_ones = [d for d in train if len(d[0]) + 1 + len(d[1]) > 12]
    probe = random.Random(seed + 2).sample(long_ones, min(100, len(long_ones)))
    t0 = time.time()
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        said = pool.map(_answer, [c for c, _ in test], chunksize=1)
        again = pool.map(_answer, [c for c, _ in probe], chunksize=1)
    answered = time.time() - t0
    watched = sum(got == " ".join(answer) for (_, answer), got in zip(probe, again))
    return {"seed": seed, "train": len(train), "moments": mind.t, "ties": mind.S.n_ties, "lived_s": round(lived, 1),
            "answered_s": round(answered, 1), "watched_again": [watched, len(probe)], "split_sizes": sizes,
            "rows": rows_of(test, said)}


def run_transformer(seed: int, n_train: int, n_test: int, epochs: int, steps: int) -> dict:
    """The Transformer of ``t_scan.py`` (3 layers, 4 heads, width 64; greedy decoding after the turn end), trained
    from scratch on the dialogues of this split's life."""
    import t_scan as S
    S.split, S.life = split, life                        # (its training loop, with this script's split and life)
    r = S.run_transformer(seed, n_train, n_test, epochs, steps)
    _, test = split(seed, n_train, n_test)
    r["rows"] = rows_of(test, [x["said"] for x in r["rows"]])
    return r


def main() -> None:
    sys.path.insert(0, HERE)
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["length"] + sorted(TEMPLATES), default="length")
    ap.add_argument("--who", choices=["mind", "transformer"], default="mind")
    ap.add_argument("--steps", type=int, default=6000, help="transformer: training steps (batches of 32)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", type=int, default=1000, help="training commands with and / after (0: all)")
    ap.add_argument("--test", type=int, default=500, help="test commands (0: all)")
    ap.add_argument("--window", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--set", default="")
    ap.add_argument("--shuffled", action="store_true", help="no curriculum: the first epoch in a shuffled order too")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    global SPLIT, SHUFFLED
    SPLIT, SHUFFLED = args.split, args.shuffled
    torch.set_num_threads(1)
    switches = {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, args.set.split(",")))}
    if args.who == "transformer":
        r = run_transformer(args.seed, args.train, args.test, args.epochs, args.steps)
        switches = {"transformer": True}
    else:
        r = run(args.seed, switches, args.train, args.test, args.window, args.epochs, args.workers)
    rows = r["rows"]

    def of(keep) -> str:
        kept = [x for x in rows if keep(x)]
        return "%d / %d" % (sum(x["right"] for x in kept), len(kept))
    wrong = [x for x in rows if not x["right"]]
    lines = ["== SCAN %s%s: %d training dialogues an epoch (%d epochs%s, window %d), switches %s: exact %s ==" % (
        SPLIT, ", a Transformer" if args.who == "transformer" else "", r["train"], args.epochs,
        ", shuffled" if SHUFFLED else "", args.window, switches or "-", of(lambda x: True)),
        "  not exact: %d silent, %d wrong" % (sum(x["silent"] for x in wrong), sum(not x["silent"] for x in wrong)),
        "  no connective: %s; and: %s; after: %s" % (of(lambda x: x["joined"] == "-"), of(lambda x: x["joined"] == "and"),
                                                    of(lambda x: x["joined"] == "after")),
        "  parts outside the training set: none %s; one %s; two %s" % (
            of(lambda x: x["new_parts"] == 0), of(lambda x: x["new_parts"] == 1), of(lambda x: x["new_parts"] == 2)),
        "  answers of 1-8 actions: %s; 9-16: %s; 17-24: %s; 25-32: %s; 33-48: %s" % (
            of(lambda x: x["answer"] <= 8), of(lambda x: 8 < x["answer"] <= 16), of(lambda x: 16 < x["answer"] <= 24),
            of(lambda x: 24 < x["answer"] <= 32), of(lambda x: x["answer"] > 32))]
    if args.who == "transformer":
        lines += ["  its training commands (300 of them): %d / %d" % tuple(r["watched_again"]),
                  "  trained %d steps in %.0f s (loss %.4f); decoded in %.0f s" % (
                      r["steps"], r["lived_s"], r["loss"], r["answered_s"])]
    else:
        lines += ["  watched commands that do not fit 12 items, asked again: %d / %d" % tuple(r["watched_again"]),
                  "  lived %d moments in %.0f s (%d ties); answered in %.0f s" % (
                      r["moments"], r["lived_s"], r["ties"], r["answered_s"])]
    lines += ["  not exact: %s -> %s" % (x["command"], "(silent)" if x["silent"] else x["said"][:160]) for x in wrong[:6]]
    for line in lines:
        print(line)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"split": SPLIT, "who": args.who, "seed": args.seed, "train": args.train, "test": args.test,
                            "window": args.window, "epochs": args.epochs, "shuffled": SHUFFLED, "switches": switches,
                            "summary": lines, **r}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
