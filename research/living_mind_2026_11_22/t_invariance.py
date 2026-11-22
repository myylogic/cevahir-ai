# -*- coding: utf-8 -*-
"""t_invariance.py: does anything in the mind, or in the tests, hang on the words themselves?

    python research/living_mind_2026_11_22/t_invariance.py [--split add_jump|around_right|length] [--train 150]
                                                           [--test 100] [--renamings 3] [--workers 3]

Three checks on one split of SCAN (the splits of ``t_scan.py`` / ``t_scan_splits.py``, generated from SCAN's grammar):

1. **No leak.** No test command is among the training commands (the two sets are compared as sets).
2. **Renaming.** Every command word and every action token is replaced by a random string of letters (a bijection,
   drawn afresh for each renaming). The mind lives the renamed dialogues and answers the renamed test commands. If
   nothing depends on the words themselves (no rule for "jump", no list of SCAN's words, no canned answer), then
   every answer, translated back through the bijection, is the answer the mind gave with the original words: not
   only the same score, the same answer to every test command, right or wrong.
3. **Order.** The training dialogues are watched in a shuffled order in every epoch (no curriculum), with a different
   shuffle each time; and the training commands are drawn with other seeds. The score may change (the life is
   another life); how much is what is reported.

The mind is the speaking configuration with ``--set`` (default wholes=1,told_apart=1), window 13, two epochs.
Appends to ``results/invariance.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import string
import sys
import time
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "invariance.jsonl")
MIND = None
Dialogue = Tuple[tuple, tuple]


def the_split(name: str, seed: int, n_train: int, n_test: int) -> Tuple[List[Dialogue], List[Dialogue]]:
    if name == "add_jump":
        import t_scan as S
        return S.split(seed, n_train, n_test)
    import t_scan_splits as X
    X.SPLIT = name
    return X.split(seed, n_train, n_test)


def the_life(name: str, seed: int, train, epochs: int, shuffled: bool) -> List[Dialogue]:
    if shuffled:
        rng, out = random.Random(1000 + seed), []
        for _ in range(epochs):
            block = list(train) + ([(("jump",), ("i_jump",))] * 4 if name == "add_jump" else [])
            rng.shuffle(block)
            out += block
        return out
    if name == "add_jump":
        import t_scan as S
        return S.life(seed, train, epochs)
    import t_scan_splits as X
    return X.life(seed, train, epochs)


def renaming(words, seed: int) -> Dict[str, str]:
    """A bijection from the words to random strings of 4 to 7 letters (none of them a word of the original)."""
    rng, taken, out = random.Random(seed), set(words), {}
    for w in sorted(words):
        while True:
            new = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(4, 7)))
            if new not in taken:
                break
        taken.add(new)
        out[w] = new
    return out


def _answer(command: str) -> str:
    from zihin.__main__ import Session
    return Session(MIND).reply(command, limit=400)


def live_and_answer(life, test, switches, window: int, workers: int, rename: Dict[str, str] = None):
    """The mind lives the life and answers the test; with ``rename``, in the renamed words (its answers are
    translated back). Returns (answers in the original words, moments, seconds)."""
    import t_relations as R
    from zihin import Mind
    from zihin.__main__ import CHANNEL, Session
    global MIND
    ren = (lambda w: rename[w]) if rename else (lambda w: w)
    back = {v: k for k, v in rename.items()} if rename else None
    torch.manual_seed(0)
    mind = Mind(R.speaking_config(switches))
    mind.add_channel(CHANNEL, "act")
    mind.knob["F"] = window
    s = Session(mind)
    t0 = time.time()
    for command, answer in life:
        s.watch(" ".join(ren(w) for w in command), " ".join(ren(w) for w in answer) + " .")
    MIND = mind
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        said = pool.map(_answer, [" ".join(ren(w) for w in c) for c, _ in test], chunksize=1)
    if back:
        said = [" ".join(back.get(w, w) for w in a.split()) for a in said]
    return said, mind.t, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="add_jump", choices=["add_jump", "around_right", "length"])
    ap.add_argument("--train", type=int, default=150)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--renamings", type=int, default=3)
    ap.add_argument("--orders", type=int, default=2, help="shuffled orders, and other draws of the training commands")
    ap.add_argument("--window", type=int, default=13)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--set", default="wholes=1,told_apart=1")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    switches = {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, args.set.split(",")))}
    train, test = the_split(args.split, 0, args.train, args.test)
    leak = sorted(set(c for c, _ in train) & set(c for c, _ in test))
    life = the_life(args.split, 0, train, 2, False)
    truth = [" ".join(a) for _, a in test]
    words = {w for c, a in train + test for w in c + a}

    base, moments, seconds = live_and_answer(life, test, switches, args.window, args.workers)
    right = [x == y for x, y in zip(base, truth)]
    lines = ["== invariance, SCAN %s: %d training dialogues, %d tests, switches %s ==" % (
        args.split, len(train), len(test), switches or "-"),
        "  no leak: %d test commands are among the training commands" % len(leak),
        "  original words: %d / %d exact (%d moments, %.0f s)" % (sum(right), len(test), moments, seconds)]
    rows = []
    for k in range(args.renamings):
        ren = renaming(words, 100 + k)
        said, _, _ = live_and_answer(life, test, switches, args.window, args.workers, ren)
        same = sum(x == y for x, y in zip(said, base))
        rows.append({"renaming": k, "exact": sum(x == y for x, y in zip(said, truth)), "same_answer_as_original": same,
                     "example": dict(list(sorted(ren.items()))[:4])})
        lines.append("  renaming %d (e.g. %s): %d / %d exact; the same answer as with the original words to %d / %d "
                     "test commands" % (k, ", ".join("%s->%s" % kv for kv in list(sorted(ren.items()))[:3]),
                                        rows[-1]["exact"], len(test), same, len(test)))
    orders = []
    for k in range(args.orders):
        said, _, _ = live_and_answer(the_life(args.split, k, train, 2, True), test, switches, args.window, args.workers)
        orders.append({"what": "shuffled order %d" % k, "exact": sum(x == y for x, y in zip(said, truth))})
        tr2, te2 = the_split(args.split, 10 + k, args.train, args.test)
        said2, _, _ = live_and_answer(the_life(args.split, 10 + k, tr2, 2, False), te2, switches, args.window,
                                      args.workers)
        orders.append({"what": "another draw (seed %d)" % (10 + k),
                       "exact": sum(x == " ".join(a) for x, (_, a) in zip(said2, te2)),
                       "leak": len(set(c for c, _ in tr2) & set(c for c, _ in te2))})
    for o in orders:
        lines.append("  %s: %d / %d exact%s" % (o["what"], o["exact"], len(test),
                                               ("; leak %d" % o["leak"]) if "leak" in o else ""))
    for line in lines:
        print(line, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"split": args.split, "train": len(train), "test": len(test), "switches": switches,
                            "leak": len(leak), "exact_original": sum(right), "renamings": rows, "orders": orders,
                            "summary": lines}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
