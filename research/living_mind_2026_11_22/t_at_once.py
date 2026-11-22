# -*- coding: utf-8 -*-
"""t_at_once.py: many questions answered at the same time (zihin/paralel.py): the same answers, and how much sooner?

    python research/living_mind_2026_11_22/t_at_once.py [--train 150] [--test 60] [--workers 4]

A mind lives SCAN's add-jump life of ``t_scan.py`` (F43 and F45 on, window 13: answers in parts, the mind's most
expensive answers). The held-out commands are then answered twice by the same mind: one after another, each on a
copy of the mind (``Mind.clone``); and all at the same time, each in a process that is the mind as it stands
(``paralel.answers``). Reported: whether every answer is the same word for word, whether the mind's substance digest
is what it was before either, exact answers, and the seconds of both ways.
Appends to ``results/at_once.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "at_once.jsonl")


def main() -> None:
    import t_relations as R
    import t_scan as S
    from zihin import Mind, paralel
    from zihin.__main__ import CHANNEL, Session
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=150)
    ap.add_argument("--test", type=int, default=60)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    mind = Mind(R.speaking_config({"wholes": True, "told_apart": True}))
    mind.add_channel(CHANNEL, "act")
    mind.knob["F"] = 13
    s = Session(mind)
    train, test = S.split(0, args.train, args.test)
    for command, answer in S.life(0, train, 2):
        s.watch(" ".join(command), " ".join(answer) + " .")
    s.close_turn()
    questions = [" ".join(c) for c, _ in test]
    digest = mind.S.digest()
    t0 = time.perf_counter()
    in_turn = [Session(mind.clone()).reply(q, limit=400) for q in questions]
    turn_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    at_once = paralel.answers(s, questions, limit=400, workers=args.workers)
    once_s = time.perf_counter() - t0
    exact = sum(x == " ".join(a) for x, (_, a) in zip(at_once, test))
    lines = ["== many questions at the same time: SCAN add-jump, %d training dialogues, %d held-out commands ==" % (
        len(train), len(test)),
        "  one after another, on copies: %.1f s; at the same time in %d processes: %.1f s (%.1fx)" % (
            turn_s, args.workers, once_s, turn_s / max(1e-9, once_s)),
        "  the same answer, word for word: %d of %d; exact answers: %d of %d" % (
            sum(x == y for x, y in zip(in_turn, at_once)), len(test), exact, len(test)),
        "  the mind's substance digest unchanged by either: %s" % (mind.S.digest() == digest)]
    for line in lines:
        print(line, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"test": "t_at_once", "train": len(train), "asked": len(test), "workers": args.workers,
                            "in_turn_s": round(turn_s, 1), "at_once_s": round(once_s, 1),
                            "same_answers": sum(x == y for x, y in zip(in_turn, at_once)), "exact": exact,
                            "digest_unchanged": mind.S.digest() == digest, "summary": lines}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
