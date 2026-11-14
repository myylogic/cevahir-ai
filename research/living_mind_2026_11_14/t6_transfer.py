# -*- coding: utf-8 -*-
"""T6: is a second, unseen domain learned faster by a mind that has lived another domain? And is the first kept?

    python research/living_mind_2026_11_14/t6_transfer.py --seeds 0 1 2 3 4 --variant members

For each seed and each order (A = nitelik then B = iliski, and the reverse) two minds meet the identical B stream:
a fresh mind, and the mind that first lived A (same seed). Vocabularies are drawn independently per domain, so no
content can carry over; what can carry over is how to learn (learned trust in sources, the habits of the core).
Recorded: B's checkpoints for both minds (T1, T2, T3, T5 rows as in run.py) and, for the experienced mind, A's
checkpoint after B (is A's knowledge kept: members, exceptions, held-out, early facts bit-identical).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run import World, checkpoint, live  # noqa: E402


def one(seed, first, second, variant, n_a, n_b):
    a, b = World.make(seed, first), World.make(seed, second)
    _, fresh_rows, _ = live(b, variant, n_b)
    mind, a_rows, a_state = live(a, variant, n_a)
    a_end = [r for r in a_rows if r.get("at") == n_a and r["test"] in ("T3", "T5")]
    _, exp_rows, _ = live(b, variant, n_b, mind=mind)
    a_after = checkpoint(a, mind, a_state, n_a)
    keep = ("T1", "T2", "T3", "T5")
    return {"seed": seed, "first": first, "second": second, "variant": variant, "n_a": n_a, "n_b": n_b,
            "fresh": [r for r in fresh_rows if r["test"] in keep],
            "experienced": [r for r in exp_rows if r["test"] in keep],
            "a_before_b": a_end, "a_after_b": [r for r in a_after if r["test"] in ("T3", "T5")]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--variant", default="members")
    ap.add_argument("--orders", nargs="+", default=["nitelik:iliski", "iliski:nitelik"])
    ap.add_argument("--n-a", type=int, default=600)
    ap.add_argument("--n-b", type=int, default=600)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "t6_transfer.jsonl"))
    args = ap.parse_args()
    for seed in args.seeds:
        for order in args.orders:
            first, second = order.split(":")
            r = one(seed, first, second, args.variant, args.n_a, args.n_b)
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(seed, order, "done", flush=True)


if __name__ == "__main__":
    main()
