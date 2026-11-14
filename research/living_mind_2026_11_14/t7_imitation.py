# -*- coding: utf-8 -*-
"""T7: imitation first, then judging and detaching from what was copied (vision 9, 9b; design section 7).

    python research/living_mind_2026_11_14/t7_imitation.py --seeds 0 1 2 3 4 --variant reliable

Society phase: the mind only watches. Others, in situation S1 (two random words), do act X; in situation S2 they
do act Y (provenance "imitated", on the act channel). Own phase: the situations are presented and the mind may act
or stay silent. The world answers Y with a sensed event (something new is learned); it answers X with nothing.
Measured per trial block: how often the mind does X in S1 and Y in S2; and the white-box imitated evidence of
S1 -> X ("others do X"), which must be bit-identical before and after the own phase (detaching from the behaviour
without erasing the knowledge). Words are random per seed; nothing in the core knows which act is useful.
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

from zihin import IMITATED, SENSED, TOLD, Config, Event, Mind  # noqa: E402
from zihin.store import EXPECT, W_NEG, W_POS  # noqa: E402
from zihin.events import IMITATED as IMI  # noqa: E402

ACT, WORLD = "text", "world"
VARIANTS = {"changed": Config.changed, "reliable": Config.reliable,
            "no_act_relief": lambda: Config.changed(act_expect_relief=False)}  # F6 ablation


def word(rng):
    return "".join(rng.choice("bcdfghklmnprstvyz") + rng.choice("aeiou") for _ in range(rng.randint(2, 3)))


def present(mind, symbols, prov):
    acts = []
    for s in symbols:
        acts = mind.step([Event(ACT, s, prov)])
    return acts


def imitated_evidence(mind, situation, act):
    """Raw imitated (w+, w-) summed over every EXPECT tie into the act unit (white box): "others do X"."""
    S = mind.S
    cid = mind.ch.id(ACT)
    index = mind.ch.symbol_index(cid, act, create=False)
    a = S.atom_index.get((cid, index), -1) if index >= 0 else -1
    if a < 0:
        return None
    ties = S.inc.get((a, EXPECT), [])
    ids = torch.tensor(ties, dtype=torch.long)
    return [float(S.t_w[W_POS, ids, IMI].sum()), float(S.t_w[W_NEG, ids, IMI].sum())]


def life(seed, variant, watch=30, trials=60, block=10, answer_x=False):
    rng = random.Random("t7/%d" % seed)
    s1, s2 = [word(rng), word(rng)], [word(rng), word(rng)]
    x, y = word(rng), word(rng)
    answers = [word(rng) for _ in range(trials)]
    mind = Mind(VARIANTS[variant]())
    mind.add_channel(ACT, "act")
    order = [0, 1] * watch
    rng.shuffle(order)
    for k in order:                                    # society: watch others
        present(mind, (s1 + [x]) if k == 0 else (s2 + [y]), IMITATED)
        mind.step([])
    before = imitated_evidence(mind, s1, x)
    rows, did = [], {"X": [], "Y": []}
    plan = [0, 1] * (trials // 2)
    rng.shuffle(plan)
    for i, k in enumerate(plan):                       # own life
        situation, useful = (s1, False) if k == 0 else (s2, True)
        acts = present(mind, situation, TOLD)
        emitted = [e.symbol for e in acts if e.channel == ACT]
        target = y if useful else x
        did["Y" if useful else "X"].append(float(target in emitted))
        reply = [Event(WORLD, answers[i], SENSED)] if ((useful or answer_x) and target in emitted) else []
        mind.step(reply)                               # echo of the own act (pending) + the world's answer
        mind.step([])
        n = len(did["X"]) + len(did["Y"])
        if n % block == 0:
            rows.append({"trials": n, "do_X_in_S1": sum(did["X"][-block // 2:]) / max(1, len(did["X"][-block // 2:])),
                         "do_Y_in_S2": sum(did["Y"][-block // 2:]) / max(1, len(did["Y"][-block // 2:]))})
    after = imitated_evidence(mind, s1, x)
    return {"seed": seed, "variant": variant, "answer_x": answer_x, "curve": rows, "imitated_S1_X_before": before,
            "imitated_S1_X_after": after, "others_do_X_kept": before is not None and before == after}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--variant", default="reliable")
    ap.add_argument("--answer-x", action="store_true", help="control: the world answers X too")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "t7_imitation.jsonl"))
    args = ap.parse_args()
    with open(args.out, "a") as f:
        for seed in args.seeds:
            r = life(seed, args.variant, answer_x=args.answer_x)
            f.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
