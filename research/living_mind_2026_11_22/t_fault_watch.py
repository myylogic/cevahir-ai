# -*- coding: utf-8 -*-
"""t_fault_watch.py: F49 while the mind only watches (``Config.silent_claims``), measured.

    python research/living_mind_2026_11_22/t_fault_watch.py [--seeds 5]

Sections 11 and 12 settle a claim only where the mind was free to speak: a mind that is never asked learns nothing
about its ways. With ``silent_claims`` it makes its reading also while it listens to something new, and what its
ways claim there is settled by what it hears next. Here the mind is **never asked** during its life: it only reads
and watches. Then it is asked on copies (nothing it says is lived).

``kinds`` (the world of ``t_fault_kinds.py``): what a thing eats follows its kind, read once in a sentence about
each thing; the 8 watched dialogues at the start are all of one kind; then the mind watches n more dialogues, about
new things of all four kinds. Asked afterwards about 8 fresh things, two of each kind.

``shared`` (the misleading frame of ``t_fault.py``): 8 watched things share one value; then the mind watches n more
dialogues in which the value is one of four at random. Asked afterwards about 8 fresh things.

Conditions (the same life in each): F49 off; F49 on (nothing is ever settled: the mind is never asked); F49 on with
silent claims.

``cost``: the seconds each condition took to live the same watching.

Appends to ``results/fault_watch.jsonl``.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import t_fault as T  # noqa: E402

OUT = os.path.join(HERE, "results", "fault_watch.jsonl")
FOODS = {"evcil": "ot", "yırtıcı": "et", "sucul": "balık", "kanatlı": "tohum"}
WATCHED = [0, 4, 8, 16]
CONDITIONS = [("off", {}), ("on (never asked: nothing is settled)", {"faults": True}),
              ("on with silent claims", {"faults": True, "silent_claims": True})]


def life(seed: int, switches: dict, world: str):
    """One life; yields (n watched so far, the marks on the 8 fresh things) at every checkpoint, then the seconds."""
    from zihin import IMITATED
    import t_relations as R
    from zihin import Mind
    from zihin.__main__ import CHANNEL, Session
    rng = random.Random(seed)
    things = list(T.THINGS)
    rng.shuffle(things)
    members = things[:8]
    names = list(FOODS)
    values = T.FRAMES["V1"][2]
    if world == "kinds":
        kind = {x: names[0] if x in members else names[i % 4] for i, x in enumerate(things)}
        food = {x: FOODS[kind[x]] for x in things}
    else:
        food = {x: values[0] if x in members else rng.choice(values) for x in things}
    torch.manual_seed(seed)
    mind = Mind(R.speaking_config(dict(switches)))
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)

    def qa(x):
        return "%s ne yer ?" % x, "%s %s yer ." % (x, food[x])
    t0 = time.time()
    for _ in range(2):
        if world == "kinds":
            for x in things:                             # the kind of every thing, read once a round
                s.company(None)
                s.listen("%s %s bir hayvandır ." % (x, kind[x]), IMITATED)
                s.end_of_turn(IMITATED, tone=".")
                mind.step([])
        for x in members:
            s.watch(*qa(x))
    s.close_turn()
    took = time.time() - t0
    pool, fresh = things[8:40], things[40:]
    out = []
    for i in range(WATCHED[-1] + 1):
        if i in WATCHED:
            out.append((i, "".join(T.probe(s, *qa(x)) for x in fresh)))
        if i < WATCHED[-1]:
            t0 = time.time()
            s.watch(*qa(pool[i]))                        # watched, never asked
            s.close_turn()
            took += time.time() - t0
    return out, took, mind


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
    rows, lines = [], []
    for world, title in (("kinds", "what a thing eats follows its kind (read in a sentence)"),
                         ("shared", "8 watched things share a value that the others do not")):
        lines.append("== never asked, only watching; %s: %d seeds; 8 fresh things asked on copies after n more "
                     "watched dialogues (right / wrong / silent) ==" % (title, args.seeds))
        for label, switches in CONDITIONS:
            at = {n: {"right": 0, "wrong": 0, "silent": 0} for n in WATCHED}
            seconds, first, settled = 0.0, None, 0
            for seed in range(args.seeds):
                marks, took, mind = life(seed, switches, world)
                seconds += took
                for n, m in marks:
                    for key, k in T.count(m).items():
                        at[n][key] += k
                first = first or " ".join("%d:%s" % nm for nm in marks)
                if switches.get("faults"):
                    from zihin import hata
                    settled += sum(1 for key in hata.votes(mind) if isinstance(key, tuple) and len(key) == 3)
            rows.append({"test": "fault_watch", "world": world, "condition": label, "seeds": args.seeds,
                         "fresh_after": at, "watch_s": round(seconds, 1), "ways_with_a_record": settled,
                         "first": first})
            lines.append("  %-38s %s ; watching took %.1f s ; ways with a record of their own: %d ; the first seed: %s" % (
                label, "  ".join("n=%d %d/%d/%d" % (n, at[n]["right"], at[n]["wrong"], at[n]["silent"]) for n in WATCHED),
                seconds, settled, first))
    T.finish(args, rows, lines)


if __name__ == "__main__":
    main()
