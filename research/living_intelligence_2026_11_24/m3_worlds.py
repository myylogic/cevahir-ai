# -*- coding: utf-8 -*-
"""M3: worlds of the earlier study that a parrot cannot answer, lived by Cevahir and by a Transformer trained on
the very stream Cevahir lived (NOTES.md, 8).

    python m3_worlds.py [deduce chain ...] [--seeds 3] [--variant speaking] [--set inherit=1] [--minutes 5]

- ``deduce`` (research/living_mind_2026_11_15/t_deduce.py, unchanged): classes with a property stated of the class
  only, members stated to be members, one exception; facts read as documents in a shuffled order, never two facts of
  one chain together; three members of each class asked what they do (watched), the other three never. Asked "kartal
  ne yapar ?", the answer follows only by putting two facts read apart together. The exception is asked too.
- ``chain`` (t_chain.py, unchanged): the same with two steps (a member, its class, the class's kind and the kind's
  property), a class with a nearer property of its own, and the exception.

Cevahir reads each fact on a line of its own (``Cevahir.tell``) and watches each dialogue (``Cevahir.watch``); every
question is answered by a process that is the mind as it stands, in three ways (zihin, harmony, cevahir). The
Transformer (m3_transformer.py) is trained from scratch on exactly the characters Cevahir's harmony lived and asked
the same questions on a new line. An answer is right when its words, without the closing mark, are the member and
the property that holds for it.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "research", "living_mind_2026_11_15"))
from living.cevahir import Cevahir  # noqa: E402


def deduce_world(seed: int, days: int = 3):
    import t_deduce as D
    rng = random.Random(seed)
    asked, unasked = [], []
    for name, (copula, plural, prop, members) in D.CLASSES.items():
        order = list(members)
        rng.shuffle(order)
        asked += [(m, prop) for m in order[:3]]
        unasked += [(m, prop) for m in order[3:]]
    life = []
    for _ in range(days):
        sentences = []
        for name, (copula, plural, prop, members) in D.CLASSES.items():
            sentences += ["%s bir %s ." % (m, copula) for m in members]
            sentences += ["%s %s ." % (plural, prop)] * 2
        sentences += ["%s bir %s ." % (D.EXCEPTION[0], D.CLASSES[D.EXCEPTION[1]][0]),
                      "%s %s ." % (D.EXCEPTION[0], D.EXCEPTION[2])]
        rng.shuffle(sentences)
        dialogues = [("%s ne yapar ?" % m, "%s %s ." % (m, p)) for m, p in asked]
        rng.shuffle(dialogues)
        for i, sentence in enumerate(sentences):
            life.append(("tell", sentence))
            if i % 3 == 2 and dialogues:
                life.append(("watch",) + dialogues.pop())
        life += [("watch",) + d for d in dialogues]
    probes = [("%s ne yapar ?" % m, [m, p], "two facts") for m, p in unasked]
    probes.append(("%s ne yapar ?" % D.EXCEPTION[0], [D.EXCEPTION[0], D.EXCEPTION[2]], "exception"))
    return life, probes


def chain_world(seed: int, days: int = 3):
    import t_chain as C
    rng = random.Random(seed)
    asked, unasked = [], []
    for name, (copula, plural, kind, own, members) in C.CLASSES.items():
        order = list(members)
        rng.shuffle(order)
        asked += [(m, name) for m in order[:3]]
        unasked += [(m, name) for m in order[3:]]
    life = []
    for _ in range(days):
        sentences = []
        for kind, (copula, plural, prop) in C.KINDS.items():
            sentences += ["%s %s ." % (plural, prop)] * 2
        for name, (copula, plural, kind, own, members) in C.CLASSES.items():
            sentences += ["%s bir %s ." % (m, copula) for m in members]
            sentences += ["%s bir %s ." % (plural, C.KINDS[kind][0])] * 2
            if own:
                sentences += ["%s %s ." % (plural, own)] * 2
        sentences += ["%s bir %s ." % (C.EXCEPTION[0], C.CLASSES[C.EXCEPTION[1]][0]),
                      "%s %s ." % (C.EXCEPTION[0], C.EXCEPTION[2])]
        rng.shuffle(sentences)
        dialogues = [("%s ne yapar ?" % m, "%s %s ." % (m, C.holds(c))) for m, c in asked]
        rng.shuffle(dialogues)
        for i, sentence in enumerate(sentences):
            life.append(("tell", sentence))
            if i % 3 == 2 and dialogues:
                life.append(("watch",) + dialogues.pop())
        life += [("watch",) + d for d in dialogues]
    probes = []
    for m, c in unasked:
        probes.append(("%s ne yapar ?" % m, [m, C.holds(c)], "one step" if C.CLASSES[c][3] else "two steps"))
    probes.append(("%s ne yapar ?" % C.EXCEPTION[0], [C.EXCEPTION[0], C.EXCEPTION[2]], "exception"))
    return life, probes


WORLDS = {"deduce": deduce_world, "chain": chain_world}


def run(world: str, seed: int, args, switches) -> dict:
    import torch
    torch.set_num_threads(1)
    life, probes = WORLDS[world](seed)
    c = Cevahir(variant=args.variant, switches=switches, seed=seed, keep_stream=True)
    t0 = time.time()
    for item in life:
        if item[0] == "tell":
            c.tell(item[1])
        else:
            c.watch(item[1], item[2])
    c.session.close_turn()
    lived_s = time.time() - t0
    said = c.answers([q for q, _, _ in probes], way="every", workers=args.workers)
    stream = "".join(c.stream)
    row = {"test": "m3_worlds", "code": code_version(), "world": world, "seed": seed, "variant": args.variant, "switches": switches,
           "life_items": len(life), "stream_chars": len(stream), "life_s": round(lived_s, 1), "asked": len(probes)}
    ways = {}
    for way in ("zihin", "harmony", "cevahir"):
        ways[way] = [c.session.body(s[way]) if s[way] != "(sessiz)" else [] for s in said]
    if args.minutes > 0:
        import m3_transformer as MT
        tf = MT.train_and_answer(stream, [q for q, _, _ in probes], minutes=args.minutes, threads=1, seed=seed)
        ways["transformer"] = [c.session.body(a) for a in tf["answers"]]
        row["transformer_run"] = {k: tf[k] for k in ("params", "steps", "train_s", "final_loss", "chars")}
    for way, got in ways.items():
        tally = {}
        for (q, truth, kind), g in zip(probes, got):
            t = tally.setdefault(kind, [0, 0, 0])            # right, silent, asked
            t[2] += 1
            t[0] += g == truth
            t[1] += not g
        row[way] = tally
    row["examples"] = [{"q": q, **{w: " ".join(ways[w][i]) for w in ways}} for i, (q, _, _) in enumerate(probes)][:6]
    return row


def code_version() -> str:
    """The commit of the code that wrote a row."""
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("worlds", nargs="*", default=["deduce", "chain"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variant", default="speaking")
    ap.add_argument("--set", default="inherit=1")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m3_worlds.jsonl"))
    args = ap.parse_args()
    switches = {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, args.set.split(",")))}
    for world in args.worlds:
        for seed in range(args.seeds):
            row = run(world, seed, args, switches)
            print(json.dumps({k: row[k] for k in row if k != "examples"}, ensure_ascii=False), flush=True)
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
