# -*- coding: utf-8 -*-
"""t_own_time.py: when does the mind think, when the decision is its own (F48)?

    python research/living_mind_2026_11_22/t_own_time.py [--seeds 3] [--silences 5]

A mind with T-want's beginning (twelve things read about; ``t_want.py``) is told sentences of three kinds, in a
shuffled order, by someone who asks nothing of it; after each it is left in silence for ``--silences`` moments:
  expected:  a sentence it has read ("gül kırmızı bir çiçektir .");
  against:   a thing it knows said to be otherwise ("gül sarı bir çiçektir .": another adjective of its world);
  nothing:   a sentence of words it never met ("zambak mor bir çiçektir ." with T-want's library things: it could
             expect nothing).
No command is given. With F48 the mind decides in each silent moment whether to think about what it last took in.
Reported for each kind: its mean surprise over the sentence (the mind's own feeling), how many sentences it thought
about, the thoughts' mean reality score and how many were empty (nothing said inwardly); what its thoughts are
expected to come to after the run (1/2 at birth); and what the program's clock would have done (a thought after
every sentence, whatever it was).
Appends to ``results/own_time.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
sys.path.insert(0, HERE)
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "own_time.jsonl")


def run(seed: int, silences: int) -> dict:
    import t_relations as R
    import t_want as W
    from zihin import IMITATED, TOLD, Mind
    from zihin.__main__ import CHANNEL, Session
    from zihin.read import think_expectation
    torch.manual_seed(seed)
    rng = random.Random(seed)
    mind = Mind(R.speaking_config({"think": True}))
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    known = list(W.KNOWN)
    for _ in range(3):
        for a in known:
            s.company(None)
            s.listen(W.fact(a), IMITATED)
            s.end_of_turn(IMITATED, tone=".")
            mind.step([])
    s.close_turn()
    adjectives = sorted({W.KNOWN[a].split()[0] for a in known})
    told = [("expected", W.fact(a)) for a in known[:8]]
    for a in known[:8]:
        other = rng.choice([x for x in adjectives if x != W.KNOWN[a].split()[0]])
        told.append(("against", "%s %s %s ." % (a, other, " ".join(W.KNOWN[a].split()[1:]))))
    told += [("nothing", W.fact(a)) for a in W.LIBRARY]
    rng.shuffle(told)
    quiet = lambda line: None   # noqa: E731
    rows = {kind: {"told": 0, "surprise": 0.0, "thought": 0, "empty": 0, "reality": 0.0} for kind in
            ("expected", "against", "nothing")}
    for kind, sentence in told:
        n = len(s.thoughts)
        s.live(sentence, TOLD)
        felt = None
        for i in range(silences):
            s.silent_moment(say=quiet)
            if i == 0:
                felt = mind.heard if len(s.thoughts) == n else felt
        row = rows[kind]
        row["told"] += 1
        if len(s.thoughts) > n:
            topic, words, reality = s.thoughts[-1]
            row["thought"] += len(s.thoughts) - n
            row["empty"] += not words
            row["reality"] += reality
        row.setdefault("felt", []).append(felt)
    return {"seed": seed, "rows": rows, "think_expectation": round(think_expectation(mind), 3),
            "thoughts": [(t, " ".join(w), round(r, 3)) for t, w, r in s.thoughts][:6]}


def surprise_of(kind: str, seeds: int) -> float:
    """The mean surprise the mind feels over a sentence of each kind (measured on copies, nothing lived)."""
    import t_relations as R
    import t_want as W
    from zihin import IMITATED, TOLD, Mind
    from zihin.__main__ import CHANNEL, Session
    torch.manual_seed(0)
    mind = Mind(R.speaking_config({"think": True}))
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    for _ in range(3):
        for a in W.KNOWN:
            s.company(None)
            s.listen(W.fact(a), IMITATED)
            s.end_of_turn(IMITATED, tone=".")
            mind.step([])
    s.close_turn()
    sentences = {"expected": [W.fact(a) for a in list(W.KNOWN)[:8]],
                 "against": ["%s sarı %s ." % (a, " ".join(W.KNOWN[a].split()[1:])) for a in list(W.KNOWN)[:8]],
                 "nothing": [W.fact(a) for a in W.LIBRARY]}[kind]
    felt = []
    for sentence in sentences:
        copy = Session(mind.clone())
        copy.live(sentence, TOLD)
        copy.mind.step([])
        felt.append(copy.mind.heard)
    return sum(felt) / len(felt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--silences", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    runs = [run(seed, args.silences) for seed in range(args.seeds)]
    lines = ["== when the mind thinks by its own decision (F48): %d seeds, 24 sentences told in each, %d silent "
             "moments after each ==" % (args.seeds, args.silences)]
    total = 0
    for kind in ("expected", "against", "nothing"):
        told = sum(r["rows"][kind]["told"] for r in runs)
        thought = sum(r["rows"][kind]["thought"] for r in runs)
        empty = sum(r["rows"][kind]["empty"] for r in runs)
        reality = sum(r["rows"][kind]["reality"] for r in runs)
        total += told
        lines.append("  %-9s mean surprise %.3f; thought about %d of %d sentences (empty thoughts %d; mean reality of "
                     "the others %s)" % (kind, surprise_of(kind, args.seeds), thought, told, empty,
                                         "%.2f" % (reality / (thought - empty)) if thought > empty else "-"))
    lines.append("  the program's clock would have thought after every sentence: %d of %d" % (total, total))
    lines.append("  what its thoughts are expected to come to after the run (1/2 at birth): %s" % ", ".join(
        "%.3f" % r["think_expectation"] for r in runs))
    lines.append("  seed 0, its first thoughts (topic: inner words, reality): %s" % runs[0]["thoughts"][:4])
    for line in lines:
        print(line, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"test": "t_own_time", "seeds": args.seeds, "runs": runs, "summary": lines},
                           ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
