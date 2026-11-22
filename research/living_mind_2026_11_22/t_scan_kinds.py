# -*- coding: utf-8 -*-
"""t_scan_kinds.py: F56 The Kind of a Whole on the published SCAN files: the answers that were wrong, and whether
anything else changes.

    python research/living_mind_2026_11_22/t_scan_kinds.py wrong  --scan FOLDER [--only right] [--workers 3]
    python research/living_mind_2026_11_22/t_scan_kinds.py sample --scan FOLDER --split length [--test 300]
    python research/living_mind_2026_11_22/t_scan_kinds.py run    --scan FOLDER --split length

The four answers of the published splits that were not exact (``t_scan_official.py``: one of "right", two of
"around right", one of "length") have one source: a lived command whose answer reads as the replies of its two
wholes in either order ("walk twice and walk": I_WALK I_WALK I_WALK) had no form, stood as a bare whole, and took
what an action phrase takes ("[walk twice and walk] right"). With ``Config.whole_kinds`` the form of a whole says
which wholes stand in it also there.

``wrong``: the life of each of the three splits lived twice (the life of ``t_scan_official.py``: the same sample of
training commands, the same order), with the switch off and on; the wrong commands asked of each.

``sample``: one split; the life lived with the switch off and on; ``--test`` sampled commands of the published test
file asked of both (the same commands). How many are exact in each, and the commands answered differently.

``run``: one split with the switch on, the whole published test file (the run of ``t_scan_official.py``).

Appends to ``results/scan_kinds.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
import torch  # noqa: E402

import t_scan_official as O  # noqa: E402

OUT = os.path.join(HERE, "results", "scan_kinds.jsonl")
WRONG = {"right": ["walk twice and walk right"],
         "around_right": ["look after look around right", "run thrice and run around right"],
         "length": ["jump right and jump around right thrice"]}


def lived_life(scan: str, split: str, seed: int, train_n: int, on: bool):
    """The life of ``t_scan_official.run`` for a split, with F56 off or on: (session, test commands, seconds)."""
    import t_relations as R
    from zihin import Mind
    from zihin.__main__ import CHANNEL, Session
    train_file, test_file = O.FILES[split]
    train = sorted(set(O.published(os.path.join(scan, train_file))))
    test = O.published(os.path.join(scan, test_file))
    rng = random.Random(seed)
    simple = [d for d in train if "and" not in d[0] and "after" not in d[0]]
    longer = [d for d in train if "and" in d[0] or "after" in d[0]]
    if train_n:
        longer = rng.sample(longer, min(train_n, len(longer)))
    lived = simple + longer
    rng = random.Random(seed + 1)
    life = []
    for epoch in range(2):
        block = list(lived)
        rng.shuffle(block)
        if epoch == 0:
            block.sort(key=lambda d: len(d[0]))
        life += block
    torch.manual_seed(seed)
    mind = Mind(R.speaking_config({"wholes": True, "told_apart": True, "whole_kinds": on}))
    mind.add_channel(CHANNEL, "act")
    mind.knob["F"] = 13
    s = Session(mind)
    t0 = time.time()
    for command, answer in life:
        s.watch(" ".join(command), " ".join(answer) + " .")
    s.close_turn()
    return s, test, len(lived), len(train), time.time() - t0


def wrong(args) -> None:
    from zihin import paralel
    rows, lines = [], ["== the answers of the published SCAN splits that were not exact, with F56 off and on =="]
    for split, commands in WRONG.items():
        if args.only and split != args.only:
            continue
        for on in (False, True):
            s, test, lived, total, life_s = lived_life(args.scan, split, args.seed, args.train, on)
            truth = {" ".join(c): " ".join(s.words(" ".join(a))) for c, a in test}
            said = paralel.answers(s, commands, limit=400, workers=min(args.workers, len(commands)))
            for command, x in zip(commands, said):
                ok = x == truth[command]
                rows.append({"test": "scan_kinds_wrong", "split": split, "whole_kinds": on, "command": command,
                             "exact": ok, "said_actions": 0 if x == "(sessiz)" else len(x.split()),
                             "true_actions": len(truth[command].split()), "moments": s.mind.t,
                             "life_s": round(life_s, 1)})
                lines.append("  %-13s F56 %-3s %-42s %s (%d actions said, %d true)" % (
                    split, "on" if on else "off", command, "exact" if ok else "NOT exact",
                    rows[-1]["said_actions"], rows[-1]["true_actions"]))
    finish(args, rows, lines)


def sample(args) -> None:
    from zihin import paralel
    answers, lives = {}, {}
    for on in (False, True):
        s, test, lived, total, life_s = lived_life(args.scan, args.split, args.seed, args.train, on)
        rng = random.Random(args.seed + 2)
        asked = rng.sample(test, min(args.test, len(test))) if args.test else test
        t0 = time.time()
        said = paralel.answers(s, [" ".join(c) for c, _ in asked], limit=400, workers=args.workers)
        truth = [" ".join(s.words(" ".join(a))) for _, a in asked]
        answers[on] = (said, truth, [" ".join(c) for c, _ in asked])
        lives[on] = (s.mind.t, round(life_s, 1), round(time.time() - t0, 1), lived, total)
    off, on = answers[False], answers[True]
    exact = {flag: sum(x == y for x, y in zip(answers[flag][0], answers[flag][1])) for flag in (False, True)}
    differ = [(c, a == t, b == t) for c, a, b, t in zip(off[2], off[0], on[0], off[1]) if a != b]
    lines = ["== the published SCAN %s split, %d sampled commands of the published test file, F56 off and on (%d of "
             "the %d training commands watched twice) ==" % (args.split, len(off[2]), lives[False][3], lives[False][4]),
             "  F56 off: %d of %d exact ; life %d moments in %.0f s" % (exact[False], len(off[2]), lives[False][0],
                                                                      lives[False][1]),
             "  F56 on:  %d of %d exact ; life %d moments in %.0f s" % (exact[True], len(on[2]), lives[True][0],
                                                                      lives[True][1]),
             "  answered differently: %d" % len(differ)]
    for command, a, b in differ[:8]:
        lines.append("    %s : off %s, on %s" % (command, "exact" if a else "not exact", "exact" if b else "not exact"))
    finish(args, [{"test": "scan_kinds_sample", "split": args.split, "seed": args.seed, "asked": len(off[2]),
                   "exact_off": exact[False], "exact_on": exact[True], "answered_differently": len(differ),
                   "moments_off": lives[False][0], "moments_on": lives[True][0], "train_watched": lives[False][3],
                   "train_published": lives[False][4]}], lines)


def run(args) -> None:
    from zihin import paralel
    s, test, lived, total, life_s = lived_life(args.scan, args.split, args.seed, args.train, True)
    t0 = time.time()
    said = paralel.answers(s, [" ".join(c) for c, _ in test], limit=400, workers=args.workers)
    answered_s = time.time() - t0
    truth = [" ".join(s.words(" ".join(a))) for _, a in test]
    right = [x == y for x, y in zip(said, truth)]
    silent = sum(x == "(sessiz)" for x in said)
    lines = ["== the published SCAN %s split with F56: %d of the %d published training commands watched twice; the "
             "published test file ==" % (args.split, lived, total),
             "  exact: %d of %d (not exact: %d silent, %d wrong)" % (sum(right), len(test), silent,
                                                                    len(test) - sum(right) - silent),
             "  life %d moments in %.0f s; answers in %.0f s with %d processes" % (s.mind.t, life_s, answered_s,
                                                                                 args.workers)]
    for (c, _), x, ok in zip(test, said, right):
        if not ok and len(lines) < 9:
            lines.append("  not exact: %s -> %s" % (" ".join(c), x[:120]))
    finish(args, [{"test": "scan_kinds_run", "split": args.split, "seed": args.seed, "train_watched": lived,
                   "train_published": total, "asked": len(test), "exact": sum(right), "silent": silent,
                   "moments": s.mind.t, "life_s": round(life_s, 1), "answers_s": round(answered_s, 1),
                   "workers": args.workers}], lines)


def finish(args, rows, lines) -> None:
    for line in lines:
        print(line, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        for row in rows:
            row = dict(row)
            row["summary"] = lines
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["wrong", "sample", "run"])
    ap.add_argument("--scan", required=True, help="the folder of the published repository")
    ap.add_argument("--split", choices=sorted(O.FILES), default="length")
    ap.add_argument("--train", type=int, default=1000, help="training commands with a connective (0: all)")
    ap.add_argument("--test", type=int, default=300, help="sample: test commands (0: the whole file)")
    ap.add_argument("--only", default="", help="wrong: one split only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    {"wrong": wrong, "sample": sample, "run": run}[args.what](args)


if __name__ == "__main__":
    main()
