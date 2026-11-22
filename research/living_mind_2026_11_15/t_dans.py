# -*- coding: utf-8 -*-
"""T-dans: moves with a direction, in Turkish; a template split. A public test, generated here. It was written before
F45 (Told Apart) was run on it: F45 was developed on SCAN's "around right" split (``t_scan_splits.py``) only. This
world differs from SCAN in its words, in where the direction stands (before the move, not after it), in the words
that make a pattern (two words, "iki kez" / "üç kez", and "uzun"), and in what is held out (one direction with two
pattern words at once).

    python research/living_mind_2026_11_15/t_dans.py [--seeds 3] [--train 150] [--window 13] [--set key=0|1,...]
    python research/living_mind_2026_11_15/t_dans.py --who transformer [--steps 6000]
    python research/living_mind_2026_11_15/t_dans.py --dump dans.tsv      (the training dialogues of seed 0, for !ogret)

The world. Four moves, each answered with one step when said alone:
    zıpla -> sıçra        eğil -> bük        dön -> çevir        yürü -> adım
Four directions, never said alone; each stands for one step wherever it is said:
    sağa -> sağ        sola -> sol        öne -> ön        geriye -> geri
A part is a move ("zıpla"), or a move with a direction:
    "sağa zıpla" -> sağ sıçra                    "sağa uzun zıpla" -> sağ sağ sıçra
    "sağa iki kez zıpla" -> sağ sıçra sağ sıçra   "sağa üç kez zıpla" -> sağ sıçra sağ sıçra sağ sıçra
A command is a part, or "P sonra Q" (the steps of P, then of Q). 68 parts, 4,692 commands; answers of up to 12
steps, commands of up to 9 words.

The split. Held out: every command in which "geriye iki kez" or "geriye üç kez" stands (1,032 commands). "geriye" is
met alone with a move and with "uzun"; "iki kez" and "üç kez" are met with the three other directions. Training: the
60 parts in which neither stands, and ``--train`` of the 3,600 commands that join two of them, drawn at random; each
watched once an epoch, two epochs, the shorter commands first in the first epoch. Test: ``--test`` of the held-out
commands drawn at random (0: all), each told to a copy of the mind as it was at the end of its life. Exact match of
the whole answer.

Reported: all; commands that are one part, and commands that join two; by how many of a command's parts are held
out; silent (nothing said) and wrong apart.

``--tek`` (one dialogue; added after the first run of the split above, and written before it was run itself).
"geriye" is met in one dialogue only, "geriye zıpla" (watched once an epoch). Training: that dialogue, the 52 parts
without "geriye" and ``--train`` of the 2,704 commands that join two of them. Test: every other command in which
"geriye" stands (1,935), with a move it was never met with, with "uzun", with "iki kez" and "üç kez", alone and
joined.

``--who transformer``: the decoder-only Transformer of ``t_scan.py`` trained from scratch on the very dialogues the
mind watches (seed 0 only).

Appends to ``results/dans.jsonl``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "dans.jsonl")
MOVES = {"zıpla": "sıçra", "eğil": "bük", "dön": "çevir", "yürü": "adım"}
DIRECTIONS = {"sağa": "sağ", "sola": "sol", "öne": "ön", "geriye": "geri"}
NEW = "geriye"
ONE = (("geriye", "zıpla"), ("geri", "sıçra"))         # --tek: the one dialogue in which the new direction is met
TEK = False
Dialogue = Tuple[tuple, tuple]


def parts() -> List[Dialogue]:
    out: List[Dialogue] = [((move,), (step,)) for move, step in MOVES.items()]
    for (direction, d), (move, step) in itertools.product(DIRECTIONS.items(), MOVES.items()):
        out += [((direction, move), (d, step)), ((direction, "uzun", move), (d, d, step)),
                ((direction, "iki", "kez", move), (d, step) * 2), ((direction, "üç", "kez", move), (d, step) * 3)]
    return out


def commands() -> List[Dialogue]:
    every = list(parts())
    for (p, a), (q, b) in itertools.product(parts(), parts()):
        every.append((p + ("sonra",) + q, a + b))
    return every


def held_out(command: tuple) -> bool:
    if TEK:
        return NEW in command and command != ONE[0]
    return any(command[i] == NEW and command[i + 2] == "kez" for i in range(len(command) - 2))


def pieces(command: tuple) -> List[tuple]:
    if "sonra" in command:
        i = command.index("sonra")
        return [command[:i], command[i + 1:]]
    return [command]


def split(seed: int, n_train: int, n_test: int = 0) -> Tuple[List[Dialogue], List[Dialogue]]:
    rng = random.Random(seed)
    every = commands()
    plain = [d for d in every if not held_out(d[0])]
    simple = [d for d in plain if "sonra" not in d[0]]
    joined = [d for d in plain if "sonra" in d[0]]
    test = [d for d in every if held_out(d[0])]
    if n_train:
        joined = rng.sample(joined, min(n_train, len(joined)))
    if n_test:
        test = rng.sample(test, min(n_test, len(test)))
    return simple + joined, test


def life(seed: int, train, epochs: int) -> List[Dialogue]:
    rng = random.Random(seed + 1)
    out: List[Dialogue] = []
    for epoch in range(epochs):
        block = list(train)
        rng.shuffle(block)
        if epoch == 0:
            block.sort(key=lambda d: len(d[0]))          # stable: the shorter commands first, shuffled among equals
        out += block
    return out


def rows_of(test, said) -> List[dict]:
    rows = []
    for (command, answer), got in zip(test, said):
        rows.append({"command": " ".join(command), "said": got, "right": got == " ".join(answer),
                     "silent": got.strip() in ("", "(sessiz)"), "joined": "sonra" in command,
                     "held_parts": sum(held_out(p) for p in pieces(command)), "answer": len(answer)})
    return rows


def run(seed: int, switches: Dict[str, bool], n_train: int, n_test: int, window: int, epochs: int) -> dict:
    import t_relations as R
    from zihin import Mind
    from zihin.__main__ import CHANNEL, Session
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
    t0 = time.time()
    said = [Session(mind.clone()).reply(" ".join(command), limit=400) for command, _ in test]
    return {"seed": seed, "train": len(train), "moments": mind.t, "lived_s": round(lived, 1),
            "answered_s": round(time.time() - t0, 1), "rows": rows_of(test, said)}


def run_transformer(seed: int, n_train: int, n_test: int, epochs: int, steps: int) -> dict:
    """The Transformer of ``t_scan.py`` (3 layers, 4 heads, width 64; greedy decoding after the turn end), trained
    from scratch on the dialogues of this world's life."""
    import t_scan as S
    S.split, S.life = split, life
    r = S.run_transformer(seed, n_train, n_test, epochs, steps)
    _, test = split(seed, n_train, n_test)
    r["rows"] = rows_of(test, [x["said"] for x in r["rows"]])
    return r


def main() -> None:
    sys.path.insert(0, HERE)
    ap = argparse.ArgumentParser()
    ap.add_argument("--who", choices=["mind", "transformer"], default="mind")
    ap.add_argument("--steps", type=int, default=6000, help="transformer: training steps (batches of 32)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--train", type=int, default=150, help="training commands that join two parts (0: all 3,600)")
    ap.add_argument("--test", type=int, default=200, help="held-out commands asked, per seed (0: all 1,032)")
    ap.add_argument("--window", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--set", default="")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--tek", action="store_true", help="the new direction is met in one dialogue only (geriye zıpla)")
    ap.add_argument("--dump", help="write the training dialogues of seed 0 (one epoch, the shorter first) as "
                    "'command<TAB>answer' lines, for the chat's !ogret, and stop")
    args = ap.parse_args()
    global TEK
    TEK = args.tek
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for command, answer in life(0, split(0, args.train)[0], 1):
                f.write("%s\t%s .\n" % (" ".join(command), " ".join(answer)))
        print("%s: %d dialogues" % (args.dump, len(split(0, args.train)[0])))
        return
    torch.set_num_threads(1)
    switches = {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, args.set.split(",")))}
    if args.who == "transformer":
        runs = [run_transformer(0, args.train, args.test, args.epochs, args.steps)]
        switches = {"transformer": True}
    else:
        runs = [run(seed, switches, args.train, args.test, args.window, args.epochs) for seed in range(args.seeds)]
    rows = [x for r in runs for x in r["rows"]]

    def of(keep) -> str:
        kept = [x for x in rows if keep(x)]
        return "%d / %d" % (sum(x["right"] for x in kept), len(kept))
    wrong = [x for x in rows if not x["right"]]
    lines = ["== T-dans%s%s: %d seeds, %d training dialogues an epoch (%d epochs, window %d), switches %s: exact %s ==" % (
        ", one dialogue with the new direction" if TEK else "",
        ", a Transformer" if args.who == "transformer" else "", len(runs), runs[0]["train"], args.epochs, args.window,
        switches or "-", of(lambda x: True)),
        "  not exact: %d silent, %d wrong" % (sum(x["silent"] for x in wrong), sum(not x["silent"] for x in wrong)),
        "  one part: %s; two parts joined: %s (one of them held out: %s; both: %s)" % (
            of(lambda x: not x["joined"]), of(lambda x: x["joined"]),
            of(lambda x: x["joined"] and x["held_parts"] == 1), of(lambda x: x["held_parts"] == 2))]
    if args.who == "transformer":
        lines += ["  its training commands (300 of them): %d / %d" % tuple(runs[0]["watched_again"]),
                  "  trained %d steps in %.0f s (loss %.4f)" % (runs[0]["steps"], runs[0]["lived_s"], runs[0]["loss"])]
    else:
        lines += ["  lived %s moments (seed 0) in %.0f s; answered in %.0f s" % (
            runs[0]["moments"], runs[0]["lived_s"], runs[0]["answered_s"])]
    lines += ["  not exact (seed 0): %s -> %s" % (x["command"], "(silent)" if x["silent"] else x["said"][:150])
              for x in [x for x in runs[0]["rows"] if not x["right"]][:6]]
    for line in lines:
        print(line)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"who": args.who, "tek": TEK, "seeds": len(runs), "train": args.train, "test": args.test,
                            "window": args.window, "epochs": args.epochs, "switches": switches, "summary": lines,
                            "runs": runs}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
