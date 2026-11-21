# -*- coding: utf-8 -*-
"""T-scan: SCAN's add-jump split (Lake & Baroni 2018) with commands and answers of every length. A public test,
generated here from SCAN's own grammar (``t_compose.scan_commands``: all 20,910 commands with their meanings).

    python research/living_mind_2026_11_15/t_scan.py [--train 1000] [--test 500] [--window 13] [--set key=0|1,...]
    python research/living_mind_2026_11_15/t_scan.py --who transformer [--train 1000] [--test 500] [--steps 6000]

The split. Training: commands without "jump", and "jump" alone (its answer "i_jump"). Test: every command in which
jump stands (7,706 commands; "jump" alone is not one of them). SCAN's whole training set is 13,204 commands; the
mind here watches a part of it, written in the result: the 81 commands without a connective (and "jump"), and
``--train`` commands with "and" / "after" drawn at random (0: all 13,122). ``--test``: that many test commands drawn at
random (0: all 7,706).

The life. Each training dialogue is watched once per epoch (someone says the command, someone answers it); in the
first epoch the shorter commands come first (a curriculum: a part is met before the commands it stands in), in the
later epochs the order is shuffled (``--shuffled``: no curriculum, every epoch shuffled). "jump" alone is watched as
often as the other one-word commands and, as SCAN repeats it, four more times an epoch.

The test. Each test command is told to a copy of the mind as it was at the end of its life (a forked process: the
copy is the mind, bit for bit, and nothing one answer leaves is there for the next). Exact match of the whole answer.
Reported: all, and by the size of the dialogue (the command, its end and the answer fit a window of 12 items, or do
not), by connective, and by the length of the answer. Also: a hundred of the watched commands whose dialogue does not
fit 12 items, asked again (what was lived, not the split's test).

A sequence-to-sequence network sees "jump" only alone and fails this split almost entirely (Lake & Baroni 2018: 1.2%
at best). ``--who transformer``: a decoder-only Transformer trained from scratch on the very dialogues the mind
watches, greedy decoding after the turn end; for it "watched again" is 300 of its training commands, and the times
are training and decoding.

Appends to ``results/scan.jsonl``.
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

OUT = os.path.join(HERE, "results", "scan.jsonl")
MIND = None                                             # the mind at the end of its life (read by forked copies)


def split(seed: int, n_train: int, n_test: int) -> Tuple[List[Tuple[tuple, tuple]], List[Tuple[tuple, tuple]]]:
    import t_compose as T
    every = [(tuple(c), tuple(a)) for c, a in T.scan_commands()]
    rng = random.Random(seed)
    plain = [d for d in every if "jump" not in d[0]]
    simple = [d for d in plain if "and" not in d[0] and "after" not in d[0]]
    joined = [d for d in plain if "and" in d[0] or "after" in d[0]]
    test = [d for d in every if "jump" in d[0] and d[0] != ("jump",)]
    if n_train:
        joined = rng.sample(joined, min(n_train, len(joined)))
    if n_test:
        test = rng.sample(test, min(n_test, len(test)))
    return simple + joined + [(("jump",), ("i_jump",))], test


SHUFFLED = False                                        # --shuffled: no curriculum (every epoch in a shuffled order)


def life(seed: int, train, epochs: int) -> List[Tuple[tuple, tuple]]:
    rng = random.Random(seed + 1)
    out: List[Tuple[tuple, tuple]] = []
    for epoch in range(epochs):
        block = list(train) + [(("jump",), ("i_jump",))] * 4
        rng.shuffle(block)
        if epoch == 0 and not SHUFFLED:
            block.sort(key=lambda d: len(d[0]))          # stable: the shorter commands first, shuffled among equals
        out += block
    return out


def _answer(command: tuple) -> str:
    from zihin.__main__ import Session
    return Session(MIND).reply(" ".join(command), limit=400)


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
    probe = random.Random(seed + 2).sample([d for d in train if len(d[0]) + 1 + len(d[1]) > 12], 100)
    t0 = time.time()
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        said = pool.map(_answer, [c for c, _ in test], chunksize=1)
        again = pool.map(_answer, [c for c, _ in probe], chunksize=1)
    answered = time.time() - t0
    watched = sum(got == " ".join(answer) for (_, answer), got in zip(probe, again))
    rows = []
    for (command, answer), got in zip(test, said):
        rows.append({"command": " ".join(command), "said": got, "right": got == " ".join(answer),
                     "fits12": len(command) + 1 + len(answer) <= 12, "answer": len(answer),
                     "joined": "and" if "and" in command else ("after" if "after" in command else "-")})
    return {"seed": seed, "train": len(train), "moments": mind.t, "ties": mind.S.n_ties, "lived_s": round(lived, 1),
            "answered_s": round(answered, 1), "watched_again": [watched, len(probe)], "rows": rows}


def run_transformer(seed: int, n_train: int, n_test: int, epochs: int, steps: int) -> dict:
    """The comparison: a decoder-only Transformer (3 layers, 4 heads, width 64) trained from scratch on the very
    dialogues the mind watches ("command ⏎ answer ."), greedy decoding after the turn end."""
    import torch.nn as nn
    torch.manual_seed(seed)
    train, test = split(seed, n_train, n_test)
    seqs = [list(c) + ["⏎"] + list(a) + ["."] for c, a in life(seed, train, epochs)]
    vocab = sorted({w for q in seqs for w in q} | {w for c, a in test for w in c + a} | {"<pad>"})
    ix = {w: i for i, w in enumerate(vocab)}
    L = max(max(len(q) for q in seqs), max(len(c) + len(a) + 2 for c, a in test)) + 2

    class Net(nn.Module):
        def __init__(self, d=64, h=4, layers=3):
            super().__init__()
            self.emb, self.pos = nn.Embedding(len(vocab), d), nn.Embedding(L, d)
            block = nn.TransformerEncoderLayer(d, h, 4 * d, dropout=0.1, batch_first=True)
            self.body = nn.TransformerEncoder(block, layers)
            self.out = nn.Linear(d, len(vocab))

        def forward(self, x):
            n = x.shape[1]
            mask = torch.triu(torch.full((n, n), float("-inf")), 1)
            return self.out(self.body(self.emb(x) + self.pos(torch.arange(n)), mask=mask))

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    data = torch.full((len(seqs), L), ix["<pad>"])
    for i, q in enumerate(seqs):
        data[i, :len(q)] = torch.tensor([ix[w] for w in q])
    t0 = time.time()
    for _ in range(steps):
        batch = data[torch.randint(len(seqs), (32,))]
        logits = net(batch[:, :-1])
        loss = nn.functional.cross_entropy(logits.reshape(-1, len(vocab)), batch[:, 1:].reshape(-1),
                                           ignore_index=ix["<pad>"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    trained = time.time() - t0
    net.eval()

    def decode(c) -> tuple:
        seq, out = [ix[w] for w in list(c) + ["⏎"]], []
        while len(seq) < L - 1:
            nxt = int(net(torch.tensor([seq]))[0, -1].argmax())
            if vocab[nxt] in (".", "<pad>"):
                break
            out.append(vocab[nxt])
            seq.append(nxt)
        return tuple(out)
    rows = []
    with torch.no_grad():
        probe = random.Random(seed + 2).sample(train, min(300, len(train)))
        learned = sum(decode(c) == a for c, a in probe)
        for c, a in test:
            got = decode(c)
            rows.append({"command": " ".join(c), "said": " ".join(got), "right": got == a,
                         "fits12": len(c) + 1 + len(a) <= 12, "answer": len(a),
                         "joined": "and" if "and" in c else ("after" if "after" in c else "-")})
    return {"seed": seed, "train": len(train), "moments": 0, "ties": 0, "lived_s": round(trained, 1),
            "answered_s": round(time.time() - t0 - trained, 1), "watched_again": [learned, len(probe)], "rows": rows,
            "loss": round(float(loss.detach()), 4), "steps": steps}


def main() -> None:
    sys.path.insert(0, HERE)
    ap = argparse.ArgumentParser()
    ap.add_argument("--who", choices=["mind", "transformer"], default="mind")
    ap.add_argument("--steps", type=int, default=6000, help="transformer: training steps (batches of 32)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", type=int, default=1000, help="training commands with and / after (0: all 13,122)")
    ap.add_argument("--test", type=int, default=500, help="test commands (0: all 7,706)")
    ap.add_argument("--window", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--set", default="")
    ap.add_argument("--shuffled", action="store_true", help="no curriculum: the first epoch in a shuffled order too")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    global SHUFFLED
    SHUFFLED = args.shuffled
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
    lines = ["== SCAN add-jump: %d training dialogues an epoch (%d epochs%s, window %d), switches %s: exact %s ==" % (
        r["train"], args.epochs, ", shuffled" if SHUFFLED else "", args.window, switches or "-", of(lambda x: True)),
        "  the dialogue fits 12 items: %s; it does not: %s" % (of(lambda x: x["fits12"]), of(lambda x: not x["fits12"])),
        "  no connective: %s; and: %s; after: %s" % (of(lambda x: x["joined"] == "-"), of(lambda x: x["joined"] == "and"),
                                                    of(lambda x: x["joined"] == "after")),
        "  answers of 1-8 actions: %s; 9-16: %s; 17-24: %s; 25-48: %s" % (
            of(lambda x: x["answer"] <= 8), of(lambda x: 8 < x["answer"] <= 16), of(lambda x: 16 < x["answer"] <= 24),
            of(lambda x: x["answer"] > 24)),
        "  watched commands that do not fit 12 items, asked again: %d / %d" % tuple(r["watched_again"]),
        "  lived %d moments in %.0f s (%d ties); answered in %.0f s" % (r["moments"], r["lived_s"], r["ties"],
                                                                         r["answered_s"])]
    if args.who == "transformer":
        lines[0] = lines[0].replace("== SCAN add-jump:", "== SCAN add-jump, a Transformer:")
        lines[-2:] = ["  its training commands (300 of them): %d / %d" % tuple(r["watched_again"]),
                      "  trained %d steps in %.0f s (loss %.4f); decoded in %.0f s" % (
                          r["steps"], r["lived_s"], r["loss"], r["answered_s"])]
    lines += ["  wrong: %s -> %s" % (x["command"], x["said"][:160]) for x in [x for x in rows if not x["right"]][:6]]
    for line in lines:
        print(line)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"who": args.who, "seed": args.seed, "train": args.train, "test": args.test,
                            "window": args.window, "epochs": args.epochs, "shuffled": SHUFFLED, "switches": switches,
                            "summary": lines, **r},
                           ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
