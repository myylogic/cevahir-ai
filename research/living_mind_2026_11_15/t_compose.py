# -*- coding: utf-8 -*-
"""T-compose: compositional generalisation in the manner of SCAN's "add primitive" split (Lake & Baroni 2018), small
enough for the mind's present window. A public test, generated here.

    python research/living_mind_2026_11_15/t_compose.py mind [--seeds 3] [--set key=0|1,...]
    python research/living_mind_2026_11_15/t_compose.py transformer [--seeds 3]

Commands are made of four primitives (walk, run, look, jump; their actions i_walk, i_run, i_look, i_jump), two
directions (left, right: i_lturn, i_rturn) and modifiers, in eleven templates whose answers have at most four actions:

    X                 -> A                    X opposite left   -> i_lturn i_lturn A
    X twice           -> A A                  X opposite right  -> i_rturn i_rturn A
    X thrice          -> A A A                X and Y           -> A B
    X left            -> i_lturn A            X after Y         -> B A
    X right           -> i_rturn A            X left twice      -> i_lturn A i_lturn A
                                              X twice and Y     -> A A B

Training: every command of walk, run and look in every template, and "jump" alone ("jump" -> "i_jump"), as dialogues
watched (someone says the command, someone answers it), each three times in a shuffled order. Test: every command in
which jump stands in a template (jump as X or as Y). A sequence-to-sequence network sees jump only alone and fails
this split almost entirely (Lake & Baroni 2018: about 1%); a mind generalises if what "jump" is answered alone carries
over to every place a primitive takes.

``mind``: the chat's speaking configuration, through Session.watch and Session.reply (each test command on a fresh
clone), ``--set`` switches for the whole life. ``transformer``: a decoder-only Transformer trained from scratch on the
same dialogues ("command ⏎ answer ."), greedy decoding after the turn end; trained until its answers of the training
commands are all exact (reported). Exact match of the whole answer. Appends the scores to ``results/compose.jsonl``.
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

OUT = os.path.join(HERE, "results", "compose.jsonl")
PRIMITIVES = {"walk": "i_walk", "run": "i_run", "look": "i_look", "jump": "i_jump"}
TURN = {"left": "i_lturn", "right": "i_rturn"}


def scan_commands() -> List[Tuple[List[str], List[str]]]:
    """SCAN's grammar (Lake & Baroni 2018), all 20,910 commands with their meanings (actions as this file writes them):
    C -> S and S | S after S | S;  S -> V twice | V thrice | V;  V -> U | D | U opposite D | U around D | turn opposite D
    | turn around D;  D -> U left | U right | turn left | turn right;  U -> walk | look | run | jump."""
    turn = {"left": "i_lturn", "right": "i_rturn"}
    v = [([u], [a]) for u, a in PRIMITIVES.items()]
    for d, t in turn.items():
        v.append((["turn", d], [t]))
        v.append((["turn", "opposite", d], [t, t]))
        v.append((["turn", "around", d], [t] * 4))
        for u, a in PRIMITIVES.items():
            v += [([u, d], [t, a]), ([u, "opposite", d], [t, t, a]), ([u, "around", d], [t, a] * 4)]
    sentences = []
    for c, a in v:
        sentences += [(c, a), (c + ["twice"], a * 2), (c + ["thrice"], a * 3)]
    out = list(sentences)
    for (c1, a1), (c2, a2) in itertools.product(sentences, sentences):
        out += [(c1 + ["and"] + c2, a1 + a2), (c1 + ["after"] + c2, a2 + a1)]
    return out


SCAN = 0                                                # --scan F: SCAN's commands that fit a window of F items


def answer(words: List[str]) -> List[str]:
    """The meaning of a command (the answer key; the mind is never given it)."""
    if SCAN:
        return SCAN_MEANING[tuple(words)]
    if "and" in words or "after" in words:
        j = words.index("and") if "and" in words else words.index("after")
        a, b = answer(words[:j]), answer(words[j + 1:])
        return a + b if "and" in words else b + a
    act = PRIMITIVES[words[0]]
    rest = words[1:]
    if not rest:
        return [act]
    if rest == ["twice"]:
        return [act, act]
    if rest == ["thrice"]:
        return [act, act, act]
    if rest in (["left"], ["right"]):
        return [TURN[rest[0]], act]
    if rest[0] == "opposite":
        return [TURN[rest[1]], TURN[rest[1]], act]
    if rest[0] == "around":
        return [TURN[rest[1]], act] * 4
    if rest[1:] == ["twice"]:
        return [TURN[rest[0]], act] * 2
    raise ValueError(words)


AROUND = False                                          # --around: the templates "X around left / right" too
WINDOW = 0                                              # --window: the mind's window F (0: as configured)


def commands() -> List[List[str]]:
    out = []
    for x in PRIMITIVES:
        out += [[x], [x, "twice"], [x, "thrice"], [x, "left"], [x, "right"], [x, "opposite", "left"],
                [x, "opposite", "right"], [x, "left", "twice"]]
        if AROUND:
            out += [[x, "around", "left"], [x, "around", "right"]]
        for y in PRIMITIVES:
            if y != x:
                out += [[x, "and", y], [x, "after", y], [x, "twice", "and", y]]
    return out


SCAN_MEANING: Dict[tuple, List[str]] = {}


def split() -> Tuple[List[List[str]], List[List[str]]]:
    if SCAN:
        if not SCAN_MEANING:
            for c, a in scan_commands():
                if len(c) + 1 + len(a) <= SCAN + 1:      # the command, its end and the answer fit the window
                    SCAN_MEANING[tuple(c)] = a
        every = [list(c) for c in SCAN_MEANING]
        return [c for c in every if "jump" not in c] + [["jump"]], [c for c in every if "jump" in c and c != ["jump"]]
    train = [c for c in commands() if "jump" not in c] + [["jump"]]
    test = [c for c in commands() if "jump" in c and c != ["jump"]]
    return train, test


def life(seed: int, epochs: int = 3) -> List[List[str]]:
    rng = random.Random(seed)
    train, _ = split()
    out = []
    for _ in range(epochs):
        block = list(train) + [["jump"]] * 4           # the primitive alone, as often as SCAN repeats it
        rng.shuffle(block)
        out += block
    return out


def run_mind(seed: int, switches: Dict[str, bool]) -> dict:
    import t_relations as R
    from zihin import Mind
    from zihin.__main__ import CHANNEL, Session
    torch.manual_seed(seed)
    mind = Mind(R.speaking_config(switches))
    mind.add_channel(CHANNEL, "act")
    if WINDOW:
        mind.knob["F"] = WINDOW                         # the mind's window (a knob it can turn itself, 3..15)
    s = Session(mind)
    t0 = time.time()
    for c in life(seed):
        s.watch(" ".join(c), " ".join(answer(c)) + " .")
    lived = time.time() - t0
    _, test = split()
    right, by_template, wrong = 0, {}, []
    for c in test:
        twin = Session(mind.clone())
        said = twin.reply(" ".join(c))
        ok = said == " ".join(answer(c))
        right += ok
        key = " ".join("X" if w in PRIMITIVES else w for w in c)
        hit = by_template.setdefault(key, [0, 0])
        hit[0] += ok
        hit[1] += 1
        if not ok and len(wrong) < 6:
            wrong.append("%s -> %s" % (" ".join(c), said))
    return {"seed": seed, "exact": right, "n": len(test), "by_template": by_template, "wrong": wrong,
            "moments": mind.t, "seconds": round(lived, 1)}


def run_transformer(seed: int, steps: int = 6000) -> dict:
    import torch.nn as nn
    torch.manual_seed(seed)
    seqs = [c + ["⏎"] + answer(c) + ["."] for c in life(seed)]
    vocab = sorted({w for s in seqs for w in s} | {w for c in split()[1] for w in c + answer(c)} | {"<pad>"})
    ix = {w: i for i, w in enumerate(vocab)}
    L = max(len(s) for s in seqs) + 2

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
    for i, s in enumerate(seqs):
        data[i, :len(s)] = torch.tensor([ix[w] for w in s])
    t0 = time.time()
    loss_v = None
    for step in range(steps):
        batch = data[torch.randint(len(seqs), (32,))]
        logits = net(batch[:, :-1])
        loss = nn.functional.cross_entropy(logits.reshape(-1, len(vocab)), batch[:, 1:].reshape(-1),
                                           ignore_index=ix["<pad>"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_v = float(loss.detach())
    net.eval()
    train, test = split()

    def decode(c):
        seq = [ix[w] for w in c + ["⏎"]]
        out = []
        for _ in range(16):
            nxt = int(net(torch.tensor([seq]))[0, -1].argmax())
            if vocab[nxt] in (".", "<pad>"):
                break
            out.append(vocab[nxt])
            seq.append(nxt)
        return out
    right, by_template, wrong = 0, {}, []
    with torch.no_grad():
        learned = sum(decode(c) == answer(c) for c in train)
        for c in test:
            out = decode(c)
            ok = out == answer(c)
            right += ok
            key = " ".join("X" if w in PRIMITIVES else w for w in c)
            hit = by_template.setdefault(key, [0, 0])
            hit[0] += ok
            hit[1] += 1
            if not ok and len(wrong) < 6:
                wrong.append("%s -> %s" % (" ".join(c), " ".join(out)))
    return {"seed": seed, "exact": right, "n": len(test), "by_template": by_template, "wrong": wrong,
            "train_loss": round(loss_v, 4), "train_exact": learned, "train_n": len(train),
            "seconds": round(time.time() - t0, 1)}


def main() -> None:
    sys.path.insert(0, HERE)
    ap = argparse.ArgumentParser()
    ap.add_argument("who", choices=["mind", "transformer"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--set", default="")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--around", action="store_true")
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--scan", type=int, default=0, help="SCAN's commands that fit a window of this many items")
    args = ap.parse_args()
    global AROUND, WINDOW, SCAN
    AROUND, WINDOW, SCAN = args.around, args.window, args.scan
    if SCAN and not WINDOW:
        WINDOW = SCAN
    torch.set_num_threads(1)
    switches = {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, args.set.split(",")))}
    rows = [run_mind(s, switches) if args.who == "mind" else run_transformer(s) for s in range(args.seeds)]
    total = sum(r["exact"] for r in rows)
    n = sum(r["n"] for r in rows)
    templates: Dict[str, List[int]] = {}
    for r in rows:
        for k, (a, b) in r["by_template"].items():
            t = templates.setdefault(k, [0, 0])
            t[0] += a
            t[1] += b
    print("== %s, add-primitive split, %d seeds, switches %s: exact %d / %d = %.3f ==" % (
        args.who, args.seeds, switches or "-", total, n, total / n))
    for k, (a, b) in sorted(templates.items()):
        print("  %-22s %d / %d" % (k, a, b))
    for w in rows[0]["wrong"]:
        print("  wrong (seed 0): %s" % w)
    if args.who == "transformer":
        print("  training commands answered exactly: %s" % [(r["train_exact"], r["train_n"]) for r in rows])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"who": args.who, "seeds": args.seeds, "switches": switches, "around": AROUND,
                            "window": WINDOW, "scan": SCAN, "exact": total, "n": n,
                            "templates": templates, "rows": rows}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
