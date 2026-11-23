# -*- coding: utf-8 -*-
"""t_m1.py: milestone M1 of MODEL.md, the baseline in the grounded world (dunya.py).

    python research/living_mind_2026_11_23/t_m1.py --who mind --lang tr --seeds 0 1 2 [--scene]
    python research/living_mind_2026_11_23/t_m1.py --who transformer --lang tr --seeds 0 1 2 [--scene] [--steps 4000]

The same life (``World.training``) is lived by each learner, and the same questions (``World.tests``) asked after it:

* ``mind``: today's whole mind (``VARIANTS["full"]``, as the program lives). Facts and namings are heard, the
  dialogues watched. With ``--scene`` what is seen while a line is said is shown with it as today's core can take a
  scene: one image-like thing with its facets (``Session.view``: the features of every thing present and of the event
  going on, without which thing has which: the binding today's core lacks, MODEL.md section 2). The scene of a
  dialogue is shown with its first line: an order is heard while it is being carried out (the core answers at once;
  it does not wait for the event).
* ``transformer``: a small causal Transformer trained from scratch on the same lines ("question ⏎ answer ."; facts as
  "fact ."), greedy answers. With ``--scene`` the scene's facets come first, as tokens, before the line.

Every answer is scored exact (the words of the answer, without the closing punctuation); "X nedir ?" also by how many
of its three facts the answer says. Rows are appended to ``results/m1.jsonl``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dunya import World  # noqa: E402

DIALOGUES = ("order", "who", "doing", "what_is", "what_this")


def words(text: str):
    return [w for w in text.lower().split() if w not in (".", "?", "!")]


def facts_said(said: str, facts) -> int:
    got = " ".join(words(said))
    return sum(" ".join(words(f)) in got for f in facts)


def gist(facets) -> str:
    return "g" + hashlib.md5(" ".join(facets).encode("utf-8")).hexdigest()[:16]


def seen_with(line: dict):
    """What is shown with a line: the scene of its answer (the event going on) when it has one, else its own."""
    scene = line.get("scene_answer") or line.get("scene")
    return scene["flat"] if scene else None


def run_mind(world: World, train, tests, scene: bool, on=()) -> dict:
    import torch
    torch.set_num_threads(1)
    import zihin.__main__ as chat
    from zihin import IMITATED, Mind
    from zihin.__main__ import CHANNEL, Session
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = chat.VARIANTS["full"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    for k in on:
        setattr(cfg, k, True)
    torch.manual_seed(0)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    t0 = time.process_time()
    with torch.no_grad():
        for line in train:
            facets = seen_with(line) if scene else None
            if facets:
                s.view = (gist(facets), facets)
            if line["kind"] in DIALOGUES:
                s.watch(line["text"], line["answer"])
            else:
                s.company(None)
                s.listen(line["text"], IMITATED)
                s.end_of_turn(IMITATED, tone=".")
                mind.step([])
        s.close_turn()
        lived = time.process_time() - t0

        def answer(line):
            c = Session(mind.clone())
            facets = seen_with(line) if scene else None
            if facets:
                c.view = (gist(facets), facets)
            return c.reply(line["text"], limit=40)
        return examine(world, tests, answer, {"who": "mind", "moments": mind.t, "ties": mind.S.n_ties,
                                              "learn_cpu_s": round(lived, 1)}, heard_words(train))


def run_transformer(world: World, train, tests, scene: bool, steps: int, seed: int) -> dict:
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    def seq(line, with_answer=True):
        head = (seen_with(line) or []) + ["|"] if scene else []
        out = head + words(line["text"])
        if line["kind"] in DIALOGUES:
            out = out + ["⏎"] + (words(line["answer"]) + ["."] if with_answer else [])
        else:
            out = out + ["."]
        return out
    seqs = [seq(l) for l in train]
    vocab = sorted({w for s in seqs for w in s} | {w for items in tests.values() for l in items for w in seq(l)}
                   | {"<pad>", "⏎", ".", "|"})
    ix = {w: i for i, w in enumerate(vocab)}
    L = max(len(s) for s in seqs) + 2

    class Net(nn.Module):
        def __init__(self, d=64, h=4, layers=3):
            super().__init__()
            self.emb, self.pos = nn.Embedding(len(vocab), d), nn.Embedding(L + 48, d)
            block = nn.TransformerEncoderLayer(d, h, 4 * d, dropout=0.1, batch_first=True)
            self.body = nn.TransformerEncoder(block, layers)
            self.out = nn.Linear(d, len(vocab))

        def forward(self, x):
            n = x.shape[1]
            mask = torch.triu(torch.full((n, n), float("-inf")), 1)
            return self.out(self.body(self.emb(x) + self.pos(torch.arange(n)), mask=mask))
    net = Net()
    params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    data = torch.full((len(seqs), L), ix["<pad>"])
    for i, s in enumerate(seqs):
        data[i, :len(s)] = torch.tensor([ix[w] for w in s])
    t0 = time.process_time()
    net.train()
    for _ in range(steps):
        batch = data[torch.randint(len(seqs), (32,))]
        logits = net(batch[:, :-1])
        loss = nn.functional.cross_entropy(logits.reshape(-1, len(vocab)), batch[:, 1:].reshape(-1),
                                           ignore_index=ix["<pad>"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    lived = time.process_time() - t0
    net.eval()

    def answer(line):
        ids = [ix[w] for w in seq(line, with_answer=False)]
        out = []
        with torch.no_grad():
            for _ in range(40):
                nxt = int(net(torch.tensor([ids]))[0, -1].argmax())
                if vocab[nxt] in (".", "<pad>") and len(out) and line["kind"] != "what_is":
                    break
                if vocab[nxt] == "<pad>" or (vocab[nxt] == "." and out and out[-1] == "."):
                    break
                out.append(vocab[nxt])
                ids.append(nxt)
        while out and out[-1] == ".":
            out.pop()
        return " ".join(out) or "(sessiz)"
    return examine(world, tests, answer, {"who": "transformer", "params": params, "steps": steps,
                                          "loss": round(float(loss.detach()), 4), "learn_cpu_s": round(lived, 1)},
                   heard_words(train))


def heard_words(train) -> set:
    return {w for l in train for w in words(l["text"]) + words(l["answer"] or "")}


def examine(world: World, tests, answer, row: dict, heard: set) -> dict:
    """Scores per test: [right, asked, right among the answers made only of words heard in the life, those asked]
    (an answer with a word form never heard, a Turkish case form never met, is counted apart: MODEL.md 7.4), and
    for "X nedir ?" the facts said and the facts there were."""
    t0 = time.process_time()
    scores, shown = {}, {}
    for name, items in tests.items():
        right = told = told_of = right_heard = n_heard = 0
        for line in items:
            said = answer(line)
            ok = said != "(sessiz)" and words(said) == words(line["answer"])
            right += ok
            if all(w in heard or w in words(line["text"]) for w in words(line["answer"])):   # (the cue's words too)
                n_heard += 1
                right_heard += ok
            if line["kind"] == "what_is":
                facts = world.facts(line["tags"]["thing"])[:3]
                told += facts_said(said, facts)
                told_of += len(facts)
            shown.setdefault(name, [])
            if len(shown[name]) < 4:
                shown[name].append([line["text"], said, line["answer"]])
        scores[name] = [right, len(items), right_heard, n_heard] + ([told, told_of] if told_of else [])
    row.update({"scores": scores, "examples": shown, "answer_cpu_s": round(time.process_time() - t0, 1)})
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--who", choices=("mind", "transformer"), default="mind")
    ap.add_argument("--lang", default="tr")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--scene", action="store_true")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--on", nargs="*", default=[])
    ap.add_argument("--rows", default=os.path.join(HERE, "results", "m1.jsonl"))
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    for seed in args.seeds:
        world = World(seed, args.lang)                   # the life first, then the questions: the world's draws
        train = world.training()                         # are the same for every learner
        tests = world.tests()
        digest = hashlib.md5(json.dumps([train, tests], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if args.who == "mind":
            row = run_mind(world, train, tests, args.scene, args.on)
        else:
            row = run_transformer(world, train, tests, args.scene, args.steps, seed)
        row.update({"test": "m1", "seed": seed, "lang": args.lang, "scene": args.scene, "on": args.on,
                    "world": digest, "note": args.note})
        print("%-11s %s seed %d scene %-5s %s" % (row["who"], args.lang, seed, args.scene,
                                                  "  ".join("%s %s" % (k, "/".join(map(str, v)))
                                                            for k, v in row["scores"].items())), flush=True)
        if args.rows:
            with open(args.rows, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
