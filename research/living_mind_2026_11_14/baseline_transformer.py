# -*- coding: utf-8 -*-
"""Transformer baselines on the identical stream (design §11: "if the snapshot baselines match the mind ... the
design is wrong").

    python research/living_mind_2026_11_14/baseline_transformer.py --domains nitelik iliski --seeds 0 1 2 3 4

For each (domain, seed) the same world schedule (600 experiences) is lived by:
  zihin      the mind, variant "members" (F11)
  online     a small causal Transformer trained one step per utterance as it arrives (one pass, no replay)
  offline    the same architecture trained from scratch on the whole stream until the loss stops falling (snapshot)
and probed identically: the right value must be the top one among the values of the asked dimension.
  held_told / held_induced   instances never told that dimension (T3)
  members, exceptions        facts that were told (T3)
  early                      facts told once in the first fifth and never again (T5)
Interference (T6): online model and mind live domain A (600) then B (600); A's members are probed before and after.
Retraction (T9): the mind retracts one experience exactly (time measured); the Transformer's exact counterpart is
retraining from scratch without it (time of one offline training is reported).
Advantages given to the Transformer, stated: it knows the whole vocabulary in advance; the offline model sees every
experience many times. Hyperparameters are small standard choices, not tuned to the probes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run import Probe, World, live  # noqa: E402

torch.set_num_threads(1)
BOS, EOS = "<s>", "</s>"


class TinyGPT(nn.Module):
    def __init__(self, vocab: int, d: int = 64, layers: int = 2, heads: int = 4, ctx: int = 12) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d, vocab)
        self.ctx = ctx

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n = ids.shape[1]
        mask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1)
        h = self.tok(ids) + self.pos(torch.arange(n))
        return self.out(self.blocks(h, mask=mask, is_causal=True))


def encode(vocab, symbols):
    return [vocab[BOS]] + [vocab[s] for s in symbols] + [vocab[EOS]]


def step(model, opt, seqs):
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), -1, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
    x, y = ids[:, :-1].clamp(min=0), ids[:, 1:]
    logits = model(x)
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-1)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss)


def train_online(model, vocab, utterances, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for symbols in utterances:
        step(model, opt, [encode(vocab, symbols)])
    return model


def train_offline(vocab, utterances, seed, max_epochs=300, batch=32, lr=3e-3):
    torch.manual_seed(seed)
    model = TinyGPT(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = [encode(vocab, s) for s in utterances]
    g = torch.Generator().manual_seed(seed)
    best, stale, start = math.inf, 0, time.time()
    for epoch in range(max_epochs):
        order = torch.randperm(len(data), generator=g).tolist()
        total = 0.0
        for i in range(0, len(order), batch):
            total += step(model, opt, [data[j] for j in order[i:i + batch]])
        if total < best - 1e-3:
            best, stale = total, 0
        else:
            stale += 1
            if stale >= 10:                       # the loss stopped falling
                break
    return model, time.time() - start, epoch + 1


def top_is(dist, values, right):
    best = max(values, key=lambda v: dist.get(v, float("-inf")))
    return float(best == right)


def tf_dist(model, vocab, prefix):
    with torch.no_grad():
        ids = torch.tensor([[vocab[BOS]] + [vocab[s] for s in prefix]])
        logp = torch.log_softmax(model(ids)[0, -1], -1)
    inv = {i: s for s, i in vocab.items()}
    return {inv[i]: float(v) for i, v in enumerate(logp)}


def zihin_dist(mind, prefix):
    return Probe(mind).ask(prefix)                # expectations by symbol; missing = never expected


def probes(world, state):
    """(name, prefix, values, right) for the T3 / T5 probes, from what was actually lived."""
    out = []
    for ent in world.held_out:
        if not state["isa"].get(ent):
            continue
        for dim in world.ruled:
            kind = "held_told" if dim == world.told_rule_dim else "held_induced"
            out.append((kind, world.prefix(ent, dim), world.values[dim], world.rule[(world.cat_of[ent], dim)]))
    for (ent, dim), heard in state["heard"].items():
        if heard["lie"]:
            continue
        right = world.value[(ent, dim)]
        if (ent, dim) in world.exceptions:
            out.append(("exceptions", world.prefix(ent, dim), world.values[dim], right))
        elif dim in world.ruled:
            out.append(("members", world.prefix(ent, dim), world.values[dim], right))
    for (ent, dim) in state["early"]:
        out.append(("early", world.prefix(ent, dim), world.values[dim], world.value[(ent, dim)]))
    return out


def score(dist_fn, items):
    acc = {}
    for kind, prefix, values, right in items:
        acc.setdefault(kind, []).append(top_is(dist_fn(prefix), values, right))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def one(domain, seed, n):
    world = World.make(seed, domain)
    t0 = time.time()
    mind, _, state = live(world, "members", n, record=True)
    t_mind = time.time() - t0
    utterances = [x.symbols for x in state["schedule"]]
    vocab = {BOS: 0, EOS: 1}
    for symbols in utterances:
        for s in symbols:
            vocab.setdefault(s, len(vocab))
    items = probes(world, state)
    res = {"domain": domain, "seed": seed, "n": n, "probes": len(items)}
    res["zihin"] = score(lambda p: zihin_dist(mind, p), items)
    torch.manual_seed(seed)
    online = train_online(TinyGPT(len(vocab)), vocab, utterances)
    res["online"] = score(lambda p: tf_dist(online, vocab, p), items)
    offline, t_off, epochs = train_offline(vocab, utterances, seed)
    res["offline"] = score(lambda p: tf_dist(offline, vocab, p), items)
    m = max(state["moments"][0])
    t1 = time.time()
    clone = mind.clone()
    clone.S.log = mind.S.log
    for moment in reversed(state["moments"][0]):
        mind.retract(moment)
    res["retract"] = {"zihin_ms": 1000 * (time.time() - t1), "transformer_retrain_s": t_off, "offline_epochs": epochs,
                      "zihin_life_s": t_mind}
    return res


def interference(seed, n, first="nitelik", second="iliski"):
    a, b = World.make(seed, first), World.make(seed, second)
    mind, _, sa = live(a, "members", n, record=True)
    items_a = [it for it in probes(a, sa) if it[0] == "members"]
    before_z = score(lambda p: zihin_dist(mind, p), items_a)["members"]
    _, _, sb = live(b, "members", n, record=True, mind=mind)
    after_z = score(lambda p: zihin_dist(mind, p), items_a)["members"]
    ua, ub = [x.symbols for x in sa["schedule"]], [x.symbols for x in sb["schedule"]]
    vocab = {BOS: 0, EOS: 1}
    for symbols in ua + ub:
        for s in symbols:
            vocab.setdefault(s, len(vocab))
    torch.manual_seed(seed)
    model = train_online(TinyGPT(len(vocab)), vocab, ua)
    before_t = score(lambda p: tf_dist(model, vocab, p), items_a)["members"]
    model = train_online(model, vocab, ub)
    after_t = score(lambda p: tf_dist(model, vocab, p), items_a)["members"]
    return {"seed": seed, "test": "T6_interference", "zihin_before": before_z, "zihin_after": after_z,
            "online_before": before_t, "online_after": after_t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["nitelik", "iliski"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--interference", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "baseline_transformer.jsonl"))
    args = ap.parse_args()
    for seed in args.seeds:
        rows = [interference(seed, args.n)] if args.interference else [one(d, seed, args.n) for d in args.domains]
        with open(args.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
                print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
