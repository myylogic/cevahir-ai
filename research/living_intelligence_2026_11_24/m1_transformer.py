# -*- coding: utf-8 -*-
"""M1 baseline: a character-level Transformer trained from scratch on the same text (NOTES.md, section 2).

    python m1_transformer.py CORPUS --mode offline|online [--minutes 12] [--threads 4]

- ``offline``: the usual setting. Train on the first 90 % of the stream (minus its last 5 %, held out to choose the
  checkpoint) for a fixed time, then evaluate the last 10 % frozen. Comparable to the cortex's ``bpc_test_frozen``.
- ``online``: living through the stream in order. Each block of 1,024 characters is first predicted (its bits
  counted), then the model trains on everything seen so far. Steps are spread evenly over the blocks so the run takes
  the given time. Comparable to the cortex's prequential ``bpc_all`` and ``bpc_test_living``.

The model is a pre-norm GPT: 4 layers, width 192, 4 heads, context 256, dropout 0.1, AdamW with warm-up and (offline)
cosine decay. On texts under 100,000 characters it is 3 layers of width 128 with dropout 0.2. The vocabulary is
every character of the stream, given in advance (the cortex has to meet its symbols).
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
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import corpora  # noqa: E402
from living.lexicon import words  # noqa: E402

CTX = 256
MEETINGS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 100), (101, 10 ** 12)]


class Block(nn.Module):
    def __init__(self, d: int, heads: int, drop: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.heads = heads
        self.drop = drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (z.view(B, T, self.heads, C // self.heads).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0)
        x = x + F.dropout(self.proj(y.transpose(1, 2).reshape(B, T, C)), self.drop, self.training)
        return x + F.dropout(self.mlp(self.ln2(x)), self.drop, self.training)


class GPT(nn.Module):
    def __init__(self, vocab: int, d: int, layers: int, heads: int, drop: float):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(CTX, d)
        self.blocks = nn.ModuleList(Block(d, heads, drop) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.drop = drop

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        x = F.dropout(x, self.drop, self.training)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def score(model: GPT, data: torch.Tensor, start: int, end: int) -> list[float]:
    """Bits of each symbol data[start:end], each given at least CTX/2 symbols before it (or all there are)."""
    model.eval()
    out: list[float] = []
    S = CTX // 2
    with torch.no_grad():
        pos = start
        if pos == 0:                                       # the first symbol of the stream has no context
            out.append(math.log2(model.head.out_features))
            pos = 1
        while pos < end:
            hi = min(end, pos + S)
            lo = max(0, hi - 1 - CTX)
            logp = F.log_softmax(model(data[lo:hi - 1].unsqueeze(0))[0].float(), dim=-1)
            lp = (logp.gather(1, data[lo + 1:hi].unsqueeze(1)).squeeze(1) / -math.log(2)).tolist()
            out.extend(lp[pos - lo - 1:])
            pos = hi
    model.train()
    return out


def batch(data: torch.Tensor, limit: int, n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, max(1, limit - CTX - 1), (n,), generator=gen)
    x = torch.stack([data[i:i + CTX] for i in ix])
    y = torch.stack([data[i + 1:i + CTX + 1] for i in ix])
    return x, y


def word_stats(s: str, bits: list[float], split: int, offset: int) -> dict:
    """Bits of seen and unseen words in the test part, and (if bits cover the stream) by meeting."""
    seen_train: set[str] = set()
    meet_count: dict[str, int] = {}
    meet = {k: [0.0, 0] for k in MEETINGS}
    test = {"seen": [0.0, 0, 0], "unseen": [0.0, 0, 0]}
    for a, b in words(s):
        w = s[a:b]
        k = meet_count.get(w, 0) + 1
        meet_count[w] = k
        if a >= offset:
            wb = sum(bits[a - offset:b - offset])
            for lo, hi in MEETINGS:
                if lo <= k <= hi:
                    meet[(lo, hi)][0] += wb
                    meet[(lo, hi)][1] += b - a
                    break
        if b <= split:
            seen_train.add(w)
        elif a >= split:
            kind = "seen" if w in seen_train else "unseen"
            test[kind][0] += sum(bits[a - offset:b - offset])
            test[kind][1] += b - a
            test[kind][2] += 1
    out = {"test_words": {k: {"bpc": round(v[0] / v[1], 4) if v[1] else None, "chars": v[1], "words": v[2]}
                          for k, v in test.items()}}
    if offset == 0:
        out["word_meetings"] = [{"meeting": "%d" % lo if lo == hi else ("%d+" % lo if hi > 10 ** 9 else
                                                                        "%d-%d" % (lo, hi)),
                                 "bpc": round(v[0] / v[1], 4) if v[1] else None, "chars": v[1]}
                                for (lo, hi), v in meet.items()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--mode", choices=("offline", "online"), required=True)
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m1_transformer.jsonl"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    s = corpora.stream(a.corpus)
    split = int(len(s) * 0.9)
    vocab = sorted(set(s))
    index = {c: i for i, c in enumerate(vocab)}
    data = torch.tensor([index[c] for c in s], dtype=torch.long)
    small = split < 100_000
    d, layers, heads, drop = (128, 3, 4, 0.2) if small else (192, 4, 4, 0.1)
    model = GPT(len(vocab), d, layers, heads, drop)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=0.1)
    budget = a.minutes * 60.0
    warm = 100
    row = {"corpus": a.corpus, "mode": a.mode, "chars": len(s), "split": split, "params": params,
           "width": d, "layers": layers, "heads": heads, "dropout": drop, "context": CTX, "minutes": a.minutes,
           "threads": a.threads, "seed": a.seed, "vocab": len(vocab)}
    t0 = time.time()
    if a.mode == "offline":
        val_lo = int(split * 0.95)
        best = (float("inf"), None, 0)
        step = 0
        evals = []
        while time.time() - t0 < budget:
            frac = (time.time() - t0) / budget
            lr = 1e-3 * min(1.0, (step + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = batch(data, val_lo, 32, gen)
            loss = F.cross_entropy(model(x).view(-1, len(vocab)), y.view(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 200 == 0:
                vb = score(model, data, val_lo, split)
                v = sum(vb) / len(vb)
                evals.append({"step": step, "seconds": round(time.time() - t0), "val_bpc": round(v, 4)})
                if v < best[0]:
                    best = (v, {k: t.clone() for k, t in model.state_dict().items()}, step)
        vb = score(model, data, val_lo, split)
        v = sum(vb) / len(vb)
        evals.append({"step": step, "seconds": round(time.time() - t0), "val_bpc": round(v, 4)})
        if v < best[0] or best[1] is None:
            best = (v, {k: t.clone() for k, t in model.state_dict().items()}, step)
        model.load_state_dict(best[1])
        train_seconds = time.time() - t0
        t1 = time.time()
        tb = score(model, data, split, len(s))
        row.update({"steps": step, "best_step": best[2], "best_val_bpc": round(best[0], 4), "evals": evals,
                    "train_seconds": round(train_seconds), "test_seconds": round(time.time() - t1),
                    "bpc_test_frozen": round(sum(tb) / len(tb), 4)})
        row.update(word_stats(s, tb, split, split))
    else:
        B = 1024
        blocks = list(range(0, len(s), B))
        bits: list[float] = []
        step = 0
        t_score = 0.0
        t_train = 0.0
        for i, p in enumerate(blocks):
            q = min(len(s), p + B)
            ts = time.time()
            bits.extend(score(model, data, p, q))
            t_score += time.time() - ts
            if q <= CTX + 2:
                continue
            left = budget - (time.time() - t0)
            blocks_left = len(blocks) - i
            per_score = t_score / (i + 1)
            per_step = t_train / step if step else 0.05
            n_steps = max(1, int((left / blocks_left - per_score) / per_step))
            tt = time.time()
            for _ in range(n_steps):
                lr = 1e-3 * min(1.0, (step + 1) / warm)
                for g in opt.param_groups:
                    g["lr"] = lr
                x, y = batch(data, q, 16, gen)
                loss = F.cross_entropy(model(x).view(-1, len(vocab)), y.view(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1
            t_train += time.time() - tt
        steps_per_block = round(step / len(blocks), 1)
        row.update({"steps": step, "steps_per_block": steps_per_block, "block": B, "batch": 16,
                    "seconds": round(time.time() - t0),
                    "bpc_all": round(sum(bits) / len(s), 4),
                    "bpc_train_part": round(sum(bits[:split]) / split, 4),
                    "bpc_test_living": round(sum(bits[split:]) / (len(s) - split), 4)})
        row.update(word_stats(s, bits, split, 0))
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
