# -*- coding: utf-8 -*-
"""M3 baseline: a character Transformer trained from scratch on exactly the stream Cevahir lived, then asked the
same questions (NOTES.md, 8).

``train_and_answer(stream, questions, minutes, threads)``: the pre-norm GPT of m1_transformer.py (width 192, 4 layers,
4 heads, context 256, dropout 0.1; on streams under 100,000 characters 3 layers of width 128 with dropout 0.2),
trained offline on the whole stream for ``minutes`` (AdamW, warm-up, cosine decay, batches of 32 windows), every
character of the stream and of the questions in its vocabulary from the start. Each question is asked as a new line
("\\n" + question + "\\t", the way the harmony lived every dialogue) and the answer is its greedy decoding up to the
line end, at most 400 characters.
"""
from __future__ import annotations

import math
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m1_transformer import CTX, GPT  # noqa: E402


def train_and_answer(stream: str, questions, minutes: float = 10.0, threads: int = 1, seed: int = 0,
                     most: int = 400) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    vocab = sorted(set(stream) | set("".join(questions)) | {"\n", "\t"})
    index = {ch: i for i, ch in enumerate(vocab)}
    data = torch.tensor([index[ch] for ch in stream], dtype=torch.long)
    small = len(stream) < 100_000
    d, layers, drop = (128, 3, 0.2) if small else (192, 4, 0.1)
    model = GPT(len(vocab), d, layers, 4, drop)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3 if small else 2e-3, betas=(0.9, 0.95), weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed)
    budget = minutes * 60.0
    t0 = time.time()
    step = 0
    base = opt.param_groups[0]["lr"]
    model.train()
    while time.time() - t0 < budget:
        frac = (time.time() - t0) / budget
        lr = base * min(1.0, (step + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        ix = torch.randint(0, max(1, len(data) - CTX - 1), (32,), generator=gen)
        x = torch.stack([data[i:i + CTX] for i in ix])
        y = torch.stack([data[i + 1:i + CTX + 1] for i in ix])
        loss = F.cross_entropy(model(x).reshape(-1, len(vocab)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
    trained = time.time() - t0
    model.eval()
    answers = []
    t1 = time.time()
    with torch.no_grad():
        for q in questions:
            ids = [index[ch] for ch in ("\n" + q + "\t")]
            out = []
            for _ in range(most):
                x = torch.tensor([ids[-CTX:]], dtype=torch.long)
                nxt = int(model(x)[0, -1].argmax())
                if vocab[nxt] == "\n":
                    break
                out.append(vocab[nxt])
                ids.append(nxt)
            answers.append("".join(out).strip())
    return {"answers": answers, "params": params, "steps": step, "train_s": round(trained, 1),
            "answer_s": round(time.time() - t1, 1), "final_loss": round(float(loss), 4), "chars": len(stream)}
