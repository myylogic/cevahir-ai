# -*- coding: utf-8 -*-
"""Recall of what was read once: a sentence of the documents, its first word(s) brought to mind, and what comes next.

    python research/living_mind_2026_11_15/recall_once.py mind --data private_data --load private_data/duel/docs_only \
        --cue 2 --set starts=1,exclusive=1
    python research/living_mind_2026_11_15/recall_once.py mind --data private_data --build private_data/duel/docs_only
    python research/living_mind_2026_11_15/recall_once.py teacher --data private_data --load private_data/duel/teacher_ends
    python research/living_mind_2026_11_15/recall_once.py transformer --data private_data --cue 2 \
        --models private_data/duel/models/word-1m-epochs private_data/duel/adv/main-d256-lr0.004

No template and no dialogue: ``mind`` uses a mind that lived only the documents (``--build`` raises one with the
speaking configuration of ``zihin.__main__``: every sentence heard, its end perceived, a pause), samples sentences of the
same documents (``random.Random(0)``, ``--n`` of them, at least ``--cue`` + 3 words) and, for each, brings its first
``--cue`` words to mind with the inner voice (``zihin.dusunce.think``: the words are imagined, the mind decides from
the last one on, each decision lived as imagined) and compares what it thinks with the rest of the sentence.

- exact: the rest of that very sentence;
- valid: the rest of some sentence of the documents that begins with the same words (a mind that keeps no order of
  reading cannot know which of several such sentences was sampled);
- by_starts: exact counts split by how many sentences begin with the cue (1, 2-3, 4 or more).

``transformer`` asks the duel's saved word-unit models the same sentences with the same cue: a new document line
begins with the cue words (the models were trained on document lines ending in a line-end unit) and greedy decoding
runs to the line end. The models use the mind's units exactly. A model saved by the duel's ``train`` is the repository
model; one saved with ``"model": "gpt"`` in its meta is the minimal GPT recipe below (RoPE, RMSNorm, SwiGLU, tied
embeddings), with which the independent verifier of the duel trained its strongest model.

``teacher`` asks the held-out subjects of ``ogretmen.tsv.heldout`` in the form the teacher pairs taught,
"<X> nedir ?" (the file holds the subject alone, and the evaluations of ``f25_teacher.py`` and ``same_data_duel.py``
asked the bare word, a form the mind never lived), on a mind raised on the pairs and the documents.

Prints aggregates and appends them to ``results/recall_once.jsonl``; no sentence, cue or answer is written anywhere.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import torch  # noqa: E402

RESULTS = os.path.join(HERE, "results", "recall_once.jsonl")


def documents(data: str, cue: int, session=None):
    """Sentences of the documents as the mind hears them (``Session.body``): those of at least cue + 3 words (the
    sample) and every one longer than the cue (what a valid answer may be)."""
    from zihin.__main__ import read_document, sentences, words, FINAL
    folder = os.path.join(data, "belgeler")
    paths = sorted(os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs)
    sample, every = [], []
    for p in paths:
        try:
            text = read_document(p)
        except Exception:                                            # noqa: BLE001 (not a text file)
            continue
        for sent in sentences(text):
            if session is not None:
                body = session.body(sent)
            else:
                body = words(sent)
                while body and body[-1] in FINAL:
                    body = body[:-1]
            if len(body) > cue:
                every.append(body)
            if len(body) >= cue + 3:
                sample.append(body)
    return sample, every


def f1(o, r) -> float:
    if not o or not r:
        return 0.0
    c = sum((collections.Counter(o) & collections.Counter(r)).values())
    if c == 0:
        return 0.0
    p, q = c / len(o), c / len(r)
    return 2 * p * q / (p + q)


class Score:
    def __init__(self, sample, every, cue: int) -> None:
        self.cue = cue
        self.starts = collections.Counter(tuple(b[:cue]) for b in sample)
        self.rests = collections.defaultdict(set)
        for b in every:
            self.rests[tuple(b[:cue])].add(tuple(b[cue:]))
        self.res = collections.defaultdict(float)
        self.by = collections.defaultdict(lambda: [0, 0])
        self.n = 0

    def add(self, body, said) -> None:
        cue, ref = tuple(body[:self.cue]), body[self.cue:]
        k = 0
        while k < min(len(said), len(ref)) and said[k] == ref[k]:
            k += 1
        self.n += 1
        self.res["exact"] += said == ref
        self.res["valid"] += tuple(said) in self.rests[cue]
        self.res["silent"] += not said
        self.res["prefix"] += k / len(ref)
        self.res["f1"] += f1(said, ref)
        self.res["len"] += len(said)
        group = "1" if self.starts[cue] == 1 else ("2-3" if self.starts[cue] <= 3 else "4+")
        self.by[group][0] += 1
        self.by[group][1] += said == ref

    def row(self) -> dict:
        out = {k: (round(v / self.n, 4) if k in ("prefix", "f1", "len") else int(v)) for k, v in self.res.items()}
        out["by_starts"] = {g: "%d/%d" % (e, m) for g, (m, e) in sorted(self.by.items())}
        out["n"] = self.n
        return out


def append(row: dict, path: str = RESULTS) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def switches(text: str) -> dict:
    out = {}
    for kv in filter(None, (text or "").split(",")):
        k, v = kv.split("=")
        out[k] = bool(int(v))
    return out


def raise_on_documents(data: str, path: str):
    """A mind of the speaking configuration that lived only the documents (no dialogue, no lesson), saved at path."""
    import zihin.__main__ as chat
    from zihin import Mind
    from zihin.persist import save
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = chat.VARIANTS["speaking"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    mind = Mind(cfg)
    mind.add_channel(chat.CHANNEL, "act")
    session = chat.Session(mind)
    t0 = time.time()
    print(session.read_path(os.path.join(data, "belgeler")), flush=True)
    save(mind, path)
    return mind, round(time.time() - t0, 1)


def run_mind(args) -> None:
    from zihin.__main__ import Session
    from zihin.dusunce import think
    from zihin.persist import load
    torch.set_num_threads(1)
    built = None
    if args.build:
        mind, built = raise_on_documents(args.data, args.build)
    else:
        mind = load(args.load, ledger=False)
    for k, v in switches(args.set).items():
        setattr(mind.cfg, k, v)
    session = Session(mind)
    sample, every = documents(args.data, args.cue, session)
    score = Score(sample, every, args.cue)
    t0 = time.time()
    for body in random.Random(0).sample(sample, args.n):
        thought, _ = think(session, " ".join(body[:args.cue]), limit=60)
        score.add(body, thought)
    row = {"test": "recall_once", "who": "zihin", "cue": args.cue, "switches": switches(args.set),
           "moments": mind.t, "ties": mind.S.n_ties, "built_s": built, "sentences": len(sample),
           "answer_s": round(time.time() - t0, 1)}
    row.update(score.row())
    append(row, args.out)


class Block(torch.nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        nn = torch.nn
        self.h = h
        self.n1, self.n2 = nn.RMSNorm(d, eps=1e-6), nn.RMSNorm(d, eps=1e-6)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.w12 = nn.Linear(d, 2 * ff, bias=False)
        self.w3 = nn.Linear(ff, d, bias=False)
        self.dp = nn.Dropout(0.0)

    def forward(self, x, cos, sin, kv=None):
        F = torch.nn.functional
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        if kv is not None:
            k, v = torch.cat([kv[0], k], 2), torch.cat([kv[1], v], 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=kv is None)
        x = x + self.dp(self.o(y.transpose(1, 2).reshape(B, T, C)))
        a, b = self.w12(self.n2(x)).chunk(2, -1)
        return x + self.dp(self.w3(F.silu(a) * b)), (k, v)


def rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class GPT(torch.nn.Module):
    """The minimal decoder-only recipe of the duel's verifier: RoPE, RMSNorm, SwiGLU, tied embeddings, causal."""

    def __init__(self, vocab, d, layers, heads, ctx=256):
        super().__init__()
        nn = torch.nn
        ff = 32 * math.ceil(8 * d / 3 / 32)
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, heads, ff) for _ in range(layers))
        self.norm = nn.RMSNorm(d, eps=1e-6)
        self.dp = nn.Dropout(0.0)
        hd = d // heads
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        ang = torch.arange(ctx).float()[:, None] * inv[None]
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def forward(self, x, cache=None):
        T = x.shape[1]
        pos = 0 if cache is None else cache[0][0].shape[2]
        cos, sin = self.cos[pos:pos + T], self.sin[pos:pos + T]
        h = self.dp(self.emb(x))
        new = []
        for i, blk in enumerate(self.blocks):
            h, kv = blk(h, cos, sin, None if cache is None else cache[i])
            new.append(kv)
        return torch.nn.functional.linear(self.norm(h), self.emb.weight), new


@torch.no_grad()
def decode(model, prompt, stop, gpt: bool, max_new: int = 60):
    import same_data_duel as D
    if not gpt:
        return D.greedy(model, prompt, stop, max_new)
    x, cache, out = torch.tensor([prompt]), None, []
    for _ in range(max_new):
        lg, cache = model(x if cache is None else x[:, -1:], cache)
        nxt = int(lg[0, -1].float().argmax())
        if nxt == stop:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]])], 1)
    return out


def run_transformer(args) -> None:
    import same_data_duel as D
    torch.set_num_threads(args.threads)
    units = D.load_units(args.data, "word")
    sample, every = documents(args.data, args.cue)
    chosen = random.Random(0).sample(sample, args.n)
    for name in args.models:
        ck = torch.load(os.path.join(name, "model.pt"))
        meta = ck.get("meta") or json.load(open(os.path.join(name, "train.json")))
        gpt = meta.get("model") == "gpt"
        model = GPT(units.size, meta["d"], meta["layers"], meta["heads"]) if gpt else \
            D.build_model(units.size, meta["d"], meta["layers"], meta["heads"], 0.0)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        score = Score(sample, every, args.cue)
        t0 = time.time()
        for body in chosen:
            ids = decode(model, [units.nl_id] + units.ids(body[:args.cue]), units.nl_id, gpt)
            score.add(body, [units.itos[i] for i in ids if i >= len(units.SPECIAL)])
        row = {"test": "recall_once", "who": "transformer", "model": os.path.basename(name.rstrip("/")),
               "recipe": "gpt" if gpt else "repository", "cue": args.cue, "params": meta.get("params"),
               "best_epoch": meta.get("best_epoch"), "train_s": meta.get("train_s"), "sentences": len(sample),
               "answer_s": round(time.time() - t0, 1)}
        row.update(score.row())
        append(row, args.out)


def run_teacher(args) -> None:
    from zihin.__main__ import CHANNEL, Session, words
    from zihin.persist import load
    torch.set_num_threads(1)
    mind = load(args.load, ledger=False)
    for k, v in switches(args.set).items():
        setattr(mind.cfg, k, v)
    session = Session(mind)
    with open(os.path.join(args.data, "ogretmen.tsv.heldout"), encoding="utf-8") as f:
        pairs = [line.rstrip("\n").split("\t", 1) for line in f if "\t" in line]
    random.Random(0).shuffle(pairs)
    pairs = pairs[:args.n]
    res = collections.defaultdict(float)
    t0 = time.time()
    for subject, answer in pairs:
        said = session.reply(subject if args.bare else "%s nedir ?" % subject)
        o = words(said) if said != "(sessiz)" else []
        r = words(answer)
        body = r[:-1] if r and r[-1] in ".!?" else r
        k = 0
        while k < min(len(o), len(body)) and o[k] == body[k]:
            k += 1
        res["exact"] += o == body
        res["starts_with_subject"] += bool(o) and o[0] == r[0]
        res["silent"] += not o
        res["f1"] += f1(o, body)
        res["prefix"] += k / len(body)
        res["len"] += len(o)
        res["overrun"] += k == len(body) and len(o) > len(body)
    n = len(pairs)
    row = {"test": "teacher_taught_form" if not args.bare else "teacher_bare_word", "who": "zihin",
           "switches": switches(args.set), "moments": mind.t, "ties": mind.S.n_ties, "n": n,
           "answer_s": round(time.time() - t0, 1)}
    row.update({k: (round(v / n, 4) if k in ("f1", "prefix", "len") else int(v)) for k, v in res.items()})
    append(row, args.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["mind", "transformer", "teacher"])
    ap.add_argument("--data", default="private_data")
    ap.add_argument("--load", default=None)
    ap.add_argument("--build", default=None, help="raise a documents-only mind and save it here")
    ap.add_argument("--set", default="", help="reading switches for a loaded mind, e.g. starts=1,exclusive=1")
    ap.add_argument("--cue", type=int, default=2)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--bare", action="store_true", help="teacher: ask the bare word, as the earlier evaluations did")
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args()
    {"mind": run_mind, "transformer": run_transformer, "teacher": run_teacher}[args.command](args)


if __name__ == "__main__":
    main()
