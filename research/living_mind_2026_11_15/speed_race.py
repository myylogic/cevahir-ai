# -*- coding: utf-8 -*-
"""Speed race: the Zihin core against small neural networks on the identical stream (CPU only).

    python research/living_mind_2026_11_15/speed_race.py throughput --reps 3
    python research/living_mind_2026_11_15/speed_race.py quality --domains nitelik --seeds 0 1 2 3 4
    python research/living_mind_2026_11_15/speed_race.py latency --mind PATH/cevahir2 --label cevahir2 \
        --dialogues PATH/dialogues.json --reps 3
    python research/living_mind_2026_11_15/speed_race.py table

The stream is the one of research/living_mind_2026_11_14/baseline_transformer.py (RESULTS.md, "Against Transformer
baselines on the identical stream"): World.make(seed, domain).schedule(600), 600 experiences of 3 symbols each. Zihin
lives an experience as 3 moments (one symbol each) and one silent moment (run.utter); a network is given
"<s> a b c </s>" and trained on its 4 next-token targets (baseline_transformer.step). So one experience is 4 items
for both: 4 moments, or 4 predicted tokens. Both scripts of 2026_11_14 are imported unchanged.

Parts (every row is appended to results/speed_race.jsonl):
  throughput  learning throughput, items per second, one pass over the stream, online: Zihin (variant "members", F11,
              as in the baseline; also without its ledger) against the baseline's TinyGPT (one Adam step per
              utterance), a GRU and an LSTM of the same width; 1 and 4 torch threads. Also the networks the usual
              way: batches of 32 utterances, one epoch (median of 10). Memory per item is in the same rows.
  latency     answer latency: Zihin's Session.reply on a saved mind (per produced word end to end, the moments that
              produce a word, the first word) and its watching speed there; against next-token inference of networks
              with the mind's vocabulary size (TinyGPT without cache, a cached decoder of the same size, GRU, LSTM,
              and a decoder of GPT-2-small size), 1 and 4 threads, random weights (latency does not depend on them).
  quality     held-out accuracy against wall-clock seconds of learning, with the probes of baseline_transformer.py
              and the clock paused while probing: Zihin's life (checkpoints), the online networks' one pass, and the
              networks trained offline as in the baseline (batches of 32, epochs until the loss stops falling).
  table       medians and means of the rows, as markdown tables.
Counts, timings and digests only: nothing a private mind says or holds is printed or stored.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import statistics
import sys
import time
import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OLD = os.path.join(ROOT, "research", "living_mind_2026_11_14")
sys.path.insert(0, ROOT)
sys.path.insert(0, OLD)
warnings.filterwarnings("ignore")

import baseline_transformer as bt  # noqa: E402  (unchanged; sets torch threads to 1)
from run import CHANNEL, PROV, Probe, World, live, utter, VARIANTS  # noqa: E402
from zihin import Mind  # noqa: E402

OUT = os.path.join(HERE, "results", "speed_race.jsonl")
ITEMS_PER_EXPERIENCE = 4          # Zihin: 3 symbol moments + 1 silent moment; network: 4 next-token targets
QUESTIONS = ["merhaba", "nasılsın ?", "bugün ne yaptın ?", "kedi ne yapar ?", "adın ne ?", "ne düşünüyorsun ?",
             "bana bir şey anlat .", "okula gittin mi ?", "en sevdiğin renk ne ?", "yarın ne yapacaksın ?"]
ACC =("held_told", "held_induced", "members", "exceptions", "early")


# ------------------------------------------------------------------------------------------------ networks
class TinyRNN(nn.Module):
    """Embedding, one recurrent layer (GRU or LSTM) of width d, linear read-out: TinyGPT's recurrent counterpart."""

    def __init__(self, vocab: int, d: int = 64, kind: str = "gru") -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.rnn = (nn.GRU if kind == "gru" else nn.LSTM)(d, d, batch_first=True)
        self.out = nn.Linear(d, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.tok(ids))
        return self.out(h)

    def run(self, ids: torch.Tensor, state=None):
        h, state = self.rnn(self.tok(ids), state)
        return self.out(h[:, -1]), state


class Block(nn.Module):
    """A pre-norm decoder block (GPT-2 layout) that keeps keys and values for incremental decoding."""

    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.heads = heads

    def forward(self, x, kv=None):
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).split(d, dim=-1)
        q, k, v = (t.view(b, n, self.heads, d // self.heads).transpose(1, 2) for t in (q, k, v))
        if kv is not None:
            k, v = torch.cat([kv[0], k], 2), torch.cat([kv[1], v], 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=kv is None and n > 1)
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
        return x + self.ff(self.n2(x)), (k, v)


class CachedGPT(nn.Module):
    """A decoder with a key/value cache, for next-token latency only (its weights are never trained here)."""

    def __init__(self, vocab: int, d: int, layers: int, heads: int, ctx: int = 1024) -> None:
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm, self.out = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)

    def run(self, ids: torch.Tensor, cache=None):
        start = 0 if cache is None else cache[0][0].shape[2]
        h = self.tok(ids) + self.pos(torch.arange(start, start + ids.shape[1]))
        new = []
        for i, block in enumerate(self.blocks):
            h, kv = block(h, None if cache is None else cache[i])
            new.append(kv)
        return self.out(self.norm(h[:, -1])), new


NETS = {"transformer": lambda v: bt.TinyGPT(v), "gru": lambda v: TinyRNN(v, kind="gru"),
        "lstm": lambda v: TinyRNN(v, kind="lstm")}


def param_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------------------------------------------ helpers
def load1() -> float:
    return os.getloadavg()[0]


def rss() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) * 1024
    return 0


def emit(row: dict, out: str) -> None:
    row = dict(row, load1=round(load1(), 2), time=time.strftime("%H:%M:%S"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def stream(domain: str, seed: int, n: int):
    """The identical stream: world, schedule, utterances and the networks' vocabulary (as baseline_transformer)."""
    world = World.make(seed, domain)
    schedule = world.schedule(n)
    utterances = [x.symbols for x in schedule]
    vocab = {bt.BOS: 0, bt.EOS: 1}
    for symbols in utterances:
        for s in symbols:
            vocab.setdefault(s, len(vocab))
    return world, schedule, utterances, vocab


def stream_state(world: World, schedule) -> dict:
    """What run.live records about the stream for the probes (heard, isa, early), without living it."""
    state = {"heard": {}, "isa": {}, "early": {}, "schedule": schedule}
    for x in schedule:
        if x.kind == "isa":
            state["isa"][x.ent] = state["isa"].get(x.ent, 0) + 1
        elif x.kind == "fact":
            heard = state["heard"].setdefault((x.ent, x.dim), {"told": 0, "sensed": 0, "lie": 0})
            heard["lie" if x.tag == "lie" else x.prov] += 1
            if x.tag == "early":
                state["early"][(x.ent, x.dim)] = None
    return state


def batched_epochs(model, vocab, utterances, seed: int, epochs: int, batch: int = 32, lr: float = 3e-3) -> List[float]:
    """Seconds per epoch of the usual training (baseline_transformer.train_offline's loop)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = [bt.encode(vocab, s) for s in utterances]
    g = torch.Generator().manual_seed(seed)
    times = []
    for _ in range(epochs):
        t0 = time.perf_counter()
        order = torch.randperm(len(data), generator=g).tolist()
        for i in range(0, len(order), batch):
            bt.step(model, opt, [data[j] for j in order[i:i + batch]])
        times.append(time.perf_counter() - t0)
    return times


def warm_up(vocab, utterances) -> None:
    for kind in NETS:
        torch.manual_seed(0)
        bt.train_online(NETS[kind](len(vocab)), vocab, utterances[:30])
        batched_epochs(NETS[kind](len(vocab)), vocab, utterances[:64], 0, 1)


# ------------------------------------------------------------------------------------------------ throughput
def throughput(args) -> None:
    for rep in range(args.reps):
        for domain in args.domains:
            world, schedule, utterances, vocab = stream(domain, args.seed, args.n)
            experiences = len(utterances)
            items = ITEMS_PER_EXPERIENCE * experiences
            base = {"part": "throughput", "domain": domain, "seed": args.seed, "rep": rep, "experiences": experiences,
                    "items": items, "vocab": len(vocab)}
            for threads in args.threads:
                torch.set_num_threads(threads)
                warm_up(vocab, utterances)
                variants = [("zihin", {})] + ([("zihin_no_ledger", {"keep_log": False})] if threads == 1 else [])
                for name, over in variants:
                    gc.collect()
                    t0 = time.perf_counter()
                    mind, _, _ = live(world, "members", args.n, record=False, **over)
                    seconds = time.perf_counter() - t0
                    size = mind.size()
                    row = dict(base, system=name, threads=threads, seconds=seconds, moments=mind.t,
                               items_per_s=mind.t / seconds, experiences_per_s=experiences / seconds,
                               ms_per_item=1000 * seconds / mind.t, units=size["units"], ties=size["ties"],
                               core_bytes=size["core_tensors"], log_bytes=size["log"],
                               core_bytes_per_item=size["core_tensors"] / mind.t,
                               log_bytes_per_item=size["log"] / mind.t, digest=mind.S.digest()[:16])
                    if rep == 0 and threads == args.threads[0]:
                        whole = len(pickle.dumps(mind.clone(), protocol=pickle.HIGHEST_PROTOCOL))
                        row.update(pickled_bytes_without_ledger=whole, pickled_bytes_per_item=whole / mind.t)
                    emit(row, args.out)
                    del mind
                for kind in NETS:
                    torch.manual_seed(args.seed)
                    model = NETS[kind](len(vocab))
                    t0 = time.perf_counter()
                    bt.train_online(model, vocab, utterances)            # one Adam step per utterance, batch 1
                    seconds = time.perf_counter() - t0
                    pb = param_bytes(model)
                    emit(dict(base, system=kind + "_online", threads=threads, seconds=seconds,
                              items_per_s=items / seconds, experiences_per_s=experiences / seconds,
                              ms_per_item=1000 * seconds / items, params=n_params(model), param_bytes=pb,
                              adam_bytes=2 * pb, param_bytes_per_item=pb / items), args.out)
                    torch.manual_seed(args.seed)
                    model = NETS[kind](len(vocab))
                    times = batched_epochs(model, vocab, utterances, args.seed, args.epochs)
                    seconds = statistics.median(times)
                    emit(dict(base, system=kind + "_batch32", threads=threads, seconds=seconds,
                              items_per_s=items / seconds, experiences_per_s=experiences / seconds,
                              ms_per_item=1000 * seconds / items, params=n_params(model), param_bytes=pb,
                              note="median seconds of one epoch over %d epochs" % args.epochs), args.out)
    torch.set_num_threads(1)


# ------------------------------------------------------------------------------------------------ latency
def net_latency(model, vocab: int, cached: str, prompt: int = 8, tokens: int = 64, window: int = 12):
    """Prefill of a prompt and the time of each next-token step (greedy). cached: "none" (full forward over the
    last ``window`` tokens), "kv" (key/value cache) or "state" (recurrent state)."""
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, vocab, (1, prompt), generator=g)
    steps = []
    with torch.inference_mode():
        t0 = time.perf_counter()
        if cached == "none":
            logits = model(ids)[0, -1]
        else:
            logits, cache = model.run(ids)
            logits = logits[0]
        prefill = time.perf_counter() - t0
        seq = ids[0].tolist()
        for _ in range(tokens):
            t0 = time.perf_counter()
            nxt = int(torch.argmax(logits))
            seq.append(nxt)
            if cached == "none":
                logits = model(torch.tensor([seq[-window:]]))[0, -1]
            else:
                logits, cache = model.run(torch.tensor([[nxt]]), cache)
                logits = logits[0]
            steps.append(time.perf_counter() - t0)
    steps.sort()
    return 1000 * prefill, 1000 * statistics.median(steps), 1000 * steps[int(0.9 * len(steps))]


def latency_nets(vocab: int, label: str, args) -> None:
    specs = [("tinygpt_d64_l2_nocache_w12", lambda: bt.TinyGPT(vocab), "none", vocab),
             ("gpt_d64_l2_kvcache", lambda: CachedGPT(vocab, 64, 2, 4), "kv", vocab),
             ("gru_d64", lambda: TinyRNN(vocab, kind="gru"), "state", vocab),
             ("lstm_d64", lambda: TinyRNN(vocab, kind="lstm"), "state", vocab),
             ("gpt_d768_l12_kvcache", lambda: CachedGPT(vocab, 768, 12, 12), "kv", vocab)]
    if args.gpt2_vocab:
        specs.append(("gpt_d768_l12_kvcache_v50257", lambda: CachedGPT(50257, 768, 12, 12), "kv", 50257))
    for name, make, cached, v in specs:
        torch.manual_seed(0)
        model = make().eval()
        for threads in args.threads:
            torch.set_num_threads(threads)
            net_latency(model, v, cached, tokens=8)                  # warm-up
            for rep in range(args.reps):
                prefill, median, p90 = net_latency(model, v, cached)
                emit({"part": "latency", "system": name, "mind": label, "vocab": v, "threads": threads, "rep": rep,
                      "prefill8_ms": prefill, "token_ms_median": median, "token_ms_p90": p90,
                      "params": n_params(model), "param_bytes": param_bytes(model)}, args.out)
        del model
        gc.collect()
    torch.set_num_threads(1)


def latency_zihin(args) -> int:
    from zihin import mind as mind_mod
    from zihin.persist import load
    from zihin.__main__ import Session
    label = args.label or os.path.basename(args.mind)
    vocab = 0
    original = mind_mod.Mind.step
    for rep in range(args.reps):
        torch.set_num_threads(1)
        gc.collect()
        r0 = rss()
        t0 = time.perf_counter()
        mind = load(args.mind, ledger=False)
        t_load = time.perf_counter() - t0
        gc.collect()
        r1 = rss()
        vocab = mind.ch.sizes()[mind.ch.id(CHANNEL)]
        lived, size = mind.t, mind.size()
        record: List[tuple] = []                                    # (seconds, text words emitted) per moment

        def step(self, events=()):
            t = time.perf_counter()
            acts = original(self, events)
            record.append((time.perf_counter() - t, sum(1 for e in acts if e.channel == CHANNEL)))
            return acts

        mind_mod.Mind.step = step
        try:
            session = Session(mind)
            answers = hashlib.sha256()
            words = moments = 0
            total = 0.0
            first, per_reply = [], []
            for q in QUESTIONS:
                stamps: List[float] = []
                m0 = len(record)
                t0 = time.perf_counter()
                said = session.reply(q, say=lambda w: stamps.append(time.perf_counter()))
                seconds = time.perf_counter() - t0
                answers.update(said.encode("utf-8"))
                total += seconds
                moments += len(record) - m0
                words += len(stamps)
                per_reply.append((len(stamps), len(record) - m0, 1000 * seconds))
                if stamps:
                    first.append(1000 * (stamps[0] - t0))
            speaking = [1000 * s for s, n in record if n > 0]          # moments in which the mind said a word
            row = {"part": "latency", "system": "zihin", "mind": label, "rep": rep, "threads": 1, "vocab": vocab,
                   "replies": len(QUESTIONS), "words": words, "moments": moments, "reply_ms_total": 1000 * total,
                   "ms_per_word": 1000 * total / max(1, words), "ms_per_moment": 1000 * total / max(1, moments),
                   "first_word_ms_median": statistics.median(first) if first else None,
                   "speaking_moment_ms_median": statistics.median(speaking) if speaking else None,
                   "speaking_moments": len(speaking), "silent_replies": sum(1 for w, _, _ in per_reply if w == 0),
                   "answers_sha": answers.hexdigest()[:16], "moments_lived": lived, "units": size["units"],
                   "ties": size["ties"], "core_bytes": size["core_tensors"],
                   "core_bytes_per_moment": size["core_tensors"] / lived, "rss_delta_bytes": r1 - r0,
                   "rss_bytes_per_moment": (r1 - r0) / lived, "load_s": t_load}
            emit(row, args.out)
            if args.dialogues:                                          # learning throughput on the large mind
                with open(args.dialogues, encoding="utf-8") as f:
                    pairs = [(x["Soru"], x["Cevap"]) for x in json.load(f)[:args.watch]]
                m0 = len(record)
                t0 = time.perf_counter()
                for q, a in pairs:
                    session.watch(q, a)
                seconds = time.perf_counter() - t0
                n = len(record) - m0
                emit({"part": "watch", "system": "zihin", "mind": label, "rep": rep, "threads": 1, "vocab": vocab,
                      "dialogues": len(pairs), "moments": n, "seconds": seconds, "items_per_s": n / seconds,
                      "ms_per_item": 1000 * seconds / n, "digest": mind.S.digest()[:16]}, args.out)
        finally:
            mind_mod.Mind.step = original
        del mind, session
        gc.collect()
    return vocab


def watch_nets(vocab: int, label: str, args, length: int = 11) -> None:
    """The networks' learning step at the mind's vocabulary: one Adam step per utterance of ``length`` tokens."""
    g = torch.Generator().manual_seed(0)
    utterances = [torch.randint(2, vocab, (length - 2,), generator=g).tolist() for _ in range(60)]
    for kind in NETS:
        for threads in args.threads:
            torch.set_num_threads(threads)
            for rep in range(args.reps):
                torch.manual_seed(0)
                model = NETS[kind](vocab)
                opt = torch.optim.Adam(model.parameters(), lr=3e-3)
                bt.step(model, opt, [[0] + utterances[0] + [1]])     # warm-up
                t0 = time.perf_counter()
                for u in utterances:
                    bt.step(model, opt, [[0] + u + [1]])
                seconds = time.perf_counter() - t0
                items = len(utterances) * (length - 1)
                emit({"part": "watch", "system": kind + "_online", "mind": label, "rep": rep, "threads": threads,
                      "vocab": vocab, "utterance_tokens": length, "items": items, "seconds": seconds,
                      "items_per_s": items / seconds, "ms_per_item": 1000 * seconds / items,
                      "params": n_params(model), "param_bytes": param_bytes(model)}, args.out)
    torch.set_num_threads(1)


def latency(args) -> None:
    label = args.label or os.path.basename(args.mind)
    vocab = latency_zihin(args)
    latency_nets(vocab, label, args)
    if args.dialogues:
        watch_nets(vocab, label, args)


# ------------------------------------------------------------------------------------------------ quality
EPOCHS_PROBED = {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 120, 150,
                 200, 250, 300}


def quality(args) -> None:
    torch.set_num_threads(1)
    checkpoints = set(args.checkpoints)
    for domain in args.domains:
        for seed in args.seeds:
            world, schedule, utterances, vocab = stream(domain, seed, args.n)
            items = bt.probes(world, stream_state(world, schedule))
            base = {"part": "quality", "domain": domain, "seed": seed, "probes": len(items)}
            # ---- Zihin: one life; the clock runs only while it lives
            mind = Mind(VARIANTS["members"]())
            mind.add_channel(CHANNEL, "act")
            clock = 0.0
            for index, x in enumerate(schedule):
                t0 = time.perf_counter()
                utter(mind, x.symbols, PROV[x.prov])
                clock += time.perf_counter() - t0
                if index + 1 in checkpoints or index + 1 == len(schedule):
                    probe = Probe(mind)
                    acc = bt.score(probe.ask, items)
                    emit(dict(base, system="zihin", seconds=clock, experiences=index + 1, **acc), args.out)
            fresh = bt.score(lambda p: bt.zihin_dist(mind, p), items)     # the baseline's scoring: a clone per probe
            same = fresh == acc
            del mind
            # ---- networks online (one pass) and offline (the baseline's loop), the clock paused while probing
            for kind in NETS:
                torch.manual_seed(seed)
                model = NETS[kind](len(vocab))
                opt = torch.optim.Adam(model.parameters(), lr=3e-3)
                clock = 0.0
                for index, symbols in enumerate(utterances):
                    t0 = time.perf_counter()
                    bt.step(model, opt, [bt.encode(vocab, symbols)])
                    clock += time.perf_counter() - t0
                    if index + 1 in checkpoints or index + 1 == len(utterances):
                        acc = bt.score(lambda p: bt.tf_dist(model, vocab, p), items)
                        emit(dict(base, system=kind + "_online", seconds=clock, experiences=index + 1, **acc),
                             args.out)
                torch.manual_seed(seed)
                model = NETS[kind](len(vocab))
                opt = torch.optim.Adam(model.parameters(), lr=3e-3)
                data = [bt.encode(vocab, s) for s in utterances]
                g = torch.Generator().manual_seed(seed)
                best, stale, clock = math.inf, 0, 0.0
                for epoch in range(args.max_epochs):
                    t0 = time.perf_counter()
                    order = torch.randperm(len(data), generator=g).tolist()
                    total = 0.0
                    for i in range(0, len(order), 32):
                        total += bt.step(model, opt, [data[j] for j in order[i:i + 32]])
                    if total < best - 1e-3:
                        best, stale = total, 0
                    else:
                        stale += 1
                    clock += time.perf_counter() - t0
                    done = stale >= 10 or epoch + 1 == args.max_epochs
                    if epoch + 1 in EPOCHS_PROBED or done:
                        acc = bt.score(lambda p: bt.tf_dist(model, vocab, p), items)
                        emit(dict(base, system=kind + "_offline", seconds=clock, epochs=epoch + 1,
                                  experiences=len(utterances) * (epoch + 1), final=done, **acc), args.out)
                    if done:
                        break
            emit(dict(base, system="check", zihin_probe_equals_clone_per_probe=same), args.out)


# ------------------------------------------------------------------------------------------------ table
def _rows(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _med(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def table(args) -> None:
    rows = _rows(args.out)
    thr = [r for r in rows if r["part"] == "throughput"]
    if thr:
        print("\n### Learning throughput on the identical stream (one pass; medians over reps and both domains)\n")
        print("| system | threads | items/s | experiences/s | ms per item | reps |")
        print("|---|---|---|---|---|---|")
        keys = sorted({(r["system"], r["threads"]) for r in thr}, key=lambda k: (k[0] != "zihin", k[0], k[1]))
        for system, threads in keys:
            sel = [r for r in thr if r["system"] == system and r["threads"] == threads]
            print("| %s | %d | %.0f | %.1f | %.3f | %d |" % (system, threads, _med(r["items_per_s"] for r in sel),
                                                             _med(r["experiences_per_s"] for r in sel),
                                                             _med(r["ms_per_item"] for r in sel), len(sel)))
        print("\n### Memory per learned item on the stream (rep 0)\n")
        for r in thr:
            if r["rep"] == 0 and r["threads"] == 1:
                if r["system"].startswith("zihin"):
                    print("- %s %s: %d moments, core %d B (%.0f B/moment), ledger %d B (%.0f B/moment)%s" % (
                        r["system"], r["domain"], r["moments"], r["core_bytes"], r["core_bytes_per_item"],
                        r["log_bytes"], r["log_bytes_per_item"],
                        ", whole mind pickled without ledger %.0f B/moment" % r["pickled_bytes_per_item"]
                        if "pickled_bytes_per_item" in r else ""))
                elif r["system"].endswith("_online"):
                    print("- %s %s: %d params, %d B (%.0f B per item seen once), Adam state %d B" % (
                        r["system"], r["domain"], r["params"], r["param_bytes"], r["param_bytes_per_item"],
                        r["adam_bytes"]))
    lat = [r for r in rows if r["part"] == "latency"]
    for label in sorted({r["mind"] for r in lat}):
        sel = [r for r in lat if r["mind"] == label]
        z = [r for r in sel if r["system"] == "zihin"]
        print("\n### Answer latency, mind %s\n" % label)
        if z:
            r = z[0]
            print("Zihin: %d units, %d ties, %d moments lived, text alphabet %d; %d words in %d replies, %d moments" % (
                r["units"], r["ties"], r["moments_lived"], r["vocab"], r["words"], r["replies"], r["moments"]))
            print("- per produced word end to end %.1f ms; per moment while replying %.1f ms; moment that says a word "
                  "%.1f ms; first word %.1f ms (medians over %d reps; answers identical: %s)" % (
                      _med(x["ms_per_word"] for x in z), _med(x["ms_per_moment"] for x in z),
                      _med(x["speaking_moment_ms_median"] for x in z), _med(x["first_word_ms_median"] for x in z),
                      len(z), len({x["answers_sha"] for x in z}) == 1))
            print("- memory: core %.0f B/moment, process %.0f B/moment (RSS growth on load)" % (
                r["core_bytes_per_moment"], r["rss_bytes_per_moment"]))
        print("\n| network | vocab | params | threads | prefill 8 tokens ms | next token ms (median) |")
        print("|---|---|---|---|---|---|")
        for name, threads in sorted({(r["system"], r["threads"]) for r in sel if r["system"] != "zihin"}):
            s = [r for r in sel if r["system"] == name and r["threads"] == threads]
            print("| %s | %d | %d | %d | %.2f | %.3f |" % (name, s[0]["vocab"], s[0]["params"], threads,
                                                          _med(r["prefill8_ms"] for r in s),
                                                          _med(r["token_ms_median"] for r in s)))
    wat = [r for r in rows if r["part"] == "watch"]
    if wat:
        print("\n### Learning throughput at the vocabulary of the saved minds (medians)\n")
        print("| mind | system | threads | items/s | ms per item |")
        print("|---|---|---|---|---|")
        for label, system, threads in sorted({(r["mind"], r["system"], r["threads"]) for r in wat}):
            s = [r for r in wat if r["mind"] == label and r["system"] == system and r["threads"] == threads]
            print("| %s | %s | %d | %.0f | %.3f |" % (label, system, threads, _med(r["items_per_s"] for r in s),
                                                     _med(r["ms_per_item"] for r in s)))
    qual = [r for r in rows if r["part"] == "quality"]
    if qual:
        equal_time(qual)


def _at(curve: List[dict], seconds: float) -> Optional[dict]:
    """The last checkpoint reached within ``seconds`` of learning."""
    within = [r for r in curve if r["seconds"] <= seconds]
    return within[-1] if within else None


def equal_time(qual: List[dict]) -> None:
    systems = ["zihin", "transformer_online", "gru_online", "lstm_online", "transformer_offline", "gru_offline",
               "lstm_offline"]
    for domain in sorted({r["domain"] for r in qual}):
        lives = sorted({r["seed"] for r in qual if r["domain"] == domain})
        out: Dict[str, Dict[str, List[float]]] = {}
        reach: Dict[str, List[Optional[float]]] = {}
        for seed in lives:
            curves = {s: [r for r in qual if r["domain"] == domain and r["seed"] == seed and r["system"] == s]
                      for s in systems}
            if not curves["zihin"]:
                continue
            t_z = curves["zihin"][-1]["seconds"]
            t_o = curves["transformer_online"][-1]["seconds"] if curves["transformer_online"] else 0.0
            z_final = curves["zihin"][-1]
            held_z = (z_final["held_told"] + z_final["held_induced"]) / 2
            for s in systems:
                if not curves[s]:
                    continue
                end = curves[s][-1]
                cells = {"end of learning": end}
                if end["seconds"] > t_z:                       # still learning when Zihin's life is over
                    cells["at Zihin's time"] = _at(curves[s], t_z)
                if s == "zihin" or s.endswith("_offline"):     # the online networks have ended by then
                    cells["at TinyGPT's one-pass time"] = _at(curves[s], t_o)
                for when, r in cells.items():
                    key = "%s | %s" % (s, when)
                    d = out.setdefault(key, {})
                    d.setdefault("seconds", []).append(r["seconds"] if r else float("nan"))
                    for a in ACC:
                        d.setdefault(a, []).append(r.get(a, float("nan")) if r else float("nan"))
                if s.endswith("_offline"):
                    hit = next((r["seconds"] for r in curves[s] if (r["held_told"] + r["held_induced"]) / 2 >= held_z),
                               None)
                    reach.setdefault(s, []).append(hit)
        print("\n### Quality at equal learning time, %s (means over %d lives; accuracy = right value on top)\n" % (
            domain, len(lives)))
        print("| system | when | seconds | held-out told rule | held-out induced | members | exceptions | early |")
        print("|---|---|---|---|---|---|---|---|")
        for key, d in out.items():
            s, when = key.split(" | ")
            vals =[statistics.fmean(x for x in d[a] if x == x) if any(x == x for x in d[a]) else float("nan")
                    for a in ("seconds",) + ACC]
            print("| %s | %s | %s |" % (s, when, " | ".join("%.2f" % v if v == v else "-" for v in vals)))
        for s, hits in reach.items():
            got = [h for h in hits if h is not None]
            print("- %s was at or above Zihin's final held-out mean at some probed epoch in %d of %d lives%s "
                  "(the best of many noisy checkpoints: optimistic)" % (
                      s, len(got), len(hits), " (median %.1f s)" % statistics.median(got) if got else ""))


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["throughput", "latency", "quality", "table"])
    ap.add_argument("--domains", nargs="+", default=["nitelik", "iliski"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--epochs", type=int, default=10, help="throughput: batched epochs timed")
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[30, 60, 120, 180, 300, 450, 600])
    ap.add_argument("--mind", help="latency: path of a saved mind (loaded without its ledger, never written)")
    ap.add_argument("--label", help="latency: name of the mind in the rows")
    ap.add_argument("--dialogues", help="latency: dialogues (json list of {Soru, Cevap}) watched to time learning")
    ap.add_argument("--watch", type=int, default=20)
    ap.add_argument("--gpt2-vocab", action="store_true", help="latency: also a GPT-2-small decoder with 50257 tokens")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    {"throughput": throughput, "latency": latency, "quality": quality, "table": table}[args.part](args)


if __name__ == "__main__":
    main()
