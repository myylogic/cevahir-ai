# -*- coding: utf-8 -*-
"""The same data duel: Transformers trained from scratch on exactly what the fresh mind of ``f25_teacher.py
--documents`` lived, asked the same 200 held-out questions in the same order, scored with the same function.

    python research/living_mind_2026_11_15/same_data_duel.py zihin  --data PRIVATE
    python research/living_mind_2026_11_15/same_data_duel.py bpe    --data PRIVATE
    python research/living_mind_2026_11_15/same_data_duel.py train  --data PRIVATE --units bpe --size 1m --exposure once
    python research/living_mind_2026_11_15/same_data_duel.py answer --data PRIVATE --name bpe-1m-once
    python research/living_mind_2026_11_15/same_data_duel.py plan   --data PRIVATE     (the Transformer runs below)
    python research/living_mind_2026_11_15/same_data_duel.py table
    python research/living_mind_2026_11_15/same_data_duel.py rescore --data PRIVATE    (the saved answers again)

``zihin``: a fresh speaking mind (the protocol of ``f25_teacher.run``, setting off, with the documents) lives
``ogretmen.tsv``, ``lessons_about_itself(1)`` and ``belgeler/``; every utterance that reaches it goes through
``Session.listen``, which is wrapped to record (role, text) without changing what is lived. The record is the corpus
(private, ``PRIVATE/duel/corpus.jsonl``). The mind then answers the 200 held-out questions (seed 0, first 200) with
``Session.reply``, then the 14 self questions of ``zihin.kendilik.LESSONS`` and 12 questions about words nobody knows
(fresh syllable words, seed 1); peak RSS and per-answer latency on 1 thread are measured.

``bpe``: the repository's byte-level BPE (``tokenizer_management/byte_bpe.py``) trained on the corpus only.

``train``: the repository's Transformer (``src/neural_network.py``, ``CevahirNeuralNetwork``: RoPE, RMSNorm, SwiGLU,
tied embeddings, causal) trained from scratch on the stream. Exchanges are "<soru> question <cevap> answer <son>",
documents plain text, one sentence per line, packed into windows. Units: ``bpe`` (the raw utterance text, repository
BPE) or ``word`` (the units the mind perceives: ``Session.body`` of each utterance, plus its tone for exchanges).
Exposure ``once``: the stream in lived order, each item once, a step per small batch, no validation (as the mind
lives it). Exposure ``epochs``: 10% of the teacher pairs held out for validation (the model sees 90% of the pairs),
many shuffled epochs, the checkpoint chosen by validation token F1 of greedy answers, stopped when neither that F1
nor the validation answer loss has improved for a few evaluations.

``answer``: greedy decoding after "<soru> question <cevap>" until <son> or 60 tokens, 1 thread; the decoded text is
turned into the mind's units (``words``) and scored by ``f25_teacher.score``.

Fixed choices (none made on the asked questions): context 128 units; AdamW (0.9, 0.95), gradient clip 1, warm-up then
cosine to 10%; once: batches of 8 sequences, fp32, no dropout, the learning rate chosen among 1e-3, 2e-3 and 4e-3 by the
validation answer loss of the 1m BPE model (``plan``) and used by every one-pass model (half for 16m); epochs: batches
of 32 (BPE) or 16 (word) sequences, bf16 autocast, dropout 0.1, weight decay 0.1, learning rate 2e-3 (1m), 1e-3 (4m),
6e-4 (16m), validation every second epoch. Peak RSS is the process peak (Python, torch and zihin take about 515 MB
before anything is built).

Only aggregates and per-question score vectors are written (``results/same_data_duel.jsonl``); questions, answers,
document words and generated answers stay under ``PRIVATE/duel/`` (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import resource
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import torch  # noqa: E402

RESULTS = os.path.join(HERE, "results", "same_data_duel.jsonl")
QUESTIONS = 200
SOR, CEV, SON, NL, UNK = "<soru>", "<cevap>", "<son>", "<nl>", "<unk>"
MAX_NEW = 60


# ----------------------------------------------------------------------------------------------- measuring
def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def private_dir(data: str) -> str:
    out = os.path.join(data, "duel")
    os.makedirs(out, exist_ok=True)
    return out


def held_out(data: str, n: int = QUESTIONS):
    """The asked questions: ``ogretmen.tsv.heldout`` shuffled with random.Random(0), the first n (as f25_teacher)."""
    with open(os.path.join(data, "ogretmen.tsv.heldout"), encoding="utf-8") as f:
        pairs = [line.rstrip("\n").split("\t", 1) for line in f if "\t" in line]
    random.Random(0).shuffle(pairs)
    return pairs[:n]


def unknown_word_questions(n: int = 12):
    """Questions about words nobody knows, made like the lessons (zihin.kendilik) but with seed 1 and never one of
    the lived words; the taught answer starts with "bilmiyorum"."""
    from zihin.kendilik import not_knowing_lessons
    lived = {q for q, _ in not_knowing_lessons()}
    lived_words = {q.split()[0] if not q.startswith("sence") else q.split()[1] for q in lived}
    out = []
    for q, a in not_knowing_lessons(60, seed=1):
        word = q.split()[1] if q.startswith("sence") else q.split()[0]
        if word not in lived_words and len(out) < n:
            out.append((q, a))
    return out


def aggregate(per):
    n = len(per)
    out = {"questions": n}
    for key in ("exact", "f1", "bilmiyorum", "silent", "asked", "length"):
        out[key] = sum(p[key] for p in per) / n
    out["per_question"] = [[p["exact"], round(p["f1"], 4), p["bilmiyorum"], p["silent"], p["asked"], p["length"]]
                           for p in per]
    return out


def latency_summary(ms):
    ms = sorted(ms)
    return {"mean_ms": round(sum(ms) / len(ms), 2), "median_ms": round(statistics.median(ms), 2),
            "p95_ms": round(ms[int(0.95 * (len(ms) - 1))], 2), "max_ms": round(ms[-1], 2)}


def append_row(row, path=RESULTS):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ----------------------------------------------------------------------------------------------- the mind
class Recorder:
    """Records every utterance that reaches the mind through ``Session.listen`` (and which exchange it belongs to)
    while the session lives exactly as before: the wrapped methods are called unchanged."""

    def __init__(self, session) -> None:
        self.session, self.items, self.phase, self._pair = session, [], None, None
        listen, watch, lesson = session.listen, session.watch, session.not_knowing_lesson

        def rec_listen(text, prov, speaker=None):
            t = session.mind.t
            listen(text, prov, speaker=speaker)
            entry = {"text": text, "prov": int(prov), "speaker": speaker, "word_moments": session.mind.t - t}
            if self._pair is not None:
                self._pair.append(entry)
            else:
                self.items.append(dict(entry, kind="doc", phase=self.phase, t=t))

        def rec_exchange(method, kind):
            def run(first, answer, *args, **kwargs):
                t, self._pair = session.mind.t, []
                try:
                    method(first, answer, *args, **kwargs)
                finally:
                    parts, self._pair = self._pair, None
                self.items.append({"kind": kind, "phase": self.phase, "t": t, "moments": session.mind.t - t,
                                   "q": parts[0]["text"], "a": parts[1]["text"] if len(parts) > 1 else None,
                                   "word_moments": sum(p["word_moments"] for p in parts),
                                   "prov": parts[0]["prov"]})
            return run

        session.listen = rec_listen
        session.watch = rec_exchange(watch, "pair")
        session.not_knowing_lesson = rec_exchange(lesson, "unknown")

    def stop(self) -> None:
        """Back to the session's own methods (the answers are given exactly as in f25_teacher)."""
        for name in ("listen", "watch", "not_knowing_lesson"):
            self.session.__dict__.pop(name, None)


def moment_account(items, session) -> dict:
    """Where the lived moments come from, recomputed from the recorded texts (Session.body / end_of_turn / steps)."""
    body = session.body
    acc = {"word_moments": 0, "turn_ends": 0, "silent_steps": 0}
    for it in items:
        if it["kind"] == "doc":
            acc["word_moments"] += len(body(it["text"]))
            acc["silent_steps"] += 1
        elif it["kind"] == "pair":
            acc["word_moments"] += len(body(it["q"])) + len(body(it["a"]))
            acc["turn_ends"] += 2
            acc["silent_steps"] += 1
        else:                                            # a lesson in not knowing (Session.not_knowing_lesson)
            acc["word_moments"] += len(body(it["q"])) + (len(body(it["a"])) if it["a"] is not None else 0)
            acc["turn_ends"] += 1 + (it["a"] is not None)
            acc["silent_steps"] += 3
    acc["total"] = sum(acc.values())
    return acc


def run_zihin(args) -> None:
    from f25_teacher import SPEAKING, score
    from zihin import Config, Mind
    from zihin.__main__ import CHANNEL, Session
    from zihin.kendilik import LESSONS
    torch.set_num_threads(1)
    out_dir = private_dir(args.data)
    rss0 = rss_mb()
    mind = Mind(Config.changed(keep_log=False, **SPEAKING))            # f25_teacher.run, setting "off"
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    rec = Recorder(session)
    t0 = time.time()
    rec.phase = "teacher"
    session.read_path(os.path.join(args.data, "ogretmen.tsv"))
    rec.phase = "lessons"
    session.lessons_about_itself(1)
    lessons_end = mind.t
    rec.phase = "documents"
    session.read_path(args.documents or os.path.join(args.data, "belgeler"))
    watched, watch_s = mind.t, time.time() - t0
    rec.stop()
    peak_watch = peak_rss_mb()
    size = mind.size()
    kinds = {}
    for it in rec.items:
        kinds[it["kind"] + ":" + it["phase"]] = kinds.get(it["kind"] + ":" + it["phase"], 0) + 1
    with open(os.path.join(out_dir, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for it in rec.items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    account = moment_account(rec.items, session)
    print({"watched_moments": watched, "after_lessons": lessons_end, "watch_s": round(watch_s, 1),
           "peak_rss_mb": round(peak_watch), "account": account, "ties": mind.S.n_ties}, flush=True)
    answers, per, ms = [], [], []
    t1 = time.time()
    for question, answer in held_out(args.data):
        a = time.perf_counter()
        said = session.reply(question)
        ms.append(1000 * (time.perf_counter() - a))
        per.append(score(said, answer))
        answers.append(said)
    answer_s = time.time() - t1
    probe, probe_out = [], []
    for question, answer in LESSONS:
        said = session.reply(question)
        probe.append(score(said, answer))
        probe_out.append(said)
    unknown, unknown_out = [], []
    for question, answer in unknown_word_questions():
        said = session.reply(question)
        unknown.append(score(said, answer))
        unknown_out.append(said)
    with open(os.path.join(out_dir, "answers_zihin.json"), "w", encoding="utf-8") as f:
        json.dump({"test": answers, "self": probe_out, "unknown": unknown_out}, f, ensure_ascii=False)
    row = {"model": "zihin", "units": "zihin words", "exposure": "lived once (no training phase)",
           "watched_moments": watched, "moments_after_lessons": lessons_end, "moment_account": account,
           "corpus_items": kinds, "train_s": round(watch_s, 1), "answer_s": round(answer_s, 1),
           "ties": mind.S.n_ties, "units_count": mind.S.n_units, "core_tensor_bytes": size.get("core_tensors"),
           "rss_mb_before": round(rss0), "peak_rss_mb_living": round(peak_watch),
           "peak_rss_mb_answering": round(peak_rss_mb()), "rss_mb_end": round(rss_mb()),
           "latency_1_thread": latency_summary(ms), "threads": 1, "moments_end": mind.t}
    row.update(aggregate(per))
    row["self_probe"] = {"asked": len(probe), "as_taught": sum(p["exact"] for p in probe),
                         "f1": round(sum(p["f1"] for p in probe) / len(probe), 4),
                         "per_question": [p["exact"] for p in probe]}
    row["unknown_probe"] = {"asked": len(unknown), "bilmiyorum": sum(p["bilmiyorum"] for p in unknown),
                            "per_question": [p["bilmiyorum"] for p in unknown]}
    append_row(row, args.out)
    print({k: v for k, v in row.items() if k != "per_question"}, flush=True)


# ----------------------------------------------------------------------------------------------- the stream
def load_corpus(data: str):
    with open(os.path.join(private_dir(data), "corpus.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def validation_indices(items, share: float = 0.1, seed: int = 1234) -> set:
    """About 10% of the watched teacher pairs (never the asked questions), fixed by a seed."""
    teacher = [i for i, it in enumerate(items) if it["kind"] == "pair" and it["phase"] == "teacher"]
    return set(random.Random(seed).sample(teacher, max(1, round(share * len(teacher)))))


def body(text: str):
    """``Session.body`` of a fresh mind: the words of an utterance without its closing punctuation."""
    from zihin.__main__ import FINAL, words
    symbols = words(text)
    while symbols and symbols[-1] in FINAL:
        symbols = symbols[:-1]
    return symbols


def corpus_texts(items):
    for it in items:
        if it["kind"] == "doc":
            yield it["text"]
        else:
            yield it["q"]
            if it["a"] is not None:
                yield it["a"]


class BPEUnits:
    """The repository's byte-level BPE on the raw utterance text, plus three markers after its vocabulary."""
    kind = "bpe"

    def __init__(self, path: str) -> None:
        from tokenizer_management.byte_bpe import ByteLevelBPE
        self.path = path
        self.bpe = ByteLevelBPE.load(path)
        v = self.bpe.vocab_size
        self.sor, self.cev, self.son, self.size = v, v + 1, v + 2, v + 3
        self.nl = self.bpe.encode("\n")
        self.pad = 0

    def prompt(self, q: str):
        return [self.sor] + self.bpe.encode(q) + [self.cev]

    def exchange(self, q: str, a):
        return self.prompt(q) + (self.bpe.encode(a) if a is not None else []) + [self.son]

    def document(self, text: str):
        return self.bpe.encode(text) + self.nl

    def is_line_end(self, i: int) -> bool:
        return [i] == self.nl

    def text(self, ids) -> str:
        return self.bpe.decode([i for i in ids if i < self.bpe.vocab_size], errors="replace")


class WordUnits:
    """The units the mind perceives: ``Session.body`` of every utterance, with its tone for an exchange
    (end_of_turn perceives "?", "!" or "."), and a line end for a document sentence (the silent moment after it)."""
    kind = "word"
    SPECIAL = (SOR, CEV, SON, NL, UNK)

    def __init__(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            vocab = json.load(f)
        self.path = path
        self.itos = list(self.SPECIAL) + vocab
        self.stoi = {u: i for i, u in enumerate(self.itos)}
        self.sor, self.cev, self.son, self.nl_id, self.unk = range(5)
        self.size = len(self.itos)
        self.pad = self.unk

    @staticmethod
    def said(text: str):
        from zihin.__main__ import Session
        return body(text) + [Session.tone(text)]

    @classmethod
    def build_vocab(cls, items):
        counts = {}
        for it in items:
            seq = body(it["text"]) if it["kind"] == "doc" else cls.said(it["q"]) + (
                cls.said(it["a"]) if it["a"] is not None else [])
            for u in seq:
                counts[u] = counts.get(u, 0) + 1
        return sorted(counts, key=lambda u: (-counts[u], u))

    def ids(self, units):
        return [self.stoi.get(u, self.unk) for u in units]

    def prompt(self, q: str):
        return [self.sor] + self.ids(self.said(q)) + [self.cev]

    def exchange(self, q: str, a):
        return self.prompt(q) + (self.ids(self.said(a)) if a is not None else []) + [self.son]

    def document(self, text: str):
        return self.ids(body(text)) + [self.nl_id]

    def is_line_end(self, i: int) -> bool:
        return i == self.nl_id

    def text(self, ids) -> str:
        return " ".join(self.itos[i] for i in ids if i >= len(self.SPECIAL))


def load_units(data: str, kind: str):
    d = private_dir(data)
    return BPEUnits(os.path.join(d, "bpe4000_corpus.json")) if kind == "bpe" else WordUnits(
        os.path.join(d, "word_vocab.json"))


def sequences(items, units, ctx: int, skip=frozenset()):
    """Training sequences in lived order: each exchange alone (it starts at position 0, as an asked question
    does); consecutive document sentences packed into windows of at most ``ctx`` units."""
    seqs, buf = [], []
    for i, it in enumerate(items):
        if i in skip:
            continue
        if it["kind"] == "doc":
            ids = units.document(it["text"])
            while len(ids) > ctx:
                if buf:
                    seqs.append(buf)
                    buf = []
                seqs.append(ids[:ctx])
                ids = ids[ctx:]
            if len(buf) + len(ids) > ctx:
                seqs.append(buf)
                buf = []
            buf = buf + ids
        else:
            if buf:
                seqs.append(buf)
                buf = []
            seqs.append(units.exchange(it["q"], it["a"])[:ctx])
    if buf:
        seqs.append(buf)
    return [s for s in seqs if len(s) >= 2]


# ----------------------------------------------------------------------------------------------- the model
SIZES = {("bpe", "1m"): dict(d=128, layers=3, heads=4), ("bpe", "4m"): dict(d=256, layers=4, heads=4),
         ("bpe", "16m"): dict(d=384, layers=8, heads=6),
         ("word", "1m"): dict(d=56, layers=2, heads=2), ("word", "4m"): dict(d=160, layers=4, heads=4),
         ("word", "16m"): dict(d=384, layers=6, heads=6)}


def build_model(vocab: int, d: int, layers: int, heads: int, dropout: float):
    from src.neural_network import CevahirNeuralNetwork
    return CevahirNeuralNetwork(learning_rate=1e-3, dropout=dropout, vocab_size=vocab, embed_dim=d, seq_proj_dim=d,
                                num_heads=heads, num_layers=layers, use_gradient_checkpointing=False,
                                use_kv_cache=True, pe_max_len=512, rope_original_max_len=512,
                                log_level=logging.WARNING)


def batch_tensors(seqs, pad: int, multiple: int = 1):
    """Right-padded inputs and targets (-100 where there is nothing to predict); the width is rounded up to a
    multiple so that a bf16 run meets few shapes (each new shape costs a kernel and memory)."""
    width = max(len(s) for s in seqs)
    width = 1 + multiple * math.ceil((width - 1) / multiple)
    x = torch.full((len(seqs), width - 1), pad, dtype=torch.long)
    y = torch.full((len(seqs), width - 1), -100, dtype=torch.long)
    for i, s in enumerate(seqs):
        t = torch.tensor(s)
        x[i, :len(s) - 1] = t[:-1]
        y[i, :len(s) - 1] = t[1:]
    return x, y


def loss_of(model, x, y, bf16: bool):
    if bf16:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits, _ = model(x)
    else:
        logits, _ = model(x)
    return torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1),
                                             ignore_index=-100)


@torch.no_grad()
def greedy(model, prompt, stop: int, max_new: int = MAX_NEW):
    """Greedy decoding with the model's own KV cache, until ``stop`` or ``max_new`` units."""
    model.clear_kv_cache()
    logits = model(torch.tensor([prompt]), use_cache=True)[0]
    out = []
    for _ in range(max_new):
        nxt = int(logits[0, -1].argmax())
        if nxt == stop:
            break
        out.append(nxt)
        logits = model(torch.tensor([[nxt]]), use_cache=True)[0]
    model.clear_kv_cache()
    return out


def as_said(units, ids) -> str:
    """What the model said, in the mind's units (``words``), as the string the score function reads."""
    from zihin.__main__ import words
    said = " ".join(words(units.text(ids)))
    return said if said else "(sessiz)"


def cut_at_line_end(units, ids):
    for k, i in enumerate(ids):
        if units.is_line_end(i):
            return ids[:k]
    return ids


@torch.no_grad()
def answer_loss(model, units, pairs) -> float:
    """Teacher-forced cross entropy of the answer units (and <son>) given the question (validation)."""
    total, n = 0.0, 0
    for q, a in pairs:
        p = units.prompt(q)
        s = units.exchange(q, a)
        x = torch.tensor([s[:-1]])
        logits = model(x)[0][0]
        lp = torch.log_softmax(logits.float(), -1)
        for k in range(len(p) - 1, len(s) - 1):
            total -= float(lp[k, s[k + 1]])
            n += 1
    return total / max(1, n)


def validate(model, units, pairs):
    from f25_teacher import score
    model.eval()
    per = [score(as_said(units, greedy(model, units.prompt(q), units.son)), a) for q, a in pairs]
    loss = answer_loss(model, units, pairs)
    model.train()
    return {"exact": sum(p["exact"] for p in per) / len(per), "f1": sum(p["f1"] for p in per) / len(per),
            "loss": loss}


def run_train(args) -> None:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rss0 = rss_mb()
    items = load_corpus(args.data)
    units = load_units(args.data, args.units)
    val = validation_indices(items) if (args.exposure == "epochs" or args.validate) else set()
    val_pairs = [(items[i]["q"], items[i]["a"]) for i in sorted(val)]
    seqs = sequences(items, units, args.ctx, skip=val)
    cfg = dict(SIZES[(args.units, args.size)])
    dropout = args.dropout if args.dropout is not None else (0.0 if args.exposure == "once" else 0.1)
    model = build_model(units.size, cfg["d"], cfg["layers"], cfg["heads"], dropout)
    params = sum(p.numel() for p in model.parameters())
    emb = model.embedding.embedding.weight.numel()
    lr = args.lr or (2e-3 if args.exposure == "once" else {"1m": 2e-3, "4m": 1e-3, "16m": 6e-4}[args.size])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0 if args.exposure ==
                            "once" else 0.1)
    name = args.name or "%s-%s-%s%s" % (args.units, args.size, args.exposure, "-val" if args.validate else "")
    out_dir = os.path.join(private_dir(args.data), "models", name)
    os.makedirs(out_dir, exist_ok=True)
    log = {"curve": []}
    model.train()
    tokens, steps, t_train, t_val = 0, 0, 0.0, 0.0
    if args.exposure == "once":                                   # the stream in lived order, each item once
        batches = [seqs[i:i + args.batch] for i in range(0, len(seqs), args.batch)]
        warm = max(1, len(batches) // 20)
        t0 = time.time()
        for b in batches:
            for g in opt.param_groups:                          # warm-up, then cosine to 10% over the pass
                g["lr"] = lr * min(1.0, (steps + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(
                    math.pi * steps / len(batches))))
            x, y = batch_tensors(b, units.pad, 32 if args.bf16 else 1)
            loss = loss_of(model, x, y, args.bf16)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tokens += int((y != -100).sum())
            steps += 1
        t_train = time.time() - t0
        best_state, epochs_run, best_epoch = None, 1, 1
        if val_pairs:
            t1 = time.time()
            model.eval()                                          # the learning rate trial: answer loss only
            log["curve"].append(dict(epoch=1, loss=answer_loss(model, units, val_pairs)))
            model.train()
            t_val = time.time() - t1
    else:                                                         # many shuffled epochs, chosen by validation
        rng = random.Random(args.seed)
        per_epoch = math.ceil(len(seqs) / args.batch)
        total = per_epoch * args.max_epochs
        warm = min(200, total // 20)
        best, best_state, best_epoch, stale, epochs_run, best_loss = -1.0, None, 0, 0, 0, float("inf")
        for epoch in range(1, args.max_epochs + 1):
            order = list(range(len(seqs)))
            rng.shuffle(order)
            chunks = [sorted(order[i:i + 16 * args.batch], key=lambda j: len(seqs[j]))
                      for i in range(0, len(order), 16 * args.batch)]
            batches = [c[i:i + args.batch] for c in chunks for i in range(0, len(c), args.batch)]
            rng.shuffle(batches)
            t0, epoch_loss = time.time(), []
            for b in batches:
                frac = steps / max(1, total)
                for g in opt.param_groups:
                    g["lr"] = lr * min(1.0, (steps + 1) / max(1, warm)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(
                        math.pi * min(1.0, frac))))
                x, y = batch_tensors([seqs[j] for j in b], units.pad, 32 if args.bf16 else 1)
                loss = loss_of(model, x, y, args.bf16)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tokens += int((y != -100).sum())
                steps += 1
                epoch_loss.append(loss.item())
            t_train += time.time() - t0
            epochs_run = epoch
            if epoch % args.eval_every and epoch != args.max_epochs:
                continue
            t1 = time.time()
            v = validate(model, units, val_pairs)
            t_val += time.time() - t1
            log["curve"].append(dict(epoch=epoch, train_s=round(t_train, 1),
                                     train_loss=round(sum(epoch_loss) / max(1, len(epoch_loss)), 4), **v))
            print(json.dumps(log["curve"][-1]), flush=True)
            improved = v["loss"] < best_loss                        # still learning: the answer loss falls
            best_loss = min(best_loss, v["loss"])
            if v["f1"] > best:                                       # the checkpoint: the best validation F1
                best, best_epoch, improved = v["f1"], epoch, True
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            stale = 0 if improved else stale + 1
            if stale >= args.patience:                               # neither F1 nor the loss improved lately
                break
        model.load_state_dict(best_state)
    pairs_seen = sum(1 for i, it in enumerate(items) if it["kind"] == "pair" and it["phase"] == "teacher"
                     and i not in val)
    meta = {"name": name, "units": args.units, "size": args.size, "exposure": args.exposure, "vocab": units.size,
            "d": cfg["d"], "layers": cfg["layers"], "heads": cfg["heads"], "dropout": dropout, "lr": lr,
            "batch": args.batch, "ctx": args.ctx, "bf16_training": bool(args.bf16), "threads": args.threads,
            "params": params, "embedding_params": emb, "non_embedding_params": params - emb,
            "sequences": len(seqs), "teacher_pairs_seen": pairs_seen, "validation_pairs": len(val_pairs),
            "epochs_run": epochs_run, "best_epoch": best_epoch, "steps": steps, "tokens_seen": tokens,
            "train_s": round(t_train, 1), "validation_s": round(t_val, 1), "rss_mb_after_imports": round(rss0),
            "peak_rss_mb_training": round(peak_rss_mb()), "curve": log["curve"], "seed": args.seed}
    path = os.path.join(out_dir, "model.pt")
    reserve = os.path.join(private_dir(args.data), "reserve.bin")   # space kept free on a shared disk, if any
    for attempt in range(60):                     # a full shared disk must not lose a finished training
        try:
            if os.path.exists(reserve):
                os.remove(reserve)
            torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
            break
        except (OSError, RuntimeError) as exc:
            try:
                print("saving failed (%s), retrying in 30 s" % type(exc).__name__, flush=True)
            except OSError:
                pass
            time.sleep(30)
    meta["checkpoint_bytes"] = os.path.getsize(path)
    with open(os.path.join(out_dir, "train.json"), "w") as f:
        json.dump(meta, f)
    print(json.dumps({k: v for k, v in meta.items() if k != "curve"}), flush=True)


def run_answer(args) -> None:
    from f25_teacher import score
    from zihin.kendilik import LESSONS
    torch.set_num_threads(1)
    rss0 = rss_mb()
    out_dir = os.path.join(private_dir(args.data), "models", args.name)
    with open(os.path.join(out_dir, "train.json")) as f:
        meta = json.load(f)
    units = load_units(args.data, meta["units"])
    model = build_model(units.size, meta["d"], meta["layers"], meta["heads"], 0.0)
    model.load_state_dict(torch.load(os.path.join(out_dir, "model.pt"))["state_dict"])
    model.eval()
    rss_loaded = rss_mb()
    per, cut, ms, said_all = [], [], [], []
    for question, answer in held_out(args.data):
        a = time.perf_counter()
        ids = greedy(model, units.prompt(question), units.son)
        said = as_said(units, ids)
        ms.append(1000 * (time.perf_counter() - a))
        per.append(score(said, answer))
        cut.append(score(as_said(units, cut_at_line_end(units, ids)), answer))
        said_all.append(said)
    probe, probe_out = [], []
    for question, answer in LESSONS:
        said = as_said(units, greedy(model, units.prompt(question), units.son))
        probe.append(score(said, answer))
        probe_out.append(said)
    unknown, unknown_out = [], []
    for question, answer in unknown_word_questions():
        said = as_said(units, greedy(model, units.prompt(question), units.son))
        unknown.append(score(said, answer))
        unknown_out.append(said)
    with open(os.path.join(out_dir, "answers.json"), "w", encoding="utf-8") as f:
        json.dump({"test": said_all, "self": probe_out, "unknown": unknown_out}, f, ensure_ascii=False)
    row = {"model": "transformer", **{k: v for k, v in meta.items()}, "answer_s": round(sum(ms) / 1000, 1),
           "rss_mb_answer_start": round(rss0), "rss_mb_model_loaded": round(rss_loaded),
           "peak_rss_mb_answering": round(peak_rss_mb()), "latency_1_thread": latency_summary(ms)}
    row.update(aggregate(per))
    c = aggregate(cut)
    row["cut_at_line_end"] = {k: c[k] for k in ("exact", "f1", "bilmiyorum", "silent", "asked", "length")}
    row["self_probe"] = {"asked": len(probe), "as_taught": sum(p["exact"] for p in probe),
                         "f1": round(sum(p["f1"] for p in probe) / len(probe), 4),
                         "per_question": [p["exact"] for p in probe]}
    row["unknown_probe"] = {"asked": len(unknown), "bilmiyorum": sum(p["bilmiyorum"] for p in unknown),
                            "per_question": [p["bilmiyorum"] for p in unknown]}
    append_row(row, args.out)
    print(json.dumps({k: v for k, v in row.items() if k not in ("per_question", "curve")}), flush=True)


def run_bpe(args) -> None:
    from tokenizer_management.byte_bpe import ByteLevelBPE
    items = load_corpus(args.data)
    d = private_dir(args.data)
    t0 = time.time()
    tok = ByteLevelBPE.train(corpus_texts(items), vocab_size=4000, min_frequency=2)
    tok.save(os.path.join(d, "bpe4000_corpus.json"))
    vocab = WordUnits.build_vocab(items)
    with open(os.path.join(d, "word_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    earlier = None
    try:
        earlier = ByteLevelBPE.load(os.path.join(args.data, "bpe4000.json")).identity() == tok.identity()
    except Exception as exc:                                             # noqa: BLE001
        earlier = "unreadable: %s" % type(exc).__name__
    bpe_units = BPEUnits(os.path.join(d, "bpe4000_corpus.json"))
    word_units = WordUnits(os.path.join(d, "word_vocab.json"))
    counts = {}
    for name, u in (("bpe", bpe_units), ("word", word_units)):
        seqs = sequences(items, u, 128)
        counts[name] = {"vocab": u.size, "sequences": len(seqs), "units": sum(len(s) for s in seqs)}
    print(json.dumps({"bpe_s": round(time.time() - t0, 1), "bpe_vocab": tok.vocab_size,
                      "same_as_earlier_bpe4000": earlier, "stream": counts}), flush=True)


ONCE_LRS = (1e-3, 2e-3, 4e-3)
ORDER = (("1m", "once"), ("1m", "epochs"), ("4m", "once"), ("4m", "epochs"), ("16m", "once"), ("16m", "epochs"))
EPOCHS = {"1m": dict(max_epochs=20, eval_every=2, patience=3), "4m": dict(max_epochs=10, eval_every=2, patience=2),
          "16m": dict(max_epochs=6, eval_every=2, patience=2)}


def run_plan(args) -> None:
    """The whole duel, one process per job (so that each job's peak RSS is its own), smallest models first.
    The one-pass learning rate is chosen by the answer loss of the validation pairs with the 1m BPE model (every
    one-pass model takes it, the 16m ones half of it); every other choice is fixed above. Jobs already done are
    skipped, so the plan can resume."""
    import subprocess
    me = [sys.executable, os.path.abspath(__file__)]
    common = ["--data", args.data, "--out", args.out]
    models = os.path.join(private_dir(args.data), "models")
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(line).get("name") for line in open(args.out, encoding="utf-8")}

    def meta(name):
        path = os.path.join(models, name, "train.json")
        return json.load(open(path)) if os.path.exists(path) else None

    def call(*extra):
        print("$ " + " ".join(extra), flush=True)
        r = subprocess.run(me + list(extra) + common)
        if r.returncode:
            print("job failed (%d): %s" % (r.returncode, " ".join(extra)), flush=True)
        return r.returncode == 0

    def train_and_answer(units, size, exposure, lr=None):
        name = "%s-%s-%s" % (units, size, exposure)
        if meta(name) is None:
            extra = ["train", "--units", units, "--size", size, "--exposure", exposure, "--name", name]
            if lr:
                extra += ["--lr", "%g" % lr]
            if exposure == "epochs":
                e = EPOCHS[size]
                extra += ["--max-epochs", str(e["max_epochs"]), "--eval-every", str(e["eval_every"]),
                          "--patience", str(e["patience"])]
            if not call(*extra):
                return
        if name not in done:
            call("answer", "--name", name)

    best = {}
    for units in ("bpe",):
        trials = []

        def trial(lr):
            name = "%s-1m-once-val-lr%g" % (units, lr)
            if meta(name) is None:
                call("train", "--units", units, "--size", "1m", "--exposure", "once", "--validate", "--lr",
                     "%g" % lr, "--name", name)
            m = meta(name)
            if m:
                trials.append((-m["curve"][-1]["loss"], lr))

        for lr in ONCE_LRS:
            trial(lr)
        if trials and max(trials)[1] == ONCE_LRS[-1]:          # the best at the edge: one step further
            trial(2 * ONCE_LRS[-1])
        best[units] = max(trials)[1] if trials else 2e-3
        print(json.dumps({"once_lr": units, "trials": trials, "chosen": best[units]}), flush=True)
    best["word"] = best["bpe"]
    for size, exposure in ORDER:                                # smallest first; 16m last (it may not fit)
        if size not in args.sizes or exposure not in args.exposures:
            continue
        for units in (("bpe", "word") if exposure == "once" else args.epoch_units.get(size, ("bpe", "word"))):
            train_and_answer(units, size, exposure, (best[units] if size != "16m" else best[units] / 2)
                             if exposure == "once" else None)


def rescore(args) -> None:
    """Score the saved private answers again with f25_teacher.score and compare with the written vectors."""
    from f25_teacher import score
    test = held_out(args.data)
    for r in (json.loads(line) for line in open(args.out, encoding="utf-8")):
        name = r.get("name", r["model"])
        path = os.path.join(private_dir(args.data), "answers_zihin.json") if r["model"] == "zihin" else os.path.join(
            private_dir(args.data), "models", name, "answers.json")
        said = json.load(open(path, encoding="utf-8"))["test"]
        again = aggregate([score(s, a) for s, (_, a) in zip(said, test)])
        print("%-24s exact %.3f F1 %.4f  same per-question vector: %s" % (
            name, again["exact"], again["f1"], again["per_question"] == r["per_question"]))


def table(path: str) -> None:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    print("| model | params | exposure | pairs seen | units seen | exact | F1 | bilmiyorum | silent | asked | "
          "length | self probe | learn s | peak RSS learn / answer MB | answer ms (1 thread) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["model"] == "zihin":
            name, params, exposure, pairs, seen = "Zihin (fresh, f25 protocol)", "%.2fM ties" % (
                r["ties"] / 1e6), "lived once", 1358, "%d moments" % r["watched_moments"]
            learn, rss = r["train_s"], "%d / %d" % (r["peak_rss_mb_living"], r["peak_rss_mb_answering"])
        else:
            name = "%s %s %s" % (r["units"], r["size"], r["name"].split("-", 2)[-1])
            params, exposure = "%.2fM" % (r["params"] / 1e6), (
                "once" if r["exposure"] == "once" else "%d ep (best %d)" % (r["epochs_run"], r["best_epoch"]))
            pairs, seen = r["teacher_pairs_seen"], "%d tokens" % r["tokens_seen"]
            learn, rss = r["train_s"], "%d / %d" % (r["peak_rss_mb_training"], r["peak_rss_mb_answering"])
        print("| %s | %s | %s | %s | %s | %.3f (%d) | %.3f | %.3f | %.3f | %.3f | %.1f | %d/%d | %.0f | %s | %.0f |" % (
            name, params, exposure, pairs, seen, r["exact"], round(r["exact"] * r["questions"]), r["f1"],
            r["bilmiyorum"], r["silent"], r["asked"], r["length"], r["self_probe"]["as_taught"],
            r["self_probe"]["asked"], learn, rss, r["latency_1_thread"]["mean_ms"]))
    zihin = next((r for r in rows if r["model"] == "zihin"), None)
    if zihin is None:
        return
    print("\npaired over the same questions (model - Zihin): exact, F1 (mean +- se; questions higher / equal / lower)")
    for r in rows:
        if r["model"] == "zihin":
            continue
        line = []
        for i, key in ((0, "exact"), (1, "F1")):
            d = [a[i] - b[i] for a, b in zip(r["per_question"], zihin["per_question"])]
            mu = sum(d) / len(d)
            se = (sum((x - mu) ** 2 for x in d) / (len(d) - 1) / len(d)) ** 0.5
            line.append("%s %+.3f +- %.3f (%d/%d/%d)" % (key, mu, se, sum(x > 0 for x in d), sum(x == 0 for x in d),
                                                       sum(x < 0 for x in d)))
        both = sum(1 for a, b in zip(r["per_question"], zihin["per_question"]) if a[0] and b[0])
        print("  %-22s %s; exact by both %d" % (r["name"], "; ".join(line), both))


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["zihin", "bpe", "train", "answer", "plan", "table", "rescore"])
    ap.add_argument("--sizes", nargs="+", default=["1m", "4m", "16m"])
    ap.add_argument("--exposures", nargs="+", default=["once", "epochs"])
    ap.add_argument("--data", default="private_data")
    ap.add_argument("--documents", default=None)
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--units", choices=["bpe", "word"], default="bpe")
    ap.add_argument("--size", choices=["1m", "4m", "16m"], default="1m")
    ap.add_argument("--exposure", choices=["once", "epochs"], default="once")
    ap.add_argument("--validate", action="store_true", help="hold out the validation pairs in a once run too")
    ap.add_argument("--name", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", type=int, default=None, help="default: on for epochs (batches of 16-32), off once")
    args = ap.parse_args()
    if args.batch is None:
        args.batch = 8 if args.exposure == "once" else (32 if args.units == "bpe" else 16)
    if args.bf16 is None:
        args.bf16 = int(args.exposure == "epochs")
    if args.command == "zihin":
        run_zihin(args)
    elif args.command == "bpe":
        run_bpe(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "answer":
        run_answer(args)
    elif args.command == "plan":
        args.epoch_units = {"16m": ("bpe",)}
        run_plan(args)
    elif args.command == "table":
        table(args.out)
    elif args.command == "rescore":
        rescore(args)


if __name__ == "__main__":
    main()
