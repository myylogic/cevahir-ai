# -*- coding: utf-8 -*-
"""t_race.py: the same machine, the same data, the same questions: the mind living a folder of documents once
against a Transformer trained on it from nothing (section 40).

    python research/living_mind_2026_11_22/t_race.py mind --data FOLDER [--variant speaking|full] [--off a,b] [--set a=1]
    python research/living_mind_2026_11_22/t_race.py transformer --data FOLDER [--minutes 60] [--threads 1]
                                                            [--at 2 5 10 20 40 60] [--d 256 --layers 4 --heads 4]

``FOLDER`` holds ``belgeler``: the documents. Both read exactly the same sentences (the program's sentence split,
``sentences``) in the same units (the mind's own words, ``Session.body``). The questions are the same 200 for
both: sentences of at least five words drawn with ``random.Random(0)``; each is cued by its first two words, and
what comes is scored as ``recall_once.py`` scores it: exact (the rest of that sentence), valid (the rest of some
sentence that begins with the same two words), silent, the share of the words in place, word F1.

``mind``: a new mind lives the folder once as the program reads a file (``Session.read_path``: every sentence
heard, its end perceived, a pause), on one thread; then it brings each cue to mind with the inner voice
(``dusunce.think``, its words only: the reality score, a second reading of every step that the words do not depend
on, is not read, as the Transformer is not asked for one).

``transformer``: the minimal decoder recipe of the earlier duel (``recall_once.GPT``: RoPE, RMSNorm, SwiGLU, tied
embeddings, causal; d 256, 4 layers, 4 heads by default), initialised at random (seed 0), trained on the
sentences in reading order, one sentence a line ([BOS] words [EOS]), packed into windows of 128 units, AdamW
(0.9, 0.95), learning rate 4e-3 (the earlier duel's choice for this size) warmed up over 100 steps and lowered on
a cosine of the time spent (to a tenth at ``--minutes``), batches of 16 windows, gradient clip 1, as many epochs
(shuffled windows after the first, which is in reading order) as the time allows. It is asked the same cues by
greedy decoding after [BOS] cue, up to [EOS] or 60 units, at each of ``--at`` minutes of training (the clock is
stopped while it is asked) and at the end of its first epoch (one pass, as the mind reads).

Every row holds the processor seconds of the process (what a busy machine does not change much) and the wall
seconds, the answering seconds, the peak memory of the process, and the scores. Nothing of the documents and
nothing either one said is printed or written: counts and times only. Appends to ``results/race.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "race.jsonl")
CUE, N, MAX_NEW = 2, 200, 60


def peak_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def corpus(data: str):
    """(every sentence of the documents as the program splits them, each as the mind's words; a session to read)."""
    import zihin.__main__ as chat
    from zihin import Mind
    from zihin.__main__ import CHANNEL, Session, read_document, sentences
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        probe = Session(Mind(chat.VARIANTS["speaking"]()))
    finally:
        chat._log_path = made
    folder = os.path.join(data, "belgeler")
    bodies = []
    for path in sorted(os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs):
        try:
            text = read_document(path)
        except Exception:                                  # noqa: BLE001 (not a text file)
            continue
        for sentence in sentences(text, probe.dil):
            body = probe.body(sentence)
            if body:
                bodies.append(body)
    return bodies, CHANNEL


def questions(bodies):
    import recall_once as RO
    sample = [b for b in bodies if len(b) >= CUE + 3]
    every = [b for b in bodies if len(b) > CUE]
    return random.Random(0).sample(sample, min(N, len(sample))), RO.Score(sample, every, CUE)


def append(row: dict, path: str) -> None:
    print(json.dumps(row), flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_mind(args) -> None:
    import zihin.__main__ as chat
    from zihin import Mind
    from zihin.__main__ import Session
    from zihin.dusunce import think
    torch.set_num_threads(1)
    bodies, channel = corpus(args.data)
    chosen, score = questions(bodies)
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = chat.VARIANTS[args.variant]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    off = [k for k in args.off.split(",") if k]
    on = [kv.split("=")[0] for kv in args.set.split(",") if kv and kv.endswith("=1")]
    for name in off + on:
        if not hasattr(cfg, name):
            raise SystemExit("no such switch: %s" % name)
        setattr(cfg, name, name in on)
    torch.manual_seed(0)
    mind = Mind(cfg)
    mind.add_channel(channel, "act")
    session = Session(mind)
    w0, c0 = time.time(), time.process_time()
    session.read_path(os.path.join(args.data, "belgeler"))
    read_wall, read_cpu = time.time() - w0, time.process_time() - c0
    each = []
    w0, c0 = time.time(), time.process_time()
    for body in chosen:
        t = time.time()
        thought, _ = think(session, " ".join(body[:CUE]), limit=MAX_NEW, reality=False)
        each.append(time.time() - t)
        score.add(body, thought)
    each.sort()
    row = {"test": "race", "who": "zihin", "variant": args.variant, "off": off, "on": on,
           "sentences": len(bodies), "moments": mind.t, "ties": mind.S.n_ties,
           "learn_cpu_s": round(read_cpu, 1), "learn_wall_s": round(read_wall, 1), "passes": 1,
           "answer_cpu_s": round(time.process_time() - c0, 1), "answer_wall_s": round(time.time() - w0, 1),
           "answer_median_s": round(each[len(each) // 2], 3), "answer_p90_s": round(each[int(len(each) * 0.9)], 3),
           "threads": 1, "peak_mb": peak_mb(), "note": args.note,
           "recall_or_reason": mind.__dict__.get("_recall_counts")}
    row.update(score.row())
    append(row, args.out)


def run_transformer(args) -> None:
    import recall_once as RO
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    bodies, _ = corpus(args.data)
    chosen, _ = questions(bodies)
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2}
    for body in bodies:
        for w in body:
            vocab.setdefault(w, len(vocab))
    itos = {i: w for w, i in vocab.items()}
    stream = []
    for body in bodies:
        stream += [1] + [vocab[w] for w in body] + [2]
    ctx, rows = 128, 16
    windows = [stream[i:i + ctx + 1] for i in range(0, len(stream) - 1, ctx)]
    windows = [w + [0] * (ctx + 1 - len(w)) for w in windows]
    data = torch.tensor(windows, dtype=torch.long)
    model = RO.GPT(len(vocab), args.d, args.layers, args.heads, ctx)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    budget = args.minutes * 60.0
    marks = sorted(set(m * 60.0 for m in args.at if m * 60.0 <= budget) | {budget})
    trained_wall = trained_cpu = 0.0
    step = epoch = 0
    rng = random.Random(0)

    def ask(label):
        import recall_once as R2
        _, score = questions(bodies)
        model.eval()
        each = []
        w0, c0 = time.time(), time.process_time()
        with torch.no_grad():
            for body in chosen:
                t = time.time()
                ids = R2.decode(model, [1] + [vocab[w] for w in body[:CUE]], 2, True, MAX_NEW)
                each.append(time.time() - t)
                score.add(body, [itos[i] for i in ids if i > 2])
        model.train()
        each.sort()
        row = {"test": "race", "who": "transformer", "recipe": "gpt", "d": args.d, "layers": args.layers,
               "heads": args.heads, "params": params, "vocab": len(vocab), "sentences": len(bodies),
               "units": len(stream), "at": label, "epochs": round(epoch_progress(), 3), "steps": step,
               "learn_cpu_s": round(trained_cpu, 1), "learn_wall_s": round(trained_wall, 1),
               "answer_cpu_s": round(time.process_time() - c0, 1), "answer_wall_s": round(time.time() - w0, 1),
               "answer_median_s": round(each[len(each) // 2], 4), "answer_p90_s": round(each[int(len(each) * 0.9)], 4),
               "threads": args.threads, "lr": args.lr, "peak_mb": peak_mb(), "note": args.note}
        row.update(score.row())
        append(row, args.out)

    order = list(range(len(windows)))
    position = [0]

    def epoch_progress():
        return epoch + position[0] / len(order)

    one_pass_asked = False
    model.train()
    while marks:
        if position[0] >= len(order):
            epoch += 1
            position[0] = 0
            rng.shuffle(order)
            if not one_pass_asked:
                one_pass_asked = True
                ask("one pass")
        idx = order[position[0]:position[0] + rows]
        position[0] += len(idx)
        w0, c0 = time.time(), time.process_time()
        batch = data[idx]
        x, y = batch[:, :-1], batch[:, 1:]
        frac = min(1.0, trained_wall / budget)
        lr = args.lr * min(1.0, (step + 1) / 100.0) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bool(args.bf16)):
            logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1),
                                                 ignore_index=0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        trained_wall += time.time() - w0
        trained_cpu += time.process_time() - c0
        while marks and trained_wall >= marks[0]:
            ask("%g min" % (marks.pop(0) / 60.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("who", choices=["mind", "transformer"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", default="speaking")
    ap.add_argument("--off", default="")
    ap.add_argument("--set", default="")
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--at", type=float, nargs="*", default=[2, 5, 10, 20, 40, 60])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--bf16", type=int, default=1)
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    (run_mind if args.who == "mind" else run_transformer)(args)


if __name__ == "__main__":
    main()
