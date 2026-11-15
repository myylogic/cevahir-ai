# -*- coding: utf-8 -*-
"""Which of them runs on a low-end machine: Zihin minds and Transformers measured on ONE core without a GPU, and what
happens as the data grows.

Every job is one process, pinned to cores by the caller, and appends one row of numbers to
``results/lowend_profile.jsonl``. No question, answer, document word or generated answer is printed or written; the
reply probes are ten everyday questions of this script's own (QUESTIONS), and only their latencies and word counts
are kept. Anything built from private data is saved only under PRIVATE/duel/.

    P=private_data; R=research/living_mind_2026_11_15/lowend_profile.py
    nice -n 10 taskset -c 3 python $R corpus --data $P            # token counts (bpe4000) and pair lengths
    nice -n 10 taskset -c 3 python $R teacher --data $P           # the F25 teacher mind -> $P/duel/lowend_teacher.zihin
    nice -n 10 taskset -c 3 python $R mind --data $P --name cevahir2 --path $P/cevahir2.zihin --breakdown
    nice -n 10 taskset -c 3 python $R traced --data $P --build    # tracemalloc by file/line over a teacher life
    nice -n 10 taskset -c 3 python $R traced --data $P --name teacher --path $P/duel/lowend_teacher.zihin
    nice -n 10 taskset -c 3 python $R convert --data $P --name cevahir3b --path $P/cevahir3b.zihin   # today's layout
    nice -n 10 taskset -c 3 python $R mind --data $P --name cevahir3b_today --path $P/duel/lowend_today_cevahir3b.zihin
    nice -n 10 taskset -c 3 python $R transformer --size 20m --mode infer
    nice -n 10 taskset -c 2,3 python $R transformer --size 20m --mode train --threads 2
    python $R project                                             # low-end summary and projection rows

The committed rows were made in this order, one process at a time: corpus; teacher; mind for teacher, ilk_nefes,
cevahir2 and cevahir3b (``--breakdown``, ``--resave`` for the three old-layout minds, whose resaved files were
measured and removed); transformer infer and train for the four sizes (1 thread on core 3, 2 threads on cores 2-3);
traced (``--build``, teacher, cevahir2); convert + mind for cevahir2_today and cevahir3b_today (removed after); a
second sample of the 2-thread training (the first shared core 2 with another workflow; ``project`` keeps the faster
of repeated runs); project. All four cores were busy with other workflows throughout (load 3-6), so the one-core
costs in the summary and projection rows are CPU times of the process; the wall times kept beside them are what
was observed.

mind: file size; load time (and its parts: sha256, unpickling with the old-layout conversion, digest); RSS after
loading, its peak during loading, during 10 replies and during watching the first 60 dialogues of the private
daily-conversation file (``*gunluk-konusma.json``); reply latency; watching ms per moment; units, ties, moments.
CPU time (process_time) is kept beside every wall time, because the machine was shared. ``--breakdown`` walks
the object graph after the timings (tensor storage bytes by name; Python bytes of each index and cache, each object
counted once, small cached ints not counted, pymalloc's 16-byte rounding applied). traced: tracemalloc, either over
a teacher life built in the process (allocation sites by file and line) or over a loaded mind (Python bytes freed
when each structure is dropped, an exact check of the walk). transformer: a GPT-style decoder (pre-LayerNorm, causal
self-attention, GELU MLP, tied embeddings, vocabulary 4000 as bpe4000, context 128), untrained weights: inference
RSS, ms per generated token with a KV cache, ms per answer of 15 words (in bpe4000 tokens per Zihin word, from the
corpus row); training ms per token (AdamW, cross-entropy) on the threads given.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "lowend_profile.jsonl")
QUESTIONS = ("merhaba", "nasılsın ?", "adın ne ?", "ne yapıyorsun ?", "bugün hava nasıl ?",
             "en sevdiğin renk hangisi ?", "kitap okumayı sever misin ?", "yapay zeka nedir ?",
             "bana kendinden bahset", "teşekkür ederim , iyi geceler")
DIALOGUES = "gunluk-konusma.json"                # found in the data folder by this ending
CORPUS_TOKENS = 260_000                       # the duel's corpus size, as given (the corpus row measures ours)
MiB = float(2 ** 20)


def status() -> dict:
    out = {}
    with open("/proc/self/status") as f:
        for line in f:
            key = line.split(":", 1)[0]
            if key in ("VmRSS", "VmHWM"):
                out[key] = int(line.split()[1]) * 1024
    return out


def reset_peak() -> bool:
    """Linux: writing 5 to clear_refs resets the peak RSS (VmHWM) to the current RSS."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def trim() -> None:
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def mb(x: float) -> float:
    return round(x / MiB, 1)


def machine() -> dict:
    return {"cpus": sorted(os.sched_getaffinity(0)), "torch_threads": torch.get_num_threads(), "nice": os.nice(0),
            "loadavg_1m": round(os.getloadavg()[0], 2)}


def write(row: dict, path: str) -> None:
    row["when"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def rows_of(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def latest(rows: list, **match) -> dict:
    found = [r for r in rows if all(r.get(k) == v for k, v in match.items())]
    return found[-1] if found else {}


# ---------------------------------------------------------------------------------------------------------- corpus
def corpus(args) -> None:
    """Counts only: teacher pairs and documents in bpe4000 tokens and in Zihin's words (symbols)."""
    from tokenizer_management.byte_bpe import ByteLevelBPE
    from zihin.__main__ import CODE, read_document, sentences, words
    bpe = ByteLevelBPE.load(os.path.join(args.data, "bpe4000.json"))
    with open(os.path.join(args.data, "ogretmen.tsv"), encoding="utf-8", errors="replace") as f:
        pairs = [line.rstrip("\n").split("\t", 1) for line in f if "\t" in line]
    q_tok = [len(bpe.encode(q)) for q, _ in pairs]
    a_tok = [len(bpe.encode(a)) for _, a in pairs]
    q_sym = [len(words(q)) for q, _ in pairs]
    a_sym = [len(words(a)) for _, a in pairs]
    a_ws = [len(a.split()) for _, a in pairs]
    folder = os.path.join(args.data, "belgeler")
    files = sorted(os.path.join(r, n) for r, _, ns in os.walk(folder) for n in ns)
    d_sent = d_tok = d_sym = 0
    for name in files:
        text = read_document(name)
        lines = [ln.strip() for ln in text.splitlines() if len(words(ln)) >= 2] if name.lower().endswith(CODE) \
            else sentences(text)
        d_sent += len(lines)
        d_tok += sum(len(bpe.encode(s)) for s in lines)
        d_sym += sum(len(words(s)) for s in lines)
    teacher_tokens = sum(q_tok) + sum(a_tok)
    write({"kind": "corpus", "bpe": {"format": "cevahir-byte-bpe v1 (tokenizer_management/byte_bpe.py)",
                                     "vocab": bpe.vocab_size},
           "pairs": len(pairs), "question_tokens_mean": round(statistics.mean(q_tok), 2),
           "answer_tokens_mean": round(statistics.mean(a_tok), 2),
           "question_symbols_mean": round(statistics.mean(q_sym), 2),
           "answer_symbols_mean": round(statistics.mean(a_sym), 2),
           "tokens_per_symbol": round(sum(a_tok) / sum(a_sym), 3), "tokens_per_word": round(sum(a_tok) / sum(a_ws), 3),
           "pair_tokens_mean": round(teacher_tokens / len(pairs), 2), "teacher_tokens": teacher_tokens,
           "doc_files": len(files), "doc_sentences": d_sent, "doc_tokens": d_tok, "doc_symbols": d_sym,
           "teacher_plus_docs_tokens": teacher_tokens + d_tok}, args.out)


# --------------------------------------------------------------------------------------------------------- teacher
def teacher(args) -> None:
    """The F25 teacher protocol without documents (setting off): watch ogretmen.tsv, lessons once, answer the 200
    held-out questions; scores aggregated only (a reproduction check), the mind saved under PRIVATE/duel/."""
    torch.set_num_threads(1)
    import random
    from f25_teacher import SPEAKING, score
    from zihin import Config, Mind
    from zihin.__main__ import CHANNEL, Session
    from zihin.persist import save
    gc.collect()
    base = status()
    mind = Mind(Config.changed(keep_log=False, **SPEAKING))
    mind.add_channel(CHANNEL, "act")
    session = Session(mind)
    t0, c0 = time.perf_counter(), time.process_time()
    session.read_path(os.path.join(args.data, "ogretmen.tsv"))
    pairs_s, pair_moments, pair_ties = time.perf_counter() - t0, mind.t, mind.S.n_ties
    pairs_cpu = time.process_time() - c0
    session.lessons_about_itself(1)
    watch_s, watch_cpu, watched = time.perf_counter() - t0, time.process_time() - c0, mind.t
    with open(os.path.join(args.data, "ogretmen.tsv.heldout"), encoding="utf-8") as f:
        held = [line.rstrip("\n").split("\t", 1) for line in f if "\t" in line]
    random.Random(0).shuffle(held)
    t1, c1 = time.perf_counter(), time.process_time()
    per = [score(session.reply(q), a) for q, a in held[:200]]
    answer_s, answer_cpu = time.perf_counter() - t1, time.process_time() - c1
    gc.collect()
    trim()
    lived = status()
    os.makedirs(os.path.join(args.data, "duel"), exist_ok=True)
    path = os.path.join(args.data, "duel", "lowend_teacher.zihin")
    t2 = time.perf_counter()
    save(mind, path)
    save_s = time.perf_counter() - t2
    n = len(per)
    with open(os.path.join(args.data, "ogretmen.tsv"), encoding="utf-8", errors="replace") as f:
        watched_pairs = sum(1 for line in f if "\t" in line)
    write({"kind": "teacher_build", **machine(), "pairs": watched_pairs,
           "pair_moments": pair_moments, "pair_ties": pair_ties, "pairs_s": round(pairs_s, 1),
           "pairs_cpu_s": round(pairs_cpu, 1), "watched_moments": watched, "watch_s": round(watch_s, 1),
           "watch_cpu_s": round(watch_cpu, 1), "watch_ms_per_moment": round(1000 * watch_s / watched, 3),
           "watch_cpu_ms_per_moment": round(1000 * watch_cpu / watched, 3), "answer_s": round(answer_s, 1),
           "answer_cpu_s": round(answer_cpu, 1),
           "moments": mind.t, "ties": mind.S.n_ties, "units": mind.S.n_units,
           "exact": sum(p["exact"] for p in per) / n, "f1": round(sum(p["f1"] for p in per) / n, 4),
           "bilmiyorum": sum(p["bilmiyorum"] for p in per) / n, "silent": sum(p["silent"] for p in per) / n,
           "rss_base_mb": mb(base["VmRSS"]), "rss_lived_mb": mb(lived["VmRSS"]), "peak_lived_mb": mb(lived["VmHWM"]),
           "file_mb": mb(os.path.getsize(path)), "save_s": round(save_s, 1)}, args.out)


# ---------------------------------------------------------------------------------------------------------- minds
def _small_int(x: int) -> bool:
    return -5 <= x <= 256


def _rounded(n: int) -> int:
    return (n + 15) // 16 * 16 if n <= 512 else n


def walk(roots: list, flat_keys=("tie_index",)) -> dict:
    """Python bytes and tensor storage bytes reachable from each root; an object is counted once, for the first root
    that reaches it. Small ints are cached by Python and not counted; other ints and floats are counted where they
    are found (a loaded mind shares none: pickle does not memoise numbers). The keys of a ``flat_keys`` root (unique
    tuples of ints) are counted without remembering their ids."""
    import types
    skip = (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType, type)
    seen, storages, out = set(), set(), {}
    int_size, float_size = _rounded(sys.getsizeof(2 ** 20)), _rounded(sys.getsizeof(0.5))
    for name, root in roots:
        py = ten = objs = tensors = 0
        stack = [root]
        while stack:
            o = stack.pop()
            t = type(o)
            if t is int:
                if not _small_int(o):
                    py += int_size if -2 ** 30 < o < 2 ** 30 else _rounded(sys.getsizeof(o))
                    objs += 1
                continue
            if t is float:
                py += float_size
                objs += 1
                continue
            if o is None or t is bool:
                continue
            i = id(o)
            if i in seen:
                continue
            seen.add(i)
            if isinstance(o, torch.Tensor):
                tensors += 1
                s = o.untyped_storage()
                if s.data_ptr() not in storages:
                    storages.add(s.data_ptr())
                    ten += s.nbytes()
                continue
            if isinstance(o, skip):
                continue
            py += _rounded(sys.getsizeof(o))
            objs += 1
            if t is dict:
                if o is root and name in flat_keys:
                    for k, v in o.items():
                        py += _rounded(sys.getsizeof(k))
                        objs += 1
                        for x in k:
                            if type(x) is int and not _small_int(x):
                                py += int_size
                                objs += 1
                        stack.append(v)
                else:
                    stack.extend(o.keys())
                    stack.extend(o.values())
            elif t in (list, tuple, set, frozenset):
                stack.extend(o)
            elif t in (str, bytes, bytearray):
                pass
            else:
                d = getattr(o, "__dict__", None)
                if d is not None:
                    stack.append(d)
                for cls in t.__mro__:
                    for slot in getattr(cls, "__slots__", ()):
                        if isinstance(slot, str) and hasattr(o, slot):
                            stack.append(getattr(o, slot))
        out[name] = {"py_mb": mb(py), "tensor_mb": mb(ten), "objects": objs, "tensors": tensors,
                     "entries": len(root) if isinstance(root, (dict, list)) else None}
    return out


def roots_of(mind) -> list:
    S = mind.S
    return [("tie_index", S.tie_index), ("out_lists", S.out), ("inc_lists", S.inc),
            ("tuple_index", S.tuple_index), ("frame_first", S.frame_first), ("frame_of", S.frame_of),
            ("atom_index", S.atom_index), ("out_ids_cache", S._out_ids), ("channels", mind.ch),
            ("log", S.log), ("substance_rest", S), ("mind_rest", mind)]


def tensor_table(mind) -> dict:
    """Allocated storage bytes of every tensor attribute of the substance, and the bytes in use."""
    S = mind.S
    out = {}
    for name, v in S.__dict__.items():
        if isinstance(v, torch.Tensor):
            if name.startswith("u_"):
                used = S.n_units * v.element_size()
            elif name == "t_ev":
                used = 2 * S.n_ties * 8
            elif name.startswith("t_"):
                used = S.n_ties * v.element_size()
            elif name == "ex":
                used = 4 * S.n_rows * S.P * 8
            elif name.startswith("r_"):
                used = S.n_rows * v.element_size()
            else:
                used = v.numel() * v.element_size()
            out[name] = {"alloc_mb": mb(v.untyped_storage().nbytes()), "used_mb": mb(used)}
    return out


def breakdown(mind, base_rss: int, settled_rss: int) -> dict:
    gc.collect()
    trim()
    settled_rss = status()["VmRSS"]                    # the RSS of what is walked (after the replies and watching)
    t0 = time.perf_counter()
    tensors = tensor_table(mind)
    parts = walk(roots_of(mind))
    walk_s = time.perf_counter() - t0
    S = mind.S
    py_total = sum(p["py_mb"] for p in parts.values())
    tensor_total = sum(p["tensor_mb"] for p in parts.values())
    # what a tensor object costs beyond its storage (many small ones live in the out_ids cache)
    gc.collect()
    trim()
    before = status()["VmRSS"]
    probe = [torch.tensor([1, 2, 3], dtype=torch.int64) for _ in range(20000)]
    per_tensor = (status()["VmRSS"] - before) / len(probe)
    del probe
    consumers = [(name, p["py_mb"] + p["tensor_mb"]) for name, p in parts.items()
                 if name not in ("substance_rest", "mind_rest")]
    consumers += [("tensor " + name, v["alloc_mb"]) for name, v in tensors.items()]
    consumers.append(("out_ids tensor objects (estimated)", mb(parts["out_ids_cache"]["tensors"] * per_tensor)))
    consumers.append(("substance_rest (python)", parts["substance_rest"]["py_mb"]))
    consumers.append(("mind_rest (python + tensors)", parts["mind_rest"]["py_mb"] + parts["mind_rest"]["tensor_mb"]))
    consumers.sort(key=lambda kv: -kv[1])
    rss_mind = settled_rss - base_rss
    explained = (py_total + tensor_total) * MiB + parts["out_ids_cache"]["tensors"] * per_tensor
    return {"walk_s": round(walk_s, 1), "tensors": tensors, "python_and_tensors_by_root": parts,
            "python_total_mb": round(py_total, 1), "tensor_total_mb": round(tensor_total, 1),
            "tensor_object_bytes_measured": round(per_tensor, 1),
            "rss_at_walk_mb": mb(settled_rss), "rss_mind_mb": mb(rss_mind), "explained_mb": mb(explained),
            "unexplained_mb": mb(rss_mind - explained),
            "top5": [[name, round(v, 1)] for name, v in consumers[:5]],
            "ram_bytes_per_tie": round(rss_mind / max(1, S.n_ties), 1),
            "ram_bytes_per_moment": round(rss_mind / max(1, mind.t), 1),
            "python_bytes_per_tie": round(py_total * MiB / max(1, S.n_ties), 1),
            "tensor_bytes_per_tie": round(tensor_total * MiB / max(1, S.n_ties), 1),
            "index_bytes_per_tie": round(sum(parts[k]["py_mb"] for k in ("tie_index", "out_lists", "inc_lists"))
                                         * MiB / max(1, S.n_ties), 1)}


def mind_profile(args) -> None:
    torch.set_num_threads(1)
    import zihin.persist as persist
    from zihin.store import Substance
    from zihin.__main__ import Session, pieces_for
    timers = {"sha256": 0.0, "convert": 0.0, "digest": 0.0}

    def timed(key, fn):
        def inner(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                timers[key] += time.perf_counter() - t
        return inner

    persist._sha256 = timed("sha256", persist._sha256)
    Substance._from_tables = timed("convert", Substance._from_tables)
    Substance.digest = timed("digest", Substance.digest)
    gc.collect()
    base = status()
    reset_peak()
    t0, c0 = time.perf_counter(), time.process_time()
    mind = persist.load(args.path, ledger=False)
    load_s, load_cpu = time.perf_counter() - t0, time.process_time() - c0
    loaded = status()
    gc.collect()
    trim()
    settled = status()
    session = Session(mind, pieces_for(mind))
    reset_peak()
    lat = []
    for q in QUESTIONS:
        t, c = time.perf_counter(), time.process_time()
        said = session.reply(q)
        lat.append((time.perf_counter() - t, 0 if said == "(sessiz)" else len(said.split()), time.process_time() - c))
        del said
    replies = status()
    path = os.path.join(args.data, next(n for n in sorted(os.listdir(args.data)) if n.endswith(DIALOGUES)))
    reset_peak()
    m0, t, c = mind.t, time.perf_counter(), time.process_time()
    session.read_path(path, cap=60)
    watch_s, watch_cpu, watched = time.perf_counter() - t, time.process_time() - c, mind.t - m0
    watching = status()
    ms = [1000 * d for d, _, _ in lat]
    cpu_ms = [1000 * c for _, _, c in lat]
    said_words = sum(n for _, n, _ in lat)
    lat = [(d, n) for d, n, _ in lat]
    row = {"kind": "mind", "name": args.name, **machine(), "file_mb": mb(os.path.getsize(args.path)),
           "load_s": round(load_s, 2), "load_cpu_s": round(load_cpu, 2),
           "load_parts_s": {k: round(v, 2) for k, v in timers.items()},
           "reply_cpu_ms_median": round(statistics.median(cpu_ms), 1),
           "reply_cpu_ms_per_word": round(sum(cpu_ms) / said_words, 1) if said_words else None,
           "watch_cpu_s": round(watch_cpu, 2), "watch_cpu_ms_per_moment": round(1000 * watch_cpu / max(1, watched), 3),
           "cpu_share": round((sum(cpu_ms) / 1000 + watch_cpu) / (sum(ms) / 1000 + watch_s), 3),
           "rss_base_mb": mb(base["VmRSS"]), "rss_after_load_mb": mb(loaded["VmRSS"]),
           "peak_during_load_mb": mb(loaded["VmHWM"]), "rss_after_load_trimmed_mb": mb(settled["VmRSS"]),
           "peak_during_replies_mb": mb(replies["VmHWM"]), "peak_during_watch_mb": mb(watching["VmHWM"]),
           "rss_end_mb": mb(watching["VmRSS"]),
           "replies": len(lat), "reply_ms_median": round(statistics.median(ms), 1),
           "reply_ms_max": round(max(ms), 1), "reply_words_total": said_words,
           "silent_replies": sum(1 for _, n in lat if n == 0),
           "reply_ms_per_word": round(sum(ms) / said_words, 1) if said_words else None,
           "reply_ms_per_word_median": round(statistics.median([m / n for m, (_, n) in zip(ms, lat) if n]), 1)
           if said_words else None,
           "watch_dialogues": 60, "watch_moments": watched, "watch_s": round(watch_s, 2),
           "watch_ms_per_moment": round(1000 * watch_s / max(1, watched), 3),
           "moments": mind.t, "ties": mind.S.n_ties, "units": mind.S.n_units, "full_rows": mind.S.n_rows - 1,
           "pieces": session.pieces is not None}
    row["moments_before_watch"] = m0
    if args.breakdown:
        row["breakdown"] = breakdown(mind, base["VmRSS"], settled["VmRSS"])
    if args.resave:
        out = os.path.join(args.data, "duel", "lowend_resave_%s.zihin" % args.name)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        reset_peak()
        t = time.perf_counter()
        persist.save(mind, out)
        row["resave"] = {"s": round(time.perf_counter() - t, 1), "file_mb": mb(os.path.getsize(out)),
                         "ties": mind.S.n_ties, "moments": mind.t, "peak_mb": mb(status()["VmHWM"]),
                         "file_bytes_per_tie": round(os.path.getsize(out) / mind.S.n_ties, 1)}
        if not args.keep_resave:
            os.remove(out)
            os.remove(out + ".json")
    write(row, args.out)


def convert(args) -> None:
    """Save a mind of the old full-table layout in today's layout under PRIVATE/duel/ (no row: the ``mind`` job on
    the new file measures it, and the caller removes the file afterwards)."""
    from zihin.persist import load, save
    out = os.path.join(args.data, "duel", "lowend_today_%s.zihin" % args.name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save(load(args.path, ledger=False), out)
    print(out, flush=True)


# --------------------------------------------------------------------------------------------------- tracemalloc
def traced(args) -> None:
    import tracemalloc
    torch.set_num_threads(1)

    def sites(snapshot, key: str, top: int) -> list:
        out = []
        for stat in snapshot.statistics(key)[:top]:
            frame = stat.traceback[0]
            name = os.path.relpath(frame.filename, REPO) if frame.filename.startswith(REPO) else \
                os.path.basename(frame.filename)
            out.append([name + (":%d" % frame.lineno if key == "lineno" else ""), mb(stat.size), stat.count])
        return out

    if args.build:
        from f25_teacher import SPEAKING
        from zihin import Config, Mind
        from zihin.__main__ import CHANNEL, Session
        tracemalloc.start(1)
        mind = Mind(Config.changed(keep_log=False, **SPEAKING))
        mind.add_channel(CHANNEL, "act")
        session = Session(mind)
        session.read_path(os.path.join(args.data, "ogretmen.tsv"))
        session.lessons_about_itself(1)
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        snap = tracemalloc.take_snapshot()
        write({"kind": "traced_build", "moments": mind.t, "ties": mind.S.n_ties, "units": mind.S.n_units,
               "traced_mb": mb(current), "traced_peak_mb": mb(peak),
               "traced_bytes_per_tie": round(current / mind.S.n_ties, 1),
               "by_line": sites(snap, "lineno", 15), "by_file": sites(snap, "filename", 8)}, args.out)
        return
    from zihin.persist import load
    tracemalloc.start(1)
    mind = load(args.path, ledger=False)
    gc.collect()
    current0, _ = tracemalloc.get_traced_memory()
    estimate = walk(roots_of(mind))
    gc.collect()
    S = mind.S
    freed = {}
    for name, drop in (("tie_index", lambda: setattr(S, "tie_index", {})), ("out_lists", lambda: setattr(S, "out", {})),
                       ("inc_lists", lambda: setattr(S, "inc", {})),
                       ("tuple_index", lambda: setattr(S, "tuple_index", {})),
                       ("frame_first", lambda: setattr(S, "frame_first", {})),
                       ("frame_of", lambda: setattr(S, "frame_of", {})),
                       ("atom_index", lambda: setattr(S, "atom_index", {})),
                       ("channels", lambda: setattr(mind, "ch", None))):
        before = tracemalloc.get_traced_memory()[0]
        drop()
        gc.collect()
        freed[name] = mb(before - tracemalloc.get_traced_memory()[0])
    rest = tracemalloc.get_traced_memory()[0]
    del mind, S
    gc.collect()
    write({"kind": "traced_loaded", "name": args.name, "traced_after_load_mb": mb(current0),
           "freed_mb": freed, "traced_rest_mb": mb(rest),
           "walk_estimate_py_mb": {k: v["py_mb"] for k, v in estimate.items()}}, args.out)


# --------------------------------------------------------------------------------------------------- transformer
SIZES = {"1m": (128, 3, 4), "4m": (256, 4, 4), "20m": (512, 6, 8), "100m": (768, 14, 12)}   # d_model, layers, heads
VOCAB, CTX = 4000, 128


def gpt(d: int, layers: int, heads: int):
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.fc, self.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

        def forward(self, x, past=None):
            B, T, C = x.shape
            q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, heads, C // heads).permute(2, 0, 3, 1, 4)
            if past is not None:                       # one new token attends to every cached position
                k, v = torch.cat([past[0], k], dim=2), torch.cat([past[1], v], dim=2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=past is None)
            x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
            return x + self.out(F.gelu(self.fc(self.ln2(x)))), (k, v)

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok, self.pos = nn.Embedding(VOCAB, d), nn.Embedding(CTX, d)
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.ln = nn.LayerNorm(d)
            self.head = nn.Linear(d, VOCAB, bias=False)
            self.head.weight = self.tok.weight         # tied

        def forward(self, idx, past=None):
            start = 0 if past is None else past[0][0].shape[2]
            x = self.tok(idx) + self.pos(torch.arange(start, start + idx.shape[1]))
            cache = []
            for i, block in enumerate(self.blocks):
                x, kv = block(x, None if past is None else past[i])
                cache.append(kv)
            return self.head(self.ln(x)), cache

    model = GPT()
    for name, p in model.named_parameters():           # GPT-2 initialisation (std 0.02). With N(0, 1) embeddings
        if p.dim() >= 2:                               # tied to the head, untrained outputs have std ~ sqrt(d) and
            torch.nn.init.normal_(p, std=0.02)         # the softmax gradients are full of denormal floats, which
        elif name.endswith("bias"):                    # slow the head's matmul about 50 times on this CPU
            torch.nn.init.zeros_(p)
    return model


def transformer(args) -> None:
    import torch.nn.functional as F
    torch.set_num_threads(args.threads)
    torch.set_flush_denormal(True)
    torch.manual_seed(0)
    gc.collect()
    base = status()
    reset_peak()
    d, layers, heads = SIZES[args.size]
    model = gpt(d, layers, heads)
    params = sum(p.numel() for p in model.parameters())
    built = status()
    row = {"kind": "transformer", "size": args.size, "mode": args.mode, **machine(), "params": params,
           "d_model": d, "layers": layers, "heads": heads, "vocab": VOCAB, "ctx": CTX,
           "weights_fp32_mb": mb(4 * params), "rss_base_mb": mb(base["VmRSS"]), "rss_built_mb": mb(built["VmRSS"])}
    if args.mode == "infer":
        c = latest(rows_of(args.out), kind="corpus")
        prompt = int(round(c.get("question_tokens_mean", 20)))
        answer = int(round(15 * c.get("tokens_per_symbol", 1.5)))
        model.eval()
        steps, answers, prefills, cpu_steps, cpu_answers = [], [], [], [], []
        with torch.inference_mode():
            for rep in range(1 + args.answers):        # the first answer warms up
                idx = torch.randint(4, VOCAB, (1, prompt))
                t, c = time.perf_counter(), time.process_time()
                logits, past = model(idx)
                nxt = logits[:, -1].argmax(-1, keepdim=True)
                prefill, cpu_prefill = time.perf_counter() - t, time.process_time() - c
                times, cpus = [], []
                for _ in range(answer - 1):
                    t, c = time.perf_counter(), time.process_time()
                    logits, past = model(nxt, past)
                    nxt = logits[:, -1].argmax(-1, keepdim=True)
                    times.append(time.perf_counter() - t)
                    cpus.append(time.process_time() - c)
                if rep:
                    steps += times
                    cpu_steps += cpus
                    prefills.append(prefill)
                    answers.append(prefill + sum(times))
                    cpu_answers.append(cpu_prefill + sum(cpus))
        peak = status()
        row.update({"prompt_tokens": prompt, "answer_tokens": answer, "answer_words": 15,
                    "ms_per_token": round(1000 * statistics.median(steps), 2),
                    "cpu_ms_per_token": round(1000 * statistics.median(cpu_steps), 2),
                    "prefill_ms": round(1000 * statistics.median(prefills), 1),
                    "answer_ms": round(1000 * statistics.median(answers), 1),
                    "answer_cpu_ms": round(1000 * statistics.median(cpu_answers), 1),
                    "peak_infer_mb": mb(peak["VmHWM"]), "rss_infer_over_base_mb": mb(peak["VmHWM"] - base["VmRSS"])})
    else:
        batch = args.batch or (8 if args.size == "100m" else 16)
        steps = args.steps or (3 if args.size == "100m" else 6)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        x = torch.randint(0, VOCAB, (batch, CTX + 1))
        times, cpus = [], []
        for step in range(2 + steps):                  # two warm-up steps
            t, c = time.perf_counter(), time.process_time()
            logits, _ = model(x[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step >= 2:
                times.append(time.perf_counter() - t)
                cpus.append(time.process_time() - c)
        step_s, step_cpu = statistics.median(times), statistics.median(cpus)
        tokens = batch * CTX
        ms_tok = 1000 * step_s / tokens
        cpu_tok = 1000 * step_cpu / tokens             # CPU ms over all threads per token
        peak = status()
        row.update({"batch": batch, "tokens_per_step": tokens, "steps_measured": steps,
                    "step_s": round(step_s, 3), "step_cpu_s": round(step_cpu, 3),
                    "train_ms_per_token": round(ms_tok, 4), "train_cpu_ms_per_token": round(cpu_tok, 4),
                    "achieved_gflops": round(6 * params * tokens / step_s / 1e9, 1),
                    "epoch_260k_s": round(CORPUS_TOKENS * ms_tok / 1000, 1),
                    "epochs20_260k_s": round(20 * CORPUS_TOKENS * ms_tok / 1000, 1),
                    "epoch_260k_cpu_s": round(CORPUS_TOKENS * cpu_tok / 1000, 1),
                    "epochs20_260k_cpu_s": round(20 * CORPUS_TOKENS * cpu_tok / 1000, 1),
                    "peak_train_mb": mb(peak["VmHWM"])})
    write(row, args.out)


# ------------------------------------------------------------------------------------------------------- project
def project(args) -> None:
    """The low-end summary and the projection, from the rows measured above. Times on one core are taken from the CPU
    time of the process (process_time), because the machine was shared: wall times are kept beside them."""
    rows = rows_of(args.out)
    corpus_row, build = latest(rows, kind="corpus"), latest(rows, kind="teacher_build")
    minds = {r["name"]: r for r in rows if r.get("kind") == "mind"}
    trans = {}                                     # repeated runs: the least disturbed one (shortest wall time) counts
    for r in rows:
        if r.get("kind") == "transformer":
            key, speed = (r["size"], r["mode"], r["torch_threads"]), r.get("step_s", r.get("ms_per_token"))
            if key not in trans or speed < trans[key].get("step_s", trans[key].get("ms_per_token")):
                trans[key] = r
    usable = {"2GB": 2 * 1024 * 0.75, "4GB": 4 * 1024 * 0.75, "8GB": 8 * 1024 * 0.75}   # 1/4 left to the system
    summary = []
    for name, r in sorted(minds.items(), key=lambda kv: kv[1]["ties"]):
        need = max(r["peak_during_load_mb"], r["peak_during_replies_mb"], r["peak_during_watch_mb"])
        steady = max(r["rss_after_load_trimmed_mb"], r["peak_during_replies_mb"], r["peak_during_watch_mb"])
        summary.append({"system": "zihin " + name, "ties": r["ties"], "moments": r["moments"],
                        "need_mb": need, "need_mb_without_load_peak": steady,
                        "need_over_base_mb": round(need - r["rss_base_mb"], 1),
                        "reply_cpu_ms_median": r.get("reply_cpu_ms_median"), "reply_ms_median": r["reply_ms_median"],
                        "watch_cpu_ms_per_moment": r.get("watch_cpu_ms_per_moment"),
                        "learns_while_living": True, "fits": {k: need <= v for k, v in usable.items()}})
    for size in SIZES:
        inf, tr1, tr2 = trans.get((size, "infer", 1)), trans.get((size, "train", 1)), trans.get((size, "train", 2))
        if not inf:
            continue
        summary.append({"system": "transformer " + size, "params": inf["params"],
                        "need_mb_infer": inf["peak_infer_mb"], "need_infer_over_base_mb": inf["rss_infer_over_base_mb"],
                        "answer_cpu_ms": inf.get("answer_cpu_ms"), "answer_ms": inf["answer_ms"],
                        "need_mb_train": tr1["peak_train_mb"] if tr1 else None,
                        "fits_infer": {k: inf["peak_infer_mb"] <= v for k, v in usable.items()},
                        "fits_train": {k: tr1["peak_train_mb"] <= v for k, v in usable.items()} if tr1 else None,
                        "train_260k_1ep_cpu_s": tr1.get("epoch_260k_cpu_s") if tr1 else None,
                        "train_260k_20ep_cpu_s": tr1.get("epochs20_260k_cpu_s") if tr1 else None,
                        "train_260k_20ep_wall_s_1thread": tr1["epochs20_260k_s"] if tr1 else None,
                        "train_260k_20ep_wall_s_2threads": tr2["epochs20_260k_s"] if tr2 else None})
    write({"kind": "lowend_summary", "usable_fraction_of_ram": 0.75, "rows": summary}, args.out)
    if not minds or not build:
        return
    # Zihin's growth from the measured minds: RAM and file per moment; CPU ms per moment against ties (a power law)
    big = max(minds.values(), key=lambda r: r["ties"])
    over = (big["rss_after_load_trimmed_mb"] - big["rss_base_mb"]) * MiB
    ram_per_moment, ram_per_tie = over / big["moments"], over / big["ties"]
    ties_per_moment = big["ties"] / big["moments"]
    resaved = [r for r in minds.values() if "resave" in r]
    big_resaved = max(resaved, key=lambda r: r["resave"]["ties"]) if resaved else None
    file_per_tie = big_resaved["resave"]["file_bytes_per_tie"] if big_resaved else \
        build["file_mb"] * MiB / build["ties"]
    pts = [(math.log(r["ties"]), math.log(r.get("watch_cpu_ms_per_moment") or r["watch_ms_per_moment"]))
           for r in minds.values()]
    n = len(pts)
    mx, my = sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
    k = sum((x - mx) * (y - my) for x, y in pts) / max(1e-12, sum((x - mx) ** 2 for x, _ in pts))
    c = math.exp(my - k * mx)                      # CPU ms per moment = c * ties ** k

    def live_seconds(moments: float) -> float:     # the integral of c * (r * m) ** k over the moments lived
        return c * ties_per_moment ** k * moments ** (k + 1) / (k + 1) / 1000.0

    moments_per_pair = build["pair_moments"] / build["pairs"]
    tokens_per_pair = corpus_row.get("pair_tokens_mean", 0.0)
    tokens_per_moment = tokens_per_pair / moments_per_pair if moments_per_pair else 1.0
    targets = [("2M teacher examples", 2e6 * moments_per_pair), ("10M moments", 1e7), ("100M moments", 1e8),
               ("1B moments", 1e9)]
    out = []
    for label, moments in targets:
        tokens = moments * tokens_per_moment
        z = {"data": label, "moments": round(moments), "tokens_equivalent": round(tokens),
             "zihin": {"ties": round(moments * ties_per_moment), "ram_gb": round(moments * ram_per_moment / 2 ** 30, 1),
                       "file_gb": round(moments * ties_per_moment * file_per_tie / 2 ** 30, 1),
                       "cpu_ms_per_moment_at_end": round(c * (moments * ties_per_moment) ** k, 2),
                       "live_hours_one_core": round(live_seconds(moments) / 3600, 1)}}
        tr = {}
        for size in SIZES:
            t1, t2, inf = trans.get((size, "train", 1)), trans.get((size, "train", 2)), trans.get((size, "infer", 1))
            if not t1:
                continue
            per_tok = t1.get("train_cpu_ms_per_token") or t1["train_ms_per_token"]
            two = round(tokens * t2["train_ms_per_token"] / 3.6e6, 1) if t2 else None
            tr[size] = {"params": t1["params"], "flops_per_epoch": float("%.3g" % (6 * t1["params"] * tokens)),
                        "epoch_hours_one_core": round(tokens * per_tok / 3.6e6, 1),
                        "epochs20_hours_one_core": round(20 * tokens * per_tok / 3.6e6, 1),
                        "epoch_hours_2threads_wall": two,
                        "infer_ram_mb": inf["peak_infer_mb"] if inf else None, "train_ram_mb": t1["peak_train_mb"],
                        "cpu_ms_per_token": inf.get("cpu_ms_per_token") if inf else None}
        z["transformer"] = tr
        out.append(z)
    limit = {}
    for gb in (2, 4, 8, 15):
        room = (gb * 1024 * 0.75 - big["rss_base_mb"]) * MiB
        limit["%dGB" % gb] = {"moments": int(room / ram_per_moment),
                              "teacher_examples": int(room / ram_per_moment / moments_per_pair)}
    write({"kind": "projection", "basis": {"mind": big["name"], "ram_bytes_per_moment": round(ram_per_moment, 1),
                                           "ram_bytes_per_tie": round(ram_per_tie, 1),
                                           "ties_per_moment": round(ties_per_moment, 2),
                                           "file_bytes_per_tie": round(file_per_tie, 1),
                                           "file_basis": big_resaved["name"] if big_resaved else "teacher",
                                           "cpu_ms_per_moment_fit": {"c": float("%.4g" % c), "k": round(k, 3),
                                                                     "points": n},
                                           "moments_per_teacher_example": round(moments_per_pair, 2),
                                           "tokens_per_teacher_example": tokens_per_pair,
                                           "tokens_per_moment": round(tokens_per_moment, 3)},
           "mind_fits_until": limit, "rows": out}, args.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=("corpus", "teacher", "mind", "convert", "traced", "transformer", "project"))
    ap.add_argument("--data", default="private_data")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--name")
    ap.add_argument("--path")
    ap.add_argument("--breakdown", action="store_true")
    ap.add_argument("--resave", action="store_true", help="save the mind in today's layout under PRIVATE/duel/")
    ap.add_argument("--keep-resave", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--size", choices=tuple(SIZES))
    ap.add_argument("--mode", choices=("infer", "train"), default="infer")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--answers", type=int, default=5)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0)
    args = ap.parse_args()
    {"corpus": corpus, "teacher": teacher, "mind": mind_profile, "convert": convert, "traced": traced,
     "transformer": transformer, "project": project}[args.job](args)


if __name__ == "__main__":
    main()
