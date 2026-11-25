# -*- coding: utf-8 -*-
"""M3: where Cevahir's memory and time go as it reads, and how much of a text the harmony already knows
(NOTES.md, 8.28).

    python m3_scale.py memory CORPUS --chars N [--variant full] [--ledger disk|memory|none] [--night] [--out ...]
    python m3_scale.py known CORPUS [--chars N] [--out ...]

``memory``: a new Cevahir reads the first N characters of the corpus stream (``Cevahir.read``, in ten pieces), with
Zihin's ledger on disk (``Cevahir(ledger=True)``: the file in a temporary folder, the latest moments in memory), in
memory (the ledger as a dict, as Cevahir kept it before 8.28), or none. Reported: the seconds, the resident memory
the process gained (before and after ``malloc_trim``), Zihin's units, ties and moments, the cortex's size, and the
bytes of every part of Cevahir reached by a walk of its objects (each object counted once; a tensor or an array by
the bytes it holds, capacity included), largest first, with the bytes per character; with ``--night``, the bytes
walked again after Zihin's night (``Mind.consolidate``). The cortex's store is held by the compiled core and is not
walked (its sizes are given).

``known``: the harmony alone reads the stream as Cevahir's harmony does, sentence by sentence (Zihin's sentence
split). Before it lives a sentence, a character is foreseen when the harmony believed it at the speaking line or
above (bits at most log2(12/7)), a word when all its characters were, and the sentence is known when the rule over
its words, (foreseen + 1/2) / (words + 1), reaches the speaking line. Reported: how many sentences of two words or
more are known, and by tenths of the stream: the share a gate on what the harmony already knows would leave out of
Zihin's life. Only counts are written, for every corpus.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import subprocess
import sys
import time
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import corpora  # noqa: E402

PAGE = os.sysconf("SC_PAGE_SIZE")


def code_version() -> str:
    """The commit of the code that wrote a row."""
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ""


CODE = code_version()


def resident_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * PAGE / 1e6


def deep_bytes(o, seen: set) -> int:
    """The bytes of ``o`` and of what it holds, every object (and every tensor storage) counted once."""
    import numpy as np
    import torch
    if id(o) in seen:
        return 0
    seen.add(id(o))
    if isinstance(o, torch.Tensor):
        st = o.untyped_storage()
        key = ("storage", st.data_ptr())
        if key in seen:
            return sys.getsizeof(o)
        seen.add(key)
        return st.nbytes()
    if isinstance(o, np.ndarray):
        return 100 if o.base is not None else o.nbytes + 100
    if isinstance(o, (array, bytes, bytearray, str, int, float, bool, type(None))):
        return sys.getsizeof(o)
    n = sys.getsizeof(o)
    if isinstance(o, dict):
        for k, v in o.items():
            n += deep_bytes(k, seen) + deep_bytes(v, seen)
    elif isinstance(o, (list, tuple, set, frozenset)):
        for v in o:
            n += deep_bytes(v, seen)
    else:
        for k in getattr(type(o), "__slots__", ()):
            if hasattr(o, k):
                n += deep_bytes(getattr(o, k), seen)
        if hasattr(o, "__dict__"):
            n += deep_bytes(o.__dict__, seen)
    return n


def walk(c) -> dict:
    """The bytes of every part of Cevahir and of Zihin (its substance, mind and session), the cortex's store apart."""
    parts, seen = {}, set()
    for owner, prefix in ((c.mind.S, "zihin.S."), (c.mind, "zihin.mind."), (c.session, "zihin.session."),
                          (c, "cevahir.")):
        names = sorted(set(getattr(type(owner), "__slots__", ())) | set(getattr(owner, "__dict__", {})))
        for k in names:
            if k == "cortex" or not hasattr(owner, k):
                continue
            parts[prefix + k] = deep_bytes(getattr(owner, k), seen)
        seen.add(id(owner))
    return parts


def memory(a) -> dict:
    import torch
    torch.set_num_threads(1)
    from living.cevahir import Cevahir
    text = corpora.stream(a.corpus)[:a.chars]
    base = resident_mb()
    c = Cevahir(variant=a.variant, ledger=a.ledger != "none")
    if a.ledger == "memory":                             # the ledger as Cevahir kept it before 8.28
        c.mind.S.log = {}
    t0 = time.time()
    step = max(1, len(text) // 10)
    for i in range(0, len(text), step):
        c.read(text[i:i + step])
    seconds = time.time() - t0
    c.flush()
    gc.collect()
    rss = resident_mb() - base
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    rss_trimmed = resident_mb() - base
    S = c.mind.S
    parts = walk(c)
    night = None
    if a.night:                                          # Zihin's night (Mind.consolidate), then walked again
        c.mind.consolidate()
        gc.collect()
        night = round(sum(walk(c).values()) / 1e6, 1)
    n = len(text)
    log = S.log
    row = {"what": "memory", "corpus": a.corpus, "code": CODE, "chars": n, "variant": a.variant, "ledger": a.ledger,
           "seconds": round(seconds, 1), "chars_per_second": round(n / seconds, 1),
           "resident_mb": round(rss, 1), "resident_trimmed_mb": round(rss_trimmed, 1),
           "walked_mb": round(sum(parts.values()) / 1e6, 1), "walked_after_night_mb": night,
           "zihin": {"moments": c.mind.t, "units": int(S.n_units), "ties": int(S.n_ties)},
           "ledger_disk_mb": round(log.disk_bytes() / 1e6, 1) if hasattr(log, "disk_bytes") else 0.0,
           "cortex": c.cortex.size(),
           "parts": [[k, v, round(v / n, 1)] for k, v in sorted(parts.items(), key=lambda x: -x[1]) if v >= 10000]}
    return row


def known(a) -> dict:
    from living.cevahir import CORTEX, SEPARATOR, TOKEN
    from living.evidence import SPEAKING_LINE
    from living.fast import FastCortex
    text = corpora.stream(a.corpus)
    if a.chars:
        text = text[:a.chars]
    harmony = FastCortex(voice=True, **dict(CORTEX))
    line = -math.log2(SPEAKING_LINE)
    pieces, pos = [], 0
    for m in SEPARATOR.finditer(text):
        pieces.append((pos, m.start()))
        pos = m.end()
    pieces.append((pos, len(text)))
    done, tenths = 0, [[0, 0] for _ in range(10)]
    words_known = words_all = 0
    for start, end in pieces:
        segment = text[done:end]                     # the separator before the sentence, then the sentence
        harmony.set_voice([], [], 0.0)
        _, bits = harmony.stream(segment, bits=True)  # each symbol's bits before it is learned
        tokens = [(start - done + m.start(), start - done + m.end()) for m in TOKEN.finditer(text[start:end])]
        if len(tokens) >= 2:
            foreseen = sum(1 for s, e in tokens if (bits[s:e] <= line).all())
            is_known = (foreseen + 0.5) / (len(tokens) + 1.0) >= SPEAKING_LINE
            tenths[min(9, 10 * start // max(1, len(text)))][0 if is_known else 1] += 1
            words_all += len(tokens)
            words_known += len(tokens) if is_known else 0
        done = end
    k = sum(t[0] for t in tenths)
    total = sum(t[0] + t[1] for t in tenths)
    return {"what": "known", "corpus": a.corpus, "code": CODE, "chars": len(text), "sentences": total,
            "known": k, "share_known": round(k / max(1, total), 4),
            "share_of_words_in_known": round(words_known / max(1, words_all), 4),
            "share_known_by_tenth": [round(t[0] / max(1, t[0] + t[1]), 4) for t in tenths]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["memory", "known"])
    ap.add_argument("corpus")
    ap.add_argument("--chars", type=int, default=0)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--ledger", default="disk", choices=["disk", "memory", "none"])
    ap.add_argument("--night", action="store_true", help="memory: walk again after Zihin's night")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m3_scale.jsonl"))
    a = ap.parse_args()
    if a.what == "memory" and not a.chars:
        ap.error("memory needs --chars")
    row = memory(a) if a.what == "memory" else known(a)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    show = {k: v for k, v in row.items() if k != "parts"}
    print(json.dumps(show, ensure_ascii=False))
    for k, v, per in row.get("parts", [])[:12]:
        print("  %-32s %9.2f MB %8.1f B/char" % (k, v / 1e6, per))


if __name__ == "__main__":
    main()
