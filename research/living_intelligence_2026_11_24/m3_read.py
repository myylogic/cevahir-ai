# -*- coding: utf-8 -*-
"""M3: Cevahir living through a text, with Zihin as a voice of the harmony (NOTES.md, 8).

    python m3_read.py CORPUS [--variant speaking] [--record] [--out results/m3_read.jsonl]

The corpus stream of m2_harmony.py, read as Zihin reads a document (Cevahir.read: the harmony lives every
character; Zihin lives each sentence of at least two words, its end and a pause, and before each of its words its
expectations are the voice of the harmony). The same stream is lived by the harmony alone (M2's configuration, the
voice off) for the comparison. Reported: bits per character over the whole stream and over its last tenth, both
living (Cevahir learns as it reads; Zihin cannot read without living, so there is no frozen measure here), the
store, the time. For the author's private corpus only counts are written.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import corpora  # noqa: E402
from living.cevahir import CORTEX, Cevahir  # noqa: E402
from living.fast import FastCortex  # noqa: E402


def code_version() -> str:
    """The commit of the code that wrote a row."""
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--variant", default="speaking")
    ap.add_argument("--record", action="store_true", help="also keep the record of disagreements (slower)")
    ap.add_argument("--chars", type=int, default=0, help="only the first N characters (0: all)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m3_read.jsonl"))
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    s = corpora.stream(a.corpus)
    if a.chars:
        s = s[:a.chars]
    split = int(len(s) * 0.9)
    cut = s.rfind("\n", 0, split) + 1 or split                   # the last tenth begins at a line
    row = {"corpus": a.corpus, "code": code_version(), "chars": len(s), "split": cut, "variant": a.variant, "record": a.record}
    m2 = FastCortex(**CORTEX)
    t0 = time.time()
    b1, _ = m2.stream(s[:cut])
    b2, _ = m2.stream(s[cut:])
    row["harmony"] = {"bpc_all": round((b1 + b2) / len(s), 4), "bpc_last_tenth": round(b2 / (len(s) - cut), 4),
                      "seconds": round(time.time() - t0, 1)}
    c = Cevahir(variant=a.variant, record=a.record)
    t0 = time.time()
    code = a.corpus.startswith("code")
    c1 = c.read(s[:cut], code=code)
    t1 = time.time()
    c2 = c.read(s[cut:], code=code)
    row["cevahir"] = {"bpc_all": round((c1 + c2) / len(s), 4), "bpc_last_tenth": round(c2 / (len(s) - cut), 4),
                      "seconds": round(time.time() - t0, 1), "seconds_last_tenth": round(time.time() - t1, 1),
                      "zihin_moments": c.mind.t, "voiced_tokens": c.counts["voiced"],
                      "unaligned_utterances": c.counts["unaligned"], "cortex": c.cortex.size()}
    if a.record:
        row["disagreements"] = {str(k): [c.dis_won[k], c.dis_all[k]] for k in sorted(c.dis_all)}
        row["alone"] = {str(k): [c.alone_right[k], c.alone_all[k]] for k in sorted(c.alone_all)}
    row["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    if a.corpus in corpora.PUBLIC and not a.chars:
        row["sha256"] = corpora.describe(a.corpus)["sha256"]
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
