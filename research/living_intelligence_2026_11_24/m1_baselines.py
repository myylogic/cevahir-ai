# -*- coding: utf-8 -*-
"""M1 baselines: general-purpose compressors on the same streams (NOTES.md, section 2).

    python m1_baselines.py [--out results/m1_baselines.jsonl]

A compressor that adapts as it reads is also a learner that lives through the text once. Its compressed size in bits,
divided by the number of characters, is comparable to the cortex's prequential bits per character. Its cost of the
last 10 %, given the first 90 %, is size(all) - size(first 90 %); this is comparable to the cortex's living test bits.

The compressors are gzip -9, bzip2 -9, xz -9e (Python's zlib, bz2, lzma) and PPMd variant I, the PPM of 7-Zip
(pyppmd 1.3.1), at orders 4, 8 and 16 with 1 GB of memory.
"""
from __future__ import annotations

import argparse
import bz2
import json
import lzma
import os
import sys
import zlib

import pyppmd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import corpora  # noqa: E402

COMPRESSORS = {
    "gzip-9": lambda b: len(zlib.compress(b, 9)),
    "bzip2-9": lambda b: len(bz2.compress(b, 9)),
    "xz-9e": lambda b: len(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME)),
    "ppmd-o4": lambda b: len(pyppmd.compress(b, max_order=4, mem_size=1 << 30)),
    "ppmd-o8": lambda b: len(pyppmd.compress(b, max_order=8, mem_size=1 << 30)),
    "ppmd-o16": lambda b: len(pyppmd.compress(b, max_order=16, mem_size=1 << 30)),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m1_baselines.jsonl"))
    a = ap.parse_args()
    rows = []
    for name in corpora.CORPORA:
        s = corpora.stream(name)
        if not s:
            continue
        split = int(len(s) * 0.9)
        whole, first = s.encode("utf-8"), s[:split].encode("utf-8")
        for comp, size in COMPRESSORS.items():
            all_bytes = size(whole)
            first_bytes = size(first)
            rows.append({"corpus": name, "compressor": comp, "chars": len(s),
                         "bpc_all": round(all_bytes * 8 / len(s), 4),
                         "bpc_test_living": round((all_bytes - first_bytes) * 8 / (len(s) - split), 4)})
            print(json.dumps(rows[-1]), flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
