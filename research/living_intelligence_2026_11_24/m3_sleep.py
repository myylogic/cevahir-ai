# -*- coding: utf-8 -*-
"""M3: the hippocampus's sleep, measured (NOTES.md, 8).

    python m3_sleep.py [CORPUS ...] [--every 100000] [--out results/m3_sleep.jsonl]

The harmony (M2's configuration) lives a corpus stream in order, as m2_harmony.py does. With sleep, after every
``--every`` characters of the first nine tenths it sleeps (``FastCortex.sleep(2)``: every context met fewer than two
times is forgotten, with the longer contexts extending it; what recurred stays exactly). The last tenth is lived
without a further sleep. Reported, with and without sleep: bits per character over the whole stream and the last
tenth, and the store (nodes, entries) at the end and at its largest. For the author's private corpus only counts are
written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import corpora  # noqa: E402
from living.cevahir import CORTEX  # noqa: E402
from living.fast import FastCortex  # noqa: E402


def live(s: str, cut: int, every: int) -> dict:
    m = FastCortex(**CORTEX)
    t0 = time.time()
    total, pos, peak_nodes, peak_entries, nights, forgotten = 0.0, 0, 0, 0, 0, 0
    while pos < cut:
        nxt = min(cut, pos + (every or cut))
        b, _ = m.stream(s[pos:nxt])
        total += b
        pos = nxt
        size = m.size()
        peak_nodes, peak_entries = max(peak_nodes, size["nodes"]), max(peak_entries, size["entries"])
        if every:
            forgotten += m.sleep(2)
            nights += 1
    last, _ = m.stream(s[cut:])
    size = m.size()
    return {"bpc_all": round((total + last) / len(s), 4), "bpc_last_tenth": round(last / (len(s) - cut), 4),
            "nodes_end": size["nodes"], "entries_end": size["entries"],
            "nodes_peak": max(peak_nodes, size["nodes"]), "entries_peak": max(peak_entries, size["entries"]),
            "nights": nights, "forgotten": forgotten, "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="*", default=["tr-lessons", "en-repo", "en-pydoc", "tr-private", "code-pystdlib"])
    ap.add_argument("--every", type=int, default=100000)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m3_sleep.jsonl"))
    a = ap.parse_args()
    for name in a.corpora:
        try:
            s = corpora.stream(name)
        except Exception as exc:                                  # noqa: BLE001 (the private corpus may be absent)
            print(name, "not available:", exc)
            continue
        cut = int(len(s) * 0.9)
        every = min(a.every, max(1, cut // 4))                    # at least four nights on a small text
        row = {"corpus": name, "chars": len(s), "split": cut, "every": every,
               "awake": live(s, cut, 0), "sleeping": live(s, cut, every)}
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
