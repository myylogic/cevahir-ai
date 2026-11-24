# -*- coding: utf-8 -*-
"""M1: exact retraction (NOTES.md, section 4).

    python m1_retract.py [--corpus en-pydoc] [--out results/m1_retract.jsonl]

Store A reads every document of the corpus in order, each from an empty history, and keeps each document's ledger
(its situation events). Then one document is retracted from A. Store B reads every document except that one, in the
same order. The substance of A after retraction must be identical to B's, byte for byte (the same SHA-256).

The situation records are trust learned from A's own predictions while it lived. Their difference from B's is
counted, and so is what it changes in prediction: the bits of a probe text never read by either store.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import corpora  # noqa: E402
from living.cortex import Cortex  # noqa: E402


def live(docs: list[str], skip: int | None = None) -> tuple[Cortex, list[dict]]:
    cx = Cortex(depth=8)
    ledgers = []
    for i, d in enumerate(docs):
        ledger: dict = {}
        if i != skip:
            cx.stream(d, "", ledger=ledger)
        ledgers.append(ledger)
    return cx, ledgers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="en-pydoc")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m1_retract.jsonl"))
    a = ap.parse_args()
    docs = corpora.load(a.corpus)
    probe = subprocess.run(["git", "-C", REPO, "show", "%s:docs/research/living_evidence_mind_2026_11_24/PAPER.md"
                            % corpora.COMMIT], capture_output=True, check=True).stdout.decode("utf-8")
    rows = []
    for k in (0, len(docs) // 2, len(docs) - 1, max(range(len(docs)), key=lambda i: len(docs[i]))):
        t0 = time.time()
        A, ledgers = live(docs)
        t_live = time.time() - t0
        t1 = time.time()
        A.retract(docs[k], "", ledger=ledgers[k])
        t_retract = time.time() - t1
        B, _ = live(docs, skip=k)
        diff_sit = sum(1 for key in set(A.situations) | set(B.situations)
                       if A.situations.get(key) != B.situations.get(key))
        pa, _ = A.stream(probe, "", learn=False)
        pb, _ = B.stream(probe, "", learn=False)
        rows.append({"corpus": a.corpus, "documents": len(docs), "retracted": k, "retracted_chars": len(docs[k]),
                     "substance_identical": A.digest() == B.digest(), "substance_sha256": A.digest(),
                     "store_A": A.size(), "store_B": B.size(),
                     "situation_records_that_differ": diff_sit, "situation_records": len(B.situations),
                     "probe_chars": len(probe), "probe_bpc_A": round(pa / len(probe), 6),
                     "probe_bpc_B": round(pb / len(probe), 6),
                     "seconds_living_all": round(t_live, 1), "seconds_retracting": round(t_retract, 2)})
        print(json.dumps(rows[-1]), flush=True)
    try:
        Cortex(depth=8, grow=2).retract("x")
        deferred = "offered"
    except ValueError as e:
        deferred = "not offered: %s" % e
    rows.append({"grow_2": deferred})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
