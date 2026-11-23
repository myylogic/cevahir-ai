# -*- coding: utf-8 -*-
"""t_cost.py: what a saved mind costs to answer with and to listen with, by two versions of the code, and whether
the two do exactly the same (section 30).

    python research/living_mind_2026_11_22/t_cost.py answer --mind MIND --data FOLDER --codes OLD NEW [--n 6]
    python research/living_mind_2026_11_22/t_cost.py listen --mind MIND --data FOLDER --codes OLD NEW [--n 30]
                                                            [--set silent_claims=1,relied_on=1]

``MIND``: a mind saved by the program (``zihin.persist.save``). ``FOLDER``: the folder whose ``belgeler`` holds the
documents the cues and sentences are drawn from (``recall_once.documents``). ``--codes``: folders that each hold a
``zihin`` package (and this ``research`` folder); every one is run in a process of its own on the same mind, the
same cues (seed 0) or sentences (seed 1).

``answer``: n thoughts from two-word cues (the inner voice, ``dusunce.think``: what ``recall_once.py`` measures).
``listen``: n sentences listened to as the program reads a file (``Session.read_path``'s steps).

Printed and appended to ``results/cost.jsonl`` for every code: the md5 of its ``zihin/*.py``, wall seconds, the
seconds of the process's own processor time (what to compare when the machine is busy), moments lived, and
whether everything it did is the same as the first code did: the words of every thought (answer) and the digest
of the store at the end (both). Nothing of the documents or of what was thought is printed or kept: timings,
counts and one yes or no.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "cost.jsonl")


def code_md5(code: str) -> str:
    h = hashlib.md5()
    folder = os.path.join(code, "zihin")
    for name in sorted(n for n in os.listdir(folder) if n.endswith(".py")):
        with open(os.path.join(folder, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def one(args) -> None:
    """Run in a process of its own with one code (``--code``): prints one line of JSON."""
    code = os.path.abspath(args.code)
    sys.path.insert(0, code)
    sys.path.insert(0, os.path.join(code, "research", "living_mind_2026_11_15"))
    import random
    import torch
    torch.set_num_threads(1)
    import recall_once as RO
    from zihin import IMITATED
    from zihin.__main__ import Session, read_document, sentences
    from zihin.dusunce import think
    from zihin.persist import load
    mind = load(args.mind, ledger=False)
    for item in [kv for kv in args.set.split(",") if kv]:
        name, value = item.split("=")
        if not hasattr(mind.cfg, name):
            raise SystemExit("no such switch: %s" % name)
        setattr(mind.cfg, name, value == "1")
    session = Session(mind)
    said = hashlib.md5()
    each = []
    moments = mind.t
    if args.what == "answer":
        sample, _ = RO.documents(args.data, 2, session)
        picked = random.Random(0).sample(sample, 200)[:args.n]
        t0, c0 = time.time(), time.process_time()
        for body in picked:
            t1 = time.process_time()
            thought, _ = think(session, " ".join(body[:2]), limit=60)
            each.append(round(time.process_time() - t1, 1))
            said.update((" ".join(thought) + "|").encode("utf-8"))
    else:
        folder = os.path.join(args.data, "belgeler")
        lines = []
        for path in sorted(os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs):
            try:
                lines += list(sentences(read_document(path), session.dil))
            except Exception:                              # noqa: BLE001 (not a text file)
                continue
        picked = random.Random(1).sample(lines, args.n)
        session.company(None)
        t0, c0 = time.time(), time.process_time()
        for sentence in picked:
            session.listen(sentence, IMITATED)
            session.end_of_turn(IMITATED, tone=session.tone(sentence))
            mind.step([])
    print(json.dumps({"wall_s": round(time.time() - t0, 1), "cpu_s": round(time.process_time() - c0, 1),
                      "moments": mind.t - moments, "each_s": each, "ties": mind.S.n_ties, "mind_moments": moments,
                      "did": said.hexdigest() + mind.S.digest()}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["answer", "listen"])
    ap.add_argument("--mind", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--codes", nargs="+", default=[])
    ap.add_argument("--code")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--set", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    args.n = args.n or (6 if args.what == "answer" else 30)
    if args.code:
        one(args)
        return
    rows, first = [], None
    for code in args.codes:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        out = subprocess.run([sys.executable, os.path.abspath(__file__), args.what, "--mind", args.mind, "--data",
                              args.data, "--code", code, "--n", str(args.n), "--set", args.set],
                             capture_output=True, text=True, env=env, check=True)
        got = json.loads(out.stdout.strip().split("\n")[-1])
        first = first or got["did"]
        row = {"test": "cost_" + args.what, "code_md5": code_md5(code), "n": args.n, "switches_set": args.set,
               "mind_moments": got["mind_moments"], "ties": got["ties"], "moments": got["moments"],
               "wall_s": got["wall_s"], "cpu_s": got["cpu_s"], "each_s": got["each_s"],
               "same_as_the_first_code": got["did"] == first, "note": args.note}
        if args.what == "listen":
            row["ms_a_moment_cpu"] = round(1000.0 * got["cpu_s"] / max(1, got["moments"]), 1)
        rows.append(row)
        print("%s code %s: %d %s, %.1f s wall, %.1f s of its own processor time%s ; the same as the first code: %s" % (
            args.what, row["code_md5"][:8], args.n, "thoughts" if args.what == "answer" else "sentences",
            row["wall_s"], row["cpu_s"],
            " (%.1f ms a moment, %d moments)" % (row["ms_a_moment_cpu"], row["moments"]) if args.what == "listen"
            else " (each: %s)" % row["each_s"], "yes" if row["same_as_the_first_code"] else "NO"), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
