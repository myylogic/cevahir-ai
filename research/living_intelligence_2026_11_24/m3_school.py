# -*- coding: utf-8 -*-
"""M3: the school of the earlier study (research/living_mind_2026_11_22/t_school.py, section 41), taught to
Cevahir and examined on what was never asked (NOTES.md, 8).

    python m3_school.py [--seeds 3] [--steps 8000] [--workers 2]

The teacher's seven subjects and the course, unchanged (``t_school.course``: every lesson of every subject, three
times, each time in one shuffled order): told sentences (``Cevahir.tell``) and watched dialogues (``Cevahir.watch``).
Cevahir's Zihin is the whole mind with a window of 12, as there. The examination (``t_school.subjects``) asks only
what was never asked, of three kinds (deduction, composition, relation) and a control nothing lived can answer (a sum
never added); what was asked in the lessons is asked again (recall). Each question is answered by a process that is
the mind as it stands, in the three ways (zihin, harmony, cevahir). The Transformer is t_school.py's own
(``run_transformer``: a decoder trained from scratch on the same lessons, word units, ``--steps`` steps; its
exactness on the lessons' questions is reported). An answer is right when its words, without the closing mark, are
the teacher's.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "research", "living_mind_2026_11_22"))
from living.cevahir import Cevahir  # noqa: E402


def own_record(c) -> dict:
    """Cevahir's record of Zihin alone: [right, all], and the claims by how far Zihin believed them (in twentieths)."""
    bins = {}
    for e, won in getattr(c, "za_claims", []):
        cell = bins.setdefault("%.2f" % (int(e * 20) / 20.0), [0, 0])
        cell[0] += won
        cell[1] += 1
    return {"za": [c.za_won, c.za_all], "za_by_belief": dict(sorted(bins.items()))}


def parse_switches(text: str) -> dict:
    """``--set`` as switches: "inherit=1,alone_now=1"."""
    return {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, text.split(",")))}


def run(seed: int, steps: int, workers: int, switches: Optional[dict] = None) -> dict:
    import torch
    import t_school as S
    torch.set_num_threads(1)
    c = Cevahir(variant="full", window=S.WINDOW, seed=seed, switches=switches)
    t0 = time.process_time()
    for lesson in S.course(seed):
        if lesson[0] == "tell":
            c.tell(lesson[1], tone=".")
        else:
            c.watch(lesson[1], lesson[2])
    c.session.close_turn()
    lived = time.process_time() - t0
    exams = S.subjects(seed)
    jobs = [(name, kind, q, ans) for name, (_, exam) in exams.items() for kind, q, ans in exam]
    t1 = time.time()
    said = c.answers([q for _, _, q, _ in jobs], way="every", limit=20, workers=workers)
    answered = time.time() - t1
    scores = {}
    for (name, kind, q, ans), got in zip(jobs, said):
        want = c.session.words(ans)
        cell = scores.setdefault(name, {}).setdefault(kind, {"zihin": 0, "harmony": 0, "cevahir": 0, "asked": 0})
        cell["asked"] += 1
        for way in ("zihin", "harmony", "cevahir"):
            cell[way] += got[way] != "(sessiz)" and c.session.body(got[way]) == want
    row = {"test": "m3_school", "code": CODE, "seed": seed, "switches": switches or {}, **own_record(c),
           "lessons": len(S.course(seed)), "zihin_moments": c.mind.t,
           "learn_cpu_s": round(lived, 1), "answer_wall_s": round(answered, 1), "workers": workers}
    if steps > 0:
        tf = S.run_transformer(seed, steps)
        for name, rows in tf["answers"].items():
            for kind, q, ans, got in rows:
                cell = scores[name][kind]
                cell["transformer"] = cell.get("transformer", 0) + (got == ans)
        row["transformer_run"] = {k: tf[k] for k in ("cpu_s", "train_exact", "train_n", "loss")}
    row["scores"] = scores
    return row


def code_version() -> str:
    """The commit of the code that wrote a row."""
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ""


CODE = code_version()                                   # the code that runs, read when it starts

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--set", default="", help="switches, e.g. alone_now=1")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m3_school.jsonl"))
    a = ap.parse_args()
    for seed in range(a.seeds):
        row = run(seed, a.steps, a.workers, parse_switches(a.set))
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
