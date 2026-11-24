# -*- coding: utf-8 -*-
"""M3: a study session (etüt) with Cevahir, as research/living_mind_2026_11_22/etut.py holds one with Zihin
(NOTES.md, 8).

    python m3_etut.py --lessons FILE ... [--rounds 3] [--save FOLDER] [--report out.json]

Each lesson is a text file: one line a sentence (told by the teacher) or ``question<TAB>answer`` (a dialogue
watched); ``#`` lines are notes; ``?question<TAB>answer`` lines are never taught, only asked at the end. A lesson is
studied as an etüt, with what Cevahir has for it:

- taught: every sentence read (``Cevahir.tell``) and every dialogue watched (``Cevahir.watch``) once;
- asked: every dialogue's question, in a shuffled order, answered by Cevahir itself in conversation
  (``Cevahir.converse``: the one layer, Zihin's plan said by the prefrontal decision). A right answer is confirmed
  (``feedback(True)``: the amygdala's record, the turn lived). A wrong or missing one is judged so
  (``feedback(False)``: the wrong answer is not lived as said rightly), the attempt is forgotten exactly
  (``Cevahir.forget``: Zihin's moments by its ledger, the core's counts and episode text), and the right answer is
  shown (the dialogue watched again);
- asked again, round after round, until every answer is right or ``--rounds`` run out;
- the examination lines asked last, each by a process that is the mind as it stands.

An answer is right when it begins with the whole answer's words (closing marks apart), as in etut.py; one that goes
on after it is taken back and shown again too. ``--save``: Cevahir kept in a folder (Cevahir.save). ``--report``:
counts only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "research", "living_mind_2026_11_22"))
from living.cevahir import Cevahir  # noqa: E402


def study(c: Cevahir, path: str, rounds: int, seed: int) -> dict:
    from etut import lesson_lines
    from zihin.__main__ import FINAL
    taught, asked, exam = lesson_lines(path)
    name = os.path.splitext(os.path.basename(path))[0]
    t0 = time.time()
    for line in taught:
        if "\t" in line:
            q, a = line.split("\t", 1)
            c.watch(q.strip(), a.strip())
        else:
            c.tell(line.strip())
    s = c.session

    def trimmed(text):
        words = s.words(text)
        while words and words[-1] in FINAL:
            words = words[:-1]
        return words

    def judge(said, answer):
        want, words = trimmed(answer), s.words(said)
        right = bool(said) and words[:len(want)] == want
        return right, right and any(w not in FINAL for w in words[len(want):])

    rng = random.Random(seed)
    scores, went_on = [], []
    for _ in range(rounds):
        order = list(asked)
        rng.shuffle(order)
        right = longer = 0
        for q, a in order:
            for _ in range(2):                                   # a silence between two questions
                c.mind.step([])
            said = c.converse(q, limit=40)
            ok, went = judge(said, a)
            if ok and not went:
                right += 1
                c.feedback(True)
            else:
                if ok:
                    right += 1
                    longer += 1
                c.feedback(False)
                c.forget(len(c.entries) - 1)                      # the attempt, taken back exactly
                s.open_turn, s.answered = False, None
                c.watch(q, a)                                    # and the right answer shown
        scores.append(right)
        went_on.append(longer)
        if right == len(asked):
            break
    passed = 0
    if exam:
        said = c.answers([q for q, _ in exam], way="cevahir", limit=40, workers=2)
        passed = sum(judge(x, a)[0] for x, (_, a) in zip(said, exam))
    return {"lesson": name, "lines": len(taught), "asked": len(asked), "rounds": scores, "went_on": went_on,
            "exam": [passed, len(exam)], "seconds": round(time.time() - t0, 1), "moments": c.mind.t}


def review(c: Cevahir, paths) -> dict:
    """Every dialogue of the lessons asked again once all were studied, each by a process that is the mind as it
    stands: what stayed."""
    from etut import lesson_lines
    from zihin.__main__ import FINAL
    out = {}
    for path in paths:
        _, asked, _ = lesson_lines(path)
        if not asked:
            continue
        said = c.answers([q for q, _ in asked], way="cevahir", limit=40, workers=2)
        right = 0
        for x, (_, a) in zip(said, asked):
            want = [w for w in c.session.words(a)]
            while want and want[-1] in FINAL:
                want = want[:-1]
            right += bool(x) and c.session.words(x)[:len(want)] == want
        out[os.path.splitext(os.path.basename(path))[0]] = [right, len(asked)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lessons", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="")
    ap.add_argument("--report", default="")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    c = Cevahir(variant="full", ledger=True, seed=a.seed)
    t0 = time.time()
    rows = []
    for i, path in enumerate(a.lessons):
        row = study(c, path, a.rounds, a.seed + i)
        rows.append(row)
        print("  %s: rounds %s of %d, examination %d/%d, %.0f s" % (
            row["lesson"], row["rounds"], row["asked"], row["exam"][0], row["exam"][1], row["seconds"]), flush=True)
    rev = review(c, a.lessons)
    report = {"who": "cevahir", "lessons": rows, "review": rev, "moments": c.mind.t, "core": c.cortex.size(),
              "seconds": round(time.time() - t0, 1)}
    if a.save:
        report["saved"] = c.save(a.save)["files"]
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "lessons"}, ensure_ascii=False)[:2000], flush=True)


if __name__ == "__main__":
    main()
