# -*- coding: utf-8 -*-
"""t_repair.py: what becomes of a loss the measure of what is kept found (notes 21.9, the second fault), when the
mind lives the earlier kinds again.

    python research/living_mind_2026_11_22/t_repair.py review [--seeds 5]

The curriculum of ``t_self.py`` (the same world, the same five stages, the same things), then a sixth stage, a
**review**: the five kinds read of the thing itself are asked again in the teacher's loop, four rounds each, of
the twenty things that were asked another kind of question in stages 2 to 5. The 8 things never asked about are
asked every kind on copies after stage 5 and after the review (right / wrong / silent).

Conditions: ``F57 + F58`` (the mind that sets itself) and ``set by hand`` (F41, F53, F54, F55; no trials).

Appends to ``results/repair.jsonl``.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import t_fault as T  # noqa: E402
import t_self as S  # noqa: E402

OUT = os.path.join(HERE, "results", "repair.jsonl")
DIRECT = S.KINDS[:5]


def review(args) -> None:
    rows, lines = [], ["== the curriculum, then a review of the five kinds read of the thing itself (20 rounds): %d "
                       "seeds ; 8 fresh things, right / wrong / silent ==" % args.seeds]
    for label, more in (S.CONDITIONS[2], S.CONDITIONS[3]):
        before, after, rounds = {}, {}, {}
        changes, by_seed = [], []
        for seed in range(args.seeds):
            s, things, truth, _ = S.life(seed, dict(more), measure=False)
            mind = s.mind

            def fresh(total):
                right = {}
                for q in S.KINDS:
                    marks = "".join(S.probe(s, "%s ne %s ?" % (x, q), "%s ." % truth[(x, q)])
                                    for x in things[40:48])
                    S.add(total.setdefault(q, {}), marks)
                    right[q] = marks.count("R")
                return right
            was = fresh(before)
            n = len(getattr(mind, "trial_changes", ()))
            marks = ""
            for i, x in enumerate(things[20:40]):
                q = DIRECT[i % len(DIRECT)]
                mark = S.teacher(s, "%s ne %s ?" % (x, q), "%s ." % truth[(x, q)])
                S.add(rounds.setdefault(q, {}), mark)
                marks += mark
            now = fresh(after)
            changes.append([{"t": c["t"], "way": S.name_of(c["way"]), "on": c["on"]}
                            for c in getattr(mind, "trial_changes", ())[n:]])
            by_seed.append({"review": marks, "right_before": was, "right_after": now})
        rows.append({"test": "repair_review", "condition": label, "seeds": args.seeds, "fresh_after_stage_5": before,
                     "fresh_after_the_review": after, "review_rounds": rounds, "changes_in_the_review": changes,
                     "lives": by_seed})
        lines.append("  %s" % label)
        lines.append("    after stage 5:    %s" % "  ".join("%s %s" % (q, S.rws(before[q])) for q in S.KINDS))
        lines.append("    after the review: %s" % "  ".join("%s %s" % (q, S.rws(after[q])) for q in S.KINDS))
        lines.append("    the rounds of the review: %s ; changes of configuration in it: %s" % (
            "  ".join("%s %s" % (q, S.rws(rounds[q])) for q in DIRECT),
            [c for of_life in changes for c in of_life] or "none"))
        lost = [(seed, q, life["right_before"][q], life["right_after"][q]) for seed, life in enumerate(by_seed)
                for q in S.KINDS if life["right_before"][q] < 8 or life["right_after"][q] < 8]
        lines.append("    lives and kinds not 8 of 8 (seed, kind, right before the review, after): %s" % (lost or "none"))
    T.finish(args, rows, lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["review"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
    review(args)


if __name__ == "__main__":
    main()
