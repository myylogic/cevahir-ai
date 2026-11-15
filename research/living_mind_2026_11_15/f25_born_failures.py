# -*- coding: utf-8 -*-
"""F25: one-shot evidence. Does a lag-1 tie that starts with the failures a full record would hold (setting
``born_failures``, design (a)) or a reading that lets rivals share one mass (setting ``rivals``, design (b)) change
what the mind generalises, keeps and retracts in the worlds of the first lives?

    python research/living_mind_2026_11_15/f25_born_failures.py --seeds 0 1 2 3 4 --conditions off a b --workers 2
    python research/living_mind_2026_11_15/f25_born_failures.py --summary

The protocol is that of ``research/living_mind_2026_11_14/run.py`` (imported, not changed): the same worlds,
schedules, probes and checkpoints (every 60 experiences of 600), variant "members" (F11, the reading of the RESULTS.md
tables) and the T9 retraction with its counterfactual life. The only difference between the conditions is the
setting: "off" is the code path of every earlier result (its rows must equal the archived records of the same lives,
checked by ``--summary`` where they exist), "a" sets ``born_failures``, "b" sets ``rivals``, "ab" both. ``--speaking``
runs the same worlds with the speaking configuration of ``zihin.__main__`` instead (F13 present-only learning, the
F14 floor, F16, F18, F19, F23 and exploration), whose life differs from the archived ones, so it is compared off/on
only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OLD = os.path.join(ROOT, "research", "living_mind_2026_11_14")
sys.path.insert(0, ROOT)
sys.path.insert(0, OLD)

import torch  # noqa: E402

from run import World, live, retraction  # noqa: E402

CONDITIONS = {"off": {}, "a": {"born_failures": True}, "b": {"rivals": True},
              "ab": {"born_failures": True, "rivals": True}}
SPEAKING = dict(bind_repeats=True, next_only_present=True, neg_floor=1.0 / 14, backoff=True, resemblance=True,
                race=True, situations=True, explore=True)
MARKS = (60, 120, 180, 300, 600)
OUT = os.path.join(HERE, "results", "f25_worlds.jsonl")


def one(task) -> dict:
    domain, seed, condition, n, speaking, t9 = task
    torch.set_num_threads(1)
    over = dict(CONDITIONS[condition])
    if speaking:
        over.update(SPEAKING)
    world = World.make(seed, domain)
    start = time.time()
    mind, rows, state = live(world, "members", n, **over)
    if t9:
        rows.append(retraction(world, "members", n, mind, state, **over))
    return {"domain": domain, "seed": seed, "condition": condition, "speaking": speaking, "n": n,
            "moments": mind.t, "units": mind.S.n_units, "ties": mind.S.n_ties, "seconds": time.time() - start,
            "rows": rows}


def held_out(row: dict):
    vals = [row[k] for k in ("held_told_rule_ok", "held_induced_ok") if k in row]
    return sum(vals) / len(vals) if vals else None


def at(record: dict, test: str, mark: int) -> dict:
    for r in record["rows"]:
        if r["test"] == test and r.get("at") == mark:
            return r
    return {}


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def paired(diffs) -> str:
    diffs = [d for d in diffs if d is not None and d == d]
    if len(diffs) < 2:
        return "n<2"
    mu = sum(diffs) / len(diffs)
    se = (sum((d - mu) ** 2 for d in diffs) / (len(diffs) - 1) / len(diffs)) ** 0.5
    return "%+.3f +- %.3f (%d/%d/%d)" % (mu, se, sum(d > 0 for d in diffs), sum(d == 0 for d in diffs),
                                         sum(d < 0 for d in diffs))


def archived(domain: str, seed: int) -> list:
    """The archived rows of the same life (variant members, 600 experiences), read only; [] when absent."""
    path = os.path.join(OLD, "results", "life_%s_%d_members.jsonl" % (domain, seed))
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8")]


# left out of the comparison: wall clock, the record's own head, and the ledger's size in bytes (the speed work writes
# fewer ledger records than when the archived lives were run; the substance is the same)
CLOCK = ("ms_per_moment", "ms_per_moment_max", "seed", "domain", "variant", "n", "log_bytes")


def reproduces(record: dict) -> tuple:
    """(identical, compared) rows of an "off" life against its archived record (wall-clock fields left out; T9's
    counterfactual numbers included)."""
    old = {(r["test"], r.get("at"), r.get("step"), str(r.get("fact"))): r for r in archived(record["domain"],
                                                                                          record["seed"])}
    same = compared = 0
    for r in record["rows"]:
        o = old.get((r["test"], r.get("at"), r.get("step"), str(r.get("fact"))))
        if o is None:
            continue
        compared += 1
        a = {k: v for k, v in r.items() if k not in CLOCK}
        b = {k: v for k, v in o.items() if k not in CLOCK}
        same += a == b
    return same, compared


def summary(path: str) -> None:
    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    for speaking in (False, True):
        group = [r for r in records if r["speaking"] == speaking]
        if not group:
            continue
        print("\n==== configuration: %s ====" % ("speaking" if speaking else "members (run.py)"))
        conds = [c for c in CONDITIONS if any(r["condition"] == c for r in group)]
        by = {(r["domain"], r["seed"], r["condition"]): r for r in group}
        if not speaking:
            for r in group:
                if r["condition"] == "off":
                    same, compared = reproduces(r)
                    if compared:
                        print("off reproduces the archived life %s seed %d: %d/%d rows identical" % (
                            r["domain"], r["seed"], same, compared))
        for domain in ("nitelik", "iliski"):
            seeds = sorted({s for d, s, c in by if d == domain})
            if not seeds:
                continue
            print("\n%s, seeds %s" % (domain, seeds))
            print("  held-out (mean of told-rule and induced-rule probes) by experiences lived")
            for c in conds:
                cells = []
                for m in MARKS:
                    cells.append("%d:%.3f" % (m, mean([held_out(at(by[(domain, s, c)], "T3", m))
                                                       for s in seeds if (domain, s, c) in by])))
                print("    %-3s %s" % (c, "  ".join(cells)))
            for c in conds:
                if c == "off":
                    continue
                for m in MARKS:
                    diffs = [held_out(at(by[(domain, s, c)], "T3", m)) - held_out(at(by[(domain, s, "off")], "T3", m))
                             if (domain, s, c) in by and (domain, s, "off") in by
                             and held_out(at(by[(domain, s, c)], "T3", m)) is not None
                             and held_out(at(by[(domain, s, "off")], "T3", m)) is not None else None
                             for s in seeds]
                    print("    %s - off at %3d: %s   per seed %s" % (c, m, paired(diffs), " ".join(
                        "%+.2f" % d if d is not None else "na" for d in diffs)))
            end = MARKS[-1]
            print("  at %d (mean over seeds)" % end)
            keys = [("T3", "held_told_rule_ok"), ("T3", "held_induced_ok"), ("T3", "member_ok"),
                    ("T3", "exception_ok"), ("T3", "exception_kept"), ("T5", "early_recall"),
                    ("T5", "early_evidence_identical"), ("T1", "c1"), ("T1", "e1"), ("T1", "c4"), ("T1", "e4"),
                    ("T1", "trust_told"), ("T2", "cat_jaccard"), ("T2", "categories"), ("T5", "ties"),
                    ("T5", "exact"), ("T5", "ms_per_moment")]
            print("    %-28s %s" % ("", "  ".join("%9s" % c for c in conds)))
            for test, key in keys:
                vals = []
                for c in conds:
                    vals.append(mean([at(by[(domain, s, c)], test, end).get(key) for s in seeds if (domain, s, c) in by]))
                print("    %-28s %s" % (test + " " + key, "  ".join("%9.3f" % v for v in vals)))
            print("  T1 in the stream: lies and corrections (mean over the rows of all seeds)")
            for c in conds:
                lies = [r for s in seeds if (domain, s, c) in by for r in by[(domain, s, c)]["rows"]
                        if r["test"] == "T1_lie"]
                for step in ("lie", "correction"):
                    rows = [r for r in lies if r["step"] == step]
                    if rows:
                        print("    %-3s %-10s n=%3d false_e %.3f false_c %.3f true_e %.3f true_c %.3f "
                              "true>1/2 %.2f true>false %.2f positive_untouched %s" % (
                                  c, step, len(rows), mean([r["false_e"] for r in rows]),
                                  mean([r["false_c"] for r in rows]), mean([r["true_e"] for r in rows]),
                                  mean([r["true_c"] for r in rows]), mean([float(r["true_e"] > 0.5) for r in rows]),
                                  mean([float(r["true_e"] > r["false_e"]) for r in rows]),
                                  "%d/%d" % (sum(bool(r["positive_untouched"]) for r in rows
                                                 if r["positive_untouched"] is not None),
                                             sum(r["positive_untouched"] is not None for r in rows))))
            print("  T9 exact retraction")
            for c in conds:
                t9 = [r for s in seeds if (domain, s, c) in by for r in by[(domain, s, c)]["rows"] if r["test"] == "T9"]
                if t9:
                    print("    %-3s lives %d: own evidence after 0 in %d, untouched identical %d, equals ledger %d, "
                          "other early facts identical %d, own_e after = counterfactual in %d, gap mean %.4f" % (
                              c, len(t9), sum(r["own_evidence_after"] == 0.0 for r in t9),
                              sum(bool(r["untouched_ties_identical"]) for r in t9),
                              sum(bool(r["equals_ledger_of_other_moments"]) for r in t9),
                              sum(bool(r["other_early_facts_identical"]) for r in t9),
                              sum(abs(r["own_e_after"] - r["counterfactual_own_e_after"]) < 1e-12 for r in t9),
                              mean([r["counterfactual_gap_mean"] for r in t9])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--domains", nargs="+", default=["nitelik", "iliski"])
    ap.add_argument("--conditions", nargs="+", default=["off", "a"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--speaking", action="store_true")
    ap.add_argument("--no-t9", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary(args.out)
        return
    torch.set_num_threads(1)
    tasks = [(d, s, c, args.n, args.speaking, not args.no_t9) for d in args.domains for s in args.seeds
             for c in args.conditions]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with Pool(args.workers) as pool, open(args.out, "a", encoding="utf-8") as handle:
        for rec in pool.imap_unordered(one, tasks):
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            print("%s seed %d %s%s: %d moments, %d ties, %.0f s" % (
                rec["domain"], rec["seed"], rec["condition"], " speaking" if rec["speaking"] else "", rec["moments"],
                rec["ties"], rec["seconds"]), flush=True)


if __name__ == "__main__":
    main()
