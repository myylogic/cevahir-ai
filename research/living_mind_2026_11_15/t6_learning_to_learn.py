# -*- coding: utf-8 -*-
"""T6 again, with learning to learn (F24): does a mind that has lived one domain learn a second one faster because of
what it learned about learning, and is the first domain kept?

    python research/living_mind_2026_11_15/t6_learning_to_learn.py --seeds 0 1 2 3 4 --flags on --workers 2
    python research/living_mind_2026_11_15/t6_learning_to_learn.py --seeds 0 --orders nitelik:iliski --flags off
    python research/living_mind_2026_11_15/t6_learning_to_learn.py --summary

With the setting off the code path is the one of the first T6, so the summary reads the first T6 records
(``research/living_mind_2026_11_14/results/t6_transfer.jsonl``, read only) as the setting-off baseline wherever this
file holds no own off run, and compares an own off run row by row with them (reproduction check).

The protocol of ``research/living_mind_2026_11_14/t6_transfer.py`` (imported, not changed): for each seed and each
order (A = nitelik then B = iliski, and the reverse) two minds meet the identical B stream, a fresh mind and the mind
that first lived A (same seed, 600 experiences each, independent vocabulary). The same worlds, schedules and probes
(``run.py``) are used; the only differences are the setting ``voices`` (F24, "off" reproduces the earlier T6) and
denser checkpoints (every 30 experiences instead of 60; probes run on clones and do not touch the life, so the rows at
60, 120, ... are those of the earlier protocol). Also recorded: what the mind learned about kinds of voices (held and
failed outcomes per kind) after A and after B.

Single-domain check: the fresh minds ARE single-domain lives, so on/off of the fresh rows compares T1 (conviction by
hearings), T2 (categories), T3 (held-out, members, exceptions) and T5 (early facts, integrity) within one domain.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OLD = os.path.join(ROOT, "research", "living_mind_2026_11_14")
sys.path.insert(0, ROOT)
sys.path.insert(0, OLD)

import torch  # noqa: E402

from run import World, checkpoint, live  # noqa: E402
from zihin.read import VOICE_ALL, VOICE_CHANNEL, voice_credibility  # noqa: E402
from zihin.store import EXPECT, W_NEG, W_POS  # noqa: E402

KEEP = ("T1", "T2", "T3", "T5")
MARKS = (30, 60, 90, 120, 180, 300, 600)


def kinds(mind) -> list:
    """What the mind has lived about voices: [many, agree, held, failed, credibility] per kind, and the row of every
    voice as [0, 0, held, failed, expectation]."""
    if not mind.ch.has(VOICE_CHANNEL):
        return []
    S, cid, out = mind.S, mind.ch.id(VOICE_CHANNEL), []
    for kind in mind.ch.symbols[cid]:
        tie = S.find_tie(mind.unit_of(VOICE_CHANNEL, kind), mind.holds, EXPECT, 0)
        held = float(S.t_w[W_POS, tie].sum()) if tie >= 0 else 0.0
        failed = float(S.t_w[W_NEG, tie].sum()) if tie >= 0 else 0.0
        if kind == VOICE_ALL:
            out.append([0, 0, held, failed, (held + 0.5) / (held + failed + 1.0)])
        else:
            out.append([kind[0], kind[1], held, failed, voice_credibility(mind, kind)])
    return sorted(out)


def one(task) -> dict:
    seed, first, second, flag, n_a, n_b, checkpoints = task
    torch.set_num_threads(1)
    over = {"voices": flag == "on"}
    a, b = World.make(seed, first), World.make(seed, second)
    fresh_mind, fresh_rows, _ = live(b, "members", n_b, checkpoints=checkpoints, **over)
    mind, a_rows, a_state = live(a, "members", n_a, checkpoints=checkpoints, **over)
    a_end = [r for r in a_rows if r.get("at") == n_a and r["test"] in ("T3", "T5")]
    after_a = kinds(mind)
    _, exp_rows, _ = live(b, "members", n_b, mind=mind, checkpoints=checkpoints, **over)
    a_after = checkpoint(a, mind, a_state, n_a)
    return {"seed": seed, "first": first, "second": second, "flag": flag, "variant": "members", "n_a": n_a,
            "n_b": n_b, "fresh": [r for r in fresh_rows if r["test"] in KEEP],
            "experienced": [r for r in exp_rows if r["test"] in KEEP],
            "a_before_b": a_end, "a_after_b": [r for r in a_after if r["test"] in ("T3", "T5")],
            "kinds_after_a": after_a, "kinds_after_b_fresh": kinds(fresh_mind), "kinds_after_b_experienced": kinds(mind)}


def held_out(row: dict):
    """Mean of the told-rule and induced-rule held-out shares (the measure of the T6 table), None without probes."""
    vals = [row[k] for k in ("held_told_rule_ok", "held_induced_ok") if k in row]
    return sum(vals) / len(vals) if vals else None


def curve(record: dict, cond: str) -> dict:
    return {r["at"]: held_out(r) for r in record[cond] if r["test"] == "T3"}


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


PROTECTED = os.path.join(OLD, "results", "t6_transfer.jsonl")   # the first T6 (setting off), read only


def _same_rows(mine: list, theirs: list) -> tuple:
    """(identical, compared) rows at the checkpoints both records have; wall-clock fields left out."""
    clock = ("ms_per_moment", "ms_per_moment_max")
    index = {(r["test"], r["at"]): r for r in theirs}
    same = compared = 0
    for r in mine:
        other = index.get((r["test"], r["at"]))
        if other is not None:
            compared += 1
            same += {k: v for k, v in r.items() if k not in clock} == {k: v for k, v in other.items() if k not in clock}
    return same, compared


def load(path: str, protected_off: bool = True) -> list:
    """Own records; with ``protected_off`` the first T6 records serve as the setting-off baseline for every (seed,
    order) without an own off run, and an own off run is compared row by row with the first T6 record."""
    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    if not protected_off or not os.path.exists(PROTECTED):
        return records
    own_off = {(r["seed"], r["first"]): r for r in records if r["flag"] == "off"}
    for line in open(PROTECTED, encoding="utf-8"):
        old = json.loads(line)
        mine = own_off.get((old["seed"], old["first"]))
        if mine is None:
            old["flag"], old["protected"] = "off", True
            records.append(old)
            continue
        report = []
        for part in ("fresh", "experienced", "a_after_b"):
            same, compared = _same_rows(mine[part], old[part])
            report.append("%s %d/%d" % (part, same, compared))
        print("off reproduces the first T6 (seed %d, %s -> %s), identical rows: %s" % (
            old["seed"], old["first"], old["second"], ", ".join(report)))
    return records


def summary(path: str, protected_off: bool = True) -> None:
    records = load(path, protected_off)
    seeds = sorted({r["seed"] for r in records})
    print("records:", len(records), "seeds:", seeds, "(off from the first T6 where marked: %d)" % sum(
        bool(r.get("protected")) for r in records))
    print("\nHeld-out in B (mean of told-rule and induced-rule), mean over seeds (n lives if fewer have the mark)")
    for first, second in (("nitelik", "iliski"), ("iliski", "nitelik")):
        for flag in ("off", "on"):
            group = [r for r in records if (r["first"], r["second"], r["flag"]) == (first, second, flag)]
            if not group:
                continue
            group.sort(key=lambda r: r["seed"])
            for cond in ("fresh", "experienced"):
                curves = [curve(r, cond) for r in group]
                cells = []
                for m in MARKS:
                    have = [c[m] for c in curves if c.get(m) is not None]
                    cells.append("%d:%s" % (m, "%.2f" % mean(have) + ("(%d)" % len(have) if len(have) < len(group)
                                                                      else "") if have else "na"))
                print("%-8s->%-8s %-3s %-11s %s" % (first, second, flag, cond, "  ".join(cells)))
            for m in (60, 120, 180, 300, 600):
                diffs = []
                for r in group:
                    f, e = curve(r, "fresh").get(m), curve(r, "experienced").get(m)
                    diffs.append(None if f is None or e is None else e - f)
                print("    experienced - fresh at %3d, seeds %s: %s" % (
                    m, [r["seed"] for r in group], " ".join("%+.2f" % d if d is not None else "na" for d in diffs)))
    print("\nEarly mean (checkpoints 60, 120, 180) and late mean (300, 600) of held-out in B, both orders pooled")
    for flag in ("off", "on"):
        for cond in ("fresh", "experienced"):
            group = [r for r in records if r["flag"] == flag]
            if not group:
                continue
            early = mean([mean([curve(r, cond).get(m) for m in (60, 120, 180)]) for r in group])
            late = mean([mean([curve(r, cond).get(m) for m in (300, 600)]) for r in group])
            print("  %-3s %-11s early %.3f  late %.3f  (n=%d lives)" % (flag, cond, early, late, len(group)))
    print("\nWith the setting on, experienced - fresh per life, mean of the checkpoints 30-180 (all on records)")
    for r in sorted((r for r in records if r["flag"] == "on"), key=lambda r: (r["first"], r["seed"])):
        f = mean([curve(r, "fresh").get(m) for m in (30, 60, 90, 120, 180)])
        e = mean([curve(r, "experienced").get(m) for m in (30, 60, 90, 120, 180)])
        print("  seed %d %-8s -> %-8s fresh %.3f experienced %.3f  difference %+.3f" % (
            r["seed"], r["first"], r["second"], f, e, e - f))
    print("\nSingle-domain check (fresh minds = one domain each), end of life (600) and early (60, 120), mean over seeds")
    keys = ("held_told_rule_ok", "held_induced_ok", "member_ok", "exception_ok", "exception_kept")
    for domain in ("nitelik", "iliski"):
        for flag in ("off", "on"):
            group = [r for r in records if r["second"] == domain and r["flag"] == flag]
            if not group:
                continue
            cells = []
            for at in (60, 120, 600):
                t3 = [x for r in group for x in r["fresh"] if x["test"] == "T3" and x["at"] == at]
                cells.append("@%d " % at + " ".join("%s %.2f" % (k.replace("_ok", "").replace("held_", ""),
                                                               mean([x.get(k) for x in t3])) for k in keys))
            t1 = [x for r in group for x in r["fresh"] if x["test"] == "T1" and x["at"] == 600]
            t2 = [x for r in group for x in r["fresh"] if x["test"] == "T2" and x["at"] == 600]
            t5 = [x for r in group for x in r["fresh"] if x["test"] == "T5" and x["at"] == 600]
            print("  %-8s %-3s %s" % (domain, flag, " | ".join(cells)))
            print("               T1 c1 %.3f c4 %.3f | T2 cat %.2f | T5 early %.2f identical %.2f exact %s ms %.1f" % (
                mean([x.get("c1") for x in t1]), mean([x.get("c4") for x in t1]),
                mean([x.get("cat_jaccard") for x in t2]), mean([x.get("early_recall") for x in t5]),
                mean([x.get("early_evidence_identical") for x in t5]), all(x.get("exact") for x in t5),
                mean([x.get("ms_per_moment") for x in t5])))
    print("\nIs A kept after B? (experienced minds; A's T3 and T5 before -> after B)")
    for flag in ("off", "on"):
        group = [r for r in records if r["flag"] == flag]
        same_ho = same_mem = same_early = 0
        for r in group:
            before = {x["test"]: x for x in r["a_before_b"]}
            after = {x["test"]: x for x in r["a_after_b"]}
            same_ho += held_out(before["T3"]) == held_out(after["T3"])
            same_mem += before["T3"].get("member_ok") == after["T3"].get("member_ok")
            same_early += after["T5"].get("early_evidence_identical") == 1.0
        print("  %-3s held-out unchanged %d/%d, members unchanged %d/%d, early facts bit-identical %d/%d" % (
            flag, same_ho, len(group), same_mem, len(group), same_early, len(group)))
    print("\nKinds of voices learned in A (pooled over lives with the setting on): many agree held failed credibility")
    pooled = {}
    for r in records:
        for many, agree, held, failed, _ in r.get("kinds_after_a", []):
            acc = pooled.setdefault((r["first"], many, agree), [0.0, 0.0])
            acc[0] += held
            acc[1] += failed
    for (first, many, agree), (held, failed) in sorted(pooled.items()):
        if held + failed >= 20:
            print("  %-8s many %d agree %2d  held %5.0f failed %5.0f  share %.2f" % (
                first, many, agree, held, failed, held / (held + failed)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--orders", nargs="+", default=["nitelik:iliski", "iliski:nitelik"])
    ap.add_argument("--flags", nargs="+", default=["off", "on"])
    ap.add_argument("--n-a", type=int, default=600)
    ap.add_argument("--n-b", type=int, default=600)
    ap.add_argument("--checkpoints", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "t6_learning_to_learn.jsonl"))
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--no-protected-off", action="store_true",
                    help="summary without the first T6 records as the setting-off baseline")
    args = ap.parse_args()
    if args.summary:
        summary(args.out, not args.no_protected_off)
        return
    torch.set_num_threads(1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tasks = [(seed, *order.split(":"), flag, args.n_a, args.n_b, args.checkpoints)
             for seed in args.seeds for order in args.orders for flag in args.flags]
    with Pool(args.workers) as pool:
        for record in pool.imap_unordered(one, tasks):
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(record["seed"], record["first"], "->", record["second"], record["flag"], "done", flush=True)


if __name__ == "__main__":
    main()
