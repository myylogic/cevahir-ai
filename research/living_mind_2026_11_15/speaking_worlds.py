# -*- coding: utf-8 -*-
"""The chat configuration in the generated worlds of the first lives, as it lives in a chat: listening while the world
speaks; and the same lives read with other reading switches.

    python research/living_mind_2026_11_15/speaking_worlds.py --listen --set starts=1,exclusive=1,exact_relations=1
    python research/living_mind_2026_11_15/speaking_worlds.py --read starts=1,exclusive=1,exact_relations=1
    python research/living_mind_2026_11_15/speaking_worlds.py --summary

The worlds, schedules, probes and checkpoints are those of ``research/living_mind_2026_11_14/run.py`` (imported, not
changed), with the speaking configuration of ``f25_born_failures.py`` (the switches of the chat program as of that
study; ``--set`` changes some of them for the whole life). Two changes of protocol are offered, one at a time:

- ``--listen``: while the world says an utterance the mind listens (its act threshold is raised for those moments, as
  ``Session.listen`` does in a chat), so it does not speak into the world's stream. In ``run.py`` the mind is free to act
  in every moment, and the chat configuration does, so it lives its own words between the world's.
- ``--read``: the life is lived as set; at every checkpoint the probes are read twice, as lived and with these reading
  switches on, on the same mind at the same moment (rows marked ``"reading": "on"``). What a reading does is then
  separated from how it changes a life.

Writes the probe rows of every life to ``results/speaking_worlds.jsonl`` (numbers only) and prints the summary.
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
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "results", "speaking_worlds.jsonl")
MARKS = (60, 120, 180, 300, 600)
KEYS = [("T3", "held_told_rule_ok"), ("T3", "held_induced_ok"), ("T3", "member_ok"), ("T3", "exception_ok"),
        ("T3", "exception_kept"), ("T5", "early_recall"), ("T1", "c1"), ("T1", "e1"), ("T2", "cat_jaccard"),
        ("T2", "categories"), ("T5", "ties")]


def switches(text: str) -> dict:
    return {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, (text or "").split(",")))}


def one(task) -> dict:
    import torch
    import run as R
    import f25_born_failures as F
    from zihin import Event
    domain, seed, listen, life, read = task
    torch.set_num_threads(1)
    if listen:
        def utter(mind, symbols, prov, pause=1):
            moments, theta = [], mind.knob["theta"]
            mind.knob["theta"] = 1e9
            try:
                for symbol in symbols:
                    mind.step([Event(R.CHANNEL, symbol, prov)])
                    moments.append(mind.t)
            finally:
                mind.knob["theta"] = theta
            for _ in range(pause):
                mind.step([])
            return moments
        R.utter = utter
    if read:
        plain = getattr(R, "_plain_checkpoint", R.checkpoint)   # a worker lives several lives: wrap the original once
        R._plain_checkpoint = plain

        def twice(world, mind, state, at):
            rows = plain(world, mind, state, at)
            saved = {k: getattr(mind.cfg, k) for k in read}
            for k, v in read.items():
                setattr(mind.cfg, k, v)
            try:
                on = plain(world, mind, state, at)
            finally:
                for k, v in saved.items():
                    setattr(mind.cfg, k, v)
            for r in on:
                r["reading"] = "on"
            return rows + on
        R.checkpoint = twice
    over = dict(F.SPEAKING)
    over.update(life)
    world = R.World.make(seed, domain)
    t0 = time.time()
    mind, rows, _ = R.live(world, "members", 600, **over)
    return {"domain": domain, "seed": seed, "listen": listen, "life": life, "read": read, "rows": rows,
            "moments": mind.t, "ties": mind.S.n_ties, "seconds": round(time.time() - t0, 1)}


def label(rec) -> str:
    life = ",".join("%s=%d" % kv for kv in sorted(rec["life"].items())) or "as in f25"
    return "%s, life: %s" % ("listening" if rec["listen"] else "speaking into the stream", life)


def summary(path: str) -> None:
    recs = [json.loads(line) for line in open(path, encoding="utf-8")]
    groups = {}
    for rec in recs:
        groups.setdefault((label(rec), json.dumps(rec["read"], sort_keys=True)), []).append(rec)
    for (name, read), group in groups.items():
        readings = [("off", "lived")] + ([("on", "read with " + read)] if read != "{}" else [])
        for tag, title in readings:
            print("\n== %s; %s ==" % (name, title))
            for domain in ("nitelik", "iliski"):
                g = [r for r in group if r["domain"] == domain]
                if not g:
                    continue
                cells = []
                for mark in MARKS:
                    vals = []
                    for rec in g:
                        for r in rec["rows"]:
                            if r["test"] == "T3" and r.get("at") == mark and r.get("reading", "off") == tag:
                                v = [r[k] for k in ("held_told_rule_ok", "held_induced_ok") if k in r]
                                vals.append(sum(v) / len(v))
                    cells.append("%d:%.3f" % (mark, sum(vals) / len(vals) if vals else float("nan")))
                print("  %s (%d seeds) held-out %s" % (domain, len(g), "  ".join(cells)))
                for test, key in KEYS:
                    vals = [r[key] for rec in g for r in rec["rows"] if r["test"] == test and r.get("at") == 600
                            and key in r and r.get("reading", "off") == tag]
                    if vals:
                        print("    %-24s %.3f" % (test + " " + key, sum(vals) / len(vals)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", action="store_true")
    ap.add_argument("--set", default="", help="switches for the whole life, e.g. exact_relations=1")
    ap.add_argument("--read", default="", help="reading switches applied to the probes of the same life")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary(args.out)
        return
    tasks = [(d, s, args.listen, switches(args.set), switches(args.read)) for d in ("nitelik", "iliski")
             for s in args.seeds]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with Pool(args.workers) as pool, open(args.out, "a", encoding="utf-8") as f:
        for rec in pool.imap_unordered(one, tasks):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(rec["domain"], rec["seed"], "%.0f s" % rec["seconds"], flush=True)
    summary(args.out)


if __name__ == "__main__":
    main()
