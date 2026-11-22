# -*- coding: utf-8 -*-
"""growth.py: how a mind grows with what it lives. Units, ties, bytes and time per moment along one life.

    python research/living_mind_2026_11_22/growth.py --scan 3000 [--every 5000] [--set wholes=1] [--note TEXT]
    python research/living_mind_2026_11_22/growth.py --docs FOLDER --label NAME [--cap 0]

``--scan N``: SCAN's add-jump life of ``t_scan.py`` (the 81 commands without a connective, "jump", and N joined
commands, each dialogue watched once an epoch for two epochs): a public, generated life.
``--docs FOLDER``: every document of a folder read once, as the chat's ``!oku`` reads it. Only counts are written
(no text, no file name), so a private folder leaves nothing of itself in the result.

At every ``--every`` moments: moments lived, units, ties, bytes of the core tensors, resident memory of the process,
and milliseconds per moment over the interval. At the end: the fit ties = a * moments^b (least squares on the
logarithms), bytes per tie, and what the fit gives at 10^6, 10^8 and 10^9 moments (an extrapolation, not a
measurement). Appends one row to ``results/growth.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "growth.jsonl")
PAGE = os.sysconf("SC_PAGE_SIZE")


def resident_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * PAGE / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, default=0, help="SCAN add-jump life with this many joined commands")
    ap.add_argument("--docs", default="", help="a folder of documents, read once")
    ap.add_argument("--label", default="")
    ap.add_argument("--cap", type=int, default=0, help="--docs: at most this many sentences of each document")
    ap.add_argument("--every", type=int, default=5000)
    ap.add_argument("--window", type=int, default=0, help="the mind's window (0: its own)")
    ap.add_argument("--set", default="")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--note", default="", help="kept in the row (which code, which machine)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    import t_relations as R
    from zihin import Mind
    from zihin.__main__ import CHANNEL, Session
    switches = {k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, args.set.split(",")))}
    torch.manual_seed(0)
    base = resident_mb()
    mind = Mind(R.speaking_config(switches))
    mind.add_channel(CHANNEL, "act")
    if args.window:
        mind.knob["F"] = args.window
    s = Session(mind)
    curve, last = [], [time.perf_counter(), 0]
    lived = mind.step

    def step(events=()):
        out = lived(events)
        if mind.t % args.every == 0:
            now = time.perf_counter()
            size = mind.size()
            curve.append({"moments": mind.t, "units": size["units"], "ties": size["ties"],
                          "core_bytes": size.get("core_tensors", 0), "resident_mb": round(resident_mb(), 1),
                          "ms_per_moment": round(1000.0 * (now - last[0]) / max(1, mind.t - last[1]), 3)})
            last[0], last[1] = now, mind.t
            print(curve[-1], flush=True)
        return out
    mind.step = step
    t0 = time.perf_counter()
    if args.scan:
        import t_scan as S
        train, _ = S.split(0, args.scan, 1)
        for command, answer in S.life(0, train, 2):
            s.watch(" ".join(command), " ".join(answer) + " .")
        what = "SCAN add-jump, %d training dialogues watched twice" % len(train)
    else:
        s.read_path(args.docs, args.cap)
        what = "documents read once (%s)" % (args.label or "unnamed")
    seconds = time.perf_counter() - t0
    size = mind.size()
    xs = [math.log(c["moments"]) for c in curve if c["ties"] > 0]
    ys = [math.log(c["ties"]) for c in curve if c["ties"] > 0]
    fit = None
    if len(xs) >= 3:
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        a = math.exp(my - b * mx)
        per_tie = (curve[-1]["resident_mb"] - base) * 1e6 / max(1, curve[-1]["ties"])
        fit = {"a": a, "b": b, "resident_bytes_per_tie": round(per_tie, 1),
               "ties_at": {str(m): a * m ** b for m in (10 ** 6, 10 ** 8, 10 ** 9)},
               "resident_gb_at": {str(m): a * m ** b * per_tie / 1e9 for m in (10 ** 6, 10 ** 8, 10 ** 9)}}
    lines = ["== growth: %s; switches %s ==" % (what, switches or "-"),
             "  lived %d moments in %.0f s; %d units, %d ties (%.1f ties a moment); core tensors %.0f MB; "
             "resident %.0f MB (%.0f MB before the mind)" % (mind.t, seconds, size["units"], size["ties"],
                                                             size["ties"] / max(1, mind.t),
                                                             size.get("core_tensors", 0) / 1e6, resident_mb(), base)]
    if curve:
        lines.append("  ms per moment: first interval %.2f, last interval %.2f" % (
            curve[0]["ms_per_moment"], curve[-1]["ms_per_moment"]))
    if fit:
        lines.append("  ties = %.2f x moments^%.3f; %.0f resident bytes a tie" % (fit["a"], fit["b"],
                                                                                 fit["resident_bytes_per_tie"]))
        lines.append("  the fit at 10^6 / 10^8 / 10^9 moments: %s ties; %s GB resident (an extrapolation)" % (
            " / ".join("%.2e" % fit["ties_at"][k] for k in ("1000000", "100000000", "1000000000")),
            " / ".join("%.1f" % fit["resident_gb_at"][k] for k in ("1000000", "100000000", "1000000000"))))
    for line in lines:
        print(line)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"what": what, "note": args.note, "switches": switches, "every": args.every,
                            "window": mind.knob["F"], "seconds": round(seconds, 1),
                            "moments": mind.t, "units": size["units"], "ties": size["ties"], "curve": curve,
                            "fit": fit, "summary": lines}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
