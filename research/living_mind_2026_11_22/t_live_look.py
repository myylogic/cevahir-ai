# -*- coding: utf-8 -*-
"""t_live_look.py: the whole loop on the real internet. NEEDS THE NETWORK (the Turkish Wikipedia, read only).

    python research/living_mind_2026_11_22/t_live_look.py [--things leylek,bakır,armut,vapur] [--agent TEXT]
                                                          [--set known=1] [--cache FOLDER]

A mind with T-want's beginning (twelve things read about, "X nedir ?" watched for eight; ``t_want.py``) is asked about
real things it never met. Then it is left in silence. Nobody tells it to look anything up: with F46 and F47 on it
decides by itself, and the session carries its looks out on the real Turkish Wikipedia through ``zihin.ag.Net``
(a wait between requests, tries again when the service is busy, robots obeyed for pages; here only the API the
service publishes for programs is asked). The mind then settles each look itself (the claim read again), and the
questions are asked again.

This is a demonstration, not a benchmark: a real page is not written to answer "X nedir ?" in the mind's words, and
the page read today may read differently tomorrow. What is recorded: what it looked up and in which silent moment,
the requests made, whether each claim was met by what it read, and what it then says (the beginning of the page's
own first sentence, as the mind's text adapter splits it).
Appends to ``results/live_look.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
sys.path.insert(0, HERE)
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "live_look.jsonl")


def main() -> None:
    import t_relations as R
    import t_want as W
    from zihin import IMITATED, Mind
    from zihin.__main__ import CHANNEL, Session
    from zihin.ag import Net
    from zihin.read import open_claim
    from zihin.web import Wikipedia
    ap = argparse.ArgumentParser()
    ap.add_argument("--things", default="leylek,bakır,armut,vapur")
    ap.add_argument("--agent", default="", help="how the reader names itself to the service (default: the program's)")
    ap.add_argument("--silences", type=int, default=12)
    ap.add_argument("--set", default="", help="more switches of the mind, e.g. known=1")
    ap.add_argument("--cache", default="", help="keep what is fetched in this folder (a run can then be repeated "
                                                "without asking the service again)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    things = [t for t in args.things.split(",") if t]
    switches = {"open_claims": True, "look": True}
    switches.update({k: bool(int(v)) for k, v in (kv.split("=") for kv in filter(None, args.set.split(",")))})
    mind = Mind(R.speaking_config(switches))
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)
    for _ in range(3):
        for a in W.KNOWN:
            s.company(None)
            s.listen(W.fact(a), IMITATED)
            s.end_of_turn(IMITATED, tone=".")
            mind.step([])
        for a in list(W.KNOWN)[:8]:
            s.watch(W.ASK % a, W.fact(a))
    net = Net(agent=args.agent or None, gap=1.0, cache=args.cache or None)
    s.net, s.look_source, s.may_look, s.look_gap = net, Wikipedia("tr", net=net), True, 0.0
    lines = ["== the whole loop on the real Turkish Wikipedia: %s; switches %s ==" % (", ".join(things), switches)]
    before = [s.reply(W.ASK % a, limit=40) for a in things]
    s.close_turn()
    lines.append("  asked, never met: %s" % before)
    printed = []
    for i in range(args.silences):
        n = len(s.looks)
        s.silent_moment(say=printed.append)
        if len(s.looks) > n:
            r = s.looks[-1]
            lines.append("  silent moment %d: it looked up %r by itself; read %s; %d sentences; claim met: %s" % (
                i, r.get("topic"), r.get("read"), r.get("sentences", 0), r.get("met")))
    after = [Session(mind.clone()).reply(W.ASK % a, limit=40) for a in things]
    for a, said in zip(things, after):
        lines.append("  asked again, %s: %s" % (a, said))
    lines.append("  claims still open: %d of %d; requests: %s" % (
        sum(open_claim(mind, a) > 0.5 for a in things), len(things),
        ", ".join("%s in %.1f s (%d tr%s)" % (e["status"], e["seconds"], e["tries"], "y" if e["tries"] == 1 else "ies")
                  for e in net.log)))
    for line in lines:
        print(line, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"test": "t_live_look", "switches": switches, "things": things, "before": before,
                            "after": after,
                            "looks": [{k: r.get(k) for k in ("topic", "read", "sentences", "met", "way")}
                                      for r in s.looks],
                            "requests": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items()}
                                         for e in net.log], "summary": lines}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
