# -*- coding: utf-8 -*-
"""t_paradigm.py: the owner's example as a small world, answered by today's whole mind (MODEL.md, section 0.1).

    python research/living_mind_2026_11_23/t_paradigm.py [--on SWITCH ...] [--rows results/paradigm.jsonl]

The owner's paradigm: an order ("ali ata bak") is answered by describing it as it happens ("ali ata bakıyor"); "at
nedir ?" is answered by what is known of horses. The mind reads six facts twice, watches fourteen order-description
pairs and one "kedi nedir ?" dialogue twice, and is then asked, on copies:

* known verb, new pair: an order whose verb was described before, in a pair of words never heard together;
* known verb, new object: the same with an object never heard with that verb;
* verb never described: an order whose described form was never heard (it asks for a word form the mind never met;
  scored apart, see MODEL.md section 7.4);
* topic: "at nedir ?", scored by how many of the three facts read about horses the answer says.

Only words: today's mind has no scene, no referent and no event (MODEL.md section 0.2). This is the baseline the new
levels are measured against.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

TAUGHT = [("ali ata bak", "ali ata bakıyor"), ("ayşe topa vur", "ayşe topa vuruyor"), ("can eve koş", "can eve koşuyor"),
          ("zeynep kediye bak", "zeynep kediye bakıyor"), ("ali topa vur", "ali topa vuruyor"),
          ("ayşe eve gel", "ayşe eve geliyor"), ("can suya bak", "can suya bakıyor"), ("zeynep eve koş", "zeynep eve koşuyor"),
          ("ali kediye gül", "ali kediye gülüyor"), ("ayşe ata bak", "ayşe ata bakıyor"), ("can okula git", "can okula gidiyor"),
          ("zeynep topa bak", "zeynep topa bakıyor"), ("ali eve gel", "ali eve geliyor"), ("ayşe kediye gül", "ayşe kediye gülüyor")]
FACTS = ["at bir hayvandır .", "atların dört ayağı vardır .", "atlar çok hızlı koşar .", "kedi bir hayvandır .",
         "kedilerin dört ayağı vardır .", "kediler süt içer ."]
TESTS = [("can ata bak", "can ata bakıyor", "known verb, new pair"),
         ("zeynep suya vur", "zeynep suya vuruyor", "known verb, new pair"),
         ("ali okula koş", "ali okula koşuyor", "known verb, new object"),
         ("ayşe suya dal", "ayşe suya dalıyor", "verb never described"),
         ("can topa tut", "can topa tutuyor", "verb never described"),
         ("ali eve dön", "ali eve dönüyor", "verb never described")]
TOPIC = ("at nedir ?", ["at bir hayvandır", "atların dört ayağı vardır", "atlar çok hızlı koşar"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", nargs="*", default=[], help="switches set on besides the whole mind's")
    ap.add_argument("--rows", default="")
    args = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    import zihin.__main__ as chat
    from zihin import IMITATED, Mind
    from zihin.__main__ import CHANNEL, Session
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = chat.VARIANTS["full"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    for k in args.on:
        setattr(cfg, k, True)
    torch.manual_seed(0)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    s = Session(mind)

    def tell(text):
        s.company(None)
        s.listen(text, IMITATED)
        s.end_of_turn(IMITATED, tone=".")
        mind.step([])
    with torch.no_grad():
        for t in FACTS:
            for _ in range(2):
                tell(t)
        for _ in range(2):
            for q, a in TAUGHT:
                s.watch(q, a + " .")
            s.watch("kedi nedir ?", "kedi bir hayvandır . kedilerin dört ayağı vardır . kediler süt içer .")
        s.close_turn()
        right = {}
        rows = []
        for q, want, kind in TESTS:
            said = Session(mind.clone()).reply(q, limit=40)
            ok = said.rstrip(" .") == want
            right.setdefault(kind, [0, 0])
            right[kind][0] += ok
            right[kind][1] += 1
            rows.append({"cue": q, "said": said, "want": want, "kind": kind, "right": ok})
            print("%-24s %-18s -> %s%s" % (kind, q, said, "   (doğru)" if ok else ""))
        q, facts = TOPIC
        said = Session(mind.clone()).reply(q, limit=40)
        told = sum(f in said for f in facts)
        print("%-24s %-18s -> %s   (%d / %d bilgi)" % ("topic", q, said, told, len(facts)))
    out = {"test": "paradigm", "on": args.on, "right": right, "topic": [told, len(facts)], "rows": rows,
           "topic_said": said}
    if args.rows:
        with open(args.rows, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
