# -*- coding: utf-8 -*-
"""M3: the grounded world of the earlier study (research/living_mind_2026_11_23), lived by Cevahir (NOTES.md, 8).

    python m3_ground.py --lang tr --seeds 0 1 2 [--workers 2]

The life and the questions of t_m2.py, unchanged (``World(seed, lang).training()`` and ``.tests()``): every line is
heard, and what is seen while it is said is perceived by Zihin's ground; then the questions, each answered by a
process that is the mind as it stands. Zihin's answer is t_m2.py's "both" (the ground where it understands, the core
where it does not); Cevahir's is the prefrontal decision between that plan and the harmony; the harmony alone
answers under the speaking line. T1 is asked twice (a: the order heard with only the things in view; b: the event
goes on while it is heard), as there. Scores per test: right, asked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "research", "living_mind_2026_11_23"))
from living.cevahir import Cevahir  # noqa: E402


def run(seed: int, lang: str, workers: int) -> dict:
    import torch
    from dunya import World
    from t_m1 import DIALOGUES, words
    from zihin.__main__ import SOMEONE
    torch.set_num_threads(1)
    world = World(seed, lang)
    train, tests = world.training(), world.tests()
    digest = hashlib.md5(json.dumps([train, tests], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    c = Cevahir(variant="full", seed=0, dil="tr" if lang == "tr" else "diger")
    c.ground()
    t0 = time.process_time()
    with torch.no_grad():
        for line in train:
            if line["kind"] in DIALOGUES:
                c.watch(line["text"], line["answer"], asker=SOMEONE, scene=line["scene"],
                        scene_answer=line["scene_answer"])
            else:
                c.tell(line["text"], scene=line["scene"], tone=".")
        c.mind.zemin.see(None)
        c.mind.step([])
        c.session.close_turn()
    lived = time.process_time() - t0
    jobs, keys = [], []
    for name, items in tests.items():
        modes = [("a", "scene"), ("b", "scene_answer")] if name.startswith("T1") else [("", "scene")]
        for suffix, which in modes:
            label = name + ("_" + suffix if suffix else "")
            for line in items:
                scene = line.get(which) or line.get("scene")
                before = line["scene"] if suffix == "b" else None
                jobs.append((line["text"], scene, before))
                keys.append((label, words(line["answer"])))
    t1 = time.time()
    said = c.answers(jobs, way="seen", workers=workers)
    scores, examples = {}, {}
    for (label, want), got in zip(keys, said):
        row = scores.setdefault(label, {"zihin": [0, 0], "harmony": [0, 0], "cevahir": [0, 0]})
        for way in row:
            ok = got[way] != "(sessiz)" and words(got[way]) == want
            row[way][0] += ok
            row[way][1] += 1
        ex = examples.setdefault(label, [])
        if len(ex) < 3:
            ex.append({"want": " ".join(want), **got})
    return {"test": "m3_ground", "seed": seed, "lang": lang, "world": digest, "lines": len(train),
            "zihin_moments": c.mind.t, "learn_cpu_s": round(lived, 1), "answer_wall_s": round(time.time() - t1, 1),
            "workers": workers, "scores": scores, "examples": examples,
            "disagreements": {str(k): [c.dis_won[k], c.dis_all[k]] for k in sorted(c.dis_all)},
            "alone": {str(k): [c.alone_right[k], c.alone_all[k]] for k in sorted(c.alone_all)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="tr")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "m3_ground.jsonl"))
    args = ap.parse_args()
    for seed in args.seeds:
        row = run(seed, args.lang, args.workers)
        print("%s seed %d  %s" % (args.lang, seed, "  ".join(
            "%s z%d/h%d/c%d of %d" % (k, v["zihin"][0], v["harmony"][0], v["cevahir"][0], v["zihin"][1])
            for k, v in row["scores"].items())), flush=True)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
