# -*- coding: utf-8 -*-
"""t_m2.py: the ground (``Config.grounded``, zihin/zemin.py) in the grounded world, against the baseline of t_m1.py.

    python research/living_mind_2026_11_23/t_m2.py --lang tr --seeds 0 1 2

The same life as t_m1.py (``World.training``), lived by today's whole mind (``VARIANTS["full"]``) with the ground on:
every line is heard by the core as before, and what is seen while it is said is perceived by the ground (the things
in view with their features, the event going on with its participants), so that what is perceived and what is heard
meet (MODEL.md). Then the same questions, on copies:

* ``T1_new`` / ``T1_seen`` asked twice: ``a`` (understanding): the order is heard with only the things in view, the
  event is never shown, the answer comes from what the order asked for; ``b`` (seeing): the event the order asked for
  goes on while it is heard;
* ``T3_color`` (a colour never named with this kind, another thing of the kind beside it), ``T4_name`` (a thing of a
  colour never seen, pointed at: "bu ne ?"), ``T6_situation`` (who does what goes on, what a person is doing), with
  what the line shows;
* ``T2_what_is`` (no scene: the ground has nothing to tell; the core answers).

Each question is answered three ways: by the ground alone (``zemin``: silent when it does not understand), by the
core alone as in t_m1.py (``core``), and by both (``both``: the ground first, the core where the ground is silent).
Rows are appended to ``results/m2.jsonl``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dunya import World  # noqa: E402
from t_m1 import DIALOGUES, facts_said, heard_words, words  # noqa: E402


def raise_mind(train, grounded: bool = True):
    import torch
    torch.set_num_threads(1)
    import zihin.__main__ as chat
    from zihin import IMITATED, Mind
    from zihin.__main__ import CHANNEL, SOMEONE, Session
    from zihin.zemin import Zemin
    made = chat._log_path
    chat._log_path = lambda: None
    try:
        cfg = chat.VARIANTS["full"]()
    finally:
        chat._log_path = made
    cfg.keep_log, cfg.log_path = False, None
    cfg.grounded = grounded
    torch.manual_seed(0)
    mind = Mind(cfg)
    mind.add_channel(CHANNEL, "act")
    mind.zemin = Zemin(CHANNEL)
    s = Session(mind)
    z = mind.zemin
    with torch.no_grad():
        for line in train:
            if line["kind"] in DIALOGUES:
                s.company(SOMEONE)
                z.see(line["scene"])
                s.listen(line["text"], IMITATED, speaker=SOMEONE)
                s.end_of_turn(IMITATED, tone=s.tone(line["text"]))
                z.heard(s.body(line["text"]), "soran")
                z.see(line["scene_answer"] or line["scene"])
                s.listen(line["answer"], IMITATED)
                s.end_of_turn(IMITATED, tone=s.tone(line["answer"]))
                z.heard(s.body(line["answer"]), "cevaplayan")
                mind.step([])
                if getattr(cfg, "wholes", False):
                    from zihin.butun import consolidate
                    consolidate(s, line["text"], line["answer"], IMITATED, SOMEONE)
            else:
                s.company(None)
                z.see(line["scene"])
                s.listen(line["text"], IMITATED)
                s.end_of_turn(IMITATED, tone=".")
                z.heard(s.body(line["text"]), "anlatan")
                mind.step([])
        z.see(None)
        mind.step([])
        s.close_turn()
    return mind, s


def ground_answer(mind, s, line: dict, scene, before=None) -> str:
    """The ground's answer on a copy: the scene perceived in a moment, then the question understood and answered.
    ``before``: the scene in which the question was heard, before the event of ``scene`` began (T1 b: the order is
    heard, then the person does it, then the mind speaks); the question is understood there, answered with what
    then goes on."""
    from zihin.zemin import BEFORE, answer, understand
    c = mind.clone()
    z = c.zemin
    z.pending, z.waiting, z.turn, z.event = [], None, None, None
    if before is not None:
        z.see(before)
        c.step([])
        if understand(c, z, s.body(line["text"])) is None:
            return "(sessiz)"
    z.see(scene)
    c.step([])
    got = answer(c, z, s.body(line["text"]), when=BEFORE if before is not None else None)
    return " ".join(got) if got else "(sessiz)"


def core_answer(mind, line: dict) -> str:
    from zihin.__main__ import Session
    c = mind.clone()
    c.cfg.grounded = False
    return Session(c).reply(line["text"], limit=40)


def run(world: World, train, tests, grounded: bool = True) -> dict:
    t0 = time.process_time()
    mind, s = raise_mind(train, grounded)
    lived = time.process_time() - t0
    heard = heard_words(train)
    t1 = time.process_time()
    scores, shown = {}, {}
    for name, items in tests.items():
        modes = [("a", "scene"), ("b", "scene_answer")] if name.startswith("T1") else [("", "scene")]
        for suffix, which in modes:
            label = name + ("_" + suffix if suffix else "")
            row = {"zemin": [0, 0, 0, 0], "core": [0, 0, 0, 0], "both": [0, 0, 0, 0]}
            facts = [0, 0, 0]
            for line in items:
                want = words(line["answer"])
                ok_heard = all(w in heard or w in words(line["text"]) for w in want)
                if not (line.get(which) or line.get("scene")):
                    z_said = "(sessiz)"
                elif suffix == "b":                  # the order heard before the event, answered while it goes on
                    z_said = ground_answer(mind, s, line, line[which], before=line["scene"])
                else:
                    z_said = ground_answer(mind, s, line, line.get(which) or line.get("scene"))
                c_said = core_answer(mind, line) if suffix != "b" else None
                if c_said is None:
                    c_said = shown.get(("core", name, line["text"]), "(sessiz)")
                shown[("core", name, line["text"])] = c_said
                b_said = z_said if z_said != "(sessiz)" else c_said
                for who, said in (("zemin", z_said), ("core", c_said), ("both", b_said)):
                    ok = said != "(sessiz)" and words(said) == want
                    row[who][0] += ok
                    row[who][1] += 1
                    if ok_heard:
                        row[who][2] += ok
                        row[who][3] += 1
                if line["kind"] == "what_is":
                    f = world.facts(line["tags"]["thing"])[:3]
                    facts[0] += facts_said(b_said, f)
                    facts[2] += len(f)
                ex = shown.setdefault(label, [])
                if len(ex) < 5:
                    ex.append([line["text"], z_said, c_said, line["answer"]])
            scores[label] = row
            if facts[2]:
                scores[label]["facts"] = [facts[0], facts[2]]
    examples = {k: v for k, v in shown.items() if isinstance(k, str)}
    return {"who": "mind+zemin" if grounded else "mind", "moments": mind.t, "ties": mind.S.n_ties,
            "units": mind.S.n_units, "learn_cpu_s": round(lived, 1), "answer_cpu_s": round(time.process_time() - t1, 1),
            "scores": scores, "examples": examples}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="tr")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--rows", default=os.path.join(HERE, "results", "m2.jsonl"))
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    for seed in args.seeds:
        world = World(seed, args.lang)
        train = world.training()
        tests = world.tests()
        digest = hashlib.md5(json.dumps([train, tests], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        row = run(world, train, tests)
        row.update({"test": "m2", "seed": seed, "lang": args.lang, "world": digest, "note": args.note})
        print("%s seed %d  %s" % (args.lang, seed, "  ".join(
            "%s z%d/c%d/b%d of %d" % (k, v["zemin"][0], v["core"][0], v["both"][0], v["zemin"][1])
            for k, v in row["scores"].items())), flush=True)
        if args.rows:
            with open(args.rows, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
