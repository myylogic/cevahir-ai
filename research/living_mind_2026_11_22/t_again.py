# -*- coding: utf-8 -*-
"""t_again.py: the parts of the earlier scripts of this folder, run again with more switches on.

    python research/living_mind_2026_11_22/t_again.py --ac lived_apart [--parts find_out.asked,wonder.chain,...]
                                                      [--seeds 5]

Every mind of the earlier scripts is made by ``t_relations.speaking_config(switches)``. Here the same parts are run
unchanged, with the switches of ``--ac`` turned on beside whatever each part turns on itself: a new mechanism is
measured in the worlds that were measured before it was there. Each part prints its own summary, and its rows are
appended to ``results/again.jsonl`` with the part's name (``again_of``) and the switches added (``also_on``).

The parts: ``find_out.asked``, ``find_out.half`` (F50), ``wonder.chain``, ``wonder.chains``, ``wonder.dead``,
``wonder.young`` (F51), ``fault.misled``, ``fault.regular``, ``fault.two``, ``fault.valued``, ``fault.many``,
``fault.loop`` (F49), ``fault_kinds.kinds`` (F49 restructuring), ``fault_watch`` (silent claims).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "living_mind_2026_11_15"))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import torch  # noqa: E402

OUT = os.path.join(HERE, "results", "again.jsonl")
PARTS = {
    "find_out.asked": ("t_find_out", lambda m, a: m.asked(a)),
    "find_out.half": ("t_find_out", lambda m, a: m.asked(a, True)),
    "wonder.chain": ("t_wonder", lambda m, a: m.chain(a)),
    "wonder.chains": ("t_wonder", lambda m, a: m.chain(a, (4, 4))),
    "wonder.dead": ("t_wonder", lambda m, a: m.dead(a)),
    "wonder.young": ("t_wonder", lambda m, a: m.young(a)),
    "fault.misled": ("t_fault", lambda m, a: m.misled(a)),
    "fault.regular": ("t_fault", lambda m, a: m.misled(a, True)),
    "fault.two": ("t_fault", lambda m, a: m.two(a)),
    "fault.valued": ("t_fault", lambda m, a: m.valued(a)),
    "fault.many": ("t_fault", lambda m, a: m.many(a)),
    "fault.loop": ("t_fault", lambda m, a: m.loop(a)),
    "fault_kinds.kinds": ("t_fault_kinds", lambda m, a: m.kinds(a)),
}
DEFAULT = "find_out.asked,find_out.half,wonder.chain,fault.misled,fault.regular,fault.two,fault.loop,fault_kinds.kinds"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ac", required=True, help="switches to turn on beside each part's own (comma separated)")
    ap.add_argument("--parts", default=DEFAULT)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(1)
    import t_relations as R
    more = sorted(name for name in args.ac.split(",") if name)
    made = R.speaking_config

    def with_more(switches):
        cfg = made(switches)
        for name in more:
            if not hasattr(cfg, name):
                raise SystemExit("no such switch: %s" % name)
            setattr(cfg, name, True)
        return cfg
    R.speaking_config = with_more
    try:
        for part in [p for p in args.parts.split(",") if p]:
            module, call = PARTS[part]
            mod = importlib.import_module(module)
            fd, tmp = tempfile.mkstemp(suffix=".jsonl")
            os.close(fd)
            os.remove(tmp)
            print("#### %s again, with %s on too" % (part, ", ".join(more)), flush=True)
            t0 = time.time()
            call(mod, argparse.Namespace(seeds=args.seeds, out=tmp))
            took = time.time() - t0
            rows = []
            if os.path.exists(tmp):
                with open(tmp, encoding="utf-8") as f:
                    rows = [json.loads(line) for line in f if line.strip()]
                os.remove(tmp)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "a", encoding="utf-8") as f:
                for row in rows:
                    out = {"again_of": part, "also_on": more, "part_s": round(took, 1)}
                    out.update(row)
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
    finally:
        R.speaking_config = made


if __name__ == "__main__":
    main()
