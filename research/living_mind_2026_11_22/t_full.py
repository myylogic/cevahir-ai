# -*- coding: utf-8 -*-
"""t_full.py: a measurement of this repository, run again with the WHOLE mind (the program's full configuration).

    python research/living_mind_2026_11_22/t_full.py [--off a,b] [--base speaking] [--rows results/full.jsonl]
                                                     [--name label] -- <script.py> <the script's own arguments>

Every published row names the switches it was measured with: the mind of each script is the speaking configuration
(``zihin.__main__.VARIANTS["speaking"]``) and what the script turns on beside it. Each mechanism was measured so:
alone, or with the few it was built on. The program now raises a mind with all of them (``VARIANTS["full"]``), a
combination no row measured.

This runs a script of the research folders unchanged, in this process, with the one thing replaced that all of them
build their minds from: ``VARIANTS["speaking"]`` returns the full configuration (minus ``--off``). What a script
then sets itself still holds (a script that turns a switch off for its "without" arm measures the whole mind without
that one mechanism). ``--base speaking`` replaces nothing: the same script, the same process, the configuration it
was written with (the row to compare against, on the same machine and code).

The script must be given an ``--out`` of its own: its rows are read from there, marked (``full``: the base, ``off``,
``script``, ``args``, ``seconds``, ``name``) and appended to ``--rows``; nothing is appended to the script's own
committed result file. Scripts that start worker processes inherit the replacement only where processes are forked.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "results", "full.jsonl")


def main() -> None:
    if "--" not in sys.argv:
        raise SystemExit(__doc__)
    cut = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", default="", help="mechanisms of the full configuration to leave out (comma separated)")
    ap.add_argument("--base", default="full", choices=("full", "speaking"))
    ap.add_argument("--rows", default=OUT)
    ap.add_argument("--name", default="", help="a label for these rows")
    args = ap.parse_args(sys.argv[1:cut])
    script, rest = sys.argv[cut + 1], sys.argv[cut + 2:]
    if not os.path.exists(script):
        raise SystemExit("no such script: %s" % script)
    if "--out" not in rest:
        raise SystemExit("give the script an --out of its own (its rows are read from there)")
    place = rest.index("--out") + 1
    asked_out = rest[place]
    fd, tmp = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.remove(tmp)
    rest = rest[:place] + [tmp] + rest[place + 1:]
    sys.path.insert(0, ROOT)
    import zihin.__main__ as chat
    off = sorted(k for k in args.off.split(",") if k)
    for k in off:
        if k not in chat.NEWER + chat.STUDIED:
            raise SystemExit("no such mechanism: %s" % k)
    speaking = chat.VARIANTS["speaking"]

    def whole(folder=None):
        cfg = chat.full(folder) if folder else chat.full()
        for k in off:
            setattr(cfg, k, False)
        return cfg

    if args.base == "full":
        chat.VARIANTS["speaking"] = whole
    sys.argv = [script] + rest
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    t0 = time.time()
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        chat.VARIANTS["speaking"] = speaking
        took = time.time() - t0
        rows = []
        if os.path.exists(tmp):
            with open(tmp, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            os.remove(tmp)
        os.makedirs(os.path.dirname(os.path.abspath(args.rows)), exist_ok=True)
        with open(args.rows, "a", encoding="utf-8") as f:
            for row in rows:
                out = {"full": args.base, "off": off, "name": args.name, "script": os.path.basename(script),
                       "args": [a for a in rest if a != tmp], "asked_out": os.path.basename(asked_out),
                       "seconds_all": round(took, 1)}
                out.update(row)
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print("#### %s %s: %d rows in %.0f s (base %s%s)" % (
            os.path.basename(script), " ".join(a for a in rest if a != tmp), len(rows), took, args.base,
            ", off: " + ",".join(off) if off else ""), flush=True)


if __name__ == "__main__":
    main()
