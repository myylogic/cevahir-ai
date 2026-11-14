#!/usr/bin/env python3
"""run_stream.py -- the 10^5-moment stream requested by the review brief.
usage: run_stream.py <N> <out.jsonl> <variant> [<variant> ...]"""
import sys, zsim
N = int(sys.argv[1]); out = sys.argv[2]
V = {
 "literal":        dict(),
 "literal_nointero": dict(intero="none"),
 "patched":        dict(zsim.PRESETS["patched"]),
 "patched_K64":    dict(zsim.PRESETS["patched"], K=64),
 "patched_K32":    dict(zsim.PRESETS["patched"], K=32),
}
for name in sys.argv[3:]:
    zsim.run(name, N, V[name], checkpoints=[1000, 3000, 10000, 30000, 100000, 300000, 1000000], out=out)
