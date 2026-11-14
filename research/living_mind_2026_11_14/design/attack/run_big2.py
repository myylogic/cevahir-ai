#!/usr/bin/env python3
"""run_big2.py -- as run_big.py, but a flatter and larger world (10^6 entities, Zipf exponent 0.6, 10^5 values) so that the
number of units itself passes 10^6 within ~10^6 moments.  Design with the proposed fixes + bounded bags K=64."""
import sys, zsim
N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 1200000
zsim.run("patched_K64_1M_units", N, dict(zsim.PRESETS["patched"], K=64, skip_part=True),
         world_kw=dict(E=1000000, C=1000, R=40, V=100000, zipf=0.6, p_stay=0.3),
         checkpoints=[100000, 300000, 600000, 900000, N], out="stream_big2.jsonl", ncap=3000000)
