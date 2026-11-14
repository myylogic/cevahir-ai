#!/usr/bin/env python3
"""run_big.py -- the design WITH the proposed fixes (preset 'patched' + bounded bags K=64) pushed to ~10^6 units:
an open world (300k entities, 30k values) in which most statements are new, so that units grow with moments.
Shows whether time per moment, ties per unit and memory stay flat.  (The design as written cannot be run to this
size: its cost per moment is proportional to the number of units, see stream_literal.jsonl.)"""
import sys, zsim
N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 1300000
K = int(sys.argv[2]) if len(sys.argv) > 2 else 64
zsim.run("patched_K%d_open_world" % K, N, dict(zsim.PRESETS["patched"], K=K, skip_part=True),
         world_kw=dict(E=300000, C=300, R=40, V=30000),
         checkpoints=[10000, 30000, 100000, 300000, 1000000, N], out="stream_big.jsonl", ncap=N + 500000)
