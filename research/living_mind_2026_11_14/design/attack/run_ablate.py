#!/usr/bin/env python3
"""run_ablate.py -- which single change to §2.2/§2.5/§2.6 restores perception and recall? (N moments each)"""
import sys, json, zsim
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
out = sys.argv[2] if len(sys.argv) > 2 else "ablate.jsonl"
V = {
 "literal":                    dict(),
 "literal, no interoception":  dict(intero="none"),
 "literal, decay=0.2":         dict(decay=0.2),
 "literal, decay=0.9":         dict(decay=0.9),
 "only inputs clamped":        dict(clamp=True),
 "only unweighted negatives":  dict(neg="plain"),
 "only base-rate reference":   dict(ref="base"),
 "base-rate + plain neg":      dict(ref="base", neg="plain"),
 "base-rate + plain neg + clamp": dict(ref="base", neg="plain", clamp=True),
 "base+plain+clamp+events-only ties": dict(ref="base", neg="plain", clamp=True, create="events"),
 "patched_kwta (all fixes, recurrent kWTA kept)": dict(zsim.PRESETS["patched_kwta"]),
 "patched (all fixes, monotone settle)": dict(zsim.PRESETS["patched"]),
 "patched + bag K=64": dict(zsim.PRESETS["patched"], K=64),
}
only = sys.argv[3:] 
for name, kw in V.items():
    if only and not any(o in name for o in only):
        continue
    zsim.run(name, N, kw, checkpoints=[1000, N], out=out)
