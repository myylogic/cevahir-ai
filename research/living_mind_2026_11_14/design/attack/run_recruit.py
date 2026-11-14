#!/usr/bin/env python3
"""run_recruit.py -- §2.2/§3 "if the focus holds a tuple matched by no composed unit above vigilance rho, RECRUIT a new
composed unit for it".  How many units does that make, and how often is a recruited unit ever matched again?
Two readings of "a tuple in the focus": (input) the tuple of the moment's external events; (focus) the focus content."""
import sys, zsim
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
for name, kw in [("design, tuple = external events of the moment", dict(intero="none", recruit="input")),
                 ("design, tuple = content of the focus", dict(intero="none", recruit="focus")),
                 ("design + interoception, tuple = content of the focus", dict(recruit="focus")),
                 ("fixes, tuple = content of the focus", dict(zsim.PRESETS["patched"], recruit="focus"))]:
    eng, world, rows = zsim.run(name, N, kw, checkpoints=[N], quiet=True, ncap=3 * N + 20000)
    C = eng.C
    print(f"{name:<55} moments={N} distinct facts in the world so far={len(world.seen)}  composed units recruited={C['recruited']} "
          f"({C['recruited']/N:.2f} per moment)  re-matched={C['recognised']} ({C['recognised']/N:.2%} of moments)", flush=True)
