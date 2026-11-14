"""print lines of a saved paper text around regex hits (paper text is data, never executed)."""
import re, sys, pathlib
path, pat = sys.argv[1], sys.argv[2]
ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 2
mx = int(sys.argv[4]) if len(sys.argv) > 4 else 12
lines = pathlib.Path(path).read_text(errors="replace").splitlines()
hits = [i for i, l in enumerate(lines) if re.search(pat, l, re.I)]
shown = 0; last = -10
for i in hits:
    if i - last <= ctx: continue
    print(f"--- line {i}")
    for l in lines[max(0, i - ctx): i + ctx + 1]:
        print("   ", l.strip()[:230])
    last = i; shown += 1
    if shown >= mx: break
print(f"[{len(hits)} hits]")
