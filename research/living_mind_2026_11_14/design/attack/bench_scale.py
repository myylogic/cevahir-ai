#!/usr/bin/env python3
"""bench_scale.py -- LEAD_DESIGN.md §10 "Engine form" at 10^6 units on this 4-core CPU (Xeon 2.1 GHz, 15 GB).

    "sparse tie tables (COO: src, dst, form, lag, w+[6], w-[6], use, last), settling = sparse matrix-vector
     products + top-F selection; evidence updates are scatter-adds. Float64 evidence. ... Complexity per moment
     O(F^2 + F*deg)."

Measured here (NumPy, one thread; one op repeated with torch on 4 threads for reference):
 A. one settling iteration done as the text says, a sparse matrix-vector product over the COO table
    (cost is O(all ties), independent of F), n = 10^6 units, m = 2*10^7 ties (average degree 20);
 B. the same iteration with an index by source (CSR slices): O(sum of degrees of the focus);
 C. the LEARN step for one focus unit of degree d (negative increment on every non-occurring tie, §2.5),
    in the design layout (12 float64 per tie) and in a lean layout (2 float32 + implicit negatives);
 D. bytes per tie in both layouts.
The per-tie costs are then combined with the growth laws measured by run_stream.py.
"""
import os, time, json, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

rng = np.random.default_rng(0)
n = 1_000_000
m = int(float(sys.argv[1])) if len(sys.argv) > 1 else 20_000_000
F = 7; k = 1.0
tau = np.array([1.0, 0.3, 0.3, 1.0, 0.5, 0.2])
res = {"n": n, "m": m}


def best(f, reps=3):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts)


print(f"building COO table: n={n:,} units, m={m:,} ties, design layout (w+[6], w-[6] float64) ...", flush=True)
src = np.sort(rng.integers(0, n, m).astype(np.int32)); dst = rng.integers(0, n, m).astype(np.int32)
wp = rng.random((m, 6)); wn = rng.random((m, 6))
bytes_design = 4 + 4 + 1 + 1 + 96 + 4 + 4
print(f"  evidence arrays alone: {(wp.nbytes + wn.nbytes)/1e9:.2f} GB; full design row = {bytes_design} B/tie "
      f"-> {m*bytes_design/1e9:.2f} GB for this table, {1e8*bytes_design/1e9:.1f} GB at 10^8 ties (10^6 units x degree 100)")
a = np.zeros(n); focus = rng.choice(n, F, replace=False); a[focus] = 1.0

# ---------------------------------------------------------------- A. COO sparse matrix-vector product
def coo_iter():
    Wp = wp @ tau; Wn = wn @ tau
    w = (Wp - Wn) / (Wp + Wn + k)
    y = np.bincount(dst, weights=a[src] * w, minlength=n)
    return np.argpartition(y, -F)[-F:]

tA = best(coo_iter, 2)
print(f"A. COO settle iteration (expectation from 12 evidence columns + SpMV + top-F): {tA*1e3:.0f} ms "
      f"= {tA/m*1e9:.1f} ns/tie  -> T=6: {6*tA:.2f} s per moment at 2*10^7 ties, {6*tA*5:.1f} s per moment at 10^8 ties")
res["A_coo_iter_s"] = tA
# with the weight cached once per moment (still O(m) per iteration)
w_cached = (wp @ tau - wn @ tau) / (wp @ tau + wn @ tau + k)
def coo_iter_cached():
    y = np.bincount(dst, weights=a[src] * w_cached, minlength=n)
    return np.argpartition(y, -F)[-F:]
tA2 = best(coo_iter_cached, 2)
print(f"   with weights cached per moment: {tA2*1e3:.0f} ms per iteration ({tA2/m*1e9:.1f} ns/tie)")
res["A_coo_iter_cached_s"] = tA2
try:
    import torch
    for nt in (1, 4):
        torch.set_num_threads(nt)
        tsrc = torch.from_numpy(src.astype(np.int64)); tdst = torch.from_numpy(dst.astype(np.int64))
        twp = torch.from_numpy(wp); twn = torch.from_numpy(wn); ttau = torch.from_numpy(tau); ta = torch.from_numpy(a)
        def tcoo():
            Wp = twp @ ttau; Wn = twn @ ttau
            w = (Wp - Wn) / (Wp + Wn + k)
            y = torch.zeros(n, dtype=torch.float64).index_add_(0, tdst, ta[tsrc] * w)
            return torch.topk(y, F).indices
        tt = best(tcoo, 2)
        print(f"   torch, {nt} thread(s): {tt*1e3:.0f} ms per iteration")
        res[f"A_torch_{nt}"] = tt
    del twp, twn
    torch.set_num_threads(1)
except Exception as ex:
    print("   torch not used:", ex)

# ---------------------------------------------------------------- B. indexed by source (CSR)
indptr = np.searchsorted(src, np.arange(n + 1)).astype(np.int64)
def csr_iter():
    acc = {}
    for u in focus.tolist():
        s, e = indptr[u], indptr[u + 1]
        Wp = wp[s:e] @ tau; Wn = wn[s:e] @ tau
        w = (Wp - Wn) / (Wp + Wn + k)
        for v, x in zip(dst[s:e].tolist(), w.tolist()):
            acc[v] = acc.get(v, 0.0) + x
    return sorted(acc, key=acc.get)[-F:]
tB = best(csr_iter, 5)
deg = float(np.mean(indptr[focus + 1] - indptr[focus]))
print(f"B. indexed by source, 7 focus units of degree ~{deg:.0f}: {tB*1e6:.0f} us per iteration (interpreter-bound; "
      f"{F*deg:.0f} ties) -> speed-up over A: {tA/tB:,.0f}x")
res["B_csr_iter_s"] = tB

# ---------------------------------------------------------------- C. one HUB in the focus: read + learn
print("C. a hub in the focus (a unit tied to d other units), design layout: settle read and LEARN update per moment")
for d in (10_000, 100_000, 1_000_000, 2_000_000):
    s = 0; e = d
    hub_dst = dst[s:e]; occ = np.zeros(d, bool); occ[:7] = True
    scratch = np.zeros(n)
    def read():
        Wp = wp[s:e] @ tau; Wn = wn[s:e] @ tau
        w = (Wp - Wn) / (Wp + Wn + k)
        scratch[hub_dst] += w            # scatter into the activation accumulator (random access)
        scratch[hub_dst] = 0.0
    def learn():
        Wp = wp[s:e] @ tau; Wn = wn[s:e] @ tau
        ee = (Wp + k / 2) / (Wp + Wn + k)
        inc = ee.copy(); inc[occ] = 0.0
        wn[s:e, 0] += inc                # w-[sensed] += g a e   on every tie whose target did not occur
        wp[s:e, 0][occ] += 1.0
    tr = best(read, 3); tl = best(learn, 3)
    print(f"   d={d:>9,}: settle read {tr*1e3:7.2f} ms/iteration ({tr/d*1e9:.1f} ns/tie), LEARN {tl*1e3:7.2f} ms "
          f"({tl/d*1e9:.1f} ns/tie), retraction-log entries written per moment: {d:,} = {d*16/1e6:.1f} MB")
    res[f"C_hub_{d}"] = dict(read_s=tr, learn_s=tl)
res["ns_per_tie_read"] = res["C_hub_1000000"]["read_s"] / 1e6 * 1e9
res["ns_per_tie_learn"] = res["C_hub_1000000"]["learn_s"] / 1e6 * 1e9

# ---------------------------------------------------------------- lean layout
del wp, wn
print("D. lean layout: per tie dst int32 + w+ float32 + exposure offset float32 (+ 1 byte flags) = 13 B; negatives implicit")
d = 1_000_000
lp = rng.random(d).astype(np.float32); x0 = rng.random(d).astype(np.float32); ld = dst[:d]; X = 5.0
scratch = np.zeros(n, np.float32); b = np.full(n, 0.001, np.float32)
def lean_read():
    W = X - x0
    w = (lp - b[ld] * W) / (W + 1.0)
    scratch[ld] += w; scratch[ld] = 0.0
tl = best(lean_read, 3)
print(f"   read of a row of d=10^6: {tl*1e3:.2f} ms ({tl/d*1e9:.1f} ns/tie); LEARN touches only the ties whose target occurred "
      f"(<= F per focus unit) + one exposure counter: O(F^2) per moment, independent of d")
res["lean_read_ns"] = tl / d * 1e9
for K in (32, 64, 128):
    rows = [(rng.random(K).astype(np.float32), rng.random(K).astype(np.float32), rng.integers(0, n, K)) for _ in range(F)]
    def bag_iter():
        for lp_, x0_, ld_ in rows:
            W = X - x0_
            scratch[ld_] += (lp_ - b[ld_] * W) / (W + 1.0)
        for lp_, x0_, ld_ in rows:
            scratch[ld_] = 0.0
    tb = best(bag_iter, 20)
    print(f"   bounded bags K={K}: one settle pass over F=7 units x 2 lags ~ {2*tb*1e6:.0f} us in NumPy (call overhead bound); "
          f"ties at 10^6 units <= {2*K*1e6:,.0f} = {2*K*1e6*13/1e9:.1f} GB lean / {2*K*1e6*bytes_design/1e9:.0f} GB design layout")
    res[f"bag_{K}_s"] = 2 * tb
json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)
