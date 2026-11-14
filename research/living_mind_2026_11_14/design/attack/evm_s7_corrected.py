#!/usr/bin/env python3
"""evm_s7_corrected.py -- the corrected arithmetic of attack/evidence-math.md on the design's own claims.

  A  worked numbers, DESIGN | CORRECTED, for the behaviours claimed in LEAD_DESIGN §1 and §4 (T1, T3)
  B  invariants the corrected arithmetic has and the design does not:
       order independence, exact retraction, conviction from one source bounded by its reliability,
       no path by which a tie's own evidence returns to it
"""
import math
import numpy as np
from evm_common import read, par, coupling, reliability, source_evidence, expect

RHO_T = 0.3          # reliability of a teller at birth (the design's tau_told = 0.3)
LAM = 0.5


def corrected(sources, b=0.5):
    """sources: list of (n_pos, n_neg, n_opp, chi, rho).  returns f, c, e."""
    Wp = Wn = 0.0
    for s in sources:
        a, d = source_evidence(*s)
        Wp += a; Wn += d
    W = Wp + Wn
    return (Wp / W if W > 0 else float("nan")), W / (W + 1.0), expect(Wp, Wn, b)


def design_after_silence(m, tau=0.3):
    wp, wn = 1.0, 0.0
    for _ in range(m):
        wn += read(tau * wp, tau * wn)[2]
    return read(tau * wp, tau * wn)


def row(label, d, c):
    print(f"   {label:62s} | f={d[0]:.2f} c={d[1]:.3f} e={d[2]:.3f} | f={c[0]:.2f} c={c[1]:.3f} e={c[2]:.3f}")


def part_a():
    print("== A. worked numbers.  Every row starts again from 'X -> v told once by a teller of reliability 0.3'. ==")
    print("   scenario                                                       | DESIGN (§1, §2.5, §4)      | CORRECTED")
    T1 = (1.0, 0.0, 1.0, 0.0, RHO_T)                      # one assertion, chi = 0 (open world), rho = 0.3
    row("a  first hearing", read(0.3, 0.0), corrected([T1]))
    row("b  + 3 other true statements about X by the same teller", design_after_silence(3),
        corrected([(1.0, 0.0, 4.0, 0.0, RHO_T)]))
    row("c  + the same teller repeats it 9 more times", read(3.0, 0.0), corrected([(10.0, 0.0, 10.0, 0.0, RHO_T)]))
    row("d  + a second, independent teller says it once", read(0.6, 0.0), corrected([T1, T1]))
    S_yes = (1.0, 0.0, 1.0, 1.0, 1.0)                     # one closed own sighting: v present
    S_no = (0.0, 0.0, 1.0, 1.0, 1.0)                      # one closed own sighting: v absent
    row("e  + one own sighting that confirms", read(1.3, 0.0), corrected([T1, S_yes]))
    row("f  + one own sighting that contradicts (refinement, no deletion)", read(0.3, 1.0), corrected([T1, S_no]))
    row("g  + the same teller DENIES it once ('X not v')", read(0.6, 0.0), corrected([(1.0, 1.0, 2.0, 0.0, RHO_T)]))
    print("      (design: the denial contains v, so it is booked as a second assertion; corrected: n- += 1 on the")
    print("       teller's own slot, by an absence event or a learned exclusion)")
    # category support: 6 other members, each seen once with v
    for m in (6, 600):
        S = float(m)
        b_i = (S + 0.5) / (S + 1.0)                       # rule expectation without X, horizon k_g = 1
        d = read(0.3 + LAM * S, 0.0)
        c = corrected([T1], b=b_i)
        row(f"h  + rule from {m} OTHER members seen with v (X itself not seen)", d, c)
        d = read(0.3 + LAM * S, 1.0 + LAM * 1.0)
        c = corrected([T1, S_no], b=b_i)
        row(f"i    ... then X is SEEN once not to be v (exception), m = {m}", d, c)
        d = read(0.3 + LAM * S, 2.0 + LAM * 1.0)
        c = corrected([T1, (0.0, 0.0, 2.0, 1.0, 1.0)], b=b_i)
        row(f"j    ... seen twice not to be v, m = {m}", d, c)
    print("   in rows h-j the corrected c is the conviction LIVED about X itself; what the category adds is the base rate")
    print("   b of the expectation (0.929 for m = 6, 0.999 for m = 600) and its own conviction S/(S+k_g) = 0.857 / 0.998.")


def part_b():
    print("\n== B. invariants ==")
    # ---- order independence
    print("   1. order independence.  One tie, the same ten moments (5 with v, 5 without) in three orders:")
    orders = {"+++++-----": "+++++-----", "+-----++++": "+-----++++", "+-+-+-+-+-": "+-+-+-+-+-"}
    for name, seq in orders.items():
        wp = wn = 0.0; born = False
        n_pos = n_opp = 0.0
        for ch in seq:
            if ch == "+":
                wp += 1.0; born = True; n_pos += 1.0
            elif born:
                wn += read(wp, wn)[2]
            n_opp += 1.0
        fd, cd, ed = read(wp, wn)
        fc, cc, ec = corrected([(n_pos, 0.0, n_opp, 1.0, 1.0)])
        print(f"      {name}   design f = {fd:.3f}  e = {ed:.3f}   |   corrected f = {fc:.3f}  e = {ec:.3f}")
    # ---- exact retraction and order independence on a whole world (raw integer counts)
    from evm_s1_closed_world import make_world, run_corrected
    rng = np.random.default_rng(3)
    truth, insp, events = make_world(rng, q=0.8, n_stmt=3000)
    e_full = run_corrected(truth, events)[0]
    perm = [events[i] for i in np.random.default_rng(9).permutation(len(events))]
    e_perm = run_corrected(truth, perm)[0]
    k_told = next(i for i, ev in enumerate(events) if ev[0] == "told")
    e_without = run_corrected(truth, events[:k_told] + events[k_told + 1:])[0]
    # 'retraction' = decrement the raw counters of that one event; identical to never having had it
    x_kind, x_obj, x_prop = events[k_told]
    changed = float(np.abs(e_full - e_without).max())
    print(f"   2. a world of {len(events)} experiences (evm_s1 D): the same experiences in a random order give beliefs that")
    print(f"      differ by at most {float(np.abs(e_full - e_perm).max()):.1e}.")
    print(f"      retracting one told experience = decrementing its raw counters = the life without it (exact by")
    print(f"      construction: integer counts; the retraction moved the largest belief by {changed:.3f} and trust,")
    print(f"      completeness and base rate were recomputed from the counts, not patched).")
    # ---- bound
    print("   3. conviction from one source is bounded by its reliability:  n assertions by a teller of reliability rho")
    for rho in (0.3, 0.6, 0.9):
        cs = [corrected([(float(n), 0.0, float(n), 0.0, rho)])[1] for n in (1, 10, 1000, 10 ** 6)]
        print(f"      rho = {rho}:  c = " + ", ".join(f"{c:.4f}" for c in cs) + f"   (n = 1, 10, 1e3, 1e6;  bound {rho})")
    # ---- no return path
    print("   4. no return path: a category whose only member is X gives X nothing (rule without X is empty):")
    b_i = (0.0 + 0.5) / (0.0 + 1.0)
    f, c, e = corrected([(1.0, 0.0, 1.0, 1.0, 1.0)], b=b_i)
    fd, cd, ed = read(1.0 + LAM * 1.0, 0.0)
    print(f"      corrected: c = {c:.3f}, e = {e:.3f} (own evidence only)   |   design: W_eff = 1 + lambda*min(1,cap) = 1.5, c = {cd:.3f}, e = {ed:.3f}")


if __name__ == "__main__":
    part_a(); part_b()
