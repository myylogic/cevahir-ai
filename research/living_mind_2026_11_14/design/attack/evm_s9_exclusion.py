#!/usr/bin/env python3
"""evm_s9_exclusion.py -- refinement without deletion (T1) under the corrected arithmetic, with LEARNED exclusion.

The design has no route from a contradicting statement to negative evidence (evidence-math.md, E1).  The corrected
LEARN rule CA2 has one: an occurring v' adds n- to the tie u -> v in proportion to a learned exclusion x(v', v).
Nothing about colours is in the rule; the exclusion is read from ordinary ties between property units.

    x(v', v) = c(v' -> v) * max(0, 1 - f_up(v' -> v) / b_v)
               c, f : conviction and frequency of the tie v' -> v ("when v' holds, v holds"), b_v : base rate of v
               (how much LESS often than by chance v holds when v' holds)
               f_up = (n+ + 2 sqrt(n+ + 1)) / N : the frequency two standard errors up, so that sampling noise
               between unrelated properties does not look like exclusion (with plain f it does, a little)

Phase 1  the mind looks at objects through two STATE channels (each reports exactly one symbol per look: chi = 1 by
         the adapter's declaration).  Channel A has 5 symbols, channel B has 3; an object has one of each, fixed.
Phase 2  a new object X, never looked at.  A teller of reliability 0.3 says "X a0".  Then one more experience:
           (i)   the same teller says "X b1"          (a property on the OTHER channel: compatible)
           (ii)  a second teller says "X a3"          (another symbol of the SAME channel: incompatible)
           (iii) the same teller says "X a3"
           (iv)  an own look at X: channel A reports a3
"""
import numpy as np
from evm_common import read, par, coupling, source_evidence, expect

RHO = 0.3


def phase1(seed=0, n_obj=200, n_looks=2000, nA=5, nB=3):
    rng = np.random.default_rng(seed)
    colA = rng.integers(nA, size=n_obj); colB = rng.integers(nB, size=n_obj)
    P = nA + nB                                   # property units: a0..a4, b0..b2
    n_pos = np.zeros((P, P))                      # n+[v' -> v]: looks in which both were present
    N = np.zeros(P)                               # N[v', channel of v]: looks in which v' was present (both channels report)
    occ = np.zeros(P); total = 0.0
    for _ in range(n_looks):
        x = int(rng.integers(n_obj))
        present = [int(colA[x]), nA + int(colB[x])]
        total += 1.0
        for u in present:
            N[u] += 1.0; occ[u] += 1.0
            for v in present:
                if v != u:
                    n_pos[u, v] += 1.0
    b = occ / total                               # base rate of each property on its channel
    f = n_pos / np.maximum(N[:, None], 1e-300)    # state channels: chi = 1, so E = N and f = n+/N
    f_up = (n_pos + 2.0 * np.sqrt(n_pos + 1.0)) / np.maximum(N[:, None], 1e-300)
    c = (N / (N + 1.0))[:, None] * np.ones((1, P))
    x_plain = c * np.maximum(0.0, 1.0 - f / np.maximum(b[None, :], 1e-300))
    x = c * np.maximum(0.0, 1.0 - f_up / np.maximum(b[None, :], 1e-300))
    np.fill_diagonal(x, 0.0); np.fill_diagonal(x_plain, 0.0)
    return x, b, nA, x_plain


def corrected(sources, b=0.5):
    Wp = Wn = 0.0
    for s in sources:
        a, d = source_evidence(*s)
        Wp += a; Wn += d
    W = Wp + Wn
    return (Wp / W if W > 0 else float("nan")), W / (W + 1.0), expect(Wp, Wn, b)


def main():
    x, b, nA, x_plain = phase1()
    a0, a3, b1 = 0, 3, nA + 1
    print("== phase 1: exclusion learned from 2000 looks through two state channels (no rule about them in the core) ==")
    print(f"   x(a3, a0) = {x[a3, a0]:.3f}   (two symbols of the same channel: never together)")
    print(f"   x(b1, a0) = {x[b1, a0]:.3f}   (symbols of different channels: together about as often as chance)")
    print(f"   mean exclusion within channel A = {np.mean([x[i, j] for i in range(nA) for j in range(nA) if i != j]):.3f};"
          f"  across channels = {np.mean([x[i, j] for i in range(nA) for j in range(nA, len(b))]):.3f}")
    print(f"   (with plain f instead of f_up: within {np.mean([x_plain[i, j] for i in range(nA) for j in range(nA) if i != j]):.3f},"
          f" across {np.mean([x_plain[i, j] for i in range(nA) for j in range(nA, len(b))]):.3f}: sampling noise reads as weak exclusion)")

    print("\n== phase 2: 'X a0' told once by a teller of reliability 0.3; then one more experience ==")
    print("   experience                                   | DESIGN (§2.5): f, c, e of X -> a0 | CORRECTED: f, c, e of X -> a0")
    d0 = read(RHO * 1.0, 0.0)
    c0 = corrected([(1.0, 0.0, 1.0, 0.0, RHO)])
    print(f"   (start) told once                            |   {d0[0]:.2f}  {d0[1]:.3f}  {d0[2]:.3f}             |   {c0[0]:.2f}  {c0[1]:.3f}  {c0[2]:.3f}")
    # design: any moment with X in which a0 does not occur adds w- += e (told) / (sensed)
    e0 = d0[2]
    d_told = read(RHO * 1.0, RHO * e0)
    d_look = read(RHO * 1.0, 1.0 * e0)
    rows = [
        ("(i)   same teller: 'X b1'   (compatible)", d_told,
         corrected([(1.0, x[b1, a0], 2.0, 0.0, RHO)])),
        ("(ii)  second teller: 'X a3' (incompatible)", d_told,
         corrected([(1.0, 0.0, 1.0, 0.0, RHO), (0.0, x[a3, a0], 1.0, 0.0, RHO)])),
        ("(iii) same teller: 'X a3'", d_told,
         corrected([(1.0, x[a3, a0], 2.0, 0.0, RHO)])),
        ("(iv)  own look: channel A shows a3", d_look,
         corrected([(1.0, 0.0, 1.0, 0.0, RHO), (0.0, 0.0, 1.0, 1.0, 1.0)])),
    ]
    for name, d, c in rows:
        print(f"   {name:44s} |   {d[0]:.2f}  {d[1]:.3f}  {d[2]:.3f}             |   {c[0]:.2f}  {c[1]:.3f}  {c[2]:.3f}")
    print("   -> the design moves the belief by the same amount for a compatible and for an incompatible statement")
    print("      (rows i-iii are identical): it reacts to the absence of the word, not to the contradiction.")
    print("      The corrected arithmetic leaves the belief alone in (i) and refines it in (ii)-(iv): f moves, c rises,")
    print("      nothing is deleted, and the own look weighs more than the second teller.")


if __name__ == "__main__":
    main()
