#!/usr/bin/env python3
"""evm_s4_trust.py -- is the learned trust of LEAD_DESIGN §1 identifiable, and is it the right weight?

Design text under attack (§1):
    "Evidence on a tie is a vector over provenance classes Pi = {sensed, told, imitated, acted, inferred, imagined}"
    "Trust tau_pi in [0,1] is NOT a constant: tau for a source is the expectation of the tie (source unit -> 'holds'),
     learned by C3 (told statements later confirmed or refuted by sensed evidence change the trust in the teller ...)"
    "W+ = SUM_pi tau_pi w+[pi],  W- = SUM_pi tau_pi w-[pi]"
    "one told piece of evidence with tau_told = 0.3 gives c = 0.3/1.3 = 0.23"

To isolate the trust arithmetic from the closed-world problem (evm_s1), denials are GIVEN to the design here as w-
(as if events carried a polarity), and the trust tie is updated with unweighted counts.

Parts
  a  sources at chance: tau = hit rate counts noise as evidence
  b  six class slots cannot hold "trust in the teller": an honest and a lying teller in the class 'told'
  c  the birth value and the gauge: an evidence-free trust tie has expectation 1/2 for EVERY class
  d  one teller repeating itself: conviction by repetition
  e  self-trust earned by the base rate: constructions that add nothing are 'confirmed' most of the time
"""
import math
import numpy as np
from evm_common import read, par, reliability, coupling, expect


def world(rng, n_prop, precisions, sensed_frac=0.2):
    y = rng.random(n_prop) < 0.5                                 # truth of each proposition ('holds')
    claims = np.zeros((len(precisions), n_prop), bool)
    for s, q in enumerate(precisions):
        right = rng.random(n_prop) < q
        claims[s] = np.where(right, y, ~y)
    sensed = rng.random(n_prop) < sensed_frac
    return y, claims, sensed


def hit_rates(y, claims, sensed):
    """trust tie of each source: confirmed / refuted by sensed evidence (unweighted counts, prior-free readout)"""
    a = (claims[:, sensed] == y[sensed]).sum(1).astype(float)
    d = (claims[:, sensed] != y[sensed]).sum(1).astype(float)
    return (a + 0.5) / (a + d + 1.0)


def fuse(claims, w):
    Wp = (w[:, None] * claims).sum(0)
    Wn = (w[:, None] * (~claims)).sum(0)
    return (Wp + 0.5) / (Wp + Wn + 1.0), (Wp + Wn) / (Wp + Wn + 1.0)


def report(name, e, c, y, mask):
    yy = y[mask].astype(float); ee = e[mask]
    acc = float((((ee > 0.5) & (yy > 0.5)) | ((ee < 0.5) & (yy < 0.5))).mean())
    print(f"     {name:44s} right side of 1/2: {acc:5.1%}   Brier {((ee - yy) ** 2).mean():.3f}   mean c {c[mask].mean():.3f}")


def part_a():
    print("== a. one source that is right 90% of the time, and M sources that are right 50% of the time (pure noise).")
    print("      Every source makes one polar claim on each of 4000 propositions; 20% of them are also sensed. ==")
    for M in (0, 3, 10, 30):
        rng = np.random.default_rng(100 + M)
        prec = [0.9] + [0.5] * M
        y, claims, sensed = world(rng, 4000, prec)
        A = hit_rates(y, claims, sensed)
        un = ~sensed
        print(f"   M = {M:2d} noise sources;  learned hit rates: good {A[0]:.3f}" + (f", noise mean {A[1:].mean():.3f}" if M else ""))
        e, c = fuse(claims, A)                                   # design, GIVEN one slot per source: tau = expectation
        report("design, tau = expectation of trust tie", e, c, y, un)
        rho = np.maximum(0.0, 2 * A - 1)
        e, c = fuse(claims, rho)                                 # corrected: par(1, coupling(rho)) = rho per claim
        report("corrected, rho = chance-corrected", e, c, y, un)
        llr = np.log(np.clip(A, 1e-6, 1 - 1e-6) / np.clip(1 - A, 1e-6, 1 - 1e-6))
        z = (llr[:, None] * np.where(claims, 1.0, -1.0)).sum(0)
        eb = 1 / (1 + np.exp(-z))
        report("exact Bayes with the learned hit rates", eb, np.abs(2 * eb - 1), y, un)
    print("   -> a source at chance has tau = 1/2, so it is counted as half an observation per claim.  Ten such sources")
    print("      drown a 90% source (accuracy 90% -> 59%) and the mind reads conviction c ~ 0.85 on beliefs that")
    print("      are noise.  With rho = 2A - 1 the noise weighs nothing and c stays at what the good source can carry.")


def part_b():
    print("\n== b. the class 'told' holds an honest teller (right 95%) and a lying teller (right 5%), equally talkative.")
    print("      Six class slots (§1, §10: w+[6], w-[6]) leave room for ONE tau_told. ==")
    rng = np.random.default_rng(7)
    y, claims, sensed = world(rng, 4000, [0.95, 0.05])
    un = ~sensed
    A = hit_rates(y, claims, sensed)
    a = (claims[:, sensed] == y[sensed]).sum(); d = (claims[:, sensed] != y[sensed]).sum()
    tau_class = (a + 0.5) / (a + d + 1.0)
    print(f"   learned: tau_told (one class slot) = {tau_class:.3f};   per source: honest {A[0]:.3f}, liar {A[1]:.3f}")
    e, c = fuse(claims, np.array([tau_class, tau_class]))
    report("design, one slot for the class 'told'", e, c, y, un)
    und = float((e[un] == 0.5).mean())
    print(f"       (undecided, e = 1/2 exactly: {und:.1%} of the propositions: the two tellers cancel)")
    e, c = fuse(claims, A)
    report("design GIVEN per-source slots, tau = hit rate", e, c, y, un)
    e, c = fuse(claims, np.maximum(0, 2 * A - 1))
    report("corrected, per source, rho = 2A - 1", e, c, y, un)
    print("   -> with class slots the mind learns NOTHING from being told (the liar's evidence cannot be separated after it")
    print("      was added into the same slot, and it cannot be re-weighted later).  'Trust in the teller' needs evidence")
    print("      indexed by source unit.  Per-source tau = hit rate still counts the liar at 0.05 per claim; rho does not.")


def part_c():
    print("\n== c. birth value and gauge. ==")
    f, c, e = read(0.0, 0.0)
    print(f"   an evidence-free trust tie has expectation e = {e:.2f}  ->  at birth tau_sensed = tau_told = tau_imagined = 0.5")
    print(f"   first hearing by the design's own definition: W = 0.5 -> c = {0.5/1.5:.3f}   (the text says 0.3/1.3 = 0.231)")
    print(f"   first SIGHTING by the same definition:        W = 0.5 -> c = {0.5/1.5:.3f}   (a sighting and a hearsay weigh the same)")
    print("   tau_told = 0.3 needs evidence on the trust tie, e.g. W- = 2/3 with W+ = 0:", f"e = {(0+0.5)/(2/3+1):.3f}")
    print("   The trust tie of 'sensed' is refereed by nothing (the text names sensed evidence as the referee of the others).")
    print("   If C3 referees it by itself in a world whose regularities hold with frequency p, tau_sensed -> ~p:")
    for p in (0.5, 0.7, 0.9):
        t = 0.5
        n = 2000.0
        for _ in range(200):
            t = (t * p * n + 0.5) / (t * n + 1.0)
        print(f"     p = {p}: tau_sensed = {t:.3f}  -> every sensed count is scaled by {t:.2f}; a tie seen 10 times reads c = {t*10/(t*10+1):.3f} instead of {10/11:.3f}")
    print("   f = W+/W is unchanged when all tau are scaled by a common factor, c is not: the unit of evidence is undefined")
    print("   until one class is fixed (gauge).")


def part_d():
    print("\n== d. ONE teller (tau_told = 0.3) repeats the same statement n times. ==")
    print("   n     design W   c       | corrected W = par(n, rho/(1-rho))   c      | against ONE own contrary sighting: design f, e | corrected f, e")
    K = coupling(0.3)
    for n in (1, 2, 4, 10, 100, 1000):
        Wd = 0.3 * n
        Wc = par(float(n), K)
        fd, cd, ed = read(Wd, 1.0)
        fc, cc, ec = read(Wc, 1.0)
        print(f"  {n:5d}   {Wd:7.1f}   {Wd/(Wd+1):.3f}   |        {Wc:.3f}                      {Wc/(Wc+1):.3f}   |          {fd:.2f}  {ed:.2f}                  |   {fc:.2f}  {ec:.2f}")
    print("   -> in the design four repetitions outweigh one own sighting, and every repetition is felt as relief (§5:")
    print("      c rises).  With the series coupling a belief known from one source alone can never exceed c = rho.")


def part_e():
    print("\n== e. self-trust earned by the base rate.  In this world 90% of all (instance, property) ties hold anyway.")
    print("      The mind's constructions are NOT informative: it concludes 'i -> v' for random pairs.  Later the senses")
    print("      check them. ==")
    rng = np.random.default_rng(11)
    n = 3000
    holds = rng.random(n) < 0.9
    a = float(holds.sum()); d = float((~holds).sum())
    tau = (a + 0.5) / (a + d + 1.0)
    b = 0.9
    rho = reliability(tau, b)
    print(f"   design: tau('my own construction') = {tau:.3f}  -> every construction is written at {tau:.2f} of a sensed observation")
    Wd = tau * 1.0 * 0.5                                         # iota = 1, R = 1/2 (novel)
    print(f"           a never-observed conclusion after ONE construction: W = {Wd:.2f}, c = {Wd/(Wd+1):.2f}; after 10 idle cycles see evm_s3 B")
    print(f"   corrected: base rate b = {b:.2f}, precision A = {tau:.3f} -> rho = (A - b)/(1 - b) = {rho:.3f}")
    print(f"           the same conclusion: W = {par(1.0, coupling(rho)):.3f}, c = {par(1.0, coupling(rho))/(par(1.0, coupling(rho))+1):.3f},"
          f" e = {expect(par(1.0, coupling(rho)), 0.0, b):.3f} (the base rate; nothing was learned by constructing)")
    print("   -> a hit rate is not a reliability.  The top prediction of a mind that has learned the base rates is confirmed")
    print("      most of the time; counting those hits as trust and then multiplying NEW evidence by it counts the base")
    print("      rate twice.")


if __name__ == "__main__":
    part_a(); part_b(); part_c(); part_d(); part_e()
