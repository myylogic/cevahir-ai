#!/usr/bin/env python3
"""evm_s6_boundary.py -- boundary cases of the evidence arithmetic of LEAD_DESIGN.md.

Design text under attack:
    §9 (iv) "exact retraction of any single experience (evidence is additive)"
    §10     "Every experience's evidence increments are logged (moment id -> tie increments), so one experience can be
             retracted exactly by subtracting them."   "Float64 evidence."
    §1      "f = W+/W",  "e = c*(f - 1/2) + 1/2"
    §2.5    "v occurred: w+[pi] += g*a_u   (pi = provenance of the confirming event)"

Parts
  A  retraction: subtracting the logged increments of an experience vs the life in which it never happened
  B  float64: is "subtract the increments" exact?
  C  W = 0: the literal formula, and what a NaN does to kWTA
  D  conflicting provenance: an imagined antecedent and a sensed consequent
  E  evidence weighted by a settled activation (lag-0 ties)
"""
import math
import numpy as np
from evm_common import read, e_design_literal, par, coupling, source_evidence


# ------------------------------------------------------------------------------------------- part A
def design_life(stream, tau_told=0.3):
    """one tie u->v under §2.5.  stream items: ('told', 1) = told that u->v; ('sensed', 1/0) = u occurred and v did /
    did not follow.  Returns the state and the log of increments per item."""
    w = {"told+": 0.0, "told-": 0.0, "sensed+": 0.0, "sensed-": 0.0}
    exists = False
    log = []
    for kind, y in stream:
        inc = {}
        if y == 1:
            inc[kind + "+"] = 1.0                      # v occurred (also creates the tie)
            exists = True
        elif exists:
            Wp = tau_told * w["told+"] + w["sensed+"]; Wn = tau_told * w["told-"] + w["sensed-"]
            inc[kind + "-"] = (Wp + 0.5) / (Wp + Wn + 1.0)     # predicted, did not occur: g*a*e
        for k_, v_ in inc.items():
            w[k_] += v_
        log.append(inc)
    return w, log, exists


def show(tag, w, tau_told=0.3):
    Wp = tau_told * w["told+"] + w["sensed+"]; Wn = tau_told * w["told-"] + w["sensed-"]
    f, c, e = read(Wp, Wn)
    fs = "  nan" if math.isnan(f) else f"{f:.3f}"
    print(f"   {tag:58s} W+ = {Wp:.3f}  W- = {Wn:.3f}  f = {fs}  c = {c:.3f}  e = {e:.3f}")


def part_a():
    print("== A. retraction.  Experience x = 'told once that u -> v'.  Then u occurs five times and v does not follow")
    print("      (sensed).  Then x is retracted. ==")
    stream = [("told", 1)] + [("sensed", 0)] * 5
    w, log, _ = design_life(stream)
    show("design, life with x, before retraction", w)
    for k_, v_ in log[0].items():
        w[k_] -= v_
    show("design, after subtracting the logged increments of x", w)
    w0, _, exists = design_life(stream[1:])
    show("design, the life in which x never happened", w0)
    print(f"      (in that life the tie u -> v does not even exist: exists = {exists})")
    print("   -> what is left after 'exact' retraction is 1.94 units of sensed evidence AGAINST a relation the mind would")
    print("      never have considered: c = 0.66 that u -> v is false (e = 0.17).  The five negative increments were")
    print("      weighted by an expectation that x had created; they are logged under the five later moments, not under x.")
    print("      The same holds for everything x changed downstream: trust-tie increments of the teller, chunk and")
    print("      recruitment decisions (thresholds on e), caps, and every inferred/imagined deposit computed while x was in.")
    # corrected: raw counts; silence is not stored on the tie at all
    print("   corrected (raw counts n+ per source on the tie; opportunities N on the unit u, per reporting channel):")
    n_told, N_sense, n_sense = 1.0, 5.0, 0.0
    Wp_t, Wn_t = source_evidence(n_told, 0.0, 1.0, chi=0.0, rho=0.3)
    Wp_s, Wn_s = source_evidence(n_sense, 0.0, N_sense, chi=1.0, rho=1.0)
    f, c, e = read(Wp_t + Wp_s, Wn_t + Wn_s)
    print(f"      with x:      told W+ = {Wp_t:.3f}; closed sense: 0 of 5 -> W- = {Wn_s:.1f};  f = {f:.3f}, c = {c:.3f}, e = {e:.3f}")
    Wp_s, Wn_s = source_evidence(0.0, 0.0, N_sense, chi=1.0, rho=1.0)
    f, c, e = read(Wp_s, Wn_s)
    print(f"      x retracted: n+_told = 0;        closed sense: 0 of 5 -> W- = {Wn_s:.1f};  f = {f:.3f}, c = {c:.3f}, e = {e:.3f}")
    print("      never told:  identical numbers (the five opportunities are a property of u's own history, counted whether")
    print("      or not a tie to v exists; a question 'u -> v?' reads n+ = 0 and N = 5 in both lives).")


# ------------------------------------------------------------------------------------------- part B
def part_b():
    print("\n== B. float64.  1000 increments g*a_u (random in (0,1]) are added to one slot in life order; then 999 of them are")
    print("      retracted in a different order.  Residual error against the one increment that remains: ==")
    rng = np.random.default_rng(0)
    errs = []; neg = 0
    for trial in range(2000):
        x = rng.random(1000) * 0.999 + 0.001
        total = 0.0
        for v in x:
            total += v
        keep = int(rng.integers(1000))
        for i in rng.permutation(1000):
            if i != keep:
                total -= x[i]
        errs.append(total - x[keep])
        total -= x[keep]
        if total < 0:
            neg += 1
    errs = np.abs(np.array(errs))
    print(f"   exact in {np.mean(errs == 0):.1%} of 2000 trials; max |error| = {errs.max():.2e}; after retracting everything the slot is")
    print(f"   NEGATIVE in {neg/2000:.1%} of trials (violates w >= 0; f = W+/W then has the wrong sign or divides by ~0).")
    print("   'Exact' needs integer (fixed-point) counts or recomputation from the log, not float subtraction.")


# ------------------------------------------------------------------------------------------- part C
def part_c():
    print("\n== C. W = 0.  A tie with no trust-weighted evidence (after retraction of its only experience; a PART or ABSTRACT")
    print("      tie at creation; a candidate tie created while the plasticity knob g = 0). ==")
    with np.errstate(invalid="ignore", divide="ignore"):
        lit = e_design_literal(0.0, 0.0)
    print(f"   literal §1:  f = 0/0 = nan,  c = 0,  e = c*(f - 1/2) + 1/2 = {lit}    (0 * nan = nan in IEEE 754)")
    print(f"   same quantity written as (W+ + k/2)/(W + k):  e = {read(0.0, 0.0)[2]}")
    a = np.array([0.2, 0.9, float("nan"), 0.4])
    print(f"   kWTA on activations {a.tolist()}: np.argmax -> index {int(np.argmax(a))}", end="")
    try:
        import torch
        torch.set_num_threads(1)
        v, i = torch.topk(torch.tensor(a), 2)
        print(f";  torch.topk(k=2) -> indices {i.tolist()}, values {v.tolist()}")
    except Exception as ex:                                  # torch not needed for the point
        print(f";  (torch not available: {ex})")
    print("   -> one evidence-free tie puts a NaN into settling and the NaN unit takes the focus.")


# ------------------------------------------------------------------------------------------- part D
def part_d():
    print("\n== D. conflicting provenance.  While idle the mind IMAGINES u (u never occurred).  In the next moment the real")
    print("      event v arrives (sensed).  §2.5: 'v occurred: w+[pi] += g*a_u (pi = provenance of the confirming event)'. ==")
    print("   times this coincidence happened   w+[sensed](u -> v)   c       e(u -> v)")
    for n in (1, 3, 5, 20):
        f, c, e = read(float(n), 0.0)
        print(f"            {n:3d}                         {n:5.1f}          {c:.3f}    {e:.3f}")
    print("   -> the tie is booked as SENSED evidence that u is followed by v, at full trust, though u was never sensed.")
    print("      The same happens for a category unit (active only through ABSTRACT) and for an antecedent that settling")
    print("      ignited.  A tie increment has two ends; it is as real as the less real of them.")


# ------------------------------------------------------------------------------------------- part E
def part_e():
    print("\n== E. lag-0 ties ('holds together'): the weight g*a_u is the SETTLED activation of u, and settling lets v raise u")
    print("      (the reverse tie v -> u).  If a_u = a+ in moments where v is present and a- where it is absent, even with")
    print("      unweighted negatives the frequency converges to  p a+ / (p a+ + (1-p) a-): ==")
    print("   true p   a+    a-     f the mind reads")
    for p, ap, am in ((0.5, 1.0, 0.5), (0.3, 1.0, 0.5), (0.5, 1.0, 0.25), (0.2, 1.0, 0.25)):
        print(f"    {p:.2f}    {ap:.2f}  {am:.2f}      {p*ap/(p*ap+(1-p)*am):.3f}")
    print("   -> belief -> activation -> size of the evidence increment -> belief.  The evidence weight of an end must be")
    print("      its event-driven presence, not its settled activation.")


if __name__ == "__main__":
    part_a(); part_b(); part_c(); part_d(); part_e()
