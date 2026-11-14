#!/usr/bin/env python3
"""evm_common.py -- shared arithmetic for the EVIDENCE-MATH attack on LEAD_DESIGN.md (Zihin v0).

Two arithmetics live here, side by side, so that every simulation runs both on the identical stream:

  DESIGN   the equations exactly as written in LEAD_DESIGN.md §1, §2.5, §4, §6 (readings stated per script)
  CORRECTED the arithmetic proposed in attack/evidence-math.md, section "Corrected arithmetic"

Nothing here is product code.  Pure Python/NumPy, deterministic, single thread.
"""
import math

K = 1.0          # the design's k (evidential horizon), §1


# ----------------------------------------------------------------------------- DESIGN §1 readout
def read(Wp, Wn, k=K):
    """(f, c, e) of a tie from trust-weighted evidence.  e is written in the form that is defined at W = 0;
    it is algebraically identical to the design's c*(f - 1/2) + 1/2 whenever W > 0."""
    W = Wp + Wn
    c = W / (W + k)
    f = Wp / W if W > 0 else float("nan")
    e = (Wp + k / 2.0) / (W + k)
    return f, c, e


def e_design_literal(Wp, Wn, k=K):
    """the design's formula evaluated literally in IEEE float64 (shows the W = 0 boundary case)"""
    W = Wp + Wn
    c = W / (W + k)
    f = Wp / W if W != 0 else float("nan")
    return c * (f - 0.5) + 0.5


# ----------------------------------------------------------------------------- CORRECTED primitives
def par(a, b):
    """series combination a || b = ab/(a+b): evidence a that has to pass a coupling of strength b.
    par(a, inf) = a (no bottleneck), par(a, 0) = 0."""
    if a <= 0.0 or b <= 0.0:
        return 0.0
    if math.isinf(b):
        return a
    if math.isinf(a):
        return b
    return a * b / (a + b)


def reliability(A, b=0.5):
    """chance-corrected reliability of a source.  A = expectation of its trust tie ("what this source asserts
    holds", judged only by evidence that did not come from the source).  b = base rate: how often a tie of that
    kind holds anyway.  A source whose assertions hold no more often than the base rate carries no evidence.
    With b = 1/2 this is 2A - 1."""
    if b >= 1.0:
        return 0.0
    return min(1.0, max(0.0, (A - b) / (1.0 - b)))


def expect(Wp, Wn, b=0.5, k=K):
    """expectation with a base rate: e = (W+ + b k)/(W + k).  b = 1/2 gives the design's c (f - 1/2) + 1/2.
    With W = coupling(rho) and f = 1 this returns exactly A = rho + (1 - rho) b: the precision of the source."""
    return (Wp + b * k) / (Wp + Wn + k)


def coupling(rho, k=K):
    """odds form: the most evidence a source of reliability rho can ever put on one tie.
    par(1, coupling(rho)) = rho  (one testimony = rho units)   and   c_max = coupling/(coupling+k) = rho."""
    if rho >= 1.0:
        return math.inf
    if rho <= 0.0:
        return 0.0
    return k * rho / (1.0 - rho)


def source_evidence(n_pos, n_neg, n_opp, chi, rho, k=K):
    """Evidence one source puts on one tie, in units of own direct observations.
       n_pos : raw count of assertions / observed occurrences of v with u by this source
       n_neg : raw count of explicit observed absences (closed observation or learned exclusion)
       n_opp : raw count of moments this source reported on u (opportunities to mention v)
       chi   : learned completeness of the source (1 = closed world, 0 = open world)
       rho   : learned chance-corrected reliability of the source (1 for the mind's own closed senses)
    returns (W+, W-) after the series coupling."""
    E = max(n_pos + n_neg, chi * n_opp)            # effective number of observations of 'does v hold?'
    if E <= 0.0:
        return 0.0, 0.0
    f = n_pos / E
    W = par(E, coupling(rho, k))
    return W * f, W * (1.0 - f)
