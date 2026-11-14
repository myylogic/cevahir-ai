#!/usr/bin/env python3
"""evm_s5_reality.py -- is the leave-path-out reality score of LEAD_DESIGN §6 well defined on a graph with shared ties?

Design text under attack (§6):
    "Each constructed conclusion (a tie instance i -> v reached by a chain of ties) gets a reality score
       R = e(i -> v) computed from the evidence that does NOT lie on the chain that produced it (leave-path-out)."
    "Later sensed or told evidence about the same tie confirms or refutes it, and that outcome updates (by C3) the
     trust tau in the source 'my own construction'"
together with §4 (Up: g -> v is built from the instances' ties; Down: lambda * e(i->g) * W(g->v)).

Four small graphs, computed with the design's own formulas, and one simulation.
"""
import numpy as np
from evm_common import read

LAM = 0.5


def e_of(Wp, Wn):
    return read(Wp, Wn)[2]


def g1():
    print("== 1. two routes that share NO tie and ALL their evidence.")
    print("      i is a member of g and of g'.  Both have the same six other members j1..j6; each j was seen once with v.")
    print("      (§3 creates a new category unit for every frame slot, so such twins are the normal case.) ==")
    S = 6 * 1.0                                   # Up: sum of min(W_j, cap=1)
    print(f"   W(g -> v) = W(g' -> v) = {S:.0f}   (the same six observations, copied twice by the Up rule)")
    print("   the conclusion i -> v is produced by the chain  i -> g -> v.")
    print("   'evidence that does not lie on the chain' = own evidence of i -> v (none) + what the Down rule brings")
    print("   through every other route, here i -> g' -> v:")
    for D in (1, 2, 5):
        W = LAM * S * D
        print(f"     {D} twin categor{'y' if D == 1 else 'ies'} besides g:  W_outside = {W:4.1f}  ->  R = {e_of(W, 0):.3f}")
    print("   by EVIDENCE the outside is empty (the six observations are the chain): R should be 'unknown', not 0.875-0.97.")
    print("   Leave-path-out removes ties; the Up rule has already copied their evidence onto other ties.")


def g2():
    print("\n== 2. 'the chain that produced it' is not defined by §2.2 (settling sums ALL incoming ties in parallel).")
    print("      i is a member of g (six members seen with v) and of h (four members seen WITHOUT v). ==")
    Wg_p = 6.0; Wh_n = 4.0
    R_if_g = e_of(0.0, LAM * Wh_n)                # chain = via g  -> outside evidence = via h
    R_if_h = e_of(LAM * Wg_p, 0.0)                # chain = via h  -> outside evidence = via g
    both = e_of(LAM * Wg_p, LAM * Wh_n)
    print(f"   if the producing chain is i -> g -> v:  R = e from the h-route alone = {R_if_g:.3f}")
    print(f"   if the producing chain is i -> h -> v:  R = e from the g-route alone = {R_if_h:.3f}")
    print(f"   what settling actually computes (both routes at once):               e = {both:.3f}")
    print("   Same state, same tie, three numbers.  R is a function of an attribution the dynamics do not make.")


def g3():
    print("\n== 3. shared prefix.  Two chains  i -> a -> b -> v  and  i -> a -> c -> v  (EXPECT ties). ==")
    print("   Leaving out the first chain removes the tie i -> a.  The second chain starts with the same tie, so nothing")
    print("   reaches v any more:  W_outside = 0  ->  R = 0.500, although a -> c -> v is independent evidence for the")
    print("   second half.  With one category (or one context) per instance this is the general case: every route out of")
    print("   i shares its first tie, and R = 1/2 for every conclusion that is not already known directly.")


def g4():
    print("\n== 4. the conclusion's own evidence is on BOTH sides.  i was seen three times with v; g has i and one other")
    print("      member j seen once without v. ==")
    # Up (cap = 1): g -> v gets min(3,1)=1 positive from i and 1 negative from j
    f_chain = 1.0 / 2.0
    f_chain_loo = 0.0
    R = e_of(3.0, 0.0)
    print(f"   chain i -> g -> v says f = {f_chain:.2f} (i's own evidence is half of the rule);  without i it says f = {f_chain_loo:.2f}")
    print(f"   R = e from i's own evidence = {R:.3f}.   The 'outside' evidence (i's sightings) is also inside the chain.")


def sim():
    print("\n== 5. self-trust learned in a world with NO regularity.  Members of a category have v or not, independently,")
    print("      with probability 1/2.  Every member is seen once.  The mind then constructs  i -> g -> v  for each member")
    print("      and the senses check it later (the member's value does not change).  Hit rate of the construction: ==")
    print("   m    design: rule contains i's own evidence   leave-one-out: rule built without i   (chance = 0.500)")
    rng = np.random.default_rng(4)
    for m in (2, 3, 5, 10, 30, 100):
        y = rng.random((40000, m)) < 0.5
        tot = y.sum(1, keepdims=True)
        f_in = tot / m
        f_out = (tot - y) / (m - 1)
        def rate(f):
            f = np.broadcast_to(f, y.shape)
            decided = f != 0.5
            hit = ((f > 0.5) == y) & decided
            return hit.sum() / max(1, decided.sum()), decided.mean()
        a, da = rate(f_in); b, db = rate(f_out)
        print(f"  {m:3d}        {a:.3f}  (decided {da:4.0%})                      {b:.3f}  (decided {db:4.0%})")
    print("   -> the design's self-trust rises above chance exactly where nothing can be constructed, most for small")
    print("      categories: the construction is 'confirmed' by the member's own earlier observation, which the Up rule")
    print("      put into the rule.  Trust in 'my own construction' is identifiable only if the referee evidence is")
    print("      excluded from the construction (leave-one-out by evidence, not by tie).")


if __name__ == "__main__":
    g1(); g2(); g3(); g4(); sim()
