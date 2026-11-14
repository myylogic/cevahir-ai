#!/usr/bin/env python3
"""evm_s1_closed_world.py -- negative evidence from "predicted, did not occur" (LEAD_DESIGN §2.5) is a
closed-world rule applied to an open-world stream.

Design text under attack:
    §2.5   v occurred:                      w+[pi] += g*a_u
           v was predicted, did not occur:  w-[pi] += g*a_u*e(u->v)
    §1     frequency f = W+/W  (how often it held)
           Contradiction does not delete: it adds W-, f moves, c rises
           told statements later confirmed or refuted by sensed evidence change the trust in the teller

Reading used (stated in the report): a statement "X v" is one moment whose focus holds the units X and v with
a = 1, g = 1; every existing tie out of a focus unit is a prediction (§2.3 names no threshold; a variant with the
threshold e > 1/2 is also run); tau_sensed = 1; tau_told = 0.3 where the design's own number is used, learned
(expectation of the trust tie, §1) where trust is the subject.

Parts
  A  a true fact told once, then other true facts about the same subject
  B  explicit denial "X not v"
  C  trust in a perfectly honest teller, judged "by sensed evidence" when the senses show one aspect at a time
  D  a generated world: beliefs about (object, property) pairs, DESIGN vs CORRECTED on the identical stream
"""
import math
from collections import defaultdict
import numpy as np
from evm_common import read, par, reliability, coupling, source_evidence, expect

S, T = 0, 1          # provenance index: sensed, told


def root(p):
    """stationary f of the design's rule on a tie whose target occurs with probability p"""
    return 1.0 if p >= 1 else (-p + math.sqrt(p * p + 4 * p * (1 - p))) / (2 * (1 - p))


# ------------------------------------------------------------------------------------------- part A
def part_a():
    print("== A1. told ONCE that X is v1 (tau_told = 0.3), then m other TRUE statements about X; v1 never denied ==")
    print("   m    W+     W-      f      c      e     | corrected (open world, chi_told = 0): f, c, e")
    tau = 0.3
    wp, wn = 1.0, 0.0
    for m in range(0, 51):
        f, c, e = read(tau * wp, tau * wn)
        if m in (0, 1, 2, 3, 5, 10, 20, 50):
            Wp, Wn = source_evidence(1.0, 0.0, 1.0 + m, chi=0.0, rho=0.3)
            fc, cc, ec = read(Wp, Wn)
            print(f"  {m:3d}  {tau*wp:5.2f}  {tau*wn:5.2f}  {f:6.3f} {c:6.3f} {e:6.3f}   |  {fc:5.3f} {cc:5.3f} {ec:5.3f}")
        wn += e                      # predicted (tie exists), did not occur
    print("   -> after TWO unrelated statements the design holds the uncontradicted fact at f < 1/2.")

    print("\n== A2. X has K permanent true properties; every statement names one of them (uniform); 4000 statements ==")
    print("   K   f*(theory)  f_sim   c_sim   e_sim   e_sim(threshold variant: only e>1/2 counts as predicted)")
    rng = np.random.default_rng(1)
    for Kp in (2, 3, 5, 10, 50):
        res = {}
        for variant in ("all", "thr"):
            wp = np.zeros(Kp); wn = np.zeros(Kp); born = np.zeros(Kp, bool)
            for t in range(4000):
                j = int(rng.integers(Kp))
                e = (tau * wp + 0.5) / (tau * (wp + wn) + 1.0)
                miss = born.copy(); miss[j] = False
                if variant == "thr":
                    miss &= e > 0.5
                wn[miss] += e[miss]
                wp[j] += 1.0; born[j] = True
            W = tau * (wp + wn)
            res[variant] = (float((wp / (wp + wn)).mean()), float((W / (W + 1)).mean()),
                            float(((tau * wp + 0.5) / (W + 1)).mean()))
        f, c, e = res["all"]
        print(f"  {Kp:3d}   {root(1.0/Kp):6.3f}     {f:6.3f}  {c:6.3f}  {e:6.3f}   {res['thr'][2]:6.3f}")
    print("   -> every one of these facts is TRUE and was never contradicted; the design ends convinced (c ~ 1)")
    print("      that each holds with frequency f*(1/K) < 1/2 for K >= 4; with the threshold variant it ends at")
    print("      e ~ 1/2 with c ~ 1, which §2.6 reports as 'conflict'.")


# ------------------------------------------------------------------------------------------- part B
def part_b():
    print("\n== B. explicit denial.  Stream: 'X v1' once, then n times 'X not v1' (symbols X, not, v1 in one moment) ==")
    print("   The only route to w- in §2.5 is 'predicted, did not occur'.  In a denial v1 DOES occur.")
    tau = 0.3
    print("   n denials   W+(X->v1)  W-(X->v1)   e(X->v1)")
    wp, wn = 1.0, 0.0
    for n in range(0, 11):
        if n in (0, 1, 2, 5, 10):
            f, c, e = read(tau * wp, tau * wn)
            print(f"     {n:3d}       {tau*wp:6.2f}     {tau*wn:6.2f}     {e:6.3f}")
        wp += 1.0                    # v1 occurred while X was in the focus
    print("   -> ten denials raise e(X->v1) from 0.615 to 0.884.  A told contradiction has no path to W-,")
    print("      while silence (part A) has one.  Both are the wrong way round.")


# ------------------------------------------------------------------------------------------- part C
def part_c():
    print("\n== C. trust in a 100% honest teller.  The teller told all K true properties of each of 200 objects once.")
    print("      Then the senses show an object with ONE of its true properties per moment (4000 moments). ==")
    print("   Design: sensed occurrence on a told tie confirms the teller; 'predicted, did not occur' refutes it.")
    print("   K   tau_honest(C3 literal: w- += e_trust)  tau_honest(plain: w- += 1)  theory f*(1/K), 1/K | tau_liar(literal)")
    rng = np.random.default_rng(2)
    for Kp in (2, 5, 10, 50):
        out = {}
        for mode in ("ew", "plain"):
            tp, tn = 0.0, 0.0                              # trust tie (teller -> holds), provenance sensed
            lp, ln = 0.0, 0.0                              # a second teller who asserted one FALSE property per object
            M = 200
            wp_s = np.zeros((M, Kp)); wn_s = np.zeros((M, Kp))
            wn_liar = np.zeros(M)                          # the liar's tie X -> v_false, only ever refuted
            for t in range(4000):
                x = int(rng.integers(M)); j = int(rng.integers(Kp))
                tau_h = (tp + 0.5) / (tp + tn + 1.0)
                tau_l = (lp + 0.5) / (lp + ln + 1.0)
                # honest ties of X: told evidence 1 each, sensed evidence so far
                e = (tau_h * 1.0 + wp_s[x] + 0.5) / (tau_h * 1.0 + wp_s[x] + wn_s[x] + 1.0)
                e_l = (tau_l * 1.0 + 0.5) / (tau_l * 1.0 + wn_liar[x] + 1.0)
                # confirmation of the told tie X -> v_j
                wp_s[x, j] += 1.0; tp += 1.0
                # every other told tie of X was predicted and did not occur
                for v in range(Kp):
                    if v != j:
                        wn_s[x, v] += e[v]
                        tn += tau_h if mode == "ew" else 1.0
                wn_liar[x] += e_l
                ln += tau_l if mode == "ew" else 1.0
            out[mode] = ((tp + 0.5) / (tp + tn + 1.0), (lp + 0.5) / (lp + ln + 1.0))
        print(f"  {Kp:3d}        {out['ew'][0]:6.3f}                          {out['plain'][0]:6.3f}"
              f"                 {root(1.0/Kp):6.3f}  {1.0/Kp:6.3f} |   {out['ew'][1]:6.3f}")
    print("   -> the trust earned by an honest teller measures how many properties the objects have (the co-mention")
    print("      rate), not honesty.  For K >= 4 an honest teller ends BELOW 1/2.")
    print("   corrected, same stream: the sight channel's completeness is its own repetition rate; a glimpse of another")
    print("   aspect is then not a refutation.  Precision A of each teller, refereed by sight (prior 1.3/0.7):")
    print("   K    chi_sight (true 1/K)   A_honest   A_liar")
    rng = np.random.default_rng(2)
    for Kp in (2, 5, 10, 50):
        M = 200
        n = np.zeros((M, Kp)); N = np.zeros(M)
        for t in range(4000):
            x = int(rng.integers(M)); j = int(rng.integers(Kp))
            n[x, j] += 1.0; N[x] += 1.0
        said = n > 0
        Nx = np.broadcast_to(N[:, None], n.shape)
        chi = float((n - 1.0)[said].sum() / (1.0 + (Nx - 1.0)[said].sum()))
        E = np.maximum(n, chi * Nx)
        f_ref = np.where(E > 0, n / np.maximum(E, 1e-300), 0.0); c_ref = E / (E + 1.0)
        # honest teller: one assertion on each true pair (f_T = 1, c_T = 1/2); liar: one assertion on one false pair per object
        cT = 0.5
        a_h = 1.3 + float((c_ref * cT * f_ref).sum()); d_h = 0.7 + float((c_ref * cT * (1 - f_ref)).sum())
        E_l = chi * N; c_l = E_l / (E_l + 1.0)             # the liar's pair was never seen: n = 0, E = chi*N, f_ref = 0
        a_l = 1.3; d_l = 0.7 + float((c_l * cT).sum())
        print(f"  {Kp:3d}        {chi:.3f} ({1.0/Kp:.3f})        {a_h/(a_h+d_h):.3f}     {a_l/(a_l+d_l):.3f}")


# ------------------------------------------------------------------------------------------- part D
def make_world(rng, M=200, P=20, Kp=5, n_stmt=6000, q=1.0, insp_frac=0.15, looks=3):
    truth = np.zeros((M, P), bool)
    for x in range(M):
        truth[x, rng.choice(P, Kp, replace=False)] = True
    events = []
    for _ in range(n_stmt):
        x = int(rng.integers(M))
        if rng.random() < q:
            v = int(rng.choice(np.nonzero(truth[x])[0]))
        else:
            v = int(rng.choice(np.nonzero(~truth[x])[0]))
        events.append(("told", x, v))
    insp = rng.random((M, P)) < insp_frac
    for x, v in zip(*np.nonzero(insp)):
        for _ in range(looks):
            events.append(("look", int(x), int(v)))
    order = rng.permutation(len(events))
    return truth, insp, [events[i] for i in order]


def run_design(truth, events, trust_neg="ew"):
    """literal §1/§2.5 on (object -> property) ties; trust in the teller learned as in §1."""
    M, P = truth.shape
    wp = np.zeros((2, M, P)); wn = np.zeros((2, M, P)); born = np.zeros((M, P), bool)
    tp = tn = 0.0
    for kind, x, v in events:
        tau = (tp + 0.5) / (tp + tn + 1.0)
        Wp = wp[S, x] + tau * wp[T, x]; Wn = wn[S, x] + tau * wn[T, x]
        e = (Wp + 0.5) / (Wp + Wn + 1.0)
        told_tie = wp[T, x] > 0
        if kind == "told":
            miss = born[x].copy(); miss[v] = False
            wn[T, x, miss] += e[miss]
            wp[T, x, v] += 1.0; born[x, v] = True
        else:
            present = truth[x, v]
            miss = born[x].copy()
            if present:
                miss[v] = False
                wp[S, x, v] += 1.0; born[x, v] = True
                if told_tie[v]:
                    tp += 1.0
            # every (other) existing tie of X was predicted and did not occur in this sensed moment
            wn[S, x, miss] += e[miss]
            k_ref = int((miss & told_tie).sum())
            tn += k_ref * (tau if trust_neg == "ew" else 1.0)
    tau = (tp + 0.5) / (tp + tn + 1.0)
    Wp = wp[S] + tau * wp[T]; Wn = wn[S] + tau * wn[T]
    return (Wp + 0.5) / (Wp + Wn + 1.0), (Wp + Wn) / (Wp + Wn + 1.0), tau, born


def run_corrected(truth, events, prior_A=(1.3, 0.7), prior_chi=(0.0, 1.0), use_base=True):
    """raw counts only.  Reliability rho, completeness chi and base rate b are FUNCTIONS of the raw counts,
    refereed by the closed sense (leave-source-out by construction: the referee never contains the teller's
    evidence).  prior_A = (1.3, 0.7): precision 0.65 at birth = the design's tau_told = 0.3 at base rate 1/2."""
    M, P = truth.shape
    n_T = np.zeros((M, P)); N_T = np.zeros(M)                 # teller: assertions, opportunities (mentions of X)
    n_S = np.zeros((M, P)); N_S = np.zeros((M, P))            # closed look at (X, v): present count, look count
    for kind, x, v in events:
        if kind == "told":
            n_T[x, v] += 1.0; N_T[x] += 1.0
        else:
            N_S[x, v] += 1.0
            if truth[x, v]:
                n_S[x, v] += 1.0
    f_S = np.where(N_S > 0, n_S / np.maximum(N_S, 1e-300), 0.0)
    c_S = N_S / (N_S + 1.0)
    # base rate of 'holds' for ties of this kind, from the referee alone (weak prior at 1/2)
    b = (0.5 + float((c_S * f_S).sum())) / (1.0 + float(c_S.sum())) if use_base else 0.5
    # completeness: on pairs the referee holds true, how often does the teller name v when it names X?
    w = c_S * f_S
    chi = (prior_chi[0] + float((w * n_T).sum())) / (prior_chi[0] + prior_chi[1] + float((w * N_T[:, None]).sum()))
    # the teller's standing claim on every tie: amount E, frequency f_T (silence counted at completeness chi)
    E_T = np.maximum(n_T, chi * N_T[:, None])
    f_T = np.where(E_T > 0, n_T / np.maximum(E_T, 1e-300), 0.0)
    c_T = E_T / (E_T + 1.0)
    # trust = precision of what the teller consistently asserts; at most one unit per tie (c_S * c_T <= 1)
    a = prior_A[0] + float((c_S * c_T * f_T * f_S).sum())
    d = prior_A[1] + float((c_S * c_T * f_T * (1 - f_S)).sum())
    A = a / (a + d)
    rho = reliability(A, b)
    Kc = coupling(rho)
    e = np.zeros((M, P)); c = np.zeros((M, P))
    for x in range(M):
        for v in range(P):
            Wp_s, Wn_s = source_evidence(n_S[x, v], 0.0, N_S[x, v], chi=1.0, rho=1.0)
            Wt = par(E_T[x, v], Kc)
            Wp, Wn = Wp_s + Wt * f_T[x, v], Wn_s + Wt * (1 - f_T[x, v])
            e[x, v] = expect(Wp, Wn, b); c[x, v] = (Wp + Wn) / (Wp + Wn + 1.0)
    return e, c, A, rho, chi, b


def count_told(events, shape):
    n = np.zeros(shape)
    for kind, x, v in events:
        if kind == "told":
            n[x, v] += 1.0
    return n


def score(name, e, truth, insp, told):
    rows = []
    for label, mask in (("true , told    ", truth & told & ~insp), ("true , not told", truth & ~told & ~insp),
                        ("false, not told", ~truth & ~told & ~insp), ("false, told(lie)", ~truth & told & ~insp)):
        n = int(mask.sum())
        if n == 0:
            rows.append(f"      {label}  n=    0")
            continue
        y = truth[mask].astype(float); ee = e[mask]
        right = float((((ee > 0.5) & (y > 0.5)) | ((ee < 0.5) & (y < 0.5))).mean())
        und = float((ee == 0.5).mean())
        rows.append(f"      {label}  n={n:5d}  mean e={ee.mean():5.3f}  right side of 1/2: {right:5.1%}  undecided: {und:5.1%}"
                    f"  Brier={((ee - y) ** 2).mean():.3f}")
    y = truth[~insp].astype(float)
    print(f"   {name}:  Brier over all pairs never looked at = {((e[~insp] - y) ** 2).mean():.4f}")
    for r in rows:
        print(r)


def part_d():
    print("\n== D. generated world: 200 objects x 20 properties, 5 true per object; 6000 told statements (teller names")
    print("      one property per statement); 15% of pairs are also LOOKED at (closed sense: present/absent), 3 looks. ==")
    for q in (1.0, 0.7):
        rng = np.random.default_rng(3)
        truth, insp, events = make_world(rng, q=q)
        told = np.zeros_like(truth)
        for kind, x, v in events:
            if kind == "told":
                told[x, v] = True
        print(f"\n   teller precision q = {q}")
        e_d, c_d, tau_d, born = run_design(truth, events)
        print(f"   DESIGN    learned tau_told = {tau_d:.3f}   (true precision {q})")
        score("DESIGN   ", e_d, truth, insp, told)
        e_c, c_c, A, rho, chi, b = run_corrected(truth, events)
        print(f"   CORRECTED learned: base rate b = {b:.3f} (true 0.25); precision of consistent assertions A = {A:.3f}"
              f" -> reliability rho = {rho:.3f};  completeness chi = {chi:.3f} (true {q/5:.3f})")
        score("CORRECTED", e_c, truth, insp, told)
        e_n, c_n, A, rho, chi, b = run_corrected(truth, events, use_base=False)
        print(f"   CORRECTED without the base rate (b = 1/2 as in the design): A = {A:.3f}, rho = {rho:.3f}, chi = {chi:.3f}")
        score("CORRECTED(b=1/2)", e_n, truth, insp, told)
        # the open-world default when NO closed channel exists: chi = 0 (silence says nothing), rho at its birth prior
        n_told = count_told(events, truth.shape)
        W_o = np.vectorize(lambda n: par(n, coupling(0.3)))(n_told)
        e_o = np.where(told, expect(W_o, 0.0, 0.5), 0.5)
        score("CORRECTED, no closed channel (chi = 0, rho = birth prior 0.3)", e_o, truth, insp, told)
        # no closed channel, but completeness estimated from the teller's OWN repetition rate:
        # after it has said something about X once, how often does it say it again when it speaks about X?
        N_x = n_told.sum(1, keepdims=True) * np.ones_like(n_told)
        said = n_told > 0
        chi_self = float((n_told - 1.0)[said].sum() / (1.0 + (N_x - 1.0)[said].sum()))
        E_s = np.maximum(n_told, chi_self * N_x)
        f_s = np.where(E_s > 0, n_told / np.maximum(E_s, 1e-300), 0.0)
        W_s = np.vectorize(lambda n: par(n, coupling(0.3)))(E_s)
        e_s = expect(W_s * f_s, W_s * (1 - f_s), 0.5)
        print(f"   completeness from the teller's own repetition rate: chi = {chi_self:.3f}")
        score("CORRECTED, no closed channel (chi = own repetition rate, rho = birth prior 0.3)", e_s, truth, insp, told)


if __name__ == "__main__":
    part_a(); part_b(); part_c(); part_d()
