#!/usr/bin/env python3
"""evm_s2_updown.py -- LEAD_DESIGN §4 worked by hand and in simulation: induction (up), support (down), and the loop.

Design text under attack (§4):
    Up:   g acquires EXPECT g -> v with evidence = SUM_i min(W_i, cap) discounted by the ABSTRACT expectations.
    Down: W_eff+ = W_own+ + lambda * e(i->g) * W+(g->v), same for W-, lambda = 0.5, provenance "inferred".
    "an exception stays an exception (own negative evidence outweighs inherited support)"
and §2.2 / §2.5: the instance activates its categories ("up through ABSTRACT"), and LEARN updates
    "every tie u -> v with u in the last focus".

Parts
  a  the loop when the inherited evidence is what the text says it is: evidence of provenance "inferred" on the
     instance tie (reading A), so that W_i of the Up rule contains it
  b  no loop (reading B: the Up rule sees own evidence only): self-support, and duplicate slot categories
  c  the cap of the Up rule is bypassed by LEARN on the co-active category
  d  how many own contrary observations does an exception need?
  e  generated worlds: design vs corrected vs the exact Bayes answer of the generating model
"""
import math
import numpy as np
from evm_common import read, par, reliability, coupling

LAM = 0.5


# ------------------------------------------------------------------------------------------- part a
def loop_fixed_point(m, w_own, e_m, tau_inf, cap, iters=200):
    """reading A, 'set' semantics: the inferred slot of each instance tie is overwritten (not accumulated)."""
    Wg = 0.0
    for _ in range(iters):
        Wi = w_own + tau_inf * LAM * e_m * Wg
        Wg_new = m * e_m * min(Wi, cap)
        if abs(Wg_new - Wg) < 1e-12:
            break
        Wg = Wg_new
        if Wg > 1e12:
            return math.inf, math.inf
    return Wg, w_own + tau_inf * LAM * e_m * Wg


def part_a():
    print("== a. the up/down loop.  m instances, each TOLD ONCE that v holds (own W = 0.3, c = 0.23); e(i->g) = 1 ==")
    print("   closed form without cap:  W_g = m e W_own / (1 - m tau_inf lambda e^2);  loop gain G = m tau_inf lambda e^2")
    print("   m   tau_inf  cap    G     W_g(loop)  c_g     W_i(loop)  c_i   |  no loop, others only: W_g  c_i")
    for tau_inf in (0.5, 1.0):
        for cap in (math.inf, 1.0):
            for m in (1, 2, 3, 4, 5, 10, 30):
                G = m * tau_inf * LAM
                Wg, Wi = loop_fixed_point(m, 0.3, 1.0, tau_inf, cap)
                Wg0 = m * min(0.3, cap)
                Wi0 = 0.3 + tau_inf * LAM * (m - 1) * min(0.3, cap)
                cg = 1.0 if math.isinf(Wg) else Wg / (Wg + 1)
                ci = 1.0 if math.isinf(Wi) else Wi / (Wi + 1)
                sWg = "   inf" if math.isinf(Wg) else f"{Wg:6.2f}"
                sWi = "   inf" if math.isinf(Wi) else f"{Wi:6.2f}"
                print(f"  {m:3d}   {tau_inf:3.1f}   {('inf' if math.isinf(cap) else '1.0'):>4}  {G:5.2f}   {sWg}    {cg:5.3f}    {sWi}   {ci:5.3f}"
                      f" |   {Wg0:5.2f}  {Wi0/(Wi0+1):5.3f}")
    print("   -> with G >= 1 the fixed point does not exist (cap = inf) or is the cap itself: W_g = m*cap, whatever")
    print("      the instances actually know.  Five hearsays (c = 0.23 each) become c_i = 0.74 and c_g = 0.83 (tau_inf = 1, cap = 1);")
    print("      without the loop the same five hearsays give c_i = 0.47.")
    print("\n   'accumulate' semantics (C3: evidence is ADDED at every moment), m = 3, tau_inf = 0.5, cap = 1:")
    w_inf = 0.0
    line = []
    for t in range(1, 21):
        Wi = 0.3 + 0.5 * w_inf
        Wg = 3 * min(Wi, 1.0)
        w_inf += LAM * Wg                          # the same inherited evidence written again
        if t in (1, 2, 5, 10, 20):
            Wi = 0.3 + 0.5 * w_inf
            line.append(f"t={t}: W_i={Wi:5.2f} c_i={Wi/(Wi+1):.3f}")
    print("   " + " | ".join(line))


# ------------------------------------------------------------------------------------------- part b
def part_b():
    print("\n== b. reading B (no loop: Up sees own evidence only).  Own evidence still comes back, and categories with")
    print("      the same members (§3: every frame slot creates a NEW category unit for its fillers) are summed. ==")
    print("   10 instances, each one own sensed observation (W = 1), cap = 1, e(i->g) = 1")
    print("   duplicate categories D   W_eff(design)   c_eff    own share returned   | corrected (best route, LOO, coupling 1): W_eff  c")
    for D in (1, 2, 5, 10, 30):
        Wg = 10 * 1.0
        Weff = 1.0 + LAM * D * Wg
        back = LAM * D * 1.0
        Wc = 1.0 + par(9.0, 1.0)                   # choice among routes with the same base: the best one, once
        print(f"        {D:3d}              {Weff:7.1f}      {Weff/(Weff+1):5.3f}       {back:5.1f} units        |   {Wc:5.2f}  {Wc/(Wc+1):5.3f}")
    print("   -> conviction grows with the number of sentence frames the same fillers were met in, not with evidence.")


# ------------------------------------------------------------------------------------------- part c
def part_c():
    print("\n== c. the cap is bypassed.  ONE instance i of g is observed n times with v (sensed).  g is in the focus")
    print("      each time (activated through ABSTRACT, a_g = e(i->g) = 1), so LEARN writes g -> v directly. ==")
    print("   n     W(g->v) by Up (cap 1)   W(g->v) with LEARN on g   a NEVER OBSERVED j inherits   c_j     e_j")
    for n in (1, 10, 100, 1000):
        up = min(n, 1.0)
        direct = n * 1.0 + up
        inh = LAM * direct
        f, c, e = read(inh, 0.0)
        print(f"  {n:5d}        {up:5.1f}                 {direct:8.1f}              {inh:8.1f}            {c:5.3f}  {e:5.3f}")
    print("   -> one member seen a thousand times makes the mind 99.8% convinced about every other member.")


# ------------------------------------------------------------------------------------------- part d
def need_design(m, cap=1.0):
    n = 0
    while True:
        n += 1
        Wp = LAM * m * cap
        Wn = n + LAM * min(n, cap)
        if Wn > Wp:
            return n


def gated_readout(n_pos, n_neg, Sp, Sn, A_g, s=1.0, b=0.5):
    """corrected Down with per-instance conformity (the ABSTRACT tie's own evidence).
    Sp, Sn: rule evidence WITHOUT this instance.  A_g: conformity of the category (expectation of its trust tie)."""
    W = n_pos + n_neg
    S = Sp + Sn
    if S <= 0:
        return n_pos, n_neg
    fg = Sp / S
    cg = S / (S + 1.0)
    if W > 0:
        fi = n_pos / W
        ci = W / (W + 1.0)
        A_i = (s * A_g + ci * cg * (1.0 - abs(fi - fg))) / (s + ci * cg)
    else:
        A_i = A_g
    h = par(S, coupling(reliability(A_i, b)))
    return n_pos + h * fg, n_neg + h * (1.0 - fg)


def part_d():
    print("\n== d. 'an exception stays an exception'.  m members hold v (each W+ >= cap = 1); member x is SEEN not to. ==")
    print("   own contrary sensed observations x needs before e_eff(x -> v) < 1/2:")
    print("   m      design (n- > lambda(m-1))   corrected, bounded (coupling 1)   corrected, gated (A_g = 0.9)")
    for m in (2, 5, 10, 20, 100, 1000):
        nd = need_design(m)
        nb = 0
        while True:
            nb += 1
            h = par(float(m), 1.0)
            if nb > h:
                break
        ng = 0
        while True:
            ng += 1
            Wp, Wn = gated_readout(0.0, float(ng), float(m), 0.0, 0.9)
            if Wn > Wp:
                break
        print(f"  {m:5d}          {nd:5d}                          {nb:3d}                              {ng:3d}")
    print("   design, m = 20, after ONE sighting that x is not v:", end=" ")
    Wp = LAM * 20; Wn = 1 + LAM * 1
    f, c, e = read(Wp, Wn)
    print(f"W+ = {Wp:.1f}, W- = {Wn:.1f}, f = {f:.2f}, c = {c:.2f}, e = {e:.2f}  (the mind does not believe its eyes)")


# ------------------------------------------------------------------------------------------- part e
def gen_world(rng, G, m, q, mu_hi=0.95):
    n_choices = np.array([0, 1, 2, 3, 5, 10]); n_p = np.array([0.15, 0.2, 0.2, 0.15, 0.15, 0.15])
    mu = np.where(rng.random(G) < 0.5, mu_hi, 1 - mu_hi)
    conf = rng.random((G, m)) < q
    theta = np.where(conf, mu[:, None], 1 - mu[:, None])
    n = rng.choice(n_choices, size=(G, m), p=n_p)
    npos = rng.binomial(n, theta)
    return mu, conf, theta, n.astype(float), npos.astype(float)


def est_design_B(n, npos, cap=1.0):
    W = n; f = np.where(W > 0, npos / np.maximum(W, 1e-300), 0.0)
    u = np.minimum(W, cap)
    Sp = (u * f).sum(1, keepdims=True); Sn = (u * (1 - f)).sum(1, keepdims=True)
    Wp = npos + LAM * Sp; Wn = (n - npos) + LAM * Sn
    return (Wp + 0.5) / (Wp + Wn + 1.0)


def est_design_A(n, npos, cap=1.0, tau_inf=1.0, iters=100):
    """the loop: Up reads the instance's full evidence including what it inherited."""
    G, m = n.shape
    ip = np.zeros((G, m)); im = np.zeros((G, m))            # inherited (inferred slot), set semantics
    for _ in range(iters):
        Wp = npos + tau_inf * ip; Wn = (n - npos) + tau_inf * im
        W = Wp + Wn; f = np.where(W > 0, Wp / np.maximum(W, 1e-300), 0.0)
        u = np.minimum(W, cap)
        Sp = (u * f).sum(1, keepdims=True); Sn = (u * (1 - f)).sum(1, keepdims=True)
        ip = LAM * Sp * np.ones((1, m)); im = LAM * Sn * np.ones((1, m))
    Wp = npos + tau_inf * ip; Wn = (n - npos) + tau_inf * im
    return (Wp + 0.5) / (Wp + Wn + 1.0)


def loo_rule(n, npos, cap):
    W = n; f = np.where(W > 0, npos / np.maximum(W, 1e-300), 0.0)
    u = np.where(W > 0, W * cap / (W + cap), 0.0)           # par(W, cap): one instance is at most `cap` votes
    Sp_all = (u * f).sum(1, keepdims=True); Sn_all = (u * (1 - f)).sum(1, keepdims=True)
    return Sp_all - u * f, Sn_all - u * (1 - f)              # leave-one-out: exact, the sums are additive


def conformity(n, npos, Sp, Sn, prior=(1.5, 0.5), s_pool=2.0):
    """trust tie of 'inheritance in this category': agreement of each member's OWN evidence with the rule built
    WITHOUT that member, weighted by both convictions (at most one unit per member); pooled over the domain
    (support from the category of categories), then per category."""
    W = n; S = Sp + Sn
    fi = np.where(W > 0, npos / np.maximum(W, 1e-300), 0.0)
    fg = np.where(S > 0, Sp / np.maximum(S, 1e-300), 0.5)
    w = (W / (W + 1.0)) * (S / (S + 1.0))
    agree = 1.0 - np.abs(fi - fg)
    a = (w * agree).sum(1); d = (w * (1 - agree)).sum(1)
    A_dom = (prior[0] + a.sum()) / (prior[0] + prior[1] + a.sum() + d.sum())
    A_g = (s_pool * A_dom + a) / (s_pool + a + d)
    return A_dom, A_g, w, agree


def est_corrected(n, npos, mode, cap=1.0, base=True, s_inst=1.0, horizon="learned"):
    """corrected Down.  The rule is built WITHOUT the member (leave-one-out).  It gives the member
       (1) the base rate b of its expectation (replacing the fixed 1/2), and
       (2) a BOUNDED amount h of evidence:  h = par(S, coupling)   with coupling
             'base'    : 0            (base rate only; no evidence flows down at all)
                         base1 in the table = the same with the fixed horizon k_g = 1
             'bounded' : 1            (a rule is worth at most one own observation)
             'learned' : odds of the learned conformity of the category
             'gated'   : odds of the conformity of THIS member (category conformity updated by the member's own
                         agreement with the rule: the evidence of its ABSTRACT tie)"""
    Sp, Sn = loo_rule(n, npos, cap)
    S = Sp + Sn
    fg = np.where(S > 0, Sp / np.maximum(S, 1e-300), 0.5)
    A_dom, A_g, w, agree = conformity(n, npos, Sp, Sn)
    # horizon of rules in this domain: k_g = k (1 - rho)/rho, rho = chance-corrected conformity of the domain
    rho_dom = min(max(2 * A_dom - 1, 1e-3), 1 - 1e-3)
    k_g = (1 - rho_dom) / rho_dom if horizon == "learned" else 1.0
    bg = (Sp + 0.5 * k_g) / (S + k_g) if base else np.full_like(S, 0.5)
    if mode == "base":
        Kc = np.zeros_like(S)
    elif mode == "bounded":
        Kc = np.ones_like(S)
    else:
        A = np.repeat(A_g[:, None], n.shape[1], axis=1)
        if mode == "gated":
            a_tot = (w * agree).sum(1, keepdims=True); d_tot = (w * (1 - agree)).sum(1, keepdims=True)
            A_loo = (2.0 * A_dom + a_tot - w * agree) / (2.0 + a_tot + d_tot - w)
            A = (s_inst * A_loo + w * agree) / (s_inst + w)
        rho = np.clip(2 * A - 1, 0.0, 1.0)
        Kc = np.where(rho < 1, rho / np.maximum(1 - rho, 1e-12), 1e12)
    h = np.where((S > 0) & (Kc > 0), S * Kc / np.maximum(S + Kc, 1e-300), 0.0)
    Wp = npos + h * fg; Wn = (n - npos) + h * (1 - fg)
    return (Wp + bg) / (Wp + Wn + 1.0), A_dom


def est_oracle(n, npos, q, mu_hi=0.95):
    """exact posterior predictive of the generating model (hyper-parameters known)."""
    G, m = n.shape
    nneg = n - npos
    out = np.zeros((G, m))
    mus = np.array([mu_hi, 1 - mu_hi])
    # log-likelihood of each member under each mu, conform / exception
    lc = npos[None] * np.log(mus)[:, None, None] + nneg[None] * np.log(1 - mus)[:, None, None]     # conform
    lx = npos[None] * np.log(1 - mus)[:, None, None] + nneg[None] * np.log(mus)[:, None, None]     # exception
    lik = q * np.exp(lc) + (1 - q) * np.exp(lx)                    # [2, G, m]
    logpost = np.log(lik).sum(2)                                   # [2, G]
    post = np.exp(logpost - logpost.max(0, keepdims=True)); post /= post.sum(0, keepdims=True)
    r = q * np.exp(lc) / lik                                       # P(conform | own data, mu)
    pred = r * mus[:, None, None] + (1 - r) * (1 - mus[:, None, None])
    return (post[:, :, None] * pred).sum(0)


def part_e():
    print("\n== e. generated worlds.  Category rule mu in {0.95, 0.05}; a member conforms with probability q, otherwise it is")
    print("      the opposite (theta = 1 - mu).  Own sensed observations per member n in {0,1,2,3,5,10}.  300 categories. ==")
    print("   mean squared error of e_eff against the member's true frequency theta (lower is better)")
    print("   dB = design, Up sees own evidence (reading B) | dA = design, Up sees inherited evidence too (loop, reading A)")
    print("   corrected: base = base rate only (learned horizon; base1 = horizon 1) | bnd = + evidence bounded by 1 |")
    print("              lrn = bounded by learned conformity |")
    print("              gat = bounded by this member's conformity | gat(b=1/2) = gated without the base rate")
    hdr = "   m     q    group              n     |   dB     dA   | base1   base    bnd    lrn    gat   gat(b=1/2) | exact Bayes | own only"
    for q in (0.98, 0.9, 0.5):
        print(hdr)
        for m in (5, 20, 100):
            rng = np.random.default_rng(10 + m)
            mu, conf, theta, n, npos = gen_world(rng, 300, m, q)
            E = {
                "dB": est_design_B(n, npos),
                "dA": est_design_A(n, npos),
                "base1": est_corrected(n, npos, "base", horizon="fixed")[0],
                "base": est_corrected(n, npos, "base")[0],
                "bnd": est_corrected(n, npos, "bounded")[0],
                "lrn": est_corrected(n, npos, "learned")[0],
                "gat": est_corrected(n, npos, "gated")[0],
                "gat0": est_corrected(n, npos, "gated", base=False)[0],
                "or": est_oracle(n, npos, q),
                "own": (npos + 0.5) / (n + 1.0),
            }
            A_dom = est_corrected(n, npos, "learned")[1]
            groups = [("all", np.ones_like(conf)), ("never observed", n == 0),
                      ("conformer, n>=1", conf & (n >= 1)),
                      ("exception, n=1..2", ~conf & (n >= 1) & (n <= 2)),
                      ("exception, n=3..5", ~conf & (n >= 3) & (n <= 5)),
                      ("exception, n=10", ~conf & (n == 10))]
            for name, mask in groups:
                k = int(mask.sum())
                if k == 0:
                    continue
                v = {key: float(((val - theta) ** 2)[mask].mean()) for key, val in E.items()}
                print(f"  {m:3d}  {q:4.2f}  {name:18s} {k:6d} | {v['dB']:.3f}  {v['dA']:.3f} | {v['base1']:.3f}  {v['base']:.3f}  {v['bnd']:.3f}  {v['lrn']:.3f}  {v['gat']:.3f}   {v['gat0']:.3f}"
                      f"    |   {v['or']:.3f}     |  {v['own']:.3f}")
            print(f"        learned domain conformity A = {A_dom:.3f} -> rho = {max(0, 2*A_dom-1):.3f}")
    print("   -> the design gets WORSE as the category grows; exceptions with ten own observations are still overruled")
    print("      (m = 100, q = 0.9: 0.42 against 0.006 for own evidence alone).  In the q = 0.5 world (no usable rule)")
    print("      the design is worse than ignoring categories altogether: a large mixed category overwrites what each")
    print("      member knows about itself with unbounded 50/50 evidence.")


if __name__ == "__main__":
    part_a(); part_b(); part_c(); part_d(); part_e()
