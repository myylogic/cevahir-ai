#!/usr/bin/env python3
"""Belief-graph core v0 (Substrate A) -- reference equations, numpy only.

This file is the executable form of section 2 of proposal.md. It is NOT the 2,000-line
prototype; it is the smallest program in which every equation of the proposal runs, so
that the numbers of the worked trace (section 3) come from the equations and not from a
hand. Primary state = an append-only ledger of direct evidence (+ the trust counters of
the sources). Everything else (beliefs, category rules, couplings, responsibilities,
realism) is a deterministic function of the ledger, recomputed by `solve()`.
"""
from __future__ import annotations

import math

import numpy as np

ISA, HAS, EXC = 0, 1, 2
KIND_NAME = {ISA: "ISA", HAS: "HAS", EXC: "EXC"}

# ---- constants of the prototype (all declared in proposal.md, section 2.10) ------------
K_REF = 2.0                      # readout horizon of confidence: c = w / (w + K_REF)
GRID = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])   # coupling grid (pseudo-counts)
PI0 = np.full(len(GRID), 1.0 / len(GRID))                      # log-uniform prior over the grid
NU0 = 1.0                        # weight of the grid prior in the pooling over sibling contexts
W_FREE = 1.0                     # one-sided evidence that says nothing yet about mixedness
THETA_ALLOC = 0.25               # conflict mass that allocates a refined lens (MIX unit)
THETA_RES = 0.75                 # direct evidence at which an imagination counts as resolved
RHO_REPEAT = 0.5                 # the same source repeating the same generic statement
GAMMA_CRP = 1.0                  # concentration of the unseen-member estimate
PHI_FIRST = 0.5                  # share of the unseen mass carried by the next revealed member
THETA_Q = 0.03                   # a question is uttered if its expected gain exceeds this
LOG_STRONG = math.log(10.0)      # evidence ratio demanded before a category is split or joined
SWEEPS = 6
EPS = 1e-9


def par(a, b):
    """Series combination a || b = ab/(a+b): evidence a that must pass a coupling b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    s = a + b
    return np.where(s > 0, a * b / np.where(s > 0, s, 1.0), 0.0)


def vote(n, k):
    """Pseudo-votes about a category mean carried by a member with n units of evidence when
    members deviate from the mean with coupling k.  k->inf: n (every observation counts for
    the category);  k->0: 1 (one member, one vote)."""
    return par(n, k) * (k + 1.0) / k


def handdown(k, E):
    """Prior strength that a category mean known with E pseudo-votes (plus a uniform
    hyper-prior worth 2) hands to one member:  k (E+2) / (E+k+3).
    E=0, k->inf: 2 (Laplace);  k->0: 0 (trust the first observation);  E->inf: k."""
    return k * (E + 2.0) / (E + k + 3.0)


def lgam(x):
    """Vectorised log-gamma for x > 0 (shift by 8, then Stirling; abs. error < 1e-10)."""
    x = np.asarray(x, dtype=float)
    shift = np.zeros_like(x)
    for i in range(8):
        shift = shift + np.log(x + i)
    z = x + 8.0
    zi = 1.0 / z
    st = (z - 0.5) * np.log(z) - z + 0.9189385332046727 + zi / 12.0 - zi ** 3 / 360.0 + zi ** 5 / 1260.0 - zi ** 7 / 1680.0
    return st - shift


def lbeta(a, b):
    return lgam(a) + lgam(b) - lgam(np.asarray(a, dtype=float) + np.asarray(b, dtype=float))


def pml(wp, wn):
    """Pseudo-log-marginal likelihood of a group of members as ONE homogeneous-or-not cluster:
    product of leave-one-out predictives, coupling integrated over the grid, vague prior on the mean."""
    wp, wn = np.asarray(wp, dtype=float), np.asarray(wn, dtype=float)
    w = wp + wn
    f = np.where(w > 0, wp / np.where(w > 0, w, 1.0), 0.0)
    V = vote(w[None, :], GRID[:, None])
    E = V.sum(1)[:, None] - V
    num = (V * f[None, :]).sum(1)[:, None] - V * f[None, :]
    a = np.clip(np.where(E > 1e-12, num / np.where(E > 1e-12, E, 1.0), 0.5), 0.0, 1.0)
    abar = (E * a + 1.0) / (E + 2.0)
    kp = handdown(GRID[:, None], E)
    al, be = np.maximum(kp * abar, EPS), np.maximum(kp * (1 - abar), EPS)
    ll = (lbeta(al + wp[None, :], be + wn[None, :]) - lbeta(al, be)).sum(1)
    return float(np.log(PI0 @ np.exp(ll - ll.max())) + ll.max())


def ctx_predictive(g, wp, wn):
    """log predictive of evidence (wp, wn) of an outside subject under the rule of context g."""
    E = g["E0"] + g["S0"]
    a = np.where(E > 1e-12, (g["E0"] * g["A0"] + g["S1"]) / np.where(E > 1e-12, E, 1.0), 0.5)
    abar = (E * a + 1.0) / (E + 2.0)
    kp = handdown(GRID, E)
    al, be = np.maximum(kp * abar, EPS), np.maximum(kp * (1 - abar), EPS)
    lp = lbeta(al + wp, be + wn) - lbeta(al, be)
    return float(np.log(g["pi"] @ np.exp(lp)))


class Mind:
    def __init__(self, name="mind", learn_process=True, fixed_k=None, v_fixed=None, certain_membership=False,
                 k_schema=None, allow_split=True):
        self.name = name
        self.allow_split = allow_split               # False: never form sub-categories (ablation / test hook)
        self.k_schema = k_schema                     # None: learned on the grid; a number: fixed (ablation)
        self.names: list[str] = []
        self.ix: dict[str, int] = {}
        self.src: dict[str, list[float]] = {}        # source -> [agree, disagree] beyond Beta(1,1)
        self.src0: dict[str, list[float]] = {}       # trust given before any experience
        self.ledger: list[dict] = []                 # append-only; the primary state
        self.mix: dict[int, tuple[int, int]] = {}    # MIX unit -> (pole, pole)
        self.pair2mix: dict[tuple[int, int], int] = {}
        self.sub: dict[int, tuple[int, int]] = {}    # derived sub-category -> (parent category, lens)
        self.sub_of: dict[tuple[int, int], int] = {}
        self.sub_members: dict[int, set] = {}        # sub-category -> members assigned to it
        self.split_log: dict[tuple[int, int], tuple[float, float]] = {}   # last split test per context
        self.imag: dict[tuple[int, int, int], dict] = {}   # unresolved imaginations
        self.journal: list[dict] = []                # resolved imaginations (reporting only)
        self.alloc_log: list[str] = []
        self.learn_process = learn_process           # False: ablation, operating process frozen
        self.fixed_k = fixed_k                       # ablation: one global coupling
        self.v_fixed = v_fixed                       # test hook: fix the realism / inlier prior
        self.certain_membership = certain_membership  # test hook: stored ISA links count as 1
        self.S = None
        self.t = 0

    # ------------------------------------------------------------------ units / sources
    def unit(self, name: str) -> int:
        if name not in self.ix:
            self.ix[name] = len(self.names)
            self.names.append(name)
        return self.ix[name]

    def set_prior_trust(self, q: str, agree: float, disagree: float):
        self.src0[q] = [agree, disagree]
        self.src[q] = [agree, disagree]

    def trust(self, q: str) -> float:
        a, d = self.src.setdefault(q, [0.0, 0.0])
        return (a + 1.0) / (a + d + 2.0)

    # ------------------------------------------------------------------ evidence tensors
    def tensors(self, exclude=None):
        n = len(self.names)
        Wp = np.zeros((3, n, n))
        Wn = np.zeros((3, n, n))
        mentions = np.zeros(n)
        for e in self.ledger:
            if e.get("retracted") or e["q"] == exclude:
                continue
            v = self.trust(e["q"])
            Wp[e["kind"], e["s"], e["o"]] += v * e["amt"] * e["y"]
            Wn[e["kind"], e["s"], e["o"]] += v * e["amt"] * (1.0 - e["y"])
            mentions[e["s"]] += 1.0
        return Wp, Wn, mentions

    # ------------------------------------------------------------------ the solver
    def solve(self, exclude=None):
        """Recompute every derived quantity from the ledger. `exclude` leaves one source out
        (used to judge that source against evidence that did not come from it)."""
        Wp, Wn, mentions = self.tensors(exclude)
        n = len(self.names)
        # exclusion strength: expectation with zero base rate, symmetric
        X = (Wp[EXC] + Wp[EXC].T) / (Wp[EXC] + Wp[EXC].T + Wn[EXC] + Wn[EXC].T + K_REF)
        np.fill_diagonal(X, 0.0)
        # membership weight: expectation of ISA, counted only if believed (> 1/2)
        tot = Wp[ISA] + Wn[ISA]
        Mexp = np.where(tot > 0, (Wp[ISA] + K_REF / 2) / (tot + K_REF), 0.0)
        Mbase = np.where(Mexp > 0.5, 1.0 if self.certain_membership else Mexp, 0.0)
        pole_of = {}
        for u, (o1, o2) in self.mix.items():
            pole_of.setdefault(o1, []).append(u)
            pole_of.setdefault(o2, []).append(u)

        def view(s, p):
            """direct evidence of subject s under the lens of property unit p"""
            if p in self.mix:
                o1, o2 = self.mix[p]
                a, b = Wp[HAS, s, o1], Wp[HAS, s, o2]
                return 2.0 * min(a, b), max(0.0, abs(a - b) - W_FREE) + Wn[HAS, s, o1] + Wn[HAS, s, o2]
            return Wp[HAS, s, p], Wn[HAS, s, p] + float(X[p] @ Wp[HAS, s])

        v0 = 0.5 if self.v_fixed is None else self.v_fixed
        v_schema = {"plain": v0, "pole": v0, "MIX": v0}
        prev = {}
        ctx = {}
        ks_post = {}
        M = Mbase.copy()
        kp0 = handdown(GRID, 0.0)                               # [J] strength when nothing is known
        for sweep in range(SWEEPS):
            # ---- contexts = (category, lens) with at least one informed member.  A sub-category
            #      exists for one lens of its parent: assigned members leave the parent's context for
            #      that lens; members not yet informed belong to both, in proportion to the sizes.
            M = Mbase.copy()
            ctx = {}
            cats = [c for c in range(n) if Mbase[:, c].sum() > 0]
            for c in cats:
                base = [s for s in range(n) if Mbase[s, c] > 0]
                props = set()
                for s in base:
                    for o in np.nonzero(Wp[HAS, s] + Wn[HAS, s])[0]:
                        o = int(o)
                        props.add(o)
                        for u in pole_of.get(o, []):            # the refined lens: one more view of the same
                            props.add(u)                        # evidence, and its poles are looked at together
                            props.update(self.mix[u])
                for p in props:
                    wv = np.array([view(s, p) for s in base])
                    schema = "MIX" if p in self.mix else ("pole" if X[p].max() >= 0.5 else "plain")
                    weights = {c: np.array([Mbase[s, c] for s in base])}
                    cs = self.sub_of.get((c, p))
                    if cs is not None:
                        inside = np.array([s in self.sub_members[cs] for s in base])
                        informed = wv.sum(1) >= THETA_RES
                        nB, nA = float(inside.sum()), float((informed & ~inside).sum())
                        share = nB / max(nA + nB, 1.0)
                        undecided = ~inside & ~informed
                        wpar = weights[c] * np.where(inside, 0.0, np.where(undecided, 1.0 - share, 1.0))
                        wsub = weights[c] * np.where(inside, 1.0, np.where(undecided, share, 0.0))
                        weights = {c: wpar, cs: wsub}
                        for s, x in zip(base, wsub):
                            M[s, cs] = x
                    for cat, wgt in weights.items():
                        keep = wgt > 0
                        members = [s for s, k_ in zip(base, keep) if k_]
                        old = prev.get((cat, p), {"r": {}, "pi": PI0.copy(), "psi": PI0})
                        mem = wgt[keep]
                        ctx[(cat, p)] = dict(
                            c=cat, p=p, schema=schema, members=members, mem=mem, wp=wv[keep, 0], wn=wv[keep, 1],
                            r=np.array([old["r"].get(s, v_schema[schema] * m) for s, m in zip(members, mem)]),
                            pi=old["pi"].copy(), psi=old["psi"], parent=c,
                            pool_mean=(schema in ("MIX", "pole") and cat not in self.sub))
            # ---- upward: votes of the members about the category mean, per grid value
            for g in ctx.values():
                w = g["wp"] + g["wn"]
                g["w"] = w
                g["f"] = np.where(w > 0, g["wp"] / np.where(w > 0, w, 1.0), 0.0)
                g["V"] = g["r"][None, :] * vote(w[None, :], GRID[:, None])                 # [J, M]
                g["S0"] = g["V"].sum(1)
                g["S1"] = (g["V"] * g["f"][None, :]).sum(1)
                g["S0bar"] = float(g["pi"] @ g["S0"])
                g["abar"] = float(g["pi"] @ g["S1"]) / g["S0bar"] if g["S0bar"] > 0 else 0.5
            # ---- leave-one-out messages inside one context, for a given prior (E0, A0) on its mean
            def member_level(g, E0, A0):
                v = v_schema[g["schema"]]
                E_loo = E0 + g["S0"][:, None] - g["V"]                                      # [J, M]
                num = E0 * A0 + g["S1"][:, None] - g["V"] * g["f"][None, :]
                a_loo = np.clip(np.where(E_loo > 1e-12, num / np.where(E_loo > 1e-12, E_loo, 1.0), 0.5), 0.0, 1.0)
                abar = (E_loo * a_loo + 1.0) / (E_loo + 2.0)                                # Laplace on the mean
                kp = handdown(GRID[:, None], E_loo)                                         # [J, M]
                al = np.maximum(kp * abar, EPS)
                be = np.maximum(kp * (1 - abar), EPS)
                lpin = lbeta(al + g["wp"][None, :], be + g["wn"][None, :]) - lbeta(al, be)
                lpout = lbeta(1.0 + g["wp"], 1.0 + g["wn"])                                 # vs uniform Beta(1,1)
                rho = np.clip(v * g["mem"], 0.0, 1.0)                                       # prior: member AND conforms
                mixl = rho[None, :] * np.exp(lpin) + (1 - rho)[None, :] * np.exp(lpout)[None, :]
                return dict(E_loo=E_loo, a_loo=a_loo, abar_m=abar, kp=kp, rho=rho, lpin=lpin,
                            rj=rho[None, :] * np.exp(lpin) / mixl, logL=np.log(mixl).sum(1))

            # ---- schema level: a lens with the same meaning everywhere pools its mean over categories;
            #      how strongly it carries over (K_S) is itself a belief on the grid, scored by the
            #      leave-one-out likelihood of the members it helps to predict
            ks_grid = GRID if self.k_schema is None else np.array([float(self.k_schema)])
            for g in ctx.values():
                g["E0"], g["A0"] = 0.0, 0.5
            ks_post = {}
            for sc in ("MIX", "pole"):
                pooled = [g for g in ctx.values() if g["pool_mean"] and g["schema"] == sc]
                w_ks = np.full(len(ks_grid), 1.0 / len(ks_grid))
                if pooled and self.learn_process:
                    score = np.zeros(len(ks_grid))
                    e0 = np.zeros((len(ks_grid), len(pooled)))
                    a0 = np.full((len(ks_grid), len(pooled)), 0.5)
                    ab = np.array([g["abar"] for g in pooled])
                    cat = np.array([g["parent"] for g in pooled])
                    for i, ks in enumerate(ks_grid):
                        vS = np.array([float(vote(g["S0bar"], ks)) for g in pooled])
                        for j, g in enumerate(pooled):
                            other = cat != cat[j]                 # never a context of the same category
                            rest = float(vS[other].sum())
                            if rest > 1e-12:
                                e0[i, j] = handdown(ks, rest) - handdown(ks, 0.0)
                                a0[i, j] = (float((vS[other] * ab[other]).sum()) + 1.0) / (rest + 2.0)
                            ll = member_level(g, e0[i, j], a0[i, j])["logL"]
                            pk = g.get("psi", PI0)
                            score[i] += float(np.log(pk @ np.exp(ll - ll.max())) + ll.max())
                    w_ks = np.exp(score - score.max()) / np.exp(score - score.max()).sum()
                    for j, g in enumerate(pooled):
                        g["E0"] = float(w_ks @ e0[:, j])
                        g["A0"] = float(w_ks @ (e0[:, j] * a0[:, j])) / g["E0"] if g["E0"] > 1e-12 else 0.5
                ks_post[sc] = float(np.exp(w_ks @ np.log(ks_grid)))
            # ---- likelihood of each coupling, responsibilities
            for g in ctx.values():
                g.update(member_level(g, g["E0"], g["A0"]))
                q = PI0 * np.exp(g["logL"] - g["logL"].max())
                g["q"] = q / q.sum()
            for key, g in ctx.items():
                sib = [h["q"] for k2, h in ctx.items()
                       if h["schema"] == g["schema"] and h["parent"] != g["parent"]]   # other categories only
                if self.fixed_k is not None:
                    pi = np.zeros(len(GRID))
                    pi[int(np.argmin(np.abs(GRID - self.fixed_k)))] = 1.0
                elif self.learn_process:
                    psi = (NU0 * PI0 + (np.sum(sib, axis=0) if sib else 0.0)) / (NU0 + len(sib))
                    g["psi"] = psi
                    pi = psi * np.exp(g["logL"] - g["logL"].max())
                    pi = pi / pi.sum()
                else:
                    pi = g["q"]
                g["pi"] = pi
                g["r"] = pi @ g["rj"]
                v = v_schema[g["schema"]]
                # of the unexplained part, the share that is 'a member, but an exception'
                g["exc"] = (1 - g["r"]) * g["mem"] * (1 - v) / np.maximum(1 - v * g["mem"], EPS)
                g["inl"] = g["r"] / np.maximum(g["r"] + g["exc"], EPS)                      # P(conforms | member, data)
            # ---- measured realism = share of informed members that conform to their rule
            if self.learn_process and self.v_fixed is None:
                for sc in v_schema:
                    num, den = 1.0, 2.0
                    for g in ctx.values():
                        if g["schema"] == sc and g["c"] not in self.sub:
                            ok = g["w"] >= THETA_RES
                            num += float(g["inl"][ok].sum())
                            den += float(ok.sum())
                    v_schema[sc] = num / den
            prev = {k: {"r": dict(zip(g["members"], g["r"])), "pi": g["pi"], "psi": g.get("psi", PI0),
                        "exc": dict(zip(g["members"], g["exc"]))} for k, g in ctx.items()}
        # ---- readouts
        for g in ctx.values():
            v = v_schema[g["schema"]]
            real = g["rj"] * (g["kp"] - kp0[:, None])               # evidence fed back, per grid value
            g["K"] = g["pi"] @ real                                  # [M] real evidence fed back to each member
            g["Kp"] = g["pi"] @ (real * g["abar_m"])
            g["P"] = g["pi"] @ (g["rj"] * g["kp"] + (1 - g["rj"]) * 2.0)          # total prior pseudo-count
            g["Pp"] = g["pi"] @ (g["rj"] * g["kp"] * g["abar_m"] + (1 - g["rj"]) * 1.0)
            Etot = g["E0"] + g["S0"]
            g["a_rule"] = float(g["pi"] @ ((g["E0"] * g["A0"] + g["S1"] + 1.0) / (Etot + 2.0)))
            g["E_rule"] = float(g["pi"] @ Etot)
            Knew = float(g["pi"] @ (v * (handdown(GRID, Etot) - kp0)))
            g["K_new"] = Knew
            g["c_rule"] = Knew / (Knew + K_REF)
            g["k_mean"] = float(np.exp(g["pi"] @ np.log(GRID)))
        ks_mean = ks_post.get("MIX", float("nan"))
        self.S = dict(Wp=Wp, Wn=Wn, X=X, M=M, Mbase=Mbase, ctx=ctx, v=v_schema, mentions=mentions, pole_of=pole_of,
                      ks_post=ks_post, ks_mean=ks_mean)
        return self.S

    # ------------------------------------------------------------------ answering
    def _prior(self, s: int, p: int):
        """Prior pseudo-counts for subject s under lens p from every context that holds it:
        (positive, total, real fed-back positive, real fed-back total)."""
        S = self.S
        Kp = K = 0.0
        base_p, base = 1.0, 2.0                       # nothing known: uniform Beta(1,1)
        first = True
        for (c, pp), g in S["ctx"].items():
            if pp == p and s in g["members"]:
                i = g["members"].index(s)
                K += g["K"][i]
                Kp += g["Kp"][i]
                if first:
                    base_p, base = g["Pp"][i] - g["Kp"][i], g["P"][i] - g["K"][i]
                    first = False
        return base_p + Kp, base + K, Kp, K

    def _own(self, s: int, p: int):
        S = self.S
        if p in self.mix:
            o1, o2 = self.mix[p]
            a, b = S["Wp"][HAS, s, o1], S["Wp"][HAS, s, o2]
            return 2 * min(a, b), max(0.0, abs(a - b) - W_FREE) + S["Wn"][HAS, s, o1] + S["Wn"][HAS, s, o2]
        return S["Wp"][HAS, s, p], S["Wn"][HAS, s, p] + float(S["X"][p] @ S["Wp"][HAS, s])

    def belief_view(self, s: int, p: int):
        """Belief 'subject s under lens p' = own evidence revised with the support fed back."""
        wp, wn = self._own(s, p)
        Pp, P, Kp, K = self._prior(s, p)
        w = wp + wn
        return dict(f=((wp + Kp) / (w + K) if w + K > 0 else float("nan")), w=w, K=K, Kp=Kp,
                    e=(wp + Pp) / (w + P), c=(w + K) / (w + K + K_REF), c_own=w / (w + K_REF),
                    f_own=(wp / w if w > 0 else float("nan")))

    def answer(self, kind: int, s: int, o: int):
        """Expectation e, frequency f and confidence c of a statement, from the current state."""
        S = self.S
        if kind != HAS:
            wp, wn = S["Wp"][kind, s, o], S["Wn"][kind, s, o]
            if kind == EXC:
                wp += S["Wp"][kind, o, s]
                wn += S["Wn"][kind, o, s]
            w = wp + wn
            return dict(e=(wp + 1.0) / (w + 2.0), c=w / (w + K_REF), f=(wp / w if w > 0 else float("nan")), w=w, K=0.0)
        if o in self.mix:
            return self.belief_view(s, o)
        wp, wn = self._own(s, o)
        Pp, P, Kp, K = self._prior(s, o)
        w = wp + wn
        return dict(e=(wp + Pp) / (w + P), c=(w + K) / (w + K + K_REF),
                    f=((wp + Kp) / (w + K) if w + K > 0 else float("nan")), w=w + K, K=K)

    # ------------------------------------------------------------------ perceiving
    def perceive(self, kind, s_name, o_name, y=1.0, q="stranger", event=False):
        """One experience: predict, judge the source, write one ledger line, re-solve,
        allocate, imagine. There is no other learning step."""
        self.t += 1
        s, o = self.unit(s_name), self.unit(o_name)
        if self.S is None or len(self.names) != self.S["Wp"].shape[1]:
            self.solve()
        pred = self.answer(kind, s, o)
        rep = dict(t=self.t, stmt=(KIND_NAME[kind], s_name, o_name, y, q), pred=pred)
        p_e = min(max(pred["e"], 1e-6), 1 - 1e-6)
        rep["surprise"] = -(y * math.log(p_e) + (1 - y) * math.log(1 - p_e))
        # trust in the source: judged only against what the mind believes WITHOUT this source
        ind = dict(c=0.0, f=float("nan"))
        if self.learn_process:
            if any(e["q"] == q for e in self.ledger):
                self.solve(exclude=q)
                ind = self.answer(kind, s, o)
                self.solve()
            else:
                ind = pred
            if ind["c"] > 0 and not math.isnan(ind["f"]):
                agree = 1.0 - abs(y - ind["f"])
                a = self.src.setdefault(q, [0.0, 0.0])
                a[0] += ind["c"] * agree
                a[1] += ind["c"] * (1 - agree)
        rep["independent"] = ind
        # the write: one ledger line, discounted when the same source repeats the same statement
        reps = 0 if event else sum(1 for e in self.ledger if (not e.get("retracted")) and not e["event"]
                                   and (e["q"], e["kind"], e["s"], e["o"], e["y"]) == (q, kind, s, o, y))
        self.ledger.append(dict(t=self.t, q=q, kind=kind, s=s, o=o, y=float(y),
                                amt=RHO_REPEAT ** reps, event=event))
        rep["id"] = len(self.ledger) - 1
        self.solve()
        rep["allocated"] = self._allocate()
        if rep["allocated"]:
            self.solve()
        rep["imagined"], rep["resolved"] = self._imagine()
        return rep

    def retract_evidence(self, entry_id: int):
        """O(1) retraction of one line of evidence. Exact for the evidence tensors; units that
        were allocated because of the line stay (as empty lenses), trust counters stay."""
        self.ledger[entry_id]["retracted"] = True
        self.solve()

    def replay(self, skip=()):
        """Exact retraction of the whole mind state: live again without the skipped lines."""
        m = Mind(self.name, self.learn_process, self.fixed_k, self.v_fixed, self.certain_membership, self.k_schema,
                 self.allow_split)
        for q, (a, d) in self.src0.items():
            m.set_prior_trust(q, a, d)
        for i, e in enumerate(self.ledger):
            if i in skip or e.get("retracted"):
                continue
            m.perceive(e["kind"], self.names[e["s"]], self.names[e["o"]], e["y"], e["q"], e["event"])
        return m

    # ------------------------------------------------------------------ concept formation
    def _allocate(self):
        S, out = self.S, []
        Wp, X, M = S["Wp"], S["X"], S["M"]
        n = len(self.names)
        # (a) a refined lens for two properties believed exclusive, allocated
        #     - when one subject, or the members of one category, carry positive evidence on both
        #       (unexplained evidence), or
        #     - by anticipation: the mind has already lived a lens of this kind that proved
        #       informative, and applies the same way of constructing to the new pair (P5)
        lived = self.learn_process and any(g["schema"] == "MIX" and g["pool_mean"] and (g["w"] >= THETA_RES).any()
                                           for g in S["ctx"].values())
        in_cat = (M.sum(1) > 0).astype(float)
        for o1 in range(n):
            for o2 in range(o1 + 1, n):
                if X[o1, o2] <= 0 or (o1, o2) in self.pair2mix:
                    continue
                Ts = X[o1, o2] * np.minimum(Wp[HAS, :, o1], Wp[HAS, :, o2])
                Tc = X[o1, o2] * np.minimum(M.T @ Wp[HAS, :, o1], M.T @ Wp[HAS, :, o2])
                T = max(Ts.max(), Tc.max()) if len(Tc) else Ts.max()
                antic = lived and X[o1, o2] >= 0.5 and float(in_cat @ (Wp[HAS, :, o1] + Wp[HAS, :, o2])) > 0
                if T >= THETA_ALLOC or antic:
                    u = self.unit(f"MIX({self.names[o1]}|{self.names[o2]})")
                    self.mix[u] = (o1, o2)
                    self.pair2mix[(o1, o2)] = u
                    if T >= THETA_ALLOC:
                        where = self.names[int(Ts.argmax())] if Ts.max() >= Tc.max() else "category " + self.names[int(Tc.argmax())]
                        out.append(f"new concept {self.names[u]}  (conflict mass {T:.2f} >= {THETA_ALLOC} on '{where}')")
                    else:
                        out.append(f"new concept {self.names[u]}  (anticipated: lived contrast pairs behaved as dimensions)")
        # (b) a sub-category: the informed members of a context are better explained as two groups
        #     than as one (pseudo-marginal likelihoods, CRP prior on the partition), or an informed
        #     member is better predicted by an existing sub-category than by its parent
        for (c, p), g in list(S["ctx"].items()):
            if c in self.sub or not self.allow_split:
                continue
            inf = [i for i in range(len(g["members"])) if g["w"][i] >= THETA_RES and g["mem"][i] > 0]
            cs = self.sub_of.get((c, p))
            if cs is not None:
                gs = S["ctx"].get((cs, p))
                for i in inf:
                    s = g["members"][i]
                    if gs is None or s in self.sub_members[cs]:
                        continue
                    lp_par = float(np.log(g["pi"] @ np.exp(g["lpin"][:, i])))
                    lp_sub = ctx_predictive(gs, g["wp"][i], g["wn"][i])
                    if lp_sub - lp_par > LOG_STRONG:
                        self.sub_members[cs].add(s)
                        out.append(f"'{self.names[s]}' joins category {self.names[cs]} (log predictive ratio {lp_sub - lp_par:.2f} > log 10)")
                continue
            if len(inf) < 4:
                continue
            order = sorted(inf, key=lambda i: g["f"][i])
            gaps = [g["f"][order[j + 1]] - g["f"][order[j]] for j in range(len(order) - 1)]
            j = int(np.argmax(gaps))
            lo, hi = order[: j + 1], order[j + 1:]
            if min(len(lo), len(hi)) < 2:
                continue
            one = pml(g["wp"][inf], g["wn"][inf])
            two = (pml(g["wp"][lo], g["wn"][lo]) + pml(g["wp"][hi], g["wn"][hi])
                   + float(lgam(len(lo)) + lgam(len(hi)) - lgam(len(lo) + len(hi))))
            thr = LOG_STRONG + math.log(len(inf) - 1)          # strong evidence, corrected for choosing the cut
            self.split_log[(c, p)] = (two - one, thr)
            if two - one > thr:
                grp = lo if len(lo) <= len(hi) else hi
                cs = self.unit(f"{self.names[c]}*{len(self.sub) + 1}")
                self.sub[cs] = (c, p)
                self.sub_of[(c, p)] = cs
                self.sub_members[cs] = {g["members"][i] for i in grp}
                out.append(f"new category {self.names[cs]} under '{self.names[c]}' = {sorted(self.names[g['members'][i]] for i in grp)}:"
                           f" two groups explain the evidence on {self.names[p]} better than one"
                           f" (log evidence ratio {two - one:.2f} > {thr:.2f})")
        self.alloc_log += out
        return out

    # ------------------------------------------------------------------ imagination
    def _imagine(self):
        """Constructions = beliefs held with no direct evidence. They are journalled, never written
        to the ledger, and scored when the world has spoken (direct evidence >= THETA_RES)."""
        S, new, res = self.S, [], []
        for (c, p), g in S["ctx"].items():
            parent = self.sub[c][0] if c in self.sub else c
            for i, s in enumerate(g["members"]):
                key = (s, parent, p)
                if c not in self.sub and g["w"][i] == 0 and g["K"][i] > 1e-9 and key not in self.imag:
                    b = self.belief_view(s, p)
                    self.imag[key] = dict(t=self.t, e=b["e"], c=b["c"], R=float(g["inl"][i]))
                    new.append(dict(s=self.names[s], p=self.names[p], e=b["e"], c=b["c"], R=float(g["inl"][i])))
                elif key in self.imag and g["w"][i] >= THETA_RES and g["mem"][i] > 0:
                    im = self.imag.pop(key)
                    left = c in self.sub                     # the member was moved out of the rule
                    rec = dict(s=self.names[s], p=self.names[p], e_imagined=im["e"], c_imagined=im["c"],
                               R_predicted=im["R"], f_world=float(g["f"][i]),
                               r_post=0.0 if left else float(g["inl"][i]), left_rule=left,
                               sq_err=(im["e"] - float(g["f"][i])) ** 2, t_imagined=im["t"], t_resolved=self.t)
                    self.journal.append(rec)
                    res.append(rec)
        return new, res

    # ------------------------------------------------------------------ questions
    def questions(self, c_name: str, delta: float = 0.5, gamma: float = GAMMA_CRP, phi: float = PHI_FIRST):
        """Expected gain in confidence on the next statement about category c, per candidate question."""
        S = self.S
        c = self.ix[c_name]
        gs = [(k, g) for k, g in S["ctx"].items() if k[0] == c and k[1] not in S["pole_of"]]
        if not gs:
            return [], 1.0
        members = gs[0][1]["members"]
        nc = float(sum(S["mentions"][s] for s in members))
        p_unseen = gamma / (nc + gamma)
        out = []
        m_new = (delta + 1.0) / (delta + 2.0)                 # membership after one answering statement
        g_ext = 0.0
        kp0 = handdown(GRID, 0.0)
        for (cc, p), g in gs:
            v = S["v"][g["schema"]]
            Kproj = float(g["pi"] @ (v * m_new * (handdown(GRID, g["E0"] + g["S0"]) - kp0)))
            g_ext += p_unseen * phi * (Kproj / (Kproj + K_REF)) / len(gs)
            for i, s in enumerate(members):
                Pi = (1 - p_unseen) * S["mentions"][s] / nc
                w, K = g["w"][i], g["K"][i]
                gain = Pi * ((w + delta + K) / (w + delta + K + K_REF) - (w + K) / (w + K + K_REF)) / len(gs)
                out.append((float(gain), f"tell me more about '{self.names[s]}' and {self.names[p]}"))
        out.append((float(g_ext), f"which other '{c_name}'s are there?"))
        out.sort(reverse=True)
        return [dict(gain=a, text=b, uttered=(a >= THETA_Q), top=(j == 0)) for j, (a, b) in enumerate(out)], p_unseen

    # ------------------------------------------------------------------ language out
    def say(self, s_name: str, o_name: str):
        s, o = self.ix[s_name], self.ix[o_name]
        b = self.answer(HAS, s, o)
        f, c = b["f"], b["c"]
        if c < 0.02 or math.isnan(f):
            return f"I do not know whether {s_name} is {o_name}.  (c={c:.2f})"
        freq = ("never" if f < 0.15 else "rarely" if f < 0.35 else "sometimes" if f < 0.65 else "usually" if f < 0.85 else "always")
        hedge = ("I have only heard that" if c < 0.25 else "I think" if c < 0.5 else "I am fairly sure" if c < 0.75 else "I know")
        return f"{hedge} {s_name} is {freq} {o_name}.  (f={f:.2f}, e={b['e']:.2f}, c={c:.2f})"
