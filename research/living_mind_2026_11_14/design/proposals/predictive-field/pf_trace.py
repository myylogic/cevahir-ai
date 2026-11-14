#!/usr/bin/env python3
"""Predictive Field (Substrate B) -- worked trace of the "red" example.

A deliberately tiny, numpy-only implementation of the equations of proposal.md
(section 2). It is NOT the prototype; it exists so that every number of the
worked trace (section 3) is produced by code and can be re-run:

    python3 pf_trace.py            # full trace, < 2 s on one core
    python3 pf_trace.py --json     # key numbers as JSON

Two constants are calibrated on the owner's first two numbers (0.23, 0.57):
    eta0 = prior log-odds of an isolated told fact, lam = nats per statement.
Everything else comes from stated priors and the update equations.
"""
import itertools
import json
import math
import random
import sys
import zlib

import numpy as np


def logit(p):
    return math.log(p / (1.0 - p))


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def ent(p):
    return 0.0 if p <= 0.0 or p >= 1.0 else -(p * math.log(p) + (1 - p) * math.log(1 - p))


def kl(p, q):
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


LAM = logit(0.57) - logit(0.23)      # 1.4902 nats per statement   (calibrated)
ETA0 = logit(0.23) - LAM             # -2.6985 -> p0 = 0.0631       (calibrated)
P0 = sig(ETA0)


# --------------------------------------------------------------------------- #
# operator level (the "meta-field"): graded beliefs about the operating process
# --------------------------------------------------------------------------- #
def adf(a, b, lr):
    """Beta(a,b) prior on an operator's success rate, one construction whose
    test evidence has likelihood ratio lr (success : failure).  Exact posterior
    is a two-component Beta mixture; moment-match it back to one Beta.
    lr == 1 returns (a, b) unchanged (no evidence -> no change)."""
    w1, w0 = a * lr, b
    w1, w0 = w1 / (w1 + w0), w0 / (w1 + w0)
    n = a + b
    m = (w1 * (a + 1) + w0 * a) / (n + 1)
    m2 = (w1 * (a + 1) * (a + 2) + w0 * a * (a + 1)) / ((n + 1) * (n + 2))
    s = m * (1 - m) / (m2 - m * m) - 1.0
    return m * s, (1 - m) * s


class Meta:
    def __init__(self, a=1.0, b=1.0, alpha_cat=1.0):
        self.gen_prior = (a, b)      # GENERALISE(T2): Beta over "the rule holds on new members"
        self.tests = {}              # ledger of constructions: rule key -> test LR (members other than the trigger)
        self.alpha_cat = alpha_cat   # openness of a category (CRP concentration)

    def p_gen(self, exclude=None):
        a, b = self.gen_prior
        for key in sorted(self.tests):
            if key != exclude:       # cavity: a rule never raises its own prior
                a, b = adf(a, b, self.tests[key])
        return a / (a + b), a, b


# --------------------------------------------------------------------------- #
# object level: the field
# --------------------------------------------------------------------------- #
class Field:
    def __init__(self, meta, d=64, seed=0, a=8.0, c_mode=3.0, c_unit=5.0, theta=1.0, verbose=True):
        self.meta, self.d, self.a = meta, d, a
        self.c_mode, self.c_unit, self.theta = c_mode, c_unit, theta
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.name, self.idx, self.C = [], {}, np.zeros((0, d))   # concept bank (grows)
        self.units = []      # association units: cue (S,R) -> value-slot mixture
        self.rules = []      # category-level units
        self.var = {}        # variation concepts: frozenset(modes) -> dict(centre, axis, s2)
        self.n_stmt = 0
        self.post = {}       # settled beliefs

    def say(self, *args):
        if self.verbose:
            print(*args)

    # -- concept bank ------------------------------------------------------- #
    def concept(self, name):
        if name not in self.idx:
            v = self.rng.standard_normal(self.d)
            v /= np.linalg.norm(v)
            self.idx[name] = len(self.name)
            self.name.append(name)
            self.C = np.vstack([self.C, v])
            self.say(f"      recruit concept [{name}] (no attractor explains the token)")
        return self.idx[name]

    def energy(self, x, w):
        return 0.5 * self.a * float(((x - w) ** 2).sum())

    def cue(self, i_s, i_r):
        return np.concatenate([self.C[i_s], self.C[i_r]])

    # -- perceive one frame -------------------------------------------------- #
    def perceive(self, S, R, O, sid, sign=+1, nats=LAM):
        i_s, i_r, i_o = self.concept(S), self.concept(R), self.concept(O)
        x = self.cue(i_s, i_r)
        # settle 1: which unit owns the cue?  (energies vs. the "new unit" option)
        E = [self.energy(x, self.cue(u["S"], u["R"])) for u in self.units]
        F = E + [self.c_unit]
        k = int(np.argmin(F))
        if k == len(self.units):
            u = {"S": i_s, "R": i_r, "modes": {}, "neg": {}, "id": len(self.units)}
            self.units.append(u)
            best = f"{min(E):.2f}" if E else "-"
            self.say(f"      settle: best cue energy {best} > c_unit {self.c_unit} -> recruit unit ({S},{R},.)")
        else:
            u = self.units[k]
            self.say(f"      settle: cue energy {E[k]:.2f} -> unit ({S},{R},.) owns the frame")
        refined = False
        if sign < 0:                              # "S is not O": negative evidence on that mode
            u["neg"].setdefault(i_o, {})[sid] = nats
            self.say(f"      ledger: -{nats:.2f} nats against mode [{O}]")
        else:
            # settle 2: which mode of the value slot?
            Ev = {m: self.energy(self.C[i_o], self.C[m]) for m in u["modes"]}
            m_best = min(Ev, key=Ev.get) if Ev else None
            if m_best is not None and Ev[m_best] < self.c_mode:
                u["modes"][m_best][sid] = nats
                self.say(f"      value energy {Ev[m_best]:.2f} < c_mode -> same mode, +{nats:.2f} nats")
            else:
                if Ev:
                    self.say(f"      value energy {Ev[m_best]:.2f} > c_mode {self.c_mode} -> the old value is "
                             f"NOT overwritten: new mode [{O}] in the same unit (REFINE)")
                    refined = True
                u["modes"][i_o] = {sid: nats}
        if refined:
            self.refine(u)
        return u, refined

    # -- REFINE: variation concept ------------------------------------------- #
    def refine(self, u):
        V = frozenset(u["modes"])
        if V not in self.var:
            pts = self.C[sorted(V)]
            centre = pts.mean(0)
            dv = pts[0] - pts[1]
            s2 = float((dv ** 2).sum()) / 4.0            # squared half-spread along the axis
            axis = dv / np.linalg.norm(dv)
            self.var[V] = {"centre": centre, "axis": axis, "s2": s2}
            label = "|".join(self.name[m] for m in sorted(V))
            self.say(f"      recruit variation concept V{{{label}}}: centre = mean code, slack along (difference) axis")
            for m in sorted(V):
                self.say(f"        coverage energy of [{self.name[m]}] under V: {self.cov_energy(V, self.C[m]):.2f} nats")
            others = [i for i in range(len(self.name)) if i not in V]
            if others:
                e_out = np.mean([self.cov_energy(V, self.C[i]) for i in others])
                self.say(f"        mean coverage energy of the {len(others)} other concepts: {e_out:.2f} nats (> c_mode: not in V)")
        self.generalise(u, V)

    def cov_energy(self, V, x):
        v = self.var[V]
        dx = x - v["centre"]
        along = float(dx @ v["axis"])
        kappa = 1.0 / (1.0 + self.a * v["s2"])          # precision left along the slack axis
        return 0.5 * self.a * (float(dx @ dx) - (1.0 - kappa) * along * along)

    # -- GENERALISE (T2): relax the subject slot to the category ------------- #
    def categories(self, i_s):
        isa = self.idx.get("isa")
        for u in self.units:
            if u["S"] == i_s and u["R"] == isa:
                return list(u["modes"])
        return []

    def generalise(self, u, V):
        for cat in self.categories(u["S"]):
            if any(r["cat"] == cat and r["R"] == u["R"] for r in self.rules):
                continue
            r = {"cat": cat, "R": u["R"], "V": V, "trigger": u["S"], "key": (self.name[cat], self.name[u["R"]]),
                 "licence": u["id"]}
            self.rules.append(r)
            p, a, b = self.meta.p_gen(exclude=r["key"])
            label = "|".join(self.name[m] for m in sorted(V))
            self.say(f"      GENERALISE: subject slot relaxed from [{self.name[u['S']]}] to its category "
                     f"[{self.name[cat]}] -> rule ({self.name[cat]}*, {self.name[u['R']]}, V{{{label}}}); "
                     f"prior from the operator level p_H = {p:.3f} (Beta {a:.2f},{b:.2f})")

    # -- evidence bookkeeping ------------------------------------------------ #
    def unit(self, S, R):
        i_s, i_r = self.idx.get(S), self.idx.get(R)
        for u in self.units:
            if u["S"] == i_s and u["R"] == i_r:
                return u
        return None

    def distinct_nats(self, u, modes=None):
        """nats of the unit's ledger restricted to `modes`; each statement id counted once."""
        seen = {}
        for m, led in u["modes"].items():
            if modes is None or m in modes:
                seen.update(led)
        neg = {}
        for m, led in u["neg"].items():
            if modes is None or m in modes:
                neg.update(led)
        return sum(seen.values()) - sum(neg.values())

    def members(self, r):
        """(name, membership unit, instance unit or None) for every known member of the rule's category."""
        out = []
        isa = self.idx["isa"]
        for u in self.units:
            if u["R"] == isa and r["cat"] in u["modes"]:
                inst = None
                for w in self.units:
                    if w["S"] == u["S"] and w["R"] == r["R"]:
                        inst = w
                out.append((self.name[u["S"]], u, inst))
        return out

    # -- settle the beliefs: closed-form cavity messages ---------------------- #
    def settle(self):
        post = {}
        b, th = P0, self.theta
        ruled = set()
        for r in self.rules:
            if not self.licensed(r):                 # dormant construction: kept, but sends no messages
                self.meta.tests.pop(r["key"], None)
                continue
            mem = self.members(r)
            p_h, _, _ = self.meta.p_gen(exclude=r["key"])
            mu, L, D, m1 = {}, {}, {}, {}
            for nm, um, ui in mem:
                mu[nm] = sig(ETA0 + self.distinct_nats(um, {r["cat"]}))
                L[nm] = math.exp(self.distinct_nats(ui, r["V"])) if ui is not None else 1.0
                D[nm] = b * L[nm] + 1 - b
                m1[nm] = mu[nm] * (th * L[nm] + 1 - th) + (1 - mu[nm]) * D[nm]
            up = {nm: math.log(m1[nm] / D[nm]) for nm in mu}            # upward messages (nats)
            eta_h = logit(p_h) + sum(up.values())
            post[("rule",) + r["key"]] = {"p": sig(eta_h), "prior": p_h, "up": dict(up)}
            for nm, um, ui in mem:
                cav = sig(eta_h - up[nm])                                # cavity: rule without this member
                num = L[nm] * (cav * mu[nm] * th + (1 - cav * mu[nm]) * b)
                den = num + cav * mu[nm] * (1 - th) + (1 - cav * mu[nm]) * (1 - b)
                prior_x = cav * mu[nm] * th + (1 - cav * mu[nm]) * b
                nm1 = mu[nm] * (cav * (th * L[nm] + 1 - th) + (1 - cav) * D[nm])
                nm0 = (1 - mu[nm]) * D[nm]
                own = math.log(L[nm])
                post[("x", nm, self.name[r["R"]])] = {
                    "p": num / den, "own_nats": own, "down_nats": logit(prior_x) - logit(b),
                    "cavity_rule": cav, "has_unit": ui is not None, "alone": sig(ETA0 + own)}
                post[("M", nm, self.name[r["cat"]])] = {"p": nm1 / (nm1 + nm0), "alone": mu[nm]}
                ruled.add(um["id"])
                if ui is not None:
                    ruled.add(ui["id"])
            # operator-level ledger: the rule's track record = members other than the trigger
            self.meta.tests[r["key"]] = math.exp(sum(v for nm, v in up.items() if nm != self.name[r["trigger"]]))
        for u in self.units:                                             # units under no rule: own evidence only
            if u["id"] not in ruled:
                kind = "M" if self.name[u["R"]] == "isa" else "x"
                for m in u["modes"]:
                    key = (kind, self.name[u["S"]], self.name[m] if kind == "M" else self.name[u["R"]])
                    nats = self.distinct_nats(u, {m} if kind == "M" else None)
                    post[key] = {"p": sig(ETA0 + nats), "alone": sig(ETA0 + nats), "own_nats": nats, "down_nats": 0.0}
        self.post = post
        return post

    def offer(self):
        """state-based triggers (order-independent): every unit that holds a variation is offered to GENERALISE."""
        for u in self.units:
            V = frozenset(u["modes"])
            if len(V) >= 2 and V in self.var:
                self.generalise(u, V)

    def licensed(self, r):
        """the licence of a T2 rule: its trigger unit still holds every mode of V with grounded evidence."""
        u = self.units[r["licence"]]
        return all(m in u["modes"] and u["modes"][m] for m in r["V"])

    def retract(self, sid):
        """exact retraction: remove one statement from every ledger; constructions that lose their licence go dormant."""
        for u in self.units:
            for led in list(u["modes"].values()) + list(u["neg"].values()):
                led.pop(sid, None)
            u["modes"] = {m: led for m, led in u["modes"].items() if led}
            u["neg"] = {m: led for m, led in u["neg"].items() if led}
        return self.settle()

    def content(self, S, R):
        u = self.unit(S, R)
        w = {self.name[m]: sum(led.values()) for m, led in u["modes"].items()}
        z = sum(w.values())
        return {k: v / z for k, v in w.items()}

    # -- exact check: brute-force enumeration of the same generative model ---- #
    def brute(self, r):
        mem = self.members(r)
        p_h, _, _ = self.meta.p_gen(exclude=r["key"])
        b, th = P0, self.theta
        n = len(mem)
        tot, ph, px, pm = 0.0, 0.0, [0.0] * n, [0.0] * n
        Lm = [math.exp(self.distinct_nats(um, {r["cat"]})) for _, um, _ in mem]
        Lx = [math.exp(self.distinct_nats(ui, r["V"])) if ui is not None else 1.0 for _, _, ui in mem]
        for bits in itertools.product((0, 1), repeat=1 + 2 * n):
            h, M, X = bits[0], bits[1:1 + n], bits[1 + n:]
            w = p_h if h else 1 - p_h
            for j in range(n):
                w *= (P0 * Lm[j]) if M[j] else (1 - P0)
                q = th if (h and M[j]) else b
                w *= (q * Lx[j]) if X[j] else (1 - q)
            tot += w
            ph += w * h
            for j in range(n):
                px[j] += w * X[j]
                pm[j] += w * M[j]
        return ph / tot, [v / tot for v in px], [v / tot for v in pm]

    # -- questions: epistemic value of candidate probes ----------------------- #
    def questions(self, gamma=0.5):
        r_ans = sig(LAM)                       # an answer is one statement: P(yes|true) = sigma(lam)
        out = []

        def confirm_value(p):
            py = p * r_ans + (1 - p) * (1 - r_ans)
            return ent(py) - ent(r_ans)

        for key, v in self.post.items():
            if key[0] == "rule":
                out.append((confirm_value(v["p"]), f"confirm rule: are all [{key[1]}] V?"))
            elif key[0] == "M":
                out.append((confirm_value(v["p"]), f"confirm: is [{key[1]}] a [{key[2]}]?"))
            elif key[0] == "x" and v.get("has_unit", True):
                out.append((confirm_value(v["p"]), f"confirm: is [{key[1]}] really V?"))
        for r in self.rules:
            ph = self.post[("rule",) + r["key"]]["p"]
            n_eff = sum(self.post[("M", nm, self.name[r["cat"]])]["p"] for nm, _, _ in self.members(r))
            p_unknown = self.meta.alpha_cat / (self.meta.alpha_cat + n_eff)     # CRP: next member is one I do not know
            e_k = p_unknown / (1 - p_unknown)                                    # expected number of new members
            kl_new = kl(sig(ETA0 + LAM), P0)                                     # surprise of one new membership fact
            # follow-up: one statement about the new member's value, as a test of the rule
            mu_new, b, th = sig(ETA0 + LAM), P0, self.theta
            pa = {}
            for h in (0, 1):
                px = mu_new * (th if h else b) + (1 - mu_new) * b
                pa[h] = px * r_ans + (1 - px) * (1 - r_ans)
            py = ph * pa[1] + (1 - ph) * pa[0]
            i_follow = ent(py) - ph * ent(pa[1]) - (1 - ph) * ent(pa[0])
            val = e_k * (kl_new + gamma * i_follow)
            out.append((val, f"enumerate: which other [{self.name[r['cat']]}] are there?  "
                             f"(P_unknown {p_unknown:.2f}, E[new] {e_k:.2f}, {kl_new:.3f} nats each, follow-up {i_follow:.3f})"))
        return sorted(out, reverse=True)


# --------------------------------------------------------------------------- #
# the tiny language: sentence -> frames (symbolic front end of v0)
# --------------------------------------------------------------------------- #
def tell(f, text, frames, quiet=False):
    f.n_stmt += 1
    sid = f.n_stmt
    old = f.verbose
    f.verbose = old and not quiet
    f.say(f"\n[{sid}] \"{text}\"")
    for fr in frames:
        S, R, O = fr[:3]
        sign = fr[3] if len(fr) > 3 else +1
        f.say(f"    frame ({S}, {R}, {'not ' if sign < 0 else ''}{O})")
        f.perceive(S, R, O, sid, sign)
    f.offer()
    post = f.settle()
    f.verbose = old
    return post


def show(f, keys):
    for key in keys:
        v = f.post.get(key)
        if v is None:
            continue
        extra = ""
        if key[0] == "x" and "down_nats" in v:
            tot = v["own_nats"] + max(v["down_nats"], 0.0)
            g = f"grounded share {v['own_nats'] / tot:.2f}" if v["own_nats"] >= 0 and tot > 0 else "told evidence is AGAINST"
            extra = f"   [told {v['own_nats']:+.2f} nats, constructed {v['down_nats']:+.2f} nats, {g}]"
        if key[0] == "rule":
            extra = f"   [prior {v['prior']:.3f}; upward nats " + ", ".join(f"{k} {x:+.2f}" for k, x in v["up"].items()) + "]"
        if key[0] == "M":
            extra = f"   [own evidence alone {v['alone']:.3f}]"
        f.say(f"    belief {str(key):34s} = {v['p']:.3f}{extra}")


def domain(f, cat, R, first, others, v1, v2, confirm):
    """a previous life in another domain: same two-statement pattern, then other members test the rule."""
    tell(f, f"{first} is a {v1} {cat}", [(first, "isa", cat), (first, R, v1)], quiet=True)
    tell(f, f"{first} also has {v2}", [(first, R, v2)], quiet=True)
    for o in others:
        tell(f, f"{o} is a {cat}", [(o, "isa", cat)], quiet=True)
        tell(f, f"{o} is a {v1} {cat}", [(o, "isa", cat), (o, R, v1)], quiet=True)
        if confirm:
            tell(f, f"{o} also has {v2}", [(o, R, v2)], quiet=True)
        else:
            for _ in range(3):
                tell(f, f"{o} is not {v2}", [(o, R, v2, -1)], quiet=True)


LIVES = {
    "novice (no history)": [],
    "seasoned (3 of 4 earlier generalisations held)": [
        ("animal", "size", "cat", ["dog", "horse"], "big", "small", True),
        ("tool", "weight", "hammer", ["saw", "drill"], "heavy", "lightweight", True),
        ("food", "taste", "soup", ["bread", "rice"], "salty", "sweet", True),
        ("stone", "surface", "pebble", ["granite", "slate"], "smooth", "rough", False)],
    "burnt (1 of 4 earlier generalisations held)": [
        ("animal", "size", "cat", ["dog", "horse"], "big", "small", True),
        ("tool", "weight", "hammer", ["saw", "drill"], "heavy", "lightweight", False),
        ("food", "taste", "soup", ["bread", "rice"], "salty", "sweet", False),
        ("stone", "surface", "pebble", ["granite", "slate"], "smooth", "rough", False)],
}

RED = [("rule", "colour", "is"), ("x", "red", "is"), ("M", "red", "colour")]


def red_example(f, verbose_steps=True):
    res = {}
    tell(f, "red is a dark colour", [("red", "isa", "colour"), ("red", "is", "dark")], quiet=not verbose_steps)
    res["s1_red_dark"] = f.post[("x", "red", "is")]["p"]
    res["s1_red_isa_colour"] = f.post[("M", "red", "colour")]["p"]
    if verbose_steps:
        show(f, [("x", "red", "is"), ("M", "red", "colour")])
        f.say(f"    content of (red, is, .): {f.content('red', 'is')}")
    tell(f, "red also has light shades", [("red", "is", "light")], quiet=not verbose_steps)
    v = f.post[("x", "red", "is")]
    res["s2_refined_alone"] = v["alone"]
    res["s2_rule"] = f.post[("rule", "colour", "is")]["p"]
    res["s2_rule_prior"] = f.post[("rule", "colour", "is")]["prior"]
    res["s2_red_after_feedback"] = v["p"]
    res["s2_red_isa_colour"] = f.post[("M", "red", "colour")]["p"]
    res["s2_red_down_nats"] = v["down_nats"]
    res["s2_red_own_nats"] = v["own_nats"]
    if verbose_steps:
        f.say(f"    content of (red, is, .): {f.content('red', 'is')}")
        f.say(f"    refined belief before any rule (own evidence alone)  = {v['alone']:.3f}")
        show(f, RED)
    return res


def main():
    as_json = "--json" in sys.argv
    out = {"lam": LAM, "eta0": ETA0, "p0": P0}
    if not as_json:
        print(f"calibrated: lam = {LAM:.4f} nats/statement (LR {math.exp(LAM):.2f}), eta0 = {ETA0:.4f} (p0 = {P0:.4f})")
        print("=" * 100)
        print("PART A -- the red example, novice mind (operator prior Beta(1,1): p_H = 0.5)")
        print("=" * 100)
    f = Field(Meta(), verbose=not as_json)
    out["novice"] = red_example(f)
    r = f.rules[0]
    bh, bx, bm = f.brute(r)
    out["brute_check"] = {"rule": bh, "red": bx[0], "member": bm[0]}
    err = max(abs(bh - f.post[("rule", "colour", "is")]["p"]), abs(bx[0] - f.post[("x", "red", "is")]["p"]),
              abs(bm[0] - f.post[("M", "red", "colour")]["p"]))
    out["brute_max_abs_err"] = err
    f.say(f"\n    exact enumeration of the same model: rule {bh:.6f}, red {bx[0]:.6f}, membership {bm[0]:.6f}"
          f"  (max |message - exact| = {err:.1e})")

    # questions
    qs = f.questions()
    out["questions"] = [(round(v, 4), q) for v, q in qs]
    f.say("\n    QUESTIONS (epistemic value in nats, best first):")
    for v, q in qs:
        f.say(f"      {v:.3f}  {q}")

    # the answer arrives; imagination on the new members; verification
    f.say("\n" + "=" * 100 + "\nPART B -- the question is answered; imagination, reality score, verification\n" + "=" * 100)
    tell(f, "blue is a colour", [("blue", "isa", "colour")])
    tell(f, "green is a colour", [("green", "isa", "colour")])
    for nm in ("blue", "green"):
        v = f.post[("x", nm, "is")]
        f.say(f"    IMAGINE ({nm}, is, V{{dark|light}}): reality score = {v['p']:.3f}  "
              f"[told 0.00 nats, constructed {v['down_nats']:+.2f} nats, grounded share 0.00]")
    out["imagine_blue"] = f.post[("x", "blue", "is")]["p"]
    show(f, RED)
    out["after_enumeration"] = {"rule": f.post[("rule", "colour", "is")]["p"], "red": f.post[("x", "red", "is")]["p"]}
    steps = [("blue is a dark colour", [("blue", "isa", "colour"), ("blue", "is", "dark")]),
             ("blue also has light shades", [("blue", "is", "light")]),
             ("green is a dark colour", [("green", "isa", "colour"), ("green", "is", "dark")]),
             ("green also has light shades", [("green", "is", "light")])]
    out["verification"] = []
    for text, frames in steps:
        before = f.post[("x", frames[0][0], "is")]["p"]
        tell(f, text, frames)
        show(f, RED + [("x", frames[0][0], "is")])
        out["verification"].append({"text": text, "rule": f.post[("rule", "colour", "is")]["p"],
                                    "red": f.post[("x", "red", "is")]["p"],
                                    "member_red": f.post[("M", "red", "colour")]["p"],
                                    "subject": f.post[("x", frames[0][0], "is")]["p"], "predicted_before": before})
    bh, bx, bm = f.brute(f.rules[0])
    err = abs(bh - f.post[("rule", "colour", "is")]["p"])
    out["brute_max_abs_err_3members"] = err
    f.say(f"\n    exact enumeration with three members (128 states): rule {bh:.6f} (|diff| {err:.1e})")
    mu_red = f.post[("M", "red", "colour")]["alone"]
    ceil = sig(logit(mu_red * f.theta + (1 - mu_red) * P0) + 2 * LAM)
    out["red_ceiling"] = ceil
    f.say(f"    ceiling of the red belief if the rule were certain (membership rests on one statement): {ceil:.3f}")

    # a counter-example: the rule is revised, not deleted
    f.say("\n    -- a counter-example --")
    tell(f, "black is a colour", [("black", "isa", "colour")])
    tell(f, "black is a dark colour", [("black", "isa", "colour"), ("black", "is", "dark")])
    for _ in range(3):
        tell(f, "black is not light", [("black", "is", "light", -1)], quiet=True)
    f.say("\n[.] \"black is not light\" (three times)")
    show(f, RED + [("x", "black", "is")])
    out["counterexample"] = {"rule": f.post[("rule", "colour", "is")]["p"], "red": f.post[("x", "red", "is")]["p"],
                             "black": f.post[("x", "black", "is")]["p"]}
    out["units"] = len(f.units)
    out["concepts"] = len(f.name)
    out["statements"] = f.n_stmt
    f.say(f"    memory: {len(f.name)} concepts, {len(f.units)} units, {len(f.rules)} rule, {len(f.var)} variation concept, "
          f"{f.n_stmt} statements heard")

    # PART C -- the same two statements heard by minds with different histories (P5)
    if not as_json:
        print("\n" + "=" * 100 + "\nPART C -- the same two statements, three operating histories (operator-level learning, P5)\n" + "=" * 100)
    out["histories"] = {}
    for label, life in LIVES.items():
        g = Field(Meta(), verbose=False, seed=1)
        for spec in life:
            domain(g, *spec)
        tests = dict(g.meta.tests)
        p, a, b = g.meta.p_gen()
        res = red_example(g, verbose_steps=False)
        out["histories"][label] = {"p_H": p, "beta": (a, b), "tests": {str(k): v for k, v in tests.items()},
                                   "rule": res["s2_rule"], "red": res["s2_red_after_feedback"],
                                   "refined_alone": res["s2_refined_alone"], "first": res["s1_red_dark"]}
        if not as_json:
            print(f"  {label}")
            for k, v in tests.items():
                print(f"      earlier construction {k}: test LR on new members = {v:.3f}")
            print(f"      operator belief: p_H = {p:.3f}  (Beta {a:.2f}, {b:.2f})")
            print(f"      red example -> first {res['s1_red_dark']:.2f} | refined {res['s2_refined_alone']:.2f} | "
                  f"rule {res['s2_rule']:.3f} | red after feedback {res['s2_red_after_feedback']:.3f}")

    # PART D -- what the cavity prevents: iterate the support loop WITHOUT it
    lx = math.exp(2 * LAM)
    mu = sig(ETA0 + LAM)
    px, traj = sig(ETA0 + 2 * LAM), []
    for it in range(1, 31):
        l_eff = (px / (1 - px)) / (P0 / (1 - P0))          # the instance's posterior fed upward as if it were evidence
        d = P0 * l_eff + 1 - P0
        ph = sig(math.log((mu * l_eff + (1 - mu) * d) / d))  # p_H = 0.5
        prior_x = ph * mu + (1 - ph * mu) * P0               # the rule's posterior (not cavity) fed back down
        px = sig(logit(prior_x) + math.log(lx))
        traj.append((it, ph, px))
    out["resonance"] = [(it, round(ph, 4), round(px, 4)) for it, ph, px in traj if it in (1, 2, 3, 5, 10, 30)]
    if not as_json:
        print("\n" + "=" * 100 + "\nPART D -- the same loop without the cavity correction (what section 7 guards against)\n" + "=" * 100)
        for it, ph, px in traj:
            if it in (1, 2, 3, 5, 10, 30):
                print(f"  sweep {it:2d}: rule {ph:.3f}   red {px:.3f}")
        print(f"  with the cavity (Part A) the fixed point is reached in one sweep: rule {out['novice']['s2_rule']:.3f}, "
              f"red {out['novice']['s2_red_after_feedback']:.3f}")
    # D2 -- what the grounded/constructed separation prevents: imagined instances counted as evidence
    b = P0
    d2 = []
    for n_imag in (0, 1, 3, 10, 30):
        eta = math.log((mu * lx + (1 - mu) * (b * lx + 1 - b)) / (b * lx + 1 - b))       # red's real message, p_H = 0.5
        ph = sig(eta)
        for _ in range(50):                                                              # iterate to the fixed point
            prior_x = ph * mu + (1 - ph * mu) * b                                         # imagined instance of a member heard once
            l_im = (prior_x / (1 - prior_x)) / (b / (1 - b))                              # ... treated as if it had been told
            dd = b * l_im + 1 - b
            ph = sig(eta + n_imag * math.log((mu * l_im + (1 - mu) * dd) / dd))
        d2.append((n_imag, ph))
    out["imagination_as_evidence"] = [(n, round(v, 4)) for n, v in d2]
    if not as_json:
        print("\n  D2 -- imagined instances written back as if they were evidence (no grounded/constructed separation):")
        for n, v in d2:
            print(f"    {n:2d} imagined members -> rule {v:.3f}")
        print("    with the separation the rule stays at 0.740 for any number of imagined members (they carry 0 nats)")

    # PART E -- exact retraction (the ledger): retract statement 2 and compare with the state after statement 1
    g = Field(Meta(), verbose=False)
    tell(g, "red is a dark colour", [("red", "isa", "colour"), ("red", "is", "dark")])
    before = {k: v["p"] for k, v in g.post.items()}
    tell(g, "red also has light shades", [("red", "is", "light")])
    mid = {k: v["p"] for k, v in g.post.items()}
    g.retract(2)
    after = {k: v["p"] for k, v in g.post.items()}
    diff = max(abs(after[k] - before[k]) for k in before)
    out["retraction"] = {"max_abs_diff": diff, "same_keys": sorted(map(str, after)) == sorted(map(str, before)),
                         "rule_dormant": not g.licensed(g.rules[0]), "units": len(g.units), "concepts": len(g.name)}
    if not as_json:
        print("\n" + "=" * 100 + "\nPART E -- exact retraction of one experience\n" + "=" * 100)
        print(f"  after [1]: {before}")
        print(f"  after [2]: { {k: round(v, 3) for k, v in mid.items()} }")
        print(f"  retract [2]: {after}")
        print(f"  max |belief after retraction - belief after [1]| = {diff:.1e}; rule kept but dormant: {not g.licensed(g.rules[0])}; "
              f"the concept [light] stays in the bank (it was heard)")

    # PART F -- lexical settle on character-trigram codes (Turkish inflection), same energy and thresholds
    def tri_code(word, d=256, seed=7):
        w = "^" + word + "$"
        v = np.zeros(d)
        for i in range(len(w) - 2):
            h = np.random.default_rng(zlib.crc32(f"{seed}:{w[i:i + 3]}".encode("utf-8"))).standard_normal(d)
            v += h / np.linalg.norm(h)
        return v / np.linalg.norm(v)
    base = "kırmızı"
    lex = []
    for w in ("kırmızı", "kırmızının", "kırmızıdır", "kırmızıya", "kırıcı", "mavi", "renk", "rengi"):
        ref = base if w not in ("renk", "rengi") else "renk"
        c = float(tri_code(w) @ tri_code(ref))
        lex.append((w, ref, c, 8.0 * (1 - c)))
    out["lexical"] = [(w, ref, round(c, 3), round(e, 2)) for w, ref, c, e in lex]
    if not as_json:
        print("\n" + "=" * 100 + "\nPART F -- lexical settle on character-trigram codes (d = 256, a = 8, c_mode = 3)\n" + "=" * 100)
        for w, ref, c, e in lex:
            verdict = "same concept attractor" if e < 3.0 else "new concept"
            print(f"  {w:12s} vs [{ref}]: cosine {c:.3f}, energy {e:.2f} -> {verdict}")

    # PART G -- sensitivities
    sens = {}
    f2 = Field(Meta(), verbose=False)
    red_example(f2, verbose_steps=False)
    for alpha in (0.5, 0.82, 1.0, 2.0):
        f2.meta.alpha_cat = alpha
        qs2 = f2.questions()
        sens[alpha] = [(round(v, 3), q.split("  (")[0]) for v, q in qs2][:2]
    out["question_sensitivity_alpha"] = sens
    # the owner's four numbers as one coherent posterior: membership taken as given, operator prior fitted
    lx2 = math.exp(2 * LAM)
    variants = {}
    for label, mu_v, ph_v in (("membership given, p_H = 0.27 (fitted to 0.77)", 1.0, 0.27),
                              ("membership given, p_H = 0.50", 1.0, 0.5),
                              ("membership graded 0.23, p_H = 0.50 (the trace)", sig(ETA0 + LAM), 0.5)):
        dd = P0 * lx2 + 1 - P0
        rule = sig(logit(ph_v) + math.log((mu_v * lx2 + (1 - mu_v) * dd) / dd))
        prior_x = ph_v * mu_v + (1 - ph_v * mu_v) * P0
        variants[label] = (rule, sig(logit(prior_x) + 2 * LAM))
    out["variants"] = variants
    # seed / dimension robustness of the structural decisions
    rob = {}
    for d in (64, 256):
        ok = 0
        for seed in range(200):
            h = Field(Meta(), d=d, seed=seed, verbose=False)
            r1 = red_example(h, verbose_steps=False)
            ok += int(abs(r1["s2_rule"] - out["novice"]["s2_rule"]) < 1e-12 and len(h.units) == 2 and len(h.rules) == 1)
        rob[d] = ok
    out["robust_200_seeds"] = rob
    # order of arrival: both statement orders and both frame orders inside statement 1
    orders = {}
    s1a = ("red is a dark colour", [("red", "isa", "colour"), ("red", "is", "dark")])
    s1b = ("red is a dark colour", [("red", "is", "dark"), ("red", "isa", "colour")])
    s2 = ("red also has light shades", [("red", "is", "light")])
    for label, seq in (("S1 then S2", [s1a, s2]), ("S1 (frames swapped) then S2", [s1b, s2]),
                       ("S2 then S1", [s2, s1a]), ("S2 then S1 (frames swapped)", [s2, s1b])):
        h = Field(Meta(), verbose=False)
        for text, frames in seq:
            tell(h, text, frames)
        orders[label] = (h.post[("rule", "colour", "is")]["p"], h.post[("x", "red", "is")]["p"], h.post[("M", "red", "colour")]["p"])
    out["order_invariance"] = orders
    if not as_json:
        print("\n" + "=" * 100 + "\nPART G -- sensitivities\n" + "=" * 100)
        for label, v in orders.items():
            print(f"  order [{label}]: rule {v[0]:.6f}, red {v[1]:.6f}, membership {v[2]:.6f}")
        for alpha, top in sens.items():
            print(f"  category openness alpha = {alpha}: top question = {top[0][1]} ({top[0][0]} nats); second = {top[1][1]} ({top[1][0]})")
        for label, (rule, red) in variants.items():
            print(f"  {label}: rule {rule:.3f}, red {red:.3f}")
        for d, ok in rob.items():
            print(f"  d = {d}: identical structure and beliefs in {ok} of 200 random code seeds")
    # PART H -- randomised exactness check of the messages (theta < 1, negative evidence, up to five members)
    rnd = random.Random(3)
    worst = 0.0
    for trial in range(300):
        th = rnd.choice([1.0, 0.95, 0.8, 0.6])
        h = Field(Meta(a=rnd.uniform(0.3, 4.0), b=rnd.uniform(0.3, 4.0)), theta=th, verbose=False, seed=trial)
        tell(h, "t is a v1 c", [("t", "isa", "c"), ("t", "is", "v1")])
        tell(h, "t also has v2", [("t", "is", "v2")])
        for j in range(rnd.randint(1, 4)):
            o = f"o{j}"
            for _ in range(rnd.randint(0, 3)):
                tell(h, "membership", [(o, "isa", "c")])
            for _ in range(rnd.randint(0, 3)):
                tell(h, "positive", [(o, "is", rnd.choice(["v1", "v2"]))])
            for _ in range(rnd.randint(0, 3)):
                tell(h, "negative", [(o, "is", rnd.choice(["v1", "v2"]), -1)])
        rr = h.rules[0]
        bh, bx, bm = h.brute(rr)
        e = abs(bh - h.post[("rule", "c", "is")]["p"])
        for (nm, _, _), x, m in zip(h.members(rr), bx, bm):
            e = max(e, abs(x - h.post[("x", nm, "is")]["p"]), abs(m - h.post[("M", nm, "c")]["p"]))
        worst = max(worst, e)
    out["random_trees_max_abs_err"] = worst
    a_chk, b_chk = adf(0.4, 2.3, 1.0)
    out["adf_identity_err"] = max(abs(a_chk - 0.4), abs(b_chk - 2.3))
    if not as_json:
        print("\n" + "=" * 100 + "\nPART H -- randomised exactness check\n" + "=" * 100)
        print(f"  300 random trees (theta in 1, 0.95, 0.8, 0.6; positive and negative evidence; up to five members): "
              f"max |message - enumeration| = {worst:.1e}")
        print(f"  operator filter with LR = 1 returns the prior: max deviation {out['adf_identity_err']:.1e}")
    if as_json:
        print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
