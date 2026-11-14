#!/usr/bin/env python3
"""zsim.py -- a literal, instrumented reading of LEAD_DESIGN.md §1-§2 (units, EXPECT ties lag 0/1, settle with
kWTA_F, always-on LEARN, candidate ties for every focus pair, recruitment of composed units, interoceptive
events) plus SWITCHES for the smallest fixes proposed in attack/dynamics-scale.md.

This is an attack tool, not product code.  One provenance class (sensed, tau = 1) is enough for growth, cost and
dynamics; the six-class evidence vector only multiplies the bytes per tie (accounted for analytically).

Reading of the design where the text is silent (stated in the report as well):
  * `in` = 1 for every unit carried by an event of this moment, added at every settling iteration.
  * a(0) of a moment = decay * (focus of the previous moment)            ("context units stay active with decay")
  * kWTA_F keeps the F largest positive values; then a is clipped to [0,1] (the design says a in [0,1]).
  * a tie's "use" is incremented whenever its source is in the focus (the tie is traversed by settle/predict).
  * lag-1 candidate ties are created for (u in focus(t-1), v in focus(t))  (the text only says "pairs in the focus";
    without this reading no lag-1 tie can ever exist and §2.3, chunking and imitation have nothing to work on).
  * "v occurred" = v is carried by an event of the moment (an atomic symbol, or a composed unit all of whose parts
    are events).  Units that are in the focus only because settling recalled them did NOT occur.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np, time, math, json, sys, resource

ATOM, COMP, INTERO = 0, 1, 2


class Cfg:
    def __init__(s, **kw):
        s.F = 7; s.T = 6; s.decay = 0.5; s.g = 1.0; s.k = 1.0
        s.ref = "half"       # 'half' = design: (e - 1/2)*2      | 'base' = fix: e - b_v, prior mean b_v
        s.neg = "ew"         # 'ew' = design: w- += g a e        | 'plain': w- += g a | 'lazy': implicit (fix)
        s.clamp = False      # fix: an evented unit cannot be pushed out of the focus by expectation
        s.create = "focus"   # 'focus' = design: every focus pair | 'events': both ends carried by events (fix)
        s.lag1 = True
        s.intero = "every"   # 'every' = design §2.6            | 'change': event only when the level changes | 'none'
        s.recruit = "input"  # 'input': tuple of the moment's external events | 'focus': tuple of focus atoms | 'none'
        s.K = None           # fix: per-unit bound on out-ties per lag (bag), None = unbounded (design)
        s.budget = None      # global tie budget ("budget given by the body")
        s.prune = "design"   # 'design': low c and low use, oldest first | 'z': least significant deviation from base rate
        s.drive = True       # compute D and relief (§5) for the interoceptive channel
        s.skip_part = False  # fix: no EXPECT(lag 0) tie between a composed unit and its own parts (PART says it)
        s.floor = 0.0        # activation floor of kWTA (design: none -- any positive value stays if there is room)
        s.settle = "kwta"    # 'kwta' = design §2.2 recurrence | 'monotone' = fix (clamped inputs, greedy admission)
        s.theta_r = 0.02     # admission threshold of the monotone settle
        s.__dict__.update(kw)


class Adj:
    """out-ties of one unit for one lag (struct of growable arrays + dst->pos dict)"""
    __slots__ = ("dst", "wp", "wn", "use", "born", "n", "idx")

    def __init__(s):
        s.dst = np.empty(4, np.int32); s.wp = np.empty(4); s.wn = np.empty(4)
        s.use = np.empty(4, np.int32); s.born = np.empty(4, np.int32); s.n = 0; s.idx = {}

    def reserve(s, extra):
        need = s.n + extra
        if need > len(s.dst):
            cap = max(need, 2 * len(s.dst))
            for name in ("dst", "wp", "wn", "use", "born"):
                old = getattr(s, name); new = np.empty(cap, old.dtype); new[:s.n] = old[:s.n]; setattr(s, name, new)

    def compact(s, keep):
        n = s.n; m = int(keep.sum())
        for name in ("dst", "wp", "wn", "use", "born"):
            a = getattr(s, name); a[:m] = a[:n][keep]
        s.n = m
        s.idx = {int(v): i for i, v in enumerate(s.dst[:m])}


class Engine:
    def __init__(s, cfg, ncap=300000):
        s.cfg = cfg; s.ncap = ncap; s.n = 0; s.t = 0
        s.out = [[None] * ncap, [None] * ncap]
        s.X = [np.zeros(ncap), np.zeros(ncap)]     # exposure of a source (lazy negatives)
        s.occ = np.zeros(ncap); s.birth = np.zeros(ncap, np.int32); s.kind = np.zeros(ncap, np.int8)
        s.acc = np.zeros(ncap); s.mark = np.zeros(ncap, bool)
        s.sym = {}; s.tuples = {}; s.parts = {}
        e = np.empty(0, np.int32)
        s.prev_focus = (e, np.empty(0)); s.prev_ev = set(); s.pending = []; s.last_levels = {}
        s.nties = [0, 0]
        # counters (cumulative)
        s.C = dict(reads=0, upd=0, created0=0, created1=0, created_nonevent=0, its=0, nonconv=0, period2=0,
                   ext_in=0, ext_n=0, comp_in=0, comp_n=0, intero_in=0, recalled_in=0, fsize=0, sat=0, maxraw=0.0,
                   moments=0, recruited=0, recognised=0, pruned=0, prune_passes=0, relief_pos=0, relief_neg=0,
                   dD=0.0, evict=0, boundary_ties=0, setflip=0)
        s._pe = {}; s._pf = [0.0, 0]; s._dD = 0.0; s._ops_moment = 0

    # ---------------------------------------------------------------- units
    def new_unit(s, kind):
        u = s.n
        if u >= s.ncap:
            raise RuntimeError("unit capacity exceeded")
        s.n += 1; s.birth[u] = s.t; s.kind[u] = kind
        return u

    def unit(s, sym, kind=ATOM):
        u = s.sym.get(sym)
        if u is None:
            u = s.new_unit(kind); s.sym[sym] = u
        return u

    def base(s, d):
        return s.occ[d] / (s.t - s.birth[d] + 1.0)

    # ---------------------------------------------------------------- tie maths
    def _W(s, A, n, u, lag):
        if s.cfg.neg == "lazy":
            return s.X[lag][u] - A.wn[:n]
        return A.wp[:n] + A.wn[:n]

    def weights(s, u, lag):
        """settling weight of every out-tie of u:  design (e-1/2)*2 = (W+ - W-)/(W+k);  fix e - b_v = (W+ - b W)/(W+k)"""
        A = s.out[lag][u]
        if A is None or A.n == 0:
            return None, None
        n = A.n; d = A.dst[:n]; wp = A.wp[:n]; W = s._W(A, n, u, lag); k = s.cfg.k
        if s.cfg.ref == "half":
            return d, (2.0 * wp - W) / (W + k)
        return d, (wp - s.base(d) * W) / (W + k)

    def expect(s, u, lag):
        A = s.out[lag][u]
        if A is None or A.n == 0:
            return None, None
        n = A.n; d = A.dst[:n]; wp = A.wp[:n]; W = s._W(A, n, u, lag); k = s.cfg.k
        pm = 0.5 if s.cfg.ref == "half" else s.base(d)
        return d, (wp + pm * k) / (W + k)

    # ---------------------------------------------------------------- fix: monotone admission
    def settle_monotone(s, ev, ctx_ids, ctx_vals, count=True):
        """Evented units are in the focus (a = 1, cannot be removed).  Free slots are filled one at a time by the
        best supported candidate (support = decayed context + sum_u a_u (e_uv - b_v) from the units already in
        the focus), admitted with a = min(1, support) and then frozen.  No unit ever leaves within a moment, so the
        procedure terminates after at most F - |events| admissions and cannot oscillate.  Each focus unit's ties
        are read once."""
        c = s.cfg; F = c.F; acc = s.acc; reads = 0
        ids = list(dict.fromkeys(ev.tolist()))[:F]; vals = [1.0] * len(ids); inf = set(ids)
        touched = []
        if len(ctx_ids):
            acc[ctx_ids] += ctx_vals; touched.append(ctx_ids)
        def spread(u, au):
            d, w = s.weights(u, 0)
            if d is None:
                return 0
            acc[d] += au * w; touched.append(d)
            return len(d)
        for u in ids:
            reads += spread(u, 1.0)
        its = 1
        while len(ids) < F and its < c.T + 1 and touched:
            cat = np.concatenate(touched); v = acc[cat]
            order = np.argsort(-v, kind="stable")
            pick = None
            for j in order[: F + 8].tolist():
                if v[j] <= c.theta_r:
                    break
                if int(cat[j]) not in inf:
                    pick = (int(cat[j]), float(min(1.0, v[j]))); break
            if pick is None:
                break
            ids.append(pick[0]); vals.append(pick[1]); inf.add(pick[0])
            reads += spread(pick[0], pick[1]); its += 1
        for d in touched:
            acc[d] = 0.0
        ids = np.array(ids, np.int32); vals = np.array(vals); o = np.argsort(ids)
        if count:
            C = s.C; C["reads"] += reads; C["its"] += its; s._ops_moment += reads
        return ids[o], vals[o]

    # ---------------------------------------------------------------- §2.2 settle
    def settle(s, ev, ctx_ids, ctx_vals, count=True):
        c = s.cfg; F = c.F; acc = s.acc
        if c.settle == "monotone":
            return s.settle_monotone(ev, ctx_ids, ctx_vals, count)
        ids, vals = ctx_ids, ctx_vals
        cache = {}; sets = []; reads = 0; nonconv = 1; per2 = 0; sat = 0; maxraw = 0.0; bt = 0
        for it in range(c.T):
            touched = [ev]
            acc[ev] += 1.0
            if len(ids):
                acc[ids] += c.decay * vals; touched.append(ids)
            for u, au in zip(ids.tolist(), vals.tolist()):
                dw = cache.get(u)
                if dw is None:
                    dw = s.weights(u, 0); cache[u] = dw
                d, w = dw
                if d is None:
                    continue
                acc[d] += au * w; touched.append(d); reads += len(d)
            if c.clamp:
                acc[ev] = np.maximum(acc[ev], 1.0)
            cat = np.concatenate(touched); v = acc[cat]; m = len(cat); lim = F * (F + 3)
            if m > lim:
                part = np.argpartition(v, m - lim)[m - lim:]; cand = cat[part]; cv = v[part]
            else:
                cand, cv = cat, v
            cand, ui = np.unique(cand, return_index=True); cv = cv[ui]
            pos = cv > c.floor; cand = cand[pos]; cv = cv[pos]
            if c.clamp and len(cand) > F:
                # evented units first (they cannot be suppressed), then the rest by value
                isev = s.mark[cand]; order = np.lexsort((-cv, ~isev))[:F]
                cand = cand[order]; cv = cv[order]
            elif len(cand) > F:
                order = np.argsort(-cv, kind="stable")
                if cv[order[F - 1]] == cv[order[F]]:
                    bt += 1
                order = order[:F]; cand = cand[order]; cv = cv[order]
            acc[cat] = 0.0
            if len(cv):
                maxraw = max(maxraw, float(cv.max())); sat += int((cv >= 1.0).sum())
            cv = np.minimum(cv, 1.0)
            o = np.argsort(cand); cand = cand[o]; cv = cv[o]
            fs = frozenset(cand.tolist())
            same = len(sets) and fs == sets[-1] and len(cv) == len(vals) and np.abs(cv - vals).max() < 1e-3
            if len(sets) >= 2 and fs == sets[-2] and fs != sets[-1]:
                per2 = 1
            sets.append(fs); ids, vals = cand.astype(np.int32), cv
            if same:
                nonconv = 0; per2 = 0
                break
        # nonconv = values still moving at T; setflip = the focus SET itself still changed at the last iteration
        setflip = int(len(sets) >= 2 and sets[-1] != sets[-2])
        if count:
            C = s.C; C["reads"] += reads; C["its"] += len(sets); C["nonconv"] += nonconv; C["period2"] += per2
            C["setflip"] += setflip
            C["sat"] += sat; C["maxraw"] = max(C["maxraw"], maxraw); C["boundary_ties"] += bt
            s._ops_moment += reads
        return ids, vals

    # ---------------------------------------------------------------- §2.5 learn
    def learn_from(s, u, au, lag, evl):
        c = s.cfg; A = s.out[lag][u]; ga = c.g * au; k = c.k
        if c.neg == "lazy":
            s.X[lag][u] += ga; s.C["upd"] += 1
            if A is None or A.n == 0:
                return
            idx = A.idx
            for v in evl:
                j = idx.get(v)
                if j is not None:
                    if lag == 1:
                        W = s.X[lag][u] - ga - A.wn[j]
                        pm = 0.5 if c.ref == "half" else s.occ[v] / (s.t - s.birth[v] + 1.0)
                        e = (A.wp[j] + pm * k) / (W + k)
                        if e > s._pe.get(v, 0.0):
                            s._pe[v] = e
                    A.wp[j] += ga; s.C["upd"] += 1
            return
        if A is None or A.n == 0:
            return
        n = A.n; d = A.dst[:n]; wp = A.wp[:n]; wn = A.wn[:n]; use = A.use[:n]
        occ = s.mark[d]; W = wp + wn
        if c.drive:
            Dbef = float((use * (k / (W + k))).sum())
        pm = 0.5 if c.ref == "half" else s.base(d)
        e = (wp + pm * k) / (W + k)
        io = np.nonzero(occ)[0]
        if lag == 1:
            s._pf[0] += float(e.sum() - e[io].sum()); s._pf[1] += n - len(io)
            for j in io.tolist():
                v = int(d[j])
                if e[j] > s._pe.get(v, 0.0):
                    s._pe[v] = float(e[j])
        inc = ga * e if c.neg == "ew" else np.full(n, ga)
        inc[io] = 0.0
        wn += inc; wp[io] += ga; use += 1
        s.C["upd"] += n; s._ops_moment += n
        if c.drive:
            s._dD += float((use * (k / (wp + wn + k))).sum()) - Dbef

    def create_pairs(s, us, aus, vs, avs, lag, u_ev):
        """candidate ties for every (u, v) pair; u_ev = set of units that were carried by events in u's moment"""
        c = s.cfg; t = s.t; vl = vs.tolist(); avl = avs.tolist(); made = 0; kind = s.kind
        for u, au in zip(us.tolist(), aus.tolist()):
            A = s.out[lag][u]
            if A is None:
                A = Adj(); s.out[lag][u] = A
            idx = A.idx; ue = u in u_ev
            for v, av in zip(vl, avl):
                if (v == u and lag == 0) or v in idx:
                    continue
                if c.skip_part and lag == 0 and ((kind[u] == COMP and v in s.parts[u]) or
                                                 (kind[v] == COMP and u in s.parts[v])):
                    continue          # fix: PART already says it; no EXPECT tie between a composed unit and its parts
                ve = bool(s.mark[v])
                if not (ue and ve):
                    if c.create == "events":
                        continue
                    s.C["created_nonevent"] += 1
                w0 = c.g * au * av
                if c.K is not None and A.n >= c.K:
                    if not s._evict(A, u, lag, v, w0):
                        continue
                A.reserve(1); j = A.n
                A.dst[j] = v; A.wp[j] = w0; A.use[j] = 1; A.born[j] = t
                A.wn[j] = (s.X[lag][u] - w0) if c.neg == "lazy" else 0.0
                idx[v] = j; A.n += 1; made += 1
                if c.drive and c.neg != "lazy":
                    s._dD += c.k / (w0 + c.k)
        s.nties[lag] += made; s.C["created%d" % lag] += made; s.C["upd"] += made

    def zscore(s, A, u, lag):
        n = A.n; d = A.dst[:n]; wp = A.wp[:n]; W = s._W(A, n, u, lag)
        return np.abs(wp - s.base(d) * W) / np.sqrt(W + s.cfg.k)

    def _evict(s, A, u, lag, v, w0):
        """bounded bag: the newcomer replaces the least significant tie of the same source, if it is more significant"""
        z = s.zscore(A, u, lag); j = int(np.argmin(z))
        bnew = s.occ[v] / (s.t - s.birth[v] + 1.0)
        znew = abs(w0 - bnew * w0) / math.sqrt(w0 + s.cfg.k)
        if z[j] >= znew:
            return False
        keep = np.ones(A.n, bool); keep[j] = False; A.compact(keep)
        s.nties[lag] -= 1; s.C["evict"] += 1
        return True

    # ---------------------------------------------------------------- pruning under a global budget (§8, §10)
    def prune_to(s, target):
        c = s.cfg; refs = []; Cs = []; Us = []; Bs = []; Zs = []
        for lag in (0, 1):
            out = s.out[lag]
            for u in range(s.n):
                A = out[u]
                if A is None or A.n == 0:
                    continue
                n = A.n; W = s._W(A, n, u, lag)
                refs.append((lag, u, n))
                if c.prune == "design":
                    Cs.append(W / (W + c.k))
                    Us.append(A.use[:n] if c.neg != "lazy" else (s.X[lag][u] - A.wn[:n]))
                    Bs.append(A.born[:n])
                else:
                    Zs.append(s.zscore(A, u, lag))
        total = sum(r[2] for r in refs); drop = total - target
        if drop <= 0:
            return 0
        if c.prune == "design":
            Cc = np.concatenate(Cs); Uu = np.concatenate(Us).astype(float); Bb = np.concatenate(Bs).astype(float)
            q = np.linspace(0.1, 0.9, 9)
            dc = np.searchsorted(np.quantile(Cc, q), Cc, side="right")
            du = np.searchsorted(np.quantile(Uu, q), Uu, side="right")
            key = np.maximum(dc, du) * 1e9 + Bb          # low c AND low use first; oldest first inside a class
        else:
            key = np.concatenate(Zs)
        thr = np.partition(key, drop - 1)[drop - 1]
        victim = key <= thr
        off = 0; removed = 0
        for lag, u, n in refs:
            vm = victim[off:off + n]; off += n
            nv = int(vm.sum())
            if nv:
                s.out[lag][u].compact(~vm); s.nties[lag] -= nv; removed += nv
        s.C["pruned"] += removed; s.C["prune_passes"] += 1
        return removed

    # ---------------------------------------------------------------- one moment (§2)
    def moment(s, ext_syms):
        c = s.cfg; s.t += 1; C = s.C; s._ops_moment = 0
        ext = [s.unit(x) for x in ext_syms]
        evl = list(ext); cu = None
        if c.recruit == "input" and len(ext) > 1:
            key = tuple(ext); cu = s.tuples.get(key)
            if cu is None:
                cu = s.new_unit(COMP); s.tuples[key] = cu; s.parts[cu] = key; C["recruited"] += 1
            else:
                C["recognised"] += 1
            evl.append(cu)
        nint = len(s.pending)
        evl += s.pending
        ev = np.array(evl, np.int32)
        s.occ[ev] += 1.0; s.mark[ev] = True
        pf_ids, pf_vals = s.prev_focus
        # LEARN, lag 1: ties from the last focus are judged by what this moment brought
        s._pe = {}; s._pf = [0.0, 0]; s._dD = 0.0
        for u, au in zip(pf_ids.tolist(), pf_vals.tolist()):
            s.learn_from(u, au, 1, evl)
        # RELATE
        ids, vals = s.settle(ev, pf_ids, pf_vals * c.decay)
        # LEARN, lag 0 + candidate ties
        for u, au in zip(ids.tolist(), vals.tolist()):
            s.learn_from(u, au, 0, evl)
        evset = set(evl)
        s.create_pairs(ids, vals, ids, vals, 0, evset)
        if c.lag1 and len(pf_ids):
            s.create_pairs(pf_ids, pf_vals, ids, vals, 1, s.prev_ev)
        # bookkeeping on the focus
        fset = set(ids.tolist())
        C["moments"] += 1; C["fsize"] += len(fset)
        C["ext_in"] += sum(1 for x in ext if x in fset); C["ext_n"] += len(ext)
        if cu is not None:
            C["comp_in"] += int(cu in fset); C["comp_n"] += 1
        C["intero_in"] += sum(1 for x in s.pending if x in fset)
        C["recalled_in"] += sum(1 for x in fset if x not in evset)
        if c.recruit == "focus":
            atoms = tuple(sorted(x for x in fset if s.kind[x] != COMP))
            if len(atoms) > 1:
                if atoms in s.tuples:
                    C["recognised"] += 1
                else:
                    q = s.new_unit(COMP); s.tuples[atoms] = q; s.parts[q] = atoms; C["recruited"] += 1
        s.mark[ev] = False
        s.prev_focus = (ids, vals); s.prev_ev = evset
        # FEEL (§2.6): interoceptive events enter the next moment like any event
        s.pending = []
        if c.intero != "none":
            occ_ext = [v for v in evl[:len(evl) - nint]]
            s1 = float(np.mean([1.0 - s._pe.get(v, 0.5) for v in occ_ext])) if occ_ext else 0.0
            s2 = s._pf[0] / s._pf[1] if s._pf[1] else 0.0
            surprise = s1 + s2
            C["dD"] += s._dD
            if s._dD < 0: C["relief_pos"] += 1
            elif s._dD > 0: C["relief_neg"] += 1
            levels = {
                "surprise": min(3, int(surprise / 0.4)),
                "relief": (0 if s._dD > 1e-9 else (2 if s._dD < -1e-9 else 1)),
                "load": int(math.log(s._ops_moment + 1.0, 4.0)),
                "conflict": s._conflict(ids),
            }
            for ch, lv in levels.items():
                if c.intero == "every" or s.last_levels.get(ch) != lv:
                    s.pending.append(s.unit("~%s:%d" % (ch, lv), INTERO))
            s.last_levels = levels
        if c.budget is not None and s.nties[0] + s.nties[1] > c.budget * 1.1:
            s.prune_to(int(c.budget))
        return ids, vals

    def _conflict(s, ids):
        """§2.6: mass of ties inside the focus with f near 1/2 and high c  ->  sum c*(1-|2f-1|), 3 levels"""
        k = s.cfg.k; m = 0.0; il = ids.tolist()
        for u in il:
            A = s.out[0][u]
            if A is None or A.n == 0:
                continue
            for v in il:
                j = A.idx.get(v)
                if j is None:
                    continue
                wp = A.wp[j]; W = (s.X[0][u] - A.wn[j]) if s.cfg.neg == "lazy" else wp + A.wn[j]
                if W > 0:
                    m += (W / (W + k)) * (1.0 - abs(2.0 * wp / W - 1.0))
        return 0 if m < 0.5 else (1 if m < 2.0 else 2)

    # ---------------------------------------------------------------- probes (no learning)
    def probe(s, syms, cand_ids):
        """present events (no context), settle, return (focus ids, net input from the focus to each candidate)"""
        ev = np.array([s.sym[x] for x in syms if x in s.sym], np.int32)
        if len(ev) == 0:
            return np.empty(0, np.int32), np.zeros(len(cand_ids))
        s.mark[ev] = True
        e = np.empty(0, np.int32)
        ids, vals = s.settle(ev, e, np.empty(0), count=False)
        s.mark[ev] = False
        acc = s.acc; touched = []
        for u, au in zip(ids.tolist(), vals.tolist()):
            d, w = s.weights(u, 0)
            if d is None:
                continue
            acc[d] += au * w; touched.append(d)
        net = acc[cand_ids].copy()
        for d in touched:
            acc[d] = 0.0
        return ids, net

    def degrees(s):
        d0 = np.array([0 if A is None else A.n for A in s.out[0][:s.n]])
        d1 = np.array([0 if A is None else A.n for A in s.out[1][:s.n]])
        return d0, d1


# =================================================================== toy world (random vocabulary, no domain)
class World:
    """Entities (Zipf), categories, relations, values.  A fact is (entity, relation) -> value; most facts are the
    category default, a share p_exc are exceptions owned by one entity.  A moment is one statement: the three
    symbols (entity, relation, value) arrive together; consecutive statements stay on the same entity with p_stay.
    All names are random ids: nothing is special to any domain."""

    def __init__(s, seed=1, E=3000, C=30, R=12, V=400, rel_per_cat=5, p_exc=0.1, noise=0.05, p_stay=0.5, zipf=1.0,
                 prefix=""):
        rng = np.random.default_rng(seed); s.rng = rng
        s.E, s.C, s.R, s.V, s.noise, s.p_stay, s.prefix = E, C, R, V, noise, p_stay, prefix
        s.cat = rng.integers(0, C, E)
        s.rels = np.stack([rng.choice(R, rel_per_cat, replace=False) for _ in range(C)])
        s.default = rng.integers(0, V, (C, R))
        s.is_exc = rng.random((E, R)) < p_exc
        s.exc_val = rng.integers(0, V, (E, R))
        p = 1.0 / np.arange(1, E + 1) ** zipf; s.p = p / p.sum()
        s.seen = {}

    def fact(s, e, r):
        return int(s.exc_val[e, r]) if s.is_exc[e, r] else int(s.default[s.cat[e], r])

    def stream(s, N):
        rng = s.rng; ents = rng.choice(s.E, N, p=s.p); stay = rng.random(N) < s.p_stay
        ri = rng.integers(0, s.rels.shape[1], N); nz = rng.random(N) < s.noise; rv = rng.integers(0, s.V, N)
        e = int(ents[0]); P = s.prefix
        for i in range(N):
            if not stay[i]:
                e = int(ents[i])
            r = int(s.rels[s.cat[e], ri[i]])
            if nz[i]:
                v = int(rv[i])
            else:
                v = s.fact(e, r); s.seen[(e, r)] = s.seen.get((e, r), 0) + 1
            yield ("%sE%d" % (P, e), "%sR%d" % (P, r), "%sV%d" % (P, v))


def recall(eng, world, rng, n_per_band=150):
    """facts met at least once: present (entity, relation), ask which value the focus supports.
    r1 = true value has the largest net input among all value units AND that input is > 0 (it could enter a focus);
    infocus = true value is in the settled focus."""
    P = world.prefix
    vals = [(v, eng.sym.get("%sV%d" % (P, v))) for v in range(world.V)]
    vals = [(v, u) for v, u in vals if u is not None]
    if not vals:
        return {}
    vid = np.array([u for _, u in vals], np.int32); vname = np.array([v for v, _ in vals])
    seen = list(world.seen.keys())
    bands = {"head": [], "mid": [], "tail": [], "exception": []}
    for (e, r) in seen:
        b = "head" if e < 100 else ("mid" if e < 1000 else "tail")
        bands[b].append((e, r))
        if world.is_exc[e, r]:
            bands["exception"].append((e, r))
    out = {}
    # completion through composed units (NOT in the design): a composed unit whose other parts are all active
    # offers its missing part, weighted by how often that unit occurred.  Index: (part, part) -> composed units.
    pidx = {}
    for key, cu in eng.tuples.items():
        if len(key) == 3:
            pidx.setdefault((key[0], key[1]), []).append((eng.occ[cu], key[2]))
    for b, lst in bands.items():
        if not lst:
            continue
        pick = [lst[i] for i in rng.choice(len(lst), min(n_per_band, len(lst)), replace=False)]
        r1 = inf = ra = 0; rp = 0
        for (e, r) in pick:
            ue = eng.sym.get("%sE%d" % (P, e)); ur = eng.sym.get("%sR%d" % (P, r)); uv = eng.sym.get("%sV%d" % (P, world.fact(e, r)))
            cands = pidx.get((ue, ur), [])
            if cands and max(cands)[1] == uv:
                rp += 1
        out.setdefault(b, {})["r1_part_completion"] = rp / len(pick)
        for (e, r) in pick:
            v = world.fact(e, r)
            ids, net = eng.probe(("%sE%d" % (P, e), "%sR%d" % (P, r)), vid)
            j = int(np.argmax(net))
            if vname[j] == v and net[j] > 0:
                r1 += 1
            if vname[j] == v and (net != net[j]).any():
                ra += 1
            vu = eng.sym.get("%sV%d" % (P, v))
            if vu is not None and vu in set(ids.tolist()):
                inf += 1
        out[b].update(dict(n=len(pick), r1=r1 / len(pick), infocus=inf / len(pick), r1_anysign=ra / len(pick)))
    # association recall: present the entity alone; which share of the values it was met with is among the
    # 5 best supported value units (support > 0)?  and where do its true values rank?
    ent_vals = {}
    for (e, r) in seen:
        ent_vals.setdefault(e, set()).add(world.fact(e, r))
    name2col = {int(v): i for i, v in enumerate(vname)}
    eb = {"head": [], "mid": [], "tail": []}
    for e in ent_vals:
        eb["head" if e < 100 else ("mid" if e < 1000 else "tail")].append(e)
    for b, lst in eb.items():
        if not lst:
            continue
        pick = [lst[i] for i in rng.choice(len(lst), min(n_per_band, len(lst)), replace=False)]
        a5 = 0.0; below = 0.0; tw = 0.0; ntw = 0
        for e in pick:
            ids, net = eng.probe(("%sE%d" % (P, e),), vid)
            tv = [name2col[v] for v in ent_vals[e] if v in name2col]
            top = [j for j in np.argsort(-net, kind="stable")[:5].tolist() if net[j] > 0]
            a5 += len(set(top) & set(tv)) / max(1, min(5, len(tv)))
            # share of true values whose support is BELOW that of a value never met with the entity (support 0)
            below += float(np.mean(net[tv] < 0)) if tv else 0.0
            tw += float(net[tv].sum()); ntw += len(tv)
        out[b].update(assoc5=a5 / len(pick), true_below_unrelated=below / len(pick), mean_true_support=tw / max(1, ntw))
    return out


def degree_stats(eng):
    d0, d1 = eng.degrees(); d = d0 + d1
    nz = d[d > 0]
    if len(nz) == 0:
        return {}
    srt = np.sort(nz)[::-1]; tot = int(srt.sum())
    top = np.argsort(-d)[:8]
    inv = {u: k for k, u in eng.sym.items()}
    # tail exponent (Hill estimator on the top 5% of degrees)
    kk = max(10, len(srt) // 20); hill = float(kk / np.sum(np.log(srt[:kk] / srt[kk]))) if srt[kk] > 0 else float("nan")
    return dict(ties=tot, units_with_ties=int(len(nz)), mean=float(nz.mean()), p50=float(np.median(nz)),
                p90=float(np.quantile(nz, 0.9)), p99=float(np.quantile(nz, 0.99)), max=int(srt[0]),
                top8=[(inv.get(int(u), "comp#%d" % u), int(d[u])) for u in top],
                top10_share=float(srt[:10].sum() / tot), top1pct_share=float(srt[:max(1, len(srt) // 100)].sum() / tot),
                hill_alpha=hill, max_over_n=float(srt[0] / eng.n))


TIE_BYTES_DESIGN = 4 + 4 + 1 + 1 + 6 * 8 + 6 * 8 + 4 + 4     # §10 COO row: src,dst,form,lag,w+[6],w-[6],use,last = 114
LOG_BYTES = 16                                                # retraction log: (tie id, float64 increment)


def run(name, N, cfg_kw, world_kw=None, checkpoints=None, seed=1, out=None, second=None, ncap=None, quiet=False):
    cfg = Cfg(**cfg_kw); eng = Engine(cfg, ncap=ncap or (N + 20000))
    world = World(seed=seed, **(world_kw or {}))
    cps = checkpoints or [1000, 3000, 10000, 30000, 100000, 300000]
    cps = [c for c in cps if c <= N]
    if N not in cps:
        cps.append(N)
    rng = np.random.default_rng(123)
    rows = []; last = dict(eng.C); t0 = time.time(); tl = t0; i = 0
    streams = [world.stream(N)]
    w2 = None
    if second:   # a second, disjoint domain starting at moment second["at"] (T6-like), interleaved 50/50
        w2 = World(seed=seed + 77, prefix="b", **second.get("world", world_kw or {}))
    g1 = world.stream(N); g2 = w2.stream(N) if w2 else None
    r2 = np.random.default_rng(5)
    for i in range(1, N + 1):
        if w2 is not None and i > second["at"] and r2.random() < 0.5:
            eng.moment(next(g2))
        else:
            eng.moment(next(g1))
        if i in cps:
            now = time.time(); C = eng.C; dm = C["moments"] - last["moments"]
            def per(k):
                return (C[k] - last[k]) / dm
            ties = eng.nties[0] + eng.nties[1]
            row = dict(run=name, moment=i, units=eng.n, atoms=len(eng.sym), composed=len(eng.tuples),
                       ties=ties, ties_lag0=eng.nties[0], ties_lag1=eng.nties[1],
                       tie_MB_design=ties * TIE_BYTES_DESIGN / 1e6,
                       log_entries=C["upd"], log_MB=C["upd"] * LOG_BYTES / 1e6,
                       rss_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                       ms_per_moment=1000 * (now - tl) / dm, reads_per_moment=per("reads"), upd_per_moment=per("upd"),
                       new_ties_per_moment=per("created0") + per("created1"),
                       share_new_ties_nonevent=(C["created_nonevent"] - last["created_nonevent"]) /
                       max(1, (C["created0"] - last["created0"]) + (C["created1"] - last["created1"])),
                       settle_its=per("its"), nonconv=per("nonconv"), period2=per("period2"), setflip=per("setflip"),
                       focus_size=per("fsize"), ext_survive=(C["ext_in"] - last["ext_in"]) / max(1, C["ext_n"] - last["ext_n"]),
                       comp_survive=(C["comp_in"] - last["comp_in"]) / max(1, C["comp_n"] - last["comp_n"]),
                       intero_in_focus=per("intero_in"), recalled_in_focus=per("recalled_in"),
                       saturated_per_moment=per("sat"), maxraw=C["maxraw"], boundary_ties=per("boundary_ties"),
                       relief_pos_share=per("relief_pos"), relief_neg_share=per("relief_neg"), dD_per_moment=per("dD"),
                       recruited=C["recruited"], recognised=C["recognised"], pruned=C["pruned"], evicted=C["evict"],
                       wall_s=now - t0)
            row["degree"] = degree_stats(eng)
            row["recall"] = recall(eng, world, rng)
            if w2 is not None and i > second["at"]:
                row["recall_second"] = recall(eng, w2, rng)
            rows.append(row); last = dict(C); tl = time.time()
            if not quiet:
                dg = row["degree"]; rc = row["recall"]
                print(f"[{name}] t={i:>6} units={eng.n:>6} ties={ties:>8} ({row['tie_MB_design']:.0f} MB design) "
                      f"log={C['upd']:.3g} ({row['log_MB']:.0f} MB) ms/mom={row['ms_per_moment']:.2f} "
                      f"reads/mom={row['reads_per_moment']:.0f} upd/mom={row['upd_per_moment']:.0f} "
                      f"new/mom={row['new_ties_per_moment']:.1f} deg max={dg.get('max')} p50={dg.get('p50')} "
                      f"ext_surv={row['ext_survive']:.2f} recalled={row['recalled_in_focus']:.2f} "
                      f"intero={row['intero_in_focus']:.2f} its={row['settle_its']:.2f} nonconv={row['nonconv']:.3f} "
                      f"p2={row['period2']:.3f} r1(h/m/t/x)="
                      + "/".join("%.2f" % rc[b]["r1"] if b in rc else "-" for b in ("head", "mid", "tail", "exception"))
                      + " inF=" + "/".join("%.2f" % rc[b]["infocus"] if b in rc else "-" for b in ("head", "mid", "tail", "exception"))
                      + " setflip=%.3f" % row["setflip"]
                      + " assoc5=" + "/".join("%.2f" % rc[b]["assoc5"] if b in rc else "-" for b in ("head", "mid", "tail"))
                      + " belowUnrel=" + "/".join("%.2f" % rc[b]["true_below_unrelated"] if b in rc else "-" for b in ("head", "mid", "tail"))
                      + (" second r1=" + "/".join("%.2f" % row["recall_second"][b]["r1"] if b in row["recall_second"] else "-"
                                                    for b in ("head", "mid", "tail")) if "recall_second" in row else ""),
                      flush=True)
    if out:
        with open(out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return eng, world, rows


PRESETS = {
    # the design as written
    "literal": dict(),
    # the design with every proposed fix
    "patched": dict(ref="base", neg="lazy", clamp=True, create="events", intero="change", drive=False,
                    settle="monotone"),
    "patched_kwta": dict(ref="base", neg="lazy", clamp=True, create="events", intero="change", drive=False),
}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="literal"); ap.add_argument("--N", type=int, default=10000)
    ap.add_argument("--name", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    kw = dict(PRESETS[a.preset])
    for kv in a.set:
        k, v = kv.split("=")
        try:
            v = json.loads(v)
        except Exception:
            pass
        kw[k] = v
    run(a.name or a.preset, a.N, kw, out=a.out)
