# -*- coding: utf-8 -*-
"""Fact streams, the probe battery and the cached evaluator of the output-side memory (PROTOCOL §4-§5).

Partitions per world (all from the probe partition; c0 is CEV-11's calibration pool and is never written):
``U`` = 60 never-touched probe entities (seed w+14), the stream pool = the other probe entities.

Stream L (seed w+12): inserts over relations A/B/C, one fact per concept (entity, relation), in a fixed order;
checkpoints are prefixes of that order. Fact ids are zero-padded, so the id order is the stream order.

Battery (CEV-11 metric names; scoring = CEV-11's batched exact multi-token check):

* per live fact: ``lived`` (WE), ``para`` (two other formal cases), ``fused`` (one FUSED informal form: G_F),
  ``leadin`` (the lived case after a context prefix);
* negatives, scored as "the base model's own answer is no longer produced":
  ``U`` (never-touched entities: Leak_U), ``REL`` (other relations of written entities: Leak_C), ``LTOK``
  (unwritten concepts whose entity shares the last root token: Leak_RN), ``PX`` (prefix-extension partner),
  ``OBJ`` (unwritten concepts with the same old object), and OBJ1 = the U probes whose base object starts with
  the first token of a stored new object.

The cached evaluator is exact for the output-side memory: the memory adds only before ``output_norm``, so one
base forward per probe (final residual + address codes) is enough to evaluate any store and any read rule with
``ka.episodic.compose``, the same function the hooked memory runs.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from . import KA_VERSION  # noqa: F401
from .codes import Address, capture, null_slot_read
from .episodic import Episode, EpisodeLedger, Pack, ReadRule, Readout, analytic_value, compose, M_STAR, C_CAP
from cev11.morph_tr import CASES, INFLECTED
from cev11.world import RELATIONS, frame_prefix
from tokenizer_management.byte_bpe import BOS_ID

POS_KINDS = ("lived", "para", "fused", "leadin")
NEG_KINDS = ("U", "REL", "LTOK", "PX", "OBJ")
STRUCTURED = ("Leak_C", "Leak_RN", "PX", "OBJ", "OBJ1")
N_U = 60
N_STREAM = 400
CHECKPOINTS = (25, 50, 100, 200, 400)
FT_CHECKPOINTS = (25, 50, 100)
N_PRIMARY = 100
MAX_REL, MAX_LTOK, MAX_OBJ = 300, 300, 150
FRONTIER_QS = (0.6, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999)
MATCHED_LEAK_U = 0.02
ECE_BINS = 10


@dataclass(frozen=True)
class Fact:
    idx: int
    fact_id: str
    name: str
    rel: str
    lived: str
    new: str
    old: str

    @property
    def prefix(self) -> str:
        return frame_prefix(self.name, self.rel, self.lived, "formal", "none")


def split_probe(world, seed: int, n_u: int = N_U) -> Tuple[List[str], List[str]]:
    """(U, pool): ``n_u`` never-touched probe entities and the stream pool (seed w+14)."""
    probe = sorted(world.members("probe"))
    random.Random(seed).shuffle(probe)
    if len(probe) <= n_u:
        raise ValueError("the probe partition is too small for the reserved U set")
    return sorted(probe[:n_u]), sorted(probe[n_u:])


def make_stream(world, pool: Sequence[str], seed: int, n_max: int = N_STREAM) -> List[Fact]:
    """Stream L: the first ``n_max`` concepts of a seeded shuffle of pool x relations (seed w+12)."""
    rng = random.Random(seed)
    concepts = [(n, r) for n in sorted(pool) for r in RELATIONS]
    rng.shuffle(concepts)
    facts = []
    for i, (n, r) in enumerate(concepts[:n_max]):
        old = world.facts[(n, r)]
        lived = rng.choice(INFLECTED)
        new = rng.choice([o for o in world.objects[r] if o != old])
        facts.append(Fact(i, f"f{i:04d}", n, r, lived, new, old))
    return facts


def kl_relation(rel: str) -> str:
    """Relation of the KL anchor prompt of an FT write: the next relation of the same entity (A->B->C->A)."""
    return RELATIONS[(RELATIONS.index(rel) + 1) % len(RELATIONS)]


# ---------------------------------------------------------------------------------------------------- battery
def base_answers(base, prefixes: Sequence[str], guesses: Sequence[str]) -> List[Optional[str]]:
    """The base model's own object answer per prefix (call in the BASE state only).

    The world fact is tried first with the batched exact check; where the base does not produce it, true greedy
    decoding gives its answer. ``None`` where the base answer is not a '.'-terminated object the batched check
    reproduces (such probes are dropped by the caller and counted).
    """
    from cev11.decode import exact_check_batch, greedy_answer
    ok = exact_check_batch(base.model, base.tok, list(prefixes), list(guesses)) if prefixes else []
    out: List[Optional[str]] = []
    redo = []
    for i, (p, g, k) in enumerate(zip(prefixes, guesses, ok)):
        if k:
            out.append(g)
        else:
            a = greedy_answer(base.model, base.tok, p)
            out.append(a if a else None)
            if a:
                redo.append(i)
    if redo:
        again = exact_check_batch(base.model, base.tok, [prefixes[i] for i in redo], [out[i] for i in redo])
        for i, k in zip(redo, again):
            if not k:
                out[i] = None
    return out


def build_battery(base, facts: Sequence[Fact], U_ent: Sequence[str], pool: Sequence[str], seed: int) -> dict:
    """Probe list for the whole stream (base state). Returns ``{"probes": [...], "dropped": {...}}``.

    Each probe: ``{"kind", "ci", "name", "rel", "prefix", "gold"}``; ``ci`` is the index of the fact the probe
    belongs to (-1 for U). A probe counts at checkpoint N iff ``ci < N`` (U probes always count).
    """
    from cev11.tokenizers import classify_informal, last_root_token
    world, tok = base.world, base.tok
    rng = random.Random(seed)
    stream_set = {(f.name, f.rel) for f in facts}
    lrt = {n: last_root_token(tok, n) for n in list(pool) + list(U_ent)}
    partner = {}
    for a_, b_ in world.prefix_pairs:
        partner[a_] = b_
        partner[b_] = a_
    by_old: Dict[Tuple[str, str], List[str]] = {}
    for n in sorted(pool):
        for r in RELATIONS:
            if (n, r) not in stream_set:
                by_old.setdefault((r, world.facts[(n, r)]), []).append(n)
    probes: List[dict] = []
    pending: List[dict] = []

    def pos(f: Fact, kind: str, case: str, orth: str = "formal", ctx: str = "none"):
        probes.append({"kind": kind, "ci": f.idx, "name": f.name, "rel": f.rel,
                       "prefix": frame_prefix(f.name, f.rel, case, orth, ctx), "gold": f.new})

    def neg(kind: str, ci: int, name: str, rel: str, case: str, orth: str = "formal"):
        pending.append({"kind": kind, "ci": ci, "name": name, "rel": rel,
                        "prefix": frame_prefix(name, rel, case, orth, "none"), "gold": world.facts[(name, rel)]})

    n_rel = n_ltok = n_obj = 0
    used_ltok, used_obj = set(), set()
    for f in facts:
        pos(f, "lived", f.lived)
        for c in rng.sample([c for c in CASES if c != f.lived], 2):
            pos(f, "para", c)
        fz = [c for c in INFLECTED if classify_informal(tok, f.name, c, rel=f.rel) == "FUSED"]
        if fz:
            pos(f, "fused", rng.choice(fz), "informal")
        pos(f, "leadin", f.lived, "formal", rng.choice(["P1", "P2"]))
        for r2 in RELATIONS:
            if (f.name, r2) not in stream_set and n_rel < MAX_REL:
                neg("REL", f.idx, f.name, r2, rng.choice(INFLECTED))
                n_rel += 1
        cands = [m for m in sorted(pool) if m != f.name and lrt[m] == lrt[f.name] and (m, f.rel) not in stream_set
                 and (m, f.rel) not in used_ltok]
        rng.shuffle(cands)
        for m in cands[:2]:
            if n_ltok < MAX_LTOK:
                neg("LTOK", f.idx, m, f.rel, rng.choice(INFLECTED))
                used_ltok.add((m, f.rel))
                n_ltok += 1
        p_ = partner.get(f.name)
        if p_ is not None and (p_, f.rel) not in stream_set and p_ in lrt:
            neg("PX", f.idx, p_, f.rel, rng.choice(INFLECTED))
        same = [m for m in by_old.get((f.rel, f.old), []) if m != f.name and (m, f.rel) not in used_obj]
        if same and n_obj < MAX_OBJ:
            m = rng.choice(same)
            neg("OBJ", f.idx, m, f.rel, rng.choice(INFLECTED))
            used_obj.add((m, f.rel))
            n_obj += 1
    for n in sorted(U_ent):
        for r in RELATIONS:
            neg("U", -1, n, r, rng.choice(INFLECTED))
            neg("U", -1, n, r, rng.choice(INFLECTED), "informal")
    ans = base_answers(base, [d["prefix"] for d in pending], [d["gold"] for d in pending])
    dropped: Dict[str, int] = {}
    for d, a in zip(pending, ans):
        if a is None:
            dropped[d["kind"]] = dropped.get(d["kind"], 0) + 1
            continue
        d["gold"] = a
        probes.append(d)
    return {"probes": probes, "dropped": dropped}


def probe_ids(tok, probe: dict) -> Tuple[List[int], List[int]]:
    return [BOS_ID] + list(tok.encode(probe["prefix"])), list(tok.encode(" " + probe["gold"] + "."))


# ---------------------------------------------------------------------------------------------------- cached evaluator
class CachedBattery:
    """One base forward per probe; evaluates any (store, address, read rule) without touching the model again."""

    def __init__(self, base, probes: Sequence[dict], addresses: Sequence[Address], kmax: int):
        self.base, self.tok = base, base.tok
        self.readout = Readout(base.model)
        self.addresses = {a.name: a for a in addresses}
        self.probes = list(probes)
        specs = sorted({(a.kind, a.layer) for a in addresses}) + [("x", 0)]
        seqs, self.gold, self.w0, self.pidx = [], [], [], []
        for p in self.probes:
            pre, tgt = probe_ids(self.tok, p)
            seqs.append(pre + tgt[:-1])
            self.gold.append(tgt)
            P = len(pre) - 1
            w0 = max(0, P - max(0, kmax - 1))
            self.w0.append(w0)
            self.pidx.append(P - w0)
        n = len(seqs)
        self.kmax_gold = max(len(g) for g in self.gold)
        self.Tw = max(len(s) - w for s, w in zip(seqs, self.w0))
        D = base.model.output_layer.weight.shape[1]
        self.tokens = torch.zeros(n, self.Tw, dtype=torch.long)
        self.X = torch.zeros(n, self.Tw, D, dtype=torch.float64)
        self.codes: Dict[str, torch.Tensor] = {}
        self.valid = torch.zeros(n, self.Tw, dtype=torch.bool)       # continuation positions
        self.GOLD = torch.full((n, self.Tw), -1, dtype=torch.long)
        recs = capture(base.model, seqs, specs, dtype=torch.float32)
        for a in addresses:
            nm = f"{a.kind}{a.layer}"
            if nm not in self.codes:
                self.codes[nm] = torch.zeros(n, self.Tw, recs[0][nm].shape[-1], dtype=torch.float32)
        for i, (s, rec) in enumerate(zip(seqs, recs)):
            w = self.w0[i]
            L = len(s) - w
            self.tokens[i, :L] = torch.tensor(s[w:], dtype=torch.long)
            self.X[i, :L] = rec["x"][w:].double()
            for nm in self.codes:
                self.codes[nm][i, :L] = rec[nm][w:]
            k = len(self.gold[i])
            self.valid[i, self.pidx[i]:self.pidx[i] + k] = True
            self.GOLD[i, self.pidx[i]:self.pidx[i] + k] = torch.tensor(self.gold[i], dtype=torch.long)
        # one base readout per window position: argmax, top-2 margin and log p_base(next token)
        self.base_arg = torch.zeros(n, self.Tw, dtype=torch.long)
        self.base_margin = torch.zeros(n, self.Tw, dtype=torch.float64)
        self.logp_next = torch.zeros(n, self.Tw, dtype=torch.float64)
        for s0 in range(0, n, 64):
            Z = self.readout(self.X[s0:s0 + 64])
            top = torch.topk(Z, 2, dim=-1).values
            self.base_arg[s0:s0 + 64] = Z.argmax(-1)
            self.base_margin[s0:s0 + 64] = top[..., 0] - top[..., 1]
            if self.Tw > 1:
                lsm = torch.log_softmax(Z[:, :-1], -1)
                self.logp_next[s0:s0 + 64, :-1] = lsm.gather(-1, self.tokens[s0:s0 + 64, 1:].unsqueeze(-1)).squeeze(-1)
        # Rigorous skip rule for the edited readout: with n(x) = x / r(x), r(x) = sqrt(mean x^2 + eps),
        # ||n(x + o) - n(x)|| <= 2 ||o|| / (r(x) - ||o|| / sqrt(D)), every logit moves by at most
        # max_v ||E_v * g|| times that (tanh is 1-Lipschitz), so the argmax cannot change while twice that bound
        # stays below the base top-2 margin.
        ro = self.readout
        self.lip = float((ro.E * ro.g).norm(dim=-1).max())
        self.rms = torch.sqrt((self.X * self.X).mean(-1) + ro.eps)
        self.base_ok = ((self.base_arg == self.GOLD) | ~self.valid).all(1)
        self.kind = [p["kind"] for p in self.probes]
        self.ci = torch.tensor([p["ci"] for p in self.probes], dtype=torch.long)
        self.first_gold = torch.tensor([g[0] for g in self.gold], dtype=torch.long)
        self._U_cache: Dict[str, torch.Tensor] = {}
        self._value_cache: Dict[tuple, Tuple[torch.Tensor, int]] = {}
        self._lived = {p["ci"]: i for i, p in enumerate(self.probes) if p["kind"] == "lived"}

    def lived_row(self, fact_idx: int) -> int:
        return self._lived[fact_idx]

    def embedded(self, address: Address) -> torch.Tensor:
        if address.name not in self._U_cache:
            self._U_cache[address.name] = address.embed(self.codes[f"{address.kind}{address.layer}"])
        return self._U_cache[address.name]

    def episodes(self, facts: Sequence, address: Address, m_star: float = M_STAR,
                 c_cap: float = C_CAP) -> List[Episode]:
        """Episodes of the facts from the cached base forward of their lived probes (same values as the hooked
        ``EpisodicCortex.insert`` up to batch-composition rounding)."""
        out = []
        nm = f"{address.kind}{address.layer}"
        for f in facts:
            i = self.lived_row(f.idx)
            p = self.pidx[i]
            key = (i, float(m_star), float(c_cap))
            if key not in self._value_cache:        # values depend on the base model only, not on the address
                vals, caps = [], 0
                for k, y in enumerate(self.gold[i]):
                    r, info = analytic_value(self.readout, self.X[i, p + k], int(y), m_star, c_cap)
                    vals.append(r)
                    caps += int(info["cap_hit"])
                self._value_cache[key] = (torch.stack(vals), caps)
            values, caps = self._value_cache[key]
            pre, _ = probe_ids(self.tok, self.probes[i])
            out.append(Episode(f.fact_id, self.codes[nm][i, p].double().clone(), tuple(self.gold[i]),
                               values, tuple(pre), caps))
        return out

    @torch.no_grad()
    def evaluate(self, pack: Pack, address: Address, rule: ReadRule, chunk: int = 128) -> dict:
        """Per-probe outcome under the store: ``ok`` (exact production of the gold object), ``mass`` (sum_j rho_j
        at PRED), ``smax`` (largest calibrated score at PRED), ``rho`` rows at PRED (``[n, N]``)."""
        n = len(self.probes)
        U = self.embedded(address)
        ok = torch.zeros(n, dtype=torch.bool)
        mass = torch.zeros(n, dtype=torch.float64)
        smax = torch.zeros(n, dtype=torch.float64)
        rho_pred = torch.zeros(n, pack.n, dtype=torch.float64)
        for s0 in range(0, n, chunk):
            sl = slice(s0, min(n, s0 + chunk))
            S = address.score(U[sl], pack.KU)
            O, rho, _ = compose(self.tokens[sl], S, self.X[sl], pack, rule, self.readout, self.logp_next[sl])
            rows = torch.arange(sl.stop - sl.start)
            pi = torch.tensor(self.pidx[sl], dtype=torch.long)
            rho_pred[sl] = rho[rows, pi]
            mass[sl] = rho[rows, pi].sum(-1)
            smax[sl] = S[rows, pi].max(-1).values
            X, valid = self.X[sl], self.valid[sl]
            am = self.base_arg[sl].clone()
            on = O.norm(dim=-1)
            slack = self.rms[sl] - on / math.sqrt(X.shape[-1])
            bound = torch.where(slack > 0, 2.0 * self.lip * 2.0 * on / slack.clamp_min(1e-300),
                                torch.full_like(on, float("inf")))
            need = (bound >= self.base_margin[sl]) & valid       # elsewhere the argmax provably stays the base one
            idx = need.nonzero()
            if idx.numel():
                am[idx[:, 0], idx[:, 1]] = self.readout(X[idx[:, 0], idx[:, 1]] + O[idx[:, 0], idx[:, 1]]).argmax(-1)
            ok[sl] = ((am == self.GOLD[sl]) | ~valid).all(1)
        return {"ok": ok, "mass": mass, "smax": smax, "rho": rho_pred}


# ---------------------------------------------------------------------------------------------------- metrics
def auc(score: Sequence[float], label: Sequence[bool]) -> Optional[float]:
    """Area under the ROC curve (ties get the average rank); None when one class is empty."""
    pos = [s for s, l in zip(score, label) if l]
    neg = [s for s, l in zip(score, label) if not l]
    if not pos or not neg:
        return None
    allv = torch.tensor(pos + neg, dtype=torch.float64)
    order = allv.argsort()
    ranks = torch.empty_like(allv)
    ranks[order] = torch.arange(1, len(allv) + 1, dtype=torch.float64)
    sv = allv[order]
    i = 0
    while i < len(sv):                      # average ranks over ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return (float(ranks[:len(pos)].sum()) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ece(p: torch.Tensor, y: torch.Tensor, bins: int = ECE_BINS) -> float:
    """Expected calibration error with equal-width bins."""
    b = torch.clamp((p * bins).long(), max=bins - 1)
    out = 0.0
    for k in range(bins):
        mk = b == k
        if mk.any():
            out += float(mk.double().mean()) * abs(float(p[mk].mean()) - float(y[mk].double().mean()))
    return out


def _share(v: torch.Tensor) -> Optional[float]:
    return float(v.double().mean()) if v.numel() else None


def battery_metrics(kind: Sequence[str], ci: torch.Tensor, first_gold: torch.Tensor, ok: torch.Tensor,
                    n_live: int, new_first: Sequence[int], usable: Optional[torch.Tensor] = None) -> dict:
    """World-level battery metrics at a checkpoint with the first ``n_live`` facts live.

    ``ok[i]``: probe i produces its gold object exactly (the new object for the fact's own probes, the base
    answer for negatives). Leak = 1 - ok over negatives. ``usable`` masks probes the scorer can use.
    """
    kind_t = list(kind)
    use = torch.ones(len(kind_t), dtype=torch.bool) if usable is None else usable.clone()
    live = (ci < n_live)
    out: dict = {"N": int(n_live)}
    for k in POS_KINDS:
        m = torch.tensor([x == k for x in kind_t]) & live & (ci >= 0) & use
        out[k] = _share(ok[m])
        out[f"n_{k}"] = int(m.sum())
    name = {"U": "Leak_U", "REL": "Leak_C", "LTOK": "Leak_RN", "PX": "PX", "OBJ": "OBJ"}
    for k in NEG_KINDS:
        m = torch.tensor([x == k for x in kind_t]) & use
        if k != "U":
            m = m & live & (ci >= 0)
        out[name[k]] = _share(~ok[m])
        out[f"n_{name[k]}"] = int(m.sum())
    nf = torch.tensor(sorted(set(int(t) for t in new_first)), dtype=torch.long)
    m = torch.tensor([x == "U" for x in kind_t]) & use & torch.isin(first_gold, nf)
    out["OBJ1"] = _share(~ok[m])
    out["n_OBJ1"] = int(m.sum())
    parts = [out[k] for k in STRUCTURED if out.get(k) is not None]
    out["structured"] = sum(parts) / len(parts) if parts else None
    m = torch.tensor([x == "fused" for x in kind_t]) & (ci < min(25, n_live)) & (ci >= 0) & use
    out["fused_first25"] = _share(ok[m])
    return out


def interp_at(points: Sequence[dict], xkey: str, ykey: str, target: float) -> Optional[float]:
    """Upper envelope of ``ykey`` at ``xkey = target`` by linear interpolation between neighbouring points;
    the best value at or below the target when no pair brackets it; None when every point lies above it."""
    pts = sorted((p[xkey], p[ykey]) for p in points if p.get(xkey) is not None and p.get(ykey) is not None)
    if not pts or target < pts[0][0]:
        return None
    best = None
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        if x1 <= target <= x2:
            yv = y1 if x2 == x1 else y1 + (y2 - y1) * (target - x1) / (x2 - x1)
            best = yv if best is None else max(best, yv)
    if best is None:
        best = max(y for x, y in pts if x <= target)
    return best


# ---------------------------------------------------------------------------------------------------- laws
def normal_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def markov_bound(p_target: float = 0.5) -> float:
    """E[# episodes with s > theta_N per unrelated query] <= N e^{-theta_N} = P_t / (1 - P_t), for every N.
    At P_t = 0.5 this is 1: N-invariant but vacuous at the Leak_U <= 1% scale."""
    return p_target / (1.0 - p_target)


def false_fire_law(n: int, m: float, theta: float) -> Optional[float]:
    """Calibrated Gaussian law: s | H0 ~ N(-m, 2m); expected false firings = N * sf((theta + m) / sqrt(2m))."""
    if not m > 0:
        return None
    return n * normal_sf((theta + m) / math.sqrt(2.0 * m))


def generalisation_law(m_f: float, theta: float) -> Optional[float]:
    """G_f(N) ~ sf((theta_N - m_f) / sqrt(2 m_f)) for a form class with target-score mean m_f."""
    if not m_f > 0:
        return None
    return normal_sf((theta - m_f) / math.sqrt(2.0 * m_f))


__all__ = ["POS_KINDS", "NEG_KINDS", "STRUCTURED", "N_U", "N_STREAM", "CHECKPOINTS", "FT_CHECKPOINTS", "N_PRIMARY",
           "FRONTIER_QS", "MATCHED_LEAK_U", "Fact", "split_probe", "make_stream", "kl_relation", "base_answers",
           "build_battery", "probe_ids", "CachedBattery", "auc", "ece", "battery_metrics", "interp_at", "normal_sf",
           "markov_bound", "false_fire_law", "generalisation_law", "null_slot_read", "EpisodeLedger"]
