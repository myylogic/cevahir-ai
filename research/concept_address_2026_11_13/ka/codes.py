# -*- coding: utf-8 -*-
"""Address codes of the frozen decoder and the closed-form scores built on them (PROTOCOL §2).

Residual decomposition (pre-norm block k, dropout 0, eval mode):

    g_k = Attn_k(norm1(h_k)),   f_k = FFN_k(norm2(h_k + g_k)),   h_{k+1} = h_k + g_k + f_k,   h_0 = e(x)

Codes read at block ``l`` and position ``t`` (names used by :class:`Tap`):

    ``pre``   a_pre  = h_l + g_l                 input of ``layers[l].norm2`` (FFN input before the norm)
    ``res``   norm2(a_pre)                       output of ``layers[l].norm2`` (the FFN input)
    ``key``   native fc2 key                     input of ``layers[l].ffn.fc2`` (SwiGLU hidden; ROME's key space)
    ``att``   c = sum_{k<=l} g_k                 the attention-written stream
    ``post``  a_post = h_{l+1} = a_pre + f_l     output of ``layers[l]``
    ``x``     final residual                     input of ``output_norm`` (layer argument ignored)

A write added to f_l's output can be gated only by a code computed before f_l (``pre``, ``res``, ``key``, ``att``),
never by ``post``.

Nothing here modifies the model: hooks only read, and they are removed when the tap closes.

Closed-form scores (no training, float64):

* whitening + length normalisation  z = n(C^{-1/2}(u - mu)), ridge 1e-3 tr(C)/D;
* two-covariance PLDA (Ioffe 2006; Prince & Elder 2007) with shrunk within-class covariance;
* logistic calibration of the PLDA log-likelihood ratio on cross-fitted c0 trials, shifted by the logit of the
  trial target rate, so that the calibrated score is a log likelihood ratio;
* the null-slot read  rho_j = e^{s_j} / (e^{theta_N} + sum_l e^{s_l}),  theta_N = log N - logit P_t.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from . import KA_VERSION  # noqa: F401  (import side effect: sys.path for cev11)
from tokenizer_management.byte_bpe import PAD_ID

CODE_KINDS = ("pre", "res", "key", "att", "post", "x")
P_TARGET = 0.5            # prior probability that a query instantiates some stored episode (fixed in advance)
WHITEN_RIDGE = 1e-3
SHRINK_GRID = (0.01, 0.03, 0.1, 0.3)
N_FOLDS = 5


def _first(o):
    return o if isinstance(o, torch.Tensor) else o[0]


# ---------------------------------------------------------------------------------------------------- taps
class Tap:
    """Read-only hooks capturing the requested ``(kind, layer)`` codes of every forward pass.

    ``Tap(model, [("att", 2), ("key", 2), ("x", 0)])`` used as a context manager; after ``model(x)`` the captured
    tensors are in ``tap.out["att2"]``, ``tap.out["key2"]``, ``tap.out["x"]`` (``[B, T, d]``, detached, model dtype).
    """

    def __init__(self, model, specs: Iterable[Tuple[str, int]]):
        self.model = model
        self.specs = sorted({(str(k), int(l)) for k, l in specs})
        for k, l in self.specs:
            if k not in CODE_KINDS:
                raise ValueError(f"unknown code kind {k!r}")
            if k != "x" and not 0 <= l < len(model.layers):
                raise IndexError(f"layer {l} out of range")
        self.out: Dict[str, torch.Tensor] = {}
        self._att: Dict[int, torch.Tensor] = {}
        self._handles: list = []

    @staticmethod
    def name(kind: str, layer: int) -> str:
        return "x" if kind == "x" else f"{kind}{layer}"

    def __enter__(self) -> "Tap":
        m = self.model
        att_layers = sorted(l for k, l in self.specs if k == "att")
        if att_layers:
            want = set(att_layers)
            for k in range(att_layers[-1] + 1):
                def hook(mod, inp, out, k=k):
                    g = _first(out).detach()
                    self._att[k] = g if k == 0 else self._att[k - 1] + g
                    if k in want:
                        self.out[f"att{k}"] = self._att[k]
                self._handles.append(m.layers[k].attn.register_forward_hook(hook))
        for kind, l in self.specs:
            nm = self.name(kind, l)
            if kind == "pre":
                self._handles.append(m.layers[l].norm2.register_forward_pre_hook(
                    lambda mod, inp, nm=nm: self.out.__setitem__(nm, inp[0].detach())))
            elif kind == "res":
                self._handles.append(m.layers[l].norm2.register_forward_hook(
                    lambda mod, inp, out, nm=nm: self.out.__setitem__(nm, out.detach())))
            elif kind == "key":
                self._handles.append(m.layers[l].ffn.fc2.register_forward_hook(
                    lambda mod, inp, out, nm=nm: self.out.__setitem__(nm, inp[0].detach())))
            elif kind == "post":
                self._handles.append(m.layers[l].register_forward_hook(
                    lambda mod, inp, out, nm=nm: self.out.__setitem__(nm, _first(out).detach().clone())))
            elif kind == "x":
                self._handles.append(m.output_norm.register_forward_pre_hook(
                    lambda mod, inp, nm=nm: self.out.__setitem__(nm, inp[0].detach().clone())))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._att = {}


def pad_batch(seqs: Sequence[Sequence[int]]) -> torch.Tensor:
    width = max(len(s) for s in seqs)
    x = torch.full((len(seqs), width), PAD_ID, dtype=torch.long)
    for j, s in enumerate(seqs):
        x[j, :len(s)] = torch.tensor(list(s), dtype=torch.long)
    return x


@torch.no_grad()
def capture(model, seqs: Sequence[Sequence[int]], specs: Iterable[Tuple[str, int]],
            pick: Optional[Sequence[Sequence[int]]] = None, batch: int = 64,
            dtype: torch.dtype = torch.float64) -> List[Dict[str, torch.Tensor]]:
    """Codes of every sequence: one dict ``name -> [T_i, d]`` per sequence (``[len(pick_i), d]`` with ``pick``).

    Sequences are right-padded inside a batch (a causal model: padding cannot affect earlier positions).
    """
    specs = list(specs)
    out: List[Dict[str, torch.Tensor]] = []
    was_training = model.training
    model.eval()
    try:
        with Tap(model, specs) as tap:
            for s0 in range(0, len(seqs), batch):
                chunk = seqs[s0:s0 + batch]
                model(pad_batch(chunk))
                for j, ids in enumerate(chunk):
                    rec = {}
                    for nm, t in tap.out.items():
                        row = t[j, :len(ids)]
                        if pick is not None:
                            row = row[torch.tensor(list(pick[s0 + j]), dtype=torch.long)]
                        rec[nm] = row.to(dtype).clone()
                    out.append(rec)
    finally:
        model.train(was_training)
    return out


# ---------------------------------------------------------------------------------------------------- linear algebra
def unit(Z: torch.Tensor) -> torch.Tensor:
    return Z / Z.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def inv_sqrt(C: torch.Tensor, ridge: float = WHITEN_RIDGE) -> torch.Tensor:
    """(C + ridge * tr(C)/d * I)^{-1/2} (symmetric)."""
    d = C.shape[0]
    C = C + ridge * torch.trace(C) / d * torch.eye(d, dtype=C.dtype)
    e, V = torch.linalg.eigh(C)
    return V @ torch.diag(e.clamp_min(1e-12).rsqrt()) @ V.T


def whiten_fit(X: torch.Tensor, ridge: float = WHITEN_RIDGE) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = X.mean(0)
    Xc = X - mu
    return mu, inv_sqrt(Xc.T @ Xc / X.shape[0], ridge)


def fit_plda(X: torch.Tensor, labels: Sequence, shrink: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-covariance PLDA: returns ``(m, T, psi)`` with ``T' S_w T = I`` and ``T' S_b T = diag(psi)``.

    S_w is shrunk to ``(1 - s) S_w + s tr(S_w)/d I``; S_b is the moment estimate (class-mean scatter minus the
    within-class contribution). Directions with psi <= 1e-6 max(psi) are dropped.
    """
    n, d = X.shape
    mu = X.mean(0)
    Xc = X - mu
    groups: Dict[object, List[int]] = {}
    for i, l in enumerate(labels):
        groups.setdefault(l, []).append(i)
    keys = sorted(groups)
    means = torch.stack([Xc[groups[k]].mean(0) for k in keys])
    Sw = torch.zeros(d, d, dtype=X.dtype)
    nw = 0
    for gi, k in enumerate(keys):
        G = Xc[groups[k]] - means[gi]
        Sw += G.T @ G
        nw += len(groups[k]) - 1
    Sw /= max(nw, 1)
    B = means.T @ means / len(keys) - Sw / (n / len(keys))
    Sws = (1 - shrink) * Sw + shrink * torch.trace(Sw) / d * torch.eye(d, dtype=X.dtype)
    Li = torch.linalg.inv(torch.linalg.cholesky(Sws))
    M = Li @ B @ Li.T
    M = (M + M.T) / 2
    psi, U = torch.linalg.eigh(M)
    keep = psi > 1e-6 * psi.max().clamp_min(1e-300)
    return mu, Li.T @ U[:, keep], psi[keep]


def plda_llr(U1: torch.Tensor, U2: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Log likelihood ratio same-concept vs different-concept for every pair, ``[..., n1, n2]``.

    LLR(q,k) = sum_i [ 1/2 log((psi_i+1)^2/(2psi_i+1)) - psi_i^2 (u_qi^2+u_ki^2) / (2(2psi_i+1)(psi_i+1))
                       + psi_i u_qi u_ki / (2psi_i+1) ]
    """
    a = -0.5 * psi ** 2 / ((2 * psi + 1) * (psi + 1))
    c = psi / (2 * psi + 1)
    k = 0.5 * torch.log((psi + 1) ** 2 / (2 * psi + 1))
    return k.sum() + (U1 ** 2 @ a).unsqueeze(-1) + (U2 ** 2 @ a).unsqueeze(-2) + (U1 * c) @ U2.transpose(-1, -2)


def logistic_fit(s: torch.Tensor, y: torch.Tensor, iters: int = 80) -> Tuple[float, float, float]:
    """Newton fit of ``sigmoid(a s + b)`` to labels ``y`` (step halving); returns ``(a, b, mean log-loss)``."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    a, b = 1.0 / max(float(s.std()), 1e-6), 0.0
    cur = float(bce(a * s + b, y))
    for _ in range(iters):
        p = torch.sigmoid(a * s + b)
        g = torch.stack([((p - y) * s).sum(), (p - y).sum()])
        w = p * (1 - p) + 1e-12
        H = torch.stack([torch.stack([(w * s * s).sum(), (w * s).sum()]), torch.stack([(w * s).sum(), w.sum()])])
        step = torch.linalg.solve(H + 1e-9 * torch.eye(2, dtype=s.dtype), g)
        t, new, a2, b2 = 1.0, cur, a, b
        for _ in range(20):
            a2, b2 = a - t * float(step[0]), b - t * float(step[1])
            new = float(bce(a2 * s + b2, y))
            if new <= cur:
                break
            t /= 2
        if new > cur:
            break
        converged = cur - new < 1e-13
        a, b, cur = a2, b2, new
        if converged:
            break
    return a, b, cur


def lda_fit(X: torch.Tensor, labels: Sequence, k: int, ridge: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fisher LDA projection ``(mu, P [d, k'])`` with ``k' = min(k, classes - 1)`` (supervised ceiling A-SUP)."""
    d = X.shape[1]
    mu = X.mean(0)
    groups: Dict[object, List[int]] = {}
    for i, l in enumerate(labels):
        groups.setdefault(l, []).append(i)
    Sw = torch.zeros(d, d, dtype=X.dtype)
    Sb = torch.zeros(d, d, dtype=X.dtype)
    for key in sorted(groups):
        Zc = X[groups[key]]
        mc = Zc.mean(0)
        Sw += (Zc - mc).T @ (Zc - mc)
        Sb += len(groups[key]) * torch.outer(mc - mu, mc - mu)
    Sw += ridge * torch.trace(Sw) / d * torch.eye(d, dtype=X.dtype)
    Li = torch.linalg.inv(torch.linalg.cholesky(Sw))
    M = Li @ Sb @ Li.T
    ev, V = torch.linalg.eigh((M + M.T) / 2)
    kk = max(1, min(int(k), len(groups) - 1))
    order = torch.argsort(ev, descending=True)[:kk]
    return mu, Li.T @ V[:, order]


# ---------------------------------------------------------------------------------------------------- the read
def theta_n(n: int, p_target: float = P_TARGET) -> float:
    """Bayes-prior threshold of the null slot: ``theta_N = log N - logit(P_t)`` (prior P_t/N per episode)."""
    if n <= 0:
        raise ValueError("theta_n needs a non-empty store")
    if not 0.0 < p_target < 1.0:
        raise ValueError("p_target must be in (0, 1)")
    return math.log(n) - math.log(p_target / (1.0 - p_target))


def null_slot_read(S: torch.Tensor, theta: Optional[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """``rho_j = e^{s_j} / (e^{theta} + sum_l e^{s_l})`` and ``rho_0 = 1 - sum_j rho_j`` over the last axis.

    ``theta=None`` drops the null slot: the read is then plain softmax attention over the store (rho_0 = 0).
    """
    if S.shape[-1] == 0:
        raise ValueError("empty store")
    if theta is None:
        return torch.softmax(S, -1), torch.zeros(S.shape[:-1], dtype=S.dtype)
    null = torch.full(S.shape[:-1] + (1,), float(theta), dtype=S.dtype)
    r = torch.softmax(torch.cat([S, null], -1), -1)
    return r[..., :-1], r[..., -1]


# ---------------------------------------------------------------------------------------------------- addresses
@dataclass
class Address:
    """A derived, closed-form concept code: whitening, length normalisation, PLDA projection and calibration."""
    name: str
    kind: str
    layer: int
    mu: torch.Tensor
    W: torch.Tensor
    m: torch.Tensor
    T: torch.Tensor
    psi: torch.Tensor
    a: float
    c: float
    shrink: float
    logloss: float
    target_rate: float
    stats: dict = field(default_factory=dict)

    def embed(self, codes: torch.Tensor) -> torch.Tensor:
        """Raw codes ``[..., D]`` -> PLDA coordinates ``u = T'(n(C^{-1/2}(code - mu)) - m)``."""
        z = unit((codes.to(torch.float64) - self.mu) @ self.W)
        return (z - self.m) @ self.T

    def score(self, Uq: torch.Tensor, Uk: torch.Tensor) -> torch.Tensor:
        """Calibrated log likelihood ratio ``a * LLR + c`` for every (query, key) pair."""
        return self.a * plda_llr(Uq, Uk, self.psi) + self.c

    def summary(self) -> dict:
        return {"name": self.name, "kind": self.kind, "layer": self.layer, "shrink": self.shrink,
                "a": self.a, "c": self.c, "logloss": self.logloss, "target_rate": self.target_rate,
                "plda_dims": int(self.psi.numel()), **self.stats}


def _folds(entities: Sequence[str], n_folds: int, seed: int) -> List[set]:
    ents = sorted(set(entities))
    random.Random(seed).shuffle(ents)
    return [set(ents[f::n_folds]) for f in range(n_folds)]


def fit_address(codes: torch.Tensor, items: Sequence[dict], name: str, kind: str, layer: int, seed: int,
                shrink_grid: Sequence[float] = SHRINK_GRID, n_folds: int = N_FOLDS) -> Address:
    """Fit one address on c0 codes (write-free): whitening, PLDA shrinkage by cross-fitted log-loss, calibration.

    ``items[i]`` describes code ``i``: ``{"name", "rel", "case", "orth", "ctx"}``. Concept = (name, rel).
    Trials: per test fold, one enrolment code per concept (a formal inflected form, seeded choice) against every
    other code of the fold. The calibration offset is shifted by -logit(trial target rate).
    """
    X = codes.to(torch.float64)
    n = X.shape[0]
    if n != len(items):
        raise ValueError("codes and items differ in length")
    lab = [(it["name"], it["rel"]) for it in items]
    ent = [it["name"] for it in items]
    folds = _folds(ent, n_folds, seed)
    r_enrol = random.Random(seed + 1)
    enrol: Dict[tuple, int] = {}
    for cc in sorted(set(lab)):
        cand = [i for i in range(n) if lab[i] == cc and items[i]["orth"] == "formal" and items[i]["case"] != "nom"
                and items[i].get("ctx", "none") == "none"]
        if not cand:
            cand = [i for i in range(n) if lab[i] == cc]
        enrol[cc] = r_enrol.choice(cand)

    def fold_models(shrink):
        out = []
        for f in folds:
            tr = [i for i in range(n) if ent[i] not in f]
            te = [i for i in range(n) if ent[i] in f]
            if not tr or not te:
                continue
            mu, Wm = whiten_fit(X[tr])
            m2, T2, psi = fit_plda(unit((X[tr] - mu) @ Wm), [lab[i] for i in tr], shrink)
            U = (unit((X[te] - mu) @ Wm) - m2) @ T2
            out.append((te, plda_llr(U, U, psi)))
        return out

    def trials(models):
        sc, yy, meta = [], [], []
        for te, M in models:
            pos = {i: k for k, i in enumerate(te)}
            for cc in sorted({lab[i] for i in te}):
                e_ = pos[enrol[cc]]
                for k, i in enumerate(te):
                    if k != e_:
                        sc.append(float(M[k, e_]))
                        yy.append(1.0 if lab[i] == cc else 0.0)
                        meta.append(i)
        return torch.tensor(sc, dtype=torch.float64), torch.tensor(yy, dtype=torch.float64), meta

    best = None
    for shrink in shrink_grid:
        sc, yy, meta = trials(fold_models(shrink))
        a_, b_, loss = logistic_fit(sc, yy)
        if best is None or loss < best[0]:
            best = (loss, shrink, a_, b_, sc, yy, meta)
    loss, shrink, a_, b_, sc, yy, meta = best
    ptr = float(yy.mean())
    c_ = b_ - math.log(ptr / (1 - ptr))
    cal = a_ * sc + c_
    neg, pos_ = cal[yy == 0], cal[yy == 1]
    stats = {"n_trials": int(sc.numel()),
             "h0_mean": float(neg.mean()), "h0_var": float(neg.var()),
             "h1_mean": float(pos_.mean()), "h1_var": float(pos_.var()),
             "h0_mean_exp_s": float(torch.exp(neg).mean())}
    for form in ("formal", "informal"):
        sel = torch.tensor([items[i]["orth"] == form for i in meta]) & (yy == 1)
        if sel.any():
            stats[f"h1_mean_{form}"] = float(cal[sel].mean())
    mu, Wm = whiten_fit(X)
    m2, T2, psi = fit_plda(unit((X - mu) @ Wm), lab, shrink)
    return Address(name, kind, int(layer), mu, Wm, m2, T2, psi, float(a_), float(c_), float(shrink), float(loss),
                   ptr, stats)


__all__ = ["CODE_KINDS", "P_TARGET", "SHRINK_GRID", "Tap", "pad_batch", "capture", "unit", "inv_sqrt", "whiten_fit",
           "fit_plda", "plda_llr", "logistic_fit", "lda_fit", "theta_n", "null_slot_read", "Address", "fit_address"]
