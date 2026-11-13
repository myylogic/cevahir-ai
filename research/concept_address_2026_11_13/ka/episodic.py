# -*- coding: utf-8 -*-
"""Output-side episodic memory with a calibrated null-slot read (PROTOCOL §4; KA-H2, KA-H3).

* **Address.** The code ``a_t`` of one :class:`ka.codes.Address` (PRED-RES: the residual stream after block l*;
  PRED-ATT: the attention-written stream up to block l*), read at every position.
* **Read.** ``rho_j(q) = e^{s(q,k_j)} / (e^{theta_N} + sum_l e^{s(q,k_l)})`` with calibrated PLDA scores ``s`` and
  ``theta_N = log N - logit P_t`` (:func:`ka.codes.null_slot_read`).
* **Values.** Base-relative and analytic (no optimiser): for each object token, the smallest move of the final
  residual along the margin gradient that lifts the readout margin of that token to ``m*`` (bisection), capped at
  ``c_cap * ||x||``. A value depends only on its own sentence and the base model.
* **Continuation.** Episode-anchored Bayesian update for multi-token objects: after an anchor at t', the value of
  object position k applies only if the k tokens emitted after t' equal the episode's first k object tokens, with
  weight ``rho_j M_j / (rho_0 prod p_base + sum_l rho_l M_l)``.
* **Injection.** ``x_t <- x_t + o_t`` before ``output_norm``. No block's keys or values depend on the memory, so
  writes never require clearing the KV cache.
* **Ledger.** Episodes live outside the module tree, keyed by fact id and summed in id order. insert / correct /
  retract are exact: the model's function depends only on the live set, so "write A, write B, retract A" equals
  "write B" bitwise. With an empty ledger nothing is added and the logits are bit-identical to the base model.

``compose`` is the single implementation of the read + continuation arithmetic. The hooked memory
(:class:`EpisodicCortex`) and the cached offline evaluator (``ka.streams``) both call it.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from . import KA_VERSION
from .codes import Address, Tap, null_slot_read, pad_batch, theta_n, P_TARGET
from tokenizer_management.byte_bpe import PAD_ID

M_STAR = 6.0       # target readout margin of a stored object token
C_CAP = 1.0        # value norm cap, in units of ||x||
BISECT_STEPS = 24
MASS_EPS = 1e-300


# ---------------------------------------------------------------------------------------------------- readout
class Readout:
    """Float64 replica of the model head: ``z(x) = cap * tanh(E (g * x / rms(x)) / cap)``."""

    def __init__(self, model):
        self.g = model.output_norm.scale.detach().double()
        self.eps = float(model.output_norm.eps)
        self.E = model.output_layer.weight.detach().double()
        bias = getattr(model.output_layer, "bias", None)
        self.b = None if bias is None else bias.detach().double()
        self.cap = float(getattr(model, "logit_soft_cap", 0.0) or 0.0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.double()
        n = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)
        z = (n * self.g) @ self.E.T
        if self.b is not None:
            z = z + self.b
        return self.cap * torch.tanh(z / self.cap) if self.cap > 0 else z


def margin(z: torch.Tensor, y: int) -> float:
    zo = z.clone()
    zo[y] = -float("inf")
    return float(z[y] - zo.max())


def analytic_value(readout: Readout, x: torch.Tensor, y: int, m_star: float = M_STAR,
                   c_cap: float = C_CAP) -> Tuple[torch.Tensor, dict]:
    """Base-relative value for one object token: ``r = min(gamma*, c_cap ||x||) * u`` (zero if the margin holds).

    ``u`` is the normalised gradient of ``z_y - z_yhat`` at ``x`` (``yhat`` = strongest competitor), ``gamma*`` the
    smallest step along ``u`` reaching margin ``m*`` (bisection). ``info["cap_hit"]`` is True when the cap is
    reached before ``m*``.
    """
    x = x.detach().double()
    z0 = readout(x)
    m0 = margin(z0, y)
    info = {"margin0": m0, "cap_hit": False, "gamma": 0.0}
    if m0 >= m_star:
        return torch.zeros_like(x), info
    xx = x.clone().requires_grad_(True)
    with torch.enable_grad():
        zz = readout(xx)
        zd = zz.detach().clone()
        zd[y] = -float("inf")
        (zz[y] - zz[int(zd.argmax())]).backward()
    u = xx.grad.detach()
    nu = float(u.norm())
    if not nu > 0:
        info["cap_hit"] = True
        return torch.zeros_like(x), info
    u = u / nu
    lim = c_cap * float(x.norm())
    f = lambda gm: margin(readout(x + gm * u), y)  # noqa: E731
    if f(lim) < m_star:
        info.update(cap_hit=True, gamma=lim)
        return lim * u, info
    lo, hi = 0.0, lim
    for _ in range(BISECT_STEPS):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if f(mid) >= m_star else (mid, hi)
    info["gamma"] = hi
    return hi * u, info


# ---------------------------------------------------------------------------------------------------- ledger
@dataclass(frozen=True)
class Episode:
    fact_id: str
    code: torch.Tensor                 # raw address code at the anchor (float64 copy of the float32 activation)
    obj: Tuple[int, ...]               # object tokens, the final '.' included
    values: torch.Tensor               # [len(obj), D] float64
    prefix: Tuple[int, ...]            # ids of the lived prefix ([BOS] + tokens), kept for correct()
    cap_hits: int = 0


@dataclass
class Pack:
    """Id-sorted arrays of the live set (what the read uses)."""
    ids: List[str]
    KU: torch.Tensor                   # [N, d'] embedded keys
    R: torch.Tensor                    # [N, Kmax, D] values (zero beyond each episode's length)
    Y: torch.Tensor                    # [N, Kmax] object tokens (-1 beyond the length)
    Ylen: torch.Tensor                 # [N]

    @property
    def n(self) -> int:
        return len(self.ids)


class EpisodeLedger:
    """Exact episode set: insert / correct / retract, summed in id order (history independent)."""

    def __init__(self, identity: Optional[dict] = None):
        self.identity = dict(identity or {})
        self._eps: Dict[str, Episode] = {}
        self.version = 0

    def __len__(self) -> int:
        return len(self._eps)

    def __contains__(self, fact_id: str) -> bool:
        return fact_id in self._eps

    def ids(self) -> List[str]:
        return sorted(self._eps)

    def get(self, fact_id: str) -> Episode:
        return self._eps[fact_id]

    def insert(self, ep: Episode) -> None:
        if ep.fact_id in self._eps:
            raise KeyError(f"fact {ep.fact_id!r} is already stored; use correct()")
        if len(ep.obj) == 0 or tuple(ep.values.shape[:1]) != (len(ep.obj),):
            raise ValueError("an episode needs one value per object token")
        if not (torch.isfinite(ep.code).all() and torch.isfinite(ep.values).all()):
            raise ValueError("non-finite episode")
        self._eps[ep.fact_id] = ep
        self.version += 1

    def correct(self, ep: Episode) -> None:
        """Replace the episode of ``ep.fact_id`` (delete + write: the result does not depend on the old one)."""
        if ep.fact_id not in self._eps:
            raise KeyError(f"fact {ep.fact_id!r} is not stored")
        del self._eps[ep.fact_id]
        self.insert(ep)

    def retract(self, fact_id: str) -> None:
        if fact_id not in self._eps:
            raise KeyError(f"fact {fact_id!r} is not stored")
        del self._eps[fact_id]
        self.version += 1

    def erase(self) -> None:
        self._eps = {}
        self.version += 1

    def pack(self, address: Address, ids: Optional[Sequence[str]] = None) -> Pack:
        ids = sorted(self._eps) if ids is None else sorted(ids)
        if not ids:
            raise ValueError("empty ledger")
        eps = [self._eps[i] for i in ids]
        kmax = max(len(e.obj) for e in eps)
        D = eps[0].values.shape[1]
        R = torch.zeros(len(eps), kmax, D, dtype=torch.float64)
        Y = torch.full((len(eps), kmax), -1, dtype=torch.long)
        for j, e in enumerate(eps):
            R[j, :len(e.obj)] = e.values
            Y[j, :len(e.obj)] = torch.tensor(e.obj, dtype=torch.long)
        KU = address.embed(torch.stack([e.code for e in eps]))
        return Pack(ids, KU, R, Y, torch.tensor([len(e.obj) for e in eps], dtype=torch.long))

    def sha256(self) -> str:
        """Content hash of the live set (ids, codes, objects, values), independent of the write history."""
        h = hashlib.sha256()
        h.update(json.dumps(self.identity, sort_keys=True).encode("utf-8"))
        for i in sorted(self._eps):
            e = self._eps[i]
            h.update(i.encode("utf-8"))
            h.update(json.dumps(list(e.obj)).encode("ascii"))
            h.update(e.code.contiguous().numpy().tobytes())
            h.update(e.values.contiguous().numpy().tobytes())
        return h.hexdigest()


# ---------------------------------------------------------------------------------------------------- the read
@dataclass(frozen=True)
class ReadRule:
    """``rule``: theta = theta_N (Bayes prior). ``fixed``: theta given. ``nonull``: no null slot."""
    mode: str = "rule"
    theta: Optional[float] = None
    p_target: float = P_TARGET

    def threshold(self, n: int) -> Optional[float]:
        if self.mode == "rule":
            return theta_n(n, self.p_target)
        if self.mode == "fixed":
            if self.theta is None:
                raise ValueError("a fixed read needs theta")
            return float(self.theta)
        if self.mode == "nonull":
            return None
        raise ValueError(f"unknown read mode {self.mode!r}")


def compose(tokens: torch.Tensor, S: torch.Tensor, X: torch.Tensor, pack: Pack, rule: ReadRule,
            readout: Readout, logp_next: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Injection ``O [B, T, D]`` (float64) for whole sequences, plus ``rho [B, T, N]`` and ``rho0 [B, T]``.

    ``tokens [B, T]`` input ids, ``S [B, T, N]`` calibrated scores of every position against every stored key,
    ``X [B, T, D]`` the base final residual (before the injection). Position t gets

        o_t = sum_j rho_j(q_t) r_j^(0) + sum_{k>=1} sum_j w_j^(k)(t) r_j^(k),
        w_j^(k)(t) = rho_j(q_{t-k}) M_j / (rho_0(q_{t-k}) prod_{u=t-k}^{t-1} p_base(x_{u+1} | u) + sum_l rho_l M_l),

    where M_j = 1[x_{t-k+1..t} = y_j^{1..k}] and episode j has more than k object tokens. p_base is evaluated
    only where some M_j is 1 (elsewhere the term is exactly zero). ``logp_next [B, T]`` may carry precomputed
    ``log p_base(x_{t+1} | t)`` (the cached evaluator); otherwise it is read out on demand.
    """
    B, T, N = S.shape
    rho, rho0 = null_slot_read(S.double(), rule.threshold(N))
    O = rho @ pack.R[:, 0]
    kmax = pack.R.shape[1]
    if kmax > 1 and T > 1:
        lp_cache: Dict[Tuple[int, int], float] = {}

        def lp(b: int, u: int) -> float:
            if logp_next is not None:
                return float(logp_next[b, u])
            key = (b, u)
            if key not in lp_cache:
                z = readout(X[b, u])
                lp_cache[key] = float(torch.log_softmax(z, -1)[int(tokens[b, u + 1])])
            return lp_cache[key]

        match = torch.ones(B, T, N, dtype=torch.bool)       # match[b, t, j] for the current k (anchor at t - k)
        for k in range(1, min(kmax, T)):
            # tokens[b, t] must equal y_j[k - 1]; anchor t - k; previous k - 1 tokens matched at t - 1
            prev = torch.zeros(B, T, N, dtype=torch.bool)
            prev[:, 1:] = match[:, :-1]
            match = prev & (tokens[:, :, None] == pack.Y[None, None, :, k - 1]) & (pack.Ylen > k)[None, None, :]
            match[:, :k] = False
            if not bool(match.any()):
                break
            for b, t in torch.nonzero(match.any(-1)).tolist():
                a = t - k
                wm = rho[b, a] * match[b, t]
                logpb = sum(lp(b, u) for u in range(a, t))
                den = float(rho0[b, a]) * math.exp(logpb) + float(wm.sum())
                O[b, t] += (wm / max(den, MASS_EPS)) @ pack.R[:, k]
    return O, rho, rho0


# ---------------------------------------------------------------------------------------------------- hooked memory
class EpisodicCortex:
    """The memory attached to a model: reads the address below, injects before ``output_norm``.

    ``attach()`` installs read-only hooks plus one pre-hook on ``output_norm``; with an empty ledger that hook
    returns nothing and the logits are bit-identical. ``detach()`` removes every hook. The ledger is not a
    module buffer: ``state_dict`` and the model hash are untouched.

    Cached decoding: with ``use_cache=True`` and a ``cache_position`` that does not start at 0 the call is a
    continuation of the previous one; the last ``Kmax - 1`` positions (tokens, scores, base residual) are kept so
    that the continuation rule sees its anchors. Any other call starts a new sequence.
    """

    def __init__(self, model, address: Address, rule: Optional[ReadRule] = None, m_star: float = M_STAR,
                 c_cap: float = C_CAP, identity: Optional[dict] = None):
        if address.kind not in ("post", "att"):
            raise ValueError("the output-side memory reads a 'post' or 'att' code")
        self.model, self.address = model, address
        self.rule = rule or ReadRule()
        self.m_star, self.c_cap = float(m_star), float(c_cap)
        self.readout = Readout(model)
        self.ledger = EpisodeLedger({"ka": KA_VERSION, "address": address.name, "layer": address.layer,
                                     **(identity or {})})
        self.bypass = False
        self._pack: Optional[Pack] = None
        self._pack_version = -1
        self._tap: Optional[Tap] = None
        self._handles: list = []
        self._tokens: Optional[torch.Tensor] = None
        self._continuing = False
        self._hist: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self.last_rho: Optional[torch.Tensor] = None

    # -------------------------------------------------------------- hooks
    @property
    def attached(self) -> bool:
        return bool(self._handles)

    def attach(self) -> "EpisodicCortex":
        if self._handles:
            return self
        self._tap = Tap(self.model, [(self.address.kind, self.address.layer)])
        self._tap.__enter__()
        self._handles = [self.model.register_forward_pre_hook(self._on_input, with_kwargs=True),
                         self.model.output_norm.register_forward_pre_hook(self._on_final)]
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._tap is not None:
            self._tap.__exit__(None, None, None)
            self._tap = None
        self._hist = None

    def _on_input(self, module, args, kwargs):
        ids = args[0] if args else kwargs.get("x", kwargs.get("input_ids"))
        self._tokens = ids.detach() if isinstance(ids, torch.Tensor) else None
        pos = kwargs.get("cache_position")
        self._continuing = bool(kwargs.get("use_cache")) and pos is not None and int(pos.reshape(-1)[0]) > 0
        if not self._continuing:
            self._hist = None
        return None

    def _live_pack(self) -> Optional[Pack]:
        if len(self.ledger) == 0:
            return None
        if self._pack is None or self._pack_version != self.ledger.version:
            self._pack = self.ledger.pack(self.address)
            self._pack_version = self.ledger.version
            self._hist = None
        return self._pack

    def _on_final(self, module, inputs):
        if self.bypass:
            return None
        pack = self._live_pack()
        if pack is None:
            return None                       # empty store: no add at all (bit-identical logits)
        x = inputs[0]
        code = self._tap.out[Tap.name(self.address.kind, self.address.layer)]
        if self._tokens is None or code.shape[:2] != x.shape[:2] or self._tokens.shape != x.shape[:2]:
            raise RuntimeError("EpisodicCortex: forward pass shape mismatch (call the model with token ids)")
        S = self.address.score(self.address.embed(code), pack.KU)
        tokens, X = self._tokens, x.detach().double()
        keep = pack.R.shape[1] - 1
        n_hist = 0
        if self._continuing and self._hist is not None and self._hist[0].shape[0] == tokens.shape[0]:
            n_hist = self._hist[0].shape[1]
            tokens = torch.cat([self._hist[0], tokens], 1)
            S = torch.cat([self._hist[1], S], 1)
            X = torch.cat([self._hist[2], X], 1)
        O, rho, _ = compose(tokens, S, X, pack, self.rule, self.readout)
        self.last_rho = rho[:, n_hist:]
        self._hist = (tokens[:, -keep:], S[:, -keep:], X[:, -keep:]) if keep > 0 else None
        return (x + O[:, n_hist:].to(x.dtype),)

    # -------------------------------------------------------------- writes
    @torch.no_grad()
    def _observe(self, prefix_ids: Sequence[int], obj_ids: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Base forward of the lived sentence (memory bypassed): anchor code and final residuals at the object
        positions (teacher-forced)."""
        prefix_ids, obj_ids = [int(t) for t in prefix_ids], [int(t) for t in obj_ids]
        if not prefix_ids or not obj_ids:
            raise ValueError("empty prefix or object")
        ids = prefix_ids + obj_ids[:-1]
        was_training, old = self.model.training, self.bypass
        self.model.eval()
        self.bypass = True
        try:
            with Tap(self.model, [(self.address.kind, self.address.layer), ("x", 0)]) as tap:
                self.model(pad_batch([ids]))
                code = tap.out[Tap.name(self.address.kind, self.address.layer)][0, len(prefix_ids) - 1].double().clone()
                X = tap.out["x"][0, len(prefix_ids) - 1:].double().clone()
        finally:
            self.bypass = old
            self.model.train(was_training)
            self._hist = None
        return code, X

    def make_episode(self, fact_id: str, prefix_ids: Sequence[int], obj_ids: Sequence[int]) -> Episode:
        code, X = self._observe(prefix_ids, obj_ids)
        vals, caps = [], 0
        for k, y in enumerate(obj_ids):
            r, info = analytic_value(self.readout, X[k], int(y), self.m_star, self.c_cap)
            vals.append(r)
            caps += int(info["cap_hit"])
        return Episode(str(fact_id), code, tuple(int(t) for t in obj_ids), torch.stack(vals),
                       tuple(int(t) for t in prefix_ids), caps)

    def insert(self, fact_id: str, prefix_ids: Sequence[int], obj_ids: Sequence[int]) -> Episode:
        ep = self.make_episode(fact_id, prefix_ids, obj_ids)
        self.ledger.insert(ep)
        return ep

    def correct(self, fact_id: str, obj_ids: Sequence[int]) -> Episode:
        old = self.ledger.get(fact_id)
        ep = self.make_episode(fact_id, old.prefix, obj_ids)
        self.ledger.correct(ep)
        return ep

    def retract(self, fact_id: str) -> None:
        self.ledger.retract(fact_id)

    def __len__(self) -> int:
        return len(self.ledger)


__all__ = ["M_STAR", "C_CAP", "Readout", "margin", "analytic_value", "Episode", "Pack", "EpisodeLedger", "ReadRule",
           "compose", "EpisodicCortex", "PAD_ID"]
