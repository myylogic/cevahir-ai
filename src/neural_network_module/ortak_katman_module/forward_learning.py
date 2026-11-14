# -*- coding: utf-8 -*-
"""
Forward learning for CevahirNeuralNetwork: learning in the forward pass, optional and off by default.

A ``ForwardLearner`` attaches to a built model through forward hooks. While it holds no write it returns
every ``ffn.fc2`` output untouched, so the logits are bit-identical to the plain model; ``close()`` removes
the hooks. Nothing in ``state_dict`` changes at any time: fast weights live beside the model.

What it does when it is used (one teacher-forced read of externally supplied tokens, eval mode, no autograd):

1. Output error in closed form (``output_error``): with z = U n(x_L), logits c*tanh(z/c) and p = softmax,
   delta_t = J_n(x_L)^T U^T [ s (.) (p_t - e_y) ],  s = 1 - tanh^2(z/c),  equals d(-log p_t[y]) / d x_L.
   With tied embeddings U is the embedding table, which gives the reading "expected next embedding minus
   the embedding that arrived"; the formula itself needs no tying.
2. Error fields for the ``ffn.fc2`` outputs (``error_fields``), the FEC family, all built from quantities the
   forward pass already computed (frozen attention pattern, value path only):
       FEC-TOP  delta, last block only (the exact gradient of the last fc2)
       FEC-0    delta broadcast to every block along the skip path
       FEC-1    delta + one value-path hop through each later block
       FEC-A    prod_{k>l} (I + A_k) delta, all value-path orders (the exact gradient of the graph in which
                queries, keys and FFN inputs are detached; ``stop_gradient_graph`` builds that graph)
   Controls: HEBB-0 / HEBB-TOP (target embedding only, no prediction term) and RAND-0 (random feedback).
3. Rank-one fast-weight writes into fc2 (``ForwardLearner.write``): dF_l = -mu * sum_s m_l[s] (P_l h_l[s])^T,
   normalised over the written blocks, with P_l = C_l^-1 the recursive-least-squares key projector
   (Woodbury update C_l <- C_l + xi K^T K) and a forward-only search for the smallest step mu on a grid that
   installs the read sentence. Writes are made only by ``read`` + ``write`` on tokens the caller supplies; the
   module has no path that learns from tokens the model generated.
4. Familiarity read gate: nu(h) = 1 - h^T C^-1 h / h^T C0^-1 h, the share of prior predictive variance that
   stored keys removed at h. The fast output is added only where nu >= theta; elsewhere the fc2 output is
   returned as it is, so a context in which no gate opens gives the frozen model's logits exactly.
5. Float64 ledger: every write is a list of rank-one terms (value r, gain z, key k) per block,
   F_l = sum r z^T and C_l = C0_l + xi sum k k^T. ``retract`` subtracts a write's terms and downdates C and P.
   This is exact at the ledger level; it is not a replay of the stream without that write.
6. Capacity guards and law: the generic RMS of the fast output relative to the native FFN output is bounded
   (``rms_cap``), and ``capacity_knee`` predicts from key statistics alone how many stored keys it takes
   before the gate opens on unrelated contexts (two-population form, and a mixture form with one self-energy
   per class of stored keys).

Supported configuration: pre-norm, sequential residual, RMSNorm, no MoE, standard softmax attention (no
attention operator), plain floating-point ``ffn.fc2`` layers on the CPU (no quantized layers), eval mode.
Anything else is refused. The fast state is kept on the CPU in float64. The learner's own forward passes leave
the host model's step counter and diagnostics snapshot as they were. What it is not: it is not wired into generation, training or serving, it has no learned parameters, and
whether its writes are useful is a research question (the study folder measures it), not a property of this code.

Prior work (see research/forward_learning_2026_11_13/PROTOCOL.md for the full list): the closed-form output
error applied to MLP weights is FPO; the target-embedding write is Backward Lens "forward pass shifting"; the
path-order family is Highway-BP; stop-gradient through attention scores is TinT; the RLS write is
RLSEdit / AlphaEdit / OWM; fast weights in the MLP projection are In-Place TTT and Fast Weight Layers; the
in-layer least-squares solve is MesaNet. This module claims none of these.
"""
from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

FEC_RULES: Tuple[str, ...] = ("FEC-TOP", "FEC-0", "FEC-1", "FEC-A")
CONTROL_RULES: Tuple[str, ...] = ("HEBB-TOP", "HEBB-0", "RAND-0")
RULES: Tuple[str, ...] = FEC_RULES + CONTROL_RULES
_TOP_RULES = ("FEC-TOP", "HEBB-TOP")
_ROUTED_RULES = ("FEC-1", "FEC-A")
INSTALL_CRITERIA = ("nll", "greedy")
GUARD_POLICIES = ("evict", "refuse", "off")
DEFAULT_STEP_GRID: Tuple[float, ...] = (0.0,) + tuple(0.125 * 2.0 ** (i / 4.0) for i in range(45))


class ForwardLearningError(RuntimeError):
    """Raised when the model configuration or the requested operation is outside what this module supports."""


# ----------------------------------------------------------------------------------------------- closed forms
def _is_rmsnorm(module) -> bool:
    return type(module).__name__ == "RMSNorm" and hasattr(module, "scale") and hasattr(module, "eps")


def rms_norm_vjp(norm, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """J_n(x)^T v for n(x) = scale * x / sqrt(mean(x^2) + eps); x and v are [..., D]."""
    r2 = (x * x).mean(-1, keepdim=True) + float(norm.eps)
    r = torch.sqrt(r2)
    gv = norm.scale.to(x.dtype) * v
    return gv / r - x * (x * gv).sum(-1, keepdim=True) / (x.shape[-1] * r2 * r)


def _head(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(capped logits, derivative of the cap) at final residuals x [n, D]."""
    z = model.output_layer(model.output_norm(x))
    cap = float(getattr(model, "logit_soft_cap", 0.0))
    if cap > 0.0:
        th = torch.tanh(z / cap)
        return cap * th, 1.0 - th * th
    return z, torch.ones_like(z)


def output_error(model, x_final: torch.Tensor, targets: torch.Tensor, *, feedback: str = "error") -> torch.Tensor:
    """Closed-form d(-log p[y]) / d x_final for final residuals ``x_final`` [n, D] and next tokens ``targets`` [n].

    ``feedback="error"`` is the exact gradient. ``feedback="target"`` drops the prediction term and keeps only
    the arrived token's row (the Hebbian ablation used by the HEBB rules).
    """
    if feedback not in ("error", "target"):
        raise ValueError("feedback must be 'error' or 'target'")
    logits, slope = _head(model, x_final)
    rows = torch.arange(x_final.shape[0], device=x_final.device)
    if feedback == "error":
        e = torch.softmax(logits, dim=-1)
        e[rows, targets] -= 1.0
    else:
        e = torch.zeros_like(logits)
        e[rows, targets] = -1.0
    return rms_norm_vjp(model.output_norm, x_final, (e * slope) @ model.output_layer.weight)


def expected_self_error(model, x_final: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """sum_y p[y] * output_error(y) at one final residual ``x_final`` [D]: the expected error when the next
    token is sampled from the model's own distribution at temperature 1 (the score identity says it is zero).

    Every vocabulary entry is evaluated explicitly; this is a check, not something the learner needs.
    """
    logits, _ = _head(model, x_final.unsqueeze(0))
    p = torch.softmax(logits[0], dim=-1)
    total = torch.zeros_like(x_final)
    for start in range(0, p.shape[0], chunk):
        ys = torch.arange(start, min(start + chunk, p.shape[0]), device=x_final.device)
        d = output_error(model, x_final.unsqueeze(0).expand(len(ys), -1), ys)
        total = total + (p[ys].unsqueeze(-1) * d).sum(0)
    return total


# ------------------------------------------------------------------------------------------------------- tape
class ForwardTape:
    """Forward hooks that cache what the error fields need: block inputs, attention weights, fc2 inputs and
    the final residual. Use as a context manager around one forward call."""

    def __init__(self, model, need_attention: bool):
        self.model = model
        self.need_attention = bool(need_attention)
        self.block_input: Dict[int, torch.Tensor] = {}
        self.attention: Dict[int, torch.Tensor] = {}
        self.keys: Dict[int, torch.Tensor] = {}
        self.fc2_output: Dict[int, torch.Tensor] = {}
        self.x_final: Optional[torch.Tensor] = None
        self._hooks: list = []

    def __enter__(self) -> "ForwardTape":
        for k, layer in enumerate(self.model.layers):
            self._hooks.append(layer.register_forward_pre_hook(
                lambda mod, args, k=k: self.block_input.__setitem__(k, args[0])))
            if self.need_attention:
                self._hooks.append(layer.register_forward_hook(
                    lambda mod, args, out, k=k: self.attention.__setitem__(k, out[1])))
            self._hooks.append(layer.ffn.fc2.register_forward_hook(
                lambda mod, args, out, k=k: self._fc2(k, args[0], out)))
        self._hooks.append(self.model.output_norm.register_forward_pre_hook(
            lambda mod, args: setattr(self, "x_final", args[0])))
        return self

    def _fc2(self, k: int, h: torch.Tensor, out: torch.Tensor) -> None:
        self.keys[k] = h
        self.fc2_output[k] = out

    def __exit__(self, *exc) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []


def value_path_vjp(layer, attention: torch.Tensor, block_input: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """(A m)[s] = J_norm1(x_s)^T sum_h sum_{t>=s} alpha[h,t,s] W_V^{g(h)T} W_O^{hT} m[t].

    The transpose of the attention block's value path with the attention pattern held fixed. ``attention`` is
    [H, T, T] (query t, key s), ``block_input`` [T, D] the block's residual input, ``m`` [T, D] the error at the
    block's output. Grouped-query attention is handled: query head h reads key/value head h // groups.
    """
    attn = layer.attn
    n_heads, head_dim = int(attn.num_heads), int(attn.head_dim)
    groups = int(getattr(attn, "num_kv_groups", 1))
    t_len = m.shape[0]
    b = (m @ attn.out_proj.weight).reshape(t_len, n_heads, head_dim)        # W_O^T m, per query head
    c = torch.einsum("hts,thd->shd", attention.to(m.dtype), b)             # route to the sources
    if groups > 1:
        c = c.reshape(t_len, n_heads // groups, groups, head_dim).sum(2)
    hop = c.reshape(t_len, -1) @ attn.value_proj.weight                     # W_V^T
    return rms_norm_vjp(layer.norm1, block_input, hop)


def error_fields(model, tape: ForwardTape, delta_field: torch.Tensor, rule: str) -> List[Optional[torch.Tensor]]:
    """Per block, the estimated error at the fc2 output ([T, D]) for ``rule``; None where the rule does not write.

    ``delta_field`` [T, D] holds the output error at the learned positions and zeros elsewhere.
    """
    n_blocks = len(model.layers)
    if rule in _TOP_RULES:
        return [None] * (n_blocks - 1) + [delta_field.clone()]
    if rule not in _ROUTED_RULES:
        return [delta_field.clone() for _ in range(n_blocks)]
    fields: List[Optional[torch.Tensor]] = [None] * n_blocks
    m = delta_field.clone()
    single = torch.zeros_like(delta_field)
    for k in range(n_blocks - 1, -1, -1):
        fields[k] = m.clone() if rule == "FEC-A" else delta_field + single
        if k == 0:
            break
        src = m if rule == "FEC-A" else delta_field
        hop = value_path_vjp(model.layers[k], tape.attention[k][0], tape.block_input[k][0], src)
        if rule == "FEC-A":
            m = m + hop
        else:
            single = single + hop
    return fields


@contextmanager
def stop_gradient_graph(model, *, value: bool = True, query_key: bool = False, ffn: bool = False):
    """The autograd graph in which backward path classes are cut in every block.

    ``value=False`` detaches the attention value input, ``query_key=False`` the query and key inputs,
    ``ffn=False`` the FFN input. (value=True, query_key=False, ffn=False) is the graph whose exact fc2 gradient
    is FEC-A; (False, False, False) gives FEC-0; (True, True, True) is the standard graph. Only the fc2 gradients
    are meaningful in a cut graph (some norm scales receive no gradient there).
    """
    hooks = []

    def attn_pre(mod, args, kwargs):
        q, k, v = args[0], args[1], args[2]
        if not query_key:
            q, k = q.detach(), k.detach()
        if not value:
            v = v.detach()
        return (q, k, v) + tuple(args[3:]), kwargs

    def ffn_pre(mod, args):
        return (args[0].detach(),) + tuple(args[1:])

    try:
        for layer in model.layers:
            if not (value and query_key):
                hooks.append(layer.attn.register_forward_pre_hook(attn_pre, with_kwargs=True))
            if not ffn:
                hooks.append(layer.ffn.register_forward_pre_hook(ffn_pre))
        yield
    finally:
        for h in hooks:
            h.remove()


def backprop_fields(model, tokens: Sequence[int], positions: Sequence[int]) -> dict:
    """Reference with a backward pass: the exact error at every fc2 output for the mean next-token loss at
    ``positions`` (indices into ``tokens[:-1]``), in whatever graph is active (see ``stop_gradient_graph``).

    Returns {"fields": [T, D] per block, "keys": [T, F] per block, "x_final_grad": [T, D], "nll": float}.
    Used by tests and by exact-gradient comparison arms; the learner itself never calls it.
    """
    tok = torch.as_tensor(tokens, dtype=torch.long)
    x, y = tok[:-1].unsqueeze(0), tok[1:]
    pos = torch.as_tensor(sorted(int(p) for p in positions), dtype=torch.long)
    with torch.enable_grad():
        with ForwardTape(model, need_attention=False) as tape:
            # the fc2 outputs must be part of the graph even when every parameter is frozen
            grab = [layer.ffn.fc2.register_forward_hook(
                lambda mod, args, out: out if out.requires_grad else out.requires_grad_(True))
                for layer in model.layers]
            try:
                logits = model(x)[0]
            finally:
                for h in grab:
                    h.remove()
        loss = F.cross_entropy(logits[0, pos], y[pos])
        outs = [tape.fc2_output[k] for k in range(len(model.layers))]
        grads = torch.autograd.grad(loss, outs + [tape.x_final], allow_unused=True)
    zero = torch.zeros_like(outs[0][0])
    return {"fields": [zero.clone() if g is None else g[0].detach() for g in grads[:-1]],
            "keys": [tape.keys[k][0].detach() for k in range(len(model.layers))],
            "x_final_grad": grads[-1][0].detach(), "nll": float(loss.detach())}


# ------------------------------------------------------------------------------------------------ linear algebra
def woodbury_update(P: torch.Tensor, K: torch.Tensor, xi: float) -> torch.Tensor:
    """(P^-1 + xi K^T K)^-1 for P [F, F] and keys K [n, F] (float64)."""
    PK = P @ K.t()
    S = torch.eye(K.shape[0], dtype=P.dtype) / xi + K @ PK
    out = P - PK @ torch.linalg.solve(S, PK.t())
    return 0.5 * (out + out.t())


def woodbury_downdate(P: torch.Tensor, K: torch.Tensor, xi: float) -> torch.Tensor:
    """(P^-1 - xi K^T K)^-1; requires P^-1 - xi K^T K to be positive definite."""
    PK = P @ K.t()
    S = torch.eye(K.shape[0], dtype=P.dtype) / xi - K @ PK
    out = P + PK @ torch.linalg.solve(S, PK.t())
    return 0.5 * (out + out.t())


def familiarity_from(P: torch.Tensor, P0: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """nu(k) = 1 - k^T P k / k^T P0 k in [0, 1); 0 for a dead key (k^T P0 k not positive)."""
    k = keys.to(torch.float64)
    num = ((k @ P) * k).sum(-1)
    den = ((k @ P0) * k).sum(-1)
    safe = den > 1e-300
    nu = torch.where(safe, 1.0 - num / torch.where(safe, den, torch.ones_like(den)), torch.zeros_like(den))
    return nu.clamp(min=0.0)


# ------------------------------------------------------------------------------------------------- capacity law
def whitened_second_moment(keys: torch.Tensor, C0: torch.Tensor, *, unit: bool = False) -> torch.Tensor:
    """E[k^ k^T] for k^ = C0^{-1/2} k over the rows of ``keys``; with ``unit`` every k^ is scaled to length 1."""
    w, V = torch.linalg.eigh(C0.to(torch.float64))
    if float(w.min()) <= 0.0:
        raise ValueError("C0 must be positive definite")
    kh = keys.to(torch.float64) @ (V / torch.sqrt(w)) @ V.t()
    if unit:
        kh = kh / kh.norm(dim=-1, keepdim=True).clamp(min=1e-300)
    return kh.t() @ kh / kh.shape[0]


def _law_spectrum(S_w: torch.Tensor, S_q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    sw, U = torch.linalg.eigh(S_w.to(torch.float64))
    sw = sw.clamp(min=0.0)
    sq = torch.einsum("ij,jk,ki->i", U.t(), S_q.to(torch.float64), U)
    return sw, sq


def _law_value(sw: torch.Tensor, sq: torch.Tensor, xi: float, n: float) -> float:
    delta = float(xi * sw.sum())
    for _ in range(5000):
        new = float((sw / (1.0 / xi + n * sw / (1.0 + delta))).sum())
        if abs(new - delta) <= 1e-12 * max(1.0, delta):
            delta = new
            break
        delta = 0.5 * (delta + new)
    return float(1.0 - (sq / (1.0 + xi * n * sw / (1.0 + delta))).sum() / sq.sum())


def _as_classes(S_w, shares) -> Tuple[List[torch.Tensor], List[float]]:
    if isinstance(S_w, torch.Tensor):
        if shares is not None:
            raise ValueError("shares are given only with a list of class second moments")
        return [S_w.to(torch.float64)], [1.0]
    classes = [c.to(torch.float64) for c in S_w]
    if not classes or shares is None or len(shares) != len(classes):
        raise ValueError("a mixture needs one share per class")
    sh = [float(x) for x in shares]
    if any(x < 0 for x in sh) or abs(sum(sh) - 1.0) > 1e-9:
        raise ValueError("class shares must be non-negative and sum to 1")
    return classes, sh


def _mixture_value(classes: List[torch.Tensor], shares: List[float], S_q: torch.Tensor, xi: float, n: float,
                   delta: Optional[List[float]] = None) -> Tuple[float, List[float]]:
    """Mixture form of the law: one self-energy per class, delta_c = xi tr(S_c R),
    R = (I + xi n sum_c share_c S_c / (1 + delta_c))^-1; value = 1 - tr(S_q R) / tr(S_q)."""
    S_q = S_q.to(torch.float64)
    eye = torch.eye(S_q.shape[0], dtype=torch.float64)
    delta = [float(xi * torch.trace(c)) for c in classes] if delta is None else list(delta)
    R = eye
    for _ in range(2000):
        M = eye + xi * n * sum(sh * c / (1.0 + dl) for sh, c, dl in zip(shares, classes, delta))
        R = torch.cholesky_inverse(torch.linalg.cholesky(0.5 * (M + M.t())))
        new = [float(xi * (c * R).sum()) for c in classes]
        done = max(abs(a - b) for a, b in zip(new, delta)) <= 1e-10 * max(1.0, max(delta))
        delta = new if done else [0.5 * (a + b) for a, b in zip(delta, new)]
        if done:
            break
    return float(1.0 - (S_q * R).sum() / torch.trace(S_q)), delta


def predicted_familiarity(S_w, S_q: torch.Tensor, xi: float, n: float, *, shares: Optional[Sequence[float]] = None,
                          warm: Optional[List[float]] = None, return_state: bool = False):
    """Mean familiarity of a query population after ``n`` independent stored keys (no free parameter).

    S_w = E[k^ k^T] over the population the stored keys are drawn from, S_q the same over the queries, both in
    coordinates whitened by C0 (``whitened_second_moment``). With delta solving
    delta = tr[S_w (I/xi + n S_w/(1+delta))^-1] the prediction is
    1 - tr[S_q (I + xi n S_w/(1+delta))^-1] / tr S_q  (the deterministic equivalent of a ridge resolvent).

    Mixture form: when the stored keys come from several sub-populations (e.g. the prediction site and the later
    positions of a fact), pass ``S_w`` as a list of class second moments and ``shares`` as the class
    frequencies; every class then has its own self-energy delta_c = xi tr(S_c R) with
    R = (I + xi n sum_c share_c S_c / (1 + delta_c))^-1. One class is the form above. ``warm`` takes the
    self-energies of a nearby ``n`` as a starting point; ``return_state`` also returns them.
    """
    if n < 0 or not xi > 0:
        raise ValueError("n must be >= 0 and xi > 0")
    classes, sh = _as_classes(S_w, shares)
    if len(classes) == 1 and not return_state:
        sw, sq = _law_spectrum(classes[0], S_q)
        return _law_value(sw, sq, float(xi), float(n))
    value, state = _mixture_value(classes, sh, S_q, float(xi), float(n), warm)
    return (value, state) if return_state else value


def capacity_knee(S_w, S_q: torch.Tensor, xi: float, theta: float, n_max: float = 1e7, *,
                  shares: Optional[Sequence[float]] = None) -> Optional[float]:
    """The number of stored keys N* at which ``predicted_familiarity`` reaches ``theta``; None if it never does
    below ``n_max``. ``S_w`` may be a list of class second moments with ``shares`` (mixture form)."""
    if not 0.0 < theta < 1.0:
        raise ValueError("theta must lie in (0, 1)")
    classes, sh = _as_classes(S_w, shares)
    if len(classes) == 1:
        sw, sq = _law_spectrum(classes[0], S_q)

        def value(n: float) -> float:
            return _law_value(sw, sq, float(xi), n)
        steps, tol = 200, 1e-9
    else:
        def value(n: float) -> float:
            return _mixture_value(classes, sh, S_q, float(xi), n)[0]
        steps, tol = 40, 1e-4
    if value(float(n_max)) < theta:
        return None
    lo, hi = 0.0, 1.0
    while hi < n_max and value(hi) < theta:        # bracket by doubling, then bisect
        lo, hi = hi, min(2.0 * hi, float(n_max))
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if value(mid) < theta:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------------------------------------------ learner
@dataclass(frozen=True)
class ForwardLearningConfig:
    rule: str = "FEC-A"
    blocks: Optional[Tuple[int, ...]] = None      # writable blocks; None = the rule's own set
    projector: bool = True                        # RLS key projector P = C^-1 in the write
    n0: float = 50.0                              # prior weight: C0 = n0 * (key second moment + ridge)
    xi: float = 1.0                               # information added per stored key
    ridge: float = 1e-4                           # ridge relative to tr(key second moment) / F
    theta: float = 0.0                            # read-gate threshold; 0 = always open
    read_gain: float = 1.0                        # global gain on the fast output (evaluation switch)
    install: str = "nll"                          # "nll": target NLL < stop_nll; "greedy": argmax reproduces
    stop_nll: float = 0.05
    step_grid: Tuple[float, ...] = DEFAULT_STEP_GRID
    mu_cap: Optional[float] = None                # steps above the cap are not tried
    normalize: str = "global"                     # "global" or "per_block" normalisation of the direction
    min_error_norm: float = 1e-12                 # no write below this direction norm
    guard: str = "evict"                          # capacity guard policy
    rms_cap: float = 0.25                         # generic RMS of the fast output / native FFN output
    ledger_budget: Optional[int] = None           # maximum number of writes kept

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError(f"unknown rule {self.rule!r}; choose from {RULES}")
        if self.install not in INSTALL_CRITERIA:
            raise ValueError(f"install must be one of {INSTALL_CRITERIA}")
        if self.guard not in GUARD_POLICIES:
            raise ValueError(f"guard must be one of {GUARD_POLICIES}")
        if self.normalize not in ("global", "per_block"):
            raise ValueError("normalize must be 'global' or 'per_block'")
        if not (self.n0 > 0 and self.xi > 0 and self.ridge >= 0 and self.stop_nll > 0 and self.rms_cap > 0):
            raise ValueError("n0, xi, stop_nll and rms_cap must be positive; ridge non-negative")
        if not 0.0 <= self.theta < 1.0:
            raise ValueError("theta must lie in [0, 1)")
        if not (math.isfinite(self.read_gain) and self.read_gain >= 0.0):
            raise ValueError("read_gain must be finite and non-negative")
        grid = tuple(float(g) for g in self.step_grid)
        if not grid or any(g < 0 or not math.isfinite(g) for g in grid) or list(grid) != sorted(set(grid)):
            raise ValueError("step_grid must be a strictly increasing tuple of finite non-negative steps")
        if self.ledger_budget is not None and self.ledger_budget < 1:
            raise ValueError("ledger_budget must be at least 1")


@dataclass
class Reading:
    """One teacher-forced read of external tokens: everything a write needs, nothing written yet."""
    tokens: torch.Tensor                          # [T + 1]
    positions: Tuple[int, ...]                    # learned positions (indices into tokens[:-1])
    rule: str
    fields: List[Optional[torch.Tensor]]          # per block [T, D] or None
    keys: List[torch.Tensor]                      # per block [T, F]
    nll: float                                    # mean target NLL at the learned positions before the write
    greedy: bool                                  # the sentence was already reproduced before the write


@dataclass
class LedgerEntry:
    write_id: int
    tag: Optional[str]
    rule: str
    mu: float
    terms: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]   # block -> (R [n, D], Z [n, F], K [n, F])

    @property
    def n_keys(self) -> int:
        return sum(int(k.shape[0]) for _, _, k in self.terms.values())


@dataclass
class WriteResult:
    installed: bool
    write_id: Optional[int]
    mu: Optional[float]
    nll_before: float
    nll_after: float
    greedy_after: bool
    n_forward: int
    flags: List[str] = field(default_factory=list)
    evicted: List[int] = field(default_factory=list)
    block_share: Dict[int, float] = field(default_factory=dict)


class _Block:
    """Fast state of one block (float64); the float32 copy the hook uses is derived from it."""

    def __init__(self, shape: Tuple[int, int], dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype
        self.key_moment: Optional[torch.Tensor] = None
        self.native_rms: Optional[float] = None
        self.C0: Optional[torch.Tensor] = None
        self.P0: Optional[torch.Tensor] = None
        self.C: Optional[torch.Tensor] = None
        self.P: Optional[torch.Tensor] = None
        self.F64 = torch.zeros(shape, dtype=torch.float64)
        self.F32 = torch.zeros(shape, dtype=dtype)
        self.n_entries = 0
        self.n_keys = 0
        self.trial = False

    @property
    def calibrated(self) -> bool:
        return self.C0 is not None

    @property
    def active(self) -> bool:
        return self.n_entries > 0 or self.trial

    def clear(self) -> None:
        self.F64 = torch.zeros(self.shape, dtype=torch.float64)
        self.F32 = torch.zeros(self.shape, dtype=self.dtype)
        self.n_entries = 0
        self.n_keys = 0
        self.trial = False
        if self.calibrated:
            self.C = self.C0.clone()
            self.P = self.P0.clone()

    def sync(self) -> None:
        self.F32 = self.F64.to(self.dtype)


class ForwardLearner:
    """Optional forward-learning layer of a CevahirNeuralNetwork (see the module docstring).

    Typical use::

        learner = ForwardLearner(model, ForwardLearningConfig(rule="FEC-TOP", theta=0.5))
        learner.calibrate(reference_token_sequences)           # key statistics for the projector, gate, guard
        reading = learner.read(tokens, learn_positions)        # tokens supplied from outside, one pass
        result = learner.write(reading, tag="fact-17")         # forward-only step search, ledger entry
        learner.retract(result.write_id)                       # subtract it again
        learner.close()                                        # remove the hooks
    """

    def __init__(self, model, config: Optional[ForwardLearningConfig] = None):
        self.model = model
        self.config = config or ForwardLearningConfig()
        self._check_model()
        n_blocks = len(model.layers)
        if self.config.blocks is not None:
            bad = [b for b in self.config.blocks if not 0 <= int(b) < n_blocks]
            if bad or not self.config.blocks:
                raise ValueError(f"blocks {self.config.blocks!r} out of range 0..{n_blocks - 1}")
        self.enabled = True
        self.theta = float(self.config.theta)
        self.read_gain = float(self.config.read_gain)
        self._blocks: Dict[int, _Block] = {
            b: _Block(tuple(layer.ffn.fc2.weight.shape), layer.ffn.fc2.weight.dtype)
            for b, layer in enumerate(model.layers)}
        self._ledger: Dict[int, LedgerEntry] = {}
        self._next_id = 0
        self.gate_open: Dict[int, torch.Tensor] = {}
        self.evictions = 0
        self._handles = [layer.ffn.fc2.register_forward_hook(self._make_hook(b))
                         for b, layer in enumerate(model.layers)]

    # ------------------------------------------------------------------------------------------ configuration
    def _check_model(self) -> None:
        layers = getattr(self.model, "layers", None)
        if not layers:
            raise ForwardLearningError("model has no .layers (expected a CevahirNeuralNetwork)")
        if not _is_rmsnorm(getattr(self.model, "output_norm", None)):
            raise ForwardLearningError("output_norm must be RMSNorm")
        if not hasattr(self.model, "output_layer"):
            raise ForwardLearningError("model has no output_layer")
        for i, layer in enumerate(layers):
            if getattr(layer, "use_moe", False):
                raise ForwardLearningError(f"block {i}: MoE layers are not supported")
            if not getattr(layer, "pre_norm", False):
                raise ForwardLearningError(f"block {i}: post-norm blocks are not supported")
            if getattr(layer, "parallel_residual", False):
                raise ForwardLearningError(f"block {i}: parallel residual blocks are not supported")
            if not (_is_rmsnorm(getattr(layer, "norm1", None)) and _is_rmsnorm(getattr(layer, "norm2", None))):
                raise ForwardLearningError(f"block {i}: norm1 and norm2 must be RMSNorm")
            fc2 = getattr(getattr(layer, "ffn", None), "fc2", None)
            if fc2 is None:
                raise ForwardLearningError(f"block {i}: ffn.fc2 not found")
            weight = getattr(fc2, "weight", None)
            if not (isinstance(weight, torch.Tensor) and weight.dim() == 2 and weight.is_floating_point()):
                raise ForwardLearningError(f"block {i}: ffn.fc2 is not a plain floating-point linear layer "
                                           f"(quantized or wrapped layers are not supported)")
            if weight.device.type != "cpu":
                raise ForwardLearningError(f"block {i}: ffn.fc2 is on {weight.device}; the fast state is kept on "
                                           f"the CPU and only CPU models are supported")
            operator = getattr(getattr(layer, "attn", None), "attention_operator", "standard")
            if operator != "standard":
                raise ForwardLearningError(f"block {i}: attention operator {operator!r} is not supported "
                                           f"(the value-path transport assumes plain softmax attention)")

    def _check_eval(self) -> None:
        if self.model.training:
            raise ForwardLearningError("forward learning runs in eval mode (dropout, stochastic depth and "
                                       "gradient checkpointing change the forward quantities it reads)")
        if not self._handles:
            raise ForwardLearningError("this ForwardLearner was closed")

    def writable_blocks(self, rule: Optional[str] = None) -> Tuple[int, ...]:
        rule = rule or self.config.rule
        n_blocks = len(self.model.layers)
        own = (n_blocks - 1,) if rule in _TOP_RULES else tuple(range(n_blocks))
        if self.config.blocks is None:
            return own
        return tuple(b for b in own if b in set(int(x) for x in self.config.blocks))

    def close(self) -> None:
        """Remove every hook; the model is the plain model again."""
        for h in self._handles:
            h.remove()
        self._handles = []
        self._clear_cache()

    def _clear_cache(self) -> None:
        fn = getattr(self.model, "clear_kv_cache", None)
        if callable(fn):
            fn()

    @contextmanager
    def _own_forward(self):
        """The learner's own forward passes (reads, step search, calibration) must not show up in the host
        model's bookkeeping: its step counter and last diagnostics snapshot are put back afterwards."""
        missing = object()
        saved = {name: getattr(self.model, name, missing) for name in ("_global_step", "_last_snapshot")}
        try:
            yield
        finally:
            for name, value in saved.items():
                if value is not missing:
                    setattr(self.model, name, value)

    # --------------------------------------------------------------------------------------------------- read
    def _make_hook(self, b: int):
        def hook(mod, args, out):
            st = self._blocks[b]
            if not self.enabled or not st.active or self.read_gain == 0.0:
                return None
            h = args[0]
            fast = F.linear(h, st.F32)
            if self.read_gain != 1.0:
                fast = fast * self.read_gain
            if self.theta > 0.0:
                if not st.calibrated:
                    raise ForwardLearningError("the read gate needs key statistics: call calibrate() first")
                gate = familiarity_from(st.P, st.P0, h) >= self.theta
                self.gate_open[b] = gate
                return torch.where(gate.unsqueeze(-1), out + fast, out)
            return out + fast
        return hook

    def familiarity(self, block: int, keys: torch.Tensor) -> torch.Tensor:
        """nu at fc2 inputs ``keys`` [..., F] of ``block`` under the current state (float64)."""
        st = self._blocks[block]
        if not st.calibrated:
            raise ForwardLearningError("familiarity needs key statistics: call calibrate() first")
        return familiarity_from(st.P, st.P0, keys)

    # ---------------------------------------------------------------------------------------------- statistics
    @torch.no_grad()
    def calibrate(self, sequences: Iterable[Sequence[int]], exclude_ids: Sequence[int] = ()) -> Dict[int, dict]:
        """Key second moment E[h h^T] and RMS of the native FFN output for every block, from reference token
        sequences (one forward pass each). Must run before any write."""
        self._check_eval()
        if self._ledger:
            raise ForwardLearningError("calibrate() must run before the first write")
        blocks = self._blocks
        acc = {b: torch.zeros(st.shape[1], st.shape[1], dtype=torch.float64) for b, st in blocks.items()}
        pending: Dict[int, list] = {b: [] for b in blocks}
        sq = {b: 0.0 for b in blocks}
        count = 0
        keep_holder: dict = {}
        hooks = []

        def flush(b):
            if pending[b]:
                h = torch.cat(pending[b]).to(torch.float64)
                acc[b].add_(h.t() @ h)
                pending[b].clear()

        def grab(b):
            def hook(mod, args, out):
                keep = keep_holder["keep"]
                pending[b].append(args[0][0][keep])
                if sum(int(t.shape[0]) for t in pending[b]) >= 4096:
                    flush(b)
                sq[b] += float(out[0][keep].to(torch.float64).pow(2).sum())
            return hook

        for b, layer in enumerate(self.model.layers):
            hooks.append(layer.ffn.fc2.register_forward_hook(grab(b)))
        excl = torch.tensor(sorted(set(int(i) for i in exclude_ids)), dtype=torch.long)
        try:
            with self._own_forward():
                for seq in sequences:
                    x = torch.as_tensor(list(seq), dtype=torch.long).unsqueeze(0)
                    keep = ~torch.isin(x[0], excl) if excl.numel() else torch.ones(x.shape[1], dtype=torch.bool)
                    if not bool(keep.any()):
                        continue
                    keep_holder["keep"] = keep
                    self.model(x)
                    count += int(keep.sum())
        finally:
            for h in hooks:
                h.remove()
        for b in blocks:
            flush(b)
        if count == 0:
            raise ValueError("calibrate() needs at least one token")
        out = {}
        for b, st in blocks.items():
            self.set_key_statistics(b, acc[b] / count, math.sqrt(sq[b] / (count * st.shape[0])))
            out[b] = {"tokens": count, "native_rms": st.native_rms,
                      "mean_key_norm2": float(torch.trace(st.key_moment))}
        return out

    def set_key_statistics(self, block: int, key_moment: torch.Tensor, native_rms: float) -> None:
        """Install the key second moment [F, F] and the native FFN output RMS of one block directly."""
        st = self._blocks[block]
        if st.n_entries:
            raise ForwardLearningError("key statistics cannot change while the block holds writes")
        f = st.shape[1]
        km = torch.as_tensor(key_moment).detach().to(torch.float64).clone()
        if tuple(km.shape) != (f, f) or not torch.isfinite(km).all():
            raise ValueError(f"key_moment must be a finite [{f}, {f}] matrix")
        if not (native_rms > 0 and math.isfinite(native_rms)):
            raise ValueError("native_rms must be positive")
        km = 0.5 * (km + km.t())
        ridge = self.config.ridge * float(torch.trace(km)) / f
        st.key_moment = km
        st.native_rms = float(native_rms)
        st.C0 = self.config.n0 * (km + ridge * torch.eye(f, dtype=torch.float64))
        st.P0 = torch.cholesky_inverse(torch.linalg.cholesky(st.C0))
        st.P0 = 0.5 * (st.P0 + st.P0.t())
        st.C = st.C0.clone()
        st.P = st.P0.clone()

    def prior(self, block: int) -> torch.Tensor:
        """C0 of ``block`` (float64 copy)."""
        st = self._blocks[block]
        if not st.calibrated:
            raise ForwardLearningError("call calibrate() first")
        return st.C0.clone()

    def generic_rms_ratio(self, block: int) -> float:
        """sqrt(tr(F C_key F^T) / D) / native RMS: the RMS of the ungated fast output over generic positions
        relative to the native FFN output there."""
        st = self._blocks[block]
        if not st.calibrated:
            raise ForwardLearningError("call calibrate() first")
        val = float(((st.F64 @ st.key_moment) * st.F64).sum())
        return math.sqrt(max(val, 0.0) / st.shape[0]) / st.native_rms

    # ----------------------------------------------------------------------------------------------- reading
    def _as_tokens(self, tokens) -> torch.Tensor:
        tok = torch.as_tensor(tokens, dtype=torch.long).reshape(-1).clone()
        if tok.numel() < 2:
            raise ValueError("a read needs at least two tokens (one input and its next token)")
        return tok

    def _positions(self, tok: torch.Tensor, positions) -> Tuple[int, ...]:
        t_len = tok.numel() - 1
        pos = tuple(range(t_len)) if positions is None else tuple(sorted(set(int(p) for p in positions)))
        if not pos or pos[0] < 0 or pos[-1] >= t_len:
            raise ValueError(f"learn positions must lie in 0..{t_len - 1}")
        return pos

    def _score(self, tok: torch.Tensor, pos: Tuple[int, ...]) -> Tuple[float, bool]:
        """(mean target NLL, argmax reproduces every target) at the learned positions on the standard path."""
        with torch.no_grad(), self._own_forward():
            logits = self.model(tok[:-1].unsqueeze(0))[0][0]
        idx = torch.as_tensor(pos, dtype=torch.long)
        y = tok[1:][idx]
        nll = float(F.cross_entropy(logits[idx].float(), y))
        return nll, bool((logits[idx].argmax(-1) == y).all())

    @torch.no_grad()
    def read(self, tokens, learn_positions: Optional[Sequence[int]] = None, *, rule: Optional[str] = None,
             seed: Optional[int] = None) -> Reading:
        """Read externally supplied ``tokens`` once and return the error fields of ``rule``; nothing is written.

        ``tokens`` is the whole sequence [T + 1]; position t of the input ``tokens[:-1]`` is learned from the
        arriving token ``tokens[t + 1]``. ``learn_positions`` selects the positions whose arriving token comes
        from outside (default: all). Tokens the model produced itself must not be passed here.
        """
        self._check_eval()
        rule = rule or self.config.rule
        if rule not in RULES:
            raise ValueError(f"unknown rule {rule!r}")
        tok = self._as_tokens(tokens)
        pos = self._positions(tok, learn_positions)
        idx = torch.as_tensor(pos, dtype=torch.long)
        need_att = rule in _ROUTED_RULES
        with ForwardTape(self.model, need_attention=need_att) as tape, self._own_forward():
            logits = self.model(tok[:-1].unsqueeze(0), return_attention_weights=need_att)[0][0]
        y = tok[1:][idx]
        x_final = tape.x_final[0]
        feedback = "target" if rule.startswith("HEBB") else "error"
        delta = output_error(self.model, x_final[idx], y, feedback=feedback) / len(pos)
        if rule == "RAND-0":
            if seed is None:
                raise ValueError("RAND-0 needs a seed")
            g = torch.Generator().manual_seed(int(seed))
            rnd = torch.randn(delta.shape, generator=g, dtype=delta.dtype)
            delta = rnd * delta.norm(dim=-1, keepdim=True) / math.sqrt(delta.shape[-1])
        field_ = torch.zeros_like(x_final)
        field_[idx] = delta
        fields = error_fields(self.model, tape, field_, rule)
        writable = set(self.writable_blocks(rule))
        fields = [f if b in writable else None for b, f in enumerate(fields)]
        keys = [tape.keys[b][0].clone() for b in range(len(self.model.layers))]
        nll = float(F.cross_entropy(logits[idx].float(), y))
        greedy = bool((logits[idx].argmax(-1) == y).all())
        return Reading(tok, pos, rule, fields, keys, nll, greedy)

    def learn(self, tokens, learn_positions: Optional[Sequence[int]] = None, *, tag: Optional[str] = None,
              seed: Optional[int] = None) -> "WriteResult":
        """``read`` followed by ``write`` with the configured rule and install criterion."""
        return self.write(self.read(tokens, learn_positions, seed=seed), tag=tag)

    @torch.no_grad()
    def reading_from_fields(self, tokens, learn_positions: Sequence[int],
                            fields: Sequence[Optional[torch.Tensor]], keys: Sequence[torch.Tensor],
                            rule: str = "custom") -> Reading:
        """A ``Reading`` with error fields supplied by the caller (e.g. exact gradients from
        ``backprop_fields``), so that comparison arms use the same write, projector, step search and ledger."""
        self._check_eval()
        tok = self._as_tokens(tokens)
        pos = self._positions(tok, learn_positions)
        n_blocks = len(self.model.layers)
        if len(fields) != n_blocks or len(keys) != n_blocks:
            raise ValueError("fields and keys need one entry per block")
        writable = set(range(n_blocks)) if self.config.blocks is None else set(int(b) for b in self.config.blocks)
        fl = [None if (f is None or b not in writable) else f.detach().clone() for b, f in enumerate(fields)]
        nll, greedy = self._score(tok, pos)
        return Reading(tok, pos, rule, fl, [k.detach().clone() for k in keys], nll, greedy)

    # ------------------------------------------------------------------------------------------------ writing
    def _terms(self, reading: Reading) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                                                Dict[int, torch.Tensor], float]:
        """Unit direction of a reading: per block (M, Z, K) and U = -M^T Z / norm, plus the norm."""
        raw: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        grads: Dict[int, torch.Tensor] = {}
        for b, fld in enumerate(reading.fields):
            if fld is None:
                continue
            rows = fld.abs().sum(-1) > 0
            if not bool(rows.any()):
                continue
            M = fld[rows].to(torch.float64)
            K = reading.keys[b][rows].to(torch.float64)
            st = self._blocks[b]
            if self.config.projector:
                if not st.calibrated:
                    raise ForwardLearningError("the key projector needs key statistics: call calibrate() first")
                Z = K @ st.P
            else:
                Z = K.clone()
            raw[b] = (M, Z, K)
            grads[b] = M.t() @ Z
        if not raw:
            return {}, {}, 0.0
        norms = {b: float(g.norm()) for b, g in grads.items()}
        total = math.sqrt(sum(v * v for v in norms.values()))
        if self.config.normalize == "per_block":
            live = [b for b, v in norms.items() if v > 0.0]
            scale = {b: (1.0 / (norms[b] * math.sqrt(len(live))) if norms[b] > 0.0 else 0.0) for b in norms}
        else:
            scale = {b: (1.0 / total if total > 0.0 else 0.0) for b in norms}
        terms = {b: (-scale[b] * M, Z, K) for b, (M, Z, K) in raw.items()}
        unit = {b: -scale[b] * grads[b] for b in grads}
        return terms, unit, total

    def _tracks(self, st: _Block) -> bool:
        return st.calibrated

    @torch.no_grad()
    def write(self, reading: Reading, *, mu: Optional[float] = None, install: Optional[str] = None,
              tag: Optional[str] = None) -> WriteResult:
        """Write ``reading`` into the fast weights. Without ``mu`` the smallest step on the grid that meets the
        install criterion is searched with forward passes only; a write that cannot be installed leaves the
        state unchanged."""
        self._check_eval()
        cfg = self.config
        install = install or cfg.install
        if install not in INSTALL_CRITERIA:
            raise ValueError(f"install must be one of {INSTALL_CRITERIA}")
        tok, pos = reading.tokens, reading.positions
        self._clear_cache()
        nll0, greedy0 = self._score(tok, pos)
        n_forward = 1

        def met(nll: float, greedy: bool) -> bool:
            return nll < cfg.stop_nll if install == "nll" else greedy

        if mu is None and met(nll0, greedy0):
            return WriteResult(True, None, 0.0, nll0, nll0, greedy0, n_forward, ["already_installed"])
        terms, unit, norm = self._terms(reading)
        if not terms or norm < cfg.min_error_norm:
            return WriteResult(False, None, None, nll0, nll0, greedy0, n_forward, ["no_error"])
        saved = {b: (self._blocks[b].F64, self._blocks[b].C, self._blocks[b].P) for b in terms}
        try:
            for b, (_, _, K) in terms.items():
                st = self._blocks[b]
                st.trial = True
                if self._tracks(st):
                    st.C = st.C + cfg.xi * (K.t() @ K)
                    st.P = woodbury_update(st.P, K, cfg.xi)

            def trial(step: float) -> Tuple[float, bool]:
                for b in terms:
                    st = self._blocks[b]
                    st.F32 = (saved[b][0] + step * unit[b]).to(st.dtype)
                self._clear_cache()
                return self._score(tok, pos)

            chosen: Optional[float] = None
            nll1, greedy1 = nll0, greedy0
            flags: List[str] = []
            if mu is not None:
                chosen = float(mu)
                nll1, greedy1 = trial(chosen)
                n_forward += 1
                if not met(nll1, greedy1):
                    flags.append("criterion_unmet")
            else:
                best: Optional[Tuple[float, float, bool]] = None
                for step in cfg.step_grid:
                    if step == 0.0:
                        continue
                    if cfg.mu_cap is not None and step > cfg.mu_cap:
                        flags.append("capped")
                        break
                    nll, greedy = trial(step)
                    n_forward += 1
                    if met(nll, greedy):
                        chosen, nll1, greedy1 = step, nll, greedy
                        break
                    if best is None or nll < best[1]:
                        best = (step, nll, greedy)
                if chosen is None and install == "nll" and best is not None and best[2]:
                    chosen, nll1, greedy1 = best
                    flags.append("nll_unmet")
            if chosen is None:
                raise _NotInstalled(flags or ["not_installed"], n_forward)
        except _NotInstalled as miss:
            self._rollback(terms, saved)
            return WriteResult(False, None, None, nll0, nll0, greedy0, miss.n_forward, miss.flags)
        except BaseException:
            self._rollback(terms, saved)
            raise
        entry_terms = {b: (chosen * R, Z, K) for b, (R, Z, K) in terms.items()}
        share = {}
        for b, (R, Z, K) in entry_terms.items():
            st = self._blocks[b]
            st.F64 = saved[b][0] + R.t() @ Z
            st.trial = False
            st.n_entries += 1
            st.n_keys += int(K.shape[0])
            st.sync()
            share[b] = float((unit[b] * unit[b]).sum())
        tot = sum(share.values()) or 1.0
        share = {b: v / tot for b, v in share.items()}
        wid = self._next_id
        self._next_id += 1
        self._ledger[wid] = LedgerEntry(wid, tag, reading.rule, float(chosen), entry_terms)
        self._clear_cache()
        result = WriteResult(True, wid, float(chosen), nll0, nll1, greedy1, n_forward, flags, [], share)
        self._guard(result)
        return result

    def _rollback(self, terms, saved) -> None:
        for b in terms:
            st = self._blocks[b]
            st.F64, st.C, st.P = saved[b]
            st.trial = False
            st.sync()
        self._clear_cache()

    @torch.no_grad()
    def write_terms(self, terms: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], *,
                    tag: Optional[str] = None, rule: str = "terms") -> int:
        """Low-level write: per block, values R [n, D], gain vectors Z [n, F] and keys K [n, F]; adds
        sum_i r_i z_i^T to F and xi sum_i k_i k_i^T to C. Returns the write id. No step search, no guard."""
        self._check_eval()
        entry: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for b, (R, Z, K) in terms.items():
            st = self._blocks[int(b)]
            R = torch.as_tensor(R).detach().to(torch.float64).reshape(-1, st.shape[0]).clone()
            Z = torch.as_tensor(Z).detach().to(torch.float64).reshape(-1, st.shape[1]).clone()
            K = torch.as_tensor(K).detach().to(torch.float64).reshape(-1, st.shape[1]).clone()
            if not (R.shape[0] == Z.shape[0] == K.shape[0] and R.shape[0] > 0):
                raise ValueError("R, Z and K need the same positive number of rows")
            if not (torch.isfinite(R).all() and torch.isfinite(Z).all() and torch.isfinite(K).all()):
                raise ValueError("non-finite write")
            entry[int(b)] = (R, Z, K)
        if not entry:
            raise ValueError("empty write")
        for b, (R, Z, K) in entry.items():
            st = self._blocks[b]
            st.F64 = st.F64 + R.t() @ Z
            if self._tracks(st):
                st.C = st.C + self.config.xi * (K.t() @ K)
                st.P = woodbury_update(st.P, K, self.config.xi)
            st.n_entries += 1
            st.n_keys += int(K.shape[0])
            st.sync()
        wid = self._next_id
        self._next_id += 1
        self._ledger[wid] = LedgerEntry(wid, tag, rule, 1.0, entry)
        self._clear_cache()
        return wid

    def rls_gain(self, block: int, key: torch.Tensor) -> torch.Tensor:
        """The recursive-least-squares gain P k / (1/xi + k^T P k) of one key under the current state."""
        st = self._blocks[block]
        if not st.calibrated:
            raise ForwardLearningError("call calibrate() first")
        k = torch.as_tensor(key).detach().to(torch.float64).reshape(-1)
        pk = st.P @ k
        return pk / (1.0 / self.config.xi + float(k @ pk))

    # ---------------------------------------------------------------------------------------- ledger and guards
    @property
    def ledger(self) -> List[LedgerEntry]:
        return [self._ledger[i] for i in sorted(self._ledger)]

    def __len__(self) -> int:
        return len(self._ledger)

    def n_keys(self, block: int) -> int:
        return self._blocks[block].n_keys

    def fast_weight(self, block: int) -> torch.Tensor:
        """F of ``block`` (float64 copy)."""
        return self._blocks[block].F64.clone()

    @torch.no_grad()
    def retract(self, write_id: int) -> None:
        """Subtract one write: F loses its rank-one terms, C and P are downdated. Exact at the ledger level;
        gain vectors of later writes were computed with this write's keys present and are left as they are."""
        if write_id not in self._ledger:
            raise KeyError(f"no write {write_id!r} in the ledger")
        entry = self._ledger.pop(write_id)
        for b, (R, Z, K) in entry.terms.items():
            st = self._blocks[b]
            st.n_entries -= 1
            st.n_keys -= int(K.shape[0])
            if st.n_entries == 0:
                st.clear()                       # an empty block carries no float residue
                continue
            st.F64 = st.F64 - R.t() @ Z
            if self._tracks(st):
                st.C = st.C - self.config.xi * (K.t() @ K)
                try:
                    st.P = woodbury_downdate(st.P, K, self.config.xi)
                except RuntimeError:
                    st.P = torch.linalg.inv(st.C)
                    st.P = 0.5 * (st.P + st.P.t())
            st.sync()
        self._clear_cache()

    def reset(self) -> None:
        """Drop every write; the model is the frozen model again (key statistics are kept)."""
        self._ledger.clear()
        for st in self._blocks.values():
            st.clear()
        self.gate_open = {}
        self._clear_cache()

    @torch.no_grad()
    def rebuild(self) -> Dict[str, float]:
        """Recompute F, C and P from the ledger and install them. Returns the largest absolute deviation of
        the running state from the rebuilt one (the ledger identity error)."""
        err = {"F": 0.0, "C": 0.0, "P": 0.0}
        for b, st in self._blocks.items():
            Fb = torch.zeros(st.shape, dtype=torch.float64)
            Cb = st.C0.clone() if st.calibrated else None
            for e in self.ledger:
                if b in e.terms:
                    R, Z, K = e.terms[b]
                    Fb = Fb + R.t() @ Z
                    if Cb is not None:
                        Cb = Cb + self.config.xi * (K.t() @ K)
            err["F"] = max(err["F"], float((Fb - st.F64).abs().max()))
            st.F64 = Fb
            if Cb is not None:
                Pb = torch.linalg.inv(Cb)
                Pb = 0.5 * (Pb + Pb.t())
                err["C"] = max(err["C"], float((Cb - st.C).abs().max()))
                err["P"] = max(err["P"], float((Pb - st.P).abs().max()))
                st.C, st.P = Cb, Pb
            st.sync()
        self._clear_cache()
        return err

    def _violations(self) -> List[str]:
        out = []
        cfg = self.config
        if cfg.ledger_budget is not None and len(self._ledger) > cfg.ledger_budget:
            out.append("budget")
        for b, st in self._blocks.items():
            if st.n_entries and st.calibrated and self.generic_rms_ratio(b) > cfg.rms_cap:
                out.append(f"rms@{b}")
        return out

    def _guard(self, result: WriteResult) -> None:
        cfg = self.config
        if cfg.guard == "off":
            return
        bad = self._violations()
        if not bad:
            return
        if cfg.guard == "refuse":
            self.retract(result.write_id)
            result.installed, result.write_id, result.mu = False, None, None
            result.flags.append("refused:" + ",".join(bad))
            return
        while bad and len(self._ledger) > 1:
            oldest = min(i for i in self._ledger if i != result.write_id)
            self.retract(oldest)
            result.evicted.append(oldest)
            self.evictions += 1
            bad = self._violations()
        if bad:                                   # the new write alone breaks the invariant
            self.retract(result.write_id)
            result.installed, result.write_id, result.mu = False, None, None
            result.flags.append("refused:" + ",".join(bad))

    # ------------------------------------------------------------------------------------------------ snapshots
    def snapshot(self) -> dict:
        """A copy of the whole fast state (ledger entries are immutable and shared)."""
        return {"ledger": dict(self._ledger), "next_id": self._next_id, "evictions": self.evictions,
                "blocks": {b: (st.F64.clone(), None if st.C is None else st.C.clone(),
                               None if st.P is None else st.P.clone(), st.n_entries, st.n_keys)
                           for b, st in self._blocks.items()}}

    def restore(self, snap: dict) -> None:
        self._ledger = dict(snap["ledger"])
        self._next_id = int(snap["next_id"])
        self.evictions = int(snap["evictions"])
        for b, (Fb, Cb, Pb, n_entries, n_keys) in snap["blocks"].items():
            st = self._blocks[b]
            st.F64 = Fb.clone()
            st.C = None if Cb is None else Cb.clone()
            st.P = None if Pb is None else Pb.clone()
            st.n_entries, st.n_keys, st.trial = int(n_entries), int(n_keys), False
            st.sync()
        self._clear_cache()

    def fingerprint(self) -> str:
        """sha256 over the float64 fast weights and the ledger order (equal states give equal digests)."""
        h = hashlib.sha256()
        for b in sorted(self._blocks):
            st = self._blocks[b]
            h.update(f"{b}|{st.n_entries}|{st.n_keys}|".encode("ascii"))
            h.update(st.F64.contiguous().numpy().tobytes())
        h.update(",".join(str(i) for i in sorted(self._ledger)).encode("ascii"))
        return h.hexdigest()


class _NotInstalled(Exception):
    def __init__(self, flags: List[str], n_forward: int):
        super().__init__(",".join(flags))
        self.flags = flags
        self.n_forward = n_forward


__all__ = [
    "FEC_RULES", "CONTROL_RULES", "RULES", "DEFAULT_STEP_GRID", "ForwardLearningError", "ForwardLearningConfig",
    "ForwardLearner", "ForwardTape", "Reading", "LedgerEntry", "WriteResult", "rms_norm_vjp", "output_error",
    "expected_self_error", "value_path_vjp", "error_fields", "stop_gradient_graph", "backprop_fields",
    "woodbury_update", "woodbury_downdate", "familiarity_from", "whitened_second_moment",
    "predicted_familiarity", "capacity_knee",
]
