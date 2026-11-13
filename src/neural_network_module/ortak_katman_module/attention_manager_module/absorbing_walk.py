# -*- coding: utf-8 -*-
"""
Absorbing-Walk Attention (AWA) and its comparison operators for ``MultiHeadAttention``.

Optional and OFF by default: a ``MultiHeadAttention`` whose ``attention_operator`` is "standard" never
imports or calls anything in this file, registers none of its parameters and computes exactly what it
computed before this file existed.

Operators (one head of one layer; ``x`` is the layer input the attention module receives)
-----------------------------------------------------------------------------------------
``a_i``   ordinary attention row of query ``i`` over its window ``W_i`` (the keys its mask allows).
          Hop 1 of every operator is this row, computed by the unchanged softmax path.
``awa``   prefix-closed absorbing walk (query-window normalisation)::

              kbar_l = W_K x_l                 pre-RoPE key (after QK-norm, per KV group under GQA)
              t_j    = W_T x_j                 pointer, one per query head
              E_jl   = exp(kappa * tanh(t_j . kbar_l / (kappa * sqrt(d))))      for l != j, else 0
              c_j    = lambda_max * sigmoid(rho) * sigmoid(u . x_j + b)         continue probability
              P(i)_jl = E_jl / (e^tau + sum_{l' in W_i} E_jl')                   l in W_i
              pi(0) = a_i,  pi(m+1)_l = sum_j pi(m)_j c_j P(i)_jl
              A~(K)_i = sum_{m<K} pi(m) (1 - c + c P(i)_null) + pi(K)

          K weight-tied steps (``walk_num_hops``; it can be changed at inference). The limit K -> inf is
          ``a_i (I - C P(i))^{-1} S(i)`` with ``S(i) = diag(1 - c + c P(i)_null)``
          (:func:`absorbing_walk_resolvent`, used as the reference in the tests). Rows of ``A~`` keep the
          mass of ``a_i``; nothing outside ``W_i`` receives mass, so the head stays causal for the query.
``awa_own``  the single-ingredient ablation: every hop ``j -> l`` is normalised over ``j``'s OWN window
          (``P_jl = E_jl / (e^tau + sum_{l' in W_j} E_jl')``). Under a causal mask every factor is
          lower-triangular, so only position-non-increasing paths contribute.
``chacal``  ChaCAL-style resolvent ``(1 - gamma) A (I - gamma A)^{-1}`` with one learned
          ``gamma = gamma_max * sigmoid(rho)`` per head, computed exactly by one solve. The label is
          "ChaCAL-style": the published parametrisation has not been checked against the paper.

Limits (stated, not hidden)
---------------------------
* Cost. All-row AWA is O(K n^3) per head (O(K R n^2) for R query rows); standard attention is O(n^2 d).
* KV cache. ``awa`` decodes with the KV cache (pointer, address and gate of cached positions are kept
  next to the cache and E is recomputed every step, O(n^2 d) per token and head). ``awa_own`` and
  ``chacal`` need every position's own attention row and raise ``NotImplementedError`` when the cache
  is active.
* ``awa_own`` and ``chacal`` are defined for self-attention only (queries and keys are the same
  positions); ``awa`` only needs the pointer and gate of the key positions.
* Additive masks at or below ``MASK_BLOCK_THRESHOLD`` count as "outside the window"; smaller finite
  penalties stay inside it.
* The walk runs in float32 or wider: E reaches e^kappa (4.9e8 at kappa = 20), above the fp16 range. Half
  precision never enters it: ``transition_weights``, ``absorbing_walk``, ``absorbing_walk_resolvent`` and
  ``chacal_resolvent`` widen fp16 / bf16 arguments to float32 and switch autocast off for their own
  arithmetic, so under ``torch.autocast`` (fp16 or bf16) the rest of the model runs in half precision and
  the operator in float32. Their results are float32 in that case; the caller casts them back.
* Attention dropout is applied to the effective weights (after the walk), never to hop 1.

Rows-only evaluation (:func:`attention_rows`, :func:`layer_forward_rows`) computes one encoder layer at
selected positions only; it re-uses the live module's projections, norms, RoPE, softmax code and FFN.
"""
from __future__ import annotations

import contextlib
import functools
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

LAMBDA_MAX = 0.99
GAMMA_MAX = 0.99
STANDARD = "standard"
ATTENTION_OPERATORS: Tuple[str, ...] = ("standard", "awa", "awa_own", "chacal")
WALK_OPERATORS: Tuple[str, ...] = ("awa", "awa_own")
MASK_BLOCK_THRESHOLD = -1e8
DEFAULT_OPERATOR_OPTIONS: Dict[str, Any] = {
    "layers": None,          # layer indices that carry the operator; None = the last layer only
    "num_hops": 4,           # K, walk steps after hop 1 (awa, awa_own)
    "lambda_init": 0.5,      # initial lambda = lambda_max * sigmoid(rho)
    "tau_init": 0.0,         # initial null-option logit
    "score_cap": 20.0,       # kappa; 0 disables the tanh cap
    "exclude_self": True,    # E_jj = 0
    "gamma_init": 0.5,       # initial gamma (chacal)
}


def _logit(p: float) -> float:
    return math.log(p) - math.log1p(-p)


def _wide(t: torch.Tensor) -> torch.Tensor:
    """float32 for half-precision tensors (E reaches e^kappa); float32 / float64 stay as they are."""
    return t.float() if t.dtype in (torch.float16, torch.bfloat16) else t


def _autocast_off(t: torch.Tensor):
    """Context manager that switches autocast off on the device of ``t`` (a no-op when it is not active)."""
    try:
        return torch.autocast(device_type=t.device.type, enabled=False)
    except (RuntimeError, ValueError):          # a device type without autocast: nothing to switch off
        return contextlib.nullcontext()


def _full_precision(fn):
    """Run ``fn`` in float32 or wider whatever the caller's autocast state.

    Inside ``torch.autocast(dtype=float16)`` a matmul of float32 tensors is cast back to fp16, where E (up to
    e^kappa) overflows to inf and the walk returns NaN. The wrapped primitive therefore widens every
    half-precision floating argument and computes with autocast off. Without autocast and with float32 /
    float64 arguments nothing changes: the same tensors (the same objects) reach ``fn``.
    """
    def widen(value):
        return _wide(value) if isinstance(value, torch.Tensor) and value.is_floating_point() else value

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        args = tuple(widen(a) for a in args)
        kwargs = {k: widen(v) for k, v in kwargs.items()}
        anchor = next((a for a in (*args, *kwargs.values()) if isinstance(a, torch.Tensor)), None)
        if anchor is None:
            return fn(*args, **kwargs)
        with _autocast_off(anchor):
            return fn(*args, **kwargs)

    return wrapper


# --------------------------------------------------------------------------------- configuration
def resolve_operator_config(operator: Any, options: Any = None,
                            num_layers: Optional[int] = None) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Validate ``attention_operator`` / ``attention_operator_options``.

    Returns ``(operator, options)``. For "standard" the options are ``None`` (and none may be given).
    Otherwise every default is filled in, and with ``num_layers`` the layer list is resolved to sorted
    non-negative indices. Raises ``ValueError`` on anything it does not understand.
    """
    if operator is None:
        operator = STANDARD
    if not isinstance(operator, str) or operator not in ATTENTION_OPERATORS:
        raise ValueError(f"attention_operator must be one of {ATTENTION_OPERATORS}, got {operator!r}")
    if options is not None and not isinstance(options, Mapping):
        raise ValueError("attention_operator_options must be a mapping or None")
    given = dict(options or {})
    if operator == STANDARD:
        if given:
            raise ValueError("attention_operator_options were given but attention_operator is 'standard'")
        return STANDARD, None
    unknown = sorted(set(given) - set(DEFAULT_OPERATOR_OPTIONS))
    if unknown:
        raise ValueError(f"Unknown attention_operator_options: {unknown}")
    out = {**DEFAULT_OPERATOR_OPTIONS, **given}

    def _number(name: str) -> float:
        value = out[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"attention_operator_options[{name!r}] must be a finite number")
        return float(value)

    hops = out["num_hops"]
    if isinstance(hops, bool) or not isinstance(hops, int) or hops < 0:
        raise ValueError("attention_operator_options['num_hops'] must be a non-negative integer")
    if not isinstance(out["exclude_self"], bool):
        raise ValueError("attention_operator_options['exclude_self'] must be a bool")
    if not 0.0 < _number("lambda_init") < LAMBDA_MAX:
        raise ValueError(f"attention_operator_options['lambda_init'] must be in (0, {LAMBDA_MAX})")
    if not 0.0 < _number("gamma_init") < GAMMA_MAX:
        raise ValueError(f"attention_operator_options['gamma_init'] must be in (0, {GAMMA_MAX})")
    if _number("score_cap") < 0.0:
        raise ValueError("attention_operator_options['score_cap'] must be >= 0 (0 disables the cap)")
    out["lambda_init"], out["gamma_init"] = _number("lambda_init"), _number("gamma_init")
    out["tau_init"], out["score_cap"] = _number("tau_init"), _number("score_cap")
    layers = out["layers"]
    if layers is not None:
        if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence) or not len(layers):
            raise ValueError("attention_operator_options['layers'] must be a non-empty list of layer indices")
        if any(isinstance(i, bool) or not isinstance(i, int) for i in layers):
            raise ValueError("attention_operator_options['layers'] must contain integers")
        layers = [int(i) for i in layers]
    if num_layers is not None:
        layers = [num_layers - 1] if layers is None else layers
        if any(not -num_layers <= i < num_layers for i in layers):
            raise ValueError(f"attention_operator_options['layers'] out of range for {num_layers} layers: {layers}")
        layers = sorted(i % num_layers for i in layers)
        if len(set(layers)) != len(layers):
            raise ValueError("attention_operator_options['layers'] names a layer twice")
    out["layers"] = layers
    return operator, out


def attach_operator(attn: nn.Module, operator: str, *, num_hops: int = 4, lambda_init: float = 0.5,
                    tau_init: float = 0.0, score_cap: float = 20.0, exclude_self: bool = True,
                    gamma_init: float = 0.5) -> nn.Module:
    """Register the operator's parameters on a live ``MultiHeadAttention`` and switch it on.

    awa / awa_own add ``walk_pointer`` (D x D), ``walk_gate`` (H x D weight + H bias), ``walk_rho`` (H) and
    ``walk_tau`` (H): D^2 + H(D + 1) + 2H parameters. chacal adds ``chacal_rho`` (H). The parameters are
    created here, i.e. after the module's own parameters, so the caller decides where in the random
    stream they are drawn (``CevahirNeuralNetwork`` attaches them after every other module is built:
    a standard model and an operator model built from the same seed share all common weights).
    """
    operator, opts = resolve_operator_config(operator, dict(
        num_hops=num_hops, lambda_init=lambda_init, tau_init=tau_init, score_cap=score_cap,
        exclude_self=exclude_self, gamma_init=gamma_init) if operator != STANDARD else None)
    if getattr(attn, "attention_operator", STANDARD) != STANDARD:
        raise ValueError(f"this attention module already carries the operator {attn.attention_operator!r}")
    if operator == STANDARD:
        return attn
    D, H, d = attn.embed_dim, attn.num_heads, attn.head_dim
    ref = attn.query_proj.weight
    if operator in WALK_OPERATORS:
        attn.walk_pointer = nn.Linear(D, H * d, bias=False).to(ref)
        nn.init.xavier_uniform_(attn.walk_pointer.weight)
        attn.walk_gate = nn.Linear(D, H, bias=True).to(ref)
        nn.init.zeros_(attn.walk_gate.weight)
        nn.init.zeros_(attn.walk_gate.bias)
        attn.walk_rho = nn.Parameter(torch.full((H,), _logit(opts["lambda_init"] / LAMBDA_MAX),
                                                dtype=ref.dtype, device=ref.device))
        attn.walk_tau = nn.Parameter(torch.full((H,), opts["tau_init"], dtype=ref.dtype, device=ref.device))
        attn.walk_num_hops = int(opts["num_hops"])
        attn.walk_score_cap = float(opts["score_cap"])
        attn.walk_exclude_self = bool(opts["exclude_self"])
    else:
        attn.chacal_rho = nn.Parameter(torch.full((H,), _logit(opts["gamma_init"] / GAMMA_MAX),
                                                  dtype=ref.dtype, device=ref.device))
    attn.attention_operator = operator
    return attn


def walk_lambda(attn: nn.Module) -> torch.Tensor:
    """Per-head lambda = lambda_max * sigmoid(rho) (never exactly 0 for finite rho)."""
    return LAMBDA_MAX * torch.sigmoid(attn.walk_rho)


def chacal_gamma(attn: nn.Module) -> torch.Tensor:
    return GAMMA_MAX * torch.sigmoid(attn.chacal_rho)


def continue_probability(attn: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """c_j = lambda_h * sigmoid(u_h . x_j + b_h) for the layer input x (B, S, D) -> (B, H, S)."""
    gate = _wide(attn.walk_gate(x).transpose(1, 2))
    return _wide(walk_lambda(attn)).view(1, -1, 1) * torch.sigmoid(gate)


# ------------------------------------------------------------------------------------ primitives
@_full_precision
def transition_weights(pointer: torch.Tensor, address: torch.Tensor, score_cap: float = 20.0,
                       exclude_self: bool = True) -> torch.Tensor:
    """E_jl = exp(kappa tanh(t_j . kbar_l / (kappa sqrt(d)))), zero on the diagonal if ``exclude_self``.

    pointer, address: (B, H, S, d) -> (B, H, S, S).  ``score_cap`` 0 leaves the scores uncapped.
    Half-precision inputs are widened: the result is float32 or wider, also under autocast.
    """
    scores = torch.matmul(pointer, address.transpose(-1, -2)) / math.sqrt(pointer.size(-1))
    if score_cap > 0.0:
        scores = torch.tanh(scores / score_cap) * score_cap
    weights = torch.exp(scores)
    if exclude_self:
        n = weights.size(-1)
        weights = weights.masked_fill(torch.eye(n, dtype=torch.bool, device=weights.device), 0.0)
    return weights


@_full_precision
def absorbing_walk(start: torch.Tensor, transitions: torch.Tensor, cont: torch.Tensor, null: torch.Tensor,
                   num_hops: int, *, allowed: Optional[torch.Tensor] = None,
                   own_window: Optional[torch.Tensor] = None, return_all: bool = False,
                   trace: Optional[List[torch.Tensor]] = None):
    """K steps of the absorbing walk for the given query rows.

    start        (B, H, R, S)  hop-1 rows a_i.
    transitions  (B, H, S, S)  E_jl >= 0.
    cont         (B, H, S)     continue probabilities c_j in [0, 1).
    null         (H,) or broadcastable to (B, H, 1, 1): e^tau > 0.
    allowed      bool, broadcastable to (B, H, R, S): the window W_i of each query row (None = all keys).
                 Hops are normalised over W_i and never leave it (query-window walk, ``awa``).
    own_window   bool, broadcastable to (B, H, S, S): the window of every key position. When given, hops
                 are normalised over the hopping position's own window instead (``awa_own``) and
                 ``allowed`` is not used.
    return_all   return the list [A~ at budget 0, 1, ..., num_hops] (budget 0 is ``start`` itself).
    trace        optional list; the moving mass pi(m) of every hop m = 0..num_hops is appended to it.

    Mass is conserved exactly in exact arithmetic: every row of the result sums to the row sum of
    ``start``. With cont == 0 the result equals ``start`` bit for bit. Computed in float32 or wider with
    autocast off (half-precision arguments are widened first).
    """
    if num_hops < 0:
        raise ValueError("num_hops must be non-negative")
    if null.dim() == 1:
        null = null.view(1, -1, 1, 1)
    null = null.to(start.dtype)
    if trace is not None:
        trace.append(start)
    if num_hops == 0:
        return [start] if return_all else start
    visible = None
    if own_window is not None:
        transitions = transitions * own_window.to(transitions.dtype)
        norm = transitions.sum(-1).unsqueeze(-2) + null                       # (B, H, 1, S)
    elif allowed is None:
        norm = transitions.sum(-1).unsqueeze(-2) + null
    else:
        visible = allowed.to(transitions.dtype)
        norm = torch.matmul(visible, transitions.transpose(-1, -2)) + null    # [i, j] = sum_{l in W_i} E_jl
    stop_null = null / norm
    cont = cont.unsqueeze(-2)
    pi, absorbed = start, torch.zeros_like(start)
    outs = [start] if return_all else None
    for _ in range(num_hops):
        moving = pi * cont
        absorbed = absorbed + (pi - moving) + moving * stop_null
        pi = torch.matmul(moving / norm, transitions)
        if visible is not None:
            pi = pi * visible
        if trace is not None:
            trace.append(pi)
        if return_all:
            outs.append(absorbed + pi)
    return outs if return_all else absorbed + pi


@_full_precision
def absorbing_walk_resolvent(start: torch.Tensor, transitions: torch.Tensor, cont: torch.Tensor,
                             null: torch.Tensor, *, allowed: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Closed form of the query-window walk at K = infinity: a_i (I - C P(i))^{-1} S(i).

    One dense solve per query row, O(R S^3): a reference for tests and small problems, not a fast path.
    """
    B, H, R, S = start.shape
    if null.dim() == 1:
        null = null.view(1, -1, 1, 1)
    null = null.to(start.dtype)
    visible = (torch.ones(1, 1, R, S, dtype=start.dtype, device=start.device) if allowed is None
               else allowed.to(start.dtype))
    visible = visible.expand(B, H, R, S)
    norm = torch.matmul(visible, transitions.transpose(-1, -2)) + null         # (B, H, R, S)
    # P(i)_jl = E_jl * [l in W_i] / norm[i, j]
    P = transitions.unsqueeze(2) * visible.unsqueeze(-2) / norm.unsqueeze(-1)   # (B, H, R, S, S)
    M = torch.eye(S, dtype=start.dtype, device=start.device) - cont.view(B, H, 1, S, 1) * P
    visits = torch.linalg.solve(M, start.unsqueeze(-2), left=False).squeeze(-2)
    return visits * (1.0 - cont.unsqueeze(-2) + cont.unsqueeze(-2) * null / norm)


@_full_precision
def chacal_resolvent(weights: torch.Tensor, gamma: torch.Tensor, rows: Optional[torch.Tensor] = None,
                     lower_triangular: bool = True) -> torch.Tensor:
    """ChaCAL-style resolvent (1 - gamma) A (I - gamma A)^{-1}, exact.

    weights (B, H, S, S): the square attention matrix (row j = position j's attention). ``rows`` (B, R)
    selects query rows; None returns all of them. With ``lower_triangular`` (causal attention) the solve
    is one triangular substitution; otherwise a general solve. Rows sum to 1 where A's rows do.
    """
    n = weights.size(-1)
    if weights.size(-2) != n:
        raise ValueError("the ChaCAL-style resolvent needs a square attention matrix")
    g = gamma.view(1, -1, 1, 1).to(weights.dtype)
    system = torch.eye(n, dtype=weights.dtype, device=weights.device) - g * weights
    rhs = weights if rows is None else _gather_rows(weights, rows)
    if lower_triangular:
        solved = torch.linalg.solve_triangular(system, rhs, upper=False, left=False)
    else:
        solved = torch.linalg.solve(system, rhs, left=False)
    return (1.0 - g) * solved


def _gather_rows(matrix: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """(B, H, n, m) -> (B, H, R, m) at query rows (B, R)."""
    B, H, _, m = matrix.shape
    R = rows.shape[1]
    return matrix.gather(2, rows.view(B, 1, R, 1).expand(B, H, R, m))


# ---------------------------------------------------------------------------- module-level glue
def _walk_state(attn: nn.Module, key_input: torch.Tensor, address: torch.Tensor):
    """Pointer (B, H, S, d), per-head address (B, H, S, d) inputs and gate logits (B, H, S) of the keys."""
    B, S, _ = key_input.shape
    pointer = attn.walk_pointer(key_input).view(B, S, attn.num_heads, attn.head_dim).transpose(1, 2)
    gate = attn.walk_gate(key_input).transpose(1, 2)
    return _wide(pointer), _wide(address), _wide(gate)


def _merge_cached_walk_state(attn: nn.Module, state, new_positions: torch.Tensor, key_positions: torch.Tensor):
    """Keep pointer / address / gate of cached positions next to the KV cache and align them with the keys.

    The store lives on the ``KVCache`` object, so it disappears with it. It is keyed by absolute position:
    new tokens overwrite equal positions, entries the cache no longer holds are dropped after the step.
    """
    cache = attn.kv_cache
    pointer, address, gate = (t.detach() for t in state)
    new_positions = new_positions.to(device=pointer.device, dtype=torch.long)
    store = getattr(cache, "_operator_state", None)
    usable = (store is not None and store["positions"].device == pointer.device
              and store["pointer"].shape[0] == pointer.shape[0] and store["pointer"].dtype == pointer.dtype)
    # KVCache.update() has already run for this step: a cache that holds only the new positions was empty.
    if usable and cache.attention_positions.numel() > new_positions.numel():
        keep = ~torch.isin(store["positions"], new_positions)
        positions = torch.cat((store["positions"][keep], new_positions))
        pointer = torch.cat((store["pointer"][:, :, keep], pointer), dim=2)
        address = torch.cat((store["address"][:, :, keep], address), dim=2)
        gate = torch.cat((store["gate"][:, :, keep], gate), dim=2)
        order = torch.argsort(positions)
        positions = positions[order]
        pointer, address, gate = pointer[:, :, order], address[:, :, order], gate[:, :, order]
    else:
        positions = new_positions
    key_positions = key_positions.to(device=pointer.device, dtype=torch.long)
    index = torch.searchsorted(positions, key_positions)
    if (positions.numel() == 0 or int(index.max()) >= positions.numel()
            or not torch.equal(positions[index], key_positions)):
        raise RuntimeError(
            "AWA walk state is out of sync with the KV cache (cached keys without pointer/gate state). "
            "Call clear_kv_cache() and decode the sequence again from its first token.")
    pointer, address, gate = pointer[:, :, index], address[:, :, index], gate[:, :, index]
    kept = torch.isin(key_positions, cache.positions.to(key_positions.device))
    cache._operator_state = {"positions": key_positions[kept], "pointer": pointer[:, :, kept],
                             "address": address[:, :, kept], "gate": gate[:, :, kept]}
    return pointer, address, gate


def _allowed_from_mask(add_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if add_mask is None else add_mask > MASK_BLOCK_THRESHOLD


def operator_weights_transform(attn: nn.Module, key_input: torch.Tensor, address: torch.Tensor,
                               add_mask: Optional[torch.Tensor], causal: bool, cache_active: bool,
                               query_positions: torch.Tensor, key_positions: torch.Tensor,
                               ) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the map A -> effective weights that ``MultiHeadAttention.forward`` applies after the softmax.

    key_input (B, S_new, D): the key-side input of this call; address (B, H_kv, S_new, d): its projected
    keys after QK-norm and before the GQA expand and RoPE; add_mask: the additive mask the softmax uses;
    key_positions: absolute positions of all keys of this step (cached + new).
    """
    operator = attn.attention_operator
    square = (not cache_active and query_positions.numel() == key_positions.numel()
              and torch.equal(query_positions, key_positions))
    if operator == "chacal":
        if cache_active:
            raise NotImplementedError(
                "attention_operator='chacal' has no KV-cache path (it needs every position's attention row); "
                "call the model with use_cache=False")
        if not square:
            raise ValueError("attention_operator='chacal' is defined for self-attention only")
        gamma = _wide(chacal_gamma(attn))
        return lambda weights: chacal_resolvent(weights, gamma, lower_triangular=bool(causal))
    if operator not in WALK_OPERATORS:
        raise ValueError(f"unknown attention operator {operator!r}")
    state = _walk_state(attn, key_input, address)
    if cache_active:
        if operator == "awa_own":
            raise NotImplementedError(
                "attention_operator='awa_own' has no KV-cache path (it needs every position's own window); "
                "call the model with use_cache=False")
        new_positions = key_positions[-key_input.shape[1]:] if query_positions.numel() != key_input.shape[1] \
            else query_positions
        state = _merge_cached_walk_state(attn, state, new_positions, key_positions)
    elif operator == "awa_own" and not square:
        raise ValueError("attention_operator='awa_own' is defined for self-attention only")
    pointer, address, gate = state
    if attn.num_kv_groups > 1:
        address = address.repeat_interleave(attn.num_kv_groups, dim=1)
    transitions = transition_weights(pointer, address, attn.walk_score_cap, attn.walk_exclude_self)
    cont = _wide(walk_lambda(attn)).view(1, -1, 1) * torch.sigmoid(gate)
    null = torch.exp(_wide(attn.walk_tau))
    allowed = _allowed_from_mask(add_mask)
    hops = int(attn.walk_num_hops)
    if operator == "awa":
        return lambda weights: absorbing_walk(weights, transitions, cont, null, hops, allowed=allowed)
    own = allowed if allowed is not None else torch.ones(1, 1, 1, 1, dtype=torch.bool, device=transitions.device)
    return lambda weights: absorbing_walk(weights, transitions, cont, null, hops, own_window=own)


# --------------------------------------------------------------------------------- rows-only path
def _rows_tensor(rows: torch.Tensor, batch: int, length: int) -> torch.Tensor:
    if rows.dtype not in (torch.int32, torch.int64):
        raise ValueError("rows must be an integer tensor")
    rows = rows.long()
    if rows.dim() == 1:
        rows = rows.unsqueeze(0).expand(batch, -1)
    if rows.dim() != 2 or rows.shape[0] != batch or rows.shape[1] == 0:
        raise ValueError("rows must have shape [R] or [B, R] with R >= 1")
    if int(rows.min()) < 0 or int(rows.max()) >= length:
        raise ValueError("rows out of range")
    return rows


def attention_rows(attn: nn.Module, x: torch.Tensor, rows: torch.Tensor, *, mask: Optional[torch.Tensor] = None,
                   causal_mask: bool = True, num_hops: Optional[int] = None, return_all_hops: bool = False,
                   apply_dropout: bool = True, trace: Optional[List[torch.Tensor]] = None):
    """Self-attention output of ``attn`` at the positions ``rows`` only: (B, R, D).

    x (B, S, D) is the layer input (queries, keys and values all come from it); ``rows`` is [R] or [B, R].
    Same computation as ``MultiHeadAttention.forward(x, x, x, mask, causal_mask)`` restricted to those
    rows, for every attention operator, with GQA, QK-norm, RoPE, the sliding window, the logit cap and
    ``mask`` (the formats ``forward`` accepts for a full self-attention call). With ``return_all_hops``
    (awa / awa_own) a list of outputs at walk budgets 0..num_hops is returned from one pass.
    """
    B, S, D = x.shape
    rows = _rows_tensor(rows, B, S)
    R = rows.shape[1]
    H, Hk, d = attn.num_heads, attn.num_kv_heads, attn.head_dim
    operator = getattr(attn, "attention_operator", STANDARD)
    if return_all_hops and operator not in WALK_OPERATORS:
        raise ValueError("return_all_hops needs a walk operator (awa or awa_own)")
    full_queries = operator == "chacal"                       # the resolvent needs the square matrix
    xq = x if full_queries else x.gather(1, rows.unsqueeze(-1).expand(B, R, D))
    q = attn.query_proj(xq).view(B, -1, H, d).transpose(1, 2)
    k = attn.key_proj(x).view(B, S, Hk, d).transpose(1, 2)
    v = attn.value_proj(x).view(B, S, Hk, d).transpose(1, 2)
    if attn.use_qk_norm and attn.q_norm is not None and attn.k_norm is not None:
        q, k = attn.q_norm(q), attn.k_norm(k)
    address = k
    if attn.num_kv_groups > 1:
        k = k.repeat_interleave(attn.num_kv_groups, dim=1)
        v = v.repeat_interleave(attn.num_kv_groups, dim=1)
    key_pos = torch.arange(S, device=x.device)
    if attn.use_rope and attn.positional_encoding is not None:
        q = attn.positional_encoding.apply_rotary_pos_emb(q, key_pos if full_queries else rows)
        k = attn.positional_encoding.apply_rotary_pos_emb(k, key_pos)

    # additive mask of every position (B|1, 1|H, S, S), then the requested rows
    blocked = torch.zeros(1, 1, S, S, dtype=torch.bool, device=x.device)
    if attn.sliding_window is not None:
        blocked = blocked | ((key_pos.view(S, 1) - key_pos.view(1, S)) >= attn.sliding_window)
    if causal_mask:
        blocked = blocked | (key_pos.view(1, S) > key_pos.view(S, 1))
    full_mask = torch.zeros(1, 1, S, S, dtype=torch.float32, device=x.device).masked_fill(blocked, float("-inf"))
    if mask is not None:
        prepared = attn._prepare_attention_mask(mask, B, S, S, x.device)
        if prepared.dim() == 4 and prepared.size(2) == 1:
            prepared = prepared.expand(prepared.size(0), prepared.size(1), S, S)
        full_mask = full_mask + prepared
    row_index = rows.view(B, 1, R, 1)

    def _rows_of(matrix: torch.Tensor) -> torch.Tensor:
        matrix = matrix.expand(B, matrix.size(1), S, S)
        return matrix.gather(2, row_index.expand(B, matrix.size(1), R, S))

    all_weights: List[torch.Tensor] = []
    transform = None
    if operator == "chacal":
        gamma = _wide(chacal_gamma(attn))
        transform = lambda weights: chacal_resolvent(weights, gamma, rows, lower_triangular=bool(causal_mask))  # noqa: E731
        sdpa_mask = full_mask
    else:
        sdpa_mask = _rows_of(full_mask)
        if operator in WALK_OPERATORS:
            pointer, address, gate = _walk_state(attn, x, address)
            if attn.num_kv_groups > 1:
                address = address.repeat_interleave(attn.num_kv_groups, dim=1)
            transitions = transition_weights(pointer, address, attn.walk_score_cap, attn.walk_exclude_self)
            cont = _wide(walk_lambda(attn)).view(1, -1, 1) * torch.sigmoid(gate)
            null = torch.exp(_wide(attn.walk_tau))
            hops = int(attn.walk_num_hops if num_hops is None else num_hops)
            window = dict(allowed=_allowed_from_mask(sdpa_mask)) if operator == "awa" \
                else dict(own_window=_allowed_from_mask(full_mask))

            def transform(weights: torch.Tensor) -> torch.Tensor:
                outs = absorbing_walk(weights, transitions, cont, null, hops, return_all=True, trace=trace,
                                      **window)
                all_weights.extend(outs)
                return outs[-1]
    out, _ = attn._standard_sdpa_forward(q, k, v, sdpa_mask, 1.0, apply_dropout, transform)

    def _project(heads: torch.Tensor) -> torch.Tensor:
        return attn.out_proj(heads.transpose(1, 2).contiguous().view(B, R, D))

    if not return_all_hops:
        return _project(out)
    results = []
    for weights in all_weights:
        if apply_dropout and attn.training:
            weights = attn.dropout(weights)
        results.append(_project(torch.matmul(weights.to(v.dtype), v)))
    return results


def layer_forward_rows(layer: nn.Module, h: torch.Tensor, rows: torch.Tensor, *,
                       mask: Optional[torch.Tensor] = None, causal_mask: bool = True,
                       num_hops: Optional[int] = None, return_all_hops: bool = False,
                       trace: Optional[List[torch.Tensor]] = None):
    """One ``TransformerEncoderLayer`` evaluated only at ``rows``: (B, R, D).

    h (B, S, D) is the layer's full input. Equal to ``layer(h, mask, causal_mask)[0]`` at those rows
    (dropout and stochastic depth draw their own noise in training mode). MoE layers are not supported.
    With ``return_all_hops`` a list over walk budgets 0..num_hops is returned.
    """
    if layer.use_moe:
        raise NotImplementedError("rows-only evaluation is not implemented for MoE layers")
    B, S, D = h.shape
    rows = _rows_tensor(rows, B, S)
    R = rows.shape[1]
    index = rows.unsqueeze(-1).expand(B, R, D)
    h_rows = h.gather(1, index)
    x = layer.norm1(h) if layer.pre_norm else h
    attended = attention_rows(layer.attn, x, rows, mask=mask, causal_mask=causal_mask, num_hops=num_hops,
                              return_all_hops=return_all_hops, trace=trace)

    def _residual(branch: torch.Tensor) -> torch.Tensor:
        return layer._stochastic_depth(layer.dropout(branch), layer.drop_path_rate, layer.training)

    def _finish(attn_out: torch.Tensor) -> torch.Tensor:
        if layer.pre_norm and layer.parallel_residual:
            return h_rows + _residual(attn_out) + _residual(layer.ffn(x.gather(1, index)))
        if layer.pre_norm:
            y = h_rows + _residual(attn_out)
            return y + _residual(layer.ffn(layer.norm2(y)))
        y = layer.norm1(h_rows + _residual(attn_out))
        return layer.norm2(y + _residual(layer.ffn(y)))

    if return_all_hops:
        return [_finish(o) for o in attended]
    return _finish(attended)


__all__ = [
    "LAMBDA_MAX", "GAMMA_MAX", "ATTENTION_OPERATORS", "WALK_OPERATORS", "DEFAULT_OPERATOR_OPTIONS",
    "MASK_BLOCK_THRESHOLD", "resolve_operator_config", "attach_operator", "walk_lambda", "chacal_gamma",
    "continue_probability", "transition_weights", "absorbing_walk", "absorbing_walk_resolvent",
    "chacal_resolvent", "operator_weights_transform", "attention_rows", "layer_forward_rows",
]
