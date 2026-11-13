# -*- coding: utf-8 -*-
"""Arms of the AWA-3 study (PROTOCOL section 6): model construction, answer-position forwards, analytic cost.

Every arm is a tiny Cevahir model built with ``tiny_model_config`` + ``ModelManager.initialize`` (D = 128,
4 heads, 4 KV heads, 2 layers, SwiGLU, RMSNorm pre-norm, RoPE, tied embeddings, dropout 0, fp32). The operators
(AWA, AWA-own, ChaCAL-style) are the engine's attention operators
(``src/neural_network_module/ortak_katman_module/attention_manager_module/absorbing_walk.py``), switched on for
the last layer through the model configuration; this file adds no parameters of its own.

Every arm computes its final layer only at the answer position (``CevahirNeuralNetwork.forward_rows`` /
``layer_forward_rows``), which equals the full forward at that position.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch

from src.neural_network_module.ortak_katman_module.attention_manager_module.absorbing_walk import (
    chacal_gamma,
    layer_forward_rows,
    walk_lambda,
)

from . import task

D_MODEL, N_HEADS, N_LAYERS, MAX_LEN = 128, 4, 2, 128
TRAIN_BUDGETS = (3, 4, 5)            # K_train (walks) and T_train (loops) are drawn uniformly from this set
COMPETENCE_BUDGET = 5
TEST_BUDGET = 12                     # preregistered K_test (walks) and T_test (loops)
PLATEAU_BUDGET = 24                  # walks: plateau budget
BUDGET_AXIS = {"walk": list(range(0, 25)), "looped": list(range(1, 17)), "arloop": list(range(1, 25)),
               "exact": [0], "std": [0]}

# name -> final-layer computation ("forward"), engine attention operator, ffn_dim and the asserted parameter count.
# ffn_dim 534 matches the 16,908 parameters the walk adds to a standard model (D^2 + H(D + 1) + 2H).
ARMS: Dict[str, dict] = {
    "AWA-2L": dict(forward="walk", operator="awa", ffn=None, params=551_308, seeds=5,
                   role="treatment: prefix-closed (query-window) walk"),
    "ChaCAL-2L": dict(forward="exact", operator="chacal", ffn=534, params=551_300, seeds=5,
                      role="closest published operator (ChaCAL-style resolvent, exact)"),
    "AWA-own-2L": dict(forward="walk", operator="awa_own", ffn=None, params=551_308, seeds=5,
                       role="single-ingredient ablation: own-window walk"),
    "LOOPED-2L": dict(forward="looped", operator="standard", ffn=534, params=551_296, seeds=4,
                      role="binding layer + weight-tied block applied T times to all positions"),
    "ARLOOP-2L": dict(forward="arloop", operator="standard", ffn=534, params=551_296, seeds=4,
                      role="final layer re-applied T times at the answer row"),
    "STD-2L-pm": dict(forward="std", operator="standard", ffn=534, params=551_296, seeds=3,
                      role="parameter-matched standard model: negative control for assumption B"),
}
# launch / priority order of the protocol (also the order of the extension reserve, STD is exempt)
ARM_ORDER = ("AWA-2L", "ChaCAL-2L", "AWA-own-2L", "LOOPED-2L", "ARLOOP-2L", "STD-2L-pm")
PILOT_ARMS = ("AWA-2L", "AWA-own-2L", "ChaCAL-2L", "LOOPED-2L", "ARLOOP-2L")
LOWER_TRIANGULAR_ARMS = ("ChaCAL-2L", "AWA-own-2L")


def competence_stratum(arm: str) -> str:
    """In-distribution cells an arm is judged on: lower-triangular arms can only represent sorted chains."""
    return "sorted" if arm in LOWER_TRIANGULAR_ARMS else "shuffled"


@contextmanager
def _quiet():
    names = ("ModelManager", "model_initializer", "CevahirNeuralNetwork", "MultiHeadAttention",
             "TransformerEncoderLayer", "KVCache", "PositionalEncoding", "LanguageEmbedding", "RMSNorm",
             "FeedForwardNetwork", "QuantizationManager")
    loggers = [logging.getLogger(n) for n in names]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for lg, level in zip(loggers, levels):
            lg.setLevel(level)


def arm_config(arm: str) -> dict:
    """The full model configuration of an arm (a plain dict; the engine normalizes and validates it)."""
    from model_management.config_schema import tiny_model_config

    spec = ARMS[arm]
    extra = {}
    if spec["operator"] != "standard":
        extra = {"attention_operator": spec["operator"],
                 "attention_operator_options": {"layers": [N_LAYERS - 1], "num_hops": 4}}
    return tiny_model_config(vocab_size=task.VOCAB, embed_dim=D_MODEL, seq_proj_dim=D_MODEL, num_heads=N_HEADS,
                             num_kv_heads=N_HEADS, num_layers=N_LAYERS, ffn_dim=spec["ffn"],
                             max_seq_length=MAX_LEN, max_cache_len=MAX_LEN, dropout=0.0,
                             use_gradient_checkpointing=False, **extra)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_architecture(model, arm: str) -> None:
    """Raise if the live model is not the architecture the protocol states for this arm."""
    spec = ARMS[arm]
    problems = []
    if len(model.layers) != N_LAYERS:
        problems.append(f"{len(model.layers)} layers")
    emb = model.embedding.embedding.weight
    if tuple(emb.shape) != (task.VOCAB, D_MODEL) or model.output_layer.weight is not emb:
        problems.append("embedding shape or weight tying")
    if any(float(m.p) != 0.0 for m in model.modules() if isinstance(m, torch.nn.Dropout)):
        problems.append("non-zero dropout")
    if model.pos_encoding.mode != "rope" or not model.causal_mask:
        problems.append("positional encoding or causality")
    for i, layer in enumerate(model.layers):
        a = layer.attn
        want = spec["operator"] if i == N_LAYERS - 1 else "standard"
        if a.attention_operator != want:
            problems.append(f"layer {i}: operator {a.attention_operator!r}, wanted {want!r}")
        if (a.num_heads, a.num_kv_heads, a.head_dim) != (N_HEADS, N_HEADS, D_MODEL // N_HEADS):
            problems.append(f"layer {i}: heads")
        if (not layer.pre_norm or layer.parallel_residual or layer.use_moe or layer.use_gradient_checkpointing
                or layer.drop_path_rate != 0.0 or type(layer.norm1).__name__ != "RMSNorm"
                or a.sliding_window is not None or a.use_qk_norm or a.attn_logit_cap != 0.0 or not a.use_rope):
            problems.append(f"layer {i}: block options")
        if spec["ffn"] is not None and layer.ffn.ffn_dim != spec["ffn"]:
            problems.append(f"layer {i}: ffn_dim {layer.ffn.ffn_dim}")
    n_params = count_parameters(model)
    if n_params != spec["params"]:
        problems.append(f"{n_params} parameters, the protocol states {spec['params']}")
    if problems:
        raise ValueError(f"{arm}: built model does not match the protocol: " + "; ".join(problems))


def build(arm: str, seed: int):
    """Build one arm. ``torch.manual_seed(seed)`` is called before the build (inside a forked RNG scope, so the
    caller's global stream is untouched); modules of equal shape start identical across arms for a given seed,
    because the engine draws operator parameters after every other weight. Returns (model, effective config).
    """
    from model_management.model_manager import ModelManager

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; arms: {sorted(ARMS)}")
    with _quiet(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        mm = ModelManager(dict(arm_config(arm)))
        mm.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)
    model = mm.model
    check_architecture(model, arm)
    return model, mm.effective_config


def _embed(model, x: torch.Tensor) -> torch.Tensor:
    return model.embed_dropout(model.pos_encoding(model.embedding(x)))


def answer_logits(model, arm: str, x: torch.Tensor, budget: Optional[int] = None, return_all: bool = False):
    """Logits at the answer position (the last token) of every item: (B, V).

    walk    AWA / AWA-own with ``budget`` = K walk steps; return_all gives the list for K = 0..budget.
    exact   ChaCAL-style resolvent (no budget).
    std     standard model (no budget).
    looped  layer 0, then layer 1 applied ``budget`` = T times to all positions (the last application only at
            the answer row); return_all gives the list for T = 1..budget.
    arloop  layer 0, then layer 1 applied T times at the answer row only (other positions keep their layer-0
            state); return_all gives the list for T = 1..budget.
    """
    forward = ARMS[arm]["forward"]
    rows = torch.tensor([x.shape[1] - 1], device=x.device)
    if forward in ("std", "exact"):
        if return_all:
            raise ValueError(f"{arm} has no budget axis")
        return model.forward_rows(x, rows)[:, 0]
    if budget is None:
        raise ValueError(f"{arm} needs a budget")
    if forward == "walk":
        out = model.forward_rows(x, rows, num_hops=int(budget), return_all_hops=return_all)
        return [o[:, 0] for o in out] if return_all else out[:, 0]
    if budget < 1:
        raise ValueError("loop arms need T >= 1")
    binding, block = model.layers
    h = binding(_embed(model, x), causal_mask=True)[0]
    B, _, D = h.shape
    index = rows.view(1, 1, 1).expand(B, 1, D)
    outs: List[torch.Tensor] = []
    if forward == "looped":
        if return_all:                                           # full applications; the answer row is read out
            for _ in range(int(budget)):
                h = block(h, causal_mask=True)[0]
                outs.append(model._output_head(h.gather(1, index))[:, 0])
            return outs
        for _ in range(int(budget) - 1):
            h = block(h, causal_mask=True)[0]
        return model._output_head(layer_forward_rows(block, h, rows))[:, 0]
    if forward == "arloop":
        for t in range(int(budget)):
            y = layer_forward_rows(block, h, rows)
            if return_all:
                outs.append(model._output_head(y)[:, 0])
            if t < int(budget) - 1:
                h = h.scatter(1, index, y)
        return outs if return_all else model._output_head(y)[:, 0]
    raise ValueError(f"unknown forward {forward!r}")


def operator_readout(model, arm: str) -> dict:
    """lambda per head (walk arms) or gamma per head (ChaCAL-style); empty for the other arms."""
    attn = model.layers[-1].attn
    with torch.no_grad():
        if ARMS[arm]["forward"] == "walk":
            return {"lambda": [float(v) for v in walk_lambda(attn)], "tau": [float(v) for v in attn.walk_tau]}
        if ARMS[arm]["forward"] == "exact":
            return {"gamma": [float(v) for v in chacal_gamma(attn)]}
    return {}


# ------------------------------------------------------------------------------ analytic cost model
def macs_per_answer(arm: str, budget: int = 0, n: int = task.SEQ_LEN, D: int = D_MODEL, H: int = N_HEADS,
                    vocab: int = task.VOCAB, ffn: Optional[int] = None) -> int:
    """Analytic multiply-accumulate count for ONE answer (PROTOCOL section 9, H6(ii)).

    It counts the arithmetic the operator needs for an exact single-answer evaluation, not the batched driver:
    linear maps (in x out per application), attention scores and value mixing (n * D per query row), the SwiGLU
    FFN (3 * D * F per position), and for the walk the pointer / gate projections, E (n^2 * D), the window
    normaliser and K steps of (1 x n) @ (n x n) per head. ARLOOP re-projects only the changed answer row after
    the first application. Norms, RoPE, softmax and additions are not counted (same in every arm up to O(n D)).
    """
    spec = ARMS[arm]
    F = _ffn_dim(arm) if ffn is None else ffn
    full_layer = n * (4 * D * D + 2 * n * D + 3 * D * F)        # q, k, v, out for all rows; scores + mix; FFN
    row_tail = D * D + 2 * n * D + D * D + 3 * D * F             # q, scores, mix, out, FFN for one row
    rows_layer = 2 * n * D * D + row_tail                        # + k, v projections of all positions
    head = vocab * D
    forward = spec["forward"]
    if forward == "std":
        return full_layer + rows_layer + head
    if forward == "exact":
        # all query rows are needed for A (n * D^2 + n^2 * D scores), then one triangular solve for one row
        return full_layer + rows_layer + (n - 1) * D * D + (n - 1) * n * D + H * n * n + head
    if forward == "walk":
        pointer_gate = n * D * D + n * D * H
        transitions = n * n * D                                   # E: H heads x n^2 x d
        normaliser = H * n * n
        return full_layer + rows_layer + pointer_gate + transitions + normaliser + budget * H * n * n + head
    if budget < 1:
        raise ValueError("loop arms need T >= 1")
    if forward == "looped":
        return full_layer + (budget - 1) * full_layer + rows_layer + head
    if forward == "arloop":
        refresh = 2 * D * D + row_tail                            # re-project k, v of the answer row only
        return full_layer + rows_layer + (budget - 1) * refresh + head
    raise ValueError(forward)


def _ffn_dim(arm: str) -> int:
    from src.neural_network_module.architecture_contracts import resolve_ffn_dim

    return resolve_ffn_dim(D_MODEL, ARMS[arm]["ffn"], gated=True)
