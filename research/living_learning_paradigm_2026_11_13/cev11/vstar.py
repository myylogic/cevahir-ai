# -*- coding: utf-8 -*-
"""Hedef değer (v*) araması (CEV-11 PROTOCOL §8).

``search_delta`` optimises one vector δ ∈ R^d that is substituted into the fc2 output of one layer at each prompt's
key position: the output there becomes ``v0_i + δ``, where ``v0_i`` is that row's own unedited fc2 output (including
any FFNMemory already installed, so for state Q the search runs against the current model). This is ROME's
"δ added at the subject position" search. Objective (PROTOCOL §8):

    loss = mean NLL(target | prompt) + kl_weight · KL(p_base ‖ p_edited)

with the NLL averaged over each prompt's target tokens and then over prompts (ROME's convention), and the KL taken
at the last position of the KL prompt (relation B in the lived frame), where δ is substituted at the KL prompt's
key position too (otherwise the term would be identically zero). Adam starts from δ = 0 with ``lr = step_size``; after
each step δ is projected onto the ball ``‖δ‖ ≤ cap``. The loop evaluates the current δ, stops early when the mean
NLL is below ``stop_nll``, and otherwise takes at most ``max_steps`` Adam steps; the returned δ is always the last
evaluated one, so ``info["nll"]`` is the NLL of the returned δ. No randomness is involved; at fixed threads and
shapes two calls give bitwise identical results (V7).

Write convention: with the key ``k`` used for the write (``k̄`` for STD), ``v* = W k + δ`` (``v_star`` below), so
the ROME write ``rome_update(W, C0, k, v*) = δ (C0⁻¹k)ᵀ / (kᵀC0⁻¹k)`` adds exactly δ at ``k``. PARA-ORTH "uses the
same v* as STD" (PROTOCOL §8): pass ``v* = W k̄_STD + δ_STD`` together with the PARA-ORTH key to ``rome_update``.
For the lives, ``v*`` is relative to the model the search ran on: ``W0`` for states I and J, the current weights
for state Q (``baselines.*State.v_star`` and ``memory.Ledger.residual`` compute it).

Additions to INTERFACES.md (backward compatible):
* ``kl_prompt_ids`` may be ``ids`` or ``(ids, key_pos)``. With bare ids the KL row's key position is the first
  prompt's key position, and the KL prompt must share the first prompt's tokens up to and including it (true for
  relation B in the lived frame at ROOT/STD/LAST); otherwise a ``ValueError`` asks for ``(ids, key_pos)``
  (e.g. the PRED site, whose position is the KL prompt's own last token).
* ``target_ids`` may be one id list shared by all prompts or one id list per prompt (``[[...], [...]]``); the
  latter serves the entity-joint value of the lives (state J), optimised for relation A and C prompts at once.
* ``memo`` (dict) and ``memo_context`` (str, e.g. the key rule): results are memoised under a sha256 of the
  model state (weights + active FFNMemory hooks), layer, prompts, target, KL prompt, all hyper-parameters,
  ``memo_context``, torch version and thread count. A hit returns a copy with ``info["memo_hit"] = True``.
* ``info`` has, besides ``nll, steps, delta_norm, stopped`` (True iff the NLL criterion stopped the loop):
  ``kl``, ``loss``, ``nll0`` (NLL at δ = 0), ``stop_reason``, ``cap``, ``step_size``, ``memo_hit``, ``memo_key``.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from tokenizer_management.byte_bpe import PAD_ID

from .memory import FFNMemory, _as_batch, _fc2, state_fingerprint

VSTAR_VERSION = "cev11-vstar/1"

Prompt = Tuple[Sequence[int], int]
KLPrompt = Union[Sequence[int], Tuple[Sequence[int], int], None]


# ---------------------------------------------------------------------------------------------------- objective
def _parse_kl_prompt(kl_prompt: KLPrompt, first: Optional[Prompt]) -> Optional[Tuple[List[int], Optional[int]]]:
    """(ids, key_pos) of the KL prompt; key_pos is None when no first prompt is available (FT baselines)."""
    if kl_prompt is None:
        return None
    if (isinstance(kl_prompt, tuple) and len(kl_prompt) == 2 and isinstance(kl_prompt[0], (list, tuple))
            and isinstance(kl_prompt[1], int)):
        ids, pos = list(map(int, kl_prompt[0])), int(kl_prompt[1])
        if not 0 <= pos < len(ids):
            raise ValueError(f"KL key position {pos} outside the KL prompt (length {len(ids)})")
        return ids, pos
    ids = list(map(int, kl_prompt))   # type: ignore[arg-type]
    if not ids:
        raise ValueError("empty KL prompt")
    if first is None:
        return ids, None
    p_ids, p_pos = list(first[0]), int(first[1])
    if p_pos >= len(ids) or ids[:p_pos + 1] != p_ids[:p_pos + 1]:
        raise ValueError("the KL prompt does not share the first prompt's tokens up to its key position; "
                         "pass kl_prompt_ids=(ids, key_pos)")
    return ids, p_pos


def _parse_targets(target_ids: Sequence, n_prompts: int) -> List[List[int]]:
    """One target id list per prompt, from a shared list or a list of per-prompt lists."""
    items = list(target_ids)
    if not items:
        raise ValueError("empty target")
    if isinstance(items[0], (list, tuple)):
        if len(items) != n_prompts:
            raise ValueError(f"{len(items)} targets for {n_prompts} prompts")
        out = [[int(t) for t in tg] for tg in items]
    else:
        out = [[int(t) for t in items]] * n_prompts
    if any(not tg for tg in out):
        raise ValueError("empty target")
    return out


class TeacherForcedBatch:
    """Right-padded batch of ``prompt + target[:-1]`` rows (plus an optional KL prompt row).

    Logits at ``len(prompt) − 1 + j`` predict ``target[j]``. Right padding cannot affect earlier positions of a
    causal model. ``key_pos`` holds one key position per row (None where a row has none).
    """

    def __init__(self, prompts: Sequence, target_ids: Sequence, kl_prompt: KLPrompt = None,
                 with_positions: bool = True):
        if not prompts:
            raise ValueError("no prompts")
        self.targets = _parse_targets(target_ids, len(prompts))
        rows: List[List[int]] = []
        self.key_pos: List[Optional[int]] = []
        tr, tp, ti = [], [], []
        parsed: List[Tuple[List[int], Optional[int]]] = []
        for pr in prompts:
            is_pair = (isinstance(pr, tuple) and len(pr) == 2 and isinstance(pr[0], (list, tuple))
                       and isinstance(pr[1], int))
            if with_positions:
                if not is_pair:
                    raise ValueError("prompts must be (ids, key_pos) pairs")
                ids, pos = list(map(int, pr[0])), int(pr[1])
                if not ids or not 0 <= pos < len(ids):
                    raise ValueError(f"key position {pos} outside a prompt of length {len(ids)}")
            else:
                ids = list(map(int, pr[0] if is_pair else pr))
                pos = None
                if not ids:
                    raise ValueError("empty prompt")
            parsed.append((ids, pos))
        self.slices: List[Tuple[int, int]] = []          # target tokens of prompt i: [a, b) of the flat arrays
        for i, ((ids, pos), tgt) in enumerate(zip(parsed, self.targets)):
            rows.append(ids + tgt[:-1])
            self.key_pos.append(pos)
            a = len(ti)
            for j, t in enumerate(tgt):
                tr.append(i)
                tp.append(len(ids) - 1 + j)
                ti.append(t)
            self.slices.append((a, len(ti)))
        self.n_prompts = len(parsed)
        first = parsed[0] if with_positions else None
        kl = _parse_kl_prompt(kl_prompt, first)    # type: ignore[arg-type]
        self.kl_row: Optional[int] = None
        self.kl_last: Optional[int] = None
        self.kl: Optional[Tuple[List[int], Optional[int]]] = kl
        if kl is not None:
            rows.append(kl[0])
            self.key_pos.append(kl[1] if with_positions else None)
            self.kl_row, self.kl_last = len(rows) - 1, len(kl[0]) - 1
        self.prompts = parsed
        self.X = _as_batch(rows)
        self.tgt_rows = torch.tensor(tr, dtype=torch.long)
        self.tgt_pos = torch.tensor(tp, dtype=torch.long)
        self.tgt_ids = torch.tensor(ti, dtype=torch.long)

    def nll(self, logits: torch.Tensor) -> torch.Tensor:
        """Mean over prompts of the mean NLL of that prompt's target tokens."""
        tok = F.cross_entropy(logits[self.tgt_rows, self.tgt_pos].float(), self.tgt_ids, reduction="none")
        return torch.stack([tok[a:b].mean() for a, b in self.slices]).mean()

    def base_logp(self, logits: torch.Tensor) -> Optional[torch.Tensor]:
        if self.kl_row is None:
            return None
        return F.log_softmax(logits[self.kl_row, self.kl_last].float(), dim=-1).detach()

    def kl_term(self, logits: torch.Tensor, base_logp: Optional[torch.Tensor]) -> torch.Tensor:
        """KL(p_base ‖ p_edited) = Σ p_base (log p_base − log p_edited) at the KL prompt's last position."""
        if base_logp is None or self.kl_row is None:
            return logits.new_zeros(())
        logp = F.log_softmax(logits[self.kl_row, self.kl_last].float(), dim=-1)
        return F.kl_div(logp, base_logp, log_target=True, reduction="sum")


def _check_hparams(step_size: float, cap: float, max_steps: int, stop_nll: float, kl_weight: float) -> None:
    if not (step_size > 0 and math.isfinite(step_size)):
        raise ValueError("step_size must be a positive finite number")
    if not cap > 0 or math.isnan(cap):
        raise ValueError("cap must be > 0")
    if not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a non-negative int")
    if not (math.isfinite(stop_nll) and math.isfinite(kl_weight) and kl_weight >= 0):
        raise ValueError("stop_nll and kl_weight must be finite, kl_weight >= 0")


def _memo_key(model, layer: int, batch: TeacherForcedBatch, step_size: float, cap: float, max_steps: int,
              stop_nll: float, kl_weight: float, memo_context: Optional[str]) -> str:
    payload = {
        "version": VSTAR_VERSION, "model": state_fingerprint(model), "layer": int(layer),
        "prompts": [[ids, pos] for ids, pos in batch.prompts], "targets": batch.targets,
        "kl": None if batch.kl is None else [batch.kl[0], batch.kl[1]],
        "step_size": float(step_size).hex(), "cap": float(cap).hex(), "max_steps": int(max_steps),
        "stop_nll": float(stop_nll).hex(), "kl_weight": float(kl_weight).hex(),
        "context": memo_context, "threads": torch.get_num_threads(), "torch": torch.__version__,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------------- search
def search_delta(model, layer: int, prompts: List[Prompt], target_ids: Sequence, kl_prompt_ids: KLPrompt,
                 step_size: float, cap: float, max_steps: int = 120, stop_nll: float = 0.05,
                 kl_weight: float = 1.0, memo: Optional[dict] = None,
                 memo_context: Optional[str] = None) -> Tuple[torch.Tensor, dict]:
    """v* search (see the module docstring). Returns ``(δ [d] float32, info)``; the model is left unchanged."""
    _check_hparams(step_size, cap, max_steps, stop_nll, kl_weight)
    batch = TeacherForcedBatch(prompts, target_ids, kl_prompt_ids if kl_weight > 0 else None, with_positions=True)
    key = None
    if memo is not None:
        key = _memo_key(model, layer, batch, step_size, cap, max_steps, stop_nll, kl_weight, memo_context)
        hit = memo.get(key)
        if hit is not None:
            return hit["delta"].clone(), dict(hit["info"], memo_hit=True)

    fc2 = _fc2(model, layer)
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    was_training = model.training
    sub: Optional[FFNMemory] = None
    handle = None
    try:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        rows = torch.arange(batch.X.shape[0])
        pos = torch.tensor(batch.key_pos, dtype=torch.long)
        # Base pass: unedited fc2 outputs at the key positions (after any installed memory) and base logits.
        captured: List[torch.Tensor] = []
        handle = fc2.register_forward_hook(lambda mod, inp, out: captured.append(out.detach()))
        with torch.no_grad():
            logits0 = model(batch.X)[0]
        handle.remove()
        handle = None
        if len(captured) != 1:
            raise RuntimeError(f"expected one fc2 call per forward, got {len(captured)}")
        v0 = captured[0][rows, pos].clone()                  # [R, d]
        base_logp = batch.base_logp(logits0)
        nll0 = float(batch.nll(logits0))

        sub = FFNMemory(model, layer)
        delta = torch.zeros(fc2.weight.shape[0], dtype=fc2.weight.dtype, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=step_size)
        steps = 0
        stopped = False
        while True:
            sub.substitute(batch.key_pos, v0 + delta)
            logits = model(batch.X)[0]
            nll = batch.nll(logits)
            kl = batch.kl_term(logits, base_logp)
            loss = nll + kl_weight * kl
            if float(nll.detach()) < stop_nll:
                stopped = True
                break
            if steps >= max_steps:
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
            with torch.no_grad():
                norm = delta.norm()
                if norm > cap:
                    delta.mul_(cap / norm)
        out = delta.detach().clone()
        info = {"nll": float(nll.detach()), "kl": float(kl.detach()), "loss": float(loss.detach()), "nll0": nll0,
                "steps": steps, "delta_norm": float(out.norm()), "stopped": stopped,
                "stop_reason": "nll" if stopped else "max_steps", "cap": float(cap),
                "step_size": float(step_size), "memo_hit": False, "memo_key": key}
    finally:
        if handle is not None:
            handle.remove()
        if sub is not None:
            sub.remove()
        for n, p in model.named_parameters():
            p.requires_grad_(flags[n])
        model.train(was_training)
        clear = getattr(model, "clear_kv_cache", None)
        if callable(clear):
            clear()
    if memo is not None:
        memo[key] = {"delta": out.clone(), "info": dict(info)}
    return out, info


def v_star(W: torch.Tensor, k: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """``v* = W k + δ`` in float64 (the value the ROME write installs at key ``k``)."""
    return W.detach().double() @ k.detach().double() + delta.detach().double()


@torch.no_grad()
def median_fc2_output_norm(model, layer: int, ids_batches: Iterable, exclude_ids: Sequence[int] = (PAD_ID,)) -> float:
    """Median ‖fc2 output‖ over non-excluded positions (for the shared cap: 8 × this, PROTOCOL §8).

    Measures the model as it is (installed memories included): call it on the base model.
    """
    fc2 = _fc2(model, layer)
    captured: List[torch.Tensor] = []
    handle = fc2.register_forward_hook(lambda mod, inp, out: captured.append(out.detach()))
    was_training = model.training
    model.eval()
    norms = []
    excl = torch.tensor(sorted(set(int(i) for i in exclude_ids)), dtype=torch.long)
    try:
        for b in ids_batches:
            x = _as_batch(b)
            captured.clear()
            model(x)
            keep = ~torch.isin(x, excl)
            norms.append(captured[0][keep].double().norm(dim=-1))
    finally:
        handle.remove()
        model.train(was_training)
    allv = torch.sort(torch.cat(norms)).values if norms else torch.zeros(0)
    n = allv.numel()
    if n == 0:
        raise ValueError("no positions")
    return float(allv[n // 2]) if n % 2 else float((allv[n // 2 - 1] + allv[n // 2]) / 2)


__all__ = ["search_delta", "v_star", "median_fc2_output_norm", "TeacherForcedBatch", "VSTAR_VERSION"]
