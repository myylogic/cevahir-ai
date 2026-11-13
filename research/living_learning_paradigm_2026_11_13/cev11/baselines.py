# -*- coding: utf-8 -*-
"""Karşılaştırma kolları ve yaşam durum değişkenleri (CEV-11 PROTOCOL §8, §13).

* ``ft_layer`` (FT-L): Adam on ``layers[l].ffn.fc2.weight`` only; after every step ``ΔW = W − W0`` is clamped to
  ``|ΔW_ij| ≤ eps``. Returns ``ΔW`` (float32) and restores ``W`` bitwise.
* ``ft_all`` (FT-ALL): Adam on every parameter (a dynamic-evaluation-style learner). Returns the new state_dict
  (cloned tensors) and restores the model bitwise.
  Both use the v* objective — mean target NLL + ``kl_weight`` · KL(p_base ‖ p_edited) at the last position of the
  KL prompt — and stop early when the mean NLL is below ``stop_nll``; the returned weights are the last evaluated
  ones (at most ``steps`` Adam steps). Parameters are selected by name, never by module type.
* ``normalize_informal`` (NORM): restores the apostrophe before a valid case allomorph for lexicon names.
* ``IndependentState`` (I) and ``SequentialState`` (Q): the two non-ledger state variants of the lives (the ledger
  state J is ``memory.Ledger`` with ``memory.EntityJointLedger``). Both hold float64 matrices ``M = W − W0`` for
  an ``FFNMemory`` hook (``delta()`` gives float32).

Additions to INTERFACES.md (backward compatible): ``ft_layer``/``ft_all`` take keyword arguments
``kl_prompt_ids=None, kl_weight=1.0, stop_nll=0.05, info=None`` (the protocol's "same objective as v*"; ``info``,
when a dict, receives ``nll, kl, steps, stopped``). ``prompts`` may be id lists or ``(ids, pos)`` pairs (the
position is ignored); ``target_ids`` may be shared or given per prompt, as in ``vstar.search_delta``.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from .memory import _f64, _fc2, rome_update
from .morph_tr import INFLECTED, formal, informal
from .vstar import KLPrompt, TeacherForcedBatch, _check_hparams


# ---------------------------------------------------------------------------------------------------- FT arms
def _prompt_ids(pr) -> List[int]:
    if isinstance(pr, tuple) and len(pr) == 2 and isinstance(pr[0], (list, tuple)) and isinstance(pr[1], int):
        return list(map(int, pr[0]))
    return list(map(int, pr))


def _kl_ids(kl_prompt: KLPrompt) -> Optional[List[int]]:
    if kl_prompt is None:
        return None
    return _prompt_ids(kl_prompt)


def _finetune(model, params: Dict[str, torch.nn.Parameter], prompts, target_ids, lr: float, steps: int,
              kl_prompt_ids: KLPrompt, kl_weight: float, stop_nll: float, after_step=None) -> dict:
    """Shared FT loop on ``params`` (all other parameters frozen). The caller restores the weights."""
    _check_hparams(lr, float("inf"), steps, stop_nll, kl_weight)
    batch = TeacherForcedBatch([_prompt_ids(p) for p in prompts], target_ids,
                               _kl_ids(kl_prompt_ids) if kl_weight > 0 else None, with_positions=False)
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        base_logp = batch.base_logp(model(batch.X)[0])
    for p in params.values():
        p.requires_grad_(True)
    opt = torch.optim.Adam(list(params.values()), lr=lr)
    n = 0
    stopped = False
    while True:
        logits = model(batch.X)[0]
        nll = batch.nll(logits)
        kl = batch.kl_term(logits, base_logp)
        loss = nll + kl_weight * kl
        if float(nll.detach()) < stop_nll:
            stopped = True
            break
        if n >= steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        n += 1
        if after_step is not None:
            with torch.no_grad():
                after_step()
    return {"nll": float(nll.detach()), "kl": float(kl.detach()), "steps": n, "stopped": stopped}


def ft_layer(model, layer: int, prompts, target_ids, lr: float, steps: int, eps: float, *,
             kl_prompt_ids: KLPrompt = None, kl_weight: float = 1.0, stop_nll: float = 0.05,
             info: Optional[dict] = None) -> torch.Tensor:
    """FT-L: Adam on fc2 of ``layer`` with an ℓ∞ clamp ``eps`` on ΔW. Returns ΔW (float32); W is restored."""
    if not eps > 0:
        raise ValueError("eps must be > 0")
    fc2 = _fc2(model, layer)
    name = f"layers.{layer}.ffn.fc2.weight"
    named = dict(model.named_parameters())
    if named.get(name) is not fc2.weight:
        raise RuntimeError(f"parameter {name!r} is not layers[{layer}].ffn.fc2.weight")
    W = named[name]
    W0 = W.detach().clone()
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    grads = {n: p.grad for n, p in model.named_parameters()}   # restored too: no stale .grad is left behind
    was_training = model.training

    def clamp():
        W.copy_(W0 + (W - W0).clamp(-eps, eps))

    try:
        model.eval()
        res = _finetune(model, {name: W}, prompts, target_ids, lr, steps, kl_prompt_ids, kl_weight, stop_nll,
                        after_step=clamp)
        delta = (W.detach() - W0).clone()
    finally:
        with torch.no_grad():
            W.copy_(W0)
        for n, p in model.named_parameters():
            p.requires_grad_(flags[n])
            p.grad = grads[n]
        model.train(was_training)
        _clear(model)
    if info is not None:
        info.update(res)
    return delta


def ft_all(model, prompts, target_ids, lr: float, steps: int, *, kl_prompt_ids: KLPrompt = None,
           kl_weight: float = 1.0, stop_nll: float = 0.05, info: Optional[dict] = None) -> dict:
    """FT-ALL: Adam on every parameter. Returns the fine-tuned state_dict (clones); the model is restored."""
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    grads = {n: p.grad for n, p in model.named_parameters()}   # restored too: no stale .grad is left behind
    was_training = model.training
    try:
        model.eval()
        params = dict(model.named_parameters())      # by name; tied weights appear once
        res = _finetune(model, params, prompts, target_ids, lr, steps, kl_prompt_ids, kl_weight, stop_nll)
        new_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    finally:
        model.load_state_dict(original, strict=True)
        for n, p in model.named_parameters():
            p.requires_grad_(flags[n])
            p.grad = grads[n]
        model.train(was_training)
        _clear(model)
    if info is not None:
        info.update(res)
    return new_state


def _clear(model) -> None:
    fn = getattr(model, "clear_kv_cache", None)
    if callable(fn):
        fn()


# ---------------------------------------------------------------------------------------------------- NORM
_LETTER_RUN = re.compile(r"[^\W\d_]+")


def _informal_table(lexicon: Iterable[str]) -> Dict[str, str]:
    """informal inflected form -> formal form; on a collision the longest name wins (then the smaller string)."""
    table: Dict[str, Tuple[int, str]] = {}
    for name in sorted(set(lexicon)):
        for case in INFLECTED:
            inf, form = informal(name, case), formal(name, case)
            cur = table.get(inf)
            cand = (len(name), form)
            if cur is None or (cand[0] > cur[0]) or (cand[0] == cur[0] and cand[1] < cur[1]):
                table[inf] = cand
    return {k: v[1] for k, v in table.items()}


def normalize_informal(text: str, lexicon) -> str:
    """Restore the apostrophe of informally written inflected lexicon names: ``Aytenden`` → ``Ayten'den``.

    A maximal letter run is rewritten only if it equals ``name + suffix(name, case)`` for a lexicon name and an
    inflected case (``morph_tr``), and is not itself a lexicon name. Runs right after an apostrophe (suffix
    chunks of formal text) are left alone, so the function is idempotent. If two names could explain the same
    run, the longest name wins.
    """
    names = set(lexicon)
    table = _informal_table(names)

    def repl(m: "re.Match[str]") -> str:
        w = m.group(0)
        if w in names or (m.start() > 0 and text[m.start() - 1] in "'’"):
            return w
        return table.get(w, w)

    return _LETTER_RUN.sub(repl, text)


# ---------------------------------------------------------------------------------------------------- lives: I, Q
class _DeltaState:
    def __init__(self, W0: torch.Tensor, C0: torch.Tensor):
        self.W0 = _f64(W0, "W0")
        if self.W0.dim() != 2:
            raise ValueError("W0 must be [d, f]")
        d, f = self.W0.shape
        self.C0 = _f64(C0, "C0", (f, f))
        self.keys: Dict[str, torch.Tensor] = {}
        self.M = torch.zeros(d, f, dtype=torch.float64)

    @property
    def d(self) -> int:
        return self.W0.shape[0]

    @property
    def f(self) -> int:
        return self.W0.shape[1]

    def __contains__(self, fact_id: str) -> bool:
        return fact_id in self.keys

    def __len__(self) -> int:
        return len(self.keys)

    def fact_ids(self) -> List[str]:
        return list(self.keys)

    def delta64(self) -> torch.Tensor:
        return self.M.clone()

    def delta(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """``M = W − W0`` for the FFNMemory hook (always a copy, also for ``dtype=torch.float64``)."""
        return self.M.to(dtype=dtype, copy=True)

    def weights64(self) -> torch.Tensor:
        return self.W0 + self.M


class IndependentState(_DeltaState):
    """State I: one ROME delta per fact, each computed against W0 (v* searched on the base model);
    ``W = W0 + Σ_f M_f``. correct = replace the fact's delta; retract = subtract it."""

    def __init__(self, W0: torch.Tensor, C0: torch.Tensor):
        super().__init__(W0, C0)
        self.deltas: Dict[str, torch.Tensor] = {}

    def v_star(self, k: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """``v* = W0 k + δ`` (δ from a search on the base model)."""
        return self.W0 @ _f64(k, "k", (self.f,)) + _f64(delta, "delta", (self.d,))

    def insert(self, fact_id: str, k: torch.Tensor, v_star: torch.Tensor) -> None:
        if fact_id in self.keys:
            raise KeyError(f"fact {fact_id!r} exists (use correct)")
        k = _f64(k, "k", (self.f,))
        Mf = rome_update(self.W0, self.C0, k, v_star)
        self.keys[fact_id], self.deltas[fact_id] = k, Mf
        self.M += Mf

    def correct(self, fact_id: str, v_star: torch.Tensor, k: Optional[torch.Tensor] = None) -> None:
        """Replace the fact's delta by a new ROME delta against W0 (key: ``k`` if given, else the stored one)."""
        if fact_id not in self.keys:
            raise KeyError(f"fact {fact_id!r} not present")
        k = self.keys[fact_id] if k is None else _f64(k, "k", (self.f,))
        Mf = rome_update(self.W0, self.C0, k, v_star)
        self.M -= self.deltas[fact_id]
        self.M += Mf
        self.keys[fact_id], self.deltas[fact_id] = k, Mf
        self._settle()

    def retract(self, fact_id: str) -> None:
        if fact_id not in self.keys:
            raise KeyError(f"fact {fact_id!r} not present")
        self.M -= self.deltas.pop(fact_id)
        del self.keys[fact_id]
        self._settle()

    def erase(self) -> None:
        self.keys.clear()
        self.deltas.clear()
        self._settle()

    def _settle(self) -> None:
        if not self.keys:          # no fact: exactly zero (no float residue of the running sum)
            self.M.zero_()


class SequentialState(_DeltaState):
    """State Q: sequential ROME on the current weights ``W = W0 + M``; v* is searched against the current model
    (install ``delta()`` in the FFNMemory before the search). correct = another write at the fact's key;
    retract = a write of ``W0 k`` back at the fact's key. Order-dependent by construction."""

    def current_W(self) -> torch.Tensor:
        return self.W0 + self.M

    def v_star(self, k: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """``v* = W k + δ`` with the current weights (δ from a search on the current model)."""
        return self.current_W() @ _f64(k, "k", (self.f,)) + _f64(delta, "delta", (self.d,))

    def _write(self, k: torch.Tensor, v_star: torch.Tensor) -> None:
        self.M = self.M + rome_update(self.current_W(), self.C0, k, v_star)

    def insert(self, fact_id: str, k: torch.Tensor, v_star: torch.Tensor) -> None:
        if fact_id in self.keys:
            raise KeyError(f"fact {fact_id!r} exists (use correct)")
        k = _f64(k, "k", (self.f,))
        self._write(k, v_star)
        self.keys[fact_id] = k

    def correct(self, fact_id: str, v_star: torch.Tensor, k: Optional[torch.Tensor] = None) -> None:
        if fact_id not in self.keys:
            raise KeyError(f"fact {fact_id!r} not present")
        k = self.keys[fact_id] if k is None else _f64(k, "k", (self.f,))
        self._write(k, v_star)
        self.keys[fact_id] = k

    def retract(self, fact_id: str) -> None:
        """Write the base value ``W0 k`` back at the fact's key (other facts are not restored)."""
        k = self.keys.pop(fact_id, None)
        if k is None:
            raise KeyError(f"fact {fact_id!r} not present")
        self._write(k, self.W0 @ k)

    def erase(self) -> None:
        self.keys.clear()
        self.M.zero_()


__all__ = ["ft_layer", "ft_all", "normalize_informal", "IndependentState", "SequentialState"]
