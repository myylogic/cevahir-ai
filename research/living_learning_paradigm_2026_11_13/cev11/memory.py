# -*- coding: utf-8 -*-
"""Ağ içi bellek: fc2 kancası, ROME kapalı biçimi ve float64 defter (CEV-11 PROTOCOL §0, §8, §10.3, §13).

In-network memory for CEV-11 "Paradigma".

* ``FFNMemory`` — a forward hook on ``model.layers[l].ffn.fc2`` that returns ``out + h @ M.T`` (``h`` is the fc2
  input, i.e. the key space). No repository file and no model parameter is modified; ``remove()`` detaches the
  hook, after which the forward pass is the unmodified one. Every change of ``M`` clears the model's KV cache.
  ``substitute()`` replaces the fc2 output at one position per batch row; the v* search (``vstar.py``) uses it.
* ``rome_update`` — single-edit ROME closed form ``M = (v* − W k)(C0⁻¹k)ᵀ / (kᵀC0⁻¹k)`` in float64.
* ``Ledger`` — float64 sufficient statistics ``A = Σ k kᵀ``, ``B = Σ r kᵀ`` with per-fact ``(k, r)`` records and
  ``W = W0 + B (λ C0 + A)⁻¹``. insert / correct / retract / erase / merge / delta / export / load. An identity
  block ``{base_sha256, layer, lam, C0_sha256, tokenizer_identity, code_version}`` is checked on construction,
  merge and load; any mismatch raises ``ValueError`` (refusal), never a silent repair.
* ``EntityJointLedger`` — grouping helper for the lives (state J): all facts of one entity share one key and one
  jointly optimised value; the entity's record is re-solved whenever its believed set changes.
* ``key_covariance`` — float64 mean ``h hᵀ`` over non-PAD positions (``add_ridge`` adds ``rel·tr(C)/f·I``).
* ``state_fingerprint`` — content hash of the weights plus the active FFNMemory hooks (used by the v* memo).

Conventions (documented deviations / additions relative to INTERFACES.md, all backward compatible):
* ``rome_update`` returns a float64 matrix; ``FFNMemory.set_M`` casts to the fc2 weight dtype (float32).
* ``Ledger.delta(dtype=torch.float32)`` takes an optional dtype; ``delta64()`` returns the float64 matrix.
* ``Ledger.load(path, expected_identity=None)``: ``load(path)`` alone checks the file's internal consistency
  (format, C0/W0 hashes, A/B against the records, code version); with ``expected_identity`` it also refuses any
  identity that differs from the caller's expectation (e.g. another base model or tokenizer).
* A ledger whose belief set becomes empty has exactly zero statistics (A = B = 0), so ``erase()`` and the
  retraction of the last fact both give ``M == 0`` exactly (no float residue is carried in an empty state).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from tokenizer_management.byte_bpe import PAD_ID

LEDGER_FORMAT = "cev11-ledger"
LEDGER_FORMAT_VERSION = 1
# Bump when the ledger mathematics or the file layout changes; ledgers of another version are refused.
CODE_VERSION = "cev11-memory/1"
IDENTITY_KEYS: Tuple[str, ...] = ("base_sha256", "layer", "lam", "C0_sha256", "tokenizer_identity", "code_version")

TensorLike = Union[torch.Tensor, Sequence[float]]


# ---------------------------------------------------------------------------------------------------- helpers
def tensor_sha256(t: torch.Tensor) -> str:
    """sha256 over dtype, shape and the raw bytes of a contiguous CPU copy of ``t``."""
    t = t.detach().to("cpu").contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode("ascii"))
    h.update(json.dumps(list(t.shape)).encode("ascii"))
    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def _f64(x: TensorLike, name: str, shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
    """Detached float64 CPU copy; raises on a wrong shape or a non-finite entry."""
    t = torch.as_tensor(x).detach().to(device="cpu", dtype=torch.float64).clone()
    if shape is not None and tuple(t.shape) != tuple(shape):
        raise ValueError(f"{name}: expected shape {tuple(shape)}, got {tuple(t.shape)}")
    if not torch.isfinite(t).all():
        raise ValueError(f"{name}: non-finite entries")
    return t


def _clear_kv_cache(model) -> None:
    """Same semantics as decode.clear_cache: model.clear_kv_cache() if the model has a KV cache."""
    fn = getattr(model, "clear_kv_cache", None)
    if callable(fn):
        fn()


def _fc2(model, layer: int):
    layers = getattr(model, "layers", None)
    if layers is None:
        raise AttributeError("model has no .layers (expected a CevahirNeuralNetwork)")
    if not isinstance(layer, int) or not 0 <= layer < len(layers):
        raise IndexError(f"layer {layer!r} out of range 0..{len(layers) - 1}")
    ffn = getattr(layers[layer], "ffn", None)
    fc2 = getattr(ffn, "fc2", None)
    if fc2 is None or not hasattr(fc2, "weight"):
        raise AttributeError(f"model.layers[{layer}].ffn.fc2 not found (MoE layers are not supported)")
    return fc2


# ---------------------------------------------------------------------------------------------------- FFNMemory
class FFNMemory:
    """Forward hook on ``model.layers[layer].ffn.fc2`` returning ``out + h @ M.T`` (then an optional substitution).

    ``M`` has the fc2 weight shape ``[d, f]`` and is held in the weight dtype. With ``M is None`` the hook adds
    nothing; ``zero()`` installs an explicit all-zero ``M`` (the V1 check: outputs stay bitwise identical).
    The memory never touches ``fc2.weight``.
    """

    def __init__(self, model, layer: int):
        self.model = model
        self.layer = layer
        self.fc2 = _fc2(model, layer)
        self._M: Optional[torch.Tensor] = None
        self._sub: Optional[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = None   # (batch, rows, pos, value)
        self._handle = self.fc2.register_forward_hook(self._hook)

    # -------------------------------------------------------------- state
    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.fc2.weight.shape)  # type: ignore[return-value]

    @property
    def M(self) -> Optional[torch.Tensor]:
        """The current memory matrix (weight dtype) or None."""
        return self._M

    @property
    def active(self) -> bool:
        return self._handle is not None

    def _check_active(self) -> None:
        if self._handle is None:
            raise RuntimeError("FFNMemory was removed; create a new one")

    def set_M(self, M: torch.Tensor) -> None:
        """Install ``M`` ([d, f], any float dtype; cast to the weight dtype) and clear the KV cache."""
        self._check_active()
        w = self.fc2.weight
        if tuple(M.shape) != tuple(w.shape):
            raise ValueError(f"M must have shape {tuple(w.shape)}, got {tuple(M.shape)}")
        M = M.detach().to(device=w.device, dtype=w.dtype).clone()
        if not torch.isfinite(M).all():
            raise ValueError("M has non-finite entries")
        self._M = M
        _clear_kv_cache(self.model)

    def zero(self) -> None:
        """All-zero memory (hook stays installed); clears the KV cache."""
        self._check_active()
        self._M = torch.zeros_like(self.fc2.weight, requires_grad=False)
        _clear_kv_cache(self.model)

    def remove(self) -> None:
        """Detach the hook; the model's forward pass is then exactly the unhooked one. Clears the KV cache."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._M = None
        self._sub = None
        _clear_kv_cache(self.model)

    def substitute(self, pos_by_row: Optional[Sequence[Optional[int]]], value: Optional[torch.Tensor]) -> None:
        """Replace the fc2 output (after ``M``) at ``(row i, pos_by_row[i])`` by ``value``.

        ``pos_by_row`` has one entry per batch row of the next forward passes (``None`` = leave that row alone);
        ``value`` is ``[d]`` (same vector for every substituted row) or ``[len(pos_by_row), d]`` (row-wise).
        ``substitute(None, None)`` switches substitution off. Gradients flow into ``value`` (v* search).
        """
        self._check_active()
        if pos_by_row is None or value is None:
            if (pos_by_row is None) != (value is None):
                raise ValueError("pos_by_row and value must both be None or both be given")
            self._sub = None
            _clear_kv_cache(self.model)
            return
        d = self.fc2.weight.shape[0]
        rows = [i for i, p in enumerate(pos_by_row) if p is not None]
        if not rows:
            raise ValueError("pos_by_row has no position")
        pos = [int(pos_by_row[i]) for i in rows]   # type: ignore[arg-type]
        if min(pos) < 0:
            raise ValueError("positions must be >= 0")
        if value.dim() == 1:
            if value.shape[0] != d:
                raise ValueError(f"value must have {d} entries")
            val = value.unsqueeze(0).expand(len(rows), d)
        elif value.dim() == 2 and tuple(value.shape) == (len(pos_by_row), d):
            val = value[torch.tensor(rows, dtype=torch.long)]
        else:
            raise ValueError(f"value must be [{d}] or [{len(pos_by_row)}, {d}], got {tuple(value.shape)}")
        self._sub = (len(pos_by_row), torch.tensor(rows, dtype=torch.long), torch.tensor(pos, dtype=torch.long),
                     val.to(dtype=self.fc2.weight.dtype))
        _clear_kv_cache(self.model)

    # -------------------------------------------------------------- hook
    def _hook(self, module, inputs, output):
        out = output
        if self._M is not None:
            out = out + F.linear(inputs[0], self._M)
        if self._sub is not None:
            batch, rows, pos, val = self._sub
            if out.dim() != 3 or out.shape[0] != batch:
                raise RuntimeError(f"substitution was set for batch size {batch}, forward has shape {tuple(out.shape)}")
            if int(pos.max()) >= out.shape[1]:
                raise IndexError(f"substitution position {int(pos.max())} >= sequence length {out.shape[1]}")
            out = out.index_put((rows, pos), val)
        return out if out is not output else None


def active_memories(model) -> List[Tuple[int, FFNMemory]]:
    """(layer, FFNMemory) for every FFNMemory hook currently installed on the model's fc2 modules."""
    found = []
    for li, layer in enumerate(getattr(model, "layers", [])):
        fc2 = getattr(getattr(layer, "ffn", None), "fc2", None)
        if fc2 is None:
            continue
        for hook in fc2._forward_hooks.values():
            owner = getattr(hook, "__self__", None)
            if isinstance(owner, FFNMemory):
                found.append((li, owner))
    return found


def state_fingerprint(model) -> str:
    """sha256 over every state_dict tensor (sorted names) plus the state of the active FFNMemory hooks.

    Raises ``ValueError`` if any other forward (pre-)hook is installed anywhere on the model, or a global module
    hook exists: such a hook could change the forward pass in a way this fingerprint cannot see.
    """
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"state_dict entry {name!r} is not a tensor")
        h.update(name.encode("utf-8"))
        h.update(tensor_sha256(t).encode("ascii"))
    import torch.nn.modules.module as _mm  # global hooks are module-level dicts (private API, read only)
    for attr in ("_global_forward_hooks", "_global_forward_pre_hooks"):
        if getattr(_mm, attr, None):
            raise ValueError(f"a global module hook ({attr}) is installed; cannot fingerprint the forward pass")
    for mname, module in model.named_modules():
        if module._forward_pre_hooks:
            raise ValueError(f"forward pre-hook on {mname!r}; cannot fingerprint the forward pass")
        for hook in module._forward_hooks.values():
            if not isinstance(getattr(hook, "__self__", None), FFNMemory):
                raise ValueError(f"unknown forward hook on {mname!r}; cannot fingerprint the forward pass")
    for li, mem in active_memories(model):
        h.update(f"mem:{li}".encode("ascii"))
        h.update(b"M:none" if mem.M is None else tensor_sha256(mem.M).encode("ascii"))
        if mem._sub is not None:
            batch, rows, pos, val = mem._sub
            h.update(f"sub:{batch}".encode("ascii"))
            for t in (rows, pos, val):
                h.update(tensor_sha256(t).encode("ascii"))
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------- ROME
def rome_update(W: torch.Tensor, C0: torch.Tensor, k: torch.Tensor, v_star: torch.Tensor) -> torch.Tensor:
    """ROME closed form ``M = (v* − W k)(C0⁻¹k)ᵀ / (kᵀC0⁻¹k)``, computed and returned in float64.

    ``(W + M) k = v*`` holds up to float64 rounding. ``C0`` must be symmetric positive definite.
    """
    W64 = _f64(W, "W")
    if W64.dim() != 2:
        raise ValueError("W must be [d, f]")
    d, f = W64.shape
    C64 = _f64(C0, "C0", (f, f))
    k64 = _f64(k, "k", (f,))
    v64 = _f64(v_star, "v_star", (d,))
    u = torch.linalg.solve(C64, k64)            # C0⁻¹ k (no explicit inverse)
    denom = torch.dot(k64, u)
    if not torch.isfinite(denom) or float(denom) <= 0.0:
        raise ValueError(f"kᵀC0⁻¹k = {float(denom)!r} must be positive (C0 must be SPD, k nonzero)")
    r = v64 - W64 @ k64
    return torch.outer(r, u) / denom


def add_ridge(C: torch.Tensor, rel: float = 1e-4) -> torch.Tensor:
    """``C + rel · tr(C)/f · I`` in float64 (PROTOCOL §8: 1e-4 · tr C0 / 512 · I for f = 512)."""
    C64 = _f64(C, "C")
    if C64.dim() != 2 or C64.shape[0] != C64.shape[1]:
        raise ValueError("C must be square")
    f = C64.shape[0]
    return C64 + (rel * torch.trace(C64) / f) * torch.eye(f, dtype=torch.float64)


# ---------------------------------------------------------------------------------------------------- C0
def _as_batch(batch) -> torch.Tensor:
    """LongTensor [B, T] (right-padded with PAD) from a tensor or a list of id lists."""
    if isinstance(batch, torch.Tensor):
        if batch.dim() != 2:
            raise ValueError("a tensor batch must be [B, T]")
        return batch.to(dtype=torch.long, device="cpu")
    seqs = [list(map(int, s)) for s in batch]
    if not seqs or min(len(s) for s in seqs) == 0:
        raise ValueError("a batch must contain non-empty sequences")
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return x


@torch.no_grad()
def key_covariance(model, layer: int, ids_batches: Iterable, exclude_ids: Sequence[int] = (PAD_ID,)) -> torch.Tensor:
    """float64 mean of ``h hᵀ`` over all positions whose token is not in ``exclude_ids`` (default: PAD only).

    ``ids_batches`` yields batches: LongTensor [B, T] right-padded with PAD, or lists of id lists (padded here).
    Accumulation is in float64; the result is exactly symmetric. The ridge (``add_ridge``) is added by the caller.
    """
    fc2 = _fc2(model, layer)
    captured: List[torch.Tensor] = []
    handle = fc2.register_forward_hook(lambda mod, inp, out: captured.append(inp[0].detach()))
    was_training = model.training
    model.eval()
    f = fc2.weight.shape[1]
    acc = torch.zeros(f, f, dtype=torch.float64)
    n = 0
    excl = torch.tensor(sorted(set(int(i) for i in exclude_ids)), dtype=torch.long)
    try:
        for batch in ids_batches:
            x = _as_batch(batch)
            captured.clear()
            model(x)
            if len(captured) != 1:
                raise RuntimeError(f"expected one fc2 call per forward, got {len(captured)}")
            keep = ~torch.isin(x, excl)
            hh = captured[0][keep].to(torch.float64)
            acc += hh.T @ hh
            n += hh.shape[0]
    finally:
        handle.remove()
        model.train(was_training)
    if n == 0:
        raise ValueError("no positions to average")
    C = acc / n
    return (C + C.T) / 2


# ---------------------------------------------------------------------------------------------------- Ledger
def make_identity(base_sha256: str, layer: int, lam: float, C0: torch.Tensor, tokenizer_identity: str,
                  code_version: str = CODE_VERSION, **extra) -> dict:
    """Identity block of a ledger. ``C0_sha256`` is ``tensor_sha256`` of C0 in float64."""
    ident = {"base_sha256": str(base_sha256), "layer": int(layer), "lam": float(lam),
             "C0_sha256": tensor_sha256(_f64(C0, "C0")), "tokenizer_identity": str(tokenizer_identity),
             "code_version": str(code_version)}
    for k, v in extra.items():
        if k in ident:
            raise ValueError(f"duplicate identity key {k!r}")
        ident[k] = v
    return ident


def _norm_identity(identity: Mapping) -> dict:
    missing = [k for k in IDENTITY_KEYS if k not in identity]
    if missing:
        raise ValueError(f"identity block lacks {missing}")
    out = dict(identity)
    out["layer"] = int(identity["layer"])
    out["lam"] = float(identity["lam"])
    for k in ("base_sha256", "C0_sha256", "tokenizer_identity", "code_version"):
        if not isinstance(identity[k], str) or not identity[k]:
            raise ValueError(f"identity[{k!r}] must be a non-empty string")
    return out


def _identity_diff(a: Mapping, b: Mapping) -> List[str]:
    keys = sorted(set(a) | set(b))
    return [k for k in keys if a.get(k, None) != b.get(k, None)]


@dataclass
class FactRecord:
    """One believed fact: key ``k`` [f] and residual ``r = v* − W0 k`` [d], both float64."""
    entity: str
    k: torch.Tensor
    r: torch.Tensor


class Ledger:
    """float64 ledger ``W = W0 + B (λ C0 + A)⁻¹`` with ``A = Σ k kᵀ`` and ``B = Σ r kᵀ`` over believed facts.

    ``W`` depends only on the current belief set (up to float64 rounding of the running sums), not on the order
    of operations; the per-fact ``(k, r)`` records make retraction exact even where two histories share one W
    (V5). ``r`` is the residual against the base: ``r = v* − W0 k``.
    """

    def __init__(self, W0: torch.Tensor, C0: torch.Tensor, lam: float, identity: Mapping):
        self.W0 = _f64(W0, "W0")
        if self.W0.dim() != 2:
            raise ValueError("W0 must be [d, f]")
        d, f = self.W0.shape
        self.C0 = _f64(C0, "C0", (f, f))
        self.lam = float(lam)
        if not self.lam > 0.0:
            raise ValueError("lam must be > 0")
        ident = _norm_identity(identity)
        if ident["lam"] != self.lam:
            raise ValueError(f"identity lam {ident['lam']!r} != ledger lam {self.lam!r}")
        if ident["C0_sha256"] != tensor_sha256(self.C0):
            raise ValueError("identity C0_sha256 does not match C0")
        if ident["code_version"] != CODE_VERSION:
            raise ValueError(f"identity code_version {ident['code_version']!r} != {CODE_VERSION!r}")
        self.identity = ident
        self.W0_sha256 = tensor_sha256(self.W0)
        self.records: Dict[str, FactRecord] = {}
        self.A = torch.zeros(f, f, dtype=torch.float64)
        self.B = torch.zeros(d, f, dtype=torch.float64)
        self._cache: Optional[torch.Tensor] = None

    # -------------------------------------------------------------- shape / access
    @property
    def d(self) -> int:
        return self.W0.shape[0]

    @property
    def f(self) -> int:
        return self.W0.shape[1]

    def __len__(self) -> int:
        return len(self.records)

    def __contains__(self, fact_id: str) -> bool:
        return fact_id in self.records

    def fact_ids(self) -> List[str]:
        return list(self.records)

    def residual(self, k: TensorLike, v_star: TensorLike) -> torch.Tensor:
        """``r = v* − W0 k`` (float64)."""
        return _f64(v_star, "v_star", (self.d,)) - self.W0 @ _f64(k, "k", (self.f,))

    # -------------------------------------------------------------- operations
    def _add(self, k: torch.Tensor, r: torch.Tensor, sign: float) -> None:
        self.A += sign * torch.outer(k, k)
        self.B += sign * torch.outer(r, k)

    def _settle(self) -> None:
        self._cache = None
        if not self.records:   # empty belief set: statistics are exactly zero
            self.A.zero_()
            self.B.zero_()

    def insert(self, fact_id: str, entity: str, k: TensorLike, r: TensorLike) -> None:
        """Add a new fact (``fact_id`` must be new; use ``correct`` to change a value)."""
        if fact_id in self.records:
            raise KeyError(f"fact {fact_id!r} already in the ledger (use correct)")
        rec = FactRecord(str(entity), _f64(k, "k", (self.f,)), _f64(r, "r", (self.d,)))
        self._add(rec.k, rec.r, +1.0)
        self.records[fact_id] = rec
        self._settle()

    def correct(self, fact_id: str, r_new: TensorLike) -> None:
        """Replace the residual of an existing fact (same key)."""
        rec = self.records.get(fact_id)
        if rec is None:
            raise KeyError(f"fact {fact_id!r} not in the ledger")
        r_new = _f64(r_new, "r_new", (self.d,))
        self.B -= torch.outer(rec.r, rec.k)
        self.B += torch.outer(r_new, rec.k)
        rec.r = r_new
        self._settle()

    def retract(self, fact_id: str) -> None:
        rec = self.records.pop(fact_id, None)
        if rec is None:
            raise KeyError(f"fact {fact_id!r} not in the ledger")
        self._add(rec.k, rec.r, -1.0)
        self._settle()

    def erase(self) -> None:
        """Forget every fact: ``delta()`` is exactly zero afterwards."""
        self.records.clear()
        self._settle()

    def merge(self, other: "Ledger") -> None:
        """Union of belief sets. Refuses (ValueError) another identity, another W0, or a conflicting fact id.

        A fact id present in both ledgers with bitwise identical (entity, k, r) is kept once. The merge is
        atomic: on refusal nothing changes.
        """
        if not isinstance(other, Ledger):
            raise TypeError("can only merge another Ledger")
        diff = _identity_diff(self.identity, other.identity)
        if diff:
            raise ValueError(f"identity mismatch on {diff}; refusing to merge")
        if other.W0_sha256 != self.W0_sha256:
            raise ValueError("W0 differs; refusing to merge")
        new: List[Tuple[str, FactRecord]] = []
        for fid, rec in other.records.items():
            mine = self.records.get(fid)
            if mine is None:
                new.append((fid, rec))
            elif not (mine.entity == rec.entity and torch.equal(mine.k, rec.k) and torch.equal(mine.r, rec.r)):
                raise ValueError(f"fact {fid!r} differs between the ledgers; refusing to merge")
        for fid, rec in new:
            rec = FactRecord(rec.entity, rec.k.clone(), rec.r.clone())
            self._add(rec.k, rec.r, +1.0)
            self.records[fid] = rec
        self._settle()

    # -------------------------------------------------------------- weights
    def _solve(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """``B (λ C0 + A)⁻¹`` via a linear solve (no explicit inverse)."""
        mat = self.lam * self.C0 + A
        return torch.linalg.solve(mat.T, B.T).T

    def delta64(self) -> torch.Tensor:
        """``M = W − W0 = B (λ C0 + A)⁻¹`` in float64 (a copy)."""
        if not self.records:
            return torch.zeros(self.d, self.f, dtype=torch.float64)
        if self._cache is None:
            self._cache = self._solve(self.A, self.B)
        return self._cache.clone()

    def delta(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """``M`` for the hook (float32 by default); the float64 matrix is ``delta64()``."""
        return self.delta64().to(dtype)

    def weights64(self) -> torch.Tensor:
        return self.W0 + self.delta64()

    def statistics_from_records(self, fact_ids: Optional[Iterable[str]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fresh ``(A, B)`` summed over the records in sorted fact-id order (no running-sum history)."""
        ids = sorted(self.records if fact_ids is None else fact_ids)
        A = torch.zeros(self.f, self.f, dtype=torch.float64)
        B = torch.zeros(self.d, self.f, dtype=torch.float64)
        for fid in ids:
            rec = self.records[fid]
            A += torch.outer(rec.k, rec.k)
            B += torch.outer(rec.r, rec.k)
        return A, B

    def solve_direct(self) -> torch.Tensor:
        """Direct batch solve on the current belief set: ``R Kᵀ (λ C0 + K Kᵀ)⁻¹`` (verification, V4)."""
        if not self.records:
            return torch.zeros(self.d, self.f, dtype=torch.float64)
        ids = sorted(self.records)
        K = torch.stack([self.records[i].k for i in ids], 1)    # [f, n]
        R = torch.stack([self.records[i].r for i in ids], 1)    # [d, n]
        return self._solve(K @ K.T, R @ K.T)

    def rebuild(self) -> float:
        """Recompute A and B from the records; returns the largest absolute change of any entry (drift)."""
        A, B = self.statistics_from_records()
        drift = max(float((A - self.A).abs().max()), float((B - self.B).abs().max())) if self.records else 0.0
        self.A, self.B = A, B
        self._settle()
        return drift

    # -------------------------------------------------------------- persistence
    def check_identity(self, expected: Mapping) -> None:
        """Refuse (ValueError) unless ``expected`` equals this ledger's identity on every key."""
        diff = _identity_diff(self.identity, _norm_identity(expected))
        if diff:
            raise ValueError(f"ledger identity mismatch on {diff}")

    def export(self, path) -> None:
        """Write the ledger (float64 tensors + identity) atomically to ``path`` (torch.save format)."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": LEDGER_FORMAT, "format_version": LEDGER_FORMAT_VERSION, "identity": dict(self.identity),
            "lam": self.lam, "W0": self.W0, "W0_sha256": self.W0_sha256, "C0": self.C0,
            "A": self.A, "B": self.B,
            "fact_ids": list(self.records),
            "entities": [rec.entity for rec in self.records.values()],
            "K": (torch.stack([rec.k for rec in self.records.values()]) if self.records
                  else torch.zeros(0, self.f, dtype=torch.float64)),
            "R": (torch.stack([rec.r for rec in self.records.values()]) if self.records
                  else torch.zeros(0, self.d, dtype=torch.float64)),
        }
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, expected_identity: Optional[Mapping] = None) -> "Ledger":
        """Load an exported ledger. Refuses (ValueError) on any format, hash, statistics or identity mismatch."""
        data = torch.load(pathlib.Path(path), map_location="cpu", weights_only=True)
        if data.get("format") != LEDGER_FORMAT or data.get("format_version") != LEDGER_FORMAT_VERSION:
            raise ValueError(f"not a {LEDGER_FORMAT} v{LEDGER_FORMAT_VERSION} file: {path}")
        led = cls(data["W0"], data["C0"], data["lam"], data["identity"])    # checks C0 hash, lam, code version
        if led.W0_sha256 != data["W0_sha256"]:
            raise ValueError("W0 hash mismatch (file modified)")
        if expected_identity is not None:
            led.check_identity(expected_identity)
        ids, ents, K, R = data["fact_ids"], data["entities"], data["K"], data["R"]
        if not (len(ids) == len(ents) == K.shape[0] == R.shape[0]) or len(set(ids)) != len(ids):
            raise ValueError("inconsistent fact records in file")
        for fid, ent, k, r in zip(ids, ents, K, R):
            led.records[fid] = FactRecord(ent, _f64(k, "k", (led.f,)), _f64(r, "r", (led.d,)))
        A_rec, B_rec = led.statistics_from_records()
        A, B = _f64(data["A"], "A", (led.f, led.f)), _f64(data["B"], "B", (led.d, led.f))
        for name, got, ref in (("A", A, A_rec), ("B", B, B_rec)):
            scale = max(float(ref.abs().max()), 1e-300) if ref.numel() else 1.0
            if float((got - ref).abs().max()) > 1e-9 * scale:
                raise ValueError(f"stored {name} disagrees with the fact records (file modified)")
        led.A, led.B = A, B    # running sums as exported: a load/export round trip is bitwise exact
        led._settle()
        return led


# ---------------------------------------------------------------------------------------------------- entity-joint
SolveValue = Callable[[str, torch.Tensor, Dict[str, str]], torch.Tensor]


class EntityJointLedger:
    """Lives, state J: one ledger record per entity, holding one key and one jointly optimised value.

    ``solve_value(entity, key, believed) -> r`` must return the residual ``v* − W0 k`` of the value optimised
    jointly for every believed ``{relation: object}`` of the entity against the base model W0 (MEMIT-Merge
    style). It is called whenever the entity's believed set changes; the entity's record is then inserted,
    corrected, or — when nothing is believed any more — retracted. The first key given for an entity is kept;
    a later key must agree with it (relative 1e-5, e.g. a recomputation) or the call is refused.
    """

    KEY_RTOL = 1e-5

    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self._keys: Dict[str, torch.Tensor] = {}
        self._believed: Dict[str, Dict[str, str]] = {}

    @staticmethod
    def record_id(entity: str) -> str:
        return f"entity:{entity}"

    def believed(self, entity: str) -> Dict[str, str]:
        return dict(self._believed.get(entity, {}))

    def key_of(self, entity: str) -> Optional[torch.Tensor]:
        k = self._keys.get(entity)
        return None if k is None else k.clone()

    def _key(self, entity: str, key: Optional[TensorLike]) -> torch.Tensor:
        """The entity's key: the stored one, or ``key`` if none is stored yet (stored by the caller on success)."""
        stored = self._keys.get(entity)
        if key is None:
            if stored is None:
                raise ValueError(f"entity {entity!r} has no key yet; pass key=")
            return stored
        k = _f64(key, "key", (self.ledger.f,))
        if stored is None:
            return k
        if float((k - stored).norm()) > self.KEY_RTOL * float(stored.norm()):
            raise ValueError(f"entity {entity!r} already has a different key")
        return stored

    def write(self, entity: str, rel: str, obj: str, solve_value: SolveValue,
              key: Optional[TensorLike] = None) -> Dict[str, str]:
        """Believe ``(entity, rel) = obj`` (new fact or correction); re-solve the entity's joint value."""
        k = self._key(entity, key)
        believed = dict(self._believed.get(entity, {}))
        believed[rel] = obj
        r = solve_value(entity, k.clone(), dict(believed))
        rid = self.record_id(entity)
        if rid in self.ledger:
            self.ledger.correct(rid, r)
        else:
            self.ledger.insert(rid, entity, k, r)
        self._keys.setdefault(entity, k)
        self._believed[entity] = believed
        return dict(believed)

    def retract(self, entity: str, rel: str, solve_value: SolveValue) -> Dict[str, str]:
        """Stop believing ``(entity, rel)``; re-solve the remaining facts or remove the entity's record."""
        believed = dict(self._believed.get(entity, {}))
        if rel not in believed:
            raise KeyError(f"({entity!r}, {rel!r}) is not believed")
        del believed[rel]
        rid = self.record_id(entity)
        if believed:
            r = solve_value(entity, self._keys[entity].clone(), dict(believed))
            self.ledger.correct(rid, r)
            self._believed[entity] = believed
        else:
            self.ledger.retract(rid)
            del self._believed[entity]
        return dict(believed)


__all__ = ["FFNMemory", "active_memories", "state_fingerprint", "rome_update", "add_ridge", "key_covariance",
           "make_identity", "Ledger", "FactRecord", "EntityJointLedger", "tensor_sha256", "CODE_VERSION",
           "IDENTITY_KEYS"]
