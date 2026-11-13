# -*- coding: utf-8 -*-
"""Mid-stack address codes for writing one fact at block ``l`` (PROTOCOL §3; KA-H1, S2).

One residual delta per (edit, step) comes from CEV-11's frozen ``search_delta`` at the block's STD site. It is then
installed through five addresses:

* ``A-TOK``   CEV-11's frozen ROME write on the native fc2 key (prefix-averaged key, C0 from the pool). By the
              kernel law every position x' moves by ``delta * tau(x')`` with
              ``tau(x') = k(x')' C0^-1 kbar / kbar' C0^-1 kbar`` (:func:`kernel_tau`).
* radius gates: ``delta`` is added to the block's FFN output at every position whose code lies inside the cosine
  ball of the key, ``1[cos(u(x'), u(xbar)) >= theta]``:
    ``A-TOKG``  u = native fc2 key (same code as A-TOK; radius gate instead of the kernel tau)
    ``A-RES``   u = norm2(a_pre)   (the full residual stream as the FFN sees it)
    ``A-CTX``   u = c = sum_{k<=l} g_k   (the attention-written stream)
    ``A-SUP``   u = P'(c - mu), a supervised LDA projection of c fitted on c0 entities (ceiling, never on TEST)

Every gate reads a code computed before f_l, so for a given token sequence whether a gate fires at a position is
the same before and after the write. A gate is live at every position, object positions included, so the pre-edit
indicator says where the write lands only when it is computed on the whole sequence that is scored (prefix plus
teacher-forced object tokens; ``ka.part_a.scored_ids``), not on the prefix alone (S2).

``GateMemory`` never touches a parameter. With no entry it adds nothing and the forward pass is bit-identical.
It changes the residual stream below the upper blocks, so setting or clearing entries clears the model KV cache.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from . import KA_VERSION  # noqa: F401
from .codes import Tap, capture, unit

GATE_LAYER = 2
STEPS = (0.5, 1.0, 2.0)
QS = (0.002, 0.01, 0.05)                 # matched c0 false-match rates of the radius thresholds
CAP_MULT = 8
GATE_CODE = {"A-TOKG": "key", "A-RES": "res", "A-CTX": "att", "A-SUP": "sup"}
GATE_ADDRESSES = tuple(GATE_CODE)
ADDRESSES = ("A-TOK",) + GATE_ADDRESSES
H1_BASELINES = ("A-TOK", "A-TOKG", "A-RES")
SUP_DIM = 64
GATE_EPS = 1e-6                          # cos >= theta - GATE_EPS, so theta = 1 still fires at the key itself
D_C = 64                                 # nominal code dimension of the temperature <-> radius identity


def _clear_kv(model) -> None:
    fn = getattr(model, "clear_kv_cache", None)
    if callable(fn):
        fn()


def gate_fire(u: torch.Tensor, key_unit: torch.Tensor, theta: float) -> torch.Tensor:
    """Boolean mask over the leading axes of ``u``: ``cos(u, key) >= theta`` (float32, same rule everywhere)."""
    cos = unit(u.to(torch.float32)) @ key_unit.to(torch.float32)
    return cos >= (float(theta) - GATE_EPS)


def radius_from_temperature(T: float, d_c: float = D_C) -> float:
    """Two-point deterministic annealing: a pair with cosine kappa merges iff kappa > 1 - T/d_c."""
    return 1.0 - float(T) / float(d_c)


def temperature_from_radius(theta: float, d_c: float = D_C) -> float:
    return float(d_c) * (1.0 - float(theta))


def project_sup(att: torch.Tensor, proj: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    mu, P = proj
    return (att.to(P.dtype) - mu) @ P


def code_specs(codes: Sequence[str], layer: int) -> List[Tuple[str, int]]:
    """Tap specs needed for the gate codes (``sup`` is derived from ``att``)."""
    kinds = sorted({"att" if c == "sup" else c for c in codes})
    return [(k, layer) for k in kinds]


def codes_from_tap(rec: Dict[str, torch.Tensor], code: str, layer: int,
                   proj: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
    if code == "sup":
        if proj is None:
            raise ValueError("the sup code needs its projection")
        return project_sup(rec[f"att{layer}"], proj)
    return rec[f"{code}{layer}"]


class GateMemory:
    """Side memory adding ``delta`` to ``layers[layer].ffn.fc2``'s output inside cosine balls of one code.

    ``entries``: ``(key_unit, theta, delta)`` triples. ``last_fire`` holds, after a forward pass, one boolean
    ``[B, T]`` mask per entry (where that entry fired).
    """

    def __init__(self, model, layer: int, code: str, proj: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if code not in ("key", "res", "att", "sup"):
            raise ValueError(f"unknown gate code {code!r}")
        if code == "sup" and proj is None:
            raise ValueError("the sup code needs its projection (mu, P)")
        if not 0 <= layer < len(model.layers):
            raise IndexError(f"layer {layer} out of range")
        self.model, self.layer, self.code = model, int(layer), code
        self.proj = None if proj is None else (proj[0].to(torch.float32), proj[1].to(torch.float32))
        self.entries: List[Tuple[torch.Tensor, float, torch.Tensor]] = []
        self.last_fire: List[torch.Tensor] = []
        self._code: Optional[torch.Tensor] = None
        self._tap = Tap(model, [] if code == "key" else code_specs([code], layer))
        self._tap.__enter__()
        self._handle = model.layers[layer].ffn.fc2.register_forward_hook(self._hook)

    @property
    def active(self) -> bool:
        return self._handle is not None

    def set(self, entries: Sequence[Tuple[torch.Tensor, float, torch.Tensor]]) -> None:
        if self._handle is None:
            raise RuntimeError("GateMemory was removed; create a new one")
        d = self.model.layers[self.layer].ffn.fc2.weight.shape[0]
        out = []
        for key, theta, delta in entries:
            key = key.detach().to(torch.float32)
            delta = delta.detach().to(self.model.layers[self.layer].ffn.fc2.weight.dtype)
            if key.dim() != 1 or delta.shape != (d,):
                raise ValueError("entry shapes: key [du], delta [d]")
            if not (torch.isfinite(key).all() and torch.isfinite(delta).all()):
                raise ValueError("non-finite gate entry")
            if abs(float(key.norm()) - 1.0) > 1e-4:
                raise ValueError("the gate key must be a unit vector")
            out.append((key.clone(), float(theta), delta.clone()))
        self.entries = out
        self.last_fire = []
        _clear_kv(self.model)

    def clear(self) -> None:
        self.entries = []
        self.last_fire = []
        _clear_kv(self.model)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
            self._tap.__exit__(None, None, None)
        self.entries = []
        _clear_kv(self.model)

    def _hook(self, module, inputs, output):
        if not self.entries:
            return None                      # no add at all: the forward pass stays bit-identical
        if self.code == "key":
            u = inputs[0]
        else:
            u = codes_from_tap(self._tap.out, self.code, self.layer, self.proj)
        un = unit(u.to(torch.float32))
        out = output
        fires = []
        for key, theta, delta in self.entries:
            fire = (un @ key) >= (theta - GATE_EPS)
            fires.append(fire.detach())
            if bool(fire.any()):
                out = out + fire.unsqueeze(-1).to(output.dtype) * delta
        self.last_fire = fires
        return None if out is output else out


@torch.no_grad()
def gate_key(model, layer: int, code: str, ids: Sequence[int], pos: int,
             proj: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
    """Unit key vector (float32) of ``code`` at position ``pos`` of the sequence ``ids``."""
    rec = capture(model, [list(ids)], code_specs([code], layer), dtype=torch.float32)[0]
    p = None if proj is None else (proj[0].to(torch.float32), proj[1].to(torch.float32))
    return unit(codes_from_tap(rec, code, layer, p)[pos])


def gate_thresholds(pos_codes: torch.Tensor, pos_entity: Sequence[str], key_codes: torch.Tensor,
                    key_entity: Sequence[str], qs: Sequence[float] = QS, chunk: int = 64) -> Dict[float, float]:
    """theta(q): the (1 - q)-quantile of cos(u(x'), u(xbar)) over positions x' of OTHER entities.

    ``pos_codes [n, du]`` are codes at c0 token positions, ``key_codes [m, du]`` codes at c0 key sites; pairs of
    the same entity are excluded. Computed per world from c0 only, before any write. A gate with threshold
    theta(q) therefore fires on a fraction q of unrelated (key, position) pairs of the calibration set.
    """
    P = unit(pos_codes.to(torch.float32))
    K = unit(key_codes.to(torch.float32))
    ents = sorted(set(pos_entity) | set(key_entity))
    eid = {e: i for i, e in enumerate(ents)}
    pe = torch.tensor([eid[e] for e in pos_entity])
    ke = torch.tensor([eid[e] for e in key_entity])
    vals = []
    for s in range(0, K.shape[0], chunk):
        C = K[s:s + chunk] @ P.T
        keep = ke[s:s + chunk, None] != pe[None, :]
        vals.append(C[keep])
    v = torch.cat(vals)
    n = v.numel()
    if n == 0:
        raise ValueError("no (key, position) pair of different entities")
    out = {}
    for q in qs:
        k = min(n, max(1, int(round((1.0 - float(q)) * n))))
        out[float(q)] = float(torch.kthvalue(v, k).values)
    return out


def false_match_rate(pos_codes: torch.Tensor, pos_entity: Sequence[str], key_codes: torch.Tensor,
                     key_entity: Sequence[str], theta: float, chunk: int = 64) -> float:
    """Share of (key, other-entity position) pairs with ``cos >= theta`` (check of :func:`gate_thresholds`)."""
    P = unit(pos_codes.to(torch.float32))
    K = unit(key_codes.to(torch.float32))
    ents = sorted(set(pos_entity) | set(key_entity))
    eid = {e: i for i, e in enumerate(ents)}
    pe = torch.tensor([eid[e] for e in pos_entity])
    ke = torch.tensor([eid[e] for e in key_entity])
    hit = tot = 0
    for s in range(0, K.shape[0], chunk):
        C = K[s:s + chunk] @ P.T
        keep = ke[s:s + chunk, None] != pe[None, :]
        hit += int(((C >= theta) & keep).sum())
        tot += int(keep.sum())
    return hit / max(tot, 1)


def kernel_tau(H: torch.Tensor, k_bar: torch.Tensor, C0_inv: torch.Tensor) -> torch.Tensor:
    """tau(x') = k(x')' C0^-1 kbar / (kbar' C0^-1 kbar) for every row of ``H`` (float64)."""
    u = C0_inv.double() @ k_bar.double()
    return (H.double() @ u) / (k_bar.double() @ u)


def entity_windows(F0: Dict[str, torch.Tensor], forms: Dict[str, List[torch.Tensor]],
                   ctx: Dict[str, List[torch.Tensor]], rn: Dict[str, List[str]]) -> dict:
    """Pre-edit geometry of one code at the key site (descriptive; temperature = radius).

    ``F0[e]`` is entity e's formal nominative code, ``forms[e]`` its FUSED-form codes, ``ctx[e]`` its
    context-variant codes, ``rn[e]`` its root neighbours. Entity e has a window iff every form and context
    variant is closer to its own key than the nearest root neighbour:
    ``max(T_c(FUSED_e), T_c(CTX_e)) < min_m T_c(e, m)`` with ``T_c = d_c (1 - cos)``.
    CAM = mean cos(FUSED, own) - mean cos(root neighbours).
    """
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=0))  # noqa: E731
    win, cf, cr = [], [], []
    for e in sorted(F0):
        f = [cos(v, F0[e]) for v in forms.get(e, [])]
        cx = [cos(v, F0[e]) for v in ctx.get(e, [])]
        r = [cos(F0[m], F0[e]) for m in rn.get(e, []) if m in F0 and m != e]
        if not f or not r:
            continue
        cf += f
        cr += r
        win.append(min(f + cx) > max(r))
    if not win:
        return {"n_entities": 0, "window_share": None, "CAM": None, "cos_fused_own": None, "cos_rn": None}
    m = lambda v: sum(v) / len(v)  # noqa: E731
    return {"n_entities": len(win), "window_share": m([float(w) for w in win]), "CAM": m(cf) - m(cr),
            "cos_fused_own": m(cf), "cos_rn": m(cr)}


__all__ = ["GATE_LAYER", "STEPS", "QS", "CAP_MULT", "GATE_CODE", "GATE_ADDRESSES", "ADDRESSES", "H1_BASELINES",
           "SUP_DIM", "GATE_EPS", "gate_fire", "radius_from_temperature", "temperature_from_radius", "project_sup",
           "code_specs", "codes_from_tap", "GateMemory", "gate_key", "gate_thresholds", "false_match_rate",
           "kernel_tau", "entity_windows"]
