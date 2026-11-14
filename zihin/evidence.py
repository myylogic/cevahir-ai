# -*- coding: utf-8 -*-
"""Evidence arithmetic of Zihin (design section 1). Pure functions; float64; no state.

A tie carries w+[p] and w-[p] >= 0 per provenance class p. With the trust t[p] in [0, 1] of every class

    W+ = sum_p t[p] w+[p]      W- = sum_p t[p] w-[p]      W = W+ + W-
    frequency   f = W+ / W                 (how often it held; 1/2 when nothing was lived)
    conviction  c = W / (W + k)            (how much was lived; k = 1)
    expectation e = c (f - 1/2) + 1/2      (what the mind acts on)

Stored evidence is RAW (not multiplied by trust), so that trust can change later without rewriting anything.

Exactness. Every increment is snapped to the dyadic grid 2**-30 before it is added. Sums of grid values are
exact in float64 as long as they stay below 2**22, so the stored evidence equals the sum of the logged
increments bit for bit, in any order, and subtracting the increments of one moment retracts it exactly.
"""
from __future__ import annotations

import torch

GRID_BITS = 30
GRID = float(2 ** GRID_BITS)
EXACT_LIMIT = float(2 ** (52 - GRID_BITS))  # below this magnitude every grid sum is exact in float64


def q(x: float) -> float:
    """Snap a number to the evidence grid (round half to even, like torch.round)."""
    return float(round(x * GRID)) / GRID


def quantize(x: torch.Tensor) -> torch.Tensor:
    """Snap a float64 tensor to the evidence grid."""
    return torch.round(x * GRID) / GRID


def weigh(w: torch.Tensor, trust: torch.Tensor) -> torch.Tensor:
    """Trust-weighted mass W = sum_p trust[p] w[..., p]."""
    return (w * trust).sum(-1)


def frequency(w_pos, w_neg):
    """f = W+ / (W+ + W-); 1/2 when there is no evidence. Works on floats and on tensors."""
    total = w_pos + w_neg
    if isinstance(total, torch.Tensor):
        safe = torch.where(total > 0, total, torch.ones_like(total))
        return torch.where(total > 0, w_pos / safe, torch.full_like(total, 0.5))
    return w_pos / total if total > 0 else 0.5


def conviction(w_total, k: float = 1.0):
    """c = W / (W + k)."""
    return w_total / (w_total + k)


def expectation(w_pos, w_neg, k: float = 1.0):
    """e = c (f - 1/2) + 1/2 with c = W / (W + k); 1/2 when there is no evidence."""
    return conviction(w_pos + w_neg, k) * (frequency(w_pos, w_neg) - 0.5) + 0.5
