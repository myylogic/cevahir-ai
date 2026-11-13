# -*- coding: utf-8 -*-
"""Evaluation of one trained run (PROTOCOL section 8): everything the analysis needs, as exact counts.

One pass per (depth, stratum) gives the whole budget axis: K = 0..24 for the walks, T = 1..16 for LOOPED,
T = 1..24 for ARLOOP; ChaCAL-style and STD have a single (exact) evaluation, stored as budget 0.

Recorded:
* ``test``   depth x stratum x budget: number of correct answers on the frozen test sets;
* ``val``    the same on the oracle-budget validation sets (k >= 4, 300 items per depth and stratum);
* ``zero_operator``  the operator switched off at test time (walks: K = 0, i.e. lambda := 0; ChaCAL-style:
             gamma := 0), on the test sets;
* ``by_disorder``  shuffled stratum: correct answers by the number of out-of-order adjacent aliases;
* ``traces`` (walks) normalised moving mass of hop m on the token span of chain element m - delta, per head and
             delta in {-1, 0, 1}: mean and median over items, on validation and on test items;
* ``gate``   lambda / gamma per head and the mean continue probability c by position type;
* ``macs``   analytic multiply-accumulates per answer along the budget axis;
* ``floors`` h_k and the dependency-sorted fraction P_f of the frozen test items.
"""
from __future__ import annotations

import statistics
import time
from typing import Dict, List, Optional, Sequence

import torch

from src.neural_network_module.ortak_katman_module.attention_manager_module.absorbing_walk import (
    continue_probability,
    layer_forward_rows,
)

from . import arms, task

DISORDER_DEPTHS = (3, 4, 5, 6, 8)
TRACE_DELTAS = (-1, 0, 1)
EVAL_FORMAT = "awa3-eval"


def _budget_axis(arm: str) -> List[int]:
    return list(arms.BUDGET_AXIS[arms.ARMS[arm]["forward"]])


def _correct_per_budget(model, arm: str, X: torch.Tensor, Y: torch.Tensor, chunk: int) -> List[int]:
    forward = arms.ARMS[arm]["forward"]
    axis = _budget_axis(arm)
    counts = [0] * len(axis)
    for i in range(0, X.shape[0], chunk):
        x, y = X[i:i + chunk], Y[i:i + chunk]
        if forward in ("std", "exact"):
            outs = [arms.answer_logits(model, arm, x)]
        else:
            outs = arms.answer_logits(model, arm, x, axis[-1], return_all=True)
        if len(outs) != len(axis):
            raise RuntimeError(f"{arm}: {len(outs)} outputs for a budget axis of {len(axis)}")
        for b, logits in enumerate(outs):
            counts[b] += int((logits.argmax(-1) == y).sum())
    return counts


def _grid(model, arm: str, sets: Dict[int, Sequence[dict]], chunk: int) -> dict:
    out: Dict[str, dict] = {}
    for k, progs in sets.items():
        cell: Dict[str, dict] = {}
        for stratum in task.strata_of(k):
            X, Y = task.tensors(progs, stratum)
            cell[stratum] = {"n": len(progs), "correct": _correct_per_budget(model, arm, X, Y, chunk)}
        for stratum in task.STRATA:                               # k <= 2: the four strata are the same items
            cell.setdefault(stratum, dict(cell["shuffled"]))
        out[str(k)] = cell
    return out


def _zero_operator(model, arm: str, sets: Dict[int, Sequence[dict]], chunk: int) -> Optional[dict]:
    """Accuracy with the operator off at test time (the mechanism ablation H4(b))."""
    forward = arms.ARMS[arm]["forward"]
    if forward not in ("walk", "exact"):
        return None
    attn = model.layers[-1].attn
    out: Dict[str, dict] = {}
    for k, progs in sets.items():
        cell: Dict[str, dict] = {}
        for stratum in task.strata_of(k):
            X, Y = task.tensors(progs, stratum)
            correct = 0
            for i in range(0, X.shape[0], chunk):
                if forward == "walk":
                    logits = arms.answer_logits(model, arm, X[i:i + chunk], 0)       # K = 0: lambda := 0
                else:
                    attn.attention_operator = "standard"                               # gamma := 0
                    try:
                        logits = model.forward_rows(X[i:i + chunk], torch.tensor([X.shape[1] - 1]))[:, 0]
                    finally:
                        attn.attention_operator = "chacal"
                correct += int((logits.argmax(-1) == Y[i:i + chunk]).sum())
            cell[stratum] = {"n": len(progs), "correct": correct}
        for stratum in task.STRATA:
            cell.setdefault(stratum, dict(cell["shuffled"]))
        out[str(k)] = cell
    return out


def _by_disorder(model, arm: str, sets: Dict[int, Sequence[dict]], chunk: int) -> dict:
    forward = arms.ARMS[arm]["forward"]
    budget = arms.TEST_BUDGET if forward in ("walk", "looped", "arloop") else None
    out: Dict[str, dict] = {}
    for k in DISORDER_DEPTHS:
        if k not in sets:
            continue
        progs = sets[k]
        X, Y = task.tensors(progs, "shuffled")
        hits = torch.cat([arms.answer_logits(model, arm, X[i:i + chunk], budget).argmax(-1) == Y[i:i + chunk]
                          for i in range(0, X.shape[0], chunk)])
        groups: Dict[str, List[int]] = {}
        for p, hit in zip(progs, hits.tolist()):
            g = groups.setdefault(str(task.out_of_order_adjacent(p, p["order"])), [0, 0])
            g[0] += 1
            g[1] += int(hit)
        out[str(k)] = {key: {"n": n, "correct": c} for key, (n, c) in sorted(groups.items(), key=lambda kv: int(kv[0]))}
    return out


def _last_layer_input(model, x: torch.Tensor) -> torch.Tensor:
    h = model.embed_dropout(model.pos_encoding(model.embedding(x)))
    for layer in model.layers[:-1]:
        h = layer(h, causal_mask=True)[0]
    return h


def _traces(model, progs_by_depth: Dict[int, Sequence[dict]], hops: int) -> dict:
    """Normalised moving mass of hop m on the span of chain element m - delta (walk arms).

    Result[k][stratum] = {"n", "mean": [head][delta][m], "median": [head][delta][m]} for m = 0..k-1; entries
    whose element index m - delta falls outside the chain are None. The extra stratum key "pooled" holds the
    same statistics over the items of all four strata together.
    """
    layer = model.layers[-1]
    H = layer.attn.num_heads
    out: Dict[str, dict] = {}
    for k, progs in progs_by_depth.items():
        out[str(k)] = {}
        pooled: Dict[tuple, List[float]] = {}
        for stratum in task.STRATA:
            X, _ = task.tensors(progs, stratum)
            trace: List[torch.Tensor] = []
            layer_forward_rows(layer, _last_layer_input(model, X), torch.tensor([X.shape[1] - 1]),
                               num_hops=hops, trace=trace)
            # mass[item][head][hop][element]
            spans = [task.chain_spans(p, task.stratum_order(p, stratum)) for p in progs]
            values: Dict[tuple, List[float]] = {}
            for m in range(k):
                pi = trace[m][:, :, 0, :].double()                               # (B, H, S)
                norm = pi.sum(-1, keepdim=True)
                pi = torch.where(norm > 1e-12, pi / norm.clamp_min(1e-300), torch.zeros_like(pi))
                for di, delta in enumerate(TRACE_DELTAS):
                    e = m - delta
                    if not 0 <= e < k:
                        continue
                    index = torch.tensor([[spans[b][e].start + j for j in range(4)] for b in range(len(progs))])
                    mass = pi.gather(2, index.view(len(progs), 1, 4).expand(len(progs), H, 4)).sum(-1)   # (B, H)
                    for h in range(H):
                        values[(h, di, m)] = mass[:, h].tolist()

            for key, v in values.items():
                pooled.setdefault(key, []).extend(v)
            out[str(k)][stratum] = _trace_tables(values, len(progs), k, H)
        out[str(k)]["pooled"] = _trace_tables(pooled, len(progs) * len(task.STRATA), k, H)
    return out


def _trace_tables(values: Dict[tuple, List[float]], n: int, k: int, H: int) -> dict:
    def table(fn):
        return [[[fn(values[(h, di, m)]) if (h, di, m) in values else None for m in range(k)]
                 for di in range(len(TRACE_DELTAS))] for h in range(H)]

    return {"n": n, "mean": table(statistics.fmean), "median": table(statistics.median)}


def _gate(model, arm: str, progs: Sequence[dict]) -> dict:
    out = dict(arms.operator_readout(model, arm))
    if arms.ARMS[arm]["forward"] != "walk":
        return out
    layer = model.layers[-1]
    X, _ = task.tensors(progs, "shuffled")
    c = continue_probability(layer.attn, layer.norm1(_last_layer_input(model, X))).double()      # (B, H, S)
    sums = {t: torch.zeros(c.shape[1], dtype=torch.float64) for t in task.POSITION_TYPES}
    counts = {t: 0 for t in task.POSITION_TYPES}
    for b, p in enumerate(progs):
        for s, t in enumerate(task.position_types(p, p["order"])):
            sums[t] += c[b, :, s]
            counts[t] += 1
    out["mean_c"] = {t: [float(v) for v in sums[t] / max(1, counts[t])] for t in task.POSITION_TYPES}
    return out


@torch.no_grad()
def evaluate_run(model, arm: str, *, depths: Sequence[int] = task.TEST_DEPTHS,
                 val_depths: Sequence[int] = task.VAL_ORACLE_DEPTHS, trace_depths: Sequence[int] = task.TRACE_DEPTHS,
                 limit: Optional[int] = None, chunk: int = 250, progress=None) -> dict:
    """Evaluate a trained model. ``limit`` truncates every set to its first ``limit`` items (smoke runs only;
    the truncation is recorded and the analysis refuses truncated records unless told otherwise)."""
    torch.set_num_threads(1)
    model.eval()
    t0 = time.process_time()

    def cut(progs):
        return progs if limit is None else progs[:limit]

    def say(message):
        if progress is not None:
            progress(message)

    test_sets = {k: cut(task.eval_set(k)) for k in depths}
    val_sets = {k: cut(task.val_set(k)) for k in val_depths}
    forward = arms.ARMS[arm]["forward"]
    axis = _budget_axis(arm)
    record = {
        "format": EVAL_FORMAT, "arm": arm, "forward": forward, "budgets": axis,
        "truncated_to": limit,
        "floors": {str(k): task.heuristic_floors(test_sets[k], "shuffled")["uniform"] for k in depths},
        "followable": {str(k): task.heuristic_floors(test_sets[k], "shuffled")["lowertri_followable"]
                       for k in depths},
        "macs": {str(b): arms.macs_per_answer(arm, b) for b in axis},
    }
    say(f"{arm}: test grid")
    record["test"] = _grid(model, arm, test_sets, chunk)
    say(f"{arm}: validation grid")
    record["val"] = _grid(model, arm, val_sets, chunk)
    record["zero_operator"] = _zero_operator(model, arm, test_sets, chunk)
    record["by_disorder"] = _by_disorder(model, arm, test_sets, chunk)
    gate_items = [p for k in trace_depths if k in test_sets for p in test_sets[k][:task.TRACE_ITEMS]]
    record["gate"] = _gate(model, arm, gate_items) if gate_items else dict(arms.operator_readout(model, arm))
    if forward == "walk":
        say(f"{arm}: hop traces")
        n_trace = task.TRACE_ITEMS if limit is None else min(limit, task.TRACE_ITEMS)
        record["traces"] = {
            "hops": arms.TEST_BUDGET, "deltas": list(TRACE_DELTAS),
            "val": _traces(model, {k: task.val_set(k)[:n_trace] for k in trace_depths}, arms.TEST_BUDGET),
            "test": _traces(model, {k: task.eval_set(k)[:n_trace] for k in trace_depths}, arms.TEST_BUDGET),
        }
    else:
        record["traces"] = None
    record["timing"] = {"cpu_s": time.process_time() - t0}
    return record
