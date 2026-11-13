# -*- coding: utf-8 -*-
"""The order law with hand-set weights (PROTOCOL section 4): no training, the engine operators only.

Idealised binding (assumption B): one position per fact, in program order, plus the query position. Position j
carries an address for its LHS variable, a pointer to its RHS variable (aliases) and a value for its RHS. The
walk continues at aliases and stops at literal facts. With these weights

* AWA (query-window walk) resolves every chain of depth <= K + 1 in every order;
* AWA-own (own-window walk) and the ChaCAL-style resolvent resolve a chain iff its aliases are
  dependency-sorted, i.e. iff every hop goes backwards in the text. Under uniform shuffling that happens with
  probability exactly 1 / (k - 1)!; one adjacent transposition already breaks the chain.

The operators are the engine's (``absorbing_walk``, ``chacal_resolvent``); this file only sets the weights.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch

from src.neural_network_module.ortak_katman_module.attention_manager_module.absorbing_walk import (
    absorbing_walk,
    chacal_resolvent,
)

from . import task

OPERATORS = ("AWA", "AWA-own", "ChaCAL")


def fact_level(prog: dict, order: Sequence[int], beta: float = 30.0, cap: float = 20.0, tau: float = 9.0,
               lam: float = 0.999):
    """Hand-set quantities of one program: (a_query, E, c, null, values, A_full)."""
    M = len(order)
    n = M + 1
    seq = [prog["facts"][fi] for fi in order]
    address = torch.zeros(n, task.N_VAR, dtype=torch.float64)
    pointer = torch.zeros(n, task.N_VAR, dtype=torch.float64)
    values = torch.zeros(n, task.N_VAR + task.N_LIT, dtype=torch.float64)
    cont = torch.zeros(n, dtype=torch.float64)
    logits = torch.full((n, n), float("-inf"), dtype=torch.float64)           # the resolvent's attention logits
    for j, (_, _, lhs, rhs) in enumerate(seq):
        address[j, lhs] = 1.0
        values[j, rhs] = 1.0
        if rhs < task.N_VAR:
            pointer[j, rhs] = beta
            cont[j] = lam
    query = torch.zeros(task.N_VAR, dtype=torch.float64)
    query[prog["qvar"]] = beta
    a_query = torch.softmax(address @ query, -1)                               # the query sees every fact
    for j, (_, _, lhs, rhs) in enumerate(seq):
        logits[j, : j + 1] = 0.0
        hit = [l for l in range(j + 1) if rhs < task.N_VAR and seq[l][2] == rhs]
        logits[j, hit[0] if hit else j] = beta                                 # successor if visible, else self
    logits[M] = address @ query
    A_full = torch.softmax(logits, -1)
    scores = torch.tanh((pointer @ address.T) / cap) * cap
    E = torch.exp(scores).masked_fill(torch.eye(n, dtype=torch.bool), 0.0)
    null = torch.tensor([math.exp(tau)], dtype=torch.float64)
    return a_query, E, cont, null, values, A_full


def resolve(prog: dict, order: Sequence[int], K: int = 8) -> Dict[str, bool]:
    """Does each operator put its largest readout mass on the correct literal?"""
    a_query, E, cont, null, values, A_full = fact_level(prog, order)
    n = a_query.numel()
    start, E4, c3 = a_query.view(1, 1, 1, n), E.view(1, 1, n, n), cont.view(1, 1, n)
    causal = torch.ones(n, n, dtype=torch.bool).tril().view(1, 1, n, n)
    rows = torch.tensor([[n - 1]])
    effective = {
        "AWA": absorbing_walk(start, E4, c3, null, K, allowed=torch.ones(1, 1, 1, n, dtype=torch.bool)),
        "AWA-own": absorbing_walk(start, E4, c3, null, K, own_window=causal),
        "ChaCAL": chacal_resolvent(A_full.view(1, 1, n, n), torch.tensor([0.999], dtype=torch.float64), rows),
    }
    return {name: int((weights[0, 0, 0] @ values).argmax()) == prog["answer"] for name, weights in effective.items()}


def order_law_table(depths: Sequence[int] = range(1, 10), trials: int = 150, K: int = 8,
                    split: str = "unit-orderlaw") -> Dict[Tuple[int, str], dict]:
    """Accuracy of the three operators per (depth, stratum), with the dependency-sorted share of the items."""
    table: Dict[Tuple[int, str], dict] = {}
    for k in depths:
        progs = task.item_set(split, k, trials)
        for stratum in task.STRATA:
            hits = {name: 0 for name in OPERATORS}
            followable = 0
            for prog in progs:
                order = task.stratum_order(prog, stratum)
                followable += task.followable_lower_triangular(prog, order)
                for name, ok in resolve(prog, order, K).items():
                    hits[name] += ok
            table[(k, stratum)] = {**{name: hits[name] / trials for name in OPERATORS},
                                   "followable": followable / trials}
    return table


def sorted_probability(k: int) -> float:
    """Probability that k - 1 uniformly shuffled aliases are dependency-sorted: 1 / (k - 1)!."""
    return 1.0 / math.factorial(max(0, k - 1))
