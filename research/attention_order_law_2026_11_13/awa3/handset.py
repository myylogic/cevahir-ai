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

``check_training_batch`` applies the same hand-set operators to a TRAINING batch read back from its token
tensor (PROTOCOL section 5.1): a check of the layout, the supervised positions, the visibility and the labels
the trainer consumes, without any training.
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


def readout(prog: dict, order: Sequence[int], K: int = 8) -> Dict[str, int]:
    """The token each operator puts its largest readout mass on (a variable if the walk is stuck at an alias)."""
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
    return {name: int((weights[0, 0, 0] @ values).argmax()) for name, weights in effective.items()}


def resolve(prog: dict, order: Sequence[int], K: int = 8) -> Dict[str, bool]:
    """Does each operator put its largest readout mass on the correct literal?"""
    return {name: token == prog["answer"] for name, token in readout(prog, order, K).items()}


# ------------------------------------------------------------- the training batch, read back from its tokens
def facts_from_tokens(tokens: Sequence[int], slots: Sequence[int]) -> list:
    """The facts at the program positions ``slots`` of a token sequence as (lhs, rhs), in that order; raises if
    the layout is not <bos> (v = x ;)*."""
    if tokens[0] != task.BOS:
        raise ValueError("a program starts with <bos>")
    facts = []
    for p in slots:
        lhs, eq, rhs, semi = (int(t) for t in tokens[1 + 4 * p: 5 + 4 * p])
        if eq != task.EQ or semi != task.SEMI or not 0 <= lhs < task.N_VAR or not 0 <= rhs < task.N_VAR + task.N_LIT:
            raise ValueError(f"fact {p} is not of the form 'v = x ;'")
        facts.append((lhs, rhs))
    return facts


def check_training_batch(batch, K: int = 8) -> Dict[str, int]:
    """Resolve every row of a training batch from its TOKENS with hand-set weights and compare with its labels.

    For each row the facts the row can see are parsed from the token tensor, the query is read at the supervised
    position (preceded by '?'), and the target of the batch is compared with
    * forward rows: the readout of the three hand-set operators on those facts (program order = token order);
    * inverse rows: the variable whose fact assigns the queried literal.
    Nothing here uses the generator's bookkeeping: it checks the layout, the supervised positions, the visibility
    and the labels of the tensors the trainer consumes. Returns counts; AWA must resolve every forward row,
    the lower-triangular operators exactly the rows whose chain is dependency-sorted ("followable").
    """
    counts = {"forward_rows": 0, "AWA": 0, "AWA-own": 0, "ChaCAL": 0, "followable": 0, "lower_triangular_law": 0,
              "inverse_rows": 0, "inverse": 0}
    for b in range(batch.x.shape[0]):
        tokens = batch.x[b].tolist()
        for r, row in enumerate(batch.rows.tolist()):
            if tokens[row - 1] != task.QM:
                raise ValueError("a supervised position must follow '?'")
            facts = facts_from_tokens(tokens, batch.visible[b, r].nonzero().flatten().tolist())
            query, target = tokens[row], int(batch.y[b, r])
            if int(batch.kinds[r]) == task.INVERSE:
                counts["inverse_rows"] += 1
                counts["inverse"] += [lhs for lhs, rhs in facts if rhs == query] == [target]
                continue
            position = {lhs: p for p, (lhs, _) in enumerate(facts)}
            chain, v = [], query                                    # positions of the facts the query walks through
            while v < task.N_VAR:
                chain.append(position[v])
                v = facts[position[v]][1]
            followable = all(chain[m] > chain[m + 1] for m in range(len(chain) - 1))
            prog = {"facts": [(0, 0, lhs, rhs) for lhs, rhs in facts], "qvar": query}
            tokens_out = readout(prog, range(len(facts)), K)
            counts["forward_rows"] += 1
            counts["followable"] += followable
            for name, token in tokens_out.items():
                counts[name] += token == target
            counts["lower_triangular_law"] += (tokens_out["AWA-own"] == target) == followable == \
                (tokens_out["ChaCAL"] == target)
    return counts


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
