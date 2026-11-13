# -*- coding: utf-8 -*-
"""VarDeref v2: synthetic variable dereferencing with four PAIRED order strata (PROTOCOL section 5).

Program = <bos> [literal block: one "v = n ;" per chain, shuffled] [alias block: every "v = w ;", shuffled] ? q
Target  = the literal that q resolves to, predicted at the last position.

* 72-token vocabulary: v0..v47 (variables), n0..n15 (literals), '=', ';', '?', <bos>, <eos>, <pad>.
* M = 16 facts in every program, so the length is 67 tokens at every depth (length is not confounded with depth).
* The queried chain has depth k (k - 1 aliases + one literal fact); distractor chains have depth U{1, 2, 3};
  every chain ends in a distinct literal.
* Strata re-order ONLY the queried chain's aliases among the program positions they already occupy:
  sorted (definitions first), oneswap (sorted with one adjacent transposition), anti (reading order),
  shuffled (the base order = the training distribution). The literal block and the distractors never move, so
  every guess that ignores the chain (uniform over literals, first literal, last literal) scores the same in
  all four strata: the chance floor h_k is identical across strata.

All streams are string-keyed ``random.Random`` instances (``train|<seed>``, ``budget|<seed>``, ``test|k=<k>``,
``val|k=<k>``); nothing here uses torch's or the global random state.
"""
from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Sequence, Tuple

import torch

N_VAR, N_LIT = 48, 16
EQ, SEMI, QM, BOS, EOS, PAD = 64, 65, 66, 67, 68, 69
VOCAB = 72
M_FACTS = 16
SEQ_LEN = 1 + 4 * M_FACTS + 2                 # 67, identical at every depth
ANSWER_POS = SEQ_LEN - 1
STRATA: Tuple[str, ...] = ("sorted", "oneswap", "anti", "shuffled")
TRAIN_DEPTHS: Tuple[int, ...] = (1, 2, 3)
TEST_DEPTHS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 12)
ENDPOINT_DEPTHS: Tuple[int, ...] = (6, 8)
VAL_ID_ITEMS = 600                            # in-distribution validation items per depth (k = 1, 2, 3)
VAL_ORACLE_ITEMS = 300                        # oracle-budget validation items per depth (k >= 4)
VAL_ORACLE_DEPTHS: Tuple[int, ...] = (4, 5, 6, 8, 10, 12)
TRACE_DEPTHS: Tuple[int, ...] = (3, 4, 6)
TRACE_ITEMS = 200


def eval_size(k: int) -> int:
    """Frozen number of base programs per depth: 1,000 at the endpoint depths, 500 elsewhere."""
    return 1000 if k in ENDPOINT_DEPTHS else 500


# ------------------------------------------------------------------------------------- generator
def base_program(rng: random.Random, k: int, M: int = M_FACTS, dmax: int = 3) -> dict:
    """One base program: a queried chain of depth k plus distractor chains of depth U{1..dmax}.

    facts[i] = (chain index, link index, lhs variable, rhs) with rhs < N_VAR a variable and rhs >= N_VAR a
    literal token; ``order`` is the base (shuffled) program order: literal block, then alias block.
    """
    if not 1 <= k <= M:
        raise ValueError(f"depth k must be in 1..{M}")
    vars_ = rng.sample(range(N_VAR), N_VAR)
    lits = rng.sample(range(N_LIT), N_LIT)
    chains, used, left = [vars_[:k]], k, M - k
    while left > 0:
        dd = min(rng.randint(1, dmax), left)
        chains.append(vars_[used:used + dd])
        used, left = used + dd, left - dd
    facts = []
    for ci, ch in enumerate(chains):
        for m, v in enumerate(ch):
            facts.append((ci, m, v, ch[m + 1] if m + 1 < len(ch) else N_VAR + lits[ci]))
    lit_ids = [i for i, f in enumerate(facts) if f[3] >= N_VAR]
    ali_ids = [i for i, f in enumerate(facts) if f[3] < N_VAR]
    rng.shuffle(lit_ids)
    rng.shuffle(ali_ids)
    return dict(k=k, chains=chains, facts=facts, order=lit_ids + ali_ids, n_lit=len(lit_ids),
                swap_at=rng.randrange(k - 2) if k >= 3 else None,
                answer=N_VAR + lits[0], qvar=chains[0][0], lits=[N_VAR + lits[i] for i in range(len(chains))])


def query_aliases(prog: dict) -> List[int]:
    """Fact indices of the queried chain's aliases, in chain order (links 0..k-2)."""
    return [i for i, f in enumerate(prog["facts"]) if f[0] == 0 and f[3] < N_VAR]


def chain_facts(prog: dict) -> List[int]:
    """Fact indices of the queried chain in chain order: aliases 0..k-2, then the literal fact."""
    return [i for i, f in enumerate(prog["facts"]) if f[0] == 0]


def stratum_order(prog: dict, stratum: str) -> List[int]:
    """Program order (fact index at each program position) of one stratum."""
    if stratum not in STRATA:
        raise ValueError(f"unknown stratum {stratum!r}")
    order, k = list(prog["order"]), prog["k"]
    if stratum == "shuffled" or k <= 2:
        return order
    pos = {fi: p for p, fi in enumerate(order)}
    qa = query_aliases(prog)
    slots = sorted(pos[i] for i in qa)
    if stratum in ("sorted", "oneswap"):                       # alias m after alias m+1: definitions first
        assign = {qa[m]: slots[k - 2 - m] for m in range(k - 1)}
        if stratum == "oneswap":
            a, b = qa[prog["swap_at"]], qa[prog["swap_at"] + 1]
            assign[a], assign[b] = assign[b], assign[a]
    else:                                                      # anti: reading order
        assign = {qa[m]: slots[m] for m in range(k - 1)}
    for fi, p in assign.items():
        order[p] = fi
    return order


def followable_lower_triangular(prog: dict, order: Sequence[int]) -> bool:
    """True iff every hop of the queried chain goes backwards in the text (aliases dependency-sorted).

    The last hop, into the literal block, always goes backwards.
    """
    pos = {fi: p for p, fi in enumerate(order)}
    q = [pos[i] for i in query_aliases(prog)]
    return all(q[m] > q[m + 1] for m in range(len(q) - 1))


def out_of_order_adjacent(prog: dict, order: Sequence[int]) -> int:
    """Number of adjacent alias pairs (m, m+1) whose hop goes forwards in the text (0 = dependency-sorted)."""
    pos = {fi: p for p, fi in enumerate(order)}
    q = [pos[i] for i in query_aliases(prog)]
    return sum(q[m] < q[m + 1] for m in range(len(q) - 1))


def tokens(prog: dict, order: Sequence[int]) -> List[int]:
    toks = [BOS]
    for fi in order:
        _, _, lhs, rhs = prog["facts"][fi]
        toks += [lhs, EQ, rhs, SEMI]
    return toks + [QM, prog["qvar"]]


def fact_span(p: int) -> range:
    """Token positions of the fact at program position p: lhs, '=', rhs, ';'."""
    return range(1 + 4 * p, 5 + 4 * p)


def chain_spans(prog: dict, order: Sequence[int]) -> List[range]:
    """Token spans of the queried chain's facts in chain order (element 0 = the fact whose LHS is the query)."""
    pos = {fi: p for p, fi in enumerate(order)}
    return [fact_span(pos[fi]) for fi in chain_facts(prog)]


POSITION_TYPES: Tuple[str, ...] = ("rhs_var", "rhs_lit", "semi", "other")


def position_types(prog: dict, order: Sequence[int]) -> List[str]:
    """Type of every token position: RHS of an alias, RHS of a literal fact, ';', or other."""
    types = ["other"]
    for fi in order:
        rhs = prog["facts"][fi][3]
        types += ["other", "other", "rhs_var" if rhs < N_VAR else "rhs_lit", "semi"]
    return types + ["other", "other"]


def interpret(prog: dict) -> Tuple[int, int]:
    """(literal token, depth) obtained by following the program's assignments from the query variable."""
    lhs2rhs = {f[2]: f[3] for f in prog["facts"]}
    v, depth = prog["qvar"], 0
    while v < N_VAR:
        v, depth = lhs2rhs[v], depth + 1
    return v, depth


# ----------------------------------------------------------------------------------- batches, sets
def make_batch(rng: random.Random, batch: int, depths: Sequence[int] = TRAIN_DEPTHS,
               stratum: str = "shuffled") -> Tuple[torch.Tensor, torch.Tensor]:
    """Training batch: depth uniform over ``depths`` per item, the given stratum (training = shuffled)."""
    X, Y = [], []
    for _ in range(batch):
        prog = base_program(rng, rng.choice(tuple(depths)))
        X.append(tokens(prog, stratum_order(prog, stratum)))
        Y.append(prog["answer"])
    return torch.tensor(X), torch.tensor(Y)


def item_set(split: str, k: int, n_items: int) -> List[dict]:
    """Frozen, string-keyed base programs (never collides with the ``train|seed`` streams); strata are paired."""
    rng = random.Random(f"{split}|k={k}")
    return [base_program(rng, k) for _ in range(n_items)]


def eval_set(k: int) -> List[dict]:
    return item_set("test", k, eval_size(k))


def val_set(k: int) -> List[dict]:
    return item_set("val", k, VAL_ID_ITEMS if k in TRAIN_DEPTHS else VAL_ORACLE_ITEMS)


def tensors(progs: Sequence[dict], stratum: str) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor([tokens(p, stratum_order(p, stratum)) for p in progs])
    Y = torch.tensor([p["answer"] for p in progs])
    return X, Y


def strata_of(k: int) -> Tuple[str, ...]:
    """Distinct strata at depth k: all four coincide at k <= 2 (reported as 'shuffled' for every stratum)."""
    return STRATA if k >= 3 else ("shuffled",)


def set_sha256(progs: Sequence[dict]) -> str:
    """Digest of a frozen set: the token sequences of all four strata and the answers, in order."""
    h = hashlib.sha256()
    for p in progs:
        for s in STRATA:
            h.update(bytes(tokens(p, stratum_order(p, s))))
        h.update(bytes([p["answer"]]))
    return h.hexdigest()


def frozen_set_hashes() -> Dict[str, str]:
    """Digests of every evaluation and validation set of the protocol (written into the freeze manifest)."""
    out = {f"test|k={k}|n={eval_size(k)}": set_sha256(eval_set(k)) for k in TEST_DEPTHS}
    for k in TRAIN_DEPTHS + VAL_ORACLE_DEPTHS:
        progs = val_set(k)
        out[f"val|k={k}|n={len(progs)}"] = set_sha256(progs)
    return out


# ------------------------------------------------------------------------------------------ floors
def heuristic_floors(progs: Sequence[dict], stratum: str) -> dict:
    """Accuracy of answer heuristics that ignore the chain, plus the dependency-sorted fraction.

    uniform: uniform over the distinct literals present (= h_k); last / first: the last / first literal fact.
    """
    u = last = first = foll = 0.0
    for p in progs:
        order = stratum_order(p, stratum)
        lit_facts = [p["facts"][fi][3] for fi in order if p["facts"][fi][3] >= N_VAR]
        u += 1.0 / len(set(lit_facts))
        last += lit_facts[-1] == p["answer"]
        first += lit_facts[0] == p["answer"]
        foll += followable_lower_triangular(p, order)
    n = len(progs)
    return dict(uniform=u / n, last=last / n, first=first / n, lowertri_followable=foll / n)


def floor(k: int) -> float:
    """h_k: uniform-over-literals accuracy on the frozen test set of depth k (identical in all strata)."""
    return heuristic_floors(eval_set(k), "shuffled")["uniform"]


def followable_fraction(k: int) -> float:
    """P_f: share of the frozen shuffled test items of depth k whose aliases are dependency-sorted."""
    return heuristic_floors(eval_set(k), "shuffled")["lowertri_followable"]


def train_overlap(seed: int, n_programs: int, depths: Sequence[int] = TRAIN_DEPTHS) -> int:
    """How many of the first ``n_programs`` training programs of ``train|seed`` equal a frozen test item."""
    keys = set()
    for k in depths:
        for p in eval_set(k):
            for s in strata_of(k):
                keys.add(tuple(tokens(p, stratum_order(p, s))))
    rng = random.Random(f"train|{seed}")
    hits = 0
    for _ in range(n_programs):
        p = base_program(rng, rng.choice(tuple(depths)))
        hits += tuple(tokens(p, p["order"])) in keys
    return hits
