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

Test and validation items are single-query programs in exactly this format. A TRAINING sequence is such a
program followed by more query pairs "? x" (PROTOCOL section 5.1, ``make_batch``): its own query, further
forward queries, inverse queries ("? n" -> the variable assigned n) and key-value rows that see only the first
m facts; an attention mask keeps every query pair from seeing any other pair. Loss and accuracy are computed
over the answer alphabet (the 16 literals for "? v").

All streams are string-keyed ``random.Random`` instances (``train|<seed>``, ``budget|<seed>``, ``test|k=<k>``,
``val|k=<k>``); nothing here uses torch's or the global random state.
"""
from __future__ import annotations

import hashlib
import random
from typing import Dict, List, NamedTuple, Sequence, Tuple

import torch
import torch.nn.functional as F

N_VAR, N_LIT = 48, 16
EQ, SEMI, QM, BOS, EOS, PAD = 64, 65, 66, 67, 68, 69
VOCAB = 72
M_FACTS = 16
SEQ_LEN = 1 + 4 * M_FACTS + 2                 # 67, identical at every depth
ANSWER_POS = SEQ_LEN - 1
FACT_TOKENS = 1 + 4 * M_FACTS                 # 65: <bos> and the 16 facts; query pairs "? x" follow
LITERALS = slice(N_VAR, N_VAR + N_LIT)        # answer alphabet of a forward query "? v"
VARIABLES = slice(0, N_VAR)                   # answer alphabet of an inverse query "? n"
FORWARD, INVERSE = 0, 1                       # kinds of query rows
MIN_CHAINS = 6                                # every training program has at least ceil(16 / 3) chains (literals)
SIGNAL_GROUPS: Tuple[str, ...] = ("forward", "inverse", "prefix_forward", "prefix_inverse", "chain")
# Query rows per training program, in SIGNAL_GROUPS order. Chain rows exist and are tested but are off: they did
# not make any arm learn a second hop in 3,000 steps (PROTOCOL section 15), and the measured learning curves of
# the protocol belong to this signal.
TRAIN_SIGNAL: Tuple[int, ...] = (8, 4, 4, 4, 0)
CHAIN_EXTRA: Tuple[int, int] = (1, 4)         # a chain row sees its own path plus U{1..4} other literal facts
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


# ------------------------------------------------------------------------ readout over the answer alphabet
def answer_scores(logits: torch.Tensor, kind: int = FORWARD) -> torch.Tensor:
    """The scores of the answer alphabet: the 16 literals (forward query) or the 48 variables (inverse query).

    The answer of "? v" is always a literal, so loss and accuracy are computed over the literal classes only, in
    training, validation and evaluation alike (PROTOCOL section 5.1). A loss over all 72 classes makes "is a
    literal" the first thing a tied-embedding model learns and collapses the token embeddings onto that one
    direction (measured: section 15).
    """
    return logits[..., LITERALS if kind == FORWARD else VARIABLES]


def predict(logits: torch.Tensor, kind: int = FORWARD) -> torch.Tensor:
    """Predicted token: the arg-max over the answer alphabet."""
    return answer_scores(logits, kind).argmax(-1) + (N_VAR if kind == FORWARD else 0)


def answer_loss(logits: torch.Tensor, y: torch.Tensor, kind: int = FORWARD, reduction: str = "mean") -> torch.Tensor:
    """Cross-entropy over the answer alphabet; ``y`` holds token ids."""
    scores = answer_scores(logits, kind)
    return F.cross_entropy(scores.reshape(-1, scores.shape[-1]), (y - (N_VAR if kind == FORWARD else 0)).reshape(-1),
                           reduction=reduction)


def signal_loss(logits: torch.Tensor, y: torch.Tensor, kinds: torch.Tensor) -> torch.Tensor:
    """Training loss: mean over all query rows (B, R) of the cross-entropy over each row's answer alphabet."""
    total = logits.new_zeros(())
    for kind in (FORWARD, INVERSE):
        sel = kinds == kind
        if bool(sel.any()):
            total = total + answer_loss(logits[:, sel], y[:, sel], kind, reduction="sum")
    return total / y.numel()


# --------------------------------------------------------------------------- training signal (section 5.1)
class TrainBatch(NamedTuple):
    """One training batch. Every program is a test-format program followed by more query pairs "? x".

    x        (B, n) tokens, n = 65 + 2 R: <bos>, the 16 facts, then R query pairs.
    rows     (R,)   supervised positions: the query token of every pair.
    y        (B, R) target token of every row.
    kinds    (R,)   FORWARD ("? v" -> the literal v resolves to) or INVERSE ("? n" -> the variable assigned n).
    visible  (B, R, 16) bool: the facts (by program position) the pair can see; all 16 for a full row.
    depth    (B, R) dereferencing depth of a forward row (0 for inverse rows).
    mask     (B, 1, n, n) bool, True = blocked: a pair sees <bos>, its visible facts and itself, nothing else.
    """
    x: torch.Tensor
    rows: torch.Tensor
    y: torch.Tensor
    kinds: torch.Tensor
    visible: torch.Tensor
    depth: torch.Tensor
    mask: torch.Tensor


def signal_rows(signal: Sequence[int] = TRAIN_SIGNAL) -> int:
    """Number of query rows of a training program; ``signal`` gives the rows per group, in SIGNAL_GROUPS order."""
    signal = tuple(int(v) for v in signal)
    if len(signal) != len(SIGNAL_GROUPS) or signal[0] < 1 or min(signal) < 0:
        raise ValueError(f"signal = rows of {SIGNAL_GROUPS}: forward >= 1, the others >= 0")
    if signal[0] > M_FACTS or signal[1] > MIN_CHAINS:
        raise ValueError(f"at most {M_FACTS} forward and {MIN_CHAINS} inverse queries per program")
    return sum(signal)


def signal_kinds(signal: Sequence[int] = TRAIN_SIGNAL) -> List[int]:
    forward, inverse, prefix_forward, prefix_inverse, chain = (int(v) for v in signal)
    return ([FORWARD] * forward + [INVERSE] * inverse + [FORWARD] * prefix_forward + [INVERSE] * prefix_inverse
            + [FORWARD] * chain)


def signal_groups(signal: Sequence[int] = TRAIN_SIGNAL) -> Dict[str, slice]:
    """Row ranges of the non-empty groups, in row order (SIGNAL_GROUPS)."""
    groups, start = {}, 0
    for name, count in zip(SIGNAL_GROUPS, signal):
        if count:
            groups[name] = slice(start, start + int(count))
        start += int(count)
    return groups


def train_length(signal: Sequence[int] = TRAIN_SIGNAL) -> int:
    """Length of a training sequence: it depends on the signal only, never on a depth."""
    return FACT_TOKENS + 2 * signal_rows(signal)


def resolve_variable(prog: dict, v: int) -> Tuple[int, int]:
    """(literal token, depth) of any variable of the program (``interpret`` for an arbitrary start)."""
    lhs2rhs = {f[2]: f[3] for f in prog["facts"]}
    depth = 0
    while v < N_VAR:
        v, depth = lhs2rhs[v], depth + 1
    return v, depth


def train_program(rng: random.Random, depths: Sequence[int] = TRAIN_DEPTHS,
                  signal: Sequence[int] = TRAIN_SIGNAL) -> dict:
    """One training program: a base program of the training distribution and its query rows.

    Rows, in this order (``signal`` = rows per group, SIGNAL_GROUPS):
    * forward: row 0 is the program's own query, i.e. exactly the test format (head of the queried chain, depth
      k); then forward - 1 further variables, drawn without replacement from the other 15 variables;
    * inverse: literals of the program, drawn without replacement ("? n" -> the variable assigned n);
    * prefix_forward / prefix_inverse: key-value rows. The row sees only the first m facts of the program,
      m ~ U{2..number of literal facts} (the literal block comes first, so these are m key-value pairs), and
      asks for the literal (forward) or the variable (inverse) of a fact drawn uniformly among those m;
    * chain: a small multi-hop program. The variable of a uniformly drawn alias fact (depth 2 or 3; any variable
      if the program has no alias fact), seen with the facts on its path to its literal plus e ~ U{1..4} other
      literal facts (so there is always a wrong literal in sight).
    The first two groups see all 16 facts. The draws come from ``rng`` only, after the base program's own.
    """
    forward, inverse, prefix_forward, prefix_inverse, chain = (int(v) for v in signal)
    signal_rows(signal)
    prog = base_program(rng, rng.choice(tuple(depths)))
    facts, order = prog["facts"], prog["order"]
    literal_ids = order[:prog["n_lit"]]                             # the literal block, in program order
    alias_ids = sorted(order[prog["n_lit"]:])
    if len(literal_ids) < max(inverse, 2):
        raise ValueError("the training signal needs programs with at least max(inverse, 2) chains")
    everything = set(range(len(facts)))
    queries = [(prog["qvar"], prog["answer"], everything, prog["k"])]
    others = [f[2] for f in facts if f[2] != prog["qvar"]]
    for v in rng.sample(others, forward - 1):
        literal, depth = resolve_variable(prog, v)
        queries.append((v, literal, everything, depth))
    for i in rng.sample(sorted(literal_ids), inverse):
        queries.append((facts[i][3], facts[i][2], everything, 0))
    for index in range(prefix_forward + prefix_inverse):
        m = rng.randint(2, len(literal_ids))
        fact = facts[literal_ids[rng.randrange(m)]]
        seen = set(literal_ids[:m])
        queries.append((fact[2], fact[3], seen, 1) if index < prefix_forward else (fact[3], fact[2], seen, 0))
    for _ in range(chain):
        start = rng.choice(alias_ids) if alias_ids else rng.randrange(len(facts))
        path = [i for i, f in enumerate(facts) if f[0] == facts[start][0] and f[1] >= facts[start][1]]
        pool = [i for i in sorted(literal_ids) if i not in path]    # at least 5 literal facts at depths <= 3
        extras = rng.sample(pool, rng.randint(CHAIN_EXTRA[0], min(CHAIN_EXTRA[1], len(pool))))
        literal, depth = resolve_variable(prog, facts[start][2])
        queries.append((facts[start][2], literal, set(path) | set(extras), depth))
    toks = tokens(prog, order)[:FACT_TOKENS]
    for query, _, _, _ in queries:
        toks += [QM, query]
    return dict(prog=prog, tokens=toks, targets=[q[1] for q in queries],
                visible=[[fi in q[2] for fi in order] for q in queries], depth=[q[3] for q in queries])


def query_mask(visible: torch.Tensor) -> torch.Tensor:
    """Attention mask of a training batch, (B, 1, n, n) bool with True = blocked (on top of the causal mask).

    The pair of row r sees <bos>, the facts with ``visible[b, r, p]`` and itself; it sees no other pair and no
    other fact. Fact positions are not restricted. Hence, in every arm, nothing a row computes can reach another
    row, and a row whose facts are a prefix of the program (full and key-value rows) is computed from that
    prefix alone. A chain row sees a scattered subset: the facts it does see were encoded by the binding layer
    in the presence of the earlier facts it does not see.
    """
    B, R, _ = visible.shape
    n = FACT_TOKENS + 2 * R
    pair = torch.arange(2 * R) // 2                                                  # pair of every tail position
    fact = (torch.arange(1, FACT_TOKENS) - 1) // 4                                   # fact of every fact token
    blocked = torch.zeros(B, n, n, dtype=torch.bool)
    blocked[:, FACT_TOKENS:, 1:FACT_TOKENS] = ~visible[:, pair][:, :, fact]
    blocked[:, FACT_TOKENS:, FACT_TOKENS:] = (pair.view(-1, 1) != pair.view(1, -1)).unsqueeze(0)
    return blocked.unsqueeze(1)


def make_batch(rng: random.Random, batch: int, depths: Sequence[int] = TRAIN_DEPTHS,
               signal: Sequence[int] = TRAIN_SIGNAL) -> TrainBatch:
    """Training batch (shuffled stratum): depth of the program's own query uniform over ``depths`` per item."""
    items = [train_program(rng, depths, signal) for _ in range(batch)]
    R = signal_rows(signal)
    visible = torch.tensor([it["visible"] for it in items])
    return TrainBatch(x=torch.tensor([it["tokens"] for it in items]),
                      rows=FACT_TOKENS + 1 + 2 * torch.arange(R),
                      y=torch.tensor([it["targets"] for it in items]),
                      kinds=torch.tensor(signal_kinds(signal)), visible=visible,
                      depth=torch.tensor([it["depth"] for it in items]), mask=query_mask(visible))


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


def train_overlap(seed: int, n_programs: int, depths: Sequence[int] = TRAIN_DEPTHS,
                  signal: Sequence[int] = TRAIN_SIGNAL) -> int:
    """How many of the first ``n_programs`` training programs of ``train|seed`` equal a frozen test item
    (the program with its own query, i.e. the first 67 tokens of the training sequence)."""
    keys = set()
    for k in depths:
        for p in eval_set(k):
            for s in strata_of(k):
                keys.add(tuple(tokens(p, stratum_order(p, s))))
    rng = random.Random(f"train|{seed}")
    hits = 0
    for _ in range(n_programs):
        hits += tuple(train_program(rng, depths, signal)["tokens"][:SEQ_LEN]) in keys
    return hits
