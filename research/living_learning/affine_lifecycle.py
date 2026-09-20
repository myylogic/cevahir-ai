"""A finite, falsifiable living-learning witness, not a proposed Cevahir backend.

Only Python's standard library is used. No engine, model, dataset, or network is
loaded. Run from the repository root:
    python -m research.living_learning.affine_lifecycle

World family and composition semantics are supplied inductive bias. The learned
coefficients are NOT supplied. Work counts are abstract field operations, not
CPU instructions or timings. Assertions check substantive scientific controls.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

P = 101
OPS = tuple(f"tool_{i}" for i in range(4))


def apply_rule(rule, x):
    return (rule[0] * x + rule[1]) % P


def compose(left, right):
    """Return right(left(x)); one counted composition operation."""
    return right[0] * left[0] % P, (right[0] * left[1] + right[1]) % P


class Learner:
    def __init__(self):
        self.rules = {}
        self.pending = {}
        self.versions = {op: 0 for op in OPS}
        self.macros = {}
        self.events = []

    def observe(self, op, x, y):
        """Trusted deterministic feedback; at most two uncommitted samples/op.

        Fit two distinct inputs; the third distinct input must agree before
        promotion. This is a finite consistency guard, NOT a noise guarantee.
        A trusted contradiction revokes the rule and dependent compiled code.
        """
        if op in self.rules:
            if apply_rule(self.rules[op], x) == y:
                return "consistent"
            del self.rules[op]
            self.versions[op] += 1
            self.macros = {seq: m for seq, m in self.macros.items() if op not in seq}
            self.pending[op] = []
            self.events.append({"type": "revoke", "op": op})
        points = self.pending.setdefault(op, [])
        if any(px == x for px, _ in points):
            if any(px == x and py != y for px, py in points):
                self.pending[op] = [(x, y)]
                return "inconsistent_duplicate"
            return "duplicate"
        if len(points) < 2:
            points.append((x, y))
            return "pending"
        (x0, y0), (x1, y1) = points
        a = (y1 - y0) * pow((x1 - x0) % P, -1, P) % P
        rule = (a, (y0 - a * x0) % P)
        if apply_rule(rule, x) != y:
            self.pending[op] = [points[-1], (x, y)]
            return "rejected"
        self.rules[op] = rule
        self.versions[op] += 1
        del self.pending[op]
        self.events.append({"type": "promote", "op": op})
        return "promoted"

    def predict(self, sequence, x, budget=8):
        sequence = tuple(sequence)
        if any(op not in self.rules for op in sequence):
            return None
        macro = self.macros.get(sequence)
        if macro is not None:
            rule, versions = macro
            if versions != {op: self.versions[op] for op in sequence}:
                raise AssertionError("stale compiled dependency")
            return apply_rule(rule, x) if budget >= 1 else None
        if budget < len(sequence):
            return None
        for op in sequence:
            x = apply_rule(self.rules[op], x)
        return x

    def consolidate(self, historical_sequences):
        """Ordinary algebraic compilation from already learned rules only.

        No query inputs/labels or world access. Each sequence is compiled once;
        its dependency versions are retained so revisions invalidate it.
        """
        work = 0
        for sequence in historical_sequences:
            sequence = tuple(sequence)
            if sequence in self.macros or any(op not in self.rules for op in sequence):
                continue
            rule = self.rules[sequence[0]]
            for op in sequence[1:]:
                rule = compose(rule, self.rules[op])
                work += 1
            self.macros[sequence] = (rule, {op: self.versions[op] for op in sequence})
        return work

    def persistent(self):
        """Exclude uncommitted observations, event trace and all raw episodes."""
        return {"prime": P, "rules": self.rules, "versions": self.versions,
                "macros": [{"sequence": list(seq), "rule": rule, "versions": versions}
                           for seq, (rule, versions) in self.macros.items()]}

    @classmethod
    def restore(cls, payload):
        assert payload["prime"] == P
        result = cls()
        result.rules = {op: tuple(rule) for op, rule in payload["rules"].items()}
        result.versions = dict(payload["versions"])
        result.macros = {tuple(m["sequence"]): (tuple(m["rule"]), dict(m["versions"]))
                         for m in payload["macros"]}
        return result


def make_world(rng):
    return {op: (rng.randrange(1, P), rng.randrange(P)) for op in OPS}


def world_answer(world, sequence, x):
    for op in sequence:
        x = apply_rule(world[op], x)
    return x


def teach(world):
    learner = Learner()
    raw = []
    # Predictions precede each feedback; operations are interleaved.
    for x in range(3):
        for op in OPS:
            before = learner.predict((op,), x)
            y = world_answer(world, (op,), x)
            status = learner.observe(op, x, y)
            raw.append({"op": op, "x": x, "y": y, "before": before, "status": status})
    assert not learner.pending
    return learner, raw


def score(predictions, labels):
    return {"correct": sum(a == b for a, b in zip(predictions, labels)),
            "abstentions": sum(a is None for a in predictions), "n": len(labels)}


def predictions(learner, queries, budget=8):
    return [learner.predict(seq, x, budget) for seq, x in queries]


def exact_retrieval(raw, queries):
    table = {(r["op"], r["x"]): r["y"] for r in raw}
    result = []
    for seq, x in queries:
        for op in seq:
            x = table.get((op, x))
            if x is None:
                break
        result.append(x)
    return result


def fresh_process_check(learner, queries):
    # The fresh process receives learned state and unlabelled queries only.
    with tempfile.TemporaryDirectory(prefix="cevahir_living_") as directory:
        path = Path(directory) / "input.json"
        path.write_text(json.dumps({"state": learner.persistent(), "queries": queries}), encoding="utf-8")
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--restore-worker", str(path)],
                               check=True, capture_output=True, text=True, timeout=15)
        data = json.loads(child.stdout)
    assert data["predictions"] == predictions(learner, queries)
    assert data["pending_count"] == data["event_count"] == 0
    return {"equal_predictions": True, "query_count": len(queries),
            "pending_count": data["pending_count"], "event_count": data["event_count"],
            "state_sha256": hashlib.sha256(json.dumps(learner.persistent(), sort_keys=True).encode()).hexdigest()}


def lifecycle_trial(seed):
    rng = random.Random(seed)
    world_a, world_b = make_world(rng), make_world(rng)
    a, raw_a = teach(world_a)
    b, _ = teach(world_b)
    queries = [(seq, x) for seq in [(op,) for op in OPS] + list(itertools.product(OPS, repeat=2))
               for x in range(3, P)]
    # Freeze all predictions before evaluator computes labels.
    pred_a, pred_b = predictions(a, queries), predictions(b, queries)
    pred_reset = predictions(Learner(), queries)
    pred_raw = exact_retrieval(raw_a, queries)
    transplant = Learner.restore(a.persistent())
    pred_transplant = predictions(transplant, queries)
    # Strong baseline: raw memory + the same induction, deferred until evaluation.
    raw_reasoner = Learner()
    for r in raw_a:
        raw_reasoner.observe(r["op"], r["x"], r["y"])
    pred_raw_reasoner = predictions(raw_reasoner, queries)
    labels_a = [world_answer(world_a, seq, x) for seq, x in queries]
    labels_b = [world_answer(world_b, seq, x) for seq, x in queries]
    assert pred_a == labels_a and pred_b == labels_b
    assert pred_a == pred_transplant == pred_raw_reasoner
    assert all(v is None for v in pred_reset + pred_raw)
    assert a.rules == world_a and b.rules == world_b
    return {"seed": seed, "a_on_a": score(pred_a, labels_a), "b_on_b": score(pred_b, labels_b),
            "b_on_a": score(pred_b, labels_a), "reset": score(pred_reset, labels_a),
            "raw_exact_lookup": score(pred_raw, labels_a),
            "raw_plus_same_induction": score(pred_raw_reasoner, labels_a),
            "history_dependent_disagreements": sum(x != y for x, y in zip(pred_a, pred_b)),
            "state_transplant_equal": pred_a == pred_transplant,
            "initial_rules": 0, "final_rules": len(a.rules), "feedback_count": len(raw_a),
            "early_first_eight_predictions_abstain": all(r["before"] is None for r in raw_a[:8]),
            "learned_state_bytes_json": len(json.dumps(a.persistent(), sort_keys=True).encode())}, a, queries


def offline_and_revision(seed=7001):
    rng = random.Random(seed)
    world = make_world(rng)
    online, _ = teach(world)
    sequences = []
    while len(sequences) < 12:
        seq = tuple(rng.choice(OPS) for _ in range(8))
        if seq not in sequences:
            sequences.append(seq)
    # Recorded earlier request shapes, with no future x or labels.
    for seq in sequences:
        assert online.predict(seq, 0, 8) is not None
    sleeping = Learner.restore(online.persistent())
    before_feedback = json.dumps(sleeping.rules, sort_keys=True)
    compile_work = sleeping.consolidate(sequences)
    assert before_feedback == json.dumps(sleeping.rules, sort_keys=True)
    queries = [(seq, x) for seq in sequences for x in range(3, P)]
    awake_tight = predictions(online, queries, 1)
    asleep_tight = predictions(sleeping, queries, 1)
    awake_full = predictions(online, queries, 8)
    lazy = Learner.restore(online.persistent())
    lazy_work = 0
    lazy_preds = []
    for seq, x in queries:
        lazy_work += lazy.consolidate([seq]) + 1
        lazy_preds.append(lazy.predict(seq, x, 1))
    labels = [world_answer(world, seq, x) for seq, x in queries]
    assert asleep_tight == awake_full == lazy_preds == labels
    assert compile_work + len(queries) == lazy_work
    assert all(x is None for x in awake_tight)
    # Revision: changed intercept guarantees that every x contradicts old rule.
    corrected = Learner.restore(sleeping.persistent())
    old_unaffected = {op: corrected.rules[op] for op in OPS[1:]}
    old = corrected.rules[OPS[0]]
    changed_world = dict(world)
    changed_world[OPS[0]] = (old[0], (old[1] + 1) % P)
    corrected.observe(OPS[0], 3, world_answer(changed_world, (OPS[0],), 3))
    revoked = corrected.predict((OPS[0],), 9) is None
    invalidation = all(OPS[0] not in seq for seq in corrected.macros)
    unavailable_until_revalidated = all(corrected.predict(seq, 9) is None for seq in sequences if OPS[0] in seq)
    for x in (4, 5):
        corrected.observe(OPS[0], x, world_answer(changed_world, (OPS[0],), x))
    assert {op: corrected.rules[op] for op in OPS[1:]} == old_unaffected
    corrected.consolidate(sequences)
    revised_preds = predictions(corrected, queries, 1)
    revised_labels = [world_answer(changed_world, seq, x) for seq, x in queries]
    assert revised_preds == revised_labels and revoked and invalidation and unavailable_until_revalidated
    return {"seed": seed, "external_feedback_during_consolidation": 0,
            "historical_sequence_count": len(sequences), "sequence_length": 8,
            "learned_coefficients_unchanged": True, "compile_work": compile_work,
            "no_compilation_budget_1": score(awake_tight, labels),
            "offline_compilation_budget_1": score(asleep_tight, labels),
            "no_compilation_budget_8": score(awake_full, labels),
            "sleep_total_work": compile_work + len(queries), "lazy_total_work": lazy_work,
            "interpret_total_work": len(queries) * 8,
            "cost_model": "one affine evaluation or affine composition = one work unit; indexing, bookkeeping and historical requests excluded equally; not FLOPs",
            "revision": {"feedback_count": 3, "revoked_on_first_contradiction": revoked,
                         "dependent_macros_invalidated": invalidation,
                         "dependent_requests_abstain_until_revalidated": unavailable_until_revalidated,
                         "other_three_rules_exactly_preserved": True,
                         "after_revalidation": score(revised_preds, revised_labels)}}


def counterexamples():
    # A nonlinear world agreeing at every acquired point defeats the family bias.
    hidden = Learner()
    polynomial = lambda x: (2*x + 3 + x*(x-1)*(x-2)) % P
    for x in range(3):
        hidden.observe(OPS[0], x, polynomial(x))
    queries = [((OPS[0],), x) for x in range(3, P)]
    pred = predictions(hidden, queries)
    true = [polynomial(x) for _, x in queries]
    assert score(pred, true)["correct"] == 0
    # A third point that visibly violates the supplied family prevents promotion.
    inconsistent = Learner()
    for x, y in [(0, 3), (1, 5), (2, 99)]:
        inconsistent.observe(OPS[0], x, y)
    assert OPS[0] not in inconsistent.rules
    # The trusted-label assumption matters: one wrong feedback can revoke truth.
    world = {op: (2, 3) for op in OPS}
    noisy, _ = teach(world)
    noisy.observe(OPS[0], 3, 10)  # correct result is 9
    assert noisy.predict((OPS[0],), 10) is None
    # No exploration => observationally indistinguishable affine worlds.
    no_explore = Learner()
    for _ in range(12):
        no_explore.observe(OPS[0], 0, 3)
    assert OPS[0] not in no_explore.rules
    assert no_explore.consolidate([(OPS[0], OPS[0])]) == 0
    return {"out_of_family_agrees_on_all_three_observations": True,
            "out_of_family_heldout": score(pred, true),
            "visible_third_point_inconsistency_rejected": True,
            "one_false_trusted_feedback_revokes_valid_rule": True,
            "repeated_identical_input_does_not_identify_slope": True,
            "offline_compute_cannot_resolve_missing_slope": True,
            "interpretation": "finite validation neither proves universal generalization nor handles arbitrary corruption"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "affine_lifecycle.json")
    parser.add_argument("--restore-worker", type=Path)
    args = parser.parse_args()
    if args.restore_worker:
        data = json.loads(args.restore_worker.read_text(encoding="utf-8"))
        restored = Learner.restore(data["state"])
        print(json.dumps({"predictions": predictions(restored, data["queries"]),
                          "pending_count": len(restored.pending), "event_count": len(restored.events)}))
        return
    start = time.perf_counter()
    trials = []
    for seed in range(25):
        trial, learner, queries = lifecycle_trial(seed)
        trials.append(trial)
    restored = fresh_process_check(learner, queries)
    offline = offline_and_revision()
    negative = counterexamples()
    result = {"schema": 1, "experiment": "finite affine lifecycle witness",
              "python": platform.python_version(), "seeds": list(range(25)),
              "prime": P, "operator_count": len(OPS),
              "bias": "nonzero-slope affine maps over F_101; composition, tool IDs and trusted feedback are supplied",
              "trials": trials, "fresh_process_restart": restored,
              "offline_and_revision": offline, "counterexamples": negative,
              "elapsed_seconds": time.perf_counter() - start,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "claim_scope": "known finite rule induction + ordinary compilation; no novel architecture, open-world guarantee or Cevahir integration"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seeds": len(trials),
                      "heldout_per_world": trials[0]["a_on_a"]["n"],
                      "all_assertions_passed": True, "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
