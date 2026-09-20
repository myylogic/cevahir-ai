"""Bounded online constructive-feature witness; Python standard library only.

This is known constructive induction, not an architectural novelty claim.
Run from the repository root: python -m research.living_learning_reassessment.representation_growth
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import itertools
import json
from pathlib import Path
import random
import time
import tracemalloc

P, D, DISCOVER, VALIDATE, TEST_N = 101, 5, 16, 4, 100
BASE = ((),) + tuple((i,) for i in range(D))
EXPANDED = BASE + tuple(t for k in (2, 3) for t in itertools.combinations(range(D), k))
MODES = ("fixed_linear", "growth", "fixed_expanded_sparse", "fixed_expanded_lazy")


def token(term):
    return "1" if not term else "*".join("x" + str(i) for i in term)


@dataclass
class Meter:
    counts: Counter = field(default_factory=Counter)

    def charge(self, name, amount=1):
        self.counts[name] += amount


def phi(x, term, meter=None):
    value = 1
    for i in term:
        value = value * x[i] % P
    if meter:
        meter.charge("feature_evaluations")
        meter.charge("field_multiplications", len(term))
    return value


def fit(rows, support, meter):
    """Exact finite-field elimination; no held-out labels enter selection."""
    meter.charge("candidate_fits")
    matrix = [[phi(x, term, meter) for term in support] + [y] for x, y in rows]
    width, pivot_row, pivots = len(support), 0, []
    for col in range(width):
        pivot = next((r for r in range(pivot_row, len(matrix)) if matrix[r][col]), None)
        meter.charge("pivot_row_checks", len(matrix) - pivot_row)
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        inv = pow(matrix[pivot_row][col], -1, P)
        meter.charge("field_inversions")
        matrix[pivot_row] = [(v * inv) % P for v in matrix[pivot_row]]
        meter.charge("field_multiplications", width + 1)
        for r in range(len(matrix)):
            if r != pivot_row and matrix[r][col]:
                scale = matrix[r][col]
                matrix[r] = [(a - scale * b) % P for a, b in zip(matrix[r], matrix[pivot_row])]
                meter.charge("field_multiply_subtractions", width + 1)
        pivots.append(col)
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    if any(not any(row[:width]) and row[-1] for row in matrix):
        return None
    # Free variables zero is a fixed deterministic convention, not an oracle.
    coeffs = [0] * width
    for r, col in enumerate(pivots):
        coeffs[col] = matrix[r][-1]
    return tuple((t, c) for t, c in zip(support, coeffs) if c)


class Learner:
    def __init__(self, mode):
        self.mode = mode
        self.features = {t: None for t in (EXPANDED if mode.startswith("fixed_expanded") else BASE)}
        self.active = list(BASE)
        self.heads = {}
        self.pending = {}
        self.meter = Meter()
        self.events = []
        self.max_buffer_rows = 0

    def candidates(self):
        if self.mode == "fixed_linear":
            return []
        if self.mode == "fixed_expanded_sparse":
            return [t for t in EXPANDED if t not in self.active]
        # The supplied generator multiplies an existing feature by one coordinate.
        # Triple features require a previously accepted pair; no hidden world ID.
        candidates = set()
        for left in self.active:
            if not left or len(left) >= 3:
                continue
            for i in range(D):
                self.meter.charge("candidate_generation_attempts")
                if i not in left:
                    term = tuple(sorted(left + (i,)))
                    if term not in self.active:
                        candidates.add(term)
        return sorted(candidates, key=lambda t: (len(t), t))

    def discover(self, rows):
        head = fit(rows, tuple(self.active), self.meter)
        if head is not None:
            return head, None
        for term in self.candidates():
            self.meter.charge("candidate_terms_considered")
            head = fit(rows, tuple(self.active) + (term,), self.meter)
            if head is not None:
                return head, term
        return None, None

    def evaluate(self, head, x, blocked=(), allow_unstored=False):
        if head is None:
            return None
        cache = {}
        def value(term):
            if term in cache:
                return cache[term]
            if term in blocked or (term not in self.features and not allow_unstored):
                return None
            parents = self.features.get(term)
            if parents is None:
                result = phi(x, term, self.meter)
            else:
                left, right = (value(tuple(parent)) for parent in parents)
                if left is None or right is None:
                    return None
                result = left * right % P
                self.meter.charge("feature_evaluations")
                self.meter.charge("field_multiplications")
            cache[term] = result
            return result
        total = 0
        for term, coefficient in head:
            v = value(term)
            if v is None:
                return None
            total = (total + coefficient * v) % P
            self.meter.charge("head_multiply_additions")
        return total

    def predict(self, task, x, blocked=()):
        return self.evaluate(self.heads.get(task), x, blocked)

    def observe(self, task, x, y):
        """Predict first, receive one label, possibly update; no dataset/deployment stage."""
        prediction = self.predict(task, x)
        if task in self.heads and prediction != y:
            del self.heads[task]
            self.pending.pop(task, None)
            self.events.append({"task": task, "event": "contradiction_invalidated_head"})
        if task in self.heads:
            return prediction
        pending = self.pending.setdefault(task, {"rows": [], "proposal": None, "feature": None, "ok": True})
        pending["rows"].append((tuple(x), y))
        self.max_buffer_rows = max(self.max_buffer_rows, sum(len(p["rows"]) for p in self.pending.values()))
        n = len(pending["rows"])
        if n == DISCOVER:
            pending["proposal"], pending["feature"] = self.discover(pending["rows"])
        elif n > DISCOVER:
            if self.evaluate(pending["proposal"], x, allow_unstored=True) != y:
                pending["ok"] = False
        if n == DISCOVER + VALIDATE:
            if pending["proposal"] is not None and pending["ok"]:
                new = pending["feature"]
                if new is not None:
                    parents = next(((left, (i,)) for left in self.active for i in range(D)
                                    if i not in left and tuple(sorted(left + (i,))) == new), None)
                    if self.mode == "growth" and parents is None:
                        raise AssertionError("growth attempted a feature outside its generator")
                    self.features[new] = parents
                    self.active.append(new)
                self.heads[task] = pending["proposal"]
                self.events.append({"task": task, "event": "promoted", "new_feature": token(new) if new else None,
                                    "observations": n})
            else:
                self.events.append({"task": task, "event": "no_validated_model", "observations": n})
            del self.pending[task]  # bounded raw episodes are not permanent learned state
        return prediction

    def snapshot(self):
        return {"mode": self.mode, "features": [[list(t), parents] for t, parents in self.features.items()],
                "active": [list(t) for t in self.active],
                "heads": {k: [[list(t), c] for t, c in v] for k, v in self.heads.items()}}

    @classmethod
    def restore(cls, snapshot):
        model = cls(snapshot["mode"])
        model.features = {tuple(t): parents for t, parents in snapshot["features"]}
        model.active = [tuple(t) for t in snapshot["active"]]
        model.heads = {k: tuple((tuple(t), c) for t, c in v) for k, v in snapshot["heads"].items()}
        return model


def sample_x(rng, used, count):
    result = []
    while len(result) < count:
        x = tuple(rng.randrange(P) for _ in range(D))
        if x not in used:
            used.add(x)
            result.append(x)
    return result


def world(seed):
    rng, used = random.Random(seed), set()
    perm = rng.sample(list(range(D)), D)
    a = tuple(sorted(perm[:2]))
    terms = [a, tuple(sorted(perm[:3])), tuple(sorted(perm[3:]))]
    tasks = {}
    for index, term in enumerate(terms):
        coeffs = [((), rng.randrange(P)), ((perm[index],), rng.randrange(1, P)), (term, rng.randrange(1, P))]
        target = lambda x, coeffs=coeffs: sum(c * phi(x, t) for t, c in coeffs) % P
        train = sample_x(rng, used, DISCOVER + VALIDATE)
        test = sample_x(rng, used, TEST_N)
        tasks[str(index)] = {"target": target, "term": term, "train": [(x, target(x)) for x in train],
                             "test": [(x, target(x)) for x in test]}
    return tasks, rng, used


def score(model, task, rows, blocked=()):
    predictions = [model.predict(task, x, blocked) for x, _ in rows]
    return {"n": len(rows), "correct": sum(pred == y for pred, (_, y) in zip(predictions, rows)),
            "abstentions": sum(pred is None for pred in predictions)}


def run_seed(seed):
    tasks, rng, used = world(seed)
    models, reports = {}, {}
    for mode in MODES:
        model = Learner(mode)
        retention = []
        for task, data in tasks.items():
            prequential = [model.observe(task, x, y) for x, y in data["train"]]
            retention.append({"after_task": task, "scores": {k: score(model, k, tasks[k]["test"])
                              for k in tasks if int(k) <= int(task)},
                              "stream_correct": sum(p == y for p, (_, y) in zip(prequential, data["train"])),
                              "stream_abstentions": sum(p is None for p in prequential)})
        models[mode] = model
        snapshot = json.loads(json.dumps(model.snapshot()))
        restored = Learner.restore(snapshot)
        scores = {task: score(restored, task, data["test"]) for task, data in tasks.items()}
        reports[mode] = {"scores": scores, "retention": retention, "events": model.events,
                         "active_features": [token(t) for t in model.active],
                         "stored_feature_definitions": len(model.features),
                         "snapshot_bytes": len(json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()),
                         "pending_rows_at_end": sum(len(p["rows"]) for p in model.pending.values()),
                         "maximum_buffered_rows": model.max_buffer_rows, "work": dict(model.meter.counts)}
    growth = models["growth"]
    # Causal interventions remove learned computations/heads without touching test inputs.
    ablation = {task: score(growth, task, data["test"], blocked=(data["term"],)) for task, data in tasks.items()}
    dependency_ablation = {task: score(growth, task, data["test"], blocked=(tasks["0"]["term"],))
                          for task, data in tasks.items()}
    deleted = Learner.restore(json.loads(json.dumps(growth.snapshot())))
    deleted.features.pop(tasks["0"]["term"], None)
    physical_deletion = {task: score(deleted, task, data["test"]) for task, data in tasks.items()}
    reset = {task: score(Learner("growth"), task, data["test"]) for task, data in tasks.items()}
    # A noise label looks just like a true change to this exact deterministic learner.
    poisoned = Learner.restore(json.loads(json.dumps(growth.snapshot())))
    x_noise = sample_x(rng, used, 1)[0]
    poisoned.observe("0", x_noise, (tasks["0"]["target"](x_noise) + 1) % P)
    poison_scores = {task: score(poisoned, task, data["test"]) for task, data in tasks.items()}
    # Random-label block: model should not be given an oracle nonlinear feature.
    corrupt = Learner("growth")
    for x, _ in tasks["0"]["train"]:
        corrupt.observe("0", x, rng.randrange(P))
    corrupt_score = score(corrupt, "0", tasks["0"]["test"])
    # Rule changes at the same task ID, with no world/task-reset notification.
    shifted = Learner.restore(json.loads(json.dumps(growth.snapshot())))
    shift_x = sample_x(rng, used, DISCOVER + VALIDATE)
    shift_rows = [(x, (tasks["0"]["target"](x) + 1) % P) for x in shift_x]
    shift_predictions = [shifted.observe("0", x, y) for x, y in shift_rows]
    shift_eval = [(x, (y + 1) % P) for x, y in tasks["0"]["test"]]
    shift = {"stream_wrong": sum(p is not None and p != y for p, (_, y) in zip(shift_predictions, shift_rows)),
             "stream_abstentions": sum(p is None for p in shift_predictions),
             "new_rule_score": score(shifted, "0", shift_eval),
             "old_rule_score": score(shifted, "0", tasks["0"]["test"]),
             "unaffected_scores": {k: score(shifted, k, tasks[k]["test"]) for k in ("1", "2")}}
    # Missing the prerequisite pair makes a direct triple inaccessible to growth.
    cold_triple = {}
    for mode in ("growth", "fixed_expanded_sparse"):
        model = Learner(mode)
        for x, y in tasks["1"]["train"]:
            model.observe("1", x, y)
        cold_triple[mode] = score(model, "1", tasks["1"]["test"])
    # Identical activation policy on a pre-expanded registry is an exact strong control.
    lazy_equal = all(reports["growth"][field] == reports["fixed_expanded_lazy"][field]
                     for field in ("scores", "retention", "events", "active_features", "work"))
    return {"seed": seed, "targets": {k: token(v["term"]) for k, v in tasks.items()}, "methods": reports,
            "feature_ablation": ablation, "dependency_ablation": dependency_ablation,
            "physical_feature_deletion": physical_deletion, "state_reset": reset, "single_corrupt_label": poison_scores,
            "random_labels": corrupt_score, "shift": shift, "cold_triple": cold_triple,
            "growth_equals_fixed_expanded_lazy_except_storage": lazy_equal}


def audit():
    """Small semantic checks, not implementation-mirroring test-count evidence."""
    rows = [((x, 0, 0, 0, 0), (7*x+3) % P) for x in range(9)]
    head = fit(rows, BASE, Meter())
    assert all(sum(c*phi(x, t) for t, c in head) % P == y for x, y in rows)
    assert fit(rows + [(rows[0][0], 99)], BASE, Meter()) is None
    # All linear functions have zero rectangular mixed difference; x0*x1 does not.
    square = [((0, 0, 0, 0, 0), 0), ((1, 0, 0, 0, 0), 0),
              ((0, 1, 0, 0, 0), 0), ((1, 1, 0, 0, 0), 1)]
    assert fit(square, BASE, Meter()) is None
    assert fit(square, BASE + ((0, 1),), Meter()) is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "representation_growth.json")
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.seeds <= 30:
        raise ValueError("seeds must be between 1 and 30")
    started = time.perf_counter()
    audit()
    tracemalloc.start()
    runs = [run_seed(seed) for seed in range(args.seeds)]
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    summary = {}
    for mode in MODES:
        summary[mode] = {"correct": sum(s["methods"][mode]["scores"][t]["correct"] for s in runs for t in ("0", "1", "2")),
                         "n": args.seeds * 3 * TEST_N,
                         "mean_feature_definitions": sum(s["methods"][mode]["stored_feature_definitions"] for s in runs) / args.seeds,
                         "mean_snapshot_bytes": sum(s["methods"][mode]["snapshot_bytes"] for s in runs) / args.seeds,
                         "sum_work": dict(sum((Counter(s["methods"][mode]["work"]) for s in runs), Counter()))}
    checks = {"lazy_equivalence_all_seeds": all(s["growth_equals_fixed_expanded_lazy_except_storage"] for s in runs),
              "feature_ablation_eliminates_target_answers": all(s["feature_ablation"][t]["abstentions"] == TEST_N for s in runs for t in ("0", "1", "2")),
              "reset_eliminates_answers": all(s["state_reset"][t]["abstentions"] == TEST_N for s in runs for t in ("0", "1", "2")),
              "random_labels_do_not_promote": all(s["random_labels"]["abstentions"] == TEST_N for s in runs),
              "raw_buffer_empty_after_main_stream": all(s["methods"][m]["pending_rows_at_end"] == 0 for s in runs for m in MODES)}
    report = {"schema_version": 1, "experiment": "bounded_streaming_constructive_features",
              "novel_mechanism_claimed": False, "protocol": {"field": P, "inputs": D, "discovery_observations": DISCOVER,
                "subsequent_validation_observations": VALIDATE, "heldout_inputs_per_task": TEST_N,
                "seed_count": args.seeds, "grammatical_degree_cap": 3, "task_identity_given": True,
                "old_heads_immutable_on_other_task_updates": True,
                "cost_scope": "explicit algorithmic counters; includes generation/search/feature arithmetic and evaluation; not CPU/FLOPs",
                "new_primitive_discovery": False, "raw_learning_window_allowed": True},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "elapsed_seconds": time.perf_counter() - started, "traced_python_peak_bytes": peak,
              "summary": summary, "checks": checks, "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "elapsed_seconds": report["elapsed_seconds"],
                      "traced_python_peak_bytes": peak, "summary": summary, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
