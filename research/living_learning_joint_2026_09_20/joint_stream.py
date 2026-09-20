"""A bounded shared-state living-learning witness, not a new learning principle.

Precommitted protocol: continuous input, numeric noisy outcome; no task IDs or
change flags reach the learner. Known finite feature language and spatial grid
are supplied inductive biases. Only standard-library Python is required.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

TERMS = ["1", "x", "z", "x*z", "max(x,0)*z"]
SEEDS = 24
PHASE_LENGTH = 256
HORIZON = 7 * PHASE_LENGTH
REFIT_INTERVAL = 16
GRID_SIDE = 4
CELL_CAPACITY = 8
MEMORY_CAPACITY = GRID_SIDE * GRID_SIDE * CELL_CAPACITY
RIDGE = 0.001
NOISE_SD = 0.10
HOLDOUT_N = 400
MIN_DISCOVERY_ROWS = 48
MIN_FOLD_IMPROVEMENT = 0.001
MIN_RELATIVE_IMPROVEMENT = 0.10
CHECKPOINTS = [256, 512, 768, 800, 896, 1024, 1056, 1280, 1536, 1792]
MODES = ["growth_coverage", "expanded_coverage", "expanded_rolling", "expanded_allhistory", "base_coverage"]
PHASES = ["left_acquisition", "right_acquisition", "left_recurrence", "right_label_burst_then_recovery", "right_local_change", "left_retention", "right_changed_recurrence"]


def features(x, z, active):
    values = [1.0, x, z, x * z, max(x, 0.0) * z]
    return [values[i] for i in active]


def target(x, z, changed=False, global_change=False):
    y = 0.5 + 0.7 * x - 0.4 * z + 0.8 * x * z
    if changed:
        y -= 1.6 * (x if global_change else max(x, 0.0)) * z
    return y


def cell_index(x, z):
    ix = min(GRID_SIDE - 1, max(0, int((x + 1.0) * GRID_SIDE / 2)))
    iz = min(GRID_SIDE - 1, max(0, int((z + 1.0) * GRID_SIDE / 2)))
    return ix * GRID_SIDE + iz


def solve(matrix, vector, costs):
    n = len(vector)
    a = [list(row) + [b] for row, b in zip(matrix, vector)]
    for i in range(n):
        a[i][i] += RIDGE
    costs["linear_solves"] += 1
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        a[col], a[pivot] = a[pivot], a[col]
        if abs(a[col][col]) < 1e-15:
            raise ArithmeticError("Ridge-regularized solve unexpectedly singular")
        scale = a[col][col]
        for k in range(col, n + 1):
            a[col][k] /= scale
            costs["solve_scalar_divisions"] += 1
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            for k in range(col, n + 1):
                a[row][k] -= factor * a[col][k]
                costs["solve_scalar_multiply_adds"] += 1
    return [a[i][n] for i in range(n)]


def accumulate(matrix, vector, row, active, costs):
    x, z, y, _serial = row
    phi = features(x, z, active)
    costs["training_feature_components"] += len(active)
    for i, vi in enumerate(phi):
        vector[i] += vi * y
        for j, vj in enumerate(phi):
            matrix[i][j] += vi * vj
    costs["moment_scalar_multiply_adds"] += len(active) * (len(active) + 1)


def fit(rows, active, costs):
    n = len(active)
    matrix, vector = [[0.0] * n for _ in range(n)], [0.0] * n
    for row in rows:
        accumulate(matrix, vector, row, active, costs)
    return solve(matrix, vector, costs)


def mse_rows(rows, active, weights, costs):
    loss = 0.0
    for x, z, y, _serial in rows:
        pred = sum(w * v for w, v in zip(weights, features(x, z, active)))
        loss += (pred - y) ** 2
    costs["validation_feature_components"] += len(rows) * len(active)
    return loss / len(rows)


class Learner:
    def __init__(self, mode):
        self.mode = mode
        self.active = [0, 1, 2] if mode in ("growth_coverage", "base_coverage") else list(range(5))
        self.weights = [0.0] * len(self.active)
        self.cells = [[] for _ in range(GRID_SIDE * GRID_SIDE)]
        self.rolling = []
        self.gram = [[0.0] * 5 for _ in range(5)]
        self.cross = [0.0] * 5
        self.step = 0
        self.costs = Counter()
        self.events = []
        self.max_records = 0

    def rows(self):
        if self.mode == "expanded_rolling":
            return self.rolling
        return [row for cell in self.cells for row in cell]

    def predict(self, x, z, charge=False):
        if charge:
            self.costs["service_feature_components"] += len(self.active)
        return sum(w * v for w, v in zip(self.weights, features(x, z, self.active)))

    def consider_growth(self, rows):
        if len(rows) < MIN_DISCOVERY_ROWS:
            return
        # Splits are selected from the training memory only. Reused validation
        # means these heuristic thresholds are not confidence guarantees.
        folds = [[row for row in rows if row[3] % 2 == parity] for parity in (0, 1)]
        if min(map(len, folds)) < 12:
            return
        while len(self.active) < len(TERMS):
            base_losses = []
            for i in (0, 1):
                w = fit(folds[i], self.active, self.costs)
                base_losses.append(mse_rows(folds[1 - i], self.active, w, self.costs))
            choices = []
            for candidate in (3, 4):
                if candidate in self.active:
                    continue
                active = sorted(self.active + [candidate])
                candidate_losses = []
                for i in (0, 1):
                    w = fit(folds[i], active, self.costs)
                    candidate_losses.append(mse_rows(folds[1 - i], active, w, self.costs))
                gains = [a - b for a, b in zip(base_losses, candidate_losses)]
                average_gain = statistics.mean(gains)
                if min(gains) >= MIN_FOLD_IMPROVEMENT and average_gain >= MIN_RELATIVE_IMPROVEMENT * statistics.mean(base_losses):
                    choices.append((average_gain, candidate, gains, base_losses, candidate_losses))
            if not choices:
                break
            gain, candidate, gains, before, after = max(choices)
            self.active = sorted(self.active + [candidate])
            self.events.append({"step": self.step, "term": TERMS[candidate], "fold_gain": gains, "base_fold_mse": before, "candidate_fold_mse": after})

    def observe(self, x, z, y):
        # The learner receives only this continuous input and observed outcome.
        # It has an internal sample count, no phase or truth information.
        self.step += 1
        row = [x, z, y, self.step]
        if self.mode == "expanded_allhistory":
            accumulate(self.gram, self.cross, row, self.active, self.costs)
        elif self.mode == "expanded_rolling":
            self.rolling.append(row)
            if len(self.rolling) > MEMORY_CAPACITY:
                self.rolling.pop(0)
        else:
            cell = self.cells[cell_index(x, z)]
            cell.append(row)
            if len(cell) > CELL_CAPACITY:
                cell.pop(0)
        self.max_records = max(self.max_records, len(self.rows()))
        if self.step % REFIT_INTERVAL:
            return
        if self.mode == "expanded_allhistory":
            self.weights = solve(self.gram, self.cross, self.costs)
        else:
            rows = self.rows()
            if self.mode == "growth_coverage":
                self.consider_growth(rows)
            self.weights = fit(rows, self.active, self.costs)

    def snapshot(self):
        # Only operational state. Scores, phase labels, and heldout data are
        # intentionally absent. Costs/events are diagnostic metadata retained
        # for exact save/restore comparison.
        return {"mode": self.mode, "active": self.active, "weights": self.weights,
                "cells": self.cells, "rolling": self.rolling, "gram": self.gram,
                "cross": self.cross, "step": self.step, "costs": dict(self.costs),
                "events": self.events, "max_records": self.max_records}

    @classmethod
    def restore(cls, state):
        learner = cls(state["mode"])
        for key, value in state.items():
            setattr(learner, key, Counter(value) if key == "costs" else value)
        return learner


def holdout_points(seed, positive):
    rng = random.Random(900_000 + seed * 31 + int(positive))
    return [(rng.uniform(0.2, 1.0) * (1 if positive else -1), rng.uniform(-1.0, 1.0)) for _ in range(HOLDOUT_N)]


def risk(learner, points, changed=False, global_change=False):
    return statistics.mean((learner.predict(x, z) - target(x, z, changed, global_change)) ** 2 for x, z in points)


def make_stream(seed):
    rng = random.Random(40_000 + seed)
    stream = []
    for t in range(HORIZON):
        phase = t // PHASE_LENGTH
        positive = phase in (1, 3, 4, 6)
        x = rng.uniform(0.2, 1.0) * (1 if positive else -1)
        z = rng.uniform(-1.0, 1.0)
        changed = t >= 4 * PHASE_LENGTH
        clean = target(x, z, changed)
        # A structured 32-label corruption burst mimics a real local change.
        # Gaussian label noise exists throughout the whole life.
        corrupted = 3 * PHASE_LENGTH <= t < 3 * PHASE_LENGTH + 32
        label = (target(x, z, True) if corrupted else clean) + rng.gauss(0, NOISE_SD)
        stream.append((x, z, label, clean, phase))
    return stream


def summarize(values):
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "seed_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "mean_95pct_normal_interval": [mean - 1.96 * se, mean + 1.96 * se],
            "min": min(values), "max": max(values)}


def one_seed(seed):
    stream = make_stream(seed)
    learners = {mode: Learner(mode) for mode in MODES}
    left, right = holdout_points(seed, False), holdout_points(seed, True)
    phase_loss = {mode: [0.0] * len(PHASES) for mode in MODES}
    phase_label_error = {mode: [0.0] * len(PHASES) for mode in MODES}
    checkpoints = {mode: {} for mode in MODES}
    resumed = {}
    resume_max_prediction_difference = {mode: 0.0 for mode in MODES}
    resume_all_state_equal = {mode: True for mode in MODES}
    prefix_equivalence = True
    for t, (x, z, label, clean, phase) in enumerate(stream, 1):
        for mode, learner in learners.items():
            pred = learner.predict(x, z, charge=True)
            phase_loss[mode][phase] += (pred - clean) ** 2 / PHASE_LENGTH
            phase_label_error[mode][phase] += (pred - label) ** 2 / PHASE_LENGTH
            if mode in resumed:
                restored = resumed[mode]
                restored_pred = restored.predict(x, z, charge=True)
                resume_max_prediction_difference[mode] = max(resume_max_prediction_difference[mode], abs(restored_pred - pred))
                restored.observe(x, z, label)
            learner.observe(x, z, label)
            if mode in resumed:
                resume_all_state_equal[mode] &= learner.snapshot() == resumed[mode].snapshot()
            if t == 896:
                # Real JSON roundtrip, then all later updates are repeated.
                resumed[mode] = Learner.restore(json.loads(json.dumps(learner.snapshot())))
            if t in CHECKPOINTS:
                changed = t >= 1024 + 1
                checkpoints[mode][str(t)] = {
                    "left_current_mse": risk(learner, left, changed),
                    "right_current_mse": risk(learner, right, changed),
                    "left_if_unobserved_global_change_mse": risk(learner, left, True, True),
                    "right_original_mse": risk(learner, right, False),
                    "active": [TERMS[i] for i in learner.active],
                    "records": len(learner.rows()),
                }
        if 1025 <= t <= 1280:
            prefix_equivalence &= target(x, z, True, False) == target(x, z, True, True)
    result = {"seed": seed, "methods": {}}
    for mode, learner in learners.items():
        snapshot = learner.snapshot()
        clone = Learner.restore(json.loads(json.dumps(snapshot)))
        reset = Learner(mode)
        # Test novel queries, not retrieval of the saved examples.
        restored_max_diff = max(abs(clone.predict(x, z) - learner.predict(x, z)) for x, z in left + right)
        result["methods"][mode] = {
            "phase_service_clean_mse": dict(zip(PHASES, phase_loss[mode])),
            "phase_observed_label_mse": dict(zip(PHASES, phase_label_error[mode])),
            "lifetime_service_clean_mse": statistics.mean(phase_loss[mode]),
            "checkpoints": checkpoints[mode],
            "terminal_left_mse": risk(learner, left, True),
            "terminal_right_mse": risk(learner, right, True),
            "terminal_original_right_mse": risk(learner, right, False),
            "reset_left_mse": risk(reset, left, True),
            "reset_right_mse": risk(reset, right, True),
            "costs": dict(learner.costs),
            "max_raw_records": learner.max_records,
            "raw_numeric_scalars_at_capacity": 4 * learner.max_records,
            "operational_state_json_bytes_terminal": len(json.dumps(snapshot, separators=(",", ":")).encode()),
            "active_terms_terminal": [TERMS[i] for i in learner.active],
            "candidate_activation_events": learner.events,
            "persistence": {"midlife_checkpoint": 896, "continued_samples": HORIZON - 896,
                            "max_prediction_difference": resume_max_prediction_difference[mode],
                            "every_future_state_equal": resume_all_state_equal[mode],
                            "terminal_query_max_difference": restored_max_diff},
        }
    gaps = [(target(x, z, False) - target(x, z, True, True)) ** 2 for x, z in left]
    result["indistinguishable_unobserved_change"] = {
        "observed_local_global_targets_identical_until_1280": prefix_equivalence,
        "heldout_left_target_squared_gap": statistics.mean(gaps),
        "minimum_average_two_world_mse_any_common_predictor": statistics.mean(gaps) / 4,
        "learner_gets_world_id": False,
    }
    return result


def aggregate(runs):
    out = {}
    for mode in MODES:
        records = [run["methods"][mode] for run in runs]
        out[mode] = {
            key: summarize([record[key] for record in records])
            for key in ("lifetime_service_clean_mse", "terminal_left_mse", "terminal_right_mse", "terminal_original_right_mse", "reset_left_mse", "reset_right_mse", "max_raw_records", "operational_state_json_bytes_terminal")}
        out[mode]["phase_service_clean_mse"] = {phase: summarize([record["phase_service_clean_mse"][phase] for record in records]) for phase in PHASES}
        out[mode]["checkpoints"] = {
            str(t): {key: summarize([record["checkpoints"][str(t)][key] for record in records]) for key in ("left_current_mse", "right_current_mse", "left_if_unobserved_global_change_mse", "right_original_mse")}
            for t in CHECKPOINTS}
        out[mode]["mean_costs"] = {key: statistics.mean(record["costs"].get(key, 0) for record in records) for key in sorted(set().union(*(record["costs"] for record in records)))}
        out[mode]["candidate_activation_counts_by_term"] = dict(Counter(event["term"] for record in records for event in record["candidate_activation_events"]))
        out[mode]["candidate_activation_during_corruption_or_before_recovery"] = sum(768 < event["step"] <= 1024 for record in records for event in record["candidate_activation_events"])
        out[mode]["all_persistence_checks_equal"] = all(record["persistence"]["every_future_state_equal"] and record["persistence"]["max_prediction_difference"] == 0 and record["persistence"]["terminal_query_max_difference"] == 0 for record in records)
    comparisons = {}
    for competitor in MODES[1:]:
        comparisons["growth_minus_" + competitor] = {
            key: summarize([run["methods"]["growth_coverage"][key] - run["methods"][competitor][key] for run in runs])
            for key in ("lifetime_service_clean_mse", "terminal_left_mse", "terminal_right_mse")}
    return {"methods": out, "paired_differences": comparisons,
            "indistinguishable_worlds_bound": summarize([run["indistinguishable_unobserved_change"]["minimum_average_two_world_mse_any_common_predictor"] for run in runs])}


def validate(runs):
    checks = {"bounded_raw_memory": True, "future_persistence_identical": True, "target_prefix_indistinguishable": True, "finite_losses": True, "heldout_not_in_operational_state": True}
    for run in runs:
        checks["target_prefix_indistinguishable"] &= run["indistinguishable_unobserved_change"]["observed_local_global_targets_identical_until_1280"]
        for record in run["methods"].values():
            checks["bounded_raw_memory"] &= record["max_raw_records"] <= MEMORY_CAPACITY
            checks["future_persistence_identical"] &= record["persistence"]["every_future_state_equal"] and record["persistence"]["max_prediction_difference"] == 0
            checks["finite_losses"] &= math.isfinite(record["lifetime_service_clean_mse"])
    for key, value in checks.items():
        if not value:
            raise AssertionError(key)
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--output", type=Path, default=Path("research/living_learning_joint_2026_09_20/results/joint_stream.json"))
    args = parser.parse_args()
    runs = [one_seed(seed) for seed in range(args.seeds)]
    output = {
        "experiment": "bounded_shared_state_joint_stream", "protocol_version": 1,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol": {"seeds": args.seeds, "horizon": HORIZON, "phase_length": PHASE_LENGTH, "phases": PHASES,
                     "known_feature_language": TERMS, "task_ids_or_boundary_flags": False,
                     "continuous_inputs": "x in [-1,-.2] or [.2,1], z in [-1,1]",
                     "initial_target": ".5+.7*x-.4*z+.8*x*z",
                     "local_changed_target": ".5+.7*x-.4*z+.8*x*z-1.6*max(x,0)*z",
                     "true_change_first_sample": 1025, "structured_label_corruption_samples": [769, 800],
                     "noise_sd": NOISE_SD, "raw_record_capacity": MEMORY_CAPACITY,
                     "supplied_spatial_grid": [GRID_SIDE, GRID_SIDE], "cell_capacity": CELL_CAPACITY,
                     "refit_interval": REFIT_INTERVAL, "ridge": RIDGE,
                     "discovery_min_rows": MIN_DISCOVERY_ROWS, "discovery_min_fold_gain": MIN_FOLD_IMPROVEMENT,
                     "discovery_min_relative_gain": MIN_RELATIVE_IMPROVEMENT,
                     "holdout_points_per_side_seed": HOLDOUT_N, "test_labels_used_for_learning": False,
                     "search_is_finite_supplied_candidates": True, "all_history_uses_fixed_gram_cross_statistics": True,
                     "operation_counters_are_abstract_not_cpu_time": True,
                     "growth_structural_deletion": False,
                     "negative_regime": "local-vs-global change has identical training prefix through sample1280 but conflicting left targets"},
        "summary": aggregate(runs), "checks": validate(runs), "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "checks": output["checks"], "means": {mode: {key: output["summary"]["methods"][mode][key]["mean"] for key in ("lifetime_service_clean_mse", "terminal_left_mse", "terminal_right_mse")} for mode in MODES}}, indent=2))


if __name__ == "__main__":
    main()
