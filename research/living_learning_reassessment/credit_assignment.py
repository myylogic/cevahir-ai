"""CPU-light live delayed-feedback / shared-parameter retention experiment.

Run: python -m research.living_learning_reassessment.credit_assignment
Only standard library; no offline retraining and no main-model imports.
"""

from __future__ import annotations

import hashlib
import json
import math
import argparse
import platform
import random
import statistics
from collections import OrderedDict, defaultdict
from pathlib import Path


SEEDS = tuple(range(32))
PHASE_LENGTH = 600
NOISE = 0.10
STEP_SIZE = 0.12
REPLAY_CAPACITY = 32
METHODS = (
    "initial_frozen", "frozen_after_old", "current_two", "anchored_one",
    "anchored_two", "anchored_replay", "online_rls",
)
REGIMES = {
    "compatible_delayed": {"delay": 31, "pending": 64, "world": "compatible"},
    "compatible_immediate": {"delay": 0, "pending": 64, "world": "compatible"},
    "insufficient_pending": {"delay": 127, "pending": 4, "world": "compatible"},
    "incompatible_drift": {"delay": 31, "pending": 64, "world": "conflicting"},
    "misspecified_nonlinear": {"delay": 31, "pending": 64, "world": "nonlinear"},
}


def dot(w, x):
    return w[0] * x[0] + w[1] * x[1] + w[2] * x[2]


def target(x, phase, world):
    _, u, v = x
    if world == "conflicting":
        return 0.3 + (1.2 if phase == 0 else -1.2) * u - 1.7 * v
    return 0.3 + 1.2 * u - 1.7 * v + (0.8 * u * u if world == "nonlinear" else 0.0)


def sample_x(rng, phase, world):
    u = rng.uniform(-1.0, 1.0)
    v = 0.0 if phase == 0 or world == "conflicting" else u
    return (1.0, u, v)


class Model:
    def __init__(self, method):
        self.method = method
        self.w = [0.0] * 3
        self.inverse_gram = [[float(i == j) for j in range(3)] for i in range(3)]
        self.gradient_evaluations = 0
        self.rls_updates = 0

    def gradient_update(self, x, y):
        error = dot(self.w, x) - y
        scale = STEP_SIZE * error / dot(x, x)
        self.w = [a - scale * b for a, b in zip(self.w, x)]
        self.gradient_evaluations += 1

    def rls_update(self, x, y):
        # Sherman-Morrison online ridge update, initial ridge coefficient 1.
        p_x = [dot(row, x) for row in self.inverse_gram]
        denominator = 1.0 + dot(x, p_x)
        gain = [a / denominator for a in p_x]
        residual = y - dot(self.w, x)
        self.w = [a + b * residual for a, b in zip(self.w, gain)]
        self.inverse_gram = [
            [self.inverse_gram[i][j] - gain[i] * p_x[j] for j in range(3)]
            for i in range(3)
        ]
        self.rls_updates += 1

    def receive(self, past_x, y, present_x, replay_item, t):
        if self.method == "initial_frozen":
            return
        if self.method == "frozen_after_old" and t >= PHASE_LENGTH:
            return
        if self.method == "online_rls":
            self.rls_update(past_x, y)
            return
        chosen_x = present_x if self.method == "current_two" else past_x
        self.gradient_update(chosen_x, y)
        if self.method == "anchored_one":
            return
        if self.method == "anchored_replay":
            self.gradient_update(*replay_item)
        else:
            self.gradient_update(chosen_x, y)


def evaluation_sets(seed, world):
    # Separate RNG; these continuous points and labels never enter the learner.
    rng = random.Random(10_000_000 + seed)
    sets = {}
    for name, phase in (("old", 0), ("new", 1)):
        examples = []
        for _ in range(1024):
            x = sample_x(rng, phase, world)
            examples.append((x, target(x, phase, world)))
        sets[name] = examples
    if world != "conflicting":
        sets["composition"] = []
        for _ in range(1024):
            x = (1.0, rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
            sets["composition"].append((x, target(x, 1, world)))
    return sets


def risk(w, examples):
    return statistics.fmean((dot(w, x) - y) ** 2 for x, y in examples)


def run_seed(seed, config):
    world = config["world"]
    rng = random.Random(seed)
    replay_rng = random.Random(1_000_000 + seed)
    delay_rng = random.Random(2_000_000 + seed)
    models = {name: Model(name) for name in METHODS}
    delayed = defaultdict(list)
    pending = OrderedDict()
    replay = []
    seen = 0
    delivered = 0
    dropped = 0
    evicted = 0
    max_pending = 0
    out_of_order = 0
    last_arrival_id = -1
    stream_errors = {name: [0.0, 0.0] for name in METHODS}
    heldouts = evaluation_sets(seed, world)
    old_checkpoint = None
    for t in range(2 * PHASE_LENGTH):
        phase = int(t >= PHASE_LENGTH)
        if t == PHASE_LENGTH:
            old_checkpoint = {name: risk(model.w, heldouts["old"]) for name, model in models.items()}
        x = sample_x(rng, phase, world)
        truth = target(x, phase, world)
        for name, model in models.items():
            stream_errors[name][phase] += (dot(model.w, x) - truth) ** 2
        y = truth + rng.gauss(0.0, NOISE)
        delay = delay_rng.randint(1, config["delay"]) if config["delay"] else 0
        delayed[t + delay].append((t, y))  # feedback is (episode ID, noisy label).
        pending[t] = x
        if len(pending) > config["pending"]:
            pending.popitem(last=False)
            evicted += 1
        max_pending = max(max_pending, len(pending))
        for episode_id, label in delayed.pop(t, []):
            if episode_id < last_arrival_id:
                out_of_order += 1
            last_arrival_id = episode_id
            if episode_id not in pending:
                dropped += 1
                continue
            past_x = pending.pop(episode_id)
            delivered += 1
            # Standard reservoir sampling: no phase/task metadata is stored.
            seen += 1
            item = (past_x, label)
            if len(replay) < REPLAY_CAPACITY:
                replay.append(item)
            else:
                index = replay_rng.randrange(seen)
                if index < REPLAY_CAPACITY:
                    replay[index] = item
            replay_item = replay[replay_rng.randrange(len(replay))]
            for model in models.values():
                model.receive(past_x, label, x, replay_item, t)
    results = {}
    for name, model in models.items():
        risks = {key: risk(model.w, examples) for key, examples in heldouts.items()}
        # Pure predictor reconstruction: buffers and solver state are absent.
        restored_w = json.loads(json.dumps(model.w))
        restored_risks = {key: risk(restored_w, examples) for key, examples in heldouts.items()}
        assert restored_risks == risks
        results[name] = {
            **risks,
            "old_after_acquisition": old_checkpoint[name],
            "old_risk_increase": risks["old"] - old_checkpoint[name],
            "service_phase0": stream_errors[name][0] / PHASE_LENGTH,
            "service_phase1": stream_errors[name][1] / PHASE_LENGTH,
            "gradient_evaluations": model.gradient_evaluations,
            "rls_updates": model.rls_updates,
            "final_weights": model.w,
            "predictor_restore_exact": True,
        }
    assert results["current_two"]["gradient_evaluations"] == results["anchored_two"]["gradient_evaluations"] == results["anchored_replay"]["gradient_evaluations"]
    if config["delay"] == 0:
        assert results["current_two"]["final_weights"] == results["anchored_two"]["final_weights"]
    return {
        "seed": seed, "methods": results,
        "stream": {"accepted_feedback": delivered, "discarded_feedback": dropped,
                   "evicted_pending": evicted, "max_pending": max_pending,
                   "unarrived_feedback": sum(map(len, delayed.values())),
                   "out_of_order_arrivals": out_of_order, "replay_records": len(replay)},
    }


def mean_interval(values):
    mean = statistics.fmean(values)
    # Student t critical value for 31 df, two-sided 95% interval.
    half_width = 2.039513446 * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "ci95_low": mean - half_width, "ci95_high": mean + half_width,
            "min": min(values), "max": max(values)}


def summarize(runs):
    summary = {}
    for method in METHODS:
        metrics = {}
        for key, value in runs[0]["methods"][method].items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                metrics[key] = mean_interval([run["methods"][method][key] for run in runs])
        summary[method] = metrics
    paired = {}
    for name, first, second, metric in (
        ("attribution_old", "current_two", "anchored_two", "old"),
        ("attribution_new", "current_two", "anchored_two", "new"),
        ("replay_retention", "anchored_two", "anchored_replay", "old"),
        ("replay_new", "anchored_two", "anchored_replay", "new"),
        ("rls_retention", "anchored_two", "online_rls", "old"),
    ):
        paired[name] = {"difference": f"{first} minus {second}: {metric}",
                        **mean_interval([r["methods"][first][metric] - r["methods"][second][metric] for r in runs])}
    return {"methods": summary, "paired_differences": paired,
            "stream": {key: mean_interval([r["stream"][key] for r in runs]) for key in runs[0]["stream"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "credit_assignment.json")
    args = parser.parse_args()
    all_runs = {}
    summaries = {}
    for name, config in REGIMES.items():
        runs = [run_seed(seed, config) for seed in SEEDS]
        all_runs[name] = runs
        summaries[name] = summarize(runs)
    scientific = {"summaries": summaries, "runs": all_runs}
    canonical = json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode()
    output = {
        "provenance": {
            "python": platform.python_version(), "platform": platform.platform(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scientific_sha256": hashlib.sha256(canonical).hexdigest(),
            "command": "python -m research.living_learning_reassessment.credit_assignment",
            "seed_count": len(SEEDS), "seeds": list(SEEDS),
            "phase_length": PHASE_LENGTH, "label_noise_std": NOISE,
            "normalized_gradient_step": STEP_SIZE, "replay_capacity": REPLAY_CAPACITY,
            "regimes": REGIMES,
            "protocol_selection": "one specified configuration; no parameter sweep or heldout tuning",
            "budget": {
                "shared_parameter_count": 3, "max_pending_records_main": 64,
                "pending_record": "episode ID plus three input scalars; label arrives later",
                "replay_record": "three input scalars plus noisy label",
                "all_methods_replay_budget_upper_bound": 32,
                "actual_replay_use": "only anchored_replay; others need no replay records",
                "online_rls_additional_state": "3x3 inverse Gram matrix (9 scalar slots)",
                "gradient_comparison": "current_two, anchored_two, anchored_replay have exactly two per accepted feedback",
                "rls_comparison": "strong sufficient-statistic reference, O(d^2); not FLOP matched to gradients",
            },
            "limitations": ["fixed known 3-dimensional linear representation",
                            "supplied exact episode IDs and numeric supervised labels",
                            "two scheduled environment distributions; no boundary passed to adaptive learners",
                            "frozen_after_old is explicitly a boundary-informed diagnostic baseline",
                            "no learned action-sequence temporal relevance, no sparse rewards",
                            "restore checks predictor sufficiency via serialization in the same process; not a process restart",
                            "Monte Carlo intervals cover these seeds in these synthetic worlds only"],
        },
        **scientific,
    }
    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(destination), "scientific_sha256": output["provenance"]["scientific_sha256"],
                      "main_final_mse": {method: {key: round(value["mean"], 6) for key, value in summaries["compatible_delayed"]["methods"][method].items() if key in ("old", "new", "composition")} for method in METHODS}}, indent=2))


if __name__ == "__main__":
    main()
