"""Offline replay of externally supplied, fully observed action outcome tables.

Usage from the repository root (stdlib only; no model is loaded):
    python benchmarks/research/experience_replay.py --input outcomes.jsonl --output report.json

Each JSONL row has sample_id, source_group, task_key, split (train/eval),
baseline, and outcomes: {action: {quality: [0,1], cost: [0,1]}}. Every available
action has a supplied outcome. Training observations are entered as completed
actions with evaluator feedback; held-out labels are read only after selection.
This is a controlled table replay, not a bandit/off-policy estimator. It cannot
invent the outcome of an unobserved action or establish empirical model quality.

Policies: frozen baseline, adaptive experience, seeded shuffled TRAIN feedback,
and omitted negative TRAIN feedback (quality < 0.5). Costs stay attached to the
observed action in every ablation. The store's cost/(1+cost) utility transform is
separate from the original table cost reported here. No evaluation feedback is
written. All rows are capped before evaluation and train/eval source groups must
be disjoint. Repeated task buckets across splits are intentional transfer tests.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import sys


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cognitive_management.research.experience import ExperienceStore


_ACTIONS = frozenset({"direct", "think1", "debate2", "tot"})
_POLICIES = ("frozen", "adaptive", "shuffled_training_feedback", "no_negative_feedback")
_IDENTITY = {"model_revision": "offline-table-v1", "tokenizer_revision": "no-tokenizer"}
_SCOPE = "offline-table-replay"


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _unit_number(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [0,1]")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number in [0,1]") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number in [0,1]")
    return value


def _label(name, value):
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError(f"{name} must be a nonempty string of at most 512 characters")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_rows(rows, *, max_rows=10000):
    """Consume at most max_rows+1 rows; reject leakage and invalid observations."""
    _positive_integer("max_rows", max_rows)
    validated = []
    samples, groups, splits = set(), {}, Counter()
    expected = {"sample_id", "source_group", "task_key", "split", "outcomes", "baseline"}
    for number, raw in enumerate(rows, 1):
        if number > max_rows:
            raise ValueError(f"input exceeds max_rows={max_rows}")
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError(f"row {number} requires exactly {sorted(expected)}")
        row = {key: _label(key, raw[key]) for key in ("sample_id", "source_group", "task_key", "baseline")}
        split = raw["split"]
        if split not in ("train", "eval"):
            raise ValueError(f"row {number}: split must be train or eval")
        if row["sample_id"] in samples:
            raise ValueError(f"duplicate sample_id: {row['sample_id']}")
        samples.add(row["sample_id"])
        previous_split = groups.setdefault(row["source_group"], split)
        if previous_split != split:
            raise ValueError(f"source_group crosses train/eval split: {row['source_group']}")
        outcomes = raw["outcomes"]
        if not isinstance(outcomes, dict) or not outcomes or not set(outcomes).issubset(_ACTIONS):
            raise ValueError(f"row {number}: outcomes must contain known route actions")
        if row["baseline"] not in outcomes:
            raise ValueError(f"row {number}: baseline outcome must be observed")
        row["outcomes"] = {}
        for action, observation in sorted(outcomes.items()):
            if not isinstance(observation, dict) or set(observation) != {"quality", "cost"}:
                raise ValueError(f"row {number}: {action} requires quality and cost")
            row["outcomes"][action] = {
                key: _unit_number(f"row {number} {action}.{key}", observation[key])
                for key in ("quality", "cost")
            }
        row["split"] = split
        splits[split] += 1
        validated.append(row)
    if not splits["train"] or not splits["eval"]:
        raise ValueError("input requires at least one train and one eval row")
    return validated


def load_rows(path, *, max_rows=10000, max_line_chars=131072):
    """Read bounded JSONL without allocating an unbounded line or dataset."""
    _positive_integer("max_line_chars", max_line_chars)

    def parsed():
        with Path(path).open("r", encoding="utf-8") as stream:
            line_number = 0
            while True:
                line = stream.readline(max_line_chars + 1)
                if not line:
                    break
                line_number += 1
                if len(line) > max_line_chars:
                    raise ValueError(f"line {line_number} exceeds max_line_chars={max_line_chars}")
                if not line.strip():
                    continue
                try:
                    yield json.loads(line, object_pairs_hook=_unique_object)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"line {line_number}: {exc}") from exc

    return validate_rows(parsed(), max_rows=max_rows)


def _record_key(policy, sample_id, action):
    return sha256(json.dumps([policy, sample_id, action], separators=(",", ":")).encode()).hexdigest()


def _train_stores(rows, *, seed, min_support, cost_weight, store_factory):
    observed = [(row, action, outcome) for row in rows if row["split"] == "train"
                for action, outcome in sorted(row["outcomes"].items())]
    qualities = [outcome["quality"] for _, _, outcome in observed]
    shuffled = list(qualities)
    random.Random(seed).shuffle(shuffled)
    stores, feedback_counts = {}, {}
    for policy in _POLICIES[1:]:
        store = store_factory(max_records_per_scope=max(1, len(observed)), min_support=min_support,
                              cost_weight=cost_weight, decay_half_life_seconds=604800)
        count = 0
        for index, (row, action, outcome) in enumerate(observed):
            key = _record_key(policy, row["sample_id"], action)
            experience_id = store.observe(
                scope=_SCOPE, identity=_IDENTITY, task_key=row["task_key"],
                planned_strategy=action, executed_strategy=action, status="observed",
                cost=outcome["cost"], source_ids=(row["source_group"],),
                request_id=key, sample_id=row["sample_id"],
            )
            quality = shuffled[index] if policy == "shuffled_training_feedback" else qualities[index]
            if policy == "no_negative_feedback" and quality < 0.5:
                continue
            store.feedback(experience_id=experience_id, scope=_SCOPE, identity=_IDENTITY,
                           value=quality, source="evaluator", event_id="feedback-" + key)
            count += 1
        stores[policy] = store
        feedback_counts[policy] = count
    return stores, feedback_counts, len(observed)


def evaluate_rows(rows, *, seed=42, min_support=3, cost_weight=0.1, max_rows=10000,
                  store_factory=ExperienceStore):
    """Warm up using train labels, freeze stores, select before reading eval labels."""
    rows = validate_rows(rows, max_rows=max_rows)
    _positive_integer("min_support", min_support)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    cost_weight = _unit_number("cost_weight", cost_weight)
    stores, feedback_counts, training_outcomes = _train_stores(
        rows, seed=seed, min_support=min_support, cost_weight=cost_weight, store_factory=store_factory)
    revisions = {name: store.snapshot()["revision"] for name, store in stores.items()}
    measurements = {name: [] for name in _POLICIES}
    paired = []
    for row in (item for item in rows if item["split"] == "eval"):
        choices = {"frozen": {"strategy": row["baseline"], "support": 0, "reason": "fixed_baseline"}}
        for name, store in stores.items():
            choices[name] = store.recommend(scope=_SCOPE, identity=_IDENTITY, task_key=row["task_key"],
                                            baseline=row["baseline"], allowed=sorted(row["outcomes"]))
        # No quality/cost label is used to choose an action on this held-out row.
        values = {}
        for name, choice in choices.items():
            action = choice["strategy"]
            if action not in row["outcomes"]:
                raise ValueError(f"policy selected an unobserved action: {action}")
            observation = row["outcomes"][action]
            support_fallback = name != "frozen" and action == row["baseline"] and choice["reason"] != "supported_observed_preference"
            value = {"action": action, "quality": observation["quality"], "cost": observation["cost"],
                     "support": choice["support"], "reason": choice["reason"], "support_fallback": support_fallback}
            values[name] = value
            measurements[name].append(value)
        paired.append({"sample_id": row["sample_id"], "source_group": row["source_group"],
                       "task_key": row["task_key"], "policies": values,
                       "deltas_vs_frozen": {name: {
                           "quality": values[name]["quality"] - values["frozen"]["quality"],
                           "cost": values[name]["cost"] - values["frozen"]["cost"],
                       } for name in _POLICIES[1:]}})
    for name, store in stores.items():
        if store.snapshot()["revision"] != revisions[name]:
            raise RuntimeError("evaluation modified a frozen experience store")
    summaries = {}
    for name, results in measurements.items():
        count = len(results)
        summaries[name] = {
            "eval_count": count,
            "mean_quality": math.fsum(item["quality"] for item in results) / count,
            "mean_cost": math.fsum(item["cost"] for item in results) / count,
            "action_distribution": dict(sorted(Counter(item["action"] for item in results).items())),
            "support_fallback_count": sum(item["support_fallback"] for item in results),
            "reason_distribution": dict(sorted(Counter(item["reason"] for item in results).items())),
        }
    return {
        "schema_version": 1,
        "evidence_type": "offline_fully_observed_tabulated_mechanism_evidence",
        "empirical_model_quality_validated": False,
        "limitations": [
            "Outcomes are supplied table labels, not measurements produced by this harness.",
            "Synthetic input establishes mechanism behavior only; no model quality or learning gain is inferred.",
            "All selected outcomes must be observed; this is not a counterfactual or bandit estimator.",
            "Task buckets are caller-defined; disjoint source groups do not establish semantic independence.",
            "Reported cost is the original table value; store utility uses cost/(1+cost).",
        ],
        "controls": {"seed": seed, "min_support": min_support, "cost_weight": cost_weight,
                     "max_rows": max_rows, "no_negative_feedback_threshold": 0.5,
                     "eval_feedback_events": 0, "train_outcome_count": training_outcomes,
                     "train_feedback_counts": feedback_counts},
        "dataset": {"train_rows": sum(row["split"] == "train" for row in rows),
                    "eval_rows": len(paired), "source_groups": len({row["source_group"] for row in rows})},
        "policies": summaries,
        "paired_samples": paired,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--max-line-chars", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--cost-weight", type=float, default=0.1)
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    rows = load_rows(args.input, max_rows=args.max_rows, max_line_chars=args.max_line_chars)
    report = evaluate_rows(rows, seed=args.seed, min_support=args.min_support,
                           cost_weight=args.cost_weight, max_rows=args.max_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved offline table replay: {args.output} ({report['dataset']['eval_rows']} held-out rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
