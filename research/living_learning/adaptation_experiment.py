"""Small, reproducible living-learning counterexample and transfer experiment.

Only Python's standard library is used. No Cevahir runtime is imported or changed.
Run: python research/living_learning/adaptation_experiment.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


DIMENSIONS = 6
INPUTS = 1 << DIMENSIONS
HYPOTHESES = INPUTS * 2
TRAIN_TASKS = 24
TEST_TASKS = 8
QUERY_BUDGET = 16
DEFAULT_SEEDS = 40
MODELS = ("uniform_reset", "learned_family", "scrambled_family")
PREDICTIONS = tuple(
    tuple(((h // 2 & x).bit_count() + h % 2) % 2 for x in range(INPUTS))
    for h in range(HYPOTHESES)
)
GROUPS = tuple(
    0 if (h // 2).bit_count() in (1, 2)
    else 1 if (h // 2).bit_count() in (4, 5)
    else 2
    for h in range(HYPOTHESES)
)
GROUP_SIZES = tuple(GROUPS.count(g) for g in range(3))


def probabilities(counts: list[float], groups: tuple[int, ...]) -> list[float]:
    total = sum(counts)
    sizes = [groups.count(g) for g in range(3)]
    return [counts[g] / total / sizes[g] for g in groups]


def predict_one(candidates: list[int], prior: list[float], x: int) -> float:
    denominator = sum(prior[h] for h in candidates)
    probability = sum(prior[h] for h in candidates if PREDICTIONS[h][x]) / denominator
    # Symmetric posteriors have exact half-probability; do not turn roundoff into a decision.
    return 0.5 if abs(probability - 0.5) < 1e-12 else probability


def heldout_risk(candidates: list[int], prior: list[float], xs: list[int], h: int) -> float:
    return sum(
        int(predict_one(candidates, prior, x) > 0.5) != PREDICTIONS[h][x]
        for x in xs
    ) / len(xs)


def identify_online(h: int, xs: list[int]) -> tuple[int, int]:
    """Update after each label; return only a hypothesis actually identified by data."""
    candidates = list(range(HYPOTHESES))
    for count, x in enumerate(xs, 1):
        y = PREDICTIONS[h][x]
        candidates = [candidate for candidate in candidates if PREDICTIONS[candidate][x] == y]
        if len(candidates) == 1:
            return candidates[0], count
    raise AssertionError("Full truth table did not identify a unique affine function")


def evaluate_episode(h: int, xs: list[int], prior: list[float]) -> dict:
    candidates = list(range(HYPOTHESES))
    queries, heldout = xs[:QUERY_BUDGET], xs[QUERY_BUDGET:]
    assert not set(queries).intersection(heldout)
    risks = [heldout_risk(candidates, prior, heldout, h)]
    mistakes, log_loss_bits = 0, 0.0
    identified_at = None
    for count, x in enumerate(queries, 1):
        y = PREDICTIONS[h][x]
        p = predict_one(candidates, prior, x)
        mistakes += int(int(p > 0.5) != y)
        true_probability = p if y else 1.0 - p
        assert true_probability > 0.0  # positive prior and noiseless realizability
        log_loss_bits -= math.log2(true_probability)
        candidates = [candidate for candidate in candidates if PREDICTIONS[candidate][x] == y]
        assert h in candidates
        if len(candidates) == 1 and identified_at is None:
            identified_at = count
        risks.append(heldout_risk(candidates, prior, heldout, h))
    # Retrospective evaluator statistic: the learner does not see held-out labels.
    sustained_zero = next(
        (k for k in range(len(risks)) if all(r == 0.0 for r in risks[k:])),
        QUERY_BUDGET + 1,
    )
    if identified_at is not None:
        # Independent closed-form check: noiseless Bayesian losses telescope to surprisal.
        assert math.isclose(log_loss_bits, -math.log2(prior[h]), abs_tol=1e-10)
    return {
        "prequential_mistakes": mistakes,
        "prequential_log_loss_bits": log_loss_bits,
        "heldout_risk_curve": risks,
        "heldout_risk_at_4": risks[4],
        "heldout_risk_at_8": risks[8],
        "risk_curve_mean": statistics.mean(risks),
        "labels_until_sustained_zero_risk_capped": sustained_zero,
        "sustained_zero_censored": int(sustained_zero == QUERY_BUDGET + 1),
        "labels_until_identified_capped": identified_at or QUERY_BUDGET + 1,
        "identification_censored": int(identified_at is None),
    }


def average_episodes(episodes: list[dict]) -> dict:
    output = {}
    for key in episodes[0]:
        if key == "heldout_risk_curve":
            output[key] = [statistics.mean(e[key][k] for e in episodes) for k in range(QUERY_BUDGET + 1)]
        else:
            output[key] = statistics.mean(e[key] for e in episodes)
    return output


def one_seed(seed: int) -> dict:
    rng = random.Random(seed)
    trained_family = seed % 2  # balance sparse-to-dense and dense-to-sparse shifts
    same_pool = [h for h, g in enumerate(GROUPS) if g == trained_family]
    shifted_pool = [h for h, g in enumerate(GROUPS) if g == 1 - trained_family]
    rng.shuffle(same_pool)
    rng.shuffle(shifted_pool)
    train = same_pool[:TRAIN_TASKS]
    same_test = same_pool[TRAIN_TASKS:TRAIN_TASKS + TEST_TASKS]
    shifted_test = shifted_pool[:TEST_TASKS]
    assert not set(train).intersection(same_test + shifted_test)
    assert len(set(train + same_test + shifted_test)) == TRAIN_TASKS + TEST_TASKS * 2
    scrambled_list = list(GROUPS)
    rng.shuffle(scrambled_list)
    scrambled = tuple(scrambled_list)
    # Total initial pseudocount is one; its marginal hypothesis prior is uniform.
    initial_counts = [size / HYPOTHESES for size in GROUP_SIZES]
    learned_counts = initial_counts.copy()
    scrambled_counts = initial_counts.copy()
    training_labels = 0
    learning_events = []
    for task, h in enumerate(train):
        xs = list(range(INPUTS))
        rng.shuffle(xs)
        identified_h, labels = identify_online(h, xs)
        training_labels += labels
        # This online hyperstate update uses the identified state, not the generator's h.
        learned_counts[GROUPS[identified_h]] += 1
        scrambled_counts[scrambled[identified_h]] += 1
        learning_events.append({"task": task, "labels": labels, "identified_h": identified_h})
    priors = {
        "uniform_reset": [1 / HYPOTHESES] * HYPOTHESES,
        "learned_family": probabilities(learned_counts, GROUPS),
        "scrambled_family": probabilities(scrambled_counts, scrambled),
    }
    all_results = {}
    episode_records = {}
    for condition, targets in (("same_family", same_test), ("family_shift", shifted_test)):
        episode_records[condition] = []
        results = {model: [] for model in MODELS}
        for h in targets:
            xs = list(range(INPUTS))
            rng.shuffle(xs)
            record = {"h": h, "query_inputs": xs[:QUERY_BUDGET], "heldout_inputs": xs[QUERY_BUDGET:], "models": {}}
            for model in MODELS:
                result = evaluate_episode(h, xs, priors[model])
                record["models"][model] = result
                results[model].append(result)
            assert len({record["models"][model]["labels_until_identified_capped"] for model in MODELS}) == 1
            episode_records[condition].append(record)
        all_results[condition] = {model: average_episodes(results[model]) for model in MODELS}
    return {
        "seed": seed,
        "trained_family": trained_family,
        "training_labels": training_labels,
        "meta_train_tasks": train,
        "same_family_tasks": same_test,
        "shifted_family_tasks": shifted_test,
        "meta_training_events": learning_events,
        "learned_group_counts": learned_counts,
        "scrambled_group_counts": scrambled_counts,
        "scrambled_hypothesis_groups": list(scrambled),
        "learned_mass_on_trained_family": learned_counts[trained_family] / sum(learned_counts),
        "metrics": all_results,
        "episodes": episode_records,
    }


def mean_and_ci(values: list[float]) -> dict:
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "normal_approx_95_ci": [mean - 1.96 * se, mean + 1.96 * se], "seed_count": len(values)}


def summarize(runs: list[dict]) -> dict:
    summary = {}
    for condition in ("same_family", "family_shift"):
        summary[condition] = {}
        scalar_keys = [key for key in runs[0]["metrics"][condition][MODELS[0]] if key != "heldout_risk_curve"]
        for model in MODELS:
            summary[condition][model] = {
                key: mean_and_ci([r["metrics"][condition][model][key] for r in runs])
                for key in scalar_keys
            }
            summary[condition][model]["heldout_risk_curve"] = [
                mean_and_ci([r["metrics"][condition][model]["heldout_risk_curve"][k] for r in runs])
                for k in range(QUERY_BUDGET + 1)
            ]
        summary[condition]["paired_differences_vs_uniform"] = {
            model: {
                key: mean_and_ci([
                    r["metrics"][condition][model][key] - r["metrics"][condition]["uniform_reset"][key]
                    for r in runs
                ]) for key in scalar_keys
            } for model in MODELS if model != "uniform_reset"
        }
    return summary


def stability_counterexample() -> dict:
    """Known contexts remove observable aliasing; they are privileged information."""
    a = (0b001011 * 2)
    b = a + 1  # complementary functions: every identical x has conflicting labels
    identifying_inputs = [0] + [1 << bit for bit in range(DIMENSIONS)]
    contextual = {}
    uncontextual = list(range(HYPOTHESES))
    frozen = None
    reset_events = 0
    stages = []
    for context, h in (("A", a), ("B", b)):
        contextual[context] = list(range(HYPOTHESES))
        for x in identifying_inputs:
            y = PREDICTIONS[h][x]
            contextual[context] = [candidate for candidate in contextual[context] if PREDICTIONS[candidate][x] == y]
            update = [candidate for candidate in uncontextual if PREDICTIONS[candidate][x] == y]
            if not update:
                reset_events += 1
                update = [candidate for candidate in range(HYPOTHESES) if PREDICTIONS[candidate][x] == y]
            uncontextual = update
        assert contextual[context] == [h]
        if frozen is None:
            frozen = uncontextual.copy()
        uniform = [1 / HYPOTHESES] * HYPOTHESES
        stages.append({
            "after_context": context,
            "contextual_old_A_error": heldout_risk(contextual["A"], uniform, list(range(INPUTS)), a),
            "contextual_current_error": heldout_risk(contextual[context], uniform, list(range(INPUTS)), h),
            "reset_on_contradiction_old_A_error": heldout_risk(uncontextual, uniform, list(range(INPUTS)), a),
            "reset_on_contradiction_current_error": heldout_risk(uncontextual, uniform, list(range(INPUTS)), h),
            "frozen_old_A_error": heldout_risk(frozen, uniform, list(range(INPUTS)), a),
            "frozen_current_error": heldout_risk(frozen, uniform, list(range(INPUTS)), h),
        })
    # Exhaustively check the pointwise lower bound over both deterministic predictions.
    pairwise_min_errors = [
        min(sum(pred != PREDICTIONS[h][x] for h in (a, b)) for pred in (0, 1))
        for x in range(INPUTS)
    ]
    assert all(error == 1 for error in pairwise_min_errors)
    return {
        "functions": {"A": a, "B": b},
        "labels_per_context": len(identifying_inputs),
        "identifying_inputs": identifying_inputs,
        "reset_events": reset_events,
        "stages": stages,
        "balanced_hidden_context_minimum_error": sum(pairwise_min_errors) / (2 * INPUTS),
        "contextual_stored_affine_parameter_bits": 2 * (DIMENSIONS + 1),
        "single_model_affine_parameter_bits": DIMENSIONS + 1,
        "caveat": "Context identity and task boundaries are externally supplied; per-context storage grows. The hidden-context bound assumes an unpredictable, balanced context at each query, with no informative history or side channel. This is not an equal-information or equal-memory comparison.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "adaptation_results.json")
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("at least two seeds are required")
    runs = [one_seed(seed) for seed in range(args.seeds)]
    report = {
        "experiment": "online_finite_affine_family_transfer_v1",
        "protocol": {
            "dimensions": DIMENSIONS,
            "hypothesis_count": HYPOTHESES,
            "group_sizes": GROUP_SIZES,
            "meta_train_tasks_per_seed": TRAIN_TASKS,
            "test_tasks_per_condition_per_seed": TEST_TASKS,
            "query_budget": QUERY_BUDGET,
            "heldout_input_count": INPUTS - QUERY_BUDGET,
            "seeds": list(range(args.seeds)),
            "initial_total_dirichlet_pseudocount": 1,
            "prediction_tie_rule": "predict zero; probabilities within 1e-12 of 0.5 are treated as exact ties",
            "trained_family_by_seed": "seed modulo two; sparse=0, dense=1",
            "test_hyperstate": "frozen copy of final meta-training state; task posterior still updates after every label",
            "query_selection": "uniform random permutation, paired across models; heldout inputs never queried",
            "novelty_split": "train and test function identities are disjoint within each seed",
            "confidence_intervals": "normal approximation on seed-level means and paired seed-level differences; no multiplicity correction",
            "training_budget": "all models receive the same train experience stream; uniform-reset discards cross-task information",
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summary": summarize(runs),
        "stability_counterexample": stability_counterexample(),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seeds": args.seeds, "same_family": report["summary"]["same_family"]["paired_differences_vs_uniform"]["learned_family"], "family_shift": report["summary"]["family_shift"]["paired_differences_vs_uniform"]["learned_family"]}, indent=2))


if __name__ == "__main__":
    main()
