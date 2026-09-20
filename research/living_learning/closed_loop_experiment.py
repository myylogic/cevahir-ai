"""Policy-induced evidence, endogenous retesting, and learned-limit lock-in.

No biological claims. Tiny finite simulated environment; all priors and costs
are supplied. Deterministic policies are compared with matched-rate randomness.
The Bayesian controller is exactly planned for an absorbing geometric change,
not a novel algorithm or a general safe-exploration solution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import time
from pathlib import Path

HORIZON = 240
HAZARD = 0.01
PERIOD = 20
SEEDS = 200
POLICIES = ("frozen_failure", "internal_timer", "random_retest", "irrelevant_motion", "belief_planning")


def optimal_retest_policy(horizon, hazard, failure_cost):
    """Exact finite-horizon belief-state DP for closed->open geometric change.

    A failed attempt establishes closed now. Before a decision, `age` transitions
    have elapsed since the last failure. Success establishes permanently open,
    in the model. Waiting yields no environmental signal. Internal aging alone
    changes the prior prediction, NOT the amount of external evidence.
    """
    value = [[0.0] * (horizon + 2) for _ in range(horizon + 1)]
    attempt = [[False] * (horizon + 2) for _ in range(horizon + 1)]
    for remaining in range(1, horizon + 1):
        for age in range(1, horizon - remaining + 2):
            probability_open = 1.0 - (1.0 - hazard) ** age
            try_value = probability_open * remaining + (1.0 - probability_open) * (
                -failure_cost + value[remaining - 1][1])
            wait_value = value[remaining - 1][age + 1]
            attempt[remaining][age] = try_value > wait_value + 1e-12
            value[remaining][age] = max(try_value, wait_value)
    return attempt, value[horizon][1]


def simulate(policy, change_at, failure_cost, seed, planned, transient=False, trace=False):
    # The t=0 failure is acquired experience shared by all controllers.
    # Its cost is a common sunk cost and excluded from reported t=1..T utility.
    rng = random.Random(seed + 99173)
    learned_success = False
    last_failure = 0
    first_success = None
    success_count = fail_count = information_actions = motions = 0
    rewards = 0.0
    observations = []
    no_feedback = 0
    for t in range(1, HORIZON + 1):
        age = t - last_failure
        if learned_success:
            action = "attempt"
        elif policy == "frozen_failure":
            action = "avoid"
        elif policy == "internal_timer":
            action = "attempt" if age >= PERIOD else "avoid"
        elif policy == "random_retest":
            action = "attempt" if rng.random() < 1 / PERIOD else "avoid"
        elif policy == "irrelevant_motion":
            action = "motion" if t % PERIOD == 0 else "avoid"
        elif policy == "belief_planning":
            action = "attempt" if planned[HORIZON - t + 1][age] else "avoid"
        else:
            raise ValueError(policy)
        # Only evaluator/environment sees change_at. All passive observations
        # remain the same fixed symbol, irrespective of the hidden world.
        if action == "attempt":
            information_actions += 1
            opened = change_at is not None and t >= change_at
            if transient:
                opened = opened and t < change_at + 8
            y = int(opened)
            learned_success = bool(y)
            if y:
                first_success = t if first_success is None else first_success
                success_count += 1
                rewards += 1
            else:
                last_failure = t
                fail_count += 1
                rewards -= failure_cost
        else:
            y = None
            motions += action == "motion"
            no_feedback += 1
        if trace:
            observations.append({"t": t, "outside": "unchanged", "age": age,
                                 "action": action, "feedback": y})
    change_within_horizon = change_at is not None and change_at <= HORIZON
    delay = first_success - change_at if first_success is not None else None
    # Detecting a one-step-open change on its exact opening has delay zero.
    # Capped delay does not substitute for the separately reported detection rate.
    cap = HORIZON - change_at + 1 if change_within_horizon else None
    oracle_successes = (min(8, HORIZON - change_at + 1) if transient else HORIZON - change_at + 1) if change_within_horizon else 0
    result = {"utility": rewards, "successes": success_count, "failed_attempts": fail_count,
              "attempts": information_actions, "irrelevant_motions": motions,
              "steps_without_feedback": no_feedback,
              "change_within_horizon": change_within_horizon,
              "detected": first_success is not None, "first_success": first_success,
              "detection_delay": delay,
              "detection_delay_capped": delay if delay is not None else cap,
              "oracle_utility": oracle_successes, "regret_to_informed_oracle": oracle_successes - rewards}
    if trace:
        result["trace"] = observations
    return result


def summarize(records):
    changed = [r for r in records if r["change_within_horizon"]]
    keys = ("utility", "successes", "failed_attempts", "attempts", "irrelevant_motions", "regret_to_informed_oracle")
    output = {k: statistics.mean(r[k] for r in records) for k in keys}
    output["episodes"] = len(records)
    output["changed_episodes"] = len(changed)
    output["detection_rate_given_change"] = statistics.mean(r["detected"] for r in changed) if changed else None
    output["mean_capped_delay_given_change"] = statistics.mean(r["detection_delay_capped"] for r in changed) if changed else None
    return output


def paired_difference(records, reference, key):
    differences = [a[key] - b[key] for a, b in zip(records, reference)]
    mean = statistics.mean(differences)
    se = statistics.stdev(differences) / math.sqrt(len(differences))
    return {"mean": mean, "normal_approx_95_ci": [mean - 1.96 * se, mean + 1.96 * se]}


def exact_model_utility(policy, planned, cost):
    # Integrate over EVERY possible geometric opening date (plus the tail).
    total = 0.0
    for tau in range(1, HORIZON + 1):
        weight = HAZARD * (1 - HAZARD) ** (tau - 1)
        total += weight * simulate(policy, tau, cost, 0, planned)["utility"]
    total += (1 - HAZARD) ** HORIZON * simulate(policy, None, cost, 0, planned)["utility"]
    return total


def causal_controls(planned):
    unchanged = simulate("frozen_failure", None, 1, 0, planned, trace=True)
    changed = simulate("frozen_failure", 80, 1, 0, planned, trace=True)
    assert unchanged["trace"] == changed["trace"]
    timer = simulate("internal_timer", 80, 1, 0, planned, trace=True)
    motion = simulate("irrelevant_motion", 80, 1, 0, planned, trace=True)
    assert timer["first_success"] == 80 and not motion["detected"]
    for tau in range(1, HORIZON + 1):
        r = simulate("internal_timer", tau, 1, 0, planned)
        assert r["detected"] and r["detection_delay"] < PERIOD
    # Self-capacity vs environment: same repeated barrier failure has two causes.
    explanations = {"low_capacity": {"capacity": 0, "difficulty": 1},
                    "hard_environment": {"capacity": 1, "difficulty": 2}}
    failure_histories = {name: [int(v["capacity"] >= v["difficulty"])] * 10 for name, v in explanations.items()}
    calibration = {name: int(v["capacity"] >= 1) for name, v in explanations.items()}
    assert len({tuple(x) for x in failure_histories.values()}) == 1
    assert len(set(calibration.values())) == 2
    # Large single-event action change: supplied shared latent class & cost.
    prior = 0.05
    risk_before = prior * 0.9 + (1 - prior) * 0.01
    posterior = prior * 0.9 / risk_before
    risk_after = posterior * 0.9 + (1 - posterior) * 0.01
    threshold = 1 / 10  # success benefit 1, harm cost 9
    assert risk_before < threshold < risk_after
    # One correct demonstration underdetermines a behavior. Own binary feedback
    # can refine it. All four Boolean policy templates are supplied beforehand.
    behavior_templates = ((0, 0), (1, 1), (0, 1), (1, 0))
    demo_context, demo_action = 0, 0
    after_demo = [p for p in behavior_templates if p[demo_context] == demo_action]
    own_context = 1
    attempted = after_demo[0][own_context]  # deterministic, simple first template
    own_success = attempted == own_context  # environment alone computes reward
    after_own_experience = [p for p in after_demo if (p[own_context] == attempted) == own_success]
    assert len(after_demo) == 2 and after_own_experience == [(0, 1)]
    assert not own_success
    novel_objects = [(f"new_object_{i}", i % 2) for i in range(20)]
    refined_predictions = [after_own_experience[0][context] for _, context in novel_objects]
    frozen_predictions = [after_demo[0][context] for _, context in novel_objects]
    target_actions = [context for _, context in novel_objects]
    assert refined_predictions == target_actions
    return {"avoidance_histories_identical_across_changed_and_unchanged_worlds": True,
            "deterministic_timer_detects_all_240_persistent_change_dates": True,
            "timer_worst_case_delay": PERIOD - 1,
            "irrelevant_motion_does_not_detect_change": True,
            "same_external_observation_different_internal_age_changes_action": True,
            "imitation_then_own_experience": {
                "demonstration_count": 1, "own_feedback_count": 1,
                "compatible_policies_after_demo": [list(p) for p in after_demo],
                "compatible_policies_after_own_feedback": [list(p) for p in after_own_experience],
                "first_self_attempt_succeeds": own_success,
                "refined_correct_on_new_objects": sum(a == b for a, b in zip(refined_predictions, target_actions)),
                "frozen_imitation_correct_on_new_objects": sum(a == b for a, b in zip(frozen_predictions, target_actions)),
                "new_object_count": len(novel_objects),
                "caveat": "Given two observable contexts, two actions, correct demonstrator and four supplied deterministic policy templates. Reward feedback identifies the other action after failure only because actions are binary and exactly one is correct. Novel object IDs are not novel feature values. Not general visual imitation."},
            "trace_examples": {"frozen": changed["trace"], "timer": timer["trace"]},
            "self_vs_world": {"explanations": explanations, "identical_failure_histories": failure_histories,
                              "anchored_calibration_results": calibration,
                              "caveat": "Calibration at independently known difficulty 1 is additional identifying information; without it the two causes cannot be distinguished from these failure records."},
            "one_event": {"prior_high_risk_class": prior, "harm_risk_before": risk_before,
                          "posterior_high_risk_class_after_one_harm": posterior, "harm_risk_after": risk_after,
                          "harm_risk_decision_threshold": threshold,
                          "action_before": "attempt", "action_after": "avoid",
                          "caveat": "Illustrative Bayes calculation; likelihoods, shared category and utilities are supplied; not evidence about childhood learning or a learned feature representation."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "closed_loop.json")
    args = parser.parse_args()
    start = time.perf_counter()
    report = {"experiment": "policy_induced_evidence_and_endogenous_retesting", "python": platform.python_version(),
              "protocol": {"horizon": HORIZON, "hazard": HAZARD, "timer_period": PERIOD,
                           "random_attempt_probability": 1 / PERIOD, "seeds": SEEDS,
                           "costs": [1, 20], "initial_shared_experience": "one failed attempt at t=0",
                           "conditions": ["never_changes", "absorbing_geometric", "transient_8_steps"],
                           "passive_observation": "unchanged symbol, no hidden change or capacity label",
                           "cost_scope": "environment utility counts success +1 and failure -cost, no feedback for waiting; CPU/planning cost and t=0 common sunk cost excluded",
                           "pre_specification": "constants, policies and conditions fixed in code before first run; no independent preregistration",
                           "confidence_intervals": "normal approximation on paired episode differences, no multiplicity correction"},
              "conditions": {}, "exact_geometric_checks": {}}
    plans = {}
    for cost in (1, 20):
        planned, optimum = optimal_retest_policy(HORIZON, HAZARD, cost)
        plans[cost] = planned
        exact = exact_model_utility("belief_planning", planned, cost)
        exact_timer = exact_model_utility("internal_timer", planned, cost)
        assert math.isclose(exact, optimum, abs_tol=1e-8)
        assert exact >= exact_timer - 1e-8 and exact >= -1e-8
        report["exact_geometric_checks"][str(cost)] = {"dp_value": optimum, "enumerated_value": exact,
                                                        "enumerated_timer_value": exact_timer,
                                                        "belief_state_cells": HORIZON * (HORIZON + 1) // 2}
        for condition in ("never_changes", "absorbing_geometric", "transient_8_steps"):
            records = {p: [] for p in POLICIES}
            tau_values = []
            for seed in range(SEEDS):
                rng = random.Random(seed)
                tau = math.floor(math.log1p(-rng.random()) / math.log1p(-HAZARD)) + 1
                if condition == "never_changes":
                    tau = None
                tau_values.append(tau)
                for policy in POLICIES:
                    records[policy].append(simulate(policy, tau, cost, seed, planned,
                                                    transient=condition == "transient_8_steps"))
            key = f"{condition}_cost_{cost}"
            report["conditions"][key] = {"change_dates": tau_values,
                "summary": {p: summarize(rs) for p, rs in records.items()},
                "paired_utility_vs_timer": {p: paired_difference(rs, records["internal_timer"], "utility") for p, rs in records.items()},
                "episodes": records}
    report["causal_controls"] = causal_controls(plans[1])
    report["elapsed_seconds"] = time.perf_counter() - start
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "all_assertions_passed": True,
                      "elapsed_seconds": report["elapsed_seconds"],
                      "exact_checks": report["exact_geometric_checks"],
                      "summaries": {k: v["summary"] for k, v in report["conditions"].items()}}, indent=2))


if __name__ == "__main__":
    main()
