"""Exact finite mixture-model learning of whether/when an avoided option changes.

The learner is given three possible hazard rates, including zero, not the true
rate or opening time. Only its own probes produce feedback. A complete failure
history has sufficient statistic `last_failure`: repeated failures are nested
survival events, not independent observations. Standard Python, no engine code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

T = 240
HAZARDS = (0.0, 0.002, 0.02)
WEIGHTS = (1/3, 1/3, 1/3)
POLICIES = ("updated_mixture", "frozen_hyperprior", "fixed_hazard_001", "never_probe")


def survival(t):
    return sum(w * (1 - h) ** t for w, h in zip(WEIGHTS, HAZARDS))


def posterior(last_failure):
    terms = [w * (1 - h) ** last_failure for w, h in zip(WEIGHTS, HAZARDS)]
    return [v / sum(terms) for v in terms]


def mixture_plan(cost):
    value = [[0.0] * (T + 1) for _ in range(T + 2)]
    attempt = [[False] * (T + 1) for _ in range(T + 2)]
    for t in range(T, 0, -1):
        n = T - t + 1
        for f in range(t):
            p_open = 1 - survival(t) / survival(f)
            probe = p_open * n + (1 - p_open) * (-cost + value[t+1][t])
            wait = value[t+1][f]
            attempt[t][f] = probe > wait + 1e-12
            value[t][f] = max(probe, wait)
    return attempt, value[1][0]


def age_plan(cost, fixed_hazard=None):
    """Ablation: reset hyperprior after each failure, instead of learning it.

    This is a coherent policy for a different model where the latent rate is
    redrawn after failed probes, NOT correct inference for a fixed latent rate.
    """
    value = [[0.0] * (T + 2) for _ in range(T + 1)]
    attempt = [[False] * (T + 2) for _ in range(T + 1)]
    for n in range(1, T + 1):
        for age in range(1, T - n + 2):
            p_open = 1 - (survival(age) if fixed_hazard is None else (1-fixed_hazard)**age)
            probe = p_open * n + (1-p_open) * (-cost + value[n-1][1])
            wait = value[n-1][age+1]
            attempt[n][age] = probe > wait + 1e-12
            value[n][age] = max(probe, wait)
    return attempt


def run(policy, tau, cost, plans, trace=False):
    f = 0
    success_known = False
    utility = 0
    failures = successes = 0
    first_success = None
    events = []
    for t in range(1, T+1):
        if success_known:
            probe = True
        elif policy == "updated_mixture":
            probe = plans[policy][t][f]
        elif policy in ("frozen_hyperprior", "fixed_hazard_001", "oracle_hazard"):
            probe = plans[policy][T-t+1][t-f]
        elif policy == "never_probe":
            probe = False
        else:
            raise ValueError(policy)
        if not probe:
            continue
        y = int(tau is not None and t >= tau)
        # Environment truth affects learner only through this actual feedback.
        if y:
            successes += 1
            utility += 1
            success_known = True
            if first_success is None:
                first_success = t
        else:
            f = t
            failures += 1
            utility -= cost
        if trace and not y:
            events.append({"time": t, "feedback": y, "last_failure": f,
                           "posterior_hazard_weights": posterior(f) if policy == "updated_mixture" else (list(WEIGHTS) if policy == "frozen_hyperprior" else None)})
    return {"utility": utility, "failures": failures, "successes": successes,
            "first_success": first_success, "detected": first_success is not None,
            "last_failure": f, "posterior_hazard_weights_given_last_failure": posterior(f),
            "posterior_diagnostic_scope": "conditional on last failure only; after success this is NOT the full-history posterior; policy does not use it after success",
            "failure_events": events}


def exact_evaluation(policy, hazard, cost, plans):
    # All possible opening times plus no opening by end of finite horizon.
    cases = []
    for tau in range(1, T+1):
        weight = hazard * (1-hazard) ** (tau-1)
        cases.append((weight, run(policy, tau, cost, plans)))
    tail = (1-hazard)**T
    never = run(policy, None, cost, plans, trace=True)
    cases.append((tail, never))
    assert math.isclose(sum(w for w, _ in cases), 1.0, abs_tol=1e-12)
    output = {k: sum(w * r[k] for w, r in cases) for k in ("utility", "failures", "successes")}
    mass_open = 1 - tail
    output["probability_open_by_horizon"] = mass_open
    output["detection_given_open"] = sum(w * r["detected"] for w, r in cases) / mass_open if mass_open else None
    output["no_opening_by_horizon"] = never
    return output


def controls():
    f = 150
    # Conditional survival likelihoods telescope; duplicate evidence is illegal.
    event_times = [10, 35, 90, f]
    sequential = list(WEIGHTS)
    previous = 0
    for t in event_times:
        sequential = [w * (1-h)**(t-previous) for w, h in zip(sequential, HAZARDS)]
        sequential = [v / sum(sequential) for v in sequential]
        previous = t
    expected = posterior(f)
    assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(sequential, expected))
    assert 0 < expected[0] < 1
    assert posterior(0) == list(WEIGHTS)
    t = 200
    p_by_mixture = sum(w * (1 - (1-h)**(t-f)) for w, h in zip(expected, HAZARDS))
    assert math.isclose(p_by_mixture, 1-survival(t)/survival(f), abs_tol=1e-12)
    return {"failure_times": event_times,
            "sequential_posterior_equals_latest_failure_sufficient_statistic": True,
            "posterior_after_only_a_failure_at_150": expected,
            "posterior_after_failures_at_10_35_90_150": sequential,
            "finite_failure_does_not_prove_zero_hazard": True,
            "waiting_without_feedback_keeps_model_weights_at_last_failure": True,
            "failure_count_is_not_independent_evidence_count": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "unknown_change.json")
    args = parser.parse_args()
    started = time.perf_counter()
    result = {"experiment": "closed_loop_unknown_hazard_mixture", "python": platform.python_version(),
              "protocol": {"horizon": T, "hazard_hypotheses": HAZARDS, "initial_weights": WEIGHTS,
                           "costs": [1, 20], "matched_hazards": HAZARDS,
                           "out_of_support_hazard": 0.08,
                           "exact_integration": "all opening dates 1..240 plus survival tail, no Monte Carlo intervals needed",
                           "cost_scope": "environment rewards only; DP compute and common initial t=0 failure excluded",
                           "given_bias": "known hazard candidate set, noiseless probes, absorbing opening, known reward/cost/horizon",
                           "pre_specification": "hypotheses, costs, controls fixed before first run; no external preregistration"},
              "cost_conditions": {}, "controls": controls()}
    for cost in (1, 20):
        mixture, optimum = mixture_plan(cost)
        plans = {"updated_mixture": mixture, "frozen_hyperprior": age_plan(cost),
                 "fixed_hazard_001": age_plan(cost, 0.01)}
        condition = {"dp_optimal_prior_value": optimum, "hazard_conditions": {}}
        for hazard in (*HAZARDS, 0.08):
            plans["oracle_hazard"] = age_plan(cost, hazard)
            condition["hazard_conditions"][str(hazard)] = {
                p: exact_evaluation(p, hazard, cost, plans) for p in (*POLICIES, "oracle_hazard")}
        mixture_values = {p: sum(w * condition["hazard_conditions"][str(h)][p]["utility"]
                                for w, h in zip(WEIGHTS, HAZARDS)) for p in (*POLICIES, "oracle_hazard")}
        assert math.isclose(mixture_values["updated_mixture"], optimum, abs_tol=1e-8)
        assert all(optimum >= mixture_values[p] - 1e-8 for p in POLICIES)
        assert mixture_values["oracle_hazard"] >= optimum - 1e-8
        condition["exact_prior_average_utility"] = mixture_values
        # Two histories, same current exterior and elapsed time since last probe:
        # different last-failure absolute times can yield different model weights.
        condition["no_change_attempt_times"] = {
            p: [e["time"] for e in condition["hazard_conditions"]["0.0"][p]["no_opening_by_horizon"]["failure_events"]]
            for p in POLICIES}
        result["cost_conditions"][str(cost)] = condition
    result["elapsed_seconds"] = time.perf_counter()-started
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "all_assertions_passed": True,
                     "elapsed_seconds": result["elapsed_seconds"],
                     "costs": {k: {"prior_utility": v["exact_prior_average_utility"],
                                   "attempt_times_no_change": v["no_change_attempt_times"]}
                               for k, v in result["cost_conditions"].items()}}, indent=2))


if __name__ == "__main__":
    main()
