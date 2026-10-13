"""Round 23 UPD: supplied online gain adaptation; no hidden truth in learners.

Catalog and protocol were written before performance runs. Standard library only.
Virtual counts enumerate explicit learner arithmetic/comparisons/exp and persistent
numeric writes plus prediction reads; they are not a Python runtime cost model.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random

GAINS = (0.01, 0.03, 0.1, 0.3, 0.7, 1.0)
LEARNING_RATE = 4.0
FORGET = 0.97
REGIMES = ("stationary_low", "stationary_noisy", "drift", "jumps", "reversals", "mixed")


def hypotheses():
    return json.loads(Path(__file__).with_name("catalog_update.json").read_text(encoding="utf-8"))


def _meter():
    return {"arithmetic": 0, "exponentials": 0, "comparisons": 0, "prediction_reads": 0, "persistent_writes": 0}


def _weights(logw, predictions, y, lr, meter):
    """All losses compare past proposals/predictions with current observed y."""
    k = len(logw)
    logw = [FORGET * old - lr * ((p - y) ** 2) / 16.0 for old, p in zip(logw, predictions)]
    top = max(logw)
    expw = [math.exp(v - top) for v in logw]
    total = sum(expw)
    weights = [v / total for v in expw]
    meter["arithmetic"] += 6 * k + k + (k - 1) + k
    meter["exponentials"] += k
    meter["comparisons"] += k - 1
    meter["persistent_writes"] += 2 * k
    return logw, weights


class FixedGain:
    def __init__(self, gain=0.2):
        self.gain = float(gain)
        self.m = 0.0
        self.cost = _meter()
        self.cost["persistent_writes"] = 2

    def predict(self):
        self.cost["prediction_reads"] += 1
        return self.m

    def observe(self, y):
        self.m = self.m + self.gain * (y - self.m)
        self.cost["arithmetic"] += 3
        self.cost["persistent_writes"] += 1

    def cells(self):
        return 2


class GainMeta:
    def __init__(self, learning_rate=LEARNING_RATE):
        self.m = 0.0
        self.gain = sum(GAINS) / len(GAINS)
        self.logw = [0.0] * len(GAINS)
        self.weights = [1.0 / len(GAINS)] * len(GAINS)
        self.proposals = [0.0] * len(GAINS)
        self.learning_rate = float(learning_rate)
        self.cost = _meter()
        self.cost["arithmetic"] = len(GAINS) + 1  # initial mean and uniform weight division
        self.cost["persistent_writes"] = self.cells()

    def predict(self):
        self.cost["prediction_reads"] += 1
        return self.m

    def observe(self, y):
        self.logw, self.weights = _weights(self.logw, self.proposals, y, self.learning_rate, self.cost)
        self.gain = sum(w * a for w, a in zip(self.weights, GAINS))
        old = self.m
        residual = y - old
        self.proposals = [old + a * residual for a in GAINS]
        self.m = old + self.gain * residual
        self.cost["arithmetic"] += (2 * len(GAINS) - 1) + 1 + 2 * len(GAINS) + 2
        self.cost["persistent_writes"] += len(GAINS) + 2

    def cells(self):
        return 3 + 3 * len(GAINS)


class ExpertBank:
    def __init__(self):
        self.states = [0.0] * len(GAINS)
        self.logw = [0.0] * len(GAINS)
        self.weights = [1.0 / len(GAINS)] * len(GAINS)
        self.cost = _meter()
        self.cost["arithmetic"] = 1  # initial uniform weight division
        self.cost["persistent_writes"] = self.cells()

    def predict(self):
        self.cost["prediction_reads"] += 2 * len(GAINS)
        self.cost["arithmetic"] += 2 * len(GAINS) - 1
        return sum(w * m for w, m in zip(self.weights, self.states))

    def observe(self, y):
        self.logw, self.weights = _weights(self.logw, self.states, y, LEARNING_RATE, self.cost)
        self.states = [m + a * (y - m) for a, m in zip(GAINS, self.states)]
        self.cost["arithmetic"] += 3 * len(GAINS)
        self.cost["persistent_writes"] += len(GAINS)

    def cells(self):
        return 3 * len(GAINS)


_CLASSES = {cls.__name__: cls for cls in (FixedGain, GainMeta, ExpertBank)}


def dumps(model):
    return json.dumps({"type": type(model).__name__, "state": model.__dict__}, sort_keys=True, separators=(",", ":"), allow_nan=False)


def loads(serialized):
    obj = json.loads(serialized)
    model = _CLASSES[obj["type"]].__new__(_CLASSES[obj["type"]])
    model.__dict__.update(obj["state"])
    return model


def _total(model):
    return sum(model.cost.values())


def make_stream(seed, regime, n=384, salt=0):
    """mu is evaluation-only; API run_model passes only observed y to learners."""
    if regime not in REGIMES:
        raise ValueError(regime)
    rng = random.Random(7919 * int(seed) + 104729 * salt + 313 * REGIMES.index(regime))
    mu = rng.uniform(-0.8, 0.8)
    direction = rng.choice((-1, 1))
    observations, latent = [], []
    for t in range(n):
        noise = 0.08
        if regime == "stationary_noisy":
            noise = 0.8
        elif regime == "drift":
            mu += direction * 0.012 + rng.uniform(-0.004, 0.004)
            noise = 0.18
        elif regime == "jumps":
            if t and t % 32 == 0:
                mu = rng.uniform(-0.9, 0.9)
            noise = 0.18
        elif regime == "reversals":
            if t and t % 24 == 0:
                direction *= -1
            mu += direction * 0.035
            noise = 0.18
        elif regime == "mixed":
            phase = (t // 48) % 4
            noise = (0.08, 0.8, 0.18, 0.18)[phase]
            if phase == 2:
                mu += direction * 0.025
            elif phase == 3 and t % 8 == 0:
                mu = rng.uniform(-0.9, 0.9)
        if mu > 0.9:
            mu, direction = 1.8 - mu, -1
        if mu < -0.9:
            mu, direction = -1.8 - mu, 1
        latent.append(mu)
        observations.append(mu + rng.uniform(-noise, noise))
    return {"observations": observations, "latent": latent}


def run_model(model, observations):
    """Learner runner never receives latent truth or regime identity."""
    predictions, gains = [], []
    for y in observations:
        predictions.append(model.predict())
        model.observe(y)
        if hasattr(model, "gain"):
            gains.append(model.gain)
    return predictions, gains


def _score(predictions, truth):
    return sum(((p - mu) / 3.0) ** 2 for p, mu in zip(predictions, truth)) / len(truth)


def _outcome(model, stream):
    predictions, gains = run_model(model, stream["observations"])
    return {
        "score": _score(predictions, stream["latent"]),
        "observation_score": sum(((p - y) / 4.0) ** 2 for p, y in zip(predictions, stream["observations"])) / len(predictions),
        "cost": _total(model), "cost_breakdown": dict(model.cost),
        "numeric_cells": model.cells(), "serialized_bytes": len(dumps(model).encode("utf-8")),
        "gain_min": min(gains) if gains else None,
        "gain_max": max(gains) if gains else None,
        "gain_final": gains[-1] if gains else None,
        "predictions": predictions,
    }


def _transfer(seed, mismatch=False):
    acquisition = make_stream(seed, "drift", 256, 81)
    target = make_stream(seed, "stationary_noisy" if mismatch else "drift", 256, 83)
    learned = GainMeta()
    c_acq = _outcome(learned, acquisition)
    transferred = FixedGain(learned.gain)
    c_target = _outcome(transferred, target)
    fixed_a = _outcome(FixedGain(0.2), acquisition)
    fixed_t = _outcome(FixedGain(0.2), target)
    # A single persistent scalar is copied into the new fixed model; copy cost is
    # already its two initialization writes. All original acquisition costs count.
    return {
        "candidate": (c_acq["score"] + c_target["score"]) / 2,
        "control": (fixed_a["score"] + fixed_t["score"]) / 2,
        "costs": {"candidate": c_acq["cost"] + c_target["cost"], "control": fixed_a["cost"] + fixed_t["cost"]},
        "diagnostics": {
            "acquisition_error": {"candidate": c_acq["score"], "control": fixed_a["score"]},
            "target_error": {"candidate": c_target["score"], "control": fixed_t["score"]},
            "acquisition_cost": {"candidate": c_acq["cost"], "control": fixed_a["cost"]},
            "target_cost": {"candidate": c_target["cost"], "control": fixed_t["cost"]},
            "transferred_gain": learned.gain, "transferred_cells": 1,
            "memory_peak_numeric_cells": {"candidate": learned.cells(), "control": 2},
            "final_serialized_bytes": {"candidate": c_target["serialized_bytes"], "control": fixed_t["serialized_bytes"]},
            "candidate_first_target_prediction": c_target["predictions"][0],
            "control_first_target_prediction": fixed_t["predictions"][0],
            "target_observation_error": {"candidate": c_target["observation_score"], "control": fixed_t["observation_score"]},
        },
    }


def run(seed: int):
    cache = {}
    diagnostic_oracles = {}
    for regime in REGIMES:
        stream = make_stream(seed, regime)
        gain = 0.05 if regime.startswith("stationary") else 0.2
        cache[(regime, "gain_meta")] = _outcome(GainMeta(), stream)
        cache[(regime, "fixed_005" if gain == 0.05 else "fixed_02")] = _outcome(FixedGain(gain), stream)
        if regime in ("stationary_noisy", "drift", "jumps", "mixed"):
            cache[(regime, "expert_bank")] = _outcome(ExpertBank(), stream)
        # Retrospective oracle uses hidden truth only inside this external scorer.
        # It is never selected as a primary control and never feeds any learner.
        scores = [_outcome(FixedGain(a), stream)["score"] for a in GAINS]
        index = min(range(len(scores)), key=scores.__getitem__)
        diagnostic_oracles[regime] = {"label": "retrospective_hidden_truth_fixed_gain_oracle_diagnostic_only", "gain": GAINS[index], "score": scores[index]}
    results = {}
    for h in hypotheses():
        if h["status"] != "wave1":
            continue
        if h["regime"].startswith("transfer"):
            results[h["id"]] = _transfer(seed, h["regime"] == "transfer_mismatch")
            continue
        c = cache[(h["regime"], h["contrast"]["candidate"])]
        b = cache[(h["regime"], h["contrast"]["control"])]
        results[h["id"]] = {
            "candidate": c["score"], "control": b["score"],
            "costs": {"candidate": c["cost"], "control": b["cost"]},
            "diagnostics": {
                "memory_numeric_cells": {"candidate": c["numeric_cells"], "control": b["numeric_cells"]},
                "serialized_bytes": {"candidate": c["serialized_bytes"], "control": b["serialized_bytes"]},
                "observation_error": {"candidate": c["observation_score"], "control": b["observation_score"]},
                "gain_range": [c["gain_min"], c["gain_max"]], "gain_final": c["gain_final"],
                "cost_breakdown": {"candidate": c["cost_breakdown"], "control": b["cost_breakdown"]},
                "retrospective_oracle": diagnostic_oracles[h["regime"]],
            },
        }
    checks = {
        "all_scores_bounded": all(0 <= r[s] <= 1 for r in results.values() for s in ("candidate", "control")),
        "positive_finite_costs": all(math.isfinite(v) and v > 0 for r in results.values() for v in r["costs"].values()),
        "twelve_registered_claims": len(results) == 12,
        "transfer_state_reset": all(results[h]["diagnostics"][key] == 0 for h in ("UPD11", "UPD12") for key in ("candidate_first_target_prediction", "control_first_target_prediction")),
        "gain_family_bounds": all(min(GAINS) <= r["diagnostics"]["gain_final"] <= max(GAINS) for h, r in results.items() if not h in ("UPD11", "UPD12")),
    }
    return {"hypotheses": results, "checks": checks, "meta": {
        "family": "UPD", "seed": seed, "stream_steps": 384, "transfer_steps": [256, 256],
        "supplied_gain_family": list(GAINS), "learning_rate": LEARNING_RATE, "weight_retention": FORGET,
        "scoring": "whole_life_prequential_latent_squared_error_divided_by_9",
        "cost_units": "explicit_scalar_arithmetic_comparison_exp_prediction_read_persistent_numeric_write",
        "oracle_diagnostic_cost_excluded_from_learners": True,
        "novelty_claim": False,
    }}


def selfcheck():
    """Deterministic mechanics only, not discovery/confirmation performance."""
    checks = []
    observations = [0.2, -0.4, 0.7, 0.1, -0.8, 0.9, 0.0, 0.5, -0.1]
    for gain in (0.01, 0.2, 1.0):
        fixed = FixedGain(gain)
        for t, y in enumerate(observations):
            p = fixed.predict()
            closed = sum(gain * ((1 - gain) ** (t - 1 - j)) * observations[j] for j in range(t))
            checks.append(math.isclose(p, closed, rel_tol=1e-12, abs_tol=1e-12))
            fixed.observe(y)
    # Independently reconstruct one update with ordinary scalar loops, not the
    # helper or observation method. Arbitrary nonuniform supplied old state.
    model = GainMeta()
    model.m = 0.23
    model.logw = [-0.3, 0.1, -0.7, 0.8, -0.2, 0.4]
    model.proposals = [-0.8, -0.1, 0.2, 0.6, 0.9, 1.1]
    old = copy.deepcopy(model)
    raw = []
    y = -0.31
    for j in range(6):
        error = old.proposals[j] - y
        raw.append(math.exp(0.97 * old.logw[j] - (error * error) / 4.0))
    denominator = 0.0
    for value in raw:
        denominator += value
    expected_gain = 0.0
    for j, a in enumerate(GAINS):
        expected_gain += a * raw[j] / denominator
    expected_m = (1 - expected_gain) * old.m + expected_gain * y
    expected_proposals = [(1 - a) * old.m + a * y for a in GAINS]
    model.observe(y)
    checks.extend([math.isclose(model.gain, expected_gain, abs_tol=1e-14), math.isclose(model.m, expected_m, abs_tol=1e-14)])
    checks.extend(math.isclose(a, b, abs_tol=1e-14) for a, b in zip(model.proposals, expected_proposals))
    # With learning disabled all six log-weights stay equal: effective gain is
    # the arithmetic mean. Independent fixed recurrence must match.
    disabled, fixed = GainMeta(0.0), FixedGain(sum(GAINS) / len(GAINS))
    a, ag = run_model(disabled, observations)
    b, _ = run_model(fixed, observations)
    checks.extend(math.isclose(x, z, abs_tol=1e-14) for x, z in zip(a, b))
    checks.extend(math.isclose(g, sum(GAINS) / len(GAINS), abs_tol=1e-14) for g in ag)
    active = GainMeta()
    _, active_gains = run_model(active, observations)
    checks.append(any(abs(g - ag[i]) > 1e-6 for i, g in enumerate(active_gains)))
    # Serialized continuation at an interior point, including all cost counters.
    for constructor in (FixedGain, GainMeta, ExpertBank):
        original = constructor()
        run_model(original, observations[:4])
        resumed = loads(dumps(original))
        first = run_model(original, observations[4:])
        second = run_model(resumed, observations[4:])
        checks.extend([first == second, dumps(original) == dumps(resumed)])
    # Scalar expert-bank reference after two observations.
    bank = ExpertBank()
    run_model(bank, [0.6])
    checks.extend(math.isclose(state, a * 0.6, abs_tol=1e-14) for state, a in zip(bank.states, GAINS))
    expected_bank_l = [0.97 * (-4 * 0.6 ** 2 / 16) - 4 * ((a * 0.6 + 0.2) ** 2) / 16 for a in GAINS]
    bank.observe(-0.2)
    checks.extend(math.isclose(x, z, abs_tol=1e-14) for x, z in zip(bank.logw, expected_bank_l))
    expected_states = [a * 0.6 + a * (-0.2 - a * 0.6) for a in GAINS]
    expected_w = [math.exp(l) / sum(math.exp(v) for v in expected_bank_l) for l in expected_bank_l]
    checks.append(math.isclose(bank.predict(), sum(w * m for w, m in zip(expected_w, expected_states)), abs_tol=1e-14))
    # Same observation sequence with two distinct hidden-truth arrays: identical
    # learner trace/state; only the separate evaluator's metric can change.
    one, two = GainMeta(), GainMeta()
    pred1, _ = run_model(one, observations)
    pred2, _ = run_model(two, observations)
    checks.append(dumps(one) == dumps(two) and pred1 == pred2)
    checks.append(_score(pred1, [0.0] * len(observations)) != _score(pred2, [0.8] * len(observations)))
    # Copy only gain into fresh state; no live state transfer.
    frozen = FixedGain(one.gain)
    checks.extend([frozen.m == 0.0, frozen.gain == one.gain, not hasattr(frozen, "logw")])
    # Catalog coverage is a registry check, not a scientific finding.
    catalog = hypotheses()
    checks.append(len(catalog) == 60 and len({h["id"] for h in catalog}) == 60)
    checks.extend(sum(h["family"] == family for h in catalog) == 20 for family in ("UPD", "CRD", "CTX"))
    checks.append(sum(h["status"] == "wave1" for h in catalog) == 12)
    return {"checks": len(checks), "passed": all(checks), "details": {
        "failed_check_indices": [i for i, passed in enumerate(checks) if not passed],
        "fixed_closed_form": 27, "independent_meta_step": 8,
        "mechanism_ablation": "zero_learning_rate_equals_fixed_mean_gain",
        "continuation": "all_three_learners_json_state_and_cost_exact",
        "truth_firewall": "run_model_accepts_observations_only",
        "main_seed_performance_runs": 0,
    }}


if __name__ == "__main__":
    print(json.dumps(selfcheck(), ensure_ascii=False, indent=2))
