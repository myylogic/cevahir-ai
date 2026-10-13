"""Paired nonlinear acquisition/retention experiment; see the frozen protocol.

Not a Cevahir runtime integration or a new continual-learning algorithm.
The default checks results; --write explicitly records this new experiment only.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
WIDTH = 8
PARAMS = 4 * WIDTH + 1
MODES = ("frozen", "head_only", "online", "replay", "loss_halfspace",
         "tangent_fixed", "tangent_current", "tangent_guard", "replay_guard")
TANGENT = ("tangent_fixed", "tangent_current", "tangent_guard")
SEEDS = tuple(range(12))
ANCHORS = (4, 16, 64)
RATES = (.03, .12)
STEPS = 128
BATCH = 8
DRIFT_BUDGET = 1e-4


def unpack(theta):
    return theta[:WIDTH], theta[WIDTH:3*WIDTH].reshape(WIDTH, 2), theta[3*WIDTH:4*WIDTH], theta[-1]


def forward(theta, x):
    a, w, b, c = unpack(theta)
    return np.tanh(x @ w.T + b) @ a + c


def values_jacobian(theta, x):
    a, w, b, c = unpack(theta)
    h = np.tanh(x @ w.T + b)
    d = (1 - h*h) * a
    jac = np.concatenate((h, (d[:, :, None] * x[:, None, :]).reshape(len(x), -1),
                          d, np.ones((len(x), 1))), axis=1)
    return h @ a + c, jac


def gradient(theta, x, y):
    pred, jac = values_jacobian(theta, x)
    return jac.T @ (pred - y) / len(y)


def mse(theta, x, y):
    return float(np.mean((forward(theta, x) - y)**2))


def row_basis(jac, rcond):
    _, singular, vt = np.linalg.svd(jac, full_matrices=False)
    cutoff = rcond * singular[0] if len(singular) else 0.
    return vt[singular > cutoff], singular


def project(vector, basis):
    return vector - basis.T @ (basis @ vector)


def make_world(seed):
    rng = np.random.default_rng(210921 + seed)
    angle = rng.uniform(-np.pi, np.pi)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    teacher = np.zeros(PARAMS)
    a, w, b, _ = unpack(teacher)
    a[:3] = [1.2, -.7, .6]
    w[:3] = np.array([[1.1, .5], [-.4, 1.3], [1.5, -1.1]]) @ rotation.T
    b[:3] = [0., .2, -.3]
    def inputs(n, region):
        bounds = {"old": (-1., -.2), "new": (.2, 1.), "bridge": (-.2, .2)}[region]
        local = np.column_stack((rng.uniform(*bounds, n), rng.uniform(-1., 1., n)))
        return local @ rotation.T
    old_x = inputs(128, "old")
    order = rng.permutation(128)
    new_x = inputs(STEPS * BATCH, "new").reshape(STEPS, BATCH, 2)
    continuation_x = inputs(32 * BATCH, "new").reshape(32, BATCH, 2)
    tests = {region: inputs(256, region) for region in ("old", "new", "bridge")}
    init_rng = np.random.default_rng(10000 + seed)
    theta = np.concatenate((init_rng.normal(0., .25, WIDTH),
                            init_rng.normal(0., .6, 2*WIDTH),
                            init_rng.normal(0., .1, WIDTH), [0.]))
    old_y = forward(teacher, old_x)
    for _ in range(600):
        theta -= .05 * gradient(theta, old_x, old_y)
    return dict(theta=theta, teacher=teacher, old_x=old_x[order], old_y=old_y[order],
                new_x=new_x, new_y=forward(teacher, new_x.reshape(-1, 2)).reshape(STEPS, BATCH),
                continuation_x=continuation_x,
                continuation_y=forward(teacher, continuation_x.reshape(-1, 2)).reshape(32, BATCH),
                tests={k: (x, forward(teacher, x)) for k, x in tests.items()})


class Learner:
    def __init__(self, theta, mode, rate, x, y, rcond=1e-8):
        self.theta = theta.copy()
        self.mode, self.rate, self.rcond = mode, rate, rcond
        self.x, self.y = x.copy(), y.copy()
        self.reference = forward(theta, x)
        self.basis = np.empty((0, PARAMS))
        self.costs = Counter()
        self.trace = {"raw_gradient_norm_sum": 0., "projected_ratio_sum": 0.,
                      "rank_sum": 0., "max_linearized_anchor_change": 0.,
                      "max_anchor_drift_mse": 0.}
        self.steps = 0
        if mode == "tangent_fixed":
            self.basis, _ = row_basis(values_jacobian(theta, x)[1], rcond)
            self.costs["extra_jacobian_rows"] += len(x)
            self.costs["svd_calls"] += 1

    def step(self, x, y):
        prediction = forward(self.theta, x)
        self.steps += 1
        self.costs["service_forward_rows"] += len(x)
        if self.mode == "frozen":
            return prediction
        fresh = gradient(self.theta, x, y)
        self.costs["gradient_example_rows"] += len(x)
        raw_norm = float(np.linalg.norm(fresh))
        self.trace["raw_gradient_norm_sum"] += raw_norm
        direction = fresh.copy()
        jac = None
        if self.mode == "head_only":
            direction[WIDTH:4*WIDTH] = 0.
        elif self.mode in ("replay", "replay_guard", "loss_halfspace"):
            old = gradient(self.theta, self.x, self.y)
            self.costs["gradient_example_rows"] += len(self.x)
            if self.mode == "loss_halfspace":
                dot = float(fresh @ old)
                if dot < 0. and float(old @ old) > 0.:
                    direction -= dot / float(old @ old) * old
                    self.costs["halfspace_projections"] += 1
            else:
                direction = (fresh + old) / 2.
        elif self.mode in TANGENT:
            if self.mode == "tangent_fixed":
                basis = self.basis
            else:
                _, jac = values_jacobian(self.theta, self.x)
                basis, _ = row_basis(jac, self.rcond)
                self.costs["extra_jacobian_rows"] += len(self.x)
                self.costs["svd_calls"] += 1
            direction = project(direction, basis)
            self.trace["rank_sum"] += len(basis)
        ratio = float(np.linalg.norm(direction)) / max(raw_norm, 1e-300)
        self.trace["projected_ratio_sum"] += ratio
        delta = -self.rate * direction
        if jac is not None:
            self.trace["max_linearized_anchor_change"] = max(
                self.trace["max_linearized_anchor_change"], float(np.max(np.abs(jac @ delta))))
        if self.mode in ("tangent_guard", "replay_guard"):
            accepted = False
            for attempt in range(13):
                candidate = self.theta + delta * (2. ** -attempt)
                drift = float(np.mean((forward(candidate, self.x) - self.reference)**2))
                self.costs["candidate_forward_rows"] += len(self.x)
                self.costs["candidate_attempts"] += 1
                if drift <= DRIFT_BUDGET:
                    self.theta = candidate
                    self.costs["accepted_updates"] += 1
                    self.costs["halvings_on_accepted_updates"] += attempt
                    accepted = True
                    break
            if not accepted:
                self.costs["rejected_updates"] += 1
        else:
            self.theta += delta
            self.costs["accepted_updates"] += 1
        # Evaluator-only monitoring, not available to non-guard methods.
        drift = float(np.mean((forward(self.theta, self.x) - self.reference)**2))
        self.trace["max_anchor_drift_mse"] = max(self.trace["max_anchor_drift_mse"], drift)
        if not np.isfinite(self.theta).all():
            raise ArithmeticError("Nonfinite model state")
        return prediction

    def snapshot(self):
        return dict(theta=self.theta.tolist(), mode=self.mode, rate=self.rate, rcond=self.rcond,
                    x=self.x.tolist(), y=self.y.tolist(), reference=self.reference.tolist(),
                    basis=self.basis.tolist(), costs=dict(self.costs), trace=self.trace.copy(), steps=self.steps)

    @classmethod
    def restore(cls, state):
        obj = cls.__new__(cls)
        for key in ("theta", "x", "y", "reference"):
            setattr(obj, key, np.array(state[key], dtype=np.float64))
        obj.basis = np.array(state["basis"], dtype=np.float64).reshape(-1, PARAMS)
        for key in ("mode", "rate", "rcond", "steps"):
            setattr(obj, key, state[key])
        obj.costs, obj.trace = Counter(state["costs"]), state["trace"].copy()
        return obj


def risks(theta, world):
    return {region + "_mse": mse(theta, x, y) for region, (x, y) in world["tests"].items()}


def run_cell(world, seed, n, rate, mode, rcond):
    learner = Learner(world["theta"], mode, rate, world["old_x"][:n], world["old_y"][:n], rcond)
    cumulative = 0.
    checkpoints = []
    for t, (x, y) in enumerate(zip(world["new_x"], world["new_y"]), 1):
        prediction = learner.step(x, y)
        cumulative += float(np.mean((prediction - y)**2))
        if t in (1, 16, 64, STEPS):
            checkpoints.append({"step": t, **risks(learner.theta, world)})
    after = risks(learner.theta, world)
    drift = float(np.mean((forward(learner.theta, learner.x) - learner.reference)**2))
    return {"seed": seed, "anchors": n, "rate": rate, "mode": mode, "rcond": rcond,
            "prequential_mse": cumulative / STEPS, **after, "anchor_drift_mse": drift,
            "anchor_label_mse": mse(learner.theta, learner.x, learner.y),
            "mean_raw_gradient_norm": learner.trace["raw_gradient_norm_sum"] / STEPS,
            "mean_update_to_fresh_gradient_norm_ratio": learner.trace["projected_ratio_sum"] / STEPS,
            "mean_projection_rank": learner.trace["rank_sum"] / STEPS,
            "max_linearized_anchor_change": learner.trace["max_linearized_anchor_change"],
            "max_anchor_drift_mse": learner.trace["max_anchor_drift_mse"],
            "costs": dict(learner.costs), "stored_model_scalars": PARAMS,
            "stored_anchor_scalars": 4*n,
            "stored_basis_scalars": int(learner.basis.size),
            "checkpoints": checkpoints}, learner


def continuation_probe(learner, world):
    kept = Learner.restore(learner.snapshot())
    released = Learner.restore(learner.snapshot())
    released.mode = "online"
    initial_difference = float(np.max(np.abs(forward(kept.theta, world["continuation_x"][0]) -
                                               forward(released.theta, world["continuation_x"][0]))))
    pre = {"kept": 0., "released": 0.}
    for x, y in zip(world["continuation_x"], world["continuation_y"]):
        for name, obj in (("kept", kept), ("released", released)):
            pre[name] += float(np.mean((obj.step(x, y) - y)**2)) / 32
    return {"initial_max_prediction_difference": initial_difference,
            "kept": {"prequential_mse": pre["kept"], **risks(kept.theta, world)},
            "released": {"prequential_mse": pre["released"], **risks(released.theta, world)},
            "meaning": "Identical initial weights; removing acquired constraints changes the allowed update, not the evidence. It may sacrifice old behavior."}


def summary(values):
    values = list(values)
    mean = statistics.fmean(values)
    sem = statistics.stdev(values) / len(values)**.5 if len(values) > 1 else 0.
    return {"mean": mean, "min": min(values), "max": max(values),
            "seed_mean_normal_ci95": [mean - 1.96*sem, mean + 1.96*sem]}


def aggregate(rows):
    groups = []
    keys = sorted({(r["rcond"], r["anchors"], r["rate"], r["mode"]) for r in rows})
    metrics = ("prequential_mse", "old_mse", "new_mse", "bridge_mse", "anchor_drift_mse",
               "anchor_label_mse", "mean_raw_gradient_norm", "mean_update_to_fresh_gradient_norm_ratio",
               "mean_projection_rank", "max_anchor_drift_mse")
    for rc, n, rate, mode in keys:
        subset = [r for r in rows if (r["rcond"], r["anchors"], r["rate"], r["mode"]) == (rc, n, rate, mode)]
        cost_names = sorted(set().union(*(r["costs"].keys() for r in subset)))
        groups.append({"rcond": rc, "anchors": n, "rate": rate, "mode": mode, "lives": len(subset),
                       "metrics": {k: summary(r[k] for r in subset) for k in metrics},
                       "mean_costs": {k: statistics.fmean(r["costs"].get(k, 0) for r in subset) for k in cost_names}})
    return groups


def exact_witnesses():
    h = Fraction(1, 2)
    regular = (1+h)*(1-h)
    assert regular == Fraction(3, 4)
    repaired = (1+h) * (1/(1+h))
    assert repaired == 1
    combined = h*h
    assert (combined - h*h) == 0
    return {"regular_tangent": {"initial": ["1", "1"], "delta": [str(h), str(-h)],
                               "jacobian_times_delta": "0", "prediction_after": str(regular),
                               "half_squared_error": str((regular-1)**2/2),
                               "curved_path_prediction": str(repaired)},
            "compatible_but_interacting": {"old_output": "a*b+c", "initial": [0, 0, 0],
                                           "separate_old_losses": [0, 0], "combined_half_squared_loss": str(combined**2/2),
                                           "joint_zero_loss_parameters": [str(h), str(h), str(-h*h)]},
            "loss_gradient_at_perfect_fit": [0, 0],
            "scope": "Exact counterexamples to unqualified finite-step claims, not refutations of the original papers' stated assumptions."}


def run():
    rows, starts, continuation = [], [], []
    for seed in SEEDS:
        world = make_world(seed)
        initial = {"seed": seed, **risks(world["theta"], world), "anchor_diagnostics": []}
        # Teacher is exactly in the student's model class; no optimization claim.
        assert all(mse(world["teacher"], x, y) == 0. for x, y in world["tests"].values())
        for n in ANCHORS:
            jac = values_jacobian(world["theta"], world["old_x"][:n])[1]
            _, singular = row_basis(jac, 1e-8)
            initial["anchor_diagnostics"].append({"anchors": n, "singular_values": singular.tolist(),
                                                 "rank_1e8": len(row_basis(jac, 1e-8)[0]),
                                                 "rank_1e6": len(row_basis(jac, 1e-6)[0])})
            for rate in RATES:
                for rcond, modes in ((1e-8, MODES), (1e-6, TANGENT)):
                    for mode in modes:
                        row, learner = run_cell(world, seed, n, rate, mode, rcond)
                        rows.append(row)
                        if n == 64 and rate == .12 and rcond == 1e-8 and mode in ("tangent_current", "tangent_guard", "replay_guard"):
                            continuation.append({"seed": seed, "mode": mode, **continuation_probe(learner, world)})
        starts.append(initial)
        print(f"seed {seed}: complete", file=sys.stderr, flush=True)
    return {"experiment": "learned_representation_retention_acquisition_geometry", "date": "2026-09-21",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "dtype": "float64"},
            "protocol": {"seeds": list(SEEDS), "anchors": list(ANCHORS), "rates": list(RATES), "steps": STEPS,
                         "batch": BATCH, "parameters": PARAMS, "fixed_reference_drift_budget": DRIFT_BUDGET,
                         "warmup_full_batch_steps": 600, "warmup_examples": 128, "warmup_rate": .05},
            "exact_witnesses": exact_witnesses(), "initial_states": starts, "runs": rows,
            "aggregate": aggregate(rows), "continuation_probes": continuation,
            "limits": ["A known old-data cut and a fixed finite network family are supplied",
                       "Stationary noiseless synthetic target; no label corruption or active data collection",
                       "Output preservation can preserve warmup errors; it is not old-risk nonincrease",
                       "Equal samples, not equal total computation; counters are not FLOPs",
                       "Fresh holdouts do not certify general open-world behavior",
                       "The teacher is an unlearned capacity witness, not an online competitor",
                       "No claim of a new universal computational principle or algorithm"]}


def compare(expected, actual, path="root"):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), path
        for k in expected: compare(expected[k], actual[k], path + "." + k)
    elif isinstance(expected, list):
        assert len(expected) == len(actual), path
        for i, (x, y) in enumerate(zip(expected, actual)): compare(x, y, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert np.isclose(expected, actual, atol=1e-11, rtol=1e-9), (path, expected, actual)
    else:
        assert expected == actual, (path, expected, actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--resume-input", type=Path)
    parser.add_argument("--resume-output", type=Path)
    args = parser.parse_args()
    if args.resume_input:
        payload = json.loads(args.resume_input.read_text(encoding="utf-8"))
        learner = Learner.restore(payload["state"])
        predictions = [learner.step(np.array(x), np.array(y)).tolist() for x, y in payload["batches"]]
        args.resume_output.write_text(json.dumps({"predictions": predictions, "state": learner.snapshot()}), encoding="utf-8")
        return
    result = run()
    path = HERE / "results/geometry.json"
    if args.write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    else:
        compare(json.loads(path.read_text(encoding="utf-8")), result)
    print(json.dumps({"status": "passed", "mode": "record" if args.write else "check",
                      "runs": len(result["runs"]), "groups": len(result["aggregate"]),
                      "continuation_probes": len(result["continuation_probes"])}))


if __name__ == "__main__":
    main()
