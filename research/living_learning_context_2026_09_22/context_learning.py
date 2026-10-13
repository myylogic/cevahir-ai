"""Round 14: observation-derived gates versus confounded surface cues.

Only input and observed label enter ordinary learners. Full function language
and four-bit history window are supplied. Oracle control is explicitly separate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
PROTOCOL = ROOT / "docs/research/living_learning_context_2026_09_22/PROTOCOL.md"
SEEDS = list(range(24))
REGIMES = ("aligned", "balanced", "reversed")
METHODS = ("full_table", "pooled_x", "surface_only", "history_only",
           "joint_uniform_gate", "joint_balanced_family", "oracle_gate")
CHECKPOINTS = (0, 12, 24, 48, 49, 52, 56, 64)
EPSILON = .05
INPUTS = [(h, s, x) for h in range(16) for s in range(2) for x in range(16)]
EXPERTS = [(mask, bias) for mask in range(16) for bias in range(2)]
EXPERT_OUTPUTS = np.array([[(mask & x).bit_count() % 2 ^ bias for x in range(16)]
                          for mask, bias in EXPERTS], dtype=np.uint8)
GATE_OUTPUTS = np.array([[(mask & h).bit_count() % 2 for h, s, x in INPUTS]
                         for mask in range(1, 16)] + [[s for h, s, x in INPUTS]], dtype=np.uint8)
X_IDS = np.array([x for _, _, x in INPUTS], dtype=np.int64)
FUNCTIONS = np.concatenate([np.where(gate[None, None, :],
                                     EXPERT_OUTPUTS[None, :, X_IDS],
                                     EXPERT_OUTPUTS[:, None, X_IDS]).reshape(1024, 512)
                            for gate in GATE_OUTPUTS], axis=0).astype(np.float64)
ALL_IDS = np.arange(16*1024)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def input_id(h, s, x):
    return (h*2+s)*16+x


def metrics(p, y):
    p, y = np.asarray(p), np.asarray(y)
    if not len(y):
        return None
    ties = np.isclose(p, .5, atol=1e-12, rtol=0)
    accuracy = np.where(ties, .5, (p > .5) == y)
    return dict(n=int(len(y)), mse=float(np.mean((p-y)**2)), accuracy=float(np.mean(accuracy)),
                confidently_wrong=int(np.sum(((p <= .01) & (y == 1)) | ((p >= .99) & (y == 0)))))


class Learner:
    def __init__(self, method):
        if method not in METHODS or method == "oracle_gate":
            raise ValueError("ordinary constructor cannot receive true gate")
        self.method = method
        ids = {"surface_only": np.arange(15*1024, 16*1024),
               "history_only": np.arange(15*1024),
               "joint_uniform_gate": ALL_IDS,
               "joint_balanced_family": ALL_IDS}.get(method, np.array([], dtype=np.int64))
        self._initialize(ids)

    def _initialize(self, ids):
        self.candidate_ids = np.array(ids, dtype=np.int64)
        self.errors = np.zeros(len(ids), dtype=np.int64)
        self.steps = 0
        self.candidate_label_checks = 0
        self.counts = np.zeros((512 if self.method == "full_table" else 16, 2), dtype=np.int64) if not len(ids) else np.zeros((0, 2), dtype=np.int64)
        if len(ids):
            self.prior = np.full(len(ids), 1/len(ids))
            if self.method == "joint_balanced_family":
                self.prior[:] = .5/(15*1024)
                self.prior[-1024:] = .5/1024
        else:
            self.prior = np.array([], dtype=float)

    def weights(self):
        logs = np.log(self.prior) + self.errors * np.log(EPSILON/(1-EPSILON))
        weights = np.exp(logs-logs.max())
        return weights/weights.sum()

    def probabilities(self, indices):
        indices = np.asarray(indices, dtype=int)
        if len(self.candidate_ids):
            # Slice preserves shared bank for contiguous candidates; no hidden labels.
            bank = FUNCTIONS[self.candidate_ids[0]:self.candidate_ids[-1]+1]
            assert len(bank) == len(self.candidate_ids)
            if np.array_equal(indices, np.arange(512)):
                return self.weights() @ bank
            return self.weights() @ bank[:, indices]
        keys = indices if self.method == "full_table" else X_IDS[indices]
        count = self.counts[keys]
        total = count.sum(axis=1)
        return np.divide(count[:, 1], total, out=np.full(len(keys), .5), where=total > 0)

    def observe(self, index, label):
        if not 0 <= index < 512 or label not in (0, 1):
            raise ValueError("invalid observed input or label")
        self.steps += 1
        if len(self.candidate_ids):
            self.errors += FUNCTIONS[self.candidate_ids, index] != label
            self.candidate_label_checks += len(self.candidate_ids)
        else:
            key = index if self.method == "full_table" else X_IDS[index]
            self.counts[key, label] += 1

    def posterior(self):
        if not len(self.candidate_ids):
            return None
        weights = self.weights()
        return dict(gate_mass=np.bincount(self.candidate_ids//1024, weights=weights, minlength=16).tolist(),
                    effective_candidates=float(1/np.sum(weights**2)), minimum_errors=int(self.errors.min()))

    def state(self):
        state = copy.deepcopy(self.__dict__)
        for key in ("candidate_ids", "errors", "counts", "prior"):
            state[key] = state[key].tolist()
        return state

    @classmethod
    def restore(cls, state):
        result = cls.__new__(cls)
        result.__dict__.update(copy.deepcopy(state))
        for key in ("candidate_ids", "errors", "counts"):
            setattr(result, key, np.array(state[key], dtype=np.int64))
        result.counts = result.counts.reshape(-1, 2)
        result.prior = np.array(state["prior"], dtype=float)
        return result

    def resources(self):
        arrays = (self.candidate_ids, self.errors, self.counts, self.prior)
        return dict(candidates=len(self.candidate_ids), candidate_label_checks=self.candidate_label_checks,
                    numeric_state_bytes=sum(a.nbytes for a in arrays),
                    state_json_bytes=len(encode(self.state()).encode()),
                    shared_supplied_function_bank_bytes=int(FUNCTIONS.nbytes), raw_event_rows=0)


class OracleLearner(Learner):
    """Privileged true-gate control, never instantiated for an ordinary method."""
    def __init__(self, true_gate_index):
        if not 0 <= true_gate_index < 15:
            raise ValueError("oracle requires a supplied history gate")
        self.method = "oracle_gate"
        self._initialize(np.arange(true_gate_index*1024, (true_gate_index+1)*1024))


def make_world(seed, regime):
    if regime not in REGIMES:
        raise ValueError(regime)
    rng = random.Random(703_019+seed*107_021)
    mask = rng.choice([m for m in range(1, 16) if m.bit_count() in (2, 3)])
    expert_masks = rng.sample([m for m in range(1, 16) if m.bit_count() in (1, 2)], 2)
    f0, f1 = [EXPERTS.index((m, rng.randrange(2))) for m in expert_masks]
    true_id = (mask-1)*1024+f0*32+f1
    surface_id = 15*1024+f0*32+f1
    aligned = [i for i, (h, s, x) in enumerate(INPUTS) if s == ((mask & h).bit_count() % 2)]
    reversed_ids = [i for i in range(512) if i not in set(aligned)]
    prefix = rng.sample(aligned, 48)
    remaining_aligned = [i for i in aligned if i not in set(prefix)]
    suffix_aligned, suffix_reversed = rng.sample(remaining_aligned, 16), rng.sample(reversed_ids, 16)
    if regime == "aligned":
        suffix = suffix_aligned
    elif regime == "reversed":
        suffix = suffix_reversed
    else:
        suffix = suffix_aligned[:8]+suffix_reversed[:8]
        rng.shuffle(suffix)
    truth = FUNCTIONS[true_id].astype(int).tolist()
    return dict(seed=seed, regime=regime, true_gate_index=mask-1, expert_ids=[f0, f1],
                true_candidate_id=true_id, matched_surface_candidate_id=surface_id,
                truth=truth, aligned_ids=aligned, reversed_ids=reversed_ids,
                stream=[[i, truth[i]] for i in prefix+suffix])


def make_learner(method, world):
    return OracleLearner(world["true_gate_index"]) if method == "oracle_gate" else Learner(method)


def evaluate(learner, world, observed):
    p = learner.probabilities(np.arange(512))
    truth = np.asarray(world["truth"])
    unseen = [i for i in range(512) if i not in observed]
    posterior = learner.posterior()
    return dict(t=learner.steps, full=metrics(p, truth),
                aligned=metrics(p[world["aligned_ids"]], truth[world["aligned_ids"]]),
                reversed=metrics(p[world["reversed_ids"]], truth[world["reversed_ids"]]),
                unseen=metrics(p[unseen], truth[unseen]), posterior=posterior,
                true_gate_mass=None if posterior is None else posterior["gate_mass"][world["true_gate_index"]],
                resources=learner.resources())


def run_one(world, method):
    learner = make_learner(method, world)
    observed = set()
    snapshots = [evaluate(learner, world, observed)]
    prequential = []
    for index, label in world["stream"]:
        p = float(learner.probabilities([index])[0])
        prequential.append(dict(t=learner.steps, index=index, label=label, probability=p, mse=(p-label)**2))
        learner.observe(index, label)
        observed.add(index)
        if learner.steps in CHECKPOINTS:
            snapshots.append(evaluate(learner, world, observed))
    return dict(seed=world["seed"], regime=world["regime"], method=method,
                snapshots=snapshots, prequential=prequential,
                prefix_mse=float(np.mean([p["mse"] for p in prequential[:48]])),
                suffix_mse=float(np.mean([p["mse"] for p in prequential[48:]])),
                final_state_sha256=hashlib.sha256(encode(learner.state()).encode()).hexdigest())


def run_all():
    worlds = [make_world(seed, regime) for seed in SEEDS for regime in REGIMES]
    runs = [run_one(world, method) for world in worlds for method in METHODS]
    return dict(date="2026-09-22", source_sha256=sha(__file__), protocol_sha256=sha(PROTOCOL),
                parameters=dict(epsilon=EPSILON, checkpoints=CHECKPOINTS, lives=24,
                                candidates=16384, input_count=512, shared_bank_bytes=int(FUNCTIONS.nbytes)),
                worlds=worlds, runs=runs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    data = run_all()
    path = RESULTS / "context_learning.json"
    if args.write:
        path.write_text(encode(data)+"\n", encoding="utf-8")
        print(json.dumps(dict(status="written", trajectories=len(data["runs"]), bytes=path.stat().st_size)))
    else:
        assert encode(data) == encode(json.loads(path.read_text(encoding="utf-8")))
        print(json.dumps(dict(status="exact_rerun_passed", trajectories=len(data["runs"]))))


if __name__ == "__main__":
    main()
