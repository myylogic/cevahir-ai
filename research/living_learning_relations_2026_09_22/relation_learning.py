"""Round 12: observed relation classes, independent target roles, composition.

Only observations enter Learner. World truth is confined to the evaluator.
Python + NumPy; no fitting against evaluation queries. --write is explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
from collections import defaultdict

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
PROTOCOL = ROOT / "docs/research/living_learning_relations_2026_09_22/PROTOCOL.md"
N = 6
SEEDS = list(range(24))
REGIMES = ("compatible", "cycle_shift", "local_exception")
METHODS = ("table", "permutation_prior", "source_a", "source_b",
           "source_both", "unrelated_sources", "frozen_source_both", "cycle_catalog")
PERMUTATIONS = tuple(itertools.permutations(range(N)))
CANONICAL = ((1, 2, 3, 4, 5, 0), (1, 0, 3, 2, 5, 4),
             (1, 2, 0, 4, 5, 3), (1, 2, 3, 4, 0, 5))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def conjugate(function, rename):
    result = [None] * N
    for old in range(N):
        result[rename[old]] = int(rename[function[old]])
    return tuple(result)


def learn_source(observations):
    table = [None] * N
    for x, y in observations:
        if not (0 <= x < N and 0 <= y < N):
            raise ValueError("source observation outside supplied universe")
        if table[x] is not None and table[x] != y:
            raise ValueError("non-deterministic source observations")
        table[x] = y
    if None in table:
        raise ValueError("incomplete source; no teacher completion allowed")
    return tuple(table)


def observed_class(table):
    return sorted({conjugate(table, rename) for rename in PERMUTATIONS})


def cycle_type(function):
    seen = set()
    lengths = []
    for start in range(N):
        if start in seen:
            continue
        x, length = start, 0
        while x not in seen:
            seen.add(x)
            length += 1
            x = function[x]
        lengths.append(length)
    return tuple(sorted(lengths))


def catalog():
    groups = defaultdict(list)
    for function in PERMUTATIONS:
        groups[cycle_type(function)].append(function)
    return [groups[key] for key in sorted(groups)]


class Learner:
    """Deterministic finite learner; constructor has no truth/regime/seed input."""

    def __init__(self, method, source_observations):
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        selected = {"source_a": (0,), "source_b": (1,),
                    "source_both": (0, 1), "frozen_source_both": (0, 1),
                    "unrelated_sources": (2, 3)}.get(method, ())
        self.source_tables = [list(learn_source(source_observations[i])) for i in selected]
        groups = [observed_class(table) for table in self.source_tables]
        if method == "permutation_prior":
            groups = [list(PERMUTATIONS)]
        elif method == "cycle_catalog":
            groups = catalog()
        self.group_sizes = [len(g) for g in groups]
        self.candidates = np.array([f for g in groups for f in g], dtype=np.int64).reshape(-1, N)
        self.group_ids = np.array([i for i, g in enumerate(groups) for _ in g], dtype=np.int64)
        self.prior_weights = np.array([1 / (len(groups) * len(g)) for g in groups for _ in g], dtype=float)
        self.active = np.ones((2, len(self.candidates)), dtype=bool)
        self.fallback = [method == "table"] * 2
        self.fallback_at = [0 if method == "table" else None] * 2
        self.observed = np.full((2, N), -1, dtype=np.int64)
        self.steps = 0
        self.filter_checks = 0
        self.eliminated = 0
        self.rejected = 0
        self.source_observation_count = sum(len(source_observations[i]) for i in selected)
        self.rename_attempts = len(selected) * len(PERMUTATIONS)

    def probabilities(self):
        result = np.zeros((2, N, N), dtype=float)
        for action in range(2):
            if self.fallback[action]:
                for x, y in enumerate(self.observed[action]):
                    if y < 0:
                        result[action, x] = 1 / N
                    else:
                        result[action, x, y] = 1
                continue
            mask = self.active[action]
            weights = self.prior_weights[mask]
            weights = weights / weights.sum()
            candidates = self.candidates[mask]
            for x in range(N):
                result[action, x] = np.bincount(candidates[:, x], weights=weights, minlength=N)
        return result

    def update(self, action, x, y):
        if not (0 <= action < 2 and 0 <= x < N and 0 <= y < N):
            raise ValueError("invalid target observation")
        if self.observed[action, x] not in (-1, y):
            raise ValueError("repeated conflicting target fact outside protocol")
        self.steps += 1
        self.observed[action, x] = y
        if self.fallback[action]:
            return
        old = self.active[action].copy()
        self.filter_checks += int(old.sum())
        new = old & (self.candidates[:, x] == y)
        if new.any():
            self.active[action] = new
            self.eliminated += int(old.sum() - new.sum())
        elif self.method == "frozen_source_both":
            self.rejected += 1
        else:
            self.active[action] = new
            self.eliminated += int(old.sum())
            self.fallback[action] = True
            self.fallback_at[action] = self.steps

    def posterior_summary(self):
        groups = []
        for action in range(2):
            if self.fallback[action]:
                groups.append(None)
            else:
                mask = self.active[action]
                weights = self.prior_weights[mask]
                groups.append((np.bincount(self.group_ids[mask], weights=weights,
                                          minlength=len(self.group_sizes)) / weights.sum()).tolist())
        return dict(candidate_counts=[0 if self.fallback[a] else int(self.active[a].sum()) for a in range(2)],
                    class_probabilities=groups, fallback=list(self.fallback),
                    fallback_at=list(self.fallback_at), filter_checks=self.filter_checks,
                    eliminated=self.eliminated, rejected=self.rejected)

    def state(self):
        return dict(method=self.method, source_tables=self.source_tables,
                    group_sizes=self.group_sizes, candidates=self.candidates.tolist(),
                    group_ids=self.group_ids.tolist(), prior_weights=self.prior_weights.tolist(),
                    active=self.active.tolist(), fallback=self.fallback, fallback_at=self.fallback_at,
                    observed=self.observed.tolist(), steps=self.steps, filter_checks=self.filter_checks,
                    eliminated=self.eliminated, rejected=self.rejected,
                    source_observation_count=self.source_observation_count,
                    rename_attempts=self.rename_attempts)

    @classmethod
    def restore(cls, state):
        learner = cls.__new__(cls)
        learner.__dict__.update(state)
        for key in ("candidates", "group_ids", "observed"):
            setattr(learner, key, np.array(state[key], dtype=np.int64))
        learner.candidates = learner.candidates.reshape(-1, N)
        learner.prior_weights = np.array(state["prior_weights"], dtype=float)
        learner.active = np.array(state["active"], dtype=bool).reshape(2, -1)
        return learner

    def resources(self):
        arrays = (self.candidates, self.group_ids, self.prior_weights, self.active, self.observed)
        return dict(source_observations=self.source_observation_count,
                    rename_attempts=self.rename_attempts, candidate_tables=len(self.candidates),
                    candidate_table_cells=int(self.candidates.size),
                    array_bytes=sum(array.nbytes for array in arrays),
                    serialized_state_bytes=len(canonical_json(self.state()).encode("utf-8")),
                    filter_candidate_cell_checks=self.filter_checks)


def shuffled(rng):
    values = list(range(N))
    rng.shuffle(values)
    return values


def make_sources(seed):
    observations = []
    for i, function in enumerate(CANONICAL):
        rng = random.Random(1_000_003 * seed + 73_001 * (i + 1))
        actual = conjugate(function, shuffled(rng))
        stream = [(x, actual[x]) for x in range(N)] * 2
        rng.shuffle(stream)
        observations.append(stream)
    return observations


def make_world(seed, regime):
    rng = random.Random(91_919 + 3_007 * seed)
    rename_a, rename_b = shuffled(rng), shuffled(rng)
    a = conjugate(CANONICAL[0], rename_a)
    b = conjugate(CANONICAL[1], rename_b)
    if regime == "cycle_shift":
        b = conjugate(CANONICAL[2], rename_b)
    elif regime not in REGIMES:
        raise ValueError(regime)
    exception_x = rng.randrange(N)
    exception_y = rng.choice([x for x in range(N) if x != b[exception_x]])
    if regime == "local_exception":
        b = list(b)
        b[exception_x] = exception_y
        b = tuple(b)
    action_order = [0, 1]
    rng.shuffle(action_order)
    truth = [list((a, b)[i]) for i in action_order]
    cells = list(itertools.product(range(2), range(N)))
    rng.shuffle(cells)
    stream = [(action, x, truth[action][x]) for action, x in cells]
    return dict(seed=seed, regime=regime, sources=make_sources(seed), truth=truth,
                stream=stream, changed_action=action_order.index(1),
                exception_cell=[action_order.index(1), exception_x] if regime == "local_exception" else None)


def score(probabilities, labels):
    p = np.asarray(probabilities, dtype=float).reshape(-1, N)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if len(y) == 0:
        return None
    target = np.eye(N)[y]
    maxima = p.max(axis=1)
    ties = np.isclose(p, maxima[:, None], rtol=0, atol=1e-12)
    correct_tie = ties[np.arange(len(y)), y]
    accuracy = correct_tie / ties.sum(axis=1)
    confident_wrong = (maxima >= 1 - 1e-12) & (p[np.arange(len(y)), y] < 1e-12)
    return dict(queries=len(y), brier=float(np.mean(np.sum((p - target) ** 2, axis=1))),
                tie_accuracy=float(accuracy.mean()), confident_wrong=int(confident_wrong.sum()))


def evaluate(learner, truth):
    truth = np.asarray(truth, dtype=int)
    p = learner.probabilities()
    composition = np.stack((p[0] @ p[1], p[1] @ p[0]))
    composed_truth = np.stack((truth[1, truth[0]], truth[0, truth[1]]))
    unseen = learner.observed < 0
    return dict(t=learner.steps, direct=score(p, truth), unseen=score(p[unseen], truth[unseen]),
                composition=score(composition, composed_truth),
                posterior=learner.posterior_summary(), resources=learner.resources())


def run_trajectory(world, method):
    learner = Learner(method, world["sources"])
    snapshots = [evaluate(learner, world["truth"])]
    prequential = []
    for action, x, y in world["stream"]:
        probabilities = learner.probabilities()[action, x].tolist()
        prequential.append(dict(t=learner.steps, action=action, x=x, y=y,
                                probabilities=probabilities, score=score([probabilities], [y])))
        learner.update(action, x, y)
        snapshots.append(evaluate(learner, world["truth"]))
    return dict(seed=world["seed"], regime=world["regime"], method=method,
                snapshots=snapshots, prequential=prequential,
                prequential_brier=float(np.mean([row["score"]["brier"] for row in prequential])),
                prequential_confident_wrong=sum(row["score"]["confident_wrong"] for row in prequential),
                final_state_sha256=hashlib.sha256(canonical_json(learner.state()).encode()).hexdigest())


def run_all():
    worlds = [make_world(seed, regime) for seed in SEEDS for regime in REGIMES]
    runs = [run_trajectory(world, method) for world in worlds for method in METHODS]
    return dict(date="2026-09-22", source_sha256=sha(__file__), protocol_sha256=sha(PROTOCOL),
                parameters=dict(n=N, seeds=SEEDS, regimes=REGIMES, methods=METHODS,
                                checkpoint_count=13, trajectories=len(runs)),
                worlds=worlds, runs=runs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = run_all()
    path = RESULTS / "relation_learning.json"
    encoded = canonical_json(result)
    if args.write:
        path.write_text(encoded + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(dict(status="written", trajectories=len(result["runs"]), bytes=path.stat().st_size)))
    else:
        existing = json.loads(path.read_text(encoding="utf-8"))
        assert canonical_json(existing) == encoded, "full deterministic rerun differs"
        print(json.dumps(dict(status="exact_rerun_passed", trajectories=len(result["runs"]))))


if __name__ == "__main__":
    main()
