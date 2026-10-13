"""Round 13: finite two-stage consolidation versus equal-evidence online updates.

The function language is supplied, not discovered. No neurons or sleep modeled.
Only observed (context, input, label) triples enter the learner.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
PROTOCOL = ROOT / "docs/research/living_learning_consolidation_2026_09_22/PROTOCOL.md"
D = 6
X = np.array(list(itertools.product((0, 1), repeat=D)), dtype=np.int64)
PROGRAMS = [(tuple(s), bias) for degree in range(4)
            for s in itertools.combinations(range(D), degree) for bias in (0, 1)]
OUTPUTS = np.array([((X[:, subset].sum(axis=1) + bias) % 2)
                    if subset else np.full(len(X), bias) for subset, bias in PROGRAMS], dtype=np.int64)
LINEAR_IDS = list(range(14))
RICH_IDS = list(range(84))
METHODS = ("frozen_core", "cache_only", "sleep_linear_replay", "sleep_rich_replay",
           "online_rich", "sleep_rich_recent", "sleep_self_replay", "sleep_mispaired_replay")
REGIMES = ("visible_context", "hidden_context", "noisy_day")
SEEDS = list(range(24))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def score(probability, truth):
    p, y = np.asarray(probability, dtype=float), np.asarray(truth, dtype=int)
    assert p.shape == y.shape and p.size > 0
    tie = np.isclose(p, .5, atol=1e-12, rtol=0)
    accuracy = np.where(tie, .5, (p > .5) == y)
    return dict(n=int(p.size), mse=float(np.mean((p-y)**2)), accuracy=float(np.mean(accuracy)),
                confident_wrong=int(np.sum(((p < 1e-12) & (y == 1)) | ((p > 1-1e-12) & (y == 0)))))


class Learner:
    def __init__(self, method, old_rows):
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        self.old_rows = [list(row) for row in old_rows]
        self.day_rows = []
        errors = sum((OUTPUTS[LINEAR_IDS, x] != y).astype(np.int64) for _, x, y in old_rows)
        best = [LINEAR_IDS[i] for i in np.flatnonzero(errors == errors.min())]
        if len(best) != 1 or int(errors.min()) != 0:
            raise ValueError("source observations do not uniquely determine source in supplied base language")
        self.initial_id = best[0]
        self.core_ids = [best.copy(), best.copy()]
        self.bank = LINEAR_IDS if method == "sleep_linear_replay" else RICH_IDS
        self.errors = np.zeros((2, len(self.bank)), dtype=np.int64)
        self.evidence_counts = [0, 0]
        self.common_source_checks = len(old_rows) * len(LINEAR_IDS)
        self.learning_checks = 0
        self.selection_calls = 0
        self.synthetic_labels = 0
        self.steps = 0
        self.consolidated = False
        if method == "online_rich":
            self._add(old_rows)
            self._select()

    def _add(self, rows):
        for context, x, y in rows:
            if context not in (0, 1) or not 0 <= x < 64 or y not in (0, 1):
                raise ValueError("invalid observed row")
            self.errors[context] += OUTPUTS[self.bank, x] != y
            self.evidence_counts[context] += 1
            self.learning_checks += len(self.bank)

    def _select(self):
        for context in (0, 1):
            if self.evidence_counts[context]:
                counts = self.errors[context]
                self.core_ids[context] = [self.bank[i] for i in np.flatnonzero(counts == counts.min())]
                self.selection_calls += 1

    def predict(self, context, xs, use_cache=True):
        xs = np.asarray(xs, dtype=int)
        result = OUTPUTS[self.core_ids[context]][:, xs].mean(axis=0)
        if use_cache and self.method != "frozen_core":
            cache = {(c, x): y for c, x, y in self.day_rows}
            result = np.array([cache.get((context, int(x)), float(p)) for x, p in zip(xs, result)])
        return result

    def observe(self, context, x, y):
        self.steps += 1
        if self.method != "frozen_core":
            self.day_rows.append([int(context), int(x), int(y)])
        if self.method == "online_rich":
            self._add([[context, x, y]])
            self._select()

    def consolidate(self):
        if self.consolidated:
            raise ValueError("this protocol has one consolidation boundary")
        self.consolidated = True
        if self.method in ("frozen_core", "cache_only", "online_rich"):
            return
        rows = [row.copy() for row in self.day_rows]
        if self.method == "sleep_self_replay":
            rows = [[c, x, int(OUTPUTS[self.initial_id, x])] for c, x, _ in rows]
            self.synthetic_labels += len(rows)
        elif self.method == "sleep_mispaired_replay":
            labels = [row[2] for row in rows]
            rows = [[c, x, labels[(i-1) % len(labels)]] for i, (c, x, _) in enumerate(rows)]
        if self.method != "sleep_rich_recent":
            self._add(self.old_rows)
        self._add(rows)
        self._select()

    def purge_episodes(self):
        self.old_rows = []
        self.day_rows = []

    def continue_observed(self, context, x, y):
        """Append future evidence to retained statistics without raw episode replay.

        All methods get this declared diagnostic continuation operation; it is
        not a second benchmark or an assertion that frozen/cache learned earlier.
        """
        self._add([[context, x, y]])
        self._select()

    def state(self):
        state = copy.deepcopy(self.__dict__)
        state["errors"] = self.errors.tolist()
        return state

    @classmethod
    def restore(cls, state):
        result = cls.__new__(cls)
        result.__dict__.update(copy.deepcopy(state))
        result.errors = np.asarray(state["errors"], dtype=np.int64)
        return result

    def resources(self):
        return dict(common_source_checks=self.common_source_checks,
                    learning_checks=self.learning_checks, selection_calls=self.selection_calls,
                    raw_episode_rows=len(self.old_rows)+len(self.day_rows),
                    statistic_integer_cells=int(self.errors.size),
                    statistic_array_bytes=int(self.errors.nbytes),
                    active_candidate_counts=[len(ids) for ids in self.core_ids],
                    synthetic_labels=self.synthetic_labels,
                    state_json_bytes=len(encode(self.state()).encode("utf-8")))


def make_world(seed, regime):
    if regime not in REGIMES:
        raise ValueError(regime)
    rng = random.Random(21_037 + 91_027 * seed)
    old_id = PROGRAMS.index(((rng.randrange(D),), rng.randrange(2)))
    degree = rng.choice((2, 3))
    new_subset = tuple(sorted(rng.sample(range(D), degree)))
    new_id = PROGRAMS.index((new_subset, rng.randrange(2)))
    basis = [0] + [1 << bit for bit in range(D)]
    source_xs = basis + rng.sample([i for i in range(64) if i not in basis], 17)
    rng.shuffle(source_xs)
    day_xs = rng.sample(range(64), 24)
    corrupt_positions = sorted(rng.sample(range(24), 4))
    new_context = 0 if regime == "hidden_context" else 1
    old_rows = [[0, x, int(OUTPUTS[old_id, x])] for x in source_xs]
    day_rows = [[new_context, x, int(OUTPUTS[new_id, x]) ^
                 int(regime == "noisy_day" and i in corrupt_positions)] for i, x in enumerate(day_xs)]
    future_xs = rng.sample([i for i in range(64) if i not in day_xs], 6)
    return dict(seed=seed, regime=regime, old_teacher_id=old_id, new_teacher_id=new_id,
                old_truth=OUTPUTS[old_id].tolist(), new_truth=OUTPUTS[new_id].tolist(),
                old_rows=old_rows, day_rows=day_rows, new_observed_context=new_context,
                unseen_xs=[i for i in range(64) if i not in day_xs],
                corrupt_positions=corrupt_positions if regime == "noisy_day" else [],
                future_rows=[[new_context, x, int(OUTPUTS[new_id, x])] for x in future_xs])


def evaluate(learner, world, phase):
    result = dict(phase=phase, day_steps=learner.steps, resources=learner.resources())
    for cache, key in ((False, "core"), (True, "with_cache")):
        old_p = learner.predict(0, range(64), use_cache=cache)
        new_p = learner.predict(world["new_observed_context"], range(64), use_cache=cache)
        new_truth = np.asarray(world["new_truth"])
        seen = [row[1] for row in world["day_rows"]]
        unseen = world["unseen_xs"]
        old_score, new_score = score(old_p, world["old_truth"]), score(new_p, new_truth)
        result[key] = dict(old=old_score, new=new_score,
                           unseen=score(new_p[unseen], new_truth[unseen]),
                           seen=score(new_p[seen], new_truth[seen]),
                           balanced_mse=(old_score["mse"]+new_score["mse"])/2)
    return result


def run_one(world, method):
    learner = Learner(method, world["old_rows"])
    snapshots = [evaluate(learner, world, "day_0")]
    prequential = []
    for context, x, y in world["day_rows"]:
        p = float(learner.predict(context, [x])[0])
        true_y = world["new_truth"][x]
        prequential.append(dict(t=learner.steps, context=context, x=x, observed_y=y, true_y=true_y,
                                probability=p, observed_mse=(p-y)**2, true_mse=(p-true_y)**2))
        learner.observe(context, x, y)
        if learner.steps % 6 == 0:
            snapshots.append(evaluate(learner, world, f"day_{learner.steps}"))
    learner.consolidate()
    snapshots.append(evaluate(learner, world, "consolidated_with_episodes"))
    learner.purge_episodes()
    snapshots.append(evaluate(learner, world, "consolidated_no_episodes"))
    return dict(seed=world["seed"], regime=world["regime"], method=method,
                prequential=prequential, snapshots=snapshots, final_state=learner.state(),
                prequential_true_mse=float(np.mean([p["true_mse"] for p in prequential])),
                prequential_observed_mse=float(np.mean([p["observed_mse"] for p in prequential])))


def run_all():
    worlds = [make_world(seed, regime) for seed in SEEDS for regime in REGIMES]
    return dict(date="2026-09-22", source_sha256=sha(__file__), protocol_sha256=sha(PROTOCOL),
                fixed_language=dict(dimensions=D, linear_functions=14, rich_functions=84),
                worlds=worlds, runs=[run_one(w, method) for w in worlds for method in METHODS])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = run_all()
    path = RESULTS / "consolidation.json"
    if args.write:
        path.write_text(encode(result)+"\n", encoding="utf-8")
        print(json.dumps(dict(status="written", trajectories=len(result["runs"]), bytes=path.stat().st_size)))
    else:
        assert encode(result) == encode(json.loads(path.read_text(encoding="utf-8")))
        print(json.dumps(dict(status="exact_rerun_passed", trajectories=len(result["runs"]))))


if __name__ == "__main__":
    main()
