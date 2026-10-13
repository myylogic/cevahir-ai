"""Independent structural checks, process restarts, preservation, full rerun."""
import argparse
import inspect
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import numpy as np

import relation_learning as r


def preservation():
    baseline = json.loads((r.RESULTS / "preservation_baseline.json").read_text(encoding="utf-8"))
    changed = [name for name, digest in baseline["files"].items()
               if not (r.ROOT / name).exists() or r.sha(r.ROOT / name) != digest]
    assert not changed, changed
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=r.ROOT, text=True).strip()
    assert head == baseline["base_head"]
    return dict(files_unchanged=len(baseline["files"]), head_unchanged=head)


def cycle_partition_independent(function):
    remaining = set(range(r.N))
    sizes = []
    while remaining:
        origin = min(remaining)
        current = function[origin]
        cycle = {origin}
        while current != origin:
            assert current not in cycle
            cycle.add(current)
            current = function[current]
        remaining -= cycle
        sizes.append(len(cycle))
    return tuple(sorted(sizes))


def check_classes():
    sources = r.make_sources(13)
    sizes = []
    for observations in sources:
        table = r.learn_source(observations)
        kind = cycle_partition_independent(table)
        expected = {p for p in itertools.permutations(range(r.N)) if cycle_partition_independent(p) == kind}
        actual = set(r.observed_class(table))
        assert expected == actual
        sizes.append(len(actual))
        assert r.learn_source(list(reversed(observations))) == table
        assert r.learn_source(list(set(observations))) == table
        for missing in range(r.N):
            partial = [pair for pair in observations if pair[0] != missing]
            try:
                r.learn_source(partial)
            except ValueError:
                pass
            else:
                raise AssertionError("incomplete source was silently filled")
    assert sizes == [120, 15, 40, 144]
    groups = r.catalog()
    assert len(groups) == 11 and sum(map(len, groups)) == 720
    learner = r.Learner("source_both", sources)
    assert np.allclose(learner.posterior_summary()["class_probabilities"], [[.5, .5], [.5, .5]])
    return dict(source_class_sizes=sizes, catalog_types=11, source_missing_edge_rejections=24)


def check_equivariance():
    checked = 0
    rng = random.Random(58391)
    for regime in r.REGIMES:
        world = r.make_world(7, regime)
        rename = list(range(r.N))
        rng.shuffle(rename)
        swapped_sources = []
        for observations in world["sources"]:
            source_rename = list(range(r.N))
            rng.shuffle(source_rename)
            swapped_sources.append([(source_rename[x], source_rename[y]) for x, y in reversed(observations)])
        for method in r.METHODS:
            original = r.Learner(method, world["sources"])
            renamed = r.Learner(method, swapped_sources)
            for step in range(13):
                p, q = original.probabilities(), renamed.probabilities()
                for a, x, y in itertools.product(range(2), range(r.N), range(r.N)):
                    assert abs(p[a, x, y] - q[1-a, rename[x], rename[y]]) < 1e-12
                checked += 1
                if step < 12:
                    a, x, y = world["stream"][step]
                    original.update(a, x, y)
                    renamed.update(1-a, rename[x], rename[y])
    return dict(renamed_prediction_checkpoints=checked, tolerance=1e-12,
                renamed="independent sources + target objects + target actions")


def check_composition():
    # Direct enumeration of all pairs of functions; does not multiply mean matrices.
    learner = r.Learner("source_both", r.make_sources(9))
    selected0 = np.array([0, 31, 119, 124])
    selected1 = np.array([6, 28, 122])
    learner.active[:] = False
    learner.active[0, selected0] = True
    learner.active[1, selected1] = True
    p = learner.probabilities()
    total = np.zeros((2, r.N, r.N))
    w0 = learner.prior_weights[selected0]
    w1 = learner.prior_weights[selected1]
    w0 = w0 / w0.sum()
    w1 = w1 / w1.sum()
    for i, first in enumerate(selected0):
        f = learner.candidates[first]
        for j, second in enumerate(selected1):
            g = learner.candidates[second]
            for x in range(r.N):
                total[0, x, g[f[x]]] += w0[i] * w1[j]
                total[1, x, f[g[x]]] += w0[i] * w1[j]
    assert np.allclose(total, [p[0] @ p[1], p[1] @ p[0]], atol=1e-14, rtol=0)
    # Counterexample: one unknown function is identity or a swap with equal weight.
    identity = np.eye(2)
    swap = np.array([[0., 1.], [1., 0.]])
    expected_repeat = (identity @ identity + swap @ swap) / 2
    incorrect = ((identity + swap) / 2) @ ((identity + swap) / 2)
    assert np.allclose(expected_repeat, identity) and not np.allclose(expected_repeat, incorrect)
    # Symmetry prediction made by independent mathematical audit before results.
    priors = [r.Learner(method, r.make_sources(9)).probabilities()
              for method in ("source_a", "source_b", "source_both")]
    assert all(np.allclose(p, priors[0], atol=1e-14, rtol=0) for p in priors)
    scalar_label_cases = 0
    for regime in r.REGIMES:
        world = r.make_world(11, regime)
        probe = r.Learner("source_both", world["sources"])
        for observation in world["stream"][:6]:
            probe.update(*observation)
        matrices = probe.probabilities()
        scalar_brier = []
        scalar_accuracy = []
        for first, second in ((0, 1), (1, 0)):
            for x in range(r.N):
                intermediate = world["truth"][first][x]
                target = world["truth"][second][intermediate]
                probs = [sum(float(matrices[first, x, z]) * float(matrices[second, z, y])
                             for z in range(r.N)) for y in range(r.N)]
                scalar_brier.append(sum((value - int(y == target)) ** 2 for y, value in enumerate(probs)))
                ties = [y for y, value in enumerate(probs) if abs(value - max(probs)) <= 1e-12]
                scalar_accuracy.append((1 / len(ties)) if target in ties else 0)
                scalar_label_cases += 1
        actual = r.evaluate(probe, world["truth"])["composition"]
        assert abs(actual["brier"] - sum(scalar_brier) / len(scalar_brier)) < 1e-14
        assert abs(actual["tie_accuracy"] - sum(scalar_accuracy) / len(scalar_accuracy)) < 1e-14
    return dict(joint_candidate_pairs=12, composed_queries=12,
                repeated_same_action_counterexample=True, same_initial_marginals=True,
                independent_scalar_composite_label_cases=scalar_label_cases)


def check_frozen_and_scores():
    observations = r.make_sources(8)
    frozen = r.Learner("frozen_source_both", observations)
    flexible = r.Learner("source_both", observations)
    # Both source classes prohibit fixed points: a single self-loop refutes both.
    before = frozen.probabilities().copy()
    frozen.update(0, 2, 2)
    flexible.update(0, 2, 2)
    assert frozen.rejected == 1 and not any(frozen.fallback)
    assert np.array_equal(frozen.probabilities(), before)
    assert frozen.observed[0, 2] == 2 and frozen.probabilities()[0, 2, 2] == 0
    assert flexible.fallback == [True, False]
    assert flexible.probabilities()[0, 2, 2] == 1
    assert r.score([[1, 0, 0, 0, 0, 0]], [0])["brier"] == 0
    assert r.score([[1, 0, 0, 0, 0, 0]], [1])["brier"] == 2
    assert r.score([[1, 0, 0, 0, 0, 0]], [1])["confident_wrong"] == 1
    assert abs(r.score([[1/6] * 6], [3])["tie_accuracy"] - 1/6) < 1e-14
    assert r.score(np.zeros((0, 6)), []) is None
    assert tuple(inspect.signature(r.Learner.__init__).parameters) == ("self", "method", "source_observations")
    assert tuple(inspect.signature(r.Learner.update).parameters) == ("self", "action", "x", "y")
    return dict(frozen_rejects_but_records=True, fallback_only_affected_action=True,
                score_boundary_checks=True, learner_interface_has_no_hidden_truth=True)


def continuation(payload):
    learner = r.Learner.restore(payload["state"])
    predictions = [learner.probabilities().tolist()]
    for observation in payload["remaining"]:
        learner.update(*observation)
        predictions.append(learner.probabilities().tolist())
    return dict(predictions=predictions, final_state=learner.state())


def check_restarts():
    rows = []
    with tempfile.TemporaryDirectory(prefix="cevahir_relations_resume_") as temporary:
        path = Path(temporary)
        for regime in r.REGIMES:
            world = r.make_world(17, regime)
            for method in r.METHODS:
                learner = r.Learner(method, world["sources"])
                for observation in world["stream"][:7]:
                    learner.update(*observation)
                payload = json.loads(r.canonical_json(dict(state=learner.state(), remaining=world["stream"][7:])))
                # Reference continues the living object; it never calls restore.
                uninterrupted_predictions = [learner.probabilities().tolist()]
                for observation in payload["remaining"]:
                    learner.update(*observation)
                    uninterrupted_predictions.append(learner.probabilities().tolist())
                expected = dict(predictions=uninterrupted_predictions, final_state=learner.state())
                input_path, output_path = path / "input.json", path / "output.json"
                input_path.write_text(r.canonical_json(payload), encoding="utf-8")
                subprocess.run([sys.executable, __file__, "--resume", str(input_path), str(output_path)],
                               check=True, capture_output=True, text=True)
                actual = json.loads(output_path.read_text(encoding="utf-8"))
                assert r.canonical_json(expected) == r.canonical_json(actual)
                # Raw source episodes are not in serialized state; learned source tables remain.
                assert "source_observations" not in payload["state"]
                rows.append(dict(method=method, regime=regime, exact_json_equal=True,
                                 reference="uninterrupted_live_object"))
    return rows


def check_recorded_results():
    primary = json.loads((r.RESULTS / "relation_learning.json").read_text(encoding="utf-8"))
    assert primary["source_sha256"] == r.sha(r.HERE / "relation_learning.py")
    assert primary["protocol_sha256"] == r.sha(r.PROTOCOL)
    assert len(primary["worlds"]) == 72 and len(primary["runs"]) == 576
    worlds = {(w["seed"], w["regime"]): w for w in primary["worlds"]}
    checkpoints = 0
    for row in primary["runs"]:
        world = worlds[row["seed"], row["regime"]]
        learner = r.Learner(row["method"], world["sources"])
        assert len(row["snapshots"]) == 13 and len(row["prequential"]) == 12
        for step, prequential in enumerate(row["prequential"]):
            action, x, y = world["stream"][step]
            assert prequential["t"] == learner.steps == step
            assert np.array_equal(learner.probabilities()[action, x], prequential["probabilities"])
            assert learner.observed[action, x] == -1
            learner.update(action, x, y)
        for snapshot in row["snapshots"]:
            assert snapshot["direct"]["queries"] == 12
            assert snapshot["composition"]["queries"] == 12
            if snapshot["t"] < 12:
                assert snapshot["unseen"]["queries"] == 12 - snapshot["t"]
            else:
                assert snapshot["unseen"] is None
            for metric in (snapshot["direct"], snapshot["unseen"], snapshot["composition"]):
                if metric is not None:
                    assert -1e-14 <= metric["brier"] <= 2 + 1e-14
                    assert 0 <= metric["tie_accuracy"] <= 1
            checkpoints += 1
        final = row["snapshots"][-1]
        if row["method"] != "frozen_source_both" or row["regime"] == "compatible":
            assert final["direct"]["brier"] < 1e-25
            assert final["composition"]["brier"] < 1e-25
        expected_source_count = {"source_a": 12, "source_b": 12, "source_both": 24,
                                 "unrelated_sources": 24, "frozen_source_both": 24}.get(row["method"], 0)
        assert final["resources"]["source_observations"] == expected_source_count
        assert final["resources"]["rename_attempts"] == expected_source_count // 12 * 720
    assert checkpoints == 7488
    return dict(trajectories=576, checkpoints=checkpoints, prequential_labels=6912,
                complete_evidence_endpoints_checked=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", nargs=2)
    parser.add_argument("--structural-only", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.resume:
        source, target = map(Path, args.resume)
        payload = json.loads(source.read_text(encoding="utf-8"))
        target.write_text(r.canonical_json(continuation(payload)), encoding="utf-8")
        return
    checks = dict(classes=check_classes(), equivariance=check_equivariance(),
                  composition=check_composition(), frozen_and_scores=check_frozen_and_scores(),
                  preservation=preservation())
    if not args.structural_only:
        checks["fresh_process_resume"] = check_restarts()
        checks["recorded_results"] = check_recorded_results()
        if args.rerun:
            result = subprocess.run([sys.executable, str(r.HERE / "relation_learning.py")],
                                    capture_output=True, text=True, check=True)
            checks["complete_rerun"] = dict(returncode=result.returncode, stdout=result.stdout.strip())
        (r.RESULTS / "verification.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == "__main__":
    main()
