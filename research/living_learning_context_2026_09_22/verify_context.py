"""Independent semantic, ambiguity, prior, resume and preservation checks."""
import argparse
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

import context_learning as c


def preservation():
    baseline = json.loads((c.RESULTS / "preservation_baseline.json").read_text(encoding="utf-8"))
    changed = [p for p, digest in baseline["files"].items()
               if not (c.ROOT/p).exists() or c.sha(c.ROOT/p) != digest]
    assert not changed, changed
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c.ROOT, text=True).strip()
    assert head == baseline["base_head"]
    return dict(files_unchanged=len(baseline["files"]), head_unchanged=head)


def scalar(candidate, index):
    gate, pair = divmod(candidate, 1024)
    f0, f1 = divmod(pair, 32)
    h, remaining = divmod(index, 32)
    s, x = divmod(remaining, 16)
    context = s if gate == 15 else ((gate+1) & h).bit_count() % 2
    mask, bias = divmod(f1 if context else f0, 2)
    return ((mask & x).bit_count() % 2) ^ bias


def structural():
    unique = set()
    checked = 0
    for candidate in range(16384):
        row = bytes(scalar(candidate, index) for index in range(512))
        assert row == c.FUNCTIONS[candidate].astype(np.uint8).tobytes()
        unique.add(row)
        checked += 512
    assert len(unique) == 15904
    for seed in c.SEEDS:
        reference = c.make_world(seed, "aligned")
        for regime in c.REGIMES:
            world = c.make_world(seed, regime)
            assert world["stream"][:48] == reference["stream"][:48]
            assert len({row[0] for row in world["stream"]}) == 64
            prefix = [index for index, _ in world["stream"][:48]]
            suffix = [index for index, _ in world["stream"][48:]]
            assert all(i in world["aligned_ids"] for i in prefix)
            assert sum(i in world["reversed_ids"] for i in suffix) == {"aligned": 0, "balanced": 8, "reversed": 16}[regime]
            f0, f1 = world["expert_ids"]
            assert sum(c.EXPERT_OUTPUTS[f0] != c.EXPERT_OUTPUTS[f1]) == 8
            true_id, surface_id = world["true_candidate_id"], world["matched_surface_candidate_id"]
            assert world["truth"] == [scalar(true_id, i) for i in range(512)]
            assert all(scalar(true_id, i) == scalar(surface_id, i) for i in prefix)
            assert sum(scalar(true_id, i) != scalar(surface_id, i) for i in range(512)) == 128
            assert sum(scalar(true_id, i) != scalar(surface_id, i) for i in world["reversed_ids"]) == 128
    uniform, balanced = c.Learner("joint_uniform_gate"), c.Learner("joint_balanced_family")
    assert np.array_equal(uniform.candidate_ids, balanced.candidate_ids)
    assert np.allclose(uniform.posterior()["gate_mass"], np.full(16, 1/16))
    assert np.allclose(balanced.posterior()["gate_mass"], [1/30]*15+[.5])
    world = c.make_world(13, "reversed")
    pair_odds = []
    for learner, prior_ratio in ((uniform, 1), (balanced, 1/15)):
        true_id, surface_id = world["true_candidate_id"], world["matched_surface_candidate_id"]
        k = 0
        for t, (index, label) in enumerate(world["stream"]):
            learner.observe(index, label)
            k += scalar(true_id, index) != scalar(surface_id, index)
            weights = learner.weights()
            assert np.isclose(weights[true_id]/weights[surface_id], prior_ratio*19**k, rtol=1e-12, atol=1e-12)
            if t == 47:
                assert k == 0
        pair_odds.append(dict(method=learner.method, distinguishing_records=k, multiplier=19**k))
    snapshot = uniform.state()
    encoded = c.encode(snapshot)
    first, second = c.Learner.restore(snapshot), c.Learner.restore(snapshot)
    unchanged = c.encode(second.state())
    first.observe(5, 1)
    uniform.observe(8, 0)
    assert c.encode(snapshot) == encoded and c.encode(second.state()) == unchanged
    assert tuple(inspect.signature(c.Learner.__init__).parameters) == ("self", "method")
    assert tuple(inspect.signature(c.Learner.observe).parameters) == ("self", "index", "label")
    return dict(scalar_bank_cells=checked, unique_functions=len(unique), world_checks=72,
                prefix_equality_and_suffix_balance=True, exact_pair_odds=pair_odds,
                state_isolation=True, ordinary_constructor_has_no_teacher=True)


def continuation(payload):
    learner = c.Learner.restore(payload["state"])
    trace = []
    for index, label in payload["remaining"]:
        trace.append(float(learner.probabilities([index])[0]))
        learner.observe(index, label)
    return dict(trace=trace, final_state=learner.state())


def process_restarts():
    records = []
    with tempfile.TemporaryDirectory(prefix="cevahir_context_resume_") as folder:
        source, output = Path(folder)/"source.json", Path(folder)/"output.json"
        for regime in c.REGIMES:
            world = c.make_world(17, regime)
            for method in c.METHODS:
                learner = c.make_learner(method, world)
                for row in world["stream"][:52]:
                    learner.observe(*row)
                payload = dict(state=learner.state(), remaining=world["stream"][52:])
                expected_trace = []
                for index, label in payload["remaining"]:
                    expected_trace.append(float(learner.probabilities([index])[0]))
                    learner.observe(index, label)
                expected = dict(trace=expected_trace, final_state=learner.state())
                source.write_text(c.encode(payload), encoding="utf-8")
                subprocess.run([sys.executable, __file__, "--resume", str(source), str(output)],
                               check=True, text=True, capture_output=True)
                actual = json.loads(output.read_text(encoding="utf-8"))
                assert c.encode(actual) == c.encode(expected)
                records.append(dict(method=method, regime=regime, exact_equal=True,
                                    reference="uninterrupted_live_object"))
    return records


def recorded():
    data = json.loads((c.RESULTS / "context_learning.json").read_text(encoding="utf-8"))
    assert data["source_sha256"] == c.sha(c.HERE / "context_learning.py")
    assert data["protocol_sha256"] == c.sha(c.PROTOCOL)
    assert len(data["runs"]) == 504
    worlds = {(w["seed"], w["regime"]): w for w in data["worlds"]}
    records = 0
    for row in data["runs"]:
        world = worlds[row["seed"], row["regime"]]
        learner = c.make_learner(row["method"], world)
        for i, (index, label) in enumerate(world["stream"]):
            pre = row["prequential"][i]
            assert pre["t"] == i and pre["index"] == index and pre["label"] == label
            p = float(learner.probabilities([index])[0])
            assert p == pre["probability"] and pre["mse"] == (p-label)**2
            learner.observe(index, label)
            records += 1
        assert [s["t"] for s in row["snapshots"]] == list(c.CHECKPOINTS)
        assert row["snapshots"][-1]["resources"]["candidate_label_checks"] == len(learner.candidate_ids)*64
        for snapshot in row["snapshots"]:
            assert snapshot["unseen"]["n"] == 512-snapshot["t"]
            for kind in ("full", "aligned", "reversed", "unseen"):
                assert -1e-14 <= snapshot[kind]["mse"] <= 1+1e-14
            if snapshot["posterior"] is not None:
                assert abs(sum(snapshot["posterior"]["gate_mass"])-1) < 1e-12
        assert row["final_state_sha256"] == __import__("hashlib").sha256(c.encode(learner.state()).encode()).hexdigest()
    return dict(trajectories=504, snapshots=4032, prequential_predictions=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structural-only", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--resume", nargs=2)
    args = parser.parse_args()
    if args.resume:
        source, target = map(Path, args.resume)
        payload = json.loads(source.read_text(encoding="utf-8"))
        target.write_text(c.encode(continuation(payload)), encoding="utf-8")
        return
    result = dict(structural=structural(), preservation=preservation())
    if not args.structural_only:
        result["recorded"] = recorded()
        result["fresh_process_resume"] = process_restarts()
        if args.rerun:
            replay = subprocess.run([sys.executable, str(c.HERE / "context_learning.py")],
                                    check=True, capture_output=True, text=True)
            result["full_rerun"] = dict(returncode=replay.returncode, stdout=replay.stdout.strip())
        (c.RESULTS / "verification.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
