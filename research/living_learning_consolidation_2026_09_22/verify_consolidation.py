"""Independent scalar, schedule, context, retention and process-resume checks."""
import argparse
from fractions import Fraction
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

import consolidation as c


def preservation():
    baseline = json.loads((c.RESULTS / "preservation_baseline.json").read_text(encoding="utf-8"))
    changed = [name for name, digest in baseline["files"].items()
               if not (c.ROOT / name).exists() or c.sha(c.ROOT / name) != digest]
    assert not changed, changed
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c.ROOT, text=True).strip()
    assert head == baseline["base_head"]
    return dict(files_unchanged=len(baseline["files"]), head_unchanged=head)


def scalar(program, index):
    subset, bias = program
    value = bias
    for bit in subset:
        value ^= (index >> (5-bit)) & 1
    return value


def scalar_checks():
    checked = 0
    assert len(c.PROGRAMS) == 84 and len(c.LINEAR_IDS) == 14
    functions = []
    for j, program in enumerate(c.PROGRAMS):
        values = tuple(scalar(program, index) for index in range(64))
        assert values == tuple(c.OUTPUTS[j])
        functions.append(values)
        checked += 64
    assert len(set(functions)) == 84
    truth_checks = 0
    for seed in c.SEEDS:
        reference = c.make_world(seed, "visible_context")
        for regime in c.REGIMES:
            world = c.make_world(seed, regime)
            for name in ("old", "new"):
                expected = [scalar(c.PROGRAMS[world[f"{name}_teacher_id"]], x) for x in range(64)]
                assert expected == world[f"{name}_truth"]
                truth_checks += 64
            assert world["old_rows"] == reference["old_rows"]
            assert [row[1] for row in world["day_rows"]] == [row[1] for row in reference["day_rows"]]
            assert len(set(row[1] for row in world["old_rows"])) == 24
            assert len(set(row[1] for row in world["day_rows"])) == 24
            assert len(set(world["unseen_xs"])) == 40
            assert sum(y != world["new_truth"][x] for _, x, y in world["day_rows"]) == (4 if regime == "noisy_day" else 0)
            learner = c.Learner("frozen_core", world["old_rows"])
            assert learner.initial_id == world["old_teacher_id"]
            assert learner.common_source_checks == 336
            assert sum(a != b for a, b in zip(world["old_truth"], world["new_truth"])) == 32
    assert tuple(inspect.signature(c.Learner.__init__).parameters) == ("self", "method", "old_rows")
    assert tuple(inspect.signature(c.Learner.observe).parameters) == ("self", "context", "x", "y")
    return dict(candidate_scalar_outputs=checked, independent_teacher_labels=truth_checks,
                unique_functions=84, old_acquisition_exact_worlds=72,
                learner_interface_has_no_world_or_truth=True)


def structural_checks():
    world = c.make_world(19, "visible_context")
    online = c.Learner("online_rich", world["old_rows"])
    sleep = c.Learner("sleep_rich_replay", world["old_rows"])
    for row in world["day_rows"]:
        online.observe(*row)
        sleep.observe(*row)
    sleep.consolidate()
    online.consolidate()
    assert np.array_equal(online.errors, sleep.errors)
    assert online.core_ids == sleep.core_ids
    assert online.learning_checks == sleep.learning_checks == 4032
    # Counts computed independently with scalar predictions for every candidate.
    for context in (0, 1):
        data = [row for row in world["old_rows"]+world["day_rows"] if row[0] == context]
        for i, program in enumerate(c.PROGRAMS):
            errors = sum(scalar(program, x) != y for _, x, y in data)
            assert errors == int(sleep.errors[context, i])
    reverse = c.Learner("sleep_rich_replay", list(reversed(world["old_rows"])))
    for row in reversed(world["day_rows"]):
        reverse.observe(*row)
    reverse.consolidate()
    assert np.array_equal(reverse.errors, sleep.errors)
    assert reverse.core_ids == sleep.core_ids
    # No-new-information diagnostic: frozen source self-replay preserves source.
    dreams = c.Learner("sleep_self_replay", world["old_rows"])
    for row in world["day_rows"]:
        dreams.observe(*row)
    dreams.consolidate()
    assert dreams.core_ids == [[dreams.initial_id], [dreams.initial_id]]
    dreams.purge_episodes()
    assert not dreams.old_rows and not dreams.day_rows and dreams.synthetic_labels == 24
    snapshot = sleep.state()
    snapshot_json = c.encode(snapshot)
    first, second = c.Learner.restore(snapshot), c.Learner.restore(snapshot)
    second_before = c.encode(second.state())
    first.continue_observed(*world["future_rows"][0])
    assert c.encode(snapshot) == snapshot_json
    assert c.encode(second.state()) == second_before
    sleep.continue_observed(*world["future_rows"][0])
    assert c.encode(snapshot) == snapshot_json
    # Two incompatible labels: best average squared loss is exactly 1/4.
    for denominator in (2, 10, 100):
        for numerator in range(denominator+1):
            p = Fraction(numerator, denominator)
            risk = (p*p + (p-1)*(p-1))/2
            assert risk == (p-Fraction(1, 2))**2 + Fraction(1, 4)
    # In our world half of all x conflict, giving tight balanced lower bound 1/8.
    optimum = [Fraction(a+b, 2) for a, b in zip(world["old_truth"], world["new_truth"])]
    risk = sum((p-a)**2 + (p-b)**2 for p, a, b in zip(optimum, world["old_truth"], world["new_truth"])) / 128
    assert risk == Fraction(1, 8)
    return dict(schedule_count_identity=True, independent_error_count_cells=168,
                source_and_day_order_invariance=True, self_replay_preserves_old_model=True,
                exported_and_restored_states_are_isolated=True,
                conflicting_input_exact_lower_bound="1/4", full_hidden_world_tight_bound="1/8")


def resume_payload(payload):
    learner = c.Learner.restore(payload["state"])
    values = [[learner.predict(context, range(64), False).tolist() for context in (0, 1)]]
    for row in payload["future"]:
        learner.continue_observed(*row)
        values.append([learner.predict(context, range(64), False).tolist() for context in (0, 1)])
    return dict(values=values, final_state=learner.state())


def restarts():
    records = []
    with tempfile.TemporaryDirectory(prefix="cevahir_consolidation_resume_") as folder:
        input_path, output_path = Path(folder)/"input.json", Path(folder)/"output.json"
        for regime in c.REGIMES:
            world = c.make_world(11, regime)
            for method in c.METHODS:
                learner = c.Learner(method, world["old_rows"])
                for row in world["day_rows"]:
                    learner.observe(*row)
                learner.consolidate()
                learner.purge_episodes()
                payload = json.loads(c.encode(dict(state=learner.state(), future=world["future_rows"])))
                assert payload["state"]["old_rows"] == payload["state"]["day_rows"] == []
                # Reference is the uninterrupted object, not another restore call.
                values = [[learner.predict(context, range(64), False).tolist() for context in (0, 1)]]
                for row in world["future_rows"]:
                    learner.continue_observed(*row)
                    values.append([learner.predict(context, range(64), False).tolist() for context in (0, 1)])
                expected = dict(values=values, final_state=learner.state())
                input_path.write_text(c.encode(payload), encoding="utf-8")
                subprocess.run([sys.executable, __file__, "--resume", str(input_path), str(output_path)],
                               check=True, capture_output=True, text=True)
                actual = json.loads(output_path.read_text(encoding="utf-8"))
                assert c.encode(expected) == c.encode(actual)
                records.append(dict(regime=regime, method=method, exact_equal=True,
                                    reference="uninterrupted_live_object", raw_episodes_present=False))
    return records


def check_recorded():
    data = json.loads((c.RESULTS / "consolidation.json").read_text(encoding="utf-8"))
    assert data["source_sha256"] == c.sha(c.HERE / "consolidation.py")
    assert data["protocol_sha256"] == c.sha(c.PROTOCOL)
    worlds = {(w["seed"], w["regime"]): w for w in data["worlds"]}
    runs = {(r["seed"], r["regime"], r["method"]): r for r in data["runs"]}
    assert len(worlds) == 72 and len(runs) == 576
    prequential_count = 0
    for key, row in runs.items():
        world = worlds[key[:2]]
        learner = c.Learner(row["method"], world["old_rows"])
        for i, observed in enumerate(world["day_rows"]):
            pre = row["prequential"][i]
            context, x, y = observed
            assert pre["t"] == learner.steps == i
            p = float(learner.predict(context, [x])[0])
            assert pre["probability"] == p
            assert pre["observed_mse"] == (p-y)**2
            assert pre["true_mse"] == (p-world["new_truth"][x])**2
            learner.observe(*observed)
            prequential_count += 1
        learner.consolidate()
        learner.purge_episodes()
        assert c.encode(learner.state()) == c.encode(row["final_state"])
        assert len(row["snapshots"]) == 7
        assert row["snapshots"][-1]["resources"]["raw_episode_rows"] == 0
        for snapshot in row["snapshots"]:
            if world["regime"] == "hidden_context":
                assert snapshot["core"]["balanced_mse"] >= .125 - 1e-14
                assert snapshot["with_cache"]["balanced_mse"] >= .125 - 1e-14
            for kind in ("core", "with_cache"):
                for split in ("old", "new", "unseen", "seen"):
                    metric = snapshot[kind][split]
                    assert 0 <= metric["mse"] <= 1 and 0 <= metric["accuracy"] <= 1
        # Independent scalar final predictions, labels and MSE avoid evaluate().
        for observed_context, truth_key, split in ((0, "old_truth", "old"),
                                                  (world["new_observed_context"], "new_truth", "new")):
            ids = learner.core_ids[observed_context]
            risk = sum((sum(scalar(c.PROGRAMS[i], x) for i in ids)/len(ids) - world[truth_key][x])**2 for x in range(64))/64
            assert abs(risk-row["snapshots"][-1]["core"][split]["mse"]) < 1e-14
    for seed in c.SEEDS:
        for regime in c.REGIMES:
            online, sleep = (runs[seed, regime, m] for m in ("online_rich", "sleep_rich_replay"))
            for key in ("errors", "evidence_counts", "core_ids", "bank"):
                assert online["final_state"][key] == sleep["final_state"][key]
            assert online["final_state"]["learning_checks"] == sleep["final_state"]["learning_checks"] == 4032
            assert online["snapshots"][-1]["core"] == sleep["snapshots"][-1]["core"]
            assert runs[seed, regime, "sleep_linear_replay"]["final_state"]["learning_checks"] == 672
            assert runs[seed, regime, "sleep_rich_recent"]["final_state"]["learning_checks"] == 2016
            reference = [p["probability"] for p in runs[seed, regime, "frozen_core"]["prequential"]]
            for method in c.METHODS:
                if method != "online_rich":
                    assert reference == [p["probability"] for p in runs[seed, regime, method]["prequential"]]
    return dict(trajectories=576, snapshots=4032, prequential_records=prequential_count,
                scalar_final_query_checks=576*128, schedule_equivalent_pairs=72,
                cache_purged_all_runs=True, hidden_context_lower_bound_all_snapshots=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structural-only", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--resume", nargs=2)
    args = parser.parse_args()
    if args.resume:
        src, dst = map(Path, args.resume)
        dst.write_text(c.encode(resume_payload(json.loads(src.read_text(encoding="utf-8")))), encoding="utf-8")
        return
    checks = dict(scalar=scalar_checks(), structure=structural_checks(), preservation=preservation())
    if not args.structural_only:
        checks["recorded"] = check_recorded()
        checks["fresh_process_resume"] = restarts()
        if args.rerun:
            result = subprocess.run([sys.executable, str(c.HERE / "consolidation.py")],
                                    check=True, capture_output=True, text=True)
            checks["full_rerun"] = dict(returncode=result.returncode, stdout=result.stdout.strip())
        (c.RESULTS / "verification.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
