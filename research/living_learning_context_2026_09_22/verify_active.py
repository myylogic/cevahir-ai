"""Check query choice, outcomes, full policy replay and persistent control state."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

import active_experience as a
import context_learning as c
import verify_context as v


def resume(payload):
    controller = a.Controller.restore(payload["state"])
    trace = []
    for _ in range(payload["steps"]):
        proposal = controller.choose(payload["policy"])
        label = payload["environment_truth"][proposal["index"]]
        proposal["label"] = label
        trace.append(proposal)
        controller.observe(proposal["index"], label)
    return dict(trace=trace, state=controller.state())


def check():
    data = json.loads((c.RESULTS / "active_experience.json").read_text(encoding="utf-8"))
    assert data["source_sha256"] == c.sha(c.HERE / "active_experience.py")
    assert data["primary_source_sha256"] == c.sha(c.HERE / "context_learning.py")
    assert data["protocol_sha256"] == c.sha(a.PROTOCOL)
    assert len(data["runs"]) == 96
    checked = 0
    for row in data["runs"]:
        world = c.make_world(row["seed"], "aligned")
        controller = a.Controller(row["method"], 918_431+23*row["seed"])
        for item in world["stream"][:48]:
            controller.observe(*item)
        for recorded in row["trace"]:
            index, label = recorded["index"], recorded["label"]
            assert not controller.seen[index]
            assert label == v.scalar(world["true_candidate_id"], index)
            proposal = controller.choose(row["policy"])
            for key, value in proposal.items():
                assert recorded[key] == value
            assert recorded["mse"] == (proposal["probability"]-label)**2
            if row["policy"] == "disagreement_query":
                # Independent all-input calculation of weighted disagreement.
                bank = c.FUNCTIONS[controller.learner.candidate_ids]
                weights = controller.learner.weights()
                ones = np.einsum("i,ij->j", weights, bank, optimize=False)
                scores = ones*(1-ones)
                best = float(max(scores[~controller.seen]))
                assert abs(scores[index]-best) <= 1.1e-12
            controller.observe(index, label)
            checked += 1
        assert [s["queries"] for s in row["snapshots"]] == list(a.CHECKPOINTS)
        expected_cells = 16*16384*(512 if row["policy"] == "disagreement_query" else 1)
        assert controller.scoring_candidate_input_cells == expected_cells
        assert controller.learner.candidate_label_checks == 64*16384
        assert row["final_controller_sha256"] == hashlib.sha256(c.encode(controller.state()).encode()).hexdigest()
    assert tuple(inspect.signature(a.Controller.choose).parameters) == ("self", "policy")
    return dict(trajectories=96, query_choices_checked=checked, outcome_labels_scalar_checked=checked,
                greedy_choices_independently_scored=768, policy_replay_exact=True,
                query_selection_interface_has_no_truth=True, selection_costs_counted=True)


def restarts():
    rows = []
    with tempfile.TemporaryDirectory(prefix="cevahir_active_resume_") as temporary:
        source, output = Path(temporary)/"source.json", Path(temporary)/"output.json"
        world = c.make_world(7, "aligned")
        for method in a.PRIORS:
            for policy in a.POLICIES:
                controller = a.Controller(method, 918_431+23*7)
                for item in world["stream"][:48]:
                    controller.observe(*item)
                for _ in range(3):
                    proposal = controller.choose(policy)
                    controller.observe(proposal["index"], world["truth"][proposal["index"]])
                payload = dict(state=controller.state(), steps=5, policy=policy, environment_truth=world["truth"])
                # The environment truth is held by the harness, not the restored controller.
                trace = []
                for _ in range(5):
                    proposal = controller.choose(policy)
                    label = world["truth"][proposal["index"]]
                    proposal["label"] = label
                    trace.append(proposal)
                    controller.observe(proposal["index"], label)
                expected = dict(trace=trace, state=controller.state())
                source.write_text(c.encode(payload), encoding="utf-8")
                subprocess.run([sys.executable, __file__, "--resume", str(source), str(output)],
                               capture_output=True, text=True, check=True)
                actual = json.loads(output.read_text(encoding="utf-8"))
                assert c.encode(actual) == c.encode(expected)
                rows.append(dict(method=method, policy=policy, choices_and_updates_exact=True,
                                 reference="uninterrupted_live_object"))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", nargs=2)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.resume:
        source, target = map(Path, args.resume)
        target.write_text(c.encode(resume(json.loads(source.read_text(encoding="utf-8")))), encoding="utf-8")
        return
    result = dict(checks=check(), persistent_controller=restarts(), preservation=v.preservation())
    if args.rerun:
        run = subprocess.run([sys.executable, str(c.HERE / "active_experience.py")], check=True, capture_output=True, text=True)
        result["full_rerun"] = dict(returncode=run.returncode, stdout=run.stdout.strip())
    (c.RESULTS / "active_verification.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
