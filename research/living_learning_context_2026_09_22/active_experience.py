"""Predeclared extension: learner selects queries, environment supplies outcomes."""
import argparse
import copy
import json
import random

import numpy as np

import context_learning as c

PROTOCOL = c.PROTOCOL.parent / "ACTIVE_EXPERIENCE_PROTOCOL.md"
PRIORS = ("joint_uniform_gate", "joint_balanced_family")
POLICIES = ("uniform_query", "disagreement_query")
CHECKPOINTS = (0, 1, 2, 4, 8, 16)


def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value, (tuple, list)) else value


class Controller:
    def __init__(self, method, seed):
        if method not in PRIORS:
            raise ValueError(method)
        self.learner = c.Learner(method)
        self.seen = np.zeros(512, dtype=bool)
        self.rng = random.Random(seed)
        self.scoring_candidate_input_cells = 0
        self.proposals = 0

    def observe(self, index, label):
        if self.seen[index]:
            raise ValueError("repeated experience outside protocol")
        self.learner.observe(index, label)
        self.seen[index] = True

    def choose(self, policy):
        available = np.flatnonzero(~self.seen)
        if policy == "uniform_query":
            index = int(self.rng.choice(available.tolist()))
            p = float(self.learner.probabilities([index])[0])
            self.scoring_candidate_input_cells += len(self.learner.candidate_ids)
            maximum = None
        elif policy == "disagreement_query":
            predictions = self.learner.probabilities(np.arange(512))
            self.scoring_candidate_input_cells += len(self.learner.candidate_ids)*512
            scores = predictions*(1-predictions)
            maximum = float(scores[available].max())
            tied = available[np.isclose(scores[available], maximum, atol=1e-12, rtol=0)]
            index = int(self.rng.choice(tied.tolist()))
            p = float(predictions[index])
        else:
            raise ValueError(policy)
        self.proposals += 1
        return dict(index=index, probability=p, uncertainty=p*(1-p), maximum_available=maximum)

    def state(self):
        return dict(learner=self.learner.state(), seen=self.seen.tolist(), rng_state=self.rng.getstate(),
                    scoring_candidate_input_cells=self.scoring_candidate_input_cells, proposals=self.proposals)

    @classmethod
    def restore(cls, state):
        result = cls.__new__(cls)
        saved = copy.deepcopy(state)
        result.learner = c.Learner.restore(saved["learner"])
        result.seen = np.array(saved["seen"], dtype=bool)
        result.rng = random.Random()
        result.rng.setstate(tuples(saved["rng_state"]))
        result.scoring_candidate_input_cells = saved["scoring_candidate_input_cells"]
        result.proposals = saved["proposals"]
        return result


def evaluate(controller, world):
    result = c.evaluate(controller.learner, world, set(np.flatnonzero(controller.seen).tolist()))
    result.update(queries=controller.proposals,
                  query_prediction_candidate_cells=controller.scoring_candidate_input_cells,
                  controller_json_bytes=len(c.encode(controller.state()).encode()))
    return result


def run_one(world, method, policy):
    controller = Controller(method, 918_431+23*world["seed"])
    for index, label in world["stream"][:48]:
        controller.observe(index, label)
    snapshots = [evaluate(controller, world)]
    trace = []
    for q in range(1, 17):
        proposal = controller.choose(policy)
        # Only after choice does the environment disclose the selected result.
        label = world["truth"][proposal["index"]]
        proposal.update(query=q, label=label, mse=(proposal["probability"]-label)**2)
        trace.append(proposal)
        controller.observe(proposal["index"], label)
        if q in CHECKPOINTS:
            snapshots.append(evaluate(controller, world))
    return dict(seed=world["seed"], method=method, policy=policy, trace=trace,
                snapshots=snapshots, final_controller_sha256=__import__("hashlib").sha256(c.encode(controller.state()).encode()).hexdigest())


def run_all():
    worlds = [c.make_world(seed, "aligned") for seed in c.SEEDS]
    return dict(date="2026-09-22", source_sha256=c.sha(__file__), primary_source_sha256=c.sha(c.HERE / "context_learning.py"),
                protocol_sha256=c.sha(PROTOCOL),
                runs=[run_one(world, method, policy) for world in worlds for method in PRIORS for policy in POLICIES])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    data = run_all()
    path = c.RESULTS / "active_experience.json"
    if args.write:
        path.write_text(c.encode(data)+"\n", encoding="utf-8")
        print(json.dumps(dict(status="written", trajectories=len(data["runs"]), bytes=path.stat().st_size)))
    else:
        assert c.encode(data) == c.encode(json.loads(path.read_text(encoding="utf-8")))
        print(json.dumps(dict(status="exact_rerun_passed", trajectories=len(data["runs"]))))


if __name__ == "__main__":
    main()
