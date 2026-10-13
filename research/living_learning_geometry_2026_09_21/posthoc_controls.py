"""Explicitly post-hoc mechanism controls; primary experiment stays unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import geometry as g
import numpy as np

MODES = ("online_loss_guard", "replay_loss_guard", "replay_two_steps")


class ControlLearner(g.Learner):
    def __init__(self, theta, mode, rate, x, y):
        super().__init__(theta, mode, rate, x, y)
        self.initial_anchor_label_mse = g.mse(theta, x, y)
        self.max_anchor_label_mse = self.initial_anchor_label_mse

    def step(self, x, y):
        pred = g.forward(self.theta, x)
        self.steps += 1
        self.costs["service_forward_rows"] += len(x)
        for _ in range(2 if self.mode == "replay_two_steps" else 1):
            direction = g.gradient(self.theta, x, y)
            self.costs["gradient_example_rows"] += len(x)
            if self.mode != "online_loss_guard":
                direction = .5 * (direction + g.gradient(self.theta, self.x, self.y))
                self.costs["gradient_example_rows"] += len(self.x)
            delta = -self.rate * direction
            if self.mode == "replay_two_steps":
                self.theta += delta
                self.costs["accepted_updates"] += 1
            else:
                accepted = False
                for attempt in range(13):
                    candidate = self.theta + delta * (2. ** -attempt)
                    risk = g.mse(candidate, self.x, self.y)
                    self.costs["candidate_forward_rows"] += len(self.x)
                    self.costs["candidate_attempts"] += 1
                    if risk <= self.initial_anchor_label_mse + g.DRIFT_BUDGET:
                        self.theta = candidate
                        accepted = True
                        self.costs["accepted_updates"] += 1
                        self.costs["halvings_on_accepted_updates"] += attempt
                        break
                if not accepted:
                    self.costs["rejected_updates"] += 1
            self.max_anchor_label_mse = max(self.max_anchor_label_mse, g.mse(self.theta, self.x, self.y))
        return pred


def run():
    primary = Path(g.__file__)
    rows, joint = [], []
    for seed in g.SEEDS:
        world = g.make_world(seed)
        tx = np.concatenate((world["old_x"], world["new_x"].reshape(-1, 2)))
        ty = np.concatenate((world["old_y"], world["new_y"].reshape(-1)))
        theta = world["theta"].copy()
        # Equal weighting of the two region means, as in replay; unequal sizes do not change the target mixture.
        for _ in range(1000):
            old = g.gradient(theta, tx[:128], ty[:128])
            new = g.gradient(theta, tx[128:], ty[128:])
            theta -= .05 * (old + new) / 2
        joint.append({"seed": seed, **g.risks(theta, world), "gradient_example_rows": 1000*len(tx),
                      "scope": "Offline extra-data extra-compute reference, not a matched online baseline"})
        for n in g.ANCHORS:
            for rate in g.RATES:
                for mode in MODES:
                    obj = ControlLearner(world["theta"], mode, rate, world["old_x"][:n], world["old_y"][:n])
                    pre = sum(float(np.mean((obj.step(x, y)-y)**2)) for x, y in zip(world["new_x"], world["new_y"])) / g.STEPS
                    rows.append({"seed": seed, "anchors": n, "rate": rate, "mode": mode,
                                 "prequential_mse": pre, **g.risks(obj.theta, world),
                                 "anchor_drift_mse": float(np.mean((g.forward(obj.theta, obj.x)-obj.reference)**2)),
                                 "anchor_label_mse": g.mse(obj.theta, obj.x, obj.y),
                                 "initial_anchor_label_mse": obj.initial_anchor_label_mse,
                                 "max_anchor_label_mse_increase": obj.max_anchor_label_mse-obj.initial_anchor_label_mse,
                                 "costs": dict(obj.costs)})
        print(f"posthoc seed {seed}: complete", file=sys.stderr, flush=True)
    aggregates = []
    for n in g.ANCHORS:
        for rate in g.RATES:
            for mode in MODES:
                subset = [r for r in rows if (r["anchors"],r["rate"],r["mode"])==(n,rate,mode)]
                metrics = ("old_mse","new_mse","bridge_mse","prequential_mse","anchor_drift_mse",
                           "anchor_label_mse","max_anchor_label_mse_increase")
                cost_names = sorted(set().union(*(r["costs"].keys() for r in subset)))
                aggregates.append({"anchors":n,"rate":rate,"mode":mode,
                                   "metrics":{k:g.summary(r[k] for r in subset) for k in metrics},
                                   "mean_costs":{k:sum(r["costs"].get(k,0) for r in subset)/len(subset) for k in cost_names}})
    return {"date":"2026-09-21","status":"posthoc_mechanism_controls_not_independent_validation",
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "primary_source_sha256":hashlib.sha256(primary.read_bytes()).hexdigest(),
            "runs":rows,"aggregate":aggregates,"offline_joint_references":joint,
            "limits":["Selected after inspecting primary results; same seeds and world reused",
                      "Old loss preservation and old output preservation are different objectives",
                      "All anchor decisions use training evidence; independent test labels never enter updates",
                      "Offline joint reference receives more data and compute"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true")
    args=parser.parse_args();result=run();path=Path(__file__).parent/'results/posthoc_controls.json'
    if args.write:path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    else:g.compare(json.loads(path.read_text(encoding='utf-8')),result)
    print(json.dumps({"status":"passed","mode":"record" if args.write else "check","runs":len(result['runs'])}))


if __name__=='__main__':main()
