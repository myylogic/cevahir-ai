"""Reproduce the explicitly post-hoc endpoint/path feasibility diagnostic.

The offline optimizer receives extra data and compute. A feasible final point
does not establish a feasible path under the tested per-step acceptance rule.
Default mode checks the existing result without writing; --write is explicit.
"""
import argparse
import json
from pathlib import Path

import geometry as g
import numpy as np

ANCHORS = 64
OFFLINE_STEPS = 1000
RATE = .05


def run():
    rows = []
    for seed in g.SEEDS:
        world = g.make_world(seed)
        theta = world["theta"].copy()
        old_x, old_y = world["old_x"], world["old_y"]
        new_x = world["new_x"].reshape(-1, 2)
        new_y = world["new_y"].reshape(-1)
        anchor_x, anchor_y = old_x[:ANCHORS], old_y[:ANCHORS]
        reference = g.forward(theta, anchor_x)
        initial_loss = g.mse(theta, anchor_x, anchor_y)
        max_loss = initial_loss
        for _ in range(OFFLINE_STEPS):
            old_gradient = g.gradient(theta, old_x, old_y)
            new_gradient = g.gradient(theta, new_x, new_y)
            theta -= RATE * (old_gradient + new_gradient) / 2
            max_loss = max(max_loss, g.mse(theta, anchor_x, anchor_y))
        endpoint_loss = g.mse(theta, anchor_x, anchor_y)
        rows.append({
            "seed": seed,
            "initial_anchor_label_mse": initial_loss,
            "joint_endpoint_anchor_label_mse": endpoint_loss,
            "joint_endpoint_label_budget_feasible": endpoint_loss <= initial_loss + g.DRIFT_BUDGET,
            "joint_path_max_anchor_label_mse_increase": max_loss - initial_loss,
            "joint_endpoint_output_drift": float(np.mean((g.forward(theta, anchor_x) - reference)**2)),
            "teacher_output_drift": float(np.mean((g.forward(world["teacher"], anchor_x) - reference)**2)),
            **g.risks(theta, world),
        })
    return {
        "status": "posthoc_feasibility_diagnostic_after_primary_and_control_results",
        "anchors": ANCHORS,
        "offline_steps": OFFLINE_STEPS,
        "runs": rows,
        "joint_endpoints_inside_label_budget": sum(r["joint_endpoint_label_budget_feasible"] for r in rows),
        "joint_paths_violating_label_budget": sum(r["joint_path_max_anchor_label_mse_increase"] > g.DRIFT_BUDGET for r in rows),
        "teacher_endpoints_violating_output_budget": sum(r["teacher_output_drift"] > g.DRIFT_BUDGET for r in rows),
        "scope": "A feasible endpoint need not have a feasible path under this proposal rule. Offline solution uses extra data/compute and never entered online learning.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = run()
    path = Path(__file__).parent / "results/endpoint_diagnostic.json"
    if args.write:
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    else:
        g.compare(result, json.loads(path.read_text(encoding="utf-8")))
    print(json.dumps({"status": "passed", "mode": "record" if args.write else "check",
                      "runs": len(result["runs"]),
                      "feasible_label_endpoints": result["joint_endpoints_inside_label_budget"],
                      "tested_paths_violating_label_budget": result["joint_paths_violating_label_budget"]}))


if __name__ == "__main__":
    main()
