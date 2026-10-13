"""Recompute descriptive summaries; no fitting or hyperparameter selection."""
import json
from collections import Counter
import numpy as np
import temporal_learning as t


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), min=float(a.min()), max=float(a.max()))


def summarize(data):
    groups = []
    for task in t.TASKS:
        for method in t.METHODS:
            rows = [r for r in data["runs"] if r["task"] == task and r["method"] == method]
            assert len(rows) == 24
            points = []
            for index, step in enumerate(t.CHECKPOINTS):
                snaps = [r["snapshots"][index] for r in rows]
                point = dict(labels=step)
                for split in ("short_test", "long_test"):
                    point[split] = {metric: stats([s[split][metric] for s in snaps])
                                    for metric in ("mse", "accuracy")}
                    point[split]["lives_accuracy_one"] = sum(s[split]["accuracy"] == 1 for s in snaps)
                    point[split]["lives_mse_below_1e_10"] = sum(s[split]["mse"] < 1e-10 for s in snaps)
                point["structure"] = {}
                for key in snaps[0]["structure"]:
                    values = [s["structure"][key] for s in snaps]
                    if values[0] is not None:
                        point["structure"][key] = stats(values)
                        if key in ("selected_leaves", "selected_max_depth"):
                            point["structure"][key]["histogram"] = dict(sorted(Counter(map(str,values)).items()))
                points.append(point)
            resources = {key: stats([r["snapshots"][-1]["resources"][key] for r in rows])
                         for key in rows[0]["snapshots"][-1]["resources"]}
            prequential = {period: {metric: stats([r[period][metric] for r in rows])
                                    for metric in ("mse", "accuracy")}
                           for period in ("all_prequential", "first_128", "last_128")}
            groups.append(dict(task=task, method=method, n_lives=24, checkpoints=points,
                               prequential=prequential, final_resources=resources))
    paired = []
    for task in t.TASKS:
        by_method = {m: {r["seed"]: r for r in data["runs"] if r["task"] == task and r["method"] == m}
                     for m in t.METHODS}
        for left, right in (("map_tree_8", "suffix_8"), ("mixture_tree_8", "map_tree_8"),
                            ("two_state", "mixture_tree_8"), ("suffix_5", "mixture_tree_8")):
            diffs = [by_method[left][seed]["snapshots"][-1]["long_test"]["mse"] -
                     by_method[right][seed]["snapshots"][-1]["long_test"]["mse"] for seed in t.SEEDS]
            paired.append(dict(task=task, left=left, right=right, metric="final_long_mse_left_minus_right",
                               differences=diffs, difference_summary=stats(diffs),
                               left_better=sum(v < -1e-12 for v in diffs),
                               tied_within_1e_12=sum(abs(v) <= 1e-12 for v in diffs),
                               right_better=sum(v > 1e-12 for v in diffs)))
    return dict(trajectories=len(data["runs"]), groups=groups, paired=paired,
                interpretation="Descriptive results for 24 paired lives. No universal guarantee or adaptive test-set selection.")


def main():
    data = json.loads((t.RESULTS / "temporal_learning.json").read_text(encoding="utf-8"))
    summary = summarize(data)
    (t.RESULTS / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    for group in summary["groups"]:
        final = group["checkpoints"][-1]
        print(json.dumps(dict(task=group["task"], method=group["method"],
                              short_mse=final["short_test"]["mse"]["mean"],
                              long_mse=final["long_test"]["mse"]["mean"],
                              long_accuracy=final["long_test"]["accuracy"]["mean"],
                              structure=final["structure"])))


if __name__ == "__main__": main()
