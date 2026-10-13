"""Matched life summaries for main passive and separately planned active runs."""
import argparse
import json
import statistics

import context_learning as c


def avg(values):
    return statistics.fmean(values)


def comparison(a, b):
    delta = [x-y for x, y in zip(a, b)]
    return dict(mean_difference=avg(delta), better=sum(d < -1e-12 for d in delta),
                tied=sum(abs(d) <= 1e-12 for d in delta), worse=sum(d > 1e-12 for d in delta))


def snapshots_summary(rows, key):
    result = []
    for i in range(len(rows[0]["snapshots"])):
        snaps = [r["snapshots"][i] for r in rows]
        result.append({key: snaps[0][key],
                       **{metric: {measure: avg([s[metric][measure] for s in snaps]) for measure in ("mse", "accuracy")}
                          for metric in ("full", "aligned", "reversed", "unseen")},
                       "true_gate_mass": None if snaps[0]["true_gate_mass"] is None else avg([s["true_gate_mass"] for s in snaps]),
                       "surface_mass": None if snaps[0]["posterior"] is None else avg([s["posterior"]["gate_mass"][15] for s in snaps])})
    return result


def summarize(primary, active):
    groups, active_groups, active_pairs = [], [], []
    for regime in c.REGIMES:
        for method in c.METHODS:
            rows = sorted([r for r in primary["runs"] if r["regime"] == regime and r["method"] == method], key=lambda r:r["seed"])
            final = [r["snapshots"][-1] for r in rows]
            groups.append(dict(regime=regime, method=method, prefix_mse=avg([r["prefix_mse"] for r in rows]),
                               suffix_mse=avg([r["suffix_mse"] for r in rows]), snapshots=snapshots_summary(rows, "t"),
                               final_candidate_checks=avg([s["resources"]["candidate_label_checks"] for s in final]),
                               final_state_bytes=avg([s["resources"]["state_json_bytes"] for s in final])))
    for method in ("joint_uniform_gate", "joint_balanced_family"):
        policy_rows = {}
        for policy in ("uniform_query", "disagreement_query"):
            rows = sorted([r for r in active["runs"] if r["method"] == method and r["policy"] == policy], key=lambda r:r["seed"])
            policy_rows[policy] = rows
            active_groups.append(dict(method=method, policy=policy, snapshots=snapshots_summary(rows, "queries"),
                                      query_true_mse=avg([avg([t["mse"] for t in r["trace"]]) for r in rows]),
                                      final_query_prediction_cells=avg([r["snapshots"][-1]["query_prediction_candidate_cells"] for r in rows]),
                                      final_controller_bytes=avg([r["snapshots"][-1]["controller_json_bytes"] for r in rows])))
        pairs = []
        for i in range(6):
            left = [r["snapshots"][i] for r in policy_rows["disagreement_query"]]
            right = [r["snapshots"][i] for r in policy_rows["uniform_query"]]
            pairs.append(dict(queries=left[0]["queries"],
                              full_mse=comparison([s["full"]["mse"] for s in left], [s["full"]["mse"] for s in right]),
                              unseen_mse=comparison([s["unseen"]["mse"] for s in left], [s["unseen"]["mse"] for s in right])))
        active_pairs.append(dict(method=method, left="disagreement_query", right="uniform_query", comparisons=pairs))
    return dict(source_sha256=c.sha(__file__), primary_source_sha256=primary["source_sha256"],
                active_source_sha256=active["source_sha256"], life_count=24, descriptive_only=True,
                main_groups=groups, active_groups=active_groups, active_paired=active_pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    primary = json.loads((c.RESULTS / "context_learning.json").read_text(encoding="utf-8"))
    active = json.loads((c.RESULTS / "active_experience.json").read_text(encoding="utf-8"))
    result = summarize(primary, active)
    path = c.RESULTS / "paired_summary.json"
    if args.write:
        path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    else:
        assert result == json.loads(path.read_text(encoding="utf-8"))
    for g in result["main_groups"]:
        first, final = g["snapshots"][3], g["snapshots"][-1]
        print(f'{g["regime"]:9s} {g["method"]:22s} suffix={g["suffix_mse"]:.6f} '
              f'reverse48={first["reversed"]["mse"]:.6f} reverse64={final["reversed"]["mse"]:.6f} '
              f'unseen64={final["unseen"]["mse"]:.6f} truegate={final["true_gate_mass"]}')
    for g in result["active_groups"]:
        print(g["method"],g["policy"],[(s["queries"],round(s["full"]["mse"],8)) for s in g["snapshots"]])


if __name__ == "__main__":
    main()
