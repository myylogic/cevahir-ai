"""Descriptive paired life-level summaries; no outcome-driven model selection."""
import argparse
import json
import statistics

import consolidation as c


def avg(values):
    return statistics.fmean(values)


def paired(left, right):
    differences = [a-b for a, b in zip(left, right)]
    return dict(mean_difference=avg(differences), better=sum(d < -1e-12 for d in differences),
                tied=sum(abs(d) <= 1e-12 for d in differences), worse=sum(d > 1e-12 for d in differences))


def summarize(data):
    index = {(row["seed"], row["regime"], row["method"]): row for row in data["runs"]}
    groups, comparisons = [], []
    for regime in c.REGIMES:
        for method in c.METHODS:
            rows = [index[seed, regime, method] for seed in c.SEEDS]
            phases = []
            for i in range(7):
                snaps = [row["snapshots"][i] for row in rows]
                metrics = {}
                for kind in ("core", "with_cache"):
                    metrics[kind] = {split: {metric: avg([s[kind][split][metric] for s in snaps])
                                           for metric in ("mse", "accuracy")} for split in ("old", "new", "unseen", "seen")}
                    metrics[kind]["balanced_mse"] = avg([s[kind]["balanced_mse"] for s in snaps])
                phases.append(dict(phase=snaps[0]["phase"], **metrics))
            finals = [row["snapshots"][-1] for row in rows]
            groups.append(dict(regime=regime, method=method,
                               prequential_true_mse=avg([r["prequential_true_mse"] for r in rows]),
                               prequential_observed_mse=avg([r["prequential_observed_mse"] for r in rows]),
                               exact_old_lives=sum(s["core"]["old"]["mse"] == 0 for s in finals),
                               exact_new_lives=sum(s["core"]["new"]["mse"] == 0 for s in finals),
                               exact_heldout_lives=sum(s["core"]["unseen"]["mse"] == 0 for s in finals),
                               mean_final_resources={k: avg([s["resources"][k] for s in finals]) for k in
                                                     ("common_source_checks", "learning_checks", "selection_calls", "raw_episode_rows",
                                                      "statistic_integer_cells", "statistic_array_bytes", "state_json_bytes", "synthetic_labels")},
                               snapshots=phases))
        for method in c.METHODS:
            if method == "sleep_rich_replay":
                continue
            primary = [index[seed, regime, "sleep_rich_replay"] for seed in c.SEEDS]
            other = [index[seed, regime, method] for seed in c.SEEDS]
            comparisons.append(dict(regime=regime, primary="sleep_rich_replay", other=method,
                                    prequential_true_mse=paired([r["prequential_true_mse"] for r in primary], [r["prequential_true_mse"] for r in other]),
                                    final_old_mse=paired([r["snapshots"][-1]["core"]["old"]["mse"] for r in primary],
                                                         [r["snapshots"][-1]["core"]["old"]["mse"] for r in other]),
                                    final_unseen_mse=paired([r["snapshots"][-1]["core"]["unseen"]["mse"] for r in primary],
                                                            [r["snapshots"][-1]["core"]["unseen"]["mse"] for r in other])))
    return dict(source_sha256=c.sha(__file__), primary_source_sha256=data["source_sha256"],
                lives=24, descriptive_only=True, groups=groups, comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    data = json.loads((c.RESULTS / "consolidation.json").read_text(encoding="utf-8"))
    result = summarize(data)
    path = c.RESULTS / "paired_summary.json"
    if args.write:
        path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    else:
        assert result == json.loads(path.read_text(encoding="utf-8"))
    for regime in c.REGIMES:
        print(regime)
        for g in result["groups"]:
            if g["regime"] != regime:
                continue
            final = g["snapshots"][-1]["core"]
            print(f'{g["method"]:25s} preq={g["prequential_true_mse"]:.6f} '
                  f'old={final["old"]["mse"]:.6f} new={final["new"]["mse"]:.6f} '
                  f'heldout={final["unseen"]["mse"]:.6f} balance={final["balanced_mse"]:.6f} '
                  f'exactnew={g["exact_new_lives"]}')


if __name__ == "__main__":
    main()
