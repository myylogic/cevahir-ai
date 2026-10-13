"""Paired descriptive summaries; independent unit is a life, not a checkpoint."""
import argparse
import json
import statistics

import relation_learning as r


def mean(values):
    return statistics.fmean(values)


def paired(left, right):
    deltas = [a - b for a, b in zip(left, right)]
    return dict(lives=len(deltas), mean_delta=mean(deltas),
                better=sum(d < -1e-12 for d in deltas),
                tied=sum(abs(d) <= 1e-12 for d in deltas),
                worse=sum(d > 1e-12 for d in deltas),
                min_delta=min(deltas), max_delta=max(deltas))


def summarize(data):
    groups, comparisons = [], []
    by_key = {(row["seed"], row["regime"], row["method"]): row for row in data["runs"]}
    for regime in r.REGIMES:
        for method in r.METHODS:
            rows = [by_key[seed, regime, method] for seed in r.SEEDS]
            checkpoints = []
            for t in (0, 2, 4, 6, 8, 10, 12):
                snapshots = [row["snapshots"][t] for row in rows]
                metrics = {}
                for name in ("direct", "unseen", "composition"):
                    entries = [snapshot[name] for snapshot in snapshots if snapshot[name] is not None]
                    metrics[name] = None if not entries else dict(
                        brier=mean([entry["brier"] for entry in entries]),
                        tie_accuracy=mean([entry["tie_accuracy"] for entry in entries]),
                        confident_wrong=sum(entry["confident_wrong"] for entry in entries))
                checkpoints.append(dict(t=t, **metrics))
            final = [row["snapshots"][-1] for row in rows]
            groups.append(dict(regime=regime, method=method, lives=24,
                               prequential_brier=mean([row["prequential_brier"] for row in rows]),
                               prequential_confident_wrong=sum(row["prequential_confident_wrong"] for row in rows),
                               composition_curve_mean=mean([mean([s["composition"]["brier"] for s in row["snapshots"]]) for row in rows]),
                               fallback_lives=sum(any(s["posterior"]["fallback"]) for s in final),
                               fallback_actions=sum(sum(s["posterior"]["fallback"]) for s in final),
                               rejected_updates=sum(s["posterior"]["rejected"] for s in final),
                               mean_final_filter_checks=mean([s["resources"]["filter_candidate_cell_checks"] for s in final]),
                               mean_state_bytes=mean([s["resources"]["serialized_state_bytes"] for s in final]),
                               initial_resources=rows[0]["snapshots"][0]["resources"], checkpoints=checkpoints))
        for baseline in r.METHODS:
            if baseline == "source_both":
                continue
            left = [by_key[seed, regime, "source_both"] for seed in r.SEEDS]
            right = [by_key[seed, regime, baseline] for seed in r.SEEDS]
            metrics = {"prequential_brier": paired([row["prequential_brier"] for row in left],
                                                  [row["prequential_brier"] for row in right])}
            for t in (2, 4, 6, 8, 10, 12):
                metrics[f"composition_brier_t{t}"] = paired(
                    [row["snapshots"][t]["composition"]["brier"] for row in left],
                    [row["snapshots"][t]["composition"]["brier"] for row in right])
            comparisons.append(dict(regime=regime, left="source_both", right=baseline, metrics=metrics))
    # Readiness is a descriptive diagnostic; no threshold fitted to outcomes.
    first_exact = []
    for row in data["runs"]:
        times = [s["t"] for s in row["snapshots"] if s["composition"]["brier"] < 1e-25]
        first_exact.append(dict(seed=row["seed"], regime=row["regime"], method=row["method"],
                                first_all_compositions_exact=min(times) if times else None))
    return dict(source_sha256=r.sha(__file__), primary_source_sha256=data["source_sha256"],
                descriptive_only=True, life_count=24, groups=groups,
                paired_comparisons=comparisons, first_exact=first_exact)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    data = json.loads((r.RESULTS / "relation_learning.json").read_text(encoding="utf-8"))
    summary = summarize(data)
    path = r.RESULTS / "paired_summary.json"
    if args.write:
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    else:
        assert summary == json.loads(path.read_text(encoding="utf-8"))
    for regime in r.REGIMES:
        print(regime)
        for group in summary["groups"]:
            if group["regime"] != regime:
                continue
            c = {s["t"]: s for s in group["checkpoints"]}
            print(f'{group["method"]:22s} preq={group["prequential_brier"]:.6f} '
                  f'comp4={c[4]["composition"]["brier"]:.6f} '
                  f'comp6={c[6]["composition"]["brier"]:.6f} '
                  f'comp8={c[8]["composition"]["brier"]:.6f} '
                  f'comp12={c[12]["composition"]["brier"]:.6f} '
                  f'wrong={group["prequential_confident_wrong"]} '
                  f'fallbacks={group["fallback_actions"]}')


if __name__ == "__main__":
    main()
