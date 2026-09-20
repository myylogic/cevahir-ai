"""Exact finite audit: prediction sufficiency versus correction sufficiency.

Standard library only. No model/source/research files outside this new directory
are changed. Hypothesis h encodes h(x) in bit x. Event IDs are list positions.
Corrections are trusted external label replacements, never inferred by learner.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
FULL_H = tuple(range(8))
OBSERVATIONS = tuple(itertools.product(range(3), range(2)))


def prediction(h, x):
    return (h >> x) & 1


def replay_counts(hypotheses, history):
    """Reference rescan: does not use the incremental update implementation."""
    return [sum(prediction(h, x) != y for x, y in history) for h in hypotheses]


def survivors_mask(hypotheses, counts):
    return sum(1 << h for h, count in zip(hypotheses, counts) if count == 0)


class CorrectionSummary:
    """Fixed-H mismatch counts. Event contents/IDs are NOT stored here."""

    def __init__(self, hypotheses=FULL_H):
        self.hypotheses = tuple(hypotheses)
        self.counts = [0] * len(self.hypotheses)
        self.events = 0

    def append(self, x, y):
        for i, h in enumerate(self.hypotheses):
            self.counts[i] += prediction(h, x) != y
        self.events += 1

    def replace_label(self, old_x, old_y, new_y):
        # The sender must supply the actual old event, not just its ID.
        next_counts = [count - int(prediction(h, old_x) != old_y)
                       + int(prediction(h, old_x) != new_y)
                       for h, count in zip(self.hypotheses, self.counts)]
        if any(count < 0 or count > self.events for count in next_counts):
            raise ValueError("Invalid correction for these counts")
        self.counts = next_counts

    def snapshot(self):
        return {"hypotheses": list(self.hypotheses),
                "counts": list(self.counts), "events": self.events}

    @classmethod
    def restore(cls, state):
        result = cls(state["hypotheses"])
        result.counts = list(state["counts"])
        result.events = state["events"]
        return result


class ObservationHistogram:
    """Strong control: six bin counts suffice on the fixed three-point domain."""

    def __init__(self):
        self.bins = [[0, 0] for _ in range(3)]

    def append(self, x, y):
        self.bins[x][y] += 1

    def replace_label(self, old_x, old_y, new_y):
        if self.bins[old_x][old_y] < 1:
            raise ValueError("Old observation absent")
        self.bins[old_x][old_y] -= 1
        self.bins[old_x][new_y] += 1

    def counts_for(self, hypotheses):
        return [sum(self.bins[x][1 - prediction(h, x)] for x in range(3))
                for h in hypotheses]


def version_space(hypotheses, history):
    return [h for h in hypotheses
            if all(prediction(h, x) == y for x, y in history)]


def exact_counterexamples():
    hypotheses = (0, 3, 5)
    histories = [[(0, 0), (1, 0)], [(0, 0), (2, 0)]]
    before = [version_space(hypotheses, hist) for hist in histories]
    after = [version_space(hypotheses, [(0, 1), hist[1]]) for hist in histories]
    assert before == [[0], [0]] and after == [[5], [3]]
    prediction_witness = {
        "hypotheses_as_truth_tables": [[prediction(h, x) for x in range(3)]
                                       for h in hypotheses],
        "histories": histories, "event_id": 0,
        "same_old_event": [0, 0], "same_new_label": 1,
        "before_hypothesis_ids": before, "after_hypothesis_ids": after,
        "after_predictions": [[prediction(space[0], x) for x in range(3)]
                              for space in after],
    }
    # The full count vector is permutation invariant; an ID-only edit is not.
    histories = [[(0, 0), (1, 0)], [(1, 0), (0, 0)]]
    counts = [replay_counts(FULL_H, hist) for hist in histories]
    after = [version_space(FULL_H, [(hist[0][0], 1), hist[1]]) for hist in histories]
    assert counts[0] == counts[1] and after[0] != after[1]
    provenance_witness = {"histories": histories, "same_full_counts": counts[0],
                          "same_event_id": 0, "same_new_label": 1,
                          "after_hypothesis_ids": after}
    # Counts sufficient for old H cannot in general evaluate an enlarged H.
    old_h = (0, 7)
    histories = [[(0, 0)] * 3, [(0, 0), (1, 0), (1, 0)]]
    old_counts = [replay_counts(old_h, hist) for hist in histories]
    added_h = 2  # truth table [0,1,0]
    new_counts = [replay_counts((added_h,), hist)[0] for hist in histories]
    assert old_counts == [[0, 3], [0, 3]] and new_counts == [0, 2]
    expansion_witness = {"old_hypotheses": list(old_h), "histories": histories,
                         "same_old_counts": old_counts[0],
                         "new_hypothesis_truth_table": [0, 1, 0],
                         "new_hypothesis_mismatch_counts": new_counts}
    return {"same_predictions_different_corrections": prediction_witness,
            "id_without_old_content_is_insufficient": provenance_witness,
            "new_hypothesis_can_need_discarded_information": expansion_witness}


def exhaustive_single_corrections():
    result = {"domain_size": 3, "all_boolean_hypotheses": 8,
              "nonempty_hypothesis_classes": 255, "history_lengths": [1, 2, 3, 4],
              "histories": 0, "full_universe_corrections": 0,
              "class_correction_cases": 0, "by_history_length": []}
    for n in range(1, 5):
        row = {"length": n, "histories": 6 ** n, "class_cases": 0,
               "exact_counter_mismatches": 0, "histogram_mismatches": 0,
               "survivors_only_mismatches": 0, "binary_violation_mismatches": 0,
               "nonempty_before": 0, "nonempty_after": 0,
               "nonempty_both": 0, "empty_before_nonempty_after": 0,
               "survivors_only_mismatches_nonempty_both": 0,
               "binary_false_admission_class_cases": 0}
        for history in itertools.product(OBSERVATIONS, repeat=n):
            summary = CorrectionSummary()
            histogram = ObservationHistogram()
            for x, y in history:
                summary.append(x, y)
                histogram.append(x, y)
            assert summary.counts == replay_counts(FULL_H, history)
            assert histogram.counts_for(FULL_H) == summary.counts
            before_mask = survivors_mask(FULL_H, summary.counts)
            for event_id in range(n):
                x, old_y = history[event_id]
                new_y = 1 - old_y  # enumerate all genuine binary label replacements
                corrected = list(history)
                corrected[event_id] = (x, new_y)
                gold_counts = replay_counts(FULL_H, corrected)
                candidate = CorrectionSummary.restore(summary.snapshot())
                candidate.replace_label(x, old_y, new_y)
                assert candidate.counts == gold_counts
                histogram.replace_label(x, old_y, new_y)
                histogram_counts = histogram.counts_for(FULL_H)
                assert histogram_counts == gold_counts
                histogram.replace_label(x, new_y, old_y)
                gold = survivors_mask(FULL_H, gold_counts)
                count_answer = survivors_mask(FULL_H, candidate.counts)
                histogram_answer = survivors_mask(FULL_H, histogram_counts)
                # Ablation 1: only keep previously surviving candidates.
                kept = before_mask & sum(1 << h for h in FULL_H
                                         if prediction(h, x) == new_y)
                # Ablation 2: store only whether each candidate was contradicted,
                # then try subtract-old/add-new. Multiplicity has been lost.
                binary_counts = [int(c > 0) - int(prediction(h, x) != old_y)
                                 + int(prediction(h, x) != new_y)
                                 for h, c in zip(FULL_H, summary.counts)]
                binary_answer = survivors_mask(FULL_H, binary_counts)
                result["full_universe_corrections"] += 1
                for h_class in range(1, 256):
                    expected = gold & h_class
                    old = before_mask & h_class
                    row["class_cases"] += 1
                    row["exact_counter_mismatches"] += (count_answer & h_class) != expected
                    row["histogram_mismatches"] += (histogram_answer & h_class) != expected
                    row["survivors_only_mismatches"] += (kept & h_class) != expected
                    row["binary_violation_mismatches"] += (binary_answer & h_class) != expected
                    row["nonempty_before"] += bool(old)
                    row["nonempty_after"] += bool(expected)
                    row["nonempty_both"] += bool(old and expected)
                    row["empty_before_nonempty_after"] += bool(not old and expected)
                    row["survivors_only_mismatches_nonempty_both"] += bool(
                        old and expected and (kept & h_class) != expected)
                    row["binary_false_admission_class_cases"] += bool(
                        binary_answer & h_class & ~expected)
        result["histories"] += row["histories"]
        result["class_correction_cases"] += row["class_cases"]
        result["by_history_length"].append(row)
    result["totals"] = {key: sum(row[key] for row in result["by_history_length"])
                        for key in result["by_history_length"][0]
                        if key not in ("length", "histories")}
    assert result["class_correction_cases"] == 1507050
    return result


def repeated_corrections():
    # All length-three correction-index sequences, including repeated same IDs.
    sequences = checkpoints = 0
    for n in range(1, 4):
        for history in itertools.product(OBSERVATIONS, repeat=n):
            for ids in itertools.product(range(n), repeat=3):
                current = list(history)
                state = CorrectionSummary()
                for x, y in history:
                    state.append(x, y)
                for event_id in ids:
                    x, old_y = current[event_id]
                    state.replace_label(x, old_y, 1 - old_y)
                    current[event_id] = (x, 1 - old_y)
                    assert state.counts == replay_counts(FULL_H, current)
                    checkpoints += 1
                sequences += 1
    return {"all_three_edit_sequences": sequences,
            "intermediate_states_checked_against_replay": checkpoints,
            "exact_mismatches": 0}


def persistence_probe(bundle):
    state = CorrectionSummary.restore(bundle["state"])
    for x, old_y, new_y in bundle["corrections"]:
        state.replace_label(x, old_y, new_y)
    return state.snapshot()


def verify_fresh_process():
    rng = random.Random(27019)
    history = [rng.choice(OBSERVATIONS) for _ in range(80)]
    state = CorrectionSummary()
    for x, y in history:
        state.append(x, y)
    bundle = {"state": state.snapshot(), "corrections": []}
    for _ in range(50):
        event_id = rng.randrange(len(history))
        x, old_y = history[event_id]
        bundle["corrections"].append([x, old_y, 1 - old_y])
        history[event_id] = (x, 1 - old_y)
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--probe"],
                               input=json.dumps(bundle), text=True, capture_output=True,
                               check=True)
    restored = json.loads(completed.stdout)
    expected_counts = replay_counts(FULL_H, history)
    assert restored["counts"] == expected_counts
    assert restored == persistence_probe(bundle)
    return {"initial_events": 80, "future_corrections": 50,
            "fresh_process_exact": True,
            "checkpoint_contains": ["hypotheses", "counts", "events"],
            "continuation_supplies": ["old_x", "old_y", "new_y"],
            "final_counts": expected_counts}


def prior_files():
    paths = set()
    for base in (ROOT / "docs/research", ROOT / "research"):
        for directory in base.glob("living_learning*"):
            if directory.name == HERE.name:
                continue
            paths.update(p for p in directory.rglob("*") if p.is_file()
                         and "__pycache__" not in p.parts and p.suffix != ".pyc")
    return sorted(paths)


def check_preservation():
    manifest_path = RESULTS / "prior_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {str(p.relative_to(ROOT)).replace("\\", "/"):
                    hashlib.sha256(p.read_bytes()).hexdigest() for p in prior_files()}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    changed = [p for p, sha in manifest.items()
               if not (ROOT / p).exists()
               or hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != sha]
    assert not changed, changed
    return {"files_checked": len(manifest), "changed": changed}


def run_experiment():
    RESULTS.mkdir(parents=True, exist_ok=True)
    before = check_preservation()
    result = {"experiment": "correction_sufficient_state", "date": "2026-09-20",
              "scope": "fixed finite H; trusted identity-aware binary label corrections",
              "counterexamples": exact_counterexamples(),
              "exhaustive": exhaustive_single_corrections(),
              "repeated_edits": repeated_corrections(),
              "persistence": verify_fresh_process(),
              "preservation_before": before,
              "preservation_after": check_preservation()}
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--output", type=Path, default=RESULTS / "correction_state.json")
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(persistence_probe(json.load(sys.stdin))))
        return
    result = run_experiment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "totals": result["exhaustive"]["totals"],
                      "repeated_edits": result["repeated_edits"],
                      "preservation": result["preservation_after"]}, indent=2))


if __name__ == "__main__":
    main()
