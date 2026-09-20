"""Repeat the exact audit, test boundaries, and verify preserved prior evidence."""
import hashlib
import json
from pathlib import Path

import correction_state as experiment


def main():
    result_path = experiment.RESULTS / "correction_state.json"
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    fresh = json.loads(json.dumps(experiment.run_experiment()))
    assert fresh == saved, "Exact rerun differs from recorded experiment"

    # Follow the preceding preservation chain as well as this turn's manifest.
    previous = json.loads((experiment.ROOT / "research/living_learning_state_2026_09_20/results/prior_manifest.json").read_text(encoding="utf-8"))
    combined = {path: metadata["sha256"] for path, metadata in previous["files"].items()}
    current = json.loads((experiment.RESULTS / "prior_manifest.json").read_text(encoding="utf-8"))
    for path, digest in current.items():
        assert path not in combined or combined[path] == digest, path
        combined[path] = digest
    changed = [path for path, digest in combined.items()
               if not (experiment.ROOT / path).is_file()
               or hashlib.sha256((experiment.ROOT / path).read_bytes()).hexdigest() != digest]
    assert not changed, changed

    # The arithmetic guard catches impossible counts, but it is not provenance.
    empty = experiment.CorrectionSummary()
    try:
        empty.replace_label(0, 0, 1)
    except ValueError:
        rejected_empty_edit = True
    else:
        raise AssertionError("Editing an empty history must be rejected")

    no_op = experiment.CorrectionSummary()
    no_op.append(1, 0)
    before = no_op.snapshot()
    no_op.replace_label(1, 0, 0)
    assert no_op.snapshot() == before

    # A wrong event packet can pass the numeric guard: retain this limitation.
    state = experiment.CorrectionSummary()
    actual = [(0, 0), (1, 0)]
    for x, y in actual:
        state.append(x, y)
    state.replace_label(1, 0, 1)  # purported event 0 actually has x=0
    intended_counts = experiment.replay_counts(experiment.FULL_H, [(0, 1), (1, 0)])
    assert state.counts != intended_counts

    verification = {
        "exact_rerun_equal": True,
        "experiment_source_sha256": saved["source_sha256"],
        "verifier_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "class_correction_cases": fresh["exhaustive"]["class_correction_cases"],
        "intermediate_states": fresh["repeated_edits"]["intermediate_states_checked_against_replay"],
        "fresh_process_continuation": fresh["persistence"]["fresh_process_exact"],
        "empty_history_edit_rejected": rejected_empty_edit,
        "no_op_edit_unchanged": True,
        "wrong_provenance_can_pass_count_guard": True,
        "prior_chain_files_checked": len(combined),
        "prior_files_changed": changed,
        "limits": ["Fixed finite H and trusted old event contents",
                   "Enumeration reuses projected subsets, not independent trials",
                   "No general learning, automatic error diagnosis, or performance superiority claim"],
    }
    output = experiment.RESULTS / "verification.json"
    output.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
