"""Small chapter-12 examples using the preserved finite research implementations.

This is not a repeat of every historical audit or a new learning benchmark.
Default mode compares with evidence; --write explicitly records these examples.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/book/evidence/research_walkthrough.json"
SOURCES = {
    "correction": ROOT / "research/living_learning_correction_2026_09_20/correction_state.py",
    "query_state": ROOT / "research/living_learning_query_state_2026_09_21/query_state.py",
}


def load(name, path):
    spec = importlib.util.spec_from_file_location("book_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def correction_example(module):
    hypotheses = (0, 3, 5)
    histories = [[(0, 0), (1, 0)], [(0, 0), (2, 0)]]
    before, after, updated_counts = [], [], []
    for history in histories:
        summary = module.CorrectionSummary(hypotheses)
        for x, y in history:
            summary.append(x, y)
        before.append(module.version_space(hypotheses, history))
        summary.replace_label(0, 0, 1)
        edited = [(0, 1), history[1]]
        assert summary.counts == module.replay_counts(hypotheses, edited)
        restored = module.CorrectionSummary.restore(summary.snapshot())
        assert restored.counts == summary.counts
        after.append(module.version_space(hypotheses, edited))
        updated_counts.append(summary.counts)
    assert before == [[0], [0]] and after == [[5], [3]]
    return {"hypotheses_by_integer_id": list(hypotheses),
            "truth_tables_in_input_order": [[module.prediction(h, x) for x in range(3)] for h in hypotheses],
            "histories": histories, "survivors_before": before, "survivors_after_same_edit": after,
            "counts_after": updated_counts, "incremental_equals_replay": True,
            "restored_counts_equal": True,
            "assumption": "The actual old event contents (0,0) are supplied externally."}


def query_boundary_example(module):
    controls = module.negative_controls()
    hypotheses = ((0, 0), (0, 1), (1, 0), (1, 1))
    histories = controls["signed_collision"]
    states = [module.signed(h, 2) for h in histories]
    losses = [module.query(s, hypotheses) for s in states]
    histograms = [module.histogram(h, 2) for h in histories]
    assert states[0] == states[1] and losses == [(1, 1, 1, 1)] * 2
    assert histograms[0] != histograms[1]
    return {"histories": histories, "signed_states": states, "all_boolean_loss_counts": losses,
            "observation_histograms": histograms, "negative_controls": controls,
            "meaning": "Same answers for every allowed loss query do not imply the same answers for observation-count or event-identity queries."}


def finite_cell_example(module):
    result = module.audit_cell(3, 4)
    assert result["histories"] == 1296
    assert [result[k] for k in ("risk_classes", "histogram_classes", "id_classes")] == [85, 126, 1296]
    assert result["ideal_fixed_bits"] == {"risk": 7, "histogram": 7, "id": 11}
    assert result["query_mismatches"] == result["formula_count_mismatches"] == 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    before = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in SOURCES.items()}
    correction, query_state = [load(name, p) for name, p in SOURCES.items()]
    cases = {"correction": correction_example(correction),
             "query_boundary": query_boundary_example(query_state),
             "finite_cell": finite_cell_example(query_state)}
    after = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in SOURCES.items()}
    assert before == after
    # JSON normalization makes tuple/list representation irrelevant to the check.
    cases = json.loads(json.dumps(cases))
    if args.write:
        record = {"date": "2026-09-21", "environment": {"python": platform.python_version()},
                  "cases": cases, "research_sources_unchanged": True,
                  "scope": "Three small teaching example groups; one m=3,N=4 audit cell, not the full historical audit.",
                  "limits": ["Exact finite counts, not language-model quality or open-world learning",
                             "Fixed code length bounds do not measure Python allocation or produce an efficient encoder",
                             "No historical source or result files overwritten"]}
        OUTPUT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    else:
        assert cases == json.loads(OUTPUT.read_text(encoding="utf-8"))["cases"]
    print(json.dumps({"example_groups": len(cases), "finite_histories_checked": cases["finite_cell"]["histories"],
                      "research_sources_unchanged": True, "mode": "record" if args.write else "check"}))


if __name__ == "__main__":
    main()
