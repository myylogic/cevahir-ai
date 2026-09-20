"""Reproduce selected publication evidence without rewriting historical results.

The default is read-only. Use --output for a new, explicit verification record.
This is a selected audit, not a rerun of every experiment in the repository.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def module(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    imported = importlib.util.module_from_spec(spec)
    sys.modules[name] = imported
    spec.loader.exec_module(imported)
    return imported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "docs/publications/evidence/prior_manifest.json").read_text())
    def changed():
        return [path for path, sha in manifest["files"].items()
                if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != sha]
    assert not changed()
    new = module("research/living_learning_query_state_2026_09_21/query_state.py", "query_state_publication")
    fresh = json.loads(json.dumps(new.run()))
    assert fresh == json.loads(new.RESULT.read_text())
    old = module("research/living_learning_correction_2026_09_20/correction_state.py", "correction_publication")
    correction = json.loads(json.dumps(old.run_experiment()))
    assert correction == json.loads((old.RESULTS / "correction_state.json").read_text())
    tests = subprocess.run([sys.executable, "-m", "unittest", "tests.evolution.test_representation_witness_audit"],
                           cwd=ROOT, text=True, capture_output=True, check=True)
    assert not changed()
    verification = {
        "date": "2026-09-21", "python": platform.python_version(),
        "verifier_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prior_baseline_commit": manifest["baseline_commit"],
        "prior_files_checked_byte_exact": len(manifest["files"]), "prior_files_changed": changed(),
        "query_state_exact_rerun_equal": True, "query_state_totals": fresh["totals"],
        "query_state_source_sha256": fresh["source_sha256"],
        "correction_exact_rerun_equal": True,
        "correction_class_cases": correction["exhaustive"]["class_correction_cases"],
        "correction_fresh_process_continuation": correction["persistence"]["fresh_process_exact"],
        "witness_unittest_returncode": tests.returncode,
        "witness_unittest_output": (tests.stdout + tests.stderr).strip(),
        "historical_evidence": {
            "all_previous_experiments_rerun_in_this_publication_audit": False,
            "prior_reports_and_results_preserved": True,
            "book_previous_validation": "docs/book/evidence/verification.json"
        },
        "limits": ["Selected exact reruns plus witness tests; not a complete repository or GPU test suite",
                   "Prior byte preservation is not independent scientific replication",
                   "Internal automated verification, no external peer review or novelty certification"]
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
