"""Reproduce recorded scientific outputs in temporary files and verify provenance.

This verifies reproducibility, not general validity of any hypothesis. The
substantive causal/negative controls live in each experiment. Standard library.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DOCS = ROOT / "docs" / "research" / "living_learning_2026_09_20"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scientific(payload):
    # Wall time and Python version are provenance, not experimental outcomes.
    return {k: v for k, v in payload.items() if k not in ("elapsed_seconds", "python")}


def main():
    jobs = [("affine_lifecycle", "affine_lifecycle.json"),
            ("adaptation_experiment", "adaptation_results.json"),
            ("closed_loop_experiment", "closed_loop.json"),
            ("unknown_change_experiment", "unknown_change.json")]
    verified = []
    with tempfile.TemporaryDirectory(prefix="cevahir_reproduce_") as directory:
        for module, filename in jobs:
            source = HERE / f"{module}.py"
            recorded_path = HERE / "results" / filename
            recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
            assert recorded["source_sha256"] == digest(source), f"stale source provenance: {module}"
            target = Path(directory) / filename
            started = time.perf_counter()
            proc = subprocess.run([sys.executable, "-m", f"research.living_learning.{module}",
                                   "--output", str(target)], cwd=ROOT,
                                  capture_output=True, text=True, timeout=45)
            assert proc.returncode == 0, proc.stderr
            regenerated = json.loads(target.read_text(encoding="utf-8"))
            assert scientific(recorded) == scientific(regenerated), f"scientific reproduction differs: {module}"
            verified.append({"module": module, "source_sha256": digest(source),
                             "recorded_result_sha256": digest(recorded_path),
                             "scientific_output_equal": True,
                             "byte_identical": recorded_path.read_bytes() == target.read_bytes(),
                             "elapsed_seconds": time.perf_counter() - started})
    missing = []
    count = 0
    for file in DOCS.glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", file.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            count += 1
            candidate = (file.parent / target.split("#")[0]).resolve()
            # verification.json is about to be written by this exact process.
            if not candidate.exists() and candidate != HERE / "results" / "verification.json":
                missing.append({"file": str(file.relative_to(ROOT)), "target": target})
    assert not missing, missing
    tracked_diff = subprocess.run(["git", "diff", "--name-only", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True, check=True).stdout.splitlines()
    report = {"python": platform.python_version(), "experiments": verified,
              "local_document_links_checked": count, "missing_local_links": missing,
              "tracked_files_changed_relative_to_head": tracked_diff,
              "scope": "Four reproducible finite simulations; not a general solution, biological model or main-engine capability evaluation.",
              "verification_source_sha256": digest(Path(__file__))}
    output = HERE / "results" / "verification.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
