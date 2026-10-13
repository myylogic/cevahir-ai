"""Check this research package without rewriting previous verification/results.

Writes only results/final_audit.json. This is provenance and reproducibility
checking, not a statistical validation on independent worlds.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

import geometry as g
from verify_geometry import preservation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOCS = ROOT / "docs/research/living_learning_geometry_2026_09_21"
RESULTS = HERE / "results"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_links(allow_pending_audit=False):
    checked = 0
    for path in [HERE / "README.md", *sorted(DOCS.glob("*.md"))]:
        for href in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if re.match(r"^[A-Za-z][\w+.-]*:", href) or href.startswith("#"):
                continue
            target = (path.parent / unquote(href.split("#", 1)[0])).resolve()
            pending = allow_pending_audit and target == RESULTS / "final_audit.json"
            assert target.exists() or pending, (str(path), href)
            checked += 1
    return checked


def main():
    primary = json.loads((RESULTS / "geometry.json").read_text(encoding="utf-8"))
    post = json.loads((RESULTS / "posthoc_controls.json").read_text(encoding="utf-8"))
    endpoint = json.loads((RESULTS / "endpoint_diagnostic.json").read_text(encoding="utf-8"))
    verification = json.loads((RESULTS / "verification.json").read_text(encoding="utf-8"))
    assert primary["source_sha256"] == digest(HERE / "geometry.py")
    assert post["primary_source_sha256"] == primary["source_sha256"]
    assert post["source_sha256"] == digest(HERE / "posthoc_controls.py")
    assert len(primary["runs"]) == 864 and len(post["runs"]) == 216
    assert len({(r["seed"], r["anchors"], r["rate"], r["mode"], r["rcond"]) for r in primary["runs"]}) == 864
    assert len({(r["seed"], r["anchors"], r["rate"], r["mode"]) for r in post["runs"]}) == 216
    assert set(r["seed"] for r in primary["runs"]) == set(range(12))
    for r in primary["runs"]:
        if r["mode"].endswith("_guard"):
            assert r["max_anchor_drift_mse"] <= g.DRIFT_BUDGET + 1e-15
    for r in post["runs"]:
        if r["mode"].endswith("_guard"):
            assert r["max_anchor_label_mse_increase"] <= g.DRIFT_BUDGET + 1e-15
    assert len(primary["continuation_probes"]) == 36
    assert all(r["initial_max_prediction_difference"] == 0 for r in primary["continuation_probes"])
    assert endpoint["joint_endpoints_inside_label_budget"] == 10
    assert endpoint["joint_paths_violating_label_budget"] == 12
    assert endpoint["teacher_endpoints_violating_output_budget"] == 12
    # Same optimizer's final points must agree with the separately stored reference.
    for ep, joint in zip(endpoint["runs"], post["offline_joint_references"], strict=True):
        assert ep["seed"] == joint["seed"]
        for metric in ("old_mse", "new_mse", "bridge_mse"):
            g.compare(ep[metric], joint[metric])
    assert len(verification["complete_reruns"]) == 2, "Run verify_geometry.py --rerun first"
    assert all(r["returncode"] == 0 for r in verification["complete_reruns"])
    assert len(verification["fresh_process_continuation"]) == 9
    assert all(r["exact_json_equality"] for r in verification["fresh_process_continuation"])
    rerun = subprocess.run([sys.executable, str(HERE / "endpoint_diagnostic.py")],
                           text=True, capture_output=True)
    assert rerun.returncode == 0, (rerun.stdout, rerun.stderr)
    # Allow the audit link to resolve on its first run without creating a stub.
    links = local_links(allow_pending_audit=True)
    files = sorted([*HERE.glob("*.py"), HERE / "README.md", *DOCS.glob("*.md"),
                    *(p for p in RESULTS.glob("*.json") if p.name != "final_audit.json")])
    record = {
        "date": "2026-09-21", "status": "passed", "preservation": preservation(),
        "source_result_hashes_match": True,
        "unique_primary_conditions": 864, "unique_posthoc_conditions": 216,
        "distinct_lives": 12, "local_document_links_checked": links,
        "fresh_process_main_methods_exact": 9, "prior_full_reruns": 2,
        "continuation_pairs_initially_equal": 36,
        "endpoint_diagnostic_reproduction": json.loads(rerun.stdout),
        "offline_endpoint_and_reference_agree": True,
        "artifacts_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in files},
        "scope": "Local provenance and reproducibility; not external peer review or independent-world validation. No book, prior research, runtime, commit or publication changes."
    }
    (RESULTS / "final_audit.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    assert local_links() == links
    print(json.dumps({k: record[k] for k in ("status", "preservation", "distinct_lives", "local_document_links_checked", "unique_primary_conditions", "unique_posthoc_conditions")}))


if __name__ == "__main__":
    main()
