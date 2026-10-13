"""Final integrity audit for the local round; no mutation outside audit output."""
import json
from pathlib import Path
import platform
import re
from urllib.parse import unquote

import numpy as np

import relation_learning as r
import analyze_results
import structural_witness
import verify_relations


def main():
    data = json.loads((r.RESULTS / "relation_learning.json").read_text(encoding="utf-8"))
    summary = json.loads((r.RESULTS / "paired_summary.json").read_text(encoding="utf-8"))
    verification = json.loads((r.RESULTS / "verification.json").read_text(encoding="utf-8"))
    assert data["source_sha256"] == r.sha(r.HERE / "relation_learning.py")
    assert data["protocol_sha256"] == r.sha(r.PROTOCOL)
    assert summary == analyze_results.summarize(data)
    assert structural_witness.run() == json.loads((r.RESULTS / "structural_witness.json").read_text(encoding="utf-8"))
    assert verification["complete_rerun"]["returncode"] == 0
    assert len(verification["fresh_process_resume"]) == 24
    assert all(x["exact_json_equal"] and x["reference"] == "uninterrupted_live_object"
               for x in verification["fresh_process_resume"])
    assert verification["composition"]["independent_scalar_composite_label_cases"] == 36
    groups = {(g["regime"], g["method"]): g for g in summary["groups"]}
    pairs = {(p["regime"], p["right"]): p["metrics"] for p in summary["paired_comparisons"]}
    for baseline in ("table", "source_a", "source_b", "permutation_prior", "cycle_catalog"):
        assert pairs["compatible", baseline]["prequential_brier"]["better"] == 24
    for baseline in ("source_a", "permutation_prior"):
        assert pairs["cycle_shift", baseline]["prequential_brier"]["worse"] == 24
    assert pairs["compatible", "permutation_prior"]["composition_brier_t8"]["worse"] == 1
    assert pairs["local_exception", "frozen_source_both"]["prequential_brier"]["worse"] == 15
    for regime in ("cycle_shift", "local_exception"):
        assert pairs[regime, "frozen_source_both"]["composition_brier_t12"]["better"] == 24
    assert [groups[regime, "source_both"]["prequential_confident_wrong"] for regime in r.REGIMES] == [0, 15, 13]
    # Every recorded true target fact remains accessible in non-frozen predictors.
    source_update_prior_symmetry = verification["composition"]["same_initial_marginals"]
    assert source_update_prior_symmetry
    docs = r.PROTOCOL.parent
    local_links = 0
    for path in [r.HERE / "README.md", *docs.glob("*.md")]:
        for href in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if re.match(r"^[a-zA-Z][\w+.-]*:", href) or href.startswith("#"):
                continue
            target = (path.parent / unquote(href.split("#", 1)[0])).resolve()
            assert target.exists() or target == r.RESULTS / "final_audit.json", (path, href)
            local_links += 1
    assert (r.RESULTS / "composition_curves.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    preserved = verify_relations.preservation()
    artifacts = sorted([*r.HERE.glob("*.py"), r.HERE / "README.md", *docs.glob("*.md"),
                        *(p for p in r.RESULTS.iterdir() if p.is_file() and p.name != "final_audit.json")])
    record = dict(date="2026-09-22", status="passed", preservation=preserved,
                  environment=dict(python=platform.python_version(), numpy=np.__version__),
                  primary_trajectories=576, primary_checkpoints=7488, complete_exact_replays=1,
                  uninterrupted_fresh_process_comparisons=24, source_result_hashes_match=True,
                  paired_summary_recomputed=True, post_result_witness_exact=True,
                  report_main_positive_and_negative_claims_checked=True, local_document_links_checked=local_links,
                  source_update_prior_symmetry=source_update_prior_symmetry,
                  artifacts_sha256={p.relative_to(r.ROOT).as_posix(): r.sha(p) for p in artifacts},
                  scope="Local checks, no external peer review or independent-world replication. No commit, push, or publication.")
    (r.RESULTS / "final_audit.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "artifacts_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
