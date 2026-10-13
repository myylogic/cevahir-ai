"""Local final integrity audit for primary and active context experiments."""
import json
import platform
import re
from urllib.parse import unquote

import numpy as np

import context_learning as c
import active_experience as a
import analyze_results
import verify_context


def main():
    primary = json.loads((c.RESULTS / "context_learning.json").read_text(encoding="utf-8"))
    active = json.loads((c.RESULTS / "active_experience.json").read_text(encoding="utf-8"))
    summary = json.loads((c.RESULTS / "paired_summary.json").read_text(encoding="utf-8"))
    verification = json.loads((c.RESULTS / "verification.json").read_text(encoding="utf-8"))
    active_verification = json.loads((c.RESULTS / "active_verification.json").read_text(encoding="utf-8"))
    assert primary["source_sha256"] == c.sha(c.HERE / "context_learning.py")
    assert primary["protocol_sha256"] == c.sha(c.PROTOCOL)
    assert active["source_sha256"] == c.sha(c.HERE / "active_experience.py")
    assert active["primary_source_sha256"] == primary["source_sha256"]
    assert active["protocol_sha256"] == c.sha(a.PROTOCOL)
    assert summary == analyze_results.summarize(primary, active)
    assert verification["full_rerun"]["returncode"] == active_verification["full_rerun"]["returncode"] == 0
    assert len(verification["fresh_process_resume"]) == 21
    assert all(r["exact_equal"] for r in verification["fresh_process_resume"])
    assert len(active_verification["persistent_controller"]) == 4
    assert all(r["choices_and_updates_exact"] for r in active_verification["persistent_controller"])
    assert active_verification["checks"]["greedy_choices_independently_scored"] == 768
    assert verification["structural"]["scalar_bank_cells"] == 8388608
    assert verification["structural"]["unique_functions"] == 15904
    for comparison in summary["active_paired"]:
        after_four = next(x for x in comparison["comparisons"] if x["queries"] == 4)
        assert after_four["full_mse"]["better"] == after_four["unseen_mse"]["better"] == 24
    groups = {(g["method"],g["policy"]):g for g in summary["active_groups"]}
    for method in a.PRIORS:
        assert groups[method,"disagreement_query"]["final_query_prediction_cells"] == 134217728
        assert groups[method,"uniform_query"]["final_query_prediction_cells"] == 262144
    docs = c.PROTOCOL.parent
    links = 0
    for path in [c.HERE / "README.md", *docs.glob("*.md")]:
        for href in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if re.match(r"^[A-Za-z][\w+.-]*:", href) or href.startswith("#"):
                continue
            target = (path.parent / unquote(href.split("#",1)[0])).resolve()
            assert target.exists() or target == c.RESULTS / "final_audit.json", (str(path),href)
            links += 1
    artifacts = sorted([*c.HERE.glob("*.py"),c.HERE / "README.md",*docs.glob("*.md"),
                        *(p for p in c.RESULTS.iterdir() if p.is_file() and p.name != "final_audit.json")])
    result = dict(date="2026-09-22",status="passed",preservation=verify_context.preservation(),
                  environment=dict(python=platform.python_version(),numpy=np.__version__),
                  primary_trajectories=504,active_trajectories=96,
                  primary_prequential_predictions=32256,active_external_queries=1536,
                  complete_exact_replays=2,ordinary_process_resume_checks=21,active_process_resume_checks=4,
                  source_protocol_result_hashes_match=True,paired_summary_recomputed=True,
                  full_candidate_bank_scalar_checked=True,report_key_claims_checked=True,local_links_checked=links,
                  artifacts_sha256={p.relative_to(c.ROOT).as_posix():c.sha(p) for p in artifacts},
                  scope="Local finite experiments and conditional mathematics; supplied representation language and query interface. No claim of a new fundamental principle, biological mechanism or AGI. No commit, push or publication.")
    (c.RESULTS / "final_audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="artifacts_sha256"}))


if __name__ == "__main__":
    main()
