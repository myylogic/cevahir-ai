"""Final local integrity and report-claim audit; writes only final_audit.json."""
import json
import platform
import re
from urllib.parse import unquote

import numpy as np

import consolidation as c
import analyze_results
import exact_witnesses
import verify_consolidation


def main():
    data = json.loads((c.RESULTS / "consolidation.json").read_text(encoding="utf-8"))
    summary = json.loads((c.RESULTS / "paired_summary.json").read_text(encoding="utf-8"))
    verification = json.loads((c.RESULTS / "verification.json").read_text(encoding="utf-8"))
    assert data["source_sha256"] == c.sha(c.HERE / "consolidation.py")
    assert data["protocol_sha256"] == c.sha(c.PROTOCOL)
    assert summary == analyze_results.summarize(data)
    assert c.encode(exact_witnesses.run()) == c.encode(json.loads((c.RESULTS / "exact_witnesses.json").read_text(encoding="utf-8")))
    assert verification["full_rerun"]["returncode"] == 0
    assert len(verification["fresh_process_resume"]) == 24
    assert all(item["exact_equal"] and not item["raw_episodes_present"]
               and item["reference"] == "uninterrupted_live_object" for item in verification["fresh_process_resume"])
    assert verification["structure"]["exported_and_restored_states_are_isolated"]
    groups = {(g["regime"], g["method"]): g for g in summary["groups"]}
    comparisons = {(p["regime"], p["other"]): p for p in summary["comparisons"]}
    for regime in c.REGIMES:
        for key in ("final_old_mse", "final_unseen_mse"):
            assert comparisons[regime, "online_rich"][key]["tied"] == 24
    for method in ("sleep_rich_replay", "online_rich", "sleep_rich_recent"):
        assert groups["visible_context", method]["exact_old_lives"] == 24
        assert groups["visible_context", method]["exact_new_lives"] == 24
        assert groups["noisy_day", method]["exact_new_lives"] == 23
    for method in ("frozen_core", "cache_only", "sleep_linear_replay", "sleep_self_replay", "sleep_mispaired_replay"):
        assert groups["visible_context", method]["exact_new_lives"] == 0
    assert comparisons["visible_context", "online_rich"]["prequential_true_mse"]["worse"] == 24
    assert comparisons["noisy_day", "online_rich"]["prequential_true_mse"]["worse"] == 24
    rich = groups["visible_context", "sleep_rich_replay"]
    assert rich["mean_final_resources"]["learning_checks"] == 4032
    assert groups["visible_context", "sleep_rich_recent"]["mean_final_resources"]["learning_checks"] == 2016
    assert abs(groups["hidden_context", "sleep_rich_replay"]["snapshots"][-1]["core"]["balanced_mse"]-.21875) < 1e-14
    docs = c.PROTOCOL.parent
    hypotheses = (docs / "HYPOTHESES_TR.md").read_text(encoding="utf-8")
    hypothesis_ids = [int(x) for x in re.findall(r"^\| \*\*H(\d+) —", hypotheses, flags=re.MULTILINE)]
    assert hypothesis_ids == list(range(18, 42))
    local_links = 0
    for path in [c.HERE / "README.md", *docs.glob("*.md")]:
        for href in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if re.match(r"^[a-zA-Z][\w+.-]*:", href) or href.startswith("#"):
                continue
            target = (path.parent / unquote(href.split("#", 1)[0])).resolve()
            assert target.exists() or target == c.RESULTS / "final_audit.json", (str(path), href)
            local_links += 1
    preserved = verify_consolidation.preservation()
    artifacts = sorted([*c.HERE.glob("*.py"), c.HERE / "README.md", *docs.glob("*.md"),
                        *(p for p in c.RESULTS.iterdir() if p.is_file() and p.name != "final_audit.json")])
    result = dict(date="2026-09-22", status="passed", preservation=preserved,
                  environment=dict(python=platform.python_version(), numpy=np.__version__),
                  trajectories=576, snapshots=4032, prequential_predictions=13824,
                  source_result_protocol_hashes_match=True, paired_summary_recomputed=True,
                  complete_exact_replays=1, schedule_equal_pairs=72, uninterrupted_process_comparisons=24,
                  exact_rational_and_ambiguity_witnesses=True, isolated_state_snapshots_verified=True,
                  hypothesis_ids=hypothesis_ids, report_main_claims_checked=True,
                  local_document_links_checked=local_links,
                  artifacts_sha256={p.relative_to(c.ROOT).as_posix(): c.sha(p) for p in artifacts},
                  scope="Local conditional mathematical analysis and finite experiments; no biological mechanism, consciousness or AGI proof. No commit, push, publication or independent-world replication.")
    (c.RESULTS / "final_audit.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
