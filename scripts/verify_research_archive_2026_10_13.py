"""Reproducibility and preservation check for the 2026-10-13 release note.

This deliberately does not call the historical campaign ``verify`` action: that
action writes its own historical result and assumes the old Git HEAD.  The
check here is read-only apart from the new evidence record created by the
command's ``--write`` mode.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "research/living_learning_campaign_2026_09_24"
WAVE2 = ROOT / "research/living_learning_campaign_wave2_2026_09_24"
EVIDENCE = ROOT / "docs/publications/evidence/research_validation_2026_10_13.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(args: list[str]) -> dict:
    p = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return {"command": args, "returncode": p.returncode,
            "stdout": p.stdout[-2000:], "stderr": p.stderr[-2000:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write the new evidence record after all checks pass")
    args = ap.parse_args()
    sys.path.insert(0, str(CAMPAIGN))
    import campaign  # imported after path setup; historical sources are read only

    registry, records = campaign.load_all()
    recomputed = campaign.analyze(write=False)
    saved_summary_path = CAMPAIGN / "results/summary.json"
    saved_summary = json.loads(saved_summary_path.read_text(encoding="utf-8"))
    if campaign.encoded(recomputed) != campaign.encoded(saved_summary):
        raise AssertionError("Recomputed campaign summary differs from preserved summary.json")
    statistics_selfcheck = campaign.stats_selfcheck()
    exact = 0
    for row in records:
        module = __import__(row["engine"])
        if campaign.encoded(module.run(row["seed"])) != campaign.encoded(row["result"]):
            raise AssertionError((row["engine"], row["phase"], row["seed"]))
        exact += 1

    baseline_path = WAVE2 / "results/preservation_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
    baseline_files = baseline["files"]
    if isinstance(baseline_files, list):
        baseline_files = {item["path"]: item["sha256"] for item in baseline_files}
    changed = [name for name, digest in baseline_files.items()
               if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    # The release editor may revise publication metadata and generated paper
    # editions.  All other historical research and implementation paths remain
    # immutable.  Keeping this allow-list explicit prevents a release edit from
    # silently masking an experiment change.
    mutable_release = {
        ".zenodo.json", "CITATION.cff", "docs/publications/catalog.json",
        "docs/publications/release.json", "docs/publications/README.md",
        "docs/book/evidence/reviewed_sources.json",
    }
    mutable_release.update({
        f"docs/publications/papers/{n:02d}-{slug}.md"
        for n, slug in (
            (1, "experience-conditioned-computation"),
            (2, "representation-witness-negative"),
            (3, "persistence-transfer-retesting"),
            (4, "representation-credit-state-transport"),
            (5, "joint-lifetime-temporal-relations"),
            (6, "learned-update-transfer"),
            (7, "acquired-recurrent-state"),
            (8, "correctable-learning-state"),
            (9, "future-query-state-classes"),
        )
    })
    immutable_changed = [name for name in changed if name not in mutable_release]
    if immutable_changed:
        raise AssertionError({"immutable_baseline_changed_paths": immutable_changed})
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip()
    prior_zenodo = json.loads(
        (ROOT / "docs/publications/evidence/zenodo_archive_verification_v3.json")
        .read_text(encoding="utf-8"))
    publication_commands = [
        command([sys.executable, "scripts/check_book.py"]),
        command([sys.executable, "scripts/build_publications.py"]),
    ]
    if any(item["returncode"] != 0 for item in publication_commands):
        raise AssertionError({"publication_commands": publication_commands})

    wave2_files = [p for p in WAVE2.rglob("*") if p.is_file()]
    result = {
        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "read-only reproducibility check; no historical verify action invoked",
        "historical_baseline_head": baseline.get("base_head"),
        "current_head_at_check": head,
        "preservation_baseline_files": len(baseline_files),
        "immutable_baseline_paths_unchanged_at_check": not immutable_changed,
        "immutable_baseline_paths_count": len(baseline_files) - len(changed),
        "file_hashes_sha256": {
            "baseline": sha(baseline_path),
            "preserved_summary": sha(saved_summary_path),
            "verifier_script": sha(Path(__file__)),
        },
        "release_metadata_paths_changed_after_baseline": sorted(
            name for name in changed if name in mutable_release),
        "campaign": {
            "catalog_entries": len(registry),
            "executed_claims": recomputed["executed_claims"],
            "queued_claims": recomputed["queued_claims"],
            "engine_lifetimes": len(records),
            "paired_outcomes": recomputed["paired_outcomes"],
            "decisions": recomputed["decisions"],
            "exact_deterministic_reruns": exact,
            "summary_recomputed_without_write": True,
            "summary_equals_preserved_summary": True,
            "statistics_selfcheck": statistics_selfcheck,
            "freeze_sha256": campaign.frozen_check(),
        },
        "wave2": {
            "directory": str(WAVE2.relative_to(ROOT)).replace("\\", "/"),
            "files_present": len(wave2_files),
            "performance_results_present": any("results" in p.parts and p.name != "preservation_baseline.json" for p in wave2_files),
            "scope_note": "Wave 2 has only CONTRACT.md and preservation_baseline.json at this check; no performance outcomes are claimed.",
        },
        "publication_checks": {
            "commands": publication_commands,
            "zenodo_evidence_file": "docs/publications/evidence/zenodo_archive_verification_v3.json",
            "existing_published_record": {
                "doi": prior_zenodo.get("doi"),
                "record_url": prior_zenodo.get("zenodo_record_url"),
                "version": prior_zenodo.get("version_verified"),
            },
            "current_release_archive": "No new Zenodo v4 record was verified by this local check; v4 remains a local/GitHub metadata candidate until Zenodo publishes it.",
        },
        "interpretation": "Exact replay establishes deterministic reproducibility, not independent implementation, external replication, peer review, or a general living-learning theorem.",
    }
    if args.write:
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
