# -*- coding: utf-8 -*-
"""AWA-3 verification: freeze manifest, results manifest and recomputation of the decisions (no git involved).

Usage (from the repository root):
    python research/attention_order_law_2026_11_13/run_verify.py --freeze [--results DIR]
        After Phase 0, once PROTOCOL.md no longer says "NOT FROZEN": write ``frozen_sets.json`` (digests of every
        test and validation set, floors h_k and followable fractions P_f) and ``freeze_manifest.sha256``: the
        study code tree, the two test files, the engine files the arms are built from, frozen_sets.json and the
        Phase-0 selection (learning rates, GO/NO-GO). Refuses to overwrite an existing freeze.
    python research/attention_order_law_2026_11_13/run_verify.py --check-freeze [--results DIR]
        Re-hash everything the freeze lists and regenerate the frozen sets; exit 1 on any difference. Every
        Phase-1 job depends on this check.
    python research/attention_order_law_2026_11_13/run_verify.py --manifest [--results DIR]
        Hash the code tree and every file under the results directory into <results>/manifest.sha256.
    python research/attention_order_law_2026_11_13/run_verify.py --check [--results DIR]
        Re-hash the files of <results>/manifest.sha256: missing or changed files fail, unlisted ones are reported.
    python research/attention_order_law_2026_11_13/run_verify.py --rerun [--results DIR] [--phase 1]
        --check, then recompute the decisions from the run records with analysis.compute_decisions and compare
        them with the stored decisions.json (numbers within 1e-9); the analysis code must have the hash recorded
        in decisions.json and, when a freeze exists, the hash listed in the freeze manifest.

Manifests are ``sha256sum``-compatible; labels are relative to the study folder, so
``cd research/attention_order_law_2026_11_13 && sha256sum -c freeze_manifest.sha256`` works.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

STUDY = pathlib.Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]

FREEZE_NAME = "freeze_manifest.sha256"
MANIFEST_NAME = "manifest.sha256"
FROZEN_SETS_NAME = "frozen_sets.json"
RESULTS_PREFIX = "results/"
SELECTION_LABEL = "results/phase0/selection.json"
SKIP_DIR_NAMES = frozenset({"__pycache__", ".pytest_cache", ".ipynb_checkpoints"})
SKIP_SUFFIXES = (".pyc", ".pyo", ".tmp", ".swp")
# never listed: the manifests themselves and the log of post-freeze deviations (it must stay writable)
UNLISTED = (FREEZE_NAME, MANIFEST_NAME, "AMENDMENTS.md")
TEST_FILES = ("tests/evolution/test_awa_study.py", "tests/evolution/test_awa_attention.py")
ENGINE_FILES = (
    "src/neural_network.py",
    "src/neural_network_module/ortak_katman_module/attention_manager_module/absorbing_walk.py",
    "src/neural_network_module/ortak_katman_module/attention_manager_module/multi_head_attention.py",
    "src/neural_network_module/ortak_katman_module/transformer_encoder_layer.py",
    "src/neural_network_module/ortak_katman_module/feed_forward_network.py",
    "src/neural_network_module/ortak_katman_module/rms_norm.py",
    "src/neural_network_module/dil_katmani_module/language_embedding.py",
    "src/neural_network_module/dil_katmani_module/positional_encoding.py",
    "src/neural_network_module/architecture_contracts.py",
    "model_management/config_schema.py",
    "model_management/model_initializer.py",
    "model_management/model_manager.py",
)
FLOAT_TOL = 1e-9
_LINE = re.compile(r"^([0-9a-f]{64})  (\S(?:.*\S)?)$")


def sha256_file(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _skipped(rel: pathlib.PurePath) -> bool:
    return any(p in SKIP_DIR_NAMES or p.startswith(".") for p in rel.parts) or rel.name.endswith(SKIP_SUFFIXES)


def code_files(study: pathlib.Path = STUDY, repo: pathlib.Path = REPO,
               results_dir: Optional[pathlib.Path] = None) -> List[Tuple[str, pathlib.Path]]:
    """(label, path): the study folder (without results/, the manifests and AMENDMENTS.md), the two test files
    and the engine files the arms are built from."""
    results_dir = (results_dir or study / "results").resolve()
    out = []
    for p in sorted(study.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(study)
        if _skipped(rel) or rel.parts[0] == "results" or p.resolve().is_relative_to(results_dir):
            continue
        if rel.as_posix() in UNLISTED:
            continue
        out.append((rel.as_posix(), p))
    for name in TEST_FILES + ENGINE_FILES:
        p = repo / name
        if p.is_file():
            out.append((pathlib.Path(os.path.relpath(p, study)).as_posix(), p))
    return sorted(out)


def result_files(results_dir: pathlib.Path, exclude: Sequence[pathlib.Path] = ()) -> List[Tuple[str, pathlib.Path]]:
    excluded = {e.resolve() for e in exclude}
    out = []
    if results_dir.is_dir():
        for p in sorted(results_dir.rglob("*")):
            rel = p.relative_to(results_dir)
            if p.is_file() and not _skipped(rel) and p.resolve() not in excluded and rel.parts[0] != "job_logs":
                out.append((RESULTS_PREFIX + rel.as_posix(), p))
    return out


def resolve_label(label: str, study: pathlib.Path, results_dir: pathlib.Path) -> pathlib.Path:
    if label.startswith(RESULTS_PREFIX):
        return results_dir / label[len(RESULTS_PREFIX):]
    if pathlib.PurePosixPath(label).is_absolute():
        raise ValueError(f"absolute label in manifest: {label!r}")
    return pathlib.Path(os.path.normpath(study / label))


def _write_manifest(out: pathlib.Path, entries: List[Tuple[str, pathlib.Path]]) -> int:
    entries = sorted(entries)
    labels = [label for label, _ in entries]
    if len(labels) != len(set(labels)):
        raise RuntimeError("duplicate manifest labels")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text("".join(f"{sha256_file(p)}  {label}\n" for label, p in entries), encoding="utf-8")
    os.replace(tmp, out)
    return len(entries)


def read_manifest(path: pathlib.Path) -> List[Tuple[str, str]]:
    entries = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        m = _LINE.match(line)
        if not m:
            raise ValueError(f"{path}:{lineno}: malformed manifest line")
        entries.append((m.group(1), m.group(2)))
    return entries


def check_manifest(manifest_path: pathlib.Path, results_dir: pathlib.Path, *, study: pathlib.Path = STUDY,
                   repo: pathlib.Path = REPO, with_results: bool = True) -> dict:
    """Re-hash the listed files; ok is False on missing or changed files or a malformed manifest."""
    report: Dict[str, Any] = {"manifest": str(manifest_path), "ok": False, "n_listed": 0, "missing": [],
                              "mismatched": [], "unlisted": []}
    if not manifest_path.is_file():
        report["error"] = "manifest not found"
        return report
    try:
        entries = read_manifest(manifest_path)
    except ValueError as exc:
        report["error"] = str(exc)
        return report
    listed = set()
    for digest, label in entries:
        listed.add(label)
        try:
            path = resolve_label(label, study, results_dir)
        except ValueError as exc:
            report["error"] = str(exc)
            report["missing"].append(label)
            continue
        if not path.is_file():
            report["missing"].append(label)
        elif sha256_file(path) != digest:
            report["mismatched"].append(label)
    scope = code_files(study, repo, results_dir)
    if with_results:
        scope += result_files(results_dir, exclude=(manifest_path,))
    report["unlisted"] = sorted(label for label, _ in scope if label not in listed)
    report["n_listed"] = len(entries)
    report["ok"] = not report["missing"] and not report["mismatched"] and "error" not in report
    return report


# ------------------------------------------------------------------------------------------ freeze
def frozen_sets() -> dict:
    from . import task

    return {"format": "awa3-frozen-sets", "sets": task.frozen_set_hashes(),
            "floors": {str(k): task.floor(k) for k in task.TEST_DEPTHS},
            "followable": {str(k): task.followable_fraction(k) for k in task.TEST_DEPTHS}}


def write_freeze(results_dir: pathlib.Path, *, study: pathlib.Path = STUDY, repo: pathlib.Path = REPO) -> dict:
    """Write frozen_sets.json and freeze_manifest.sha256; returns a small report."""
    out = study / FREEZE_NAME
    if out.exists():
        raise RuntimeError(f"{out} exists: the study is frozen. Deviations go to AMENDMENTS.md, not into a new freeze")
    protocol = (study / "PROTOCOL.md").read_text(encoding="utf-8")
    if "NOT FROZEN" in protocol:
        raise RuntimeError("PROTOCOL.md still says NOT FROZEN: close its open decisions and set its status first")
    selection = resolve_label(SELECTION_LABEL, study, results_dir)
    if not selection.is_file():
        raise RuntimeError(f"{selection} not found: the freeze records the Phase-0 selection")
    if json.loads(selection.read_text(encoding="utf-8")).get("extension_pending"):
        raise RuntimeError("the Phase-0 selection is not final (AWA extension pending)")
    sets_path = study / FROZEN_SETS_NAME
    tmp = sets_path.with_name(sets_path.name + ".tmp")
    tmp.write_text(json.dumps(frozen_sets(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, sets_path)
    entries = code_files(study, repo, results_dir) + [(SELECTION_LABEL, selection)]
    return {"manifest": str(out), "n_files": _write_manifest(out, entries)}


def check_freeze(results_dir: pathlib.Path, *, study: pathlib.Path = STUDY, repo: pathlib.Path = REPO) -> dict:
    """The freeze manifest still describes the code, the selection and the regenerated evaluation sets."""
    report = check_manifest(study / FREEZE_NAME, results_dir, study=study, repo=repo, with_results=False)
    report["sets_ok"] = False
    sets_path = study / FROZEN_SETS_NAME
    if sets_path.is_file():
        report["sets_ok"] = json.loads(sets_path.read_text(encoding="utf-8")) == json.loads(json.dumps(frozen_sets()))
    report["ok"] = bool(report["ok"] and report["sets_ok"] and not report["unlisted"])
    return report


# ------------------------------------------------------------------------------------------- rerun
def compare(stored: Any, fresh: Any, path: str = "$", out: Optional[List[str]] = None,
            tol: float = FLOAT_TOL) -> List[str]:
    """Differences between two JSON values; numbers compare within tol (absolute or relative)."""
    out = [] if out is None else out
    if isinstance(stored, dict) and isinstance(fresh, dict):
        for key in sorted(set(stored) | set(fresh)):
            if key not in stored:
                out.append(f"{path}.{key}: only in the recomputation")
            elif key not in fresh:
                out.append(f"{path}.{key}: only in the stored file")
            else:
                compare(stored[key], fresh[key], f"{path}.{key}", out, tol)
    elif isinstance(stored, list) and isinstance(fresh, list):
        if len(stored) != len(fresh):
            out.append(f"{path}: length {len(stored)} != {len(fresh)}")
        else:
            for i, (a, b) in enumerate(zip(stored, fresh)):
                compare(a, b, f"{path}[{i}]", out, tol)
    elif (isinstance(stored, (int, float)) and isinstance(fresh, (int, float))
          and not isinstance(stored, bool) and not isinstance(fresh, bool)):
        same = stored == fresh or (math.isnan(stored) and math.isnan(fresh))
        if not same and not abs(stored - fresh) <= tol * max(1.0, abs(stored), abs(fresh)):
            out.append(f"{path}: {stored!r} != {fresh!r}")
    elif stored != fresh:
        out.append(f"{path}: {stored!r} != {fresh!r}")
    return out


def rerun(results_dir: pathlib.Path, phase: str = "1", *, study: pathlib.Path = STUDY,
          repo: pathlib.Path = REPO) -> dict:
    from . import analysis as analyze

    report: Dict[str, Any] = {"manifest": check_manifest(results_dir / MANIFEST_NAME, results_dir, study=study,
                                                         repo=repo)}
    phase_dir = results_dir / f"phase{phase}"
    stored_path = phase_dir / "decisions.json"
    if not stored_path.is_file():
        report.update(ok=False, error=f"{stored_path} not found")
        return report
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    runs = analyze.load_runs(phase_dir)
    fresh = analyze.compute_decisions(runs, allow_truncated=bool(stored["runs"].get("truncated")))
    recorded_code = stored.get("analysis_code_sha256", {})
    live_code = analyze.analysis_code_sha256(study)
    report["decision_differences"] = compare({k: v for k, v in stored.items() if k != "analysis_code_sha256"}, fresh)
    report["analysis_code_matches_record"] = recorded_code == live_code
    freeze = study / FREEZE_NAME
    if freeze.is_file():
        frozen = {label: digest for digest, label in read_manifest(freeze)}
        report["analysis_code_matches_freeze"] = all(frozen.get(name) == digest for name, digest in recorded_code.items())
    else:
        report["analysis_code_matches_freeze"] = None
    report["ok"] = bool(report["manifest"]["ok"] and not report["decision_differences"]
                        and report["analysis_code_matches_record"]
                        and report["analysis_code_matches_freeze"] is not False)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--check-freeze", action="store_true")
    mode.add_argument("--manifest", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--rerun", action="store_true")
    ap.add_argument("--results", default=str(STUDY / "results"))
    ap.add_argument("--phase", default="1")
    args = ap.parse_args(argv)
    results_dir = pathlib.Path(args.results)
    if args.freeze:
        try:
            report = write_freeze(results_dir)
        except RuntimeError as exc:
            print(f"freeze refused: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(report))
        return 0
    if args.check_freeze:
        report = check_freeze(results_dir)
    elif args.manifest:
        out = results_dir / MANIFEST_NAME
        n = _write_manifest(out, code_files(STUDY, REPO, results_dir) + result_files(results_dir, exclude=(out,)))
        print(json.dumps({"manifest": str(out), "n_files": n}))
        return 0
    elif args.check:
        report = check_manifest(results_dir / MANIFEST_NAME, results_dir)
    else:
        report = rerun(results_dir, args.phase)
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1

