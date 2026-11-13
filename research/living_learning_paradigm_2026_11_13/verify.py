# -*- coding: utf-8 -*-
"""CEV-11 verification: SHA-256 manifest and recomputation of the decisions (independent of git).

Usage (OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1):
    python verify.py --manifest [--code-only] [--results DIR] [--out PATH]
        Hash the code tree (every file of the study folder outside results/, plus
        tests/evolution/test_cev11_*.py) and, unless --code-only, every file under the results directory, and write
        a `sha256sum`-compatible manifest (default <results>/manifest.sha256). Labels are relative to the study
        folder ("analyze.py", "cev11/stats.py", "../../tests/evolution/test_cev11_stats.py",
        "results/decisions.json"), so `cd <study> && sha256sum -c results/manifest.sha256` works with the default
        layout. --code-only is the freeze manifest written before the first confirmatory run; its default output
        is <study>/freeze_manifest.sha256 (next to PROTOCOL.md, committed with it), so the later full manifest
        cannot overwrite it. The manifest file itself is never listed in it.
    python verify.py --check [--results DIR] [--manifest-path PATH]
        Re-hash every listed file: missing or changed files fail; files in scope but not listed are reported.
    python verify.py --rerun [--results DIR] [--manifest-path PATH] [--freeze PATH] [--report PATH]
        --check, then recompute the decisions and tables from the JSONL inputs with the options recorded in
        decisions.json (analyze.compute_decisions / render_outputs, nothing is overwritten) and compare them with
        the stored files (numbers within 1e-9), and check that the analysis code (analyze.py, cev11/stats.py) still
        has the hash recorded in decisions.json. With --freeze, the recorded analysis code must also equal the code
        listed in the code-only freeze manifest (PROTOCOL §11: analyze.py frozen by hash before any confirmatory
        hypothesis metric is computed). Exit code 0 when everything agrees, 1 otherwise. The figures and
        analysis_env.json are environment-dependent and only covered by the manifest hashes.

Writes only the manifest (--manifest) and an optional JSON report (--report), both inside the study folder or the
given results directory. No git command is used anywhere, so the check does not depend on git HEAD.
"""
from __future__ import annotations

import sys
import pathlib

STUDY = pathlib.Path(__file__).resolve().parent
REPO = STUDY.parents[1]
sys.path[:0] = [str(STUDY), str(REPO)]

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
from typing import Any  # noqa: E402

MANIFEST_NAME = "manifest.sha256"
FREEZE_NAME = "freeze_manifest.sha256"   # default output of --manifest --code-only (in the study folder)
RESULTS_PREFIX = "results/"
SKIP_DIR_NAMES = frozenset({"__pycache__", ".pytest_cache", ".ipynb_checkpoints"})
SKIP_SUFFIXES = (".pyc", ".pyo", ".tmp", ".swp")
TEST_GLOB = "test_cev11_*.py"
FLOAT_TOL = 1e-9
MAX_REPORTED_DIFFS = 50
_LINE = re.compile(r"^([0-9a-f]{64})  (\S(?:.*\S)?)$")


def sha256_file(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _skipped(rel: pathlib.PurePath) -> bool:
    return any(p in SKIP_DIR_NAMES or p.startswith(".") for p in rel.parts) or rel.name.endswith(SKIP_SUFFIXES)


def _inside(path: pathlib.Path, roots: tuple[pathlib.Path, ...]) -> bool:
    p = path.resolve()
    return any(p == r.resolve() or p.is_relative_to(r.resolve()) for r in roots)


def code_files(study: pathlib.Path = STUDY, repo: pathlib.Path = REPO,
               results_dir: pathlib.Path | None = None) -> list[tuple[str, pathlib.Path]]:
    """(label, path) of the code tree: the study folder minus results/, plus the CEV-11 test files."""
    results_dir = (results_dir or study / "results").resolve()
    out = []
    for p in sorted(study.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(study)
        if _skipped(rel) or rel.parts[0] == "results" or p.resolve().is_relative_to(results_dir):
            continue
        out.append((rel.as_posix(), p))
    tests = repo / "tests" / "evolution"
    if tests.is_dir():
        for p in sorted(tests.glob(TEST_GLOB)):
            out.append((pathlib.Path(os.path.relpath(p, study)).as_posix(), p))
    return sorted(out)


def result_files(results_dir: pathlib.Path, exclude: tuple[pathlib.Path, ...] = ()) -> list[tuple[str, pathlib.Path]]:
    """(label, path) of every file under the results directory, labels prefixed with 'results/'."""
    excl = {e.resolve() for e in exclude}
    out = []
    if not results_dir.is_dir():
        return out
    for p in sorted(results_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(results_dir)
        if _skipped(rel) or p.resolve() in excl:
            continue
        out.append((RESULTS_PREFIX + rel.as_posix(), p))
    return sorted(out)


def resolve_label(label: str, study: pathlib.Path, results_dir: pathlib.Path) -> pathlib.Path:
    if label.startswith(RESULTS_PREFIX):
        return results_dir / label[len(RESULTS_PREFIX):]
    if pathlib.PurePosixPath(label).is_absolute():
        raise ValueError(f"absolute label in manifest: {label!r}")
    return pathlib.Path(os.path.normpath(study / label))


def write_manifest(results_dir: pathlib.Path | None = None, *, code_only: bool = False,
                   out: pathlib.Path | None = None, study: pathlib.Path = STUDY,
                   repo: pathlib.Path = REPO) -> tuple[pathlib.Path, int]:
    """Write the manifest; returns (path, number of files). The output must lie in the study or results folder.

    Default output: <results>/manifest.sha256, or <study>/freeze_manifest.sha256 with code_only=True, so the later
    full manifest never overwrites the freeze manifest. The output file itself is never listed (a manifest that
    lies inside the code tree would otherwise carry its own stale hash after a rewrite and fail its own check)."""
    results_dir = pathlib.Path(results_dir) if results_dir is not None else study / "results"
    if out is None:
        out = study / FREEZE_NAME if code_only else results_dir / MANIFEST_NAME
    out = pathlib.Path(out)
    if not _inside(out.parent, (study, results_dir)):
        raise ValueError(f"refusing to write {out}: outside the study folder and the results directory")
    out_resolved = out.resolve()
    entries = [(lab, p) for lab, p in code_files(study, repo, results_dir) if p.resolve() != out_resolved]
    if not code_only:
        entries += result_files(results_dir, exclude=(out,))
    entries.sort()
    labels = [lab for lab, _ in entries]
    if len(labels) != len(set(labels)):
        raise RuntimeError("duplicate manifest labels")
    text = "".join(f"{sha256_file(p)}  {lab}\n" for lab, p in entries)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, out)
    return out, len(entries)


def read_manifest(path: pathlib.Path) -> list[tuple[str, str]]:
    entries = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        m = _LINE.match(line)
        if not m:
            raise ValueError(f"{path}:{lineno}: malformed manifest line")
        entries.append((m.group(1), m.group(2)))
    return entries


def check_manifest(manifest_path: pathlib.Path, results_dir: pathlib.Path, *, study: pathlib.Path = STUDY,
                   repo: pathlib.Path = REPO) -> dict:
    """Re-hash the listed files. ok is False on missing or changed files (or a malformed manifest)."""
    report: dict[str, Any] = {"manifest": str(manifest_path), "ok": False, "n_listed": 0, "missing": [],
                              "mismatched": [], "unlisted": []}
    if not manifest_path.is_file():
        report["error"] = "manifest not found (run verify.py --manifest first)"
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
    own = manifest_path.resolve()
    in_scope = [(lab, p) for lab, p in code_files(study, repo, results_dir) if p.resolve() != own]
    if any(label.startswith(RESULTS_PREFIX) for label in listed):
        in_scope += result_files(results_dir, exclude=(manifest_path,))
    report["unlisted"] = sorted(label for label, _ in in_scope if label not in listed)
    report["n_listed"] = len(entries)
    report["ok"] = not report["missing"] and not report["mismatched"]
    return report


def compare(stored: Any, fresh: Any, path: str = "$", out: list[str] | None = None,
            tol: float = FLOAT_TOL) -> list[str]:
    """Differences between two JSON values; numbers compare within tol (relative or absolute)."""
    out = [] if out is None else out
    if isinstance(stored, dict) and isinstance(fresh, dict):
        for k in sorted(set(stored) | set(fresh)):
            if k not in stored:
                out.append(f"{path}.{k}: only in the recomputation")
            elif k not in fresh:
                out.append(f"{path}.{k}: only in the stored file")
            else:
                compare(stored[k], fresh[k], f"{path}.{k}", out, tol)
    elif isinstance(stored, list) and isinstance(fresh, list):
        if len(stored) != len(fresh):
            out.append(f"{path}: length {len(stored)} != {len(fresh)}")
        else:
            for i, (a, b) in enumerate(zip(stored, fresh)):
                compare(a, b, f"{path}[{i}]", out, tol)
    elif isinstance(stored, bool) or isinstance(fresh, bool):
        if stored is not fresh:
            out.append(f"{path}: {stored!r} != {fresh!r}")
    elif isinstance(stored, (int, float)) and isinstance(fresh, (int, float)):
        if not math.isclose(stored, fresh, rel_tol=tol, abs_tol=tol):
            out.append(f"{path}: {stored!r} != {fresh!r}")
    elif stored != fresh:
        out.append(f"{path}: {stored!r} != {fresh!r}")
    return out


def check_freeze(freeze_manifest: pathlib.Path, stored_code: dict | None) -> dict:
    """The analysis code recorded in decisions.json must equal the code in the (code-only) freeze manifest."""
    import analyze

    frozen = {label: digest for digest, label in read_manifest(freeze_manifest)}
    missing = [rel for rel in analyze.ANALYSIS_CODE if rel not in frozen]
    want = {rel: frozen.get(rel) for rel in analyze.ANALYSIS_CODE}
    return {"ok": not missing and stored_code == want, "freeze_manifest": str(freeze_manifest), "frozen": want,
            "used_for_decisions": stored_code, "missing_in_freeze": missing}


def rerun(results_dir: pathlib.Path | None = None, manifest_path: pathlib.Path | None = None, *,
          freeze_manifest: pathlib.Path | None = None, study: pathlib.Path = STUDY,
          repo: pathlib.Path = REPO) -> dict:
    """Manifest check + recomputation of decisions.json and the tables from the JSONL inputs (+ optional check
    that decisions.json was computed with the analysis code of the freeze manifest)."""
    import analyze  # the study's analysis module (STUDY is on sys.path)

    results_dir = pathlib.Path(results_dir) if results_dir is not None else study / "results"
    manifest_path = pathlib.Path(manifest_path) if manifest_path is not None else results_dir / MANIFEST_NAME
    report: dict[str, Any] = {"results_dir": str(results_dir),
                              "manifest": check_manifest(manifest_path, results_dir, study=study, repo=repo)}
    dec_path = results_dir / "decisions.json"
    if not dec_path.is_file():
        report["decisions"] = {"ok": False, "error": "decisions.json not found (run analyze.py first)"}
        report["ok"] = False
        return report
    stored = json.loads(dec_path.read_text(encoding="utf-8"))
    opts = stored.get("options")
    if not isinstance(opts, dict):
        raise ValueError("decisions.json has no 'options' block; cannot recompute")
    data = analyze.load_results(results_dir)
    fresh = analyze.compute_decisions(data, worlds=opts["worlds"], bootstrap_n=opts["bootstrap_n"],
                                      perm_n=opts["perm_n"], seed=opts["seed"], map_only=opts["map_only"],
                                      precondition_failed=opts["precondition_failed"])
    fresh = json.loads(json.dumps(fresh, allow_nan=False))
    diffs = compare(stored, fresh)
    report["decisions"] = {"ok": not diffs, "n_differences": len(diffs), "differences": diffs[:MAX_REPORTED_DIFFS],
                           "summary_stored": stored.get("summary"), "summary_recomputed": fresh.get("summary")}
    code_now = analyze.analysis_code_hashes()
    report["analysis_code"] = {"ok": stored.get("analysis_code") == code_now, "stored": stored.get("analysis_code"),
                               "current": code_now}
    table_diffs = []
    for rel, text in analyze.render_outputs(fresh, data).items():
        if rel == "decisions.json":
            continue
        path = results_dir / rel
        if not path.is_file():
            table_diffs.append(f"{rel}: missing")
        elif path.read_text(encoding="utf-8") != text:
            table_diffs.append(f"{rel}: differs from the recomputation")
    report["tables"] = {"ok": not table_diffs, "differences": table_diffs}
    ok = report["manifest"]["ok"] and report["decisions"]["ok"] and report["analysis_code"]["ok"] and \
        report["tables"]["ok"]
    if freeze_manifest is not None:
        report["freeze"] = check_freeze(pathlib.Path(freeze_manifest), stored.get("analysis_code"))
        ok = ok and report["freeze"]["ok"]
    report["ok"] = bool(ok)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CEV-11 manifest and decision recomputation (independent of git)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", action="store_true", help="write the SHA-256 manifest")
    mode.add_argument("--check", action="store_true", help="check the manifest")
    mode.add_argument("--rerun", action="store_true", help="check the manifest and recompute the decisions")
    ap.add_argument("--results", type=pathlib.Path, default=STUDY / "results")
    ap.add_argument("--code-only", action="store_true", help="with --manifest: hash the code tree only (freeze)")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="with --manifest: output path (default <results>/manifest.sha256; with --code-only "
                         "<study>/freeze_manifest.sha256)")
    ap.add_argument("--manifest-path", type=pathlib.Path, default=None)
    ap.add_argument("--freeze", type=pathlib.Path, default=None,
                    help="with --rerun: code-only freeze manifest (normally <study>/freeze_manifest.sha256); "
                         "decisions must use its analysis code")
    ap.add_argument("--report", type=pathlib.Path, default=None, help="with --check/--rerun: write a JSON report")
    args = ap.parse_args(argv)

    if args.manifest:
        path, n = write_manifest(args.results, code_only=args.code_only, out=args.out)
        print(f"wrote {path} ({n} files{', code only' if args.code_only else ''})")
        return 0
    manifest_path = args.manifest_path or args.results / MANIFEST_NAME
    report = (rerun(args.results, manifest_path, freeze_manifest=args.freeze) if args.rerun
              else check_manifest(manifest_path, args.results))
    text = json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False)
    print(text)
    if args.report is not None:
        if not _inside(args.report.parent, (STUDY, args.results)):
            raise SystemExit(f"refusing to write {args.report}: outside the study folder and the results directory")
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
