# -*- coding: utf-8 -*-
"""KA verification tooling: freeze manifest, record validation, recomputation of the decisions (no git needed).

Usage (from the repository root; OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1):

    python research/concept_address_2026_11_13/verify.py --freeze [--refreeze]
        Write freeze_manifest.sha256. Refused while PROTOCOL.md still says NOT FROZEN, while
        results/selection.json is missing, does not match its own hash or reports a failed blocking gate
        (G-KA1c, the consistency checks), and for ever once a confirmatory record exists under results/confirm.
        An existing manifest is never overwritten silently: --refreeze replaces it, and only while no
        confirmatory record exists.
    python research/concept_address_2026_11_13/verify.py --check
        Re-hash every file of the freeze manifest; missing, changed or unlisted in-scope files fail.
    python research/concept_address_2026_11_13/verify.py --preconditions
        Everything a confirmatory run needs: the freeze manifest checks out, the protocol is frozen, the selection
        is intact and reports no failed blocking gate, and CEV-11's confirmatory analysis is recorded (blinding
        rule, PROTOCOL §8).
    python research/concept_address_2026_11_13/verify.py --records --results results/confirm
        Validate every world directory: DONE.json hashes, record schema, completeness of every listed phase,
        and that every phase the protocol requires of the world is there (all four on a discovery world; on a
        confirmatory world Part A only if the selection kept KA-H1).
    python research/concept_address_2026_11_13/verify.py --rerun --results results/confirm
        --records, then recompute the decisions from the records and compare them with decisions.json
        (numbers within 1e-9) and check that the analysis code equals the frozen code.

What the freeze covers (``scope_files``): PROTOCOL.md, every Python file of the study folder, its ``jobs_*.json``
job lists, results/selection.json, the KA tests and the frozen CEV-11 engine this study imports. What it does not
cover: README.md and AMENDMENTS.md (living documents: the status and the results pointer, and the deviations
recorded after the freeze, PROTOCOL §8), logs, results, and any other file. Writing AMENDMENTS.md or updating
README.md after the freeze therefore never blocks a confirmatory run; a new or changed Python file does.

Exit code 0 when everything agrees, 1 otherwise. Nothing but the manifest (--freeze) and an optional --report
file is written.
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

KA_DIR = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(KA_DIR)]

from ka import CEV11_DIR, KA_VERSION, REPO  # noqa: E402

FREEZE_NAME = "freeze_manifest.sha256"
SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache", "results"})
SKIP_SUFFIXES = (".pyc", ".pyo", ".tmp", ".swp", ".part", ".log")
LIVING_DOCS = ("README.md", "AMENDMENTS.md")        # outside the freeze on purpose (see the module docstring)
TEST_GLOB = "test_concept_address_*.py"
CONFIRM_WORLDS = tuple(range(1000, 1008))
FLOAT_TOL = 1e-9
_LINE = re.compile(r"^([0-9a-f]{64})  (\S(?:.*\S)?)$")
ANALYSIS_CODE = ("analyze.py", "ka/analysis.py", "ka/stats.py", "ka/part_a.py", "ka/gates.py", "ka/world_run.py")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def protocol_frozen(ka_dir: pathlib.Path = KA_DIR) -> bool:
    """True iff PROTOCOL.md carries a status line starting with 'Status: FROZEN'."""
    text = (ka_dir / "PROTOCOL.md").read_text(encoding="utf-8")
    return any(line.strip().startswith("Status: FROZEN") for line in text.split("\n"))


def in_scope(rel: pathlib.PurePath) -> bool:
    """True for the files of the study folder that the freeze covers (``rel`` is relative to the folder):
    every Python file outside results/ and the caches, PROTOCOL.md, and the root-level ``jobs_*.json`` lists."""
    if any(part in SKIP_DIRS for part in rel.parts) or rel.name == FREEZE_NAME:
        return False
    if rel.suffix == ".py":
        return True
    return len(rel.parts) == 1 and (rel.name == "PROTOCOL.md"
                                    or (rel.suffix == ".json" and rel.name.startswith("jobs_")))


def scope_files(ka_dir: pathlib.Path = KA_DIR) -> Dict[str, pathlib.Path]:
    """label -> path of every file the freeze covers (labels are relative to the study folder)."""
    out: Dict[str, pathlib.Path] = {}
    for p in sorted(ka_dir.rglob("*")):
        rel = p.relative_to(ka_dir)
        if p.is_file() and in_scope(rel):
            out[rel.as_posix()] = p
    sel = ka_dir / "results" / "selection.json"
    if sel.exists():
        out["results/selection.json"] = sel
    repo = ka_dir.parents[1]
    for p in sorted((repo / "tests" / "evolution").glob(TEST_GLOB)):
        out[f"../../tests/evolution/{p.name}"] = p
    cev = repo / "research" / "living_learning_paradigm_2026_11_13"
    for p in sorted((cev / "cev11").glob("*")):
        if p.is_file() and p.suffix not in SKIP_SUFFIXES:
            out[f"../{cev.name}/cev11/{p.name}"] = p
    for name in ("freeze_manifest.sha256", "results/selection.json"):
        if (cev / name).exists():
            out[f"../{cev.name}/{name}"] = cev / name
    return out


def read_manifest(path: pathlib.Path) -> Dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        m = _LINE.match(line)
        if not m:
            raise ValueError(f"malformed manifest line: {line!r}")
        out[m.group(2)] = m.group(1)
    return out


def check_selection_file(ka_dir: pathlib.Path = KA_DIR) -> dict:
    from ka.analysis import check_selection
    path = ka_dir / "results" / "selection.json"
    if not path.exists():
        raise ValueError("results/selection.json is missing (run the discovery jobs and run_select.py first)")
    return check_selection(json.loads(path.read_text(encoding="utf-8")))


def selection_problems(ka_dir: pathlib.Path = KA_DIR) -> List[str]:
    """Why the selection forbids a freeze or a confirmatory run: missing, tampered, or a failed blocking gate."""
    from ka.analysis import selection_blockers
    try:
        return selection_blockers(check_selection_file(ka_dir))
    except ValueError as exc:
        return [str(exc)]


def confirm_records(ka_dir: pathlib.Path = KA_DIR) -> List[str]:
    """Names of the confirmatory record sets under the default results/confirm (world directories, decisions)."""
    root = ka_dir / "results" / "confirm"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if (p.is_dir() and p.name.startswith("w")) or p.name in ("decisions.json", "tables.md"))


def freeze(ka_dir: pathlib.Path = KA_DIR, refreeze: bool = False) -> List[str]:
    """Write the freeze manifest. Returns the problems that prevented it (empty list = written).

    Once a confirmatory record exists the manifest can no longer be written or replaced: a freeze made after
    confirmatory data would not be a freeze. Before that, an existing manifest is replaced only with ``refreeze``.
    """
    problems = []
    if not protocol_frozen(ka_dir):
        problems.append("PROTOCOL.md is marked NOT FROZEN: set its status line to 'Status: FROZEN <date>' first")
    problems += selection_problems(ka_dir)
    records = confirm_records(ka_dir)
    if records:
        problems.append("confirmatory records exist under results/confirm (" + ", ".join(records[:4])
                        + ("" if len(records) <= 4 else ", …") + "): the freeze manifest can no longer be written "
                        "or replaced; record the deviation in AMENDMENTS.md")
    elif (ka_dir / FREEZE_NAME).exists() and not refreeze:
        problems.append(f"{FREEZE_NAME} already exists and is not overwritten: pass --refreeze to replace it "
                        "(possible only while no confirmatory record exists)")
    if problems:
        return problems
    files = scope_files(ka_dir)
    text = "".join(f"{sha256_file(p)}  {label}\n" for label, p in sorted(files.items()))
    tmp = ka_dir / (FREEZE_NAME + ".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, ka_dir / FREEZE_NAME)
    return []


def check(ka_dir: pathlib.Path = KA_DIR) -> List[str]:
    """Problems of the freeze manifest (empty list = everything matches)."""
    path = ka_dir / FREEZE_NAME
    if not path.exists():
        return [f"{FREEZE_NAME} is missing (the study is not frozen)"]
    listed = read_manifest(path)
    files = scope_files(ka_dir)
    problems = []
    for label, digest in sorted(listed.items()):
        p = files.get(label)
        if p is None or not p.exists():
            problems.append(f"missing: {label}")
        elif sha256_file(p) != digest:
            problems.append(f"changed: {label}")
    for label in sorted(set(files) - set(listed)):
        problems.append(f"not in the manifest: {label}")
    return problems


def cev11_analysis_recorded() -> bool:
    """Blinding rule: CEV-11's confirmatory decisions.json exists and its hash is recorded in AMENDMENTS.md."""
    dec = CEV11_DIR / "results" / "decisions.json"
    am = CEV11_DIR / "AMENDMENTS.md"
    if not dec.exists() or not am.exists():
        return False
    text = am.read_text(encoding="utf-8")
    return sha256_file(dec) in text and "recorded output" in text


def preconditions(ka_dir: pathlib.Path = KA_DIR) -> List[str]:
    problems = []
    if not protocol_frozen(ka_dir):
        problems.append("PROTOCOL.md is not frozen")
    problems += check(ka_dir)
    problems += selection_problems(ka_dir)
    if not cev11_analysis_recorded():
        problems.append("CEV-11's confirmatory analysis is not recorded in its AMENDMENTS.md (blinding rule)")
    return problems


# ---------------------------------------------------------------------------------------------------- records
def validate_world(path: pathlib.Path, required: Optional[Sequence[str]] = None) -> List[str]:
    """Hash, schema and completeness problems of one world directory (empty list = valid).

    ``required`` names the phases the protocol requires of this world; with None the list the runner recorded
    in DONE.json is used. Checked: the record version; every file hashed in DONE.json; no record file outside
    DONE.json; every listed phase complete (``ka.world_run.phase_problems``); every required phase listed;
    DONE.json and setup.json agree on the protocol configuration."""
    from ka.world_run import PHASES, RECORD_FILES, phase_problems
    path = pathlib.Path(path)
    name = path.name
    done_path = path / "DONE.json"
    if not done_path.exists():
        return [f"{name}: DONE.json is missing (incomplete world)"]
    done = json.loads(done_path.read_text(encoding="utf-8"))
    problems = []
    if done.get("ka_version") != KA_VERSION:
        problems.append(f"{name}: records of version {done.get('ka_version')!r}; this code reads {KA_VERSION!r}")
    for key in ("phases", "required_phases", "files"):
        if key not in done:
            return problems + [f"{name}: DONE.json has no {key!r}"]
    if "setup.json" not in done["files"]:
        problems.append(f"{name}: DONE.json does not hash setup.json")
    for fname, digest in done["files"].items():
        if not (path / fname).exists():
            problems.append(f"{name}: {fname} is missing")
        elif sha256_file(path / fname) != digest:
            problems.append(f"{name}: {fname} differs from DONE.json")
    for phase, fname in RECORD_FILES.items():
        if fname not in done["files"]:
            if (path / fname).exists():
                problems.append(f"{name}: {fname} is not covered by DONE.json")
            if phase in done["phases"]:
                problems.append(f"{name}: DONE.json lists phase {phase!r} without its record file")
    if problems:
        return problems
    setup = json.loads((path / "setup.json").read_text(encoding="utf-8"))
    if bool(done.get("protocol_config")) != bool(setup["config"]["protocol"]):
        problems.append(f"{name}: DONE.json and setup.json disagree on the protocol configuration")
    for phase in PHASES:
        if phase in done["phases"]:
            problems += phase_problems(path, setup, phase)
    for phase in (done["required_phases"] if required is None else required):
        if phase not in done["phases"]:
            problems.append(f"{name}: phase {phase!r} is required but not complete")
    return problems


def validate_results(results: pathlib.Path, ka_dir: pathlib.Path = KA_DIR) -> Tuple[List[str], List[int]]:
    """Problems of every world directory under ``results`` and the list of valid worlds.

    The phases a world must hold are fixed by the protocol, not by the record: all four on a discovery world; on
    a confirmatory world Part A only if the selection of ``ka_dir`` kept KA-H1. A directory marked EXCLUDED.json
    (the base is not a valid unit) is skipped."""
    from ka.world_run import required_phases
    problems, worlds = [], []
    for d in sorted(pathlib.Path(results).glob("w*")):
        if not d.is_dir():
            continue
        if (d / "EXCLUDED.json").exists():
            if (d / "DONE.json").exists():
                problems.append(f"{d.name}: both EXCLUDED.json and DONE.json are present")
            continue
        required = None
        if (d / "DONE.json").exists() and (d / "setup.json").exists():
            run = json.loads((d / "setup.json").read_text(encoding="utf-8")).get("run")
            if run == "discovery":
                required = required_phases("discovery")
            elif run == "confirm":
                try:
                    required = required_phases("confirm", bool(check_selection_file(ka_dir)["h1_confirmatory"]))
                except ValueError as exc:
                    problems.append(f"{d.name}: cannot tell which phases a confirmatory world needs ({exc})")
                    continue
        p = validate_world(d, required)
        problems += p
        if not p:
            worlds.append(int(d.name[1:]))
    return problems, worlds


def _diff(a: Any, b: Any, path: str = "") -> List[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only on one side")
            else:
                out += _diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return [] if a is b else [f"{path}: {a!r} vs {b!r}"]
        if isinstance(a, bool) or isinstance(b, bool):
            return [] if a == b else [f"{path}: {a!r} vs {b!r}"]
        if math.isnan(float(a)) and math.isnan(float(b)):
            return []
        return [] if abs(float(a) - float(b)) <= FLOAT_TOL else [f"{path}: {a!r} vs {b!r}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def rerun(results: pathlib.Path, ka_dir: pathlib.Path = KA_DIR) -> List[str]:
    from ka import analysis as A
    problems, worlds = validate_results(results, ka_dir)
    dec_path = pathlib.Path(results) / "decisions.json"
    if not dec_path.exists():
        return problems + ["decisions.json is missing (run analyze.py first)"]
    stored = json.loads(dec_path.read_text(encoding="utf-8"))
    try:
        sel = check_selection_file(ka_dir)
        recomputed = json.loads(json.dumps(A.decide(A.load_worlds(results, stored["decisions"]["worlds"]), sel)))
    except ValueError as exc:
        return problems + [f"the decisions cannot be recomputed: {exc}"]
    problems += [f"decisions differ: {d}" for d in _diff(stored["decisions"], recomputed)[:50]]
    status = json.loads(json.dumps(A.world_status(results, CONFIRM_WORLDS)))
    problems += [f"world accounting differs: {d}" for d in _diff(stored.get("world_status"), status)[:10]]
    for name, digest in stored.get("code", {}).items():
        if sha256_file(ka_dir / name) != digest:
            problems.append(f"analysis code changed since the analysis: {name}")
    frz = ka_dir / FREEZE_NAME
    if frz.exists():
        listed = read_manifest(frz)
        for name, digest in stored.get("code", {}).items():
            if listed.get(name) != digest:
                problems.append(f"analysis code differs from the freeze manifest: {name}")
    else:
        problems.append(f"{FREEZE_NAME} is missing")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--refreeze", action="store_true",
                    help="with --freeze: replace an existing manifest (only while no confirmatory record exists)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--preconditions", action="store_true")
    ap.add_argument("--records", action="store_true")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--results", default=str(KA_DIR / "results" / "confirm"))
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)
    if not any((args.freeze, args.check, args.preconditions, args.records, args.rerun)):
        ap.error("choose one of --freeze, --check, --preconditions, --records, --rerun")
    if args.refreeze and not args.freeze:
        ap.error("--refreeze goes with --freeze")
    results = pathlib.Path(args.results)
    if not results.is_absolute():
        results = next((b / results for b in (pathlib.Path.cwd(), REPO, KA_DIR) if (b / results).exists()),
                       REPO / results)
    problems: List[str] = []
    report: Dict[str, Any] = {}
    if args.freeze:
        problems += freeze(KA_DIR, refreeze=args.refreeze)
        report["freeze"] = "written" if not problems else "refused"
    if args.check:
        problems += check(KA_DIR)
    if args.preconditions:
        problems += preconditions(KA_DIR)
    if args.records:
        p, worlds = validate_results(results, KA_DIR)
        problems += p
        if not worlds:
            problems.append(f"no completed world under {results}")
        report["valid_worlds"] = worlds
        report["excluded_worlds"] = sorted(int(d.name[1:]) for d in results.glob("w*")
                                           if (d / "EXCLUDED.json").exists())
    if args.rerun:
        problems += rerun(results, KA_DIR)
    report["problems"] = problems
    report["status"] = "passed" if not problems else "failed"
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
