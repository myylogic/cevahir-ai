# -*- coding: utf-8 -*-
"""Freeze manifest, preconditions, record checks, re-analysis, comparator cross-check.

    python research/forward_learning_2026_11_13/verify.py --records [--results DIR]   validate records, completeness
    python research/forward_learning_2026_11_13/verify.py --freeze                    write freeze_manifest.sha256
    python research/forward_learning_2026_11_13/verify.py --preconditions             exit 1 unless confirm may run
    python research/forward_learning_2026_11_13/verify.py --rerun                     recompute decisions, compare
    python research/forward_learning_2026_11_13/verify.py --crosscheck                FT-L / FT-ALL vs CEV-11 records

The freeze is refused unless PROTOCOL.md carries a line "Status: FROZEN <YYYY-MM-DD>" and the discovery selection
(results/selection.json) exists with a valid hash. The manifest covers the protocol, the selection, the study
code and the engine module; any later change makes --preconditions fail, and run_world.py then refuses
confirmatory worlds.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Dict, List, Optional

from . import common as C

STUDY = C.STUDY
MANIFEST = STUDY / "freeze_manifest.sha256"
SELECTION = STUDY / "results" / "selection.json"
DECISIONS = STUDY / "results" / "decisions.json"
FROZEN_RE = re.compile(r"^Status: FROZEN (\d{4}-\d{2}-\d{2})\s*$", re.M)


def protocol_frozen(text: Optional[str] = None) -> Optional[str]:
    """The freeze date if PROTOCOL.md is frozen, else None."""
    text = (STUDY / "PROTOCOL.md").read_text(encoding="utf-8") if text is None else text
    m = FROZEN_RE.search(text)
    return m.group(1) if m else None


def frozen_files(selection: pathlib.Path = SELECTION) -> Dict[str, pathlib.Path]:
    files = {"PROTOCOL.md": STUDY / "PROTOCOL.md", "results/selection.json": pathlib.Path(selection)}
    for name in ("run_world.py", "run_select.py", "analyze.py", "verify.py"):
        files[name] = STUDY / name
    for p in sorted((STUDY / "fwl").glob("*.py")):
        files[f"fwl/{p.name}"] = p
    files["engine:" + C.ENGINE_FILE.relative_to(C.REPO).as_posix()] = C.ENGINE_FILE
    return files


def current_hashes(selection: pathlib.Path = SELECTION) -> Dict[str, str]:
    return {name: C.sha256_file(path) for name, path in frozen_files(selection).items() if path.exists()}


def read_manifest(path: pathlib.Path = MANIFEST) -> Dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            digest, name = line.split("  ", 1)
            out[name] = digest
    return out


def selection_problems(selection: pathlib.Path = SELECTION) -> List[str]:
    from . import analysis as A
    if not pathlib.Path(selection).exists():
        return [f"{selection} is missing (run the discovery jobs and run_select.py)"]
    sel = json.loads(pathlib.Path(selection).read_text(encoding="utf-8"))
    problems = []
    if sel.get("schema") != "fwl.selection.v1" or not A.selection_hash_ok(sel):
        problems.append("selection.json: wrong schema or hash")
    if tuple(sel.get("worlds", ())) != C.DISCOVERY_WORLDS:
        problems.append(f"selection.json: must be built from the discovery worlds {C.DISCOVERY_WORLDS}")
    for k in ("bp_star", "fec_star", "cmp_star", "gate_arm"):
        if not sel.get(k):
            problems.append(f"selection.json: {k} is not set")
    return problems


def check_preconditions(selection: pathlib.Path = SELECTION, manifest: pathlib.Path = MANIFEST) -> List[str]:
    """Reasons why a confirmatory world must not run now (empty list = may run)."""
    problems = []
    if protocol_frozen() is None:
        problems.append("PROTOCOL.md is not frozen (no 'Status: FROZEN <date>' line)")
    problems += selection_problems(selection)
    if not manifest.exists():
        problems.append(f"{manifest.name} is missing (verify.py --freeze)")
    else:
        recorded, now = read_manifest(manifest), current_hashes(selection)
        for name in sorted(set(recorded) | set(now)):
            if recorded.get(name) != now.get(name):
                problems.append(f"changed since the freeze: {name}")
    return problems


def freeze(selection: pathlib.Path = SELECTION, manifest: pathlib.Path = MANIFEST) -> List[str]:
    problems = []
    if protocol_frozen() is None:
        problems.append("PROTOCOL.md is not frozen (no 'Status: FROZEN <date>' line)")
    problems += selection_problems(selection)
    if problems:
        return problems
    lines = [f"{digest}  {name}" for name, digest in sorted(current_hashes(selection).items())]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return []


# -------------------------------------------------------------------------------------------------------- records
def expected_keys(head: dict, records: List[dict]) -> List[str]:
    """Record keys a complete world file must hold, from its own header (no model needed)."""
    cfg = head["config"]
    keys = []
    edits = sorted({r["edit_id"] for r in records if r["kind"] == "single"})
    if len(edits) != head["n_items"]:
        keys.append(f"<{head['n_items']} edits, found {len(edits)}>")
    for e in edits:
        for arm in cfg["single_arms"]:
            for install in (("ft",) if arm in C.SINGLE_FT else cfg["installs"]):
                keys.append(f"single|{e}|{arm}|{install}")
    cps = [c for c in cfg["checkpoints"] if c <= cfg["stream_n"]]
    for s in range(cfg["streams"]):
        for arm in cfg["stream_arms"]:
            lim = min(cfg["arm_limits"].get(arm, cfg["stream_n"]), cfg["stream_n"])
            for c in ([x for x in cps if x <= lim] or [lim]):
                keys.append(f"stream|{s}|{arm}|{c}")
            if s == 0 and arm in cfg["gate_arms"]:
                keys += [f"gate|0|{arm}|theta={t:g}" for t in cfg["thetas"]]
                keys += [f"gate|0|{arm}|gain={g:g}" for g in cfg["gains"] if g != 1.0]
            if s == 0 and arm == "FEC-TOP+RLS":
                keys += [f"capacity|0|{arm}", f"side|0|{arm}"]
    return keys


def check_records(results_dir: pathlib.Path, worlds: Optional[List[int]] = None) -> dict:
    report = {"dir": str(results_dir), "worlds": {}, "problems": []}
    files = sorted(pathlib.Path(results_dir).glob("w*.jsonl"))
    if not files:
        report["problems"].append(f"no record files in {results_dir}")
    for path in files:
        world = int(path.stem[1:])
        if worlds is not None and world not in worlds:
            continue
        try:
            records = C.read_records(path)
        except ValueError as exc:
            report["problems"].append(str(exc))
            continue
        keys = [r["key"] for r in records]
        problems = []
        if len(set(keys)) != len(keys):
            problems.append("duplicate keys")
        head = next((r for r in records if r["key"] == "meta|run"), None)
        if head is None:
            problems.append("no header record")
        else:
            if any(r["world"] != world or r["run"] != head["run"] for r in records):
                problems.append("records of another world or run")
            missing = [k for k in expected_keys(head, records) if k not in set(keys)]
            if missing:
                problems.append(f"{len(missing)} records missing, e.g. {missing[:3]}")
        report["worlds"][str(world)] = {"records": len(records), "sha256": C.sha256_file(path),
                                        "complete": not problems}
        report["problems"] += [f"w{world}: {p}" for p in problems]
    return report


def load_run(results_dir: pathlib.Path, worlds) -> Dict[int, List[dict]]:
    out = {}
    for w in worlds:
        path = C.results_path(results_dir, w)
        if path.exists():
            out[w] = C.read_records(path)
    return out


def rerun(results_dir: pathlib.Path, selection: pathlib.Path = SELECTION, decisions: pathlib.Path = DECISIONS) -> List[str]:
    """Recompute the decisions from the records and compare them with the stored file; also check that the
    record files still have the hashes the analysis wrote down."""
    from . import analysis as A
    problems = []
    if not decisions.exists():
        return [f"{decisions} is missing (analyze.py)"]
    sel = json.loads(pathlib.Path(selection).read_text(encoding="utf-8"))
    stored = json.loads(decisions.read_text(encoding="utf-8"))
    for w, digest in sorted(stored.pop("records_sha256", {}).items()):
        path = C.results_path(results_dir, int(w))
        if not path.exists() or C.sha256_file(path) != digest:
            problems.append(f"records of world {w} changed since the analysis")
    now = json.loads(json.dumps(A.compute_decisions(load_run(results_dir, C.CONFIRMATORY_WORLDS), sel)))
    if C.canonical_json(now) != C.canonical_json(stored):
        problems.append("recomputed decisions differ from the stored decisions")
    return problems


def crosscheck(results_dir: pathlib.Path) -> dict:
    """The fresh FT-L / FT-ALL records against CEV-11's recorded confirmatory records of the same edits."""
    out = {"compared": 0, "mismatches": []}
    for w, records in load_run(results_dir, C.CONFIRMATORY_WORLDS).items():
        for arm in C.SINGLE_FT:
            path = C.CEV11 / "results" / "edits" / f"confirm_{w}_{arm}.jsonl"
            if not path.exists():
                continue
            ref = {}
            for line in path.read_text(encoding="utf-8").split("\n"):
                if line.strip():
                    r = json.loads(line)
                    ref[r["edit_id"]] = r
            for r in records:
                if r["kind"] != "single" or r["arm"] != arm or r["edit_id"] not in ref:
                    continue
                o = ref[r["edit_id"]]
                out["compared"] += 1
                same = (bool(o["installed"]) == r["installed"] and o["PF6"] == r["PF6"]
                        and [(t["case"], t["formal"], t["informal"]) for t in o["targets"]]
                        == [(t["case"], t["formal"], t["informal"]) for t in r["targets"]])
                if not same:
                    out["mismatches"].append(f"w{w} {arm} {r['edit_id']}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", action="store_true")
    ap.add_argument("--results", default=None)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--preconditions", action="store_true")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--crosscheck", action="store_true")
    args = ap.parse_args(argv)
    results = pathlib.Path(args.results) if args.results else STUDY / "results" / "confirm"
    status = 0
    if args.freeze:
        problems = freeze()
        print(json.dumps({"freeze": "written" if not problems else "refused", "problems": problems}, indent=1))
        status |= bool(problems)
    if args.preconditions:
        problems = check_preconditions()
        print(json.dumps({"preconditions": "ok" if not problems else "not met", "problems": problems}, indent=1))
        status |= bool(problems)
    if args.records:
        report = check_records(results)
        print(json.dumps(report, indent=1))
        status |= bool(report["problems"])
    if args.rerun:
        problems = rerun(results)
        print(json.dumps({"rerun": "equal" if not problems else "different", "problems": problems}, indent=1))
        status |= bool(problems)
    if args.crosscheck:
        report = crosscheck(results)
        print(json.dumps(report, indent=1))
        status |= bool(report["mismatches"])
    if not (args.freeze or args.preconditions or args.records or args.rerun or args.crosscheck):
        ap.print_help()
        return 2
    return int(status)

