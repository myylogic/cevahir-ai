# -*- coding: utf-8 -*-
"""KA confirmatory analysis: preregistered statistics on worlds 1000-1007 (unit = world).

Usage (from the repository root, after the confirmatory jobs finished):

    python research/concept_address_2026_11_13/analyze.py [--results results/confirm]

Reads the completed world directories (DONE.json present, hashes verified) and ``results/selection.json``, and
writes ``decisions.json`` and ``tables.md`` into the results directory. The statistics (PROTOCOL §6-§7):
exact sign-flip permutation tests over all 2^n sign patterns with the SESOI as margin, Holm step-down within
the primary and the secondary family, Student-t intervals for labels and guards, one-sided t-tests for threshold
claims. The implementation is ``ka/analysis.py`` and ``ka/stats.py``; this script only does the I/O.

Refused: discovery worlds, non-protocol (smoke) records, and an incomplete set of units. Every confirmatory world
1000-1007 must be accounted for: complete (DONE.json with every phase the selection requires) or excluded by the
driver because its base failed CEV-11's gates (EXCLUDED.json). A world whose job did not finish is not an invalid
unit; the analysis does not run on the rest without it.

Exit code 0, or 2 when a consistency check of PROTOCOL §6 failed: decisions.json and tables.md are still written,
every label reads "audit required", and nothing may be interpreted before the audit.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

KA_DIR = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(KA_DIR)]

from ka import analysis as A  # noqa: E402
from ka.world_run import sha256_file, write_json_atomic  # noqa: E402

CODE = ("analyze.py", "ka/analysis.py", "ka/stats.py", "ka/part_a.py", "ka/gates.py", "ka/world_run.py")
CONFIRM_WORLDS = tuple(range(1000, 1008))


def compute(results: pathlib.Path, selection_path: pathlib.Path) -> dict:
    try:
        sel = A.check_selection(json.loads(selection_path.read_text(encoding="utf-8")))
    except ValueError as exc:
        raise SystemExit(str(exc))
    status = A.world_status(results, CONFIRM_WORLDS)
    if status["incomplete"] or status["absent"]:
        raise SystemExit(f"the confirmatory worlds under {results} are not all accounted for: incomplete "
                         f"{status['incomplete']}, absent {status['absent']}. Each of {CONFIRM_WORLDS[0]}-"
                         f"{CONFIRM_WORLDS[-1]} must be complete (DONE.json) or excluded by the driver "
                         "(EXCLUDED.json); finish the missing jobs first")
    try:
        worlds = A.load_worlds(results, CONFIRM_WORLDS)
        if not worlds:
            raise SystemExit(f"no completed confirmatory world under {results}")
        for w, d in worlds.items():
            if d["setup"]["run"] != "confirm" or not d["setup"]["config"]["protocol"]:
                raise SystemExit(f"world {w}: not a confirmatory record set with the protocol configuration")
        decisions = A.decide(worlds, sel)
    except ValueError as exc:
        raise SystemExit(str(exc))
    return {"decisions": decisions, "selection_sha256": A.sha256_json(sel), "world_status": status,
            "code": {name: sha256_file(KA_DIR / name) for name in CODE}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(KA_DIR / "results" / "confirm"))
    ap.add_argument("--selection", default=str(KA_DIR / "results" / "selection.json"))
    args = ap.parse_args(argv)
    results = pathlib.Path(args.results)
    out = compute(results, pathlib.Path(args.selection))
    write_json_atomic(results / "decisions.json", out)
    (results / "tables.md").write_text(A.render_tables(out["decisions"]) + "\n", encoding="utf-8")
    dec = out["decisions"]
    print(json.dumps({"worlds": dec["worlds"], "excluded": out["world_status"]["excluded"],
                      "audit_required": dec["audit_required"], "consistency_failures": dec["consistency"]["failures"],
                      "primary": {h: e.get("label") for h, e in dec["primary"].items()},
                      "secondary": {h: e.get("label") for h, e in dec["secondary"].items()}}, indent=1))
    return 2 if dec["audit_required"] else 0


if __name__ == "__main__":
    sys.exit(main())
