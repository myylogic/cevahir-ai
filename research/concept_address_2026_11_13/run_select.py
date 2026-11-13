# -*- coding: utf-8 -*-
"""KA discovery selection: operating points from discovery worlds 0-1 only -> results/selection.json (+ hash).

Usage (from the repository root, after both discovery jobs finished):

    python research/concept_address_2026_11_13/run_select.py [--results results/discovery] [--out results/selection.json]

The selection fixes, before any confirmatory run:
* the prediction-position layer l* (minimum cross-fitted c0 calibration log-loss, with the preregistered
  fallback to the best discovery frontier),
* whether PRED-ATT enters S1a/S1b (G-KA1b),
* the frozen frontier mixture of every Part A address (solved on the pooled discovery TEST edits),
* whether KA-H1 enters the confirmatory family (G-KA1a) and the resulting Holm families,
* the FT stream check on the first discovery world (G-KA1c) and KA-H2's predicted direction,
* the consistency checks on the discovery worlds, and for every frozen mixture the outlook of the KA-H1 locality
  guard (how likely the guard is to pass if the confirmatory worlds behave like the discovery edits).

``selection.json`` is ``{"sha256": <hash of the body>, "body": {...}}``; the hash is also written to
``selection.sha256``. Both discovery worlds must be complete (all four phases) and run with the protocol
configuration.

Exit code 0 when the selection is written and no blocking gate failed; 2 when it is written but G-KA1c or a
consistency check failed (``verify.py --freeze`` then refuses until the cause is found and the discovery jobs are
run again).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

KA_DIR = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(KA_DIR)]

from ka import analysis as A  # noqa: E402
from ka.world_run import write_json_atomic  # noqa: E402

DISCOVERY_WORLDS = (0, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(KA_DIR / "results" / "discovery"))
    ap.add_argument("--out", default=str(KA_DIR / "results" / "selection.json"))
    args = ap.parse_args(argv)
    try:
        worlds = A.load_worlds(pathlib.Path(args.results), DISCOVERY_WORLDS)
        missing = [w for w in DISCOVERY_WORLDS if w not in worlds]
        if missing:
            raise SystemExit(f"discovery worlds {missing} are not complete under {args.results}")
        sel = A.select(worlds)
    except ValueError as exc:
        raise SystemExit(str(exc))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, sel)
    out.with_name("selection.sha256").write_text(f"{sel['sha256']}  selection.json (body)\n", encoding="utf-8")
    b = sel["body"]
    print(json.dumps({"sha256": sel["sha256"], "pred_layer": b["pred_layer"], "pred_att": b["pred_att"],
                      "h1_confirmatory": b["h1_confirmatory"], "primary_family": b["primary_family"],
                      "secondary_family": b["secondary_family"],
                      "mixture_feasible": {a: v["feasible"] for a, v in b["mixture_discovery"].items()},
                      "locality_guard_p_pass": {a: v["locality_guard"]["p_pass"]
                                                for a, v in b["mixture_discovery"].items()},
                      "screens": {k: v["pass"] for k, v in b["screens"].items()},
                      "go": b["go"], "blocking_gates": b["blocking_gates"]}, indent=1))
    return 0 if b["go"] else 2


if __name__ == "__main__":
    sys.exit(main())
