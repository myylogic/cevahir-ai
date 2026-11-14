# -*- coding: utf-8 -*-
"""Command-line entry points of the study (run_world.py, run_select.py, analyze.py are thin wrappers).

The logic lives in the package because other study folders use the same script names: nothing here imports a
bare top-level module such as ``verify`` or ``analyze``.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from . import common as C

STUDY = C.STUDY

WORLD_DOC = """Forward-learning study: run (or resume) one world.

    python research/forward_learning_2026_11_13/run_world.py --run discovery --world 0 [--smoke]
    python research/forward_learning_2026_11_13/run_world.py --run confirm --world 1000

Discovery worlds are 0 and 1, confirmatory worlds 1000-1007. The base weights are read from $CEV_BASES (the
folder that holds discovery_w0_K20/model.pt, ...); nothing in the CEV-11 study folder is written. Records go to
results/<run>/w<world>.jsonl (append-only, resumable). Confirmatory worlds are refused until the protocol is
frozen and the freeze manifest matches (verify.py --freeze / --preconditions).

--smoke runs a reduced setting that mirrors the v1 pilots (not protocol data) and prints timings and a
comparison with the pilot numbers; it writes to results/smoke unless --out is given.
"""

SELECT_DOC = """Forward-learning study: operating points from the discovery worlds (PROTOCOL §6).

    python research/forward_learning_2026_11_13/run_select.py [--results DIR] [--out FILE]

Reads results/discovery/w0.jsonl and w1.jsonl (both must be complete) and writes results/selection.json: step
caps (4 x the discovery median per arm; for stream arms the median over the first checkpoint), the best exact
one-step variant BP*, the stream learner FEC* (by net facts per write; CMP* is descriptive, H3 is tested against
every comparator), and the gate operating points (theta*, gain*) at hard-negative leak <= 2% and prose bpb
change <= 0.01. The file carries its own sha256, the engine and configuration hashes of the discovery records,
and is part of the freeze manifest.
"""

ANALYZE_DOC = """Forward-learning study: confirmatory analysis (PROTOCOL §4, §7). Unit = world, exact sign-flip test, Holm.

    python research/forward_learning_2026_11_13/analyze.py [--results DIR] [--selection FILE] [--out FILE]

Reads the confirmatory records (results/confirm/w1000..w1007.jsonl) and the frozen discovery selection, writes
results/decisions.json and prints the labels. Incomplete world files are refused; fewer than six worlds give
"not testable". ``--describe DIR`` prints descriptive tables of any results folder (no test).
"""


def world_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=WORLD_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, choices=["discovery", "confirm"])
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--base", default=None, help="base folder (default: $CEV_BASES/<name of the world's base>)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--parts", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selection", default=str(STUDY / "results" / "selection.json"))
    args = ap.parse_args(argv)

    import logging
    logging.disable(logging.ERROR)
    import torch
    torch.set_num_threads(1)
    from . import world_run as W

    if args.run == "discovery" and args.world not in C.DISCOVERY_WORLDS:
        raise SystemExit(f"discovery worlds are {C.DISCOVERY_WORLDS}")
    if args.run == "confirm" and args.world not in C.CONFIRMATORY_WORLDS:
        raise SystemExit(f"confirmatory worlds are {C.CONFIRMATORY_WORLDS[0]}..{C.CONFIRMATORY_WORLDS[-1]}")
    if args.smoke:
        if args.run != "discovery":
            raise SystemExit("--smoke runs on a discovery world only")
        run, cfg = "smoke", C.smoke_config()
        out = pathlib.Path(args.out) if args.out else STUDY / "results" / "smoke"
    elif args.run == "discovery":
        run, cfg = "discovery", C.protocol_config()
        out = pathlib.Path(args.out) if args.out else STUDY / "results" / "discovery"
    else:
        from . import verification as V
        problems = V.check_preconditions(pathlib.Path(args.selection))
        if problems:
            raise SystemExit("confirmatory run refused:\n- " + "\n- ".join(problems))
        selection = json.loads(pathlib.Path(args.selection).read_text(encoding="utf-8"))
        run, cfg = "confirm", C.protocol_config(selection)
        out = pathlib.Path(args.out) if args.out else STUDY / "results" / "confirm"

    t0 = time.time()
    info = W.run_world(args.world, run, cfg, out, base_dir=args.base, parts=args.parts,
                       log=lambda m: print(m, flush=True))
    info["wall_seconds"] = round(time.time() - t0, 1)
    if args.smoke:
        records = C.read_records(C.results_path(out, args.world))
        summary = W.summarise(records)
        info["summary"] = summary
        if args.world == 0:
            info["pilot_comparison"] = W.pilot_comparison(summary)
        from .single import protocol_items
        base, editable = C.load_base(args.world, args.base)
        info["exactness"] = W.exactness_check(base, protocol_items(base, editable, args.world, cfg))
        info["note"] = "smoke: reduced setting on a discovery world; not protocol data"
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return 0



def select_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=SELECT_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(STUDY / "results" / "discovery"))
    ap.add_argument("--out", default=str(STUDY / "results" / "selection.json"))
    args = ap.parse_args(argv)
    from . import analysis as A
    from . import verification as V
    report = V.check_records(pathlib.Path(args.results), worlds=list(C.DISCOVERY_WORLDS))
    missing = [w for w in C.DISCOVERY_WORLDS if str(w) not in report["worlds"]]
    if report["problems"] or missing:
        print(json.dumps({"selection": "refused", "problems": report["problems"],
                          "missing_worlds": missing}, indent=1))
        return 1
    discovery = V.load_run(pathlib.Path(args.results), C.DISCOVERY_WORLDS)
    if any(r["run"] != "discovery" for recs in discovery.values() for r in recs):
        print(json.dumps({"selection": "refused", "problems": ["records are not discovery records"]}, indent=1))
        return 1
    sel = A.select(discovery)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sel, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"selection": str(out), "sha256": sel["sha256"], "bp_star": sel["bp_star"],
                      "fec_star": sel["fec_star"], "cmp_star": sel["cmp_star"], "gate_arm": sel["gate_arm"],
                      "gate": {a: {"theta": g["theta"]["value"], "theta_feasible": g["theta"]["feasible"],
                                   "gain": g["gain"]["value"], "gain_feasible": g["gain"]["feasible"]}
                               for a, g in sel["gate"].items()}}, indent=1))
    return 0



def analyze_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=ANALYZE_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(STUDY / "results" / "confirm"))
    ap.add_argument("--selection", default=str(STUDY / "results" / "selection.json"))
    ap.add_argument("--out", default=str(STUDY / "results" / "decisions.json"))
    ap.add_argument("--describe", default=None, help="print descriptive tables of a results folder and exit")
    args = ap.parse_args(argv)
    from . import analysis as A
    from . import verification as V
    if args.describe:
        tables = {}
        for path in sorted(pathlib.Path(args.describe).glob("w*.jsonl")):
            records = C.read_records(path)
            tables[path.stem] = {"single": A.single_table(records), "stream": A.stream_table(records)}
        print(json.dumps(tables, ensure_ascii=False, indent=1))
        return 0
    problems = V.selection_problems(pathlib.Path(args.selection))
    report = V.check_records(pathlib.Path(args.results), worlds=list(C.CONFIRMATORY_WORLDS))
    problems += report["problems"]
    if problems:
        print(json.dumps({"analysis": "refused", "problems": problems}, indent=1))
        return 1
    selection = json.loads(pathlib.Path(args.selection).read_text(encoding="utf-8"))
    per_world = V.load_run(pathlib.Path(args.results), C.CONFIRMATORY_WORLDS)
    if any(r["run"] != "confirm" for recs in per_world.values() for r in recs):
        print(json.dumps({"analysis": "refused", "problems": ["records are not confirmatory records"]}, indent=1))
        return 1
    header_problems = V.run_header_problems(per_world, selection)
    if header_problems:
        print(json.dumps({"analysis": "refused", "problems": header_problems}, indent=1))
        return 1
    decisions = json.loads(json.dumps(A.compute_decisions(per_world, selection)))
    decisions["records_sha256"] = {w: v["sha256"] for w, v in report["worlds"].items()}
    # whether protocol, selection, study code and engine still are what was frozen (reported, not hidden)
    freeze_problems = V.check_preconditions(pathlib.Path(args.selection))
    decisions["freeze"] = {"ok": not freeze_problems, "problems": freeze_problems}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decisions, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"decisions": str(out), "worlds": decisions["worlds"], "freeze": decisions["freeze"],
                      "labels": {h: d["label"] for h, d in decisions["decisions"].items()},
                      "secondary": {h: d["label"] for h, d in decisions["secondary"].items()}}, indent=1))
    return 0

