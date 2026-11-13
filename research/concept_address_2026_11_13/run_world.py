# -*- coding: utf-8 -*-
"""KA per-world driver: discovery worlds (0-1) and confirmatory worlds (1000-1007), with resume.

Usage (from the repository root; one thread per process):

    export CEV_BASES=/path/to/living_learning_paradigm_2026_11_13/results/bases     # folder holding <name>/model.pt
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
        python research/concept_address_2026_11_13/run_world.py --run discovery --world 0
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
        python research/concept_address_2026_11_13/run_world.py --run confirm --world 1000

The base directory is ``$CEV_BASES/discovery_w<W>_K20`` or ``$CEV_BASES/confirm_w<W>_K20`` unless ``--base`` is
given. The CEV-11 base weights are read only; nothing is written next to them.

* ``--run discovery`` accepts worlds 0 and 1 only. It evaluates every candidate prediction-position layer and
  every Part A configuration; ``run_select.py`` then fixes the operating points.
* ``--run confirm`` refuses worlds 0-1 and refuses to start unless ``verify.py --preconditions`` passes: the
  protocol is frozen, the freeze manifest matches, ``results/selection.json`` is intact and reports no failed
  blocking gate, and CEV-11's confirmatory analysis is recorded (blinding rule). It uses the frozen layer and runs
  Part A only if KA-H1 passed the discovery screen. Exit code 3 when the base did not pass CEV-11's gates: the
  world directory then holds only ``EXCLUDED.json`` (not a valid unit), which the analysis reads as such.
* ``--smoke`` (discovery worlds only) runs a reduced configuration end to end; its records are marked as
  non-protocol and are refused by the selection and the analysis.
* ``--phases`` restricts one invocation to some phases (for example to resume by hand). It never changes what the
  world needs: ``DONE.json`` is written only when every phase the protocol requires of the world is complete in
  the directory (all four on a discovery world; without Part A on a confirmatory world if KA-H1 was screened out),
  and it then lists all of them, whichever invocation wrote them.

Records go to ``results/<run>/w<W>/`` (``--out`` to change the root). A restarted job continues where it stopped.
Exit code 0 when the requested phases are complete, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

KA_DIR = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(KA_DIR)]

import torch  # noqa: E402

torch.set_num_threads(1)

from ka import REPO  # noqa: E402

DISCOVERY_WORLDS = (0, 1)
CONFIRM_WORLDS = tuple(range(1000, 1008))


def load_verify():
    """This folder's verify.py under a unique module name (CEV-11 has a script with the same file name, and a
    plain ``import verify`` would return whichever was imported first in the process)."""
    import importlib.util
    name = "ka_verify_tool"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, KA_DIR / "verify.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def base_dir_for(run: str, world: int, base: str | None) -> pathlib.Path:
    if base:
        return pathlib.Path(base)
    root = os.environ.get("CEV_BASES")
    if not root:
        raise SystemExit("set CEV_BASES to the folder holding the CEV-11 bases (<name>/model.pt) or pass --base")
    name = f"discovery_w{world}_K20" if run in ("discovery", "smoke") else f"confirm_w{world}_K20"
    return pathlib.Path(root) / name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, choices=["discovery", "confirm"])
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--phases", nargs="+", default=None, choices=["parta", "episodic", "ft", "checks"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    from ka import world_run as WR
    if args.run == "discovery" and args.world not in DISCOVERY_WORLDS:
        raise SystemExit("discovery runs use worlds 0 and 1 only")
    if args.run == "confirm":
        if args.world in DISCOVERY_WORLDS:
            raise SystemExit("discovery worlds cannot be used for confirmatory runs")
        if args.world not in CONFIRM_WORLDS:
            raise SystemExit(f"confirmatory worlds are {CONFIRM_WORLDS[0]}-{CONFIRM_WORLDS[-1]}")
        if args.smoke:
            raise SystemExit("--smoke is for discovery worlds only")
    cfg = WR.RunConfig.smoke() if args.smoke else WR.RunConfig()
    required = WR.required_phases(args.run)
    pred_layer = None
    if args.run == "confirm":
        V = load_verify()
        problems = V.preconditions(KA_DIR)
        if problems:
            raise SystemExit("confirmatory run refused:\n  " + "\n  ".join(problems))
        sel = V.check_selection_file(KA_DIR)
        pred_layer = int(sel["pred_layer"])
        cfg.pred_att = bool(sel["pred_att"])
        required = WR.required_phases("confirm", bool(sel["h1_confirmatory"]))
    phases = [p for p in (args.phases or required) if p in required]
    if not phases:
        raise SystemExit(f"none of the requested phases is run on this world (it needs {list(required)}: KA-H1 did "
                         "not pass the discovery screen, so Part A is not run on confirmatory worlds)")
    out = pathlib.Path(args.out) if args.out else KA_DIR / "results" / ("smoke" if args.smoke else args.run)
    base_dir = base_dir_for(args.run, args.world, args.base)
    if not (base_dir / "model.pt").exists():
        raise SystemExit(f"{base_dir / 'model.pt'} not found (CEV_BASES must point at the bases folder)")

    from cev11 import experiment as E
    t0 = time.time()
    base = E.load_base(base_dir, REPO)
    if int(base.world.seed) != args.world:
        raise SystemExit(f"{base_dir} holds world {base.world.seed}, not world {args.world}")
    gates_path = base.base_dir / "gates.json"
    if args.run == "confirm":
        gates = json.loads(gates_path.read_text(encoding="utf-8"))
        if not gates.get("all_pass"):
            marker = WR.write_excluded(out, args.world, args.run, "the base failed CEV-11's gates")
            print(f"[ka w{args.world}] the base failed CEV-11's gates: not a valid unit ({marker})", flush=True)
            return 3
    log = lambda msg: print(msg, flush=True)  # noqa: E731
    log(f"[ka w{args.world}] run={args.run} smoke={args.smoke} base={base_dir.name} sha={base.base_sha[:12]} "
        f"phases={phases} required={list(required)} out={out}")
    wr = WR.run_world(base, args.world, args.run, cfg, out, phases=phases, pred_layer=pred_layer, log=log,
                      required=required)
    have = wr.complete_phases()
    unfinished = [p for p in phases if p not in have]
    state = "complete, DONE.json written" if wr.done else \
        f"NOT complete: required phases still missing {[p for p in required if p not in have]}, no DONE.json"
    log(f"[ka w{args.world}] {state}; {time.time() - t0:.0f}s; timings {wr.timings}")
    return 1 if unfinished else 0


if __name__ == "__main__":
    sys.exit(main())
