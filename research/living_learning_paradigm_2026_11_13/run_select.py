# -*- coding: utf-8 -*-
"""Keşif seçim ızgarası (PROTOCOL §8; yalnız keşif dünyaları 0–1).

Her kol için katman 0–4 × ızgara, 20 keşif düzenlemesi. Kayıtlar (run="select") results/select/ altına yazılır;
seçim kararları ``select_decide.py`` ile alınır. Seçim için ölçülen aileler: WF, PF6, PF5/IF5 (hedef hâller), RN,
RD, RC — bunlar keşif bağlanma-yeri haritasını da besler. Izgaralar:
  STD / ROOT / LAST : step ∈ {0.5, 2.0, 5.0} × cap_mult ∈ {8, 16}
  PARA-ORTH         : STD'nin yapılandırmalarıyla (aynı δ, kendi anahtarı; v* araması önbellekten)
  PRED (yalnız harita): step 2.0 × cap_mult ∈ {8, 16}
  FT-L              : lr ∈ {1e-3, 1e-2} × eps ∈ {0.005, 0.02}
  FT-ALL            : lr ∈ {1e-4, 3e-4, 1e-3} (katman yok)

Kullanım: python run_select.py --base results/bases/discovery_w0_K20 --world 0 --arms STD PARA-ORTH
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

STUDY = pathlib.Path(__file__).resolve().parent
REPO = STUDY.parents[1]
sys.path[:0] = [str(STUDY), str(REPO)]

import torch  # noqa: E402

torch.set_num_threads(1)

LAYERS = (0, 1, 2, 3, 4)
KEY_GRID = [{"step": s, "cap_mult": c} for s in (0.5, 2.0, 5.0) for c in (8, 16)]
PRED_GRID = [{"step": 2.0, "cap_mult": c} for c in (8, 16)]
FTL_GRID = [{"lr": lr, "eps": e, "steps": 60} for lr in (1e-3, 1e-2) for e in (0.005, 0.02)]
FTALL_GRID = [{"lr": lr, "steps": 40} for lr in (1e-4, 3e-4, 1e-3)]
FAMILIES = ("WF", "PF6", "PF5", "IF5", "RN", "RD", "RC")
N_DISCOVERY_EDITS = 20


def config_label(hp: dict) -> str:
    return ",".join(f"{k}={hp[k]}" for k in sorted(hp))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", default=str(STUDY / "results" / "select"))
    ap.add_argument("--edits", type=int, default=N_DISCOVERY_EDITS)
    args = ap.parse_args(argv)
    if args.world not in (0, 1):
        raise SystemExit("selection runs only on discovery worlds 0 and 1")
    from cev11 import experiment as E
    from cev11 import probes as P

    base = E.load_base(STUDY / args.base if not pathlib.Path(args.base).is_absolute() else args.base, REPO)
    gates = json.loads((base.base_dir / "gates.json").read_text(encoding="utf-8"))
    editable = gates["G7"].get("editable") or gates.get("editable")
    if not editable:
        raise SystemExit("no editable list in gates.json")
    items = P.edit_items(base.world, editable, seed=args.world + 2, per_case=3)[: args.edits]
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    answers = E.Answers(base); inst = E.Installer(base); memo: dict = {}
    t0 = time.time()
    preps = [E.prepare_item(base, it, answers, seed=args.world + 3) for it in items]
    print(f"[select w{args.world}] prepared {len(preps)} items in {time.time() - t0:.0f}s", flush=True)

    plan = []
    for arm in args.arms:
        if arm in ("STD", "ROOT", "LAST", "PARA-ORTH"):
            plan += [(arm, L, hp) for L in LAYERS for hp in KEY_GRID]
        elif arm == "PRED":
            plan += [(arm, L, hp) for L in LAYERS for hp in PRED_GRID]
        elif arm == "FT-L":
            plan += [(arm, L, hp) for L in LAYERS for hp in FTL_GRID]
        elif arm == "FT-ALL":
            plan += [(arm, None, hp) for hp in FTALL_GRID]
        else:
            raise SystemExit(f"unknown arm {arm}")
    for arm in args.arms:
        path = out_dir / f"select_w{args.world}_{arm}.jsonl"
        done = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").split("\n"):
                if line.strip():
                    r = json.loads(line); done.add((r["layer"], r["config"], r["edit_id"]))
        with path.open("a", encoding="utf-8") as f:
            for a, L, hp in plan:
                if a != arm:
                    continue
                cfg = config_label(hp)
                tc = time.time()
                for prep in preps:
                    if (L, cfg, prep["item"]["edit_id"]) in done:
                        continue
                    rec = E.run_arm(base, inst, answers, prep, arm, L, hp,
                                    world_seed=args.world, run="select", config=cfg, memo=memo, families=FAMILIES)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                print(f"[select w{args.world}] {arm} L{L} {cfg}: {time.time() - tc:.0f}s", flush=True)
    print(f"[select w{args.world}] done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
