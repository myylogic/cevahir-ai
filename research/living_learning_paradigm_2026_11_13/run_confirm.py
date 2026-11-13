# -*- coding: utf-8 -*-
"""Doğrulayıcı düzenlemeler (PROTOCOL §7–§9): bir dünya × tüm kollar × 70 madde → results/edits/.

Önkoşullar: dondurulmuş PROTOCOL.md ve freeze_manifest.sha256; results/selection.json (keşif seçimi); temel model
kapıları geçmiş olmalı (geçmezse çıkış kodu 3 — yedek dünya kullanılır, PROTOCOL §6).

Kullanım: python run_confirm.py --base results/bases/confirm_w1000_K20 --world 1000 [--arms STD ROOT ...]
Her (dünya, kol) dosyası satır satır yazılır ve yeniden başlatmada kaldığı yerden devam eder; analiz yalnız tam
dosyaları kabul eder (kol başına 70 madde).
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

ARMS_ORDER = ("STD", "NORM", "PARA-ORTH", "ROOT", "LAST", "FT-L", "FT-ALL", "BASE")
BPB_EDITS = 20


def hp_from_config(cfg: str) -> dict:
    hp = {}
    for part in cfg.split(","):
        k, _, v = part.partition("=")
        if k:
            hp[k] = int(v) if v.isdigit() else float(v)
    return hp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--arms", nargs="+", default=list(ARMS_ORDER))
    ap.add_argument("--run", default="confirm", choices=["confirm", "ffreq"])
    ap.add_argument("--out", default=str(STUDY / "results" / "edits"))
    args = ap.parse_args(argv)
    if args.world in (0, 1):
        raise SystemExit("discovery worlds cannot be used for confirmatory runs")
    from cev11 import experiment as E
    from cev11 import probes as P

    selection = json.loads((STUDY / "results" / "selection.json").read_text(encoding="utf-8"))
    if args.run == "confirm" and selection.get("map_only"):
        # PROTOCOL §8 map-only switch: STD's discovery WE < 50% -> H1-H3 not testable and the confirmatory budget runs
        # the binding-site map instead of the confirmatory edit arms.
        raise SystemExit("selection.json has map_only = true: the confirmatory budget runs the binding-site map "
                         "(PROTOCOL §8, §10.5), not run_confirm.py")
    missing = [a for a in args.arms if a != "BASE" and a not in selection.get("arms", {})]
    if missing:
        raise SystemExit(f"no discovery selection for arms {missing} in selection.json")
    base_dir = pathlib.Path(args.base)
    base = E.load_base(base_dir if base_dir.is_absolute() else STUDY / base_dir, REPO)
    gates = json.loads((base.base_dir / "gates.json").read_text(encoding="utf-8"))
    if not gates.get("all_pass"):
        print(f"[confirm w{args.world}] gates failed -> use the next reserve world", flush=True)
        return 3
    items = P.edit_items(base.world, gates["editable"], seed=args.world + 2, per_case=10)
    # Edit-list record (hash) and the V10 contamination check: no edit target (entity + new object, relation A) and
    # no P3-held cell may occur in any training text (world sentences of one pass cover every trained cell string;
    # prose and the lead-in pool are checked as they are).
    import hashlib
    import random as _random
    from cev11 import leadins as LI
    from cev11.keysites import PROSE_PREFIXES
    from cev11.world import contamination, pass_sentences
    items_blob = json.dumps(items, ensure_ascii=False, sort_keys=True)
    (pathlib.Path(args.out)).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(args.out) / f"items_{args.world}.json").write_text(json.dumps(
        {"world": args.world, "sha256": hashlib.sha256(items_blob.encode("utf-8")).hexdigest(), "items": items},
        ensure_ascii=False, indent=1), encoding="utf-8")
    texts = [t for t, _ in pass_sentences(base.world, 0, _random.Random(args.world + 1))]
    texts += list(base.prose_train) + LI.leadin_pool(base.prose_train, base.tok, exclude=PROSE_PREFIXES)
    viol = contamination(base.world, [(it["name"], "A", it["new"]) for it in items], texts)
    if viol:
        raise SystemExit(f"V10 contamination: {len(viol)} violations, e.g. {viol[:3]}")
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    answers = E.Answers(base); inst = E.Installer(base); memo: dict = {}
    t0 = time.time()
    preps = [E.prepare_item(base, it, answers, seed=args.world + 3) for it in items]
    base.layer_cache["bpb_base"] = E.bits_per_byte(base, base.prose_held)
    print(f"[confirm w{args.world}] prepared {len(preps)} items, base bpb {base.layer_cache['bpb_base']:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    bpb_ids = {p["item"]["edit_id"] for p in preps[:BPB_EDITS]}
    for arm in [a for a in ARMS_ORDER if a in args.arms]:
        sel = selection["arms"].get(arm, {}) if arm != "BASE" else {"layer": None, "config": "none"}
        layer, cfg = sel.get("layer"), sel.get("config", "none")
        hp = hp_from_config(cfg) if cfg != "none" else {}
        path = out_dir / f"{args.run}_{args.world}_{arm}.jsonl"
        done = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").split("\n"):
                if line.strip():
                    done.add(json.loads(line)["edit_id"])
        ta = time.time()
        with path.open("a", encoding="utf-8") as f:
            for prep in preps:
                if prep["item"]["edit_id"] in done:
                    continue
                bpb = base.prose_held if (prep["item"]["edit_id"] in bpb_ids and arm != "BASE") else None
                rec = E.run_arm(base, inst, answers, prep, arm, layer, hp, world_seed=args.world, run=args.run,
                                config=cfg, memo=memo, with_tau=True, bpb_texts=bpb)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        print(f"[confirm w{args.world}] {arm} L{layer} {cfg}: {time.time() - ta:.0f}s", flush=True)
    from cev11.model import model_sha256
    if model_sha256(base.model) != base.base_sha:
        raise SystemExit("base model changed during the run")
    print(f"[confirm w{args.world}] done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
