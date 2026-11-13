# -*- coding: utf-8 -*-
"""Bir dünya için temel modeli eğitir ve kapıları (G1–G7) değerlendirir (CEV-11 PROTOCOL §3–§6).

Kullanım (repo kökünden, tek iş parçacığı):
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python research/living_learning_paradigm_2026_11_13/run_pretrain.py \
        --world 0 --passes 20 [--layers 6] [--names 400] [--prose-share 0.20] [--d 192] [--tag discovery]

Çıktılar: results/bases/<tag>_w<world>_K<passes>/ altında model.pt, tokenizer, curve.jsonl, exposure.json,
gates.json, world.json (dünya özeti ve hash), segmentation.json. Derlem örnekleri hiçbir yerde yazdırılmaz.
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--passes", type=int, required=True)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--names", type=int, default=400)
    ap.add_argument("--prose-share", type=float, default=0.20)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--tag", default="discovery")
    ap.add_argument("--out-root", default=str(STUDY / "results" / "bases"))
    args = ap.parse_args(argv)

    from cev11 import world as W
    from cev11 import tokenizers as T
    from cev11 import train as TR
    from cev11.morph_tr import INFLECTED

    out = pathlib.Path(args.out_root) / f"{args.tag}_w{args.world}_K{args.passes}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    prose_train, prose_held = W.load_prose(REPO)
    # The forbidden-word filter (names avoid every prose word) changes the names; always build the world the
    # same way (make_study_world) and store its JSON so later stages reload the identical world.
    world = W.make_study_world(args.world, REPO, n_names=args.names)
    (out / "world_full.json").write_text(json.dumps(world.to_json(), ensure_ascii=False), encoding="utf-8")
    tok = T.train_tokenizer(prose_train, vocab_size=args.vocab)
    # G1 önkoşulu: tüm dünya dizgeleri ve ayrılmış düzyazı kayıpsız dönmeli (eğitimden önce, ucuz).
    seg = T.segmentation_stats(tok, list(world.names))
    (out / "segmentation.json").write_text(json.dumps(seg, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "world.json").write_text(json.dumps({
        "seed": world.seed, "name_hash": world.name_hash(), "n_names": len(world.names),
        "partition_sizes": {k: sum(1 for v in world.partition.values() if v == k) for k in ("p3", "c0", "probe")},
        "objects": {k: len(v) for k, v in world.objects.items()},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    log_f = open(out / "pretrain.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n"); log_f.flush()

    log(f"world {args.world}: {len(world.names)} names, tokenizer {tok.vocab_size} ids, "
        f"informal NONIDENT share {seg.get('nonident_share', 'n/a')}")
    ckpt = TR.pretrain(world, tok, prose_train, out, passes=args.passes, seed=args.world, log=log,
                       layers=args.layers, d=args.d, prose_share=args.prose_share)
    from cev11.model import load_checkpoint
    model, cfg, tok2, meta = load_checkpoint(ckpt)
    gates = TR.evaluate_gates(model, tok2, world, prose_held, seed=args.world + 3)
    (out / "gates.json").write_text(json.dumps(gates, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"gates: { {k: v for k, v in gates.items() if k != 'editable'} }")
    log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
