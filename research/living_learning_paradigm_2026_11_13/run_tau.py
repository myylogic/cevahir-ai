# -*- coding: utf-8 -*-
"""S4 için katmanlara göre τ (PROTOCOL §10.2 S4, §10.5): yalnız ileri geçiş.

Her yoklama-havuzu varlığı için, son kök tokenini paylaşan en çok 8 komşusu (RN) alınır; anahtarlar cümle başı
yalın önekte (relation A) kök-son tokeninde okunur; τ(k_komşu, k_varlık) = k_komşuᵀ C0⁻¹ k_varlık / k_varlıkᵀ C0⁻¹
k_varlık, katmana özgü C0 ile. Aynı komşular her katmanda kullanılır. Çıktı: results/tau/<run>_<world>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

STUDY = pathlib.Path(__file__).resolve().parent
REPO = STUDY.parents[1]
sys.path[:0] = [str(STUDY), str(REPO)]

import torch  # noqa: E402

torch.set_num_threads(1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--run", default="confirm")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=str(STUDY / "results" / "tau"))
    args = ap.parse_args(argv)
    if args.run == "confirm" and args.world in (0, 1):
        raise SystemExit("discovery worlds cannot produce confirmatory tau records")
    from cev11 import experiment as E
    from cev11.keysites import key_at, sites
    from cev11.tokenizers import last_root_token
    from cev11.world import frame_prefix
    base_dir = pathlib.Path(args.base)
    base = E.load_base(base_dir if base_dir.is_absolute() else STUDY / base_dir, REPO)
    w, tok = base.world, base.tok
    probe = sorted(n for n, p in w.partition.items() if p == "probe")
    lrt = {n: last_root_token(tok, n) for n in probe}
    rnd = random.Random(args.world + 3)
    neigh = {}
    for n in probe:
        cand = [m for m in probe if m != n and lrt[m] == lrt[n]]
        if cand:
            neigh[n] = rnd.sample(cand, min(8, len(cand)))
    prefix = {n: frame_prefix(n, "A", "nom", "formal", "none") for n in probe}
    root_pos = {n: sites(tok, prefix[n], n, "none")["ROOT"] for n in probe}
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.run}_{args.world}.jsonl"
    t0 = time.time()
    with path.open("w", encoding="utf-8") as f:
        for L in args.layers:
            st = E.layer_stats(base, L, args.world)
            keys = {n: key_at(base.model, L, tok, prefix[n], root_pos[n]).double() for n in probe}
            for n, nbs in sorted(neigh.items()):
                u = st["C0_inv"] @ keys[n]
                den = float(keys[n] @ u)
                taus = [float(keys[m] @ u) / den for m in nbs]
                f.write(json.dumps({"schema": "cev11.tau.v1", "run": args.run, "world": args.world, "layer": L,
                                    "entity": n, "site": "ROOT", "tau_RN": taus}, ensure_ascii=False) + "\n")
            print(f"[tau w{args.world}] layer {L}: {len(neigh)} entities ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
