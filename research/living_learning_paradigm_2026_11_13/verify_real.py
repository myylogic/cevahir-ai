# -*- coding: utf-8 -*-
"""Real-model verification checks V1, V2, V6 and V8 (PROTOCOL §10.3) on a study base with real STD writes.

Added after the freeze as a verification tool; it imports the frozen modules and changes no frozen file, no record
and no analysis (see AMENDMENTS.md). The unit tests in tests/evolution/test_cev11_*.py cover the same checks on tiny
models; this script repeats them on the study's own bases and edits:

- V1: with an all-zero memory installed at every layer, max|Δlogit| over probe prefixes is exactly 0.0, and
  detaching the hooks restores exactly 0.0.
- V2: fc2 inputs at the root-final token are bitwise identical across the 7 formal forms and the IDENT informal
  forms on truncated prefixes (every layer), and within relative 1e-5 on the full, right-padded frame queries.
- V6: with a write installed, cached and uncached greedy decoding give identical outputs on the sampled probes.
- V8: on a random share (default 5%) of all probes of the edited and the unedited model, the batched exact check
  agrees with true (uncached) greedy decoding; disagreement must be at most 1% per family and overall.

Usage: python verify_real.py --base results/bases/confirm_w1000_K20 --world 1000 [--edits 10] [--frac 0.05]
           [--selection | --layer 1 --config cap_mult=8,step=5.0] [--out results/verification]
Writes <out>/real_<world>.json; exit code 0 if every check passes, 1 otherwise.
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

NEW_FAMILIES = ("WF", "PF6", "PF5", "IF5", "CF14")
LEAK_FAMILIES = ("RN", "RD", "SS", "RB", "RC", "PX")


def _logits(model, ids_list):
    out = []
    with torch.no_grad():
        for ids in ids_list:
            out.append(model(torch.tensor([ids], dtype=torch.long))[0][0].clone())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--edits", type=int, default=10)
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selection", action="store_true", help="use STD's layer/config from results/selection.json")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--v2-entities", type=int, default=20)
    ap.add_argument("--out", default=str(STUDY / "results" / "verification"))
    args = ap.parse_args(argv)

    from cev11 import experiment as E
    from cev11 import probes as P
    from cev11.decode import clear_cache, exact_check_batch, greedy_ids, greedy_reproduces
    from cev11.keysites import fc2_inputs, sites
    from cev11.memory import FFNMemory
    from cev11.model import model_sha256
    from cev11.morph_tr import CASES
    from cev11.tokenizers import classify_informal
    from cev11.world import frame_prefix
    from tokenizer_management.byte_bpe import BOS_ID
    from run_confirm import hp_from_config

    if args.selection:
        sel = json.loads((STUDY / "results" / "selection.json").read_text(encoding="utf-8"))["arms"]["STD"]
        layer, cfg = int(sel["layer"]), sel["config"]
    else:
        if args.layer is None or args.config is None:
            raise SystemExit("give --selection or both --layer and --config")
        layer, cfg = args.layer, args.config
    hp = hp_from_config(cfg)
    base_dir = pathlib.Path(args.base)
    base = E.load_base(base_dir if base_dir.is_absolute() else STUDY / base_dir, REPO)
    gates = json.loads((base.base_dir / "gates.json").read_text(encoding="utf-8"))
    editable = gates.get("editable") or gates["G7"].get("editable")
    model, tok, w = base.model, base.tok, base.world
    n_layers = len(model.layers)
    t0 = time.time()
    report = {"world": args.world, "base_sha": base.base_sha, "layer": layer, "config": cfg, "frac": args.frac,
              "seed": args.seed}

    # ---------------------------------------------------------------------------------------------- V2 (unedited)
    rnd = random.Random(args.seed)
    probe_pool = sorted(n for n, p in w.partition.items() if p == "probe")
    ents = rnd.sample(probe_pool, min(args.v2_entities, len(probe_pool)))
    v2_trunc_max, v2_full_rel, v2_n = 0.0, 0.0, 0
    for name in ents:
        forms = [("formal", c) for c in CASES]
        forms += [("informal", c) for c in CASES if c != "nom" and classify_informal(tok, name, c) == "IDENT"]
        prefixes = [frame_prefix(name, "A", c, orth, "none") for orth, c in forms]
        roots = [sites(tok, p, name, "none")["ROOT"] for p in prefixes]
        full = [[BOS_ID] + list(tok.encode(p)) for p in prefixes]
        trunc = [ids[: r + 1] for ids, r in zip(full, roots)]
        for L in range(n_layers):
            kt = [h[r] for h, r in zip(fc2_inputs(model, L, trunc), roots)]
            kf = [h[r] for h, r in zip(fc2_inputs(model, L, full, batch_size=len(full)), roots)]
            for a, b in zip(kt, kf):
                v2_trunc_max = max(v2_trunc_max, float((a - kt[0]).abs().max()))
                v2_full_rel = max(v2_full_rel, float((b - kt[0]).norm() / kt[0].norm()))
        v2_n += len(forms)
    report["V2"] = {"entities": len(ents), "forms": v2_n, "layers": n_layers,
                    "truncated_max_abs_diff": v2_trunc_max, "full_padded_max_rel_diff": v2_full_rel,
                    "pass": v2_trunc_max == 0.0 and v2_full_rel <= 1e-5}

    # ---------------------------------------------------------------------------------------------- edits
    items = P.edit_items(w, editable, seed=args.world + 2, per_case=10)
    items = items[: args.edits]
    answers = E.Answers(base)
    inst = E.Installer(base)
    preps = [E.prepare_item(base, it, answers, seed=args.world + 3) for it in items]
    answers.locked = True

    def probe_list(prep):
        out = []
        for fam in NEW_FAMILIES + LEAK_FAMILIES:
            for d in prep["fam"].get(fam, []):
                cand = prep["item"]["new"] if fam in NEW_FAMILIES else d.get("base_answer")
                if cand is not None:
                    out.append((fam, d["prefix"], cand))
        return out

    # V1: zero memory at every layer (bitwise), then detach (bitwise)
    v1_prefixes = [[BOS_ID] + list(tok.encode(p)) for _, p, _ in probe_list(preps[0])[:40]]
    ref = _logits(model, v1_prefixes)
    mems = [FFNMemory(model, L) for L in range(n_layers)]
    for m in mems:
        m.zero()
    z = _logits(model, v1_prefixes)
    for m in mems:
        m.remove()
    d = _logits(model, v1_prefixes)
    v1_zero = max(float((a - b).abs().max()) for a, b in zip(ref, z))
    v1_detach = max(float((a - b).abs().max()) for a, b in zip(ref, d))
    report["V1"] = {"prefixes": len(v1_prefixes), "zero_memory_max_abs_dlogit": v1_zero,
                    "detached_max_abs_dlogit": v1_detach, "pass": v1_zero == 0.0 and v1_detach == 0.0}

    # V6 + V8 on the edited model (STD write at the given layer/config) and on the unedited model
    v8 = {}
    v6_n, v6_bad = 0, 0
    srnd = random.Random(args.seed + 1)

    def sample_and_check(probes, tag):
        nonlocal v6_n, v6_bad
        k = max(1, round(args.frac * len(probes)))
        pick = srnd.sample(probes, k)
        exact = exact_check_batch(model, tok, [p for _, p, _ in pick], [c for _, _, c in pick])
        for (fam, prefix, cand), ex in zip(pick, exact):
            true = greedy_reproduces(model, tok, prefix, cand, use_cache=False)
            key = f"{tag}:{fam}"
            s = v8.setdefault(key, {"n": 0, "disagree": 0})
            s["n"] += 1
            s["disagree"] += int(bool(ex) != bool(true))
            ids = [BOS_ID] + list(tok.encode(prefix))
            a = greedy_ids(model, ids, max_new=8, use_cache=True)
            clear_cache(model)
            b = greedy_ids(model, ids, max_new=8, use_cache=False)
            v6_n += 1
            v6_bad += int(a != b)

    memo: dict = {}
    for prep in preps:
        probes = probe_list(prep)
        sample_and_check(probes, "base")
        wr = E.compute_write(base, prep["item"], "STD", layer, hp, args.world, memo)
        inst.set_M(layer, wr["M"])
        sample_and_check(probes, "edited")
        inst.zero_all()
    inst.restore()
    tot_n = sum(s["n"] for s in v8.values())
    tot_bad = sum(s["disagree"] for s in v8.values())
    fam_rate = {k: s["disagree"] / s["n"] for k, s in sorted(v8.items())}
    report["V8"] = {"families": {k: {**v8[k], "rate": fam_rate[k]} for k in sorted(v8)}, "n": tot_n,
                    "disagree": tot_bad, "rate": tot_bad / max(tot_n, 1),
                    "pass": tot_bad / max(tot_n, 1) <= 0.01 and all(r <= 0.01 for r in fam_rate.values())}
    report["V6"] = {"n": v6_n, "mismatches": v6_bad, "pass": v6_bad == 0}
    if model_sha256(model) != base.base_sha:
        raise SystemExit("base model changed during verification")
    report["all_pass"] = all(report[k]["pass"] for k in ("V1", "V2", "V6", "V8"))
    report["seconds"] = round(time.time() - t0, 1)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"real_{args.world}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: (v["pass"] if isinstance(v, dict) and "pass" in v else v) for k, v in report.items()
                      if k in ("V1", "V2", "V6", "V8", "all_pass", "seconds")}), flush=True)
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
