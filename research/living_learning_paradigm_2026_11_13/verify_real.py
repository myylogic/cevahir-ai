# -*- coding: utf-8 -*-
"""Real-model verification checks V1, V2, V6 and V8 (PROTOCOL §10.3) on a study base with real STD writes.

Added after the freeze as a verification tool; it imports the frozen modules and changes no frozen file, no record
and no analysis (see AMENDMENTS.md). The unit tests in tests/evolution/test_cev11_*.py cover the same checks on tiny
models; this script repeats them on the study's own bases and edits:

- V1: with an all-zero memory installed at every layer, max|Δlogit| over probe prefixes is exactly 0.0; detaching
  the hooks restores exactly 0.0; after a real write is installed and zeroed, and after the installer's restore,
  the logits are again bitwise equal to the unhooked base.
- V2: fc2 inputs at the root-final token are bitwise identical across the 7 formal forms and the IDENT informal
  forms on truncated prefixes (every layer), and each full, right-padded frame query reproduces its own truncated
  key within relative 1e-5.
- V6: cached and uncached greedy decoding give identical outputs on the sampled probes (edited and unedited model).
- V8 (pre-run sample): for each probe family, a random share (default 5%, at least one probe) of the probes is
  taken from the SAME family-level batched exact check that scoring uses (measure / _changed composition: WF, PF6,
  OLD on PF6, PF5, IF5, CF14, RN adoption, and the leak families scored by the batched check, i.e. probes with a
  base answer and exact_ok), on the unedited model and under one STD write per edit, and compared with true
  (uncached) greedy decoding. Families with any disagreement are listed as `true_greedy_families` (the protocol's
  remedy: that family is scored by true greedy decoding). Pass: overall disagreement ≤ 1%, and ≤ 1% for every
  family with at least 100 sampled probes.

Exit code: 1 if V1 or V2 fails (deterministic checks whose failure would invalidate the write mechanics), else 0.
V6 and V8 outcomes are recorded with every disagreement; like any failed check they block publication of the H and
S results until resolved (PROTOCOL §10.3), but they do not stop the runs.

Usage: python verify_real.py --base results/bases/confirm_w1000_K20 --world 1000 [--edits 10] [--frac 0.05]
           [--selection | --layer 1 --config cap_mult=8,step=5.0] [--out results/verification]
Writes <out>/real_<world>.json.
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

LEAK_FAMILIES = ("RN", "RD", "SS", "RB", "RC", "PX")


def _logits(model, ids_list):
    out = []
    with torch.no_grad():
        for ids in ids_list:
            out.append(model(torch.tensor([ids], dtype=torch.long))[0][0].clone())
    return out


def _maxdiff(a_list, b_list) -> float:
    return max(float((a - b).abs().max()) for a, b in zip(a_list, b_list))


def scored_units(E, base, prep):
    """(family, prefix, candidate, batched_result) for every probe scored by the batched exact check, with the same
    per-family batch composition as experiment.measure / experiment._changed. Leak probes without a base answer or
    with exact_ok False are scored by true greedy decoding in the study and are skipped here (counted)."""
    fam, item = prep["fam"], prep["item"]
    new, old = item["new"], item["old"]
    units, skipped = [], 0

    def add(label, probes, cands):
        if not probes:
            return
        res = E._exact(base, [d["prefix"] for d in probes], cands)
        units.extend((label, d["prefix"], c, bool(r)) for d, c, r in zip(probes, cands, res))

    add("WF", fam["WF"][:1], [new])
    add("PF6", fam["PF6"], [new] * len(fam["PF6"]))
    add("OLD", fam["PF6"], [old] * len(fam["PF6"]))
    add("PF5", fam["PF5"], [new] * len(fam["PF5"]))
    add("IF5", fam["IF5"], [new] * len(fam["IF5"]))
    add("CF14", fam["CF14"], [new] * len(fam["CF14"]))
    for f in LEAK_FAMILIES:
        probes = fam.get(f, [])
        clean = [d for d in probes if d["base_answer"] is not None]
        # the study batches all clean probes together; only exact_ok ones are scored by the batched result
        res = E._exact(base, [d["prefix"] for d in clean], [d["base_answer"] for d in clean]) if clean else []
        for d, r in zip(clean, res):
            if d.get("exact_ok", True):
                units.append((f, d["prefix"], d["base_answer"], bool(r)))
            else:
                skipped += 1
        skipped += len(probes) - len(clean)
        if f == "RN" and probes:
            add("RN-adopt", probes, [new] * len(probes))
    return units, skipped


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
    from cev11.decode import clear_cache, greedy_ids
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
              "seed": args.seed, "edits": args.edits}

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
                v2_trunc_max = max(v2_trunc_max, float((a - kt[0]).abs().max()))   # identity across forms
                v2_full_rel = max(v2_full_rel, float((b - a).norm() / a.norm()))    # padding vs own key
        v2_n += len(forms)
    report["V2"] = {"entities": len(ents), "forms": v2_n, "layers": n_layers,
                    "truncated_max_abs_diff_across_forms": v2_trunc_max,
                    "full_padded_max_rel_diff_vs_own_truncated": v2_full_rel,
                    "pass": v2_trunc_max == 0.0 and v2_full_rel <= 1e-5}

    # ---------------------------------------------------------------------------------------------- edits
    items = P.edit_items(w, editable, seed=args.world + 2, per_case=10)[: args.edits]
    answers = E.Answers(base)
    inst = E.Installer(base)
    preps = [E.prepare_item(base, it, answers, seed=args.world + 3) for it in items]
    answers.locked = True

    # V1: zero memory at every layer (bitwise), detach (bitwise)
    v1_prefixes = [[BOS_ID] + list(tok.encode(p)) for _, p, _, _ in scored_units(E, base, preps[0])[0][:40]]
    ref = _logits(model, v1_prefixes)
    mems = [FFNMemory(model, L) for L in range(n_layers)]
    for m in mems:
        m.zero()
    v1_zero = _maxdiff(ref, _logits(model, v1_prefixes))
    for m in mems:
        m.remove()
    v1_detach = _maxdiff(ref, _logits(model, v1_prefixes))

    # V6 + V8: per family, sampled from the family-level batched results, on the unedited and the edited model
    srnd = random.Random(args.seed + 1)
    v8, disagreements, v6_mismatch = {}, [], []
    v6_n = 0
    skipped_true_greedy = 0
    v1_after_write = None

    def sample_and_check(prep, tag):
        nonlocal v6_n, skipped_true_greedy
        units, skipped = scored_units(E, base, prep)
        skipped_true_greedy += skipped
        by_fam = {}
        for u in units:
            by_fam.setdefault(u[0], []).append(u)
        for fam_label in sorted(by_fam):
            us = by_fam[fam_label]
            for fam_name, prefix, cand, ex in srnd.sample(us, max(1, round(args.frac * len(us)))):
                ids = [BOS_ID] + list(tok.encode(prefix))
                t = tok.encode(" " + cand + ".")
                clear_cache(model)
                a = greedy_ids(model, ids, max_new=8, use_cache=True)
                clear_cache(model)
                b = greedy_ids(model, ids, max_new=8, use_cache=False)
                true = len(t) <= 8 and b[: len(t)] == t      # == decode.greedy_reproduces (uncached), one decode
                s = v8.setdefault(fam_name, {"n": 0, "disagree": 0})
                s["n"] += 1
                if bool(ex) != bool(true):
                    s["disagree"] += 1
                    disagreements.append({"state": tag, "family": fam_name, "edit_id": prep["item"]["edit_id"],
                                          "prefix": prefix, "candidate": cand, "batched": bool(ex),
                                          "true_greedy": bool(true)})
                v6_n += 1
                if a != b:
                    v6_mismatch.append({"state": tag, "edit_id": prep["item"]["edit_id"], "prefix": prefix,
                                        "cached": a, "uncached": b})

    memo: dict = {}
    for i, prep in enumerate(preps):
        sample_and_check(prep, "base")
        wr = E.compute_write(base, prep["item"], "STD", layer, hp, args.world, memo)
        inst.set_M(layer, wr["M"])
        sample_and_check(prep, "edited")
        inst.zero_all()
        if i == 0:      # V1 after a real write was installed and zeroed
            v1_after_write = _maxdiff(ref, _logits(model, v1_prefixes))
    inst.restore()
    v1_restore = _maxdiff(ref, _logits(model, v1_prefixes))
    report["V1"] = {"prefixes": len(v1_prefixes), "zero_memory_max_abs_dlogit": v1_zero,
                    "detached_max_abs_dlogit": v1_detach, "after_write_zeroed_max_abs_dlogit": v1_after_write,
                    "after_restore_max_abs_dlogit": v1_restore,
                    "pass": v1_zero == 0.0 and v1_detach == 0.0 and v1_after_write == 0.0 and v1_restore == 0.0}

    tot_n = sum(s["n"] for s in v8.values())
    tot_bad = sum(s["disagree"] for s in v8.values())
    fams = {k: {**s, "rate": s["disagree"] / s["n"]} for k, s in sorted(v8.items())}
    report["V8"] = {"kind": "pre-run sample (STD writes and the unedited model), not every scored probe",
                    "families": fams, "n": tot_n, "disagree": tot_bad, "rate": tot_bad / max(tot_n, 1),
                    "skipped_scored_by_true_greedy": skipped_true_greedy,
                    "true_greedy_families": sorted(k for k, s in v8.items() if s["disagree"]),
                    "disagreements": disagreements,
                    "pass": tot_bad <= 0.01 * tot_n and all(s["disagree"] <= 0.01 * s["n"] for s in v8.values()
                                                            if s["n"] >= 100)}
    report["V6"] = {"n": v6_n, "mismatches": len(v6_mismatch), "examples": v6_mismatch[:20],
                    "pass": not v6_mismatch}
    if model_sha256(model) != base.base_sha:
        raise SystemExit("base model changed during verification")
    report["all_pass"] = all(report[k]["pass"] for k in ("V1", "V2", "V6", "V8"))
    report["seconds"] = round(time.time() - t0, 1)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"real_{args.world}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k]["pass"] for k in ("V1", "V2", "V6", "V8")} |
                     {"all_pass": report["all_pass"], "v8_n": tot_n, "v8_disagree": tot_bad,
                      "seconds": report["seconds"]}), flush=True)
    if not report["all_pass"]:
        print("WARNING: a verification check failed; H/S results must not be published until it is resolved",
              flush=True)
    return 0 if (report["V1"]["pass"] and report["V2"]["pass"]) else 1


if __name__ == "__main__":
    sys.exit(main())
