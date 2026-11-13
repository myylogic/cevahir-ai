# -*- coding: utf-8 -*-
"""S3 yaşamları (PROTOCOL §13): ortak (J) ve bağımsız (I) durum ve sıralı ROME (Q).

Bir yaşam: 60 yoklama-havuzu varlığı üzerinde tohumlu sırada 100 işlem — 60 ilk yazım (A; yaşanan hâller dengeli),
15 ikinci ilişki yazımı (C; A'sı yazılmış varlıklar), 15 A düzeltmesi, 10 A geri alması (düzeltilenlerden ayrık).
Aynı akış üç durum biçimine uygulanır (STD anahtarı, seçilmiş katman ve STD yapılandırması):
  J: float64 defter, varlık başına tek anahtar ve inanılan tüm olgular için W0'a karşı ortak v* (MEMIT-Merge).
     A geri alınınca: varlığın C'si yoksa kaydı silinir (o anahtarda model W0'a döner); varsa ortak değer A = temel
     nesneyle yeniden çözülür.
  I: olgu başına bağımsız ROME farkları (W0'a karşı); düzeltme = değiştir, geri alma = çıkar.
  Q: güncel ağırlıklar üzerinde sıralı ROME (v* güncel modele karşı); geri alma = o anahtara W0·k geri yazılır;
     4 rastgele sırada (betimsel yayılım).
Yaşam sonunda her inanılan olgu (geri alınan A'lar için inanılan = temel nesne) 7 biçimsel kalıpta ölçülür.

Kullanım: python run_life.py --base results/bases/confirm_w1000_K20 --world 1000 --variants J I Q
          python run_life.py --base results/bases/discovery_w0_K20 --world 0 --run select --variants J --lams 0.01 0.1 1
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

N_ENT, N_C, N_CORR, N_RETR = 60, 15, 15, 10
import os
if os.environ.get("CEV11_LIFE_SMOKE"): N_ENT, N_C, N_CORR, N_RETR = 8, 3, 2, 2


def make_stream(world, editable, seed):
    """İşlem akışı: [(op, entity, rel, obj, lived_case)]; bağımlılıklara uyan tohumlu rastgele sıra."""
    from cev11.morph_tr import CASES
    rnd = random.Random(seed)
    ents = rnd.sample(sorted(editable), N_ENT)
    cases = [CASES[i % len(CASES)] for i in range(N_ENT)]
    rnd.shuffle(cases)
    towns, dishes = list(world.objects["A"]), list(world.objects["C"])
    first = {e: rnd.choice([t for t in towns if t != world.facts[(e, "A")]]) for e in ents}
    c_ents = set(rnd.sample(ents, N_C))
    rest = [e for e in ents]
    corr = set(rnd.sample(rest, N_CORR))
    retr = set(rnd.sample([e for e in rest if e not in corr], N_RETR))
    ops_by_ent = {}
    for i, e in enumerate(ents):
        seq = [("write", e, "A", first[e], cases[i])]
        tail = []
        if e in c_ents:
            tail.append(("write", e, "C", rnd.choice([d for d in dishes if d != world.facts[(e, "C")]]),
                         rnd.choice(CASES)))
        if e in corr:
            tail.append(("correct", e, "A", rnd.choice([t for t in towns if t not in (first[e], world.facts[(e, "A")])]),
                         rnd.choice(CASES)))
        if e in retr:
            tail.append(("retract", e, "A", world.facts[(e, "A")], None))
        rnd.shuffle(tail)
        ops_by_ent[e] = seq + tail
    stream, cursors = [], {e: 0 for e in ents}
    while any(cursors[e] < len(ops_by_ent[e]) for e in ents):
        avail = [e for e in ents if cursors[e] < len(ops_by_ent[e])]
        e = rnd.choice(avail)
        stream.append(ops_by_ent[e][cursors[e]]); cursors[e] += 1
    assert len(stream) == N_ENT + N_C + N_CORR + N_RETR
    return stream, ents


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--run", default="confirm", choices=["confirm", "select"])
    ap.add_argument("--variants", nargs="+", default=["J", "I", "Q"])
    ap.add_argument("--lams", type=float, nargs="+", default=None, help="ledger lambda(s) for J (select runs)")
    ap.add_argument("--q-orders", type=int, default=4)
    ap.add_argument("--out", default=str(STUDY / "results" / "lives"))
    args = ap.parse_args(argv)
    if args.run == "confirm" and args.world in (0, 1):
        raise SystemExit("discovery worlds cannot produce confirmatory life records")
    from cev11 import experiment as E
    from cev11.keysites import prose_subset, site_prompts, std_key
    from cev11.memory import EntityJointLedger, Ledger, make_identity
    from cev11.baselines import IndependentState, SequentialState
    from cev11.vstar import search_delta
    from cev11.world import frame_prefix
    from cev11.decode import exact_check_batch
    from tokenizer_management.byte_bpe import BOS_ID

    selection = json.loads((STUDY / "results" / "selection.json").read_text(encoding="utf-8"))
    std = selection["arms"]["STD"]
    layer = std["layer"]
    hp = {k: float(v) for k, v in (p.split("=") for p in std["config"].split(","))}
    base_dir = pathlib.Path(args.base)
    base = E.load_base(base_dir if base_dir.is_absolute() else STUDY / base_dir, REPO)
    gates = json.loads((base.base_dir / "gates.json").read_text(encoding="utf-8"))
    w, tok, model = base.world, base.tok, base.model
    st = E.layer_stats(base, layer, args.world)
    W0, C0 = st["W"], st["C0"]
    cap = hp["cap_mult"] * st["median_fc2_norm"]
    inst = E.Installer(base); mem = inst.mem(layer)
    stream, ents = make_stream(w, gates["editable"], args.world + 5)
    keys = {e: std_key(model, layer, tok, frame_prefix(e, "A", "nom", "formal", "none"), e, "none").double()
            for e in ents}
    copies = prose_subset(args.world + 4, 4)

    def prompts_for(e, facts):
        """facts: {rel: (obj, case)} → (prompts, per-prompt targets, kl prompt)."""
        P, T = [], []
        for rel, (obj, case) in sorted(facts.items()):
            pr = site_prompts(tok, frame_prefix(e, rel, case, "formal", "none"), e, "none", site="STD", copies=copies)
            P += pr; T += [tok.encode(" " + obj + ".")] * len(pr)
        case0 = sorted(facts.values())[0][1]
        kl = [BOS_ID] + tok.encode(frame_prefix(e, "B", case0, "formal", "none"))
        return P, T, kl

    def delta_for(e, facts):
        P, T, kl = prompts_for(e, facts)
        d, info = search_delta(model, layer, P, T, kl, hp["step"], cap)
        return d.double(), info

    probe = sorted(n for n, p in w.partition.items() if p == "probe" and n not in set(ents))
    rd_ents = random.Random(args.world + 3).sample(probe, 10)
    rd_prefixes = [frame_prefix(n, "A", c, "formal", "none") for n in rd_ents for c in
                   ("nom", "gen", "dat", "acc", "loc", "abl", "ins")]
    answers = E.Answers(base)
    rd_base = [answers(p) for p in rd_prefixes]

    def evaluate(variant, believed_end, retracted, corrected_old, order, lam):
        """Kurulu durumda yaşam sonu ölçümü; believed_end: {(e, rel): obj}."""
        from cev11.morph_tr import CASES
        facts = []
        pre, obj = [], []
        for (e, rel), o in sorted(believed_end.items()):
            for c in CASES:
                pre.append(frame_prefix(e, rel, c, "formal", "none")); obj.append(o)
        res = exact_check_batch(model, tok, pre, obj)
        live_keys = {e: keys[e] for (e, _r) in believed_end if (e, "A") not in retracted or (e, "C") in believed_end}
        Cinv = st["C0_inv"]
        for i, ((e, rel), o) in enumerate(sorted(believed_end.items())):
            corr = int(sum(res[7 * i: 7 * i + 7]))
            k = keys[e]; u = Cinv @ k; den = float(k @ u)
            others = [float(keys[m] @ u) / den for m in live_keys if m != e]
            same_entity_live = sum(1 for (m, r) in believed_end if m == e) > 1 and variant != "J"
            mt = max(others + ([1.0] if same_entity_live else [])) if (others or same_entity_live) else None
            facts.append({"fact_id": f"{e}|{rel}", "entity": e, "rel": rel, "believed": o, "correct": corr, "n": 7,
                          "max_tau": mt})
        keep = exact_check_batch(model, tok, [p for p, a in zip(rd_prefixes, rd_base) if a is not None],
                                 [a for a in rd_base if a is not None])
        rd_changed = sum(1 for k_ in keep if not k_)
        r_old = [(e, o) for (e, o) in retracted.items()]
        rres = exact_check_batch(model, tok, [frame_prefix(e, "A", c, "formal", "none") for e, _ in r_old for c in CASES],
                                 [o for _, o in r_old for _ in CASES]) if r_old else []
        cres = exact_check_batch(model, tok, [frame_prefix(e, "A", c, "formal", "none") for e, _ in corrected_old.items() for c in CASES],
                                 [o for _, o in corrected_old.items() for _ in CASES]) if corrected_old else []
        return {"schema": "cev11.life.v1", "run": args.run, "world": args.world, "variant": variant, "order": order,
                "layer": layer, "lam": lam, "facts": facts,
                "retraction_residue": (sum(rres) / len(rres)) if rres else 0.0,
                "correction_residue": (sum(cres) / len(cres)) if cres else 0.0,
                "leak": {"RD": {"changed": rd_changed, "n": len(keep), "entities": len(rd_ents)}}}

    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.run}_{args.world}.jsonl"
    t0 = time.time()
    lams = args.lams or [selection.get("life_lambda", 0.1)]

    def finalize_believed():
        believed, retracted, corrected_old, first_written = {}, {}, {}, {}
        for op, e, rel, o, case in stream:
            if op == "write":
                believed[(e, rel)] = (o, case)
                if rel == "A":
                    first_written[e] = o
            elif op == "correct":
                corrected_old[e] = believed[(e, rel)][0]
                believed[(e, rel)] = (o, case)
            elif op == "retract":
                retracted[e] = believed[(e, rel)][0]
                believed[(e, rel)] = (w.facts[(e, "A")], believed[(e, rel)][1])
        return {k: v[0] for k, v in believed.items()}, retracted, corrected_old

    believed_end, retracted, corrected_old = finalize_believed()
    with path.open("a", encoding="utf-8") as f:
        for variant in args.variants:
            orders = range(args.q_orders) if variant == "Q" else [0]
            for lam in (lams if variant == "J" else [None]):
                for order in orders:
                    tv = time.time()
                    ops = list(stream)
                    if variant == "Q" and order > 0:
                        # reorder: keep each entity's own op order, interleave entities differently
                        rnd = random.Random(args.world + 5 + 1000 * order)
                        per = {}
                        for op in ops:
                            per.setdefault(op[1], []).append(op)
                        cur = {e: 0 for e in per}; ops = []
                        while any(cur[e] < len(per[e]) for e in per):
                            e = rnd.choice([x for x in per if cur[x] < len(per[x])])
                            ops.append(per[e][cur[e]]); cur[e] += 1
                    mem.zero()
                    beliefs = {}       # (e, rel) -> (obj, case)
                    if variant == "J":
                        ident = make_identity(base.base_sha, layer, lam, C0, tok.identity())
                        jl = EntityJointLedger(Ledger(W0, C0, lam, ident))

                        def solve(e, k, believed_rel_obj):
                            facts = {r: (o, beliefs[(e, r)][1]) for r, o in believed_rel_obj.items()}
                            d, _ = delta_for(e, facts)
                            return d          # residual v* - W0 k = delta (v* = W0 k + delta)
                        for op, e, rel, o, case in ops:
                            if op in ("write", "correct"):
                                beliefs[(e, rel)] = (o, case)
                                jl.write(e, rel, o, solve, key=keys[e])
                            else:
                                if (e, "C") in beliefs:
                                    beliefs[(e, "A")] = (w.facts[(e, "A")], beliefs[(e, "A")][1])
                                    jl.write(e, "A", w.facts[(e, "A")], solve, key=keys[e])
                                else:
                                    beliefs.pop((e, "A"), None)
                                    jl.retract(e, "A", solve)
                        mem.set_M(jl.ledger.delta())
                    else:
                        S = IndependentState(W0, C0) if variant == "I" else SequentialState(W0, C0)
                        for op, e, rel, o, case in ops:
                            fid = f"{e}|{rel}"
                            if op == "retract":
                                S.retract(fid)
                            else:
                                if variant == "Q":
                                    mem.set_M(S.delta())       # v* against the CURRENT model
                                d, _ = delta_for(e, {rel: (o, case)})
                                vs = S.v_star(keys[e], d)
                                if op == "write":
                                    S.insert(fid, keys[e], vs)
                                else:
                                    S.correct(fid, vs, k=keys[e])
                        mem.set_M(S.delta())
                    rec = evaluate(variant, believed_end, retracted, corrected_old, order, lam)
                    mem.zero()
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                    e_end = sum(x["correct"] for x in rec["facts"]) / (7 * len(rec["facts"]))
                    print(f"[life w{args.world}] {variant} lam={lam} order={order}: E_end={e_end:.3f} "
                          f"RD={rec['leak']['RD']['changed']}/{rec['leak']['RD']['n']} ({time.time() - tv:.0f}s)",
                          flush=True)
    from cev11.model import model_sha256
    if model_sha256(model) != base.base_sha:
        raise SystemExit("base model changed during the life run")
    print(f"[life w{args.world}] done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
