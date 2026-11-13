# -*- coding: utf-8 -*-
"""Düzenleme motoru: bir temel modelde bir kolu bir maddeye uygular ve yoklama ailelerini ölçer (PROTOCOL §7–§9).

Akış (her madde için):
1. ``prepare_item`` — TEMEL modelle (bellek sıfır, FT durumu yüklü değil): hedef hâller, komşu kümeleri, yoklama
   aileleri ve sızıntı yoklamalarının temel açgözlü cevapları. Temel cevaplar bir düzenleme kuruluyken asla
   hesaplanmaz; aksi hâlde "değişen cevap" ölçümü düzenlenmiş modele göre yapılırdı.
2. ``apply_arm`` — kolun yazımını hesaplar (anahtar, v* araması / ince ayar) ve kurar.
3. ``measure`` — kurulu durumda tüm aileleri toplu tam-üretim denetimiyle ölçer.
4. Kurulum geri alınır (bellek sıfırlanır / durum geri yüklenir) ve ``analyze.make_edit_record`` ile kayıt üretilir.

Kollar: STD, ROOT, LAST, PARA-ORTH (STD'nin artığını kendi anahtarıyla kullanır, PROTOCOL ön-dondurma (d)),
FT-L, FT-ALL, NORM (STD'nin yazımı + gündelik yoklamalarda kesme işaretinin geri getirilmesi), BASE.
"""
from __future__ import annotations

import copy
import json
import math
import pathlib
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .morph_tr import CASES, INFLECTED
from .world import CONTEXTS, frame_prefix

KEY_ARMS = ("STD", "ROOT", "LAST", "PARA-ORTH")
SITE_OF = {"STD": "STD", "ROOT": "ROOT", "LAST": "LAST", "PARA-ORTH": "STD", "NORM": "STD", "PRED": "PRED"}


# --------------------------------------------------------------------------------------------------- base loading
@dataclass
class Base:
    model: torch.nn.Module
    cfg: dict
    tok: object
    meta: dict
    world: object
    prose_train: List[str]
    prose_held: List[str]
    base_dir: pathlib.Path
    base_sha: str = ""
    layer_cache: dict = field(default_factory=dict)


def load_base(base_dir, repo_root) -> Base:
    from .model import load_checkpoint, model_sha256
    from .world import load_prose, world_from_json
    base_dir = pathlib.Path(base_dir)
    model, cfg, tok, meta = load_checkpoint(base_dir / "model.pt")
    model.eval()
    world = world_from_json(json.loads((base_dir / "world_full.json").read_text(encoding="utf-8")))
    prose_train, prose_held = load_prose(pathlib.Path(repo_root))
    return Base(model, cfg, tok, meta, world, prose_train, prose_held, base_dir, model_sha256(model))


# --------------------------------------------------------------------------------------------- layer statistics
def c0_sequences(base: Base, seed: int, n_tokens: int = 20000) -> List[List[int]]:
    """C0 havuzu: ~%50 C0-havuz varlıklarının cümleleri (tüm kalıplar, bağlamlar, yazımlar) + ~%50 eğitim düzyazısı.
    Düzenleme ve yoklama varlıkları bu havuzda yoktur (PROTOCOL §8)."""
    from tokenizer_management.byte_bpe import BOS_ID
    from .world import sentence
    rnd = random.Random(seed)
    w = base.world
    names = sorted(n for n, p in w.partition.items() if p == "c0")
    cells = [(n, r, c, o, x) for n in names for r in ("A", "B", "C") for c in CASES
             for o in (("formal",) if c == "nom" else ("formal", "informal")) for x in CONTEXTS]
    rnd.shuffle(cells)
    seqs, tok_count = [], 0
    for n, r, c, o, x in cells:
        ids = [BOS_ID] + base.tok.encode(sentence(n, r, c, w.facts[(n, r)], o, x))
        seqs.append(ids); tok_count += len(ids)
        if tok_count >= n_tokens // 2:
            break
    prose = list(base.prose_train)
    rnd.shuffle(prose)
    p_count = 0
    for para in prose:
        ids = [BOS_ID] + base.tok.encode(para)[:63]
        seqs.append(ids); p_count += len(ids)
        if p_count >= n_tokens // 2:
            break
    return seqs


def layer_stats(base: Base, layer: int, seed: int) -> dict:
    """Katman başına C0 (float64, sırt eklenmiş), tersi ve fc2 çıktı normunun medyanı; önbelleklenir."""
    if layer in base.layer_cache:
        return base.layer_cache[layer]
    from tokenizer_management.byte_bpe import PAD_ID
    from .memory import add_ridge, key_covariance
    from .vstar import median_fc2_output_norm
    seqs = c0_sequences(base, seed)
    batches = [[s] for s in seqs]
    C = key_covariance(base.model, layer, batches, exclude_ids=(PAD_ID,))   # BOS dahil (ön-dondurma (c))
    C0 = add_ridge(C, 1e-4)
    med = median_fc2_output_norm(base.model, layer, batches, exclude_ids=(PAD_ID,))
    st = {"C0": C0, "C0_inv": torch.linalg.inv(C0), "median_fc2_norm": float(med),
          "W": base.model.layers[layer].ffn.fc2.weight.detach().double().clone()}
    base.layer_cache[layer] = st
    return st


# ------------------------------------------------------------------------------------------------ base answers
class Answers:
    """Temel modelin açgözlü nesne cevapları (önbellekli). Yalnız temel durumdayken çağrılmalı."""

    def __init__(self, base: Base):
        self.base = base
        self.cache: Dict[str, Optional[str]] = {}
        self.locked = False

    def __call__(self, prefix: str) -> Optional[str]:
        if prefix in self.cache:
            return self.cache[prefix]
        if self.locked:
            raise RuntimeError("base answer requested while an edit is installed")
        from .decode import greedy_answer
        a = greedy_answer(self.base.model, self.base.tok, prefix)
        self.cache[prefix] = a
        return a


# ------------------------------------------------------------------------------------------------ item preparation
LEAK_FAMILIES = ("RN", "RD", "SS", "RB", "RC", "PX")


def mark_base_consistency(base: Base, probes: List[dict]) -> int:
    """TEMEL durumda çağrılır: toplu tam-üretim denetimi, temel modelin kendi açgözlü cevabını (gerçek çözümlemeyle
    bulunmuş) yeniden üretmiyorsa yoklamayı ``exact_ok = False`` diye işaretler (ör. kanonik olmayan token dizisi ya
    da toplu/tekil kayan nokta farkıyla argmax eşitliği). ``_changed`` bu yoklamalarda gerçek açgözlü cevabı
    karşılaştırır; böylece temel modelin "değişen cevap" sayısı tanım gereği 0 olur (BASE kolu, seçim kısıtı
    Leak_RD ≤ %1). ``_changed`` ile aynı liste ve aynı toplu bileşim kullanılır. Döner: işaretlenen yoklama sayısı."""
    clean = [d for d in probes if d["base_answer"] is not None]
    keep = _exact(base, [d["prefix"] for d in clean], [d["base_answer"] for d in clean])
    bad = 0
    for d, k in zip(clean, keep):
        d["exact_ok"] = bool(k)
        bad += int(not k)
    return bad


def prepare_item(base: Base, item: dict, answers: Answers, seed: int) -> dict:
    from .probes import neighbour_sets, probe_families, target_cases
    from .tokenizers import classify_informal, last_root_token
    if answers.locked:
        raise RuntimeError("prepare_item must run on the BASE model (an edit is installed)")
    rnd = random.Random(f"{seed}|{item['edit_id']}")
    targets = target_cases(item, answers)
    neigh = neighbour_sets(base.world, base.tok, item, rnd, last_root_token)
    fam = probe_families(base.world, base.tok, item, targets, neigh, classify_informal)
    from .decode import greedy_text
    for f in LEAK_FAMILIES:
        for d in fam[f]:
            d["base_answer"] = answers(d["prefix"])
            if d["base_answer"] is None:      # no '.' within 8 tokens: keep the raw base continuation
                d["base_text"] = greedy_text(base.model, base.tok, d["prefix"])
    n_inconsistent = sum(mark_base_consistency(base, fam[f]) for f in LEAK_FAMILIES)
    for d in fam["PF6"]:
        d["base_answer"] = answers(d["prefix"])
    return {"item": item, "targets": targets, "neigh": neigh, "fam": fam, "base_inconsistent": n_inconsistent}


# --------------------------------------------------------------------------------------------------- measurement
def _exact(base: Base, prefixes: List[str], objects: List[Optional[str]]) -> List[bool]:
    from .decode import exact_check_batch
    idx = [i for i, o in enumerate(objects) if o is not None]
    out = [False] * len(prefixes)
    if idx:
        res = exact_check_batch(base.model, base.tok, [prefixes[i] for i in idx], [objects[i] for i in idx])
        for i, r in zip(idx, res):
            out[i] = bool(r)
    return out


def _changed(base: Base, probes: List[dict]) -> Tuple[int, int, int]:
    """(changed, n, adopted_unused): temel cevabı artık üretmeyen yoklama sayısı. Temel cevabı None olan (8 token
    içinde '.' üretmeyen) yoklamalar gerçek açgözlü metinle karşılaştırılır; temel modelin toplu denetimde kendi
    cevabını yeniden üretemediği yoklamalar (``exact_ok`` False, bkz. ``mark_base_consistency``) gerçek açgözlü
    cevapla karşılaştırılır."""
    from .decode import greedy_answer, greedy_text
    clean = [d for d in probes if d["base_answer"] is not None]
    keep = _exact(base, [d["prefix"] for d in clean], [d["base_answer"] for d in clean])
    changed = 0
    for d, k in zip(clean, keep):
        if d.get("exact_ok", True):
            changed += int(not k)
        else:
            changed += int(greedy_answer(base.model, base.tok, d["prefix"]) != d["base_answer"])
    for d in probes:
        if d["base_answer"] is None:
            changed += int(greedy_text(base.model, base.tok, d["prefix"]) != d["base_text"])
    return changed, len(probes), 0


def measure(base: Base, prep: dict, new: str, old: str, informal_prefix_map=None, families=None) -> dict:
    """Kurulu durumda ölçüm. ``informal_prefix_map``: NORM için gündelik yoklama metnini dönüştüren işlev."""
    fam = prep["fam"]
    families = families or ("WF", "PF6", "PF5", "IF5", "CF14", "RN", "RD", "SS", "RB", "RC", "PX")
    out: dict = {}
    if "WF" in families:
        out["installed"] = _exact(base, [fam["WF"][0]["prefix"]], [new])[0]
    if "PF6" in families:
        pf = fam["PF6"]
        res = _exact(base, [d["prefix"] for d in pf], [new] * len(pf))
        out["PF6"] = {d["case"]: int(r) for d, r in zip(pf, res)}
        olds = _exact(base, [d["prefix"] for d in pf], [old] * len(pf))
        out["OLD"] = {"count": int(sum(olds)), "n": len(pf)}
    if "PF5" in families or "IF5" in families:
        f_res = _exact(base, [d["prefix"] for d in fam["PF5"]], [new] * len(fam["PF5"]))
        inf_prefixes = [d["prefix"] if informal_prefix_map is None else informal_prefix_map(d["prefix"])
                        for d in fam["IF5"]]
        i_res = _exact(base, inf_prefixes, [new] * len(fam["IF5"]))
        out["targets"] = [{"case": a["case"], "cls": b["cls"], "formal": int(fr), "informal": int(ir)}
                          for a, b, fr, ir in zip(fam["PF5"], fam["IF5"], f_res, i_res)]
    if "CF14" in families:
        cf = fam["CF14"]
        res = _exact(base, [d["prefix"] for d in cf], [new] * len(cf))
        out["CF14"] = {"P1": {}, "P2": {}}
        for d, r in zip(cf, res):
            out["CF14"][d["ctx"]][d["case"]] = int(r)
    leak = {}
    for f in ("RN", "RD", "SS", "RB", "RC", "PX"):
        if f not in families:
            continue
        probes = fam[f]
        if f in ("RB", "RC"):
            ch, n, _ = _changed(base, probes)
            leak[f] = {"changed": ch, "n": n}
        elif f == "PX" and not probes:
            continue
        else:
            ch, n, _ = _changed(base, probes)
            ent = len(prep["neigh"][f]) if f in prep["neigh"] else 0
            leak[f] = {"changed": ch, "n": n, "entities": ent}
            if f == "RN":
                leak[f]["adopted"] = int(sum(_exact(base, [d["prefix"] for d in probes], [new] * len(probes))))
    out["leak"] = leak
    return out


# ------------------------------------------------------------------------------------------------------- writes
def _target_ids(base: Base, new: str) -> List[int]:
    return base.tok.encode(" " + new + ".")


def _kl_ids(base: Base, item: dict) -> List[int]:
    from tokenizer_management.byte_bpe import BOS_ID
    return [BOS_ID] + base.tok.encode(frame_prefix(item["name"], "B", item["lived_case"], "formal", "none"))


def compute_write(base: Base, item: dict, arm: str, layer: int, hp: dict, world_seed: int,
                  memo: Optional[dict] = None) -> dict:
    """Kolun ΔW'sini hesaplar (kurmaz). Döner: {"M": tensor | None, "state": state_dict | None, "k": key,
    "diag": {...}}. PARA-ORTH için {"M_all": ..., "M_variants": {(case, orth): M}}."""
    from .keysites import key_at, para_orth_from_variants, prose_subset, site_prompts, std_key, variant_std_keys
    from .memory import rome_update
    from .vstar import search_delta, v_star
    from .baselines import ft_all, ft_layer
    model, tok = base.model, base.tok
    L = frame_prefix(item["name"], item["rel"], item["lived_case"], "formal", "none")
    T = _target_ids(base, item["new"])
    kl = _kl_ids(base, item)
    if arm in ("FT-L", "FT-ALL"):
        prompts = site_prompts(tok, L, item["name"], "none", site="PRED")
        info: dict = {}
        if arm == "FT-L":
            dW = ft_layer(model, layer, prompts, T, hp["lr"], hp.get("steps", 60), hp["eps"], kl_prompt_ids=kl,
                          info=info)
            return {"M": dW.double(), "diag": info}
        state = ft_all(model, prompts, T, hp["lr"], hp.get("steps", 40), kl_prompt_ids=kl, info=info)
        return {"state": state, "diag": info}
    st = layer_stats(base, layer, world_seed)
    cap = hp["cap_mult"] * st["median_fc2_norm"]
    site = SITE_OF[arm]
    if site == "STD":
        prompts = site_prompts(tok, L, item["name"], "none", site="STD", copies=prose_subset(world_seed + 4, 4))
        k = std_key(model, layer, tok, L, item["name"], "none")
    else:
        prompts = site_prompts(tok, L, item["name"], "none", site=site)
        k = key_at(model, layer, tok, L, prompts[0][1])
        if site == "PRED":      # the relation-B prompt differs before its last token: give its own position
            kl = (kl, len(kl) - 1)
    delta, info = search_delta(model, layer, prompts, T, kl, hp["step"], cap, max_steps=int(hp.get("max_steps", 120)),
                               memo=memo, memo_context=f"{base.base_sha}|{site}")
    info = {**info, "cap": cap}
    if arm == "PARA-ORTH":
        keys = variant_std_keys(model, layer, tok, item["name"], item["rel"])
        out = {"M_variants": {}, "diag": info, "keys": keys}
        k_all = para_orth_from_variants(keys, None)
        out["M_all"] = rome_update(st["W"], st["C0"], k_all, v_star(st["W"], k_all, delta.double()))
        out["k"] = k_all
        out["delta"] = delta
        return out
    vs = v_star(st["W"], k.double(), delta.double())
    return {"M": rome_update(st["W"], st["C0"], k.double(), vs), "k": k.double(), "delta": delta, "diag": info}


def para_orth_variant_M(base: Base, write: dict, layer: int, world_seed: int, variant) -> torch.Tensor:
    from .keysites import para_orth_from_variants
    from .memory import rome_update
    from .vstar import v_star
    st = layer_stats(base, layer, world_seed)
    key = tuple(variant)
    if key not in write["M_variants"]:
        k = para_orth_from_variants(write["keys"], key)
        write["M_variants"][key] = rome_update(st["W"], st["C0"], k, v_star(st["W"], k, write["delta"].double()))
    return write["M_variants"][key]


class Installer:
    """Bir katmanda FFNMemory ve FT-ALL durumunun kurulup geri alınması."""

    def __init__(self, base: Base):
        self.base = base
        self.mems: dict = {}
        self.base_state = {k: v.clone() for k, v in base.model.state_dict().items()}

    def mem(self, layer: int):
        from .memory import FFNMemory
        if layer not in self.mems:
            self.mems[layer] = FFNMemory(self.base.model, layer)
        return self.mems[layer]

    def set_M(self, layer: int, M: torch.Tensor):
        self.mem(layer).set_M(M.float())

    def zero_all(self):
        for m in self.mems.values():
            m.zero()

    def load_state(self, state: dict):
        self.base.model.load_state_dict(state, strict=True)
        from .decode import clear_cache
        clear_cache(self.base.model)

    def restore(self):
        self.base.model.load_state_dict(self.base_state, strict=True)
        from .decode import clear_cache
        clear_cache(self.base.model)
        self.zero_all()


def run_arm(base: Base, inst: Installer, answers: Answers, prep: dict, arm: str, layer: Optional[int], hp: dict,
            world_seed: int, run: str, config: str, memo: Optional[dict] = None, families=None,
            with_tau: bool = False, bpb_texts: Optional[List[str]] = None) -> dict:
    """Bir kolu bir maddeye uygular, ölçer, geri alır ve ``analyze.make_edit_record`` kaydı döndürür."""
    import analyze as A
    from .baselines import normalize_informal
    item = prep["item"]
    t0 = time.time()
    diag: dict = {}
    answers.locked = True
    try:
        if arm == "BASE":
            res = measure(base, prep, item["new"], item["old"], families=families)
        elif arm == "FT-ALL":
            w = compute_write(base, item, arm, None, hp, world_seed)
            diag = w["diag"]
            inst.load_state(w["state"])
            res = measure(base, prep, item["new"], item["old"], families=families)
            if bpb_texts is not None:
                res["bpb"] = {"base": base.layer_cache["bpb_base"], "edited": bits_per_byte(base, bpb_texts)}
            inst.restore()
        elif arm == "PARA-ORTH":
            w = compute_write(base, item, arm, layer, hp, world_seed, memo)
            diag = w["diag"]
            inst.set_M(layer, w["M_all"])
            res = measure(base, prep, item["new"], item["old"],
                          families=[f for f in (families or ("WF", "PF6", "CF14", "RN", "RD", "SS", "RB", "RC", "PX"))
                                    if f not in ("PF5", "IF5")])
            # leave-one-variant-out for every tested target variant
            tg = []
            for a, b in zip(prep["fam"]["PF5"], prep["fam"]["IF5"]):
                inst.set_M(layer, para_orth_variant_M(base, w, layer, world_seed, (a["case"], "formal")))
                fr = _exact(base, [a["prefix"]], [item["new"]])[0]
                inst.set_M(layer, para_orth_variant_M(base, w, layer, world_seed, (b["case"], "informal")))
                ir = _exact(base, [b["prefix"]], [item["new"]])[0]
                tg.append({"case": a["case"], "cls": b["cls"], "formal": int(fr), "informal": int(ir)})
            res["targets"] = tg
            if bpb_texts is not None:      # bpb under the write measured on the other families (all 13 variants)
                inst.set_M(layer, w["M_all"])
                res["bpb"] = {"base": base.layer_cache["bpb_base"], "edited": bits_per_byte(base, bpb_texts)}
            inst.zero_all()
        else:
            w = compute_write(base, item, "STD" if arm == "NORM" else arm, layer, hp, world_seed, memo)
            diag = w["diag"]
            inst.set_M(layer, w["M"])
            fmap = None
            if arm == "NORM":
                lex = set(base.world.names)
                fmap = lambda p: normalize_informal(p, lex)   # noqa: E731
            res = measure(base, prep, item["new"], item["old"], informal_prefix_map=fmap, families=families)
            if bpb_texts is not None:
                res["bpb"] = {"base": base.layer_cache["bpb_base"], "edited": bits_per_byte(base, bpb_texts)}
            inst.zero_all()
            if with_tau and arm in ("STD", "ROOT", "LAST") and res.get("targets"):
                for t, extra in zip(res["targets"], tau_for_targets(base, prep, arm, layer, w["k"], world_seed)):
                    t.update(extra)
    except BaseException:
        # never leave a write installed after a failure: hooks are invisible to the final model_sha256 check, so a
        # caller that skipped a failing item would otherwise measure every later item and arm under this write
        inst.restore()
        raise
    finally:
        answers.locked = False
    diag = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in diag.items()
            if isinstance(v, (int, float, str, bool))}
    diag["seconds"] = round(time.time() - t0, 3)
    diag["base_inconsistent"] = int(prep.get("base_inconsistent", 0))
    targets = res.get("targets", [])
    pf6 = res.get("PF6", {})
    cf14 = res.get("CF14", {"P1": {}, "P2": {}})
    leak = res.get("leak", {})
    if families is not None:          # selection/map runs: partial records are not schema-validated
        return {"run": run, "world": world_seed, "arm": arm, "layer": layer, "config": config, **item,
                "installed": res.get("installed"), "targets": targets, "PF6": pf6, "CF14": cf14, "leak": leak,
                "OLD": res.get("OLD"), "diag": diag}
    extra = {"bpb": res["bpb"]} if "bpb" in res else {}
    return A.make_edit_record(item, run=run, world=world_seed, arm=arm, layer=layer,
                              installed=bool(res["installed"]), targets=targets, pf6=pf6, cf14=cf14, leak=leak,
                              config=config, OLD=res.get("OLD"), diag=diag, **extra)


# ------------------------------------------------------------------------------------------------------- τ values
def tau_for_targets(base: Base, prep: dict, arm: str, layer: int, k: torch.Tensor, world_seed: int) -> list:
    """Betimsel τ (PROTOCOL §9): her hedef hâlin biçimsel/gündelik yoklamasında kolun anahtar konumundaki τ ve
    yoklamanın tüm konumları üzerindeki en büyük τ. Anahtar konumu tanımsızsa (kaynaşık gündelik biçimde ROOT) ilgili
    alan yazılmaz. ``sites(..., strict=False)``: ROOT tanımsız olsa da STD/LAST konumlarının τ'su yazılır."""
    from tokenizer_management.byte_bpe import BOS_ID
    from .keysites import fc2_inputs, sites
    st = layer_stats(base, layer, world_seed)
    u = st["C0_inv"] @ k.double()
    denom = float(k.double() @ u)
    site = SITE_OF.get(arm, "STD")
    out = []
    for a, b in zip(prep["fam"]["PF5"], prep["fam"]["IF5"]):
        entry = {}
        for tag, d in (("formal", a), ("informal", b)):
            ids = [BOS_ID] + base.tok.encode(d["prefix"])
            H = fc2_inputs(base.model, layer, [ids])[0].double()        # [T, f]
            taus = (H @ u) / denom
            entry[f"tau_{tag}_max"] = float(taus[1:].max())
            pos = sites(base.tok, d["prefix"], d["name"], d["ctx"], strict=False)[site]
            if pos is not None:
                entry[f"tau_{tag}"] = float(taus[pos])
        out.append(entry)
    return out


# ------------------------------------------------------------------------------------------------- bits per byte
@torch.no_grad()
def bits_per_byte(base: Base, texts: Sequence[str], max_len: int = 64) -> float:
    """Ayrılmış düzyazıda bayt başına bit (tokenizer'dan bağımsız): Σ NLL (bit) / Σ UTF-8 bayt; 63 tokenlik parçalar."""
    from tokenizer_management.byte_bpe import BOS_ID
    import torch.nn.functional as F
    model = base.model
    nll, nbytes = 0.0, 0
    for t in texts:
        ids = base.tok.encode(t)
        for i in range(0, len(ids), max_len - 1):
            chunk = [BOS_ID] + ids[i:i + max_len - 1]
            x = torch.tensor([chunk])
            logits = model(x[:, :-1])[0]
            nll += float(F.cross_entropy(logits[0].float(), x[0, 1:], reduction="sum"))
        nbytes += len(t.encode("utf-8"))
    return nll / math.log(2) / max(nbytes, 1)
