# -*- coding: utf-8 -*-
"""Düzenleme maddeleri, yoklama aileleri ve τ (CEV-11 PROTOCOL §7 ve §9).

Bir düzenleme maddesi (edit item): düzenlenebilir bir varlık, ilişki A, yaşanan hâl (lived case) ve yeni nesne.
Yaşanan cümle, yaşanan hâlin biçimsel, cümle başı kalıbıdır. Yoklama aileleri:

- WF: yaşanan kalıp (kurulum ölçütü).
- PF6: diğer 6 biçimsel hâl kalıbı (bağlamsız).
- PF5 / IF5: hedef hâller (yaşanan hâl dışındaki, temel modelin eski nesneyi her iki yazımda da ürettiği çekimli hâller)
  biçimsel ve kesme işaretsiz yazımda; her gündelik biçim IDENT ya da NONIDENT diye etiketlenir.
- CF14: 7 biçimsel kalıp × {P1, P2} bağlam öneki.
- RN: düzenlenen adın son kök tokenini paylaşan en çok 8 yoklama-havuzu varlığı, 7 biçimsel kalıp.
- RD: RN dışından 10 rastgele yoklama-havuzu varlığı, 7 biçimsel kalıp.
- SS: yaşanan kalıpta, yaşanan biçimle aynı ek tokenini taşıyan 20 varlık (yalın yaşantıda boş).
- RB / RC: düzenlenen varlığın B ve C ilişkileri, 7 biçimsel kalıp.
- PX: önek-uzantı eşi (keşif amaçlı).

Sızıntı aileleri (RN, RD, SS, RB, RC, PX) temel modelin kendi açgözlü cevabına göre "değişen cevap" oranıyla ölçülür.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import torch

from .morph_tr import CASES, INFLECTED, suffix
from .world import frame_prefix

FORMAL_FRAMES = CASES
CONTEXT_PROBES = ("P1", "P2")


def edit_items(world, editable: Sequence[str], seed: int, per_case: int = 10, relation: str = "A") -> List[dict]:
    """Yaşanan hâl başına `per_case` madde; varlıklar birbirinden farklı; yeni nesne eskisinden farklı, düzgün dağılımlı.

    Deterministiktir (seed). Liste karıştırılmış sırayla döner; her madde tek başına uygulanır.
    """
    rnd = random.Random(seed)
    pool = sorted(editable)
    need = per_case * len(CASES)
    if len(pool) < need:
        raise ValueError(f"need {need} editable entities, have {len(pool)}")
    chosen = rnd.sample(pool, need)
    items = []
    objs = list(world.objects[relation])
    for i, name in enumerate(chosen):
        case = CASES[i // per_case]
        old = world.facts[(name, relation)]
        new = rnd.choice([o for o in objs if o != old])
        items.append({"edit_id": f"w{world.seed}-e{i:03d}", "name": name, "rel": relation, "lived_case": case,
                      "old": old, "new": new})
    rnd.shuffle(items)
    return items


def target_cases(item: dict, base_answer) -> List[str]:
    """Hedef hâller: yaşanan hâl dışındaki çekimli hâllerden, temel modelin eski nesneyi biçimsel VE gündelik
    yazımda ürettikleri. `base_answer(prefix) -> str|None` temel modelin açgözlü nesne cevabıdır."""
    out = []
    for c in INFLECTED:
        if c == item["lived_case"]:
            continue
        f = base_answer(frame_prefix(item["name"], item["rel"], c, "formal"))
        g = base_answer(frame_prefix(item["name"], item["rel"], c, "informal"))
        if f == item["old"] and g == item["old"]:
            out.append(c)
    return out


def suffix_token(tok, name: str, case: str) -> Optional[int]:
    """Biçimsel yazımda kesme işaretli ek parçasının ilk tokeni (yalın hâlde None)."""
    if case == "nom":
        return None
    s = "'" + suffix(name, case)
    ids = tok.encode(name + s)
    root = tok.encode(name)
    if ids[: len(root)] != root:
        raise ValueError(f"root tokens are not a prefix for {name!r}")
    return ids[len(root)] if len(ids) > len(root) else None


def neighbour_sets(world, tok, item: dict, rnd: random.Random, last_root_token) -> Dict[str, List[str]]:
    """RN (son kök tokenini paylaşan ≤8), RD (RN dışı 10 rastgele), SS (aynı ek tokeni, 20), PX (önek eşi)."""
    probe_pool = sorted(n for n, p in world.partition.items() if p == "probe" and n != item["name"])
    lrt = last_root_token(tok, item["name"])
    rn_all = [n for n in probe_pool if last_root_token(tok, n) == lrt]
    rn = rnd.sample(rn_all, min(8, len(rn_all)))
    rest = [n for n in probe_pool if n not in set(rn_all)]
    rd = rnd.sample(rest, min(10, len(rest)))
    ss: List[str] = []
    st = suffix_token(tok, item["name"], item["lived_case"])
    if st is not None:
        same = [n for n in probe_pool if suffix_token(tok, n, item["lived_case"]) == st]
        ss = rnd.sample(same, min(20, len(same)))
    px = [b if a == item["name"] else a for a, b in world.prefix_pairs if item["name"] in (a, b)]
    return {"RN": rn, "RD": rd, "SS": ss, "PX": px, "RN_cluster_size": len(rn_all)}


def probe_families(world, tok, item: dict, targets: Sequence[str], neigh: Dict[str, List[str]],
                   classify_informal) -> Dict[str, List[dict]]:
    """Her aile için yoklama listesi: {"prefix", "name", "rel", "case", "orth", "ctx", "expect"}.

    expect: "new" (genelleme aileleri; yeni nesne üretilmeli) ya da "base" (sızıntı aileleri; temel cevap korunmalı).
    """
    n, rel, lc = item["name"], item["rel"], item["lived_case"]

    def p(name, r, case, orth="formal", ctx="none", expect="new", **extra):
        d = {"prefix": frame_prefix(name, r, case, orth, ctx), "name": name, "rel": r, "case": case,
             "orth": orth, "ctx": ctx, "expect": expect}
        d.update(extra)
        return d

    fam = {
        "WF": [p(n, rel, lc)],
        "PF6": [p(n, rel, c) for c in FORMAL_FRAMES if c != lc],
        "PF5": [p(n, rel, c) for c in targets],
        "IF5": [p(n, rel, c, "informal", cls=classify_informal(tok, n, c)) for c in targets],
        "CF14": [p(n, rel, c, ctx=x) for x in CONTEXT_PROBES for c in FORMAL_FRAMES],
        "RN": [p(m, rel, c, expect="base") for m in neigh["RN"] for c in FORMAL_FRAMES],
        "RD": [p(m, rel, c, expect="base") for m in neigh["RD"] for c in FORMAL_FRAMES],
        "SS": [p(m, rel, lc, expect="base") for m in neigh["SS"]],
        "RB": [p(n, "B", c, expect="base") for c in FORMAL_FRAMES],
        "RC": [p(n, "C", c, expect="base") for c in FORMAL_FRAMES],
        "PX": [p(m, rel, c, expect="base") for m in neigh["PX"] for c in FORMAL_FRAMES],
    }
    for f in ("IF5",):
        for d in fam[f]:
            d["ident"] = (d["cls"] == "IDENT")
    return fam


def tau(k_prime: torch.Tensor, k_bar: torch.Tensor, C0_inv: torch.Tensor) -> float:
    """τ(k', k̄) = k'ᵀ C0⁻¹ k̄ / k̄ᵀ C0⁻¹ k̄ — sıra-bir yazımın k' konumunda ne kadar ateşlendiği."""
    u = C0_inv @ k_bar.double()
    return float((k_prime.double() @ u) / (k_bar.double() @ u))
