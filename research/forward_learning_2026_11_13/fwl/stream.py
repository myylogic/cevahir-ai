# -*- coding: utf-8 -*-
"""Part B: a stream of facts read once each (PROTOCOL §5.2), and on it the read gate (§5.3), the capacity law
(§5.4) and the retraction gap (§5.5).

Every arm sees the same facts in the same order, with no replay, and is scored with the same probes and the same
install criterion. Learner arms use the engine's ``ForwardLearner`` (projector, step search, ledger); the
comparators are sequential FT-L (CEV-11's frozen configuration), a projector-matched exact-gradient step
(BP-1+RLS), the target-embedding rule (HEBB, Backward-Lens style) and an RLSEdit-style sequential rank-one
editor (CEV-11's ROOT site and value search, written with the recursive-least-squares gain).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from . import common as C
from .single import fact_tokens, release

LEARNER_ARMS = {
    "FEC-TOP+RLS": dict(rule="FEC-TOP", projector=True),
    "FEC-A+RLS": dict(rule="FEC-A", projector=True),
    "FEC-TOP": dict(rule="FEC-TOP", projector=False),
    "FEC-TOP+RLS+G": dict(rule="FEC-TOP", projector=True, guard="evict"),
    "HEBB-TOP+RLS": dict(rule="HEBB-TOP", projector=True),
    "HEBB-0+RLS": dict(rule="HEBB-0", projector=True),
    "BP-1+RLS": dict(rule="FEC-A", projector=True, exact=True),
}


# -------------------------------------------------------------------------------------------------------- stream
@dataclass
class Stream:
    index: int
    items: List[dict]
    untouched: List[str]
    ret: List[Tuple[str, str]]
    pf6: List[List[Tuple[str, str]]]
    inf: List[List[Tuple[str, str]]]
    own: List[List[Tuple[str, str]]]
    u_a: List[Tuple[str, str]]
    u_b: List[Tuple[str, str]]
    base: dict = field(default_factory=dict)


def build_stream(base, editable: Sequence[str], world: int, index: int, cfg: C.RunConfig) -> Stream:
    """Facts and probes of stream ``index`` (deterministic; stream 0 draws like the v1 stream pilot)."""
    from cev11.morph_tr import CASES
    from cev11.world import frame_prefix
    w = base.world
    rng = random.Random(100 * (index + 1) + world)
    names = rng.sample(sorted(editable), cfg.stream_n)
    towns = list(w.objects["A"])
    items = []
    for i, n in enumerate(names):
        old = w.facts[(n, "A")]
        items.append({"edit_id": f"w{world}-s{index}-f{i:03d}", "name": n, "rel": "A", "lived_case": CASES[i % 7],
                      "old": old, "new": rng.choice([o for o in towns if o != old])})
    pool = [n for n in sorted(w.names) if n not in set(names) and w.partition[n] != "p3"]
    untouched = rng.sample(pool, min(cfg.untouched, len(pool)))
    ret = [(frame_prefix(it["name"], "A", it["lived_case"]), it["new"]) for it in items]
    pf6 = [[(frame_prefix(it["name"], "A", c), it["new"]) for c in CASES if c != it["lived_case"]] for it in items]
    inf = [[(frame_prefix(it["name"], "A", c, "informal"), it["new"]) for c in CASES[1:] if c != it["lived_case"]]
           for it in items]
    own = [[(frame_prefix(it["name"], r, "gen"), w.facts[(it["name"], r)]) for r in ("B", "C")] for it in items]
    u_a = [(frame_prefix(n, "A", "gen"), w.facts[(n, "A")]) for n in untouched]
    u_b = [(frame_prefix(n, "B", "gen"), w.facts[(n, "B")]) for n in untouched]
    return Stream(index, items, untouched, ret, pf6, inf, own, u_a, u_b)


def exact(base, pairs: Sequence[Tuple[str, str]]) -> List[bool]:
    from cev11 import experiment as E
    if not pairs:
        return []
    return E._exact(base, [p for p, _ in pairs], [o for _, o in pairs])


@torch.no_grad()
def last_probs(base, prefixes: Sequence[str], batch: int = 64) -> torch.Tensor:
    """Next-token distribution after each prefix ([n, V], float32), standard batched forward."""
    from tokenizer_management.byte_bpe import BOS_ID, PAD_ID
    ids = [[BOS_ID] + base.tok.encode(p) for p in prefixes]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out: List[Optional[torch.Tensor]] = [None] * len(ids)
    for s in range(0, len(order), batch):
        rows = order[s:s + batch]
        width = max(len(ids[i]) for i in rows)
        x = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
        for r, i in enumerate(rows):
            x[r, :len(ids[i])] = torch.tensor(ids[i])
        logits = base.model(x)[0]
        for r, i in enumerate(rows):
            out[i] = torch.softmax(logits[r, len(ids[i]) - 1].float(), -1)
    return torch.stack(out) if out else torch.zeros(0, 0)


@torch.no_grad()
def block_keys(base, sequences: Sequence[Sequence[int]], positions: Sequence[Sequence[int]],
               block: int) -> List[torch.Tensor]:
    """fc2 inputs of ``block`` at the given positions of each sequence (one forward per sequence), float64."""
    grabbed: list = []
    handle = base.model.layers[block].ffn.fc2.register_forward_hook(lambda m, a, o: grabbed.append(a[0][0]))
    out = []
    try:
        for seq, pos in zip(sequences, positions):
            grabbed.clear()
            base.model(torch.tensor([list(seq)], dtype=torch.long))
            out.append(grabbed[0][list(pos)].double())
    finally:
        handle.remove()
    return out


def hard_negatives(base, stream: Stream, world: int, cfg: C.RunConfig) -> Dict[str, List[str]]:
    """Hard-negative prompts of a stream: 'rn' entities outside the stream that share the last root token with a
    written entity (same relation, the writer's lived frame); 'rel' the written entities' relations B and C in
    the lived frame; 'so' entities outside the stream whose answer is the old answer of a written fact."""
    from cev11.tokenizers import last_root_token
    from cev11.world import frame_prefix
    w = base.world
    written = {it["name"] for it in stream.items}
    outside = [n for n in sorted(w.names) if n not in written and w.partition[n] != "p3"]
    lrt = {n: last_root_token(base.tok, n) for n in outside}
    rn, rel, so = [], [], []
    olds = {}
    for it in stream.items:
        t = last_root_token(base.tok, it["name"])
        rn += [frame_prefix(m, "A", it["lived_case"]) for m in outside if lrt[m] == t]
        rel += [frame_prefix(it["name"], r, it["lived_case"]) for r in ("B", "C")]
        olds.setdefault(it["old"], 0)
    so = [frame_prefix(m, "A", "gen") for m in outside if w.facts[(m, "A")] in olds]
    rnd = random.Random(f"{world}|{stream.index}|hard")
    out = {}
    for name, lst in (("rn", rn), ("rel", rel), ("so", so)):
        lst = sorted(set(lst))
        out[name] = sorted(rnd.sample(lst, cfg.hard_negatives)) if len(lst) > cfg.hard_negatives else lst
    return out


def prepare_stream(base, stream: Stream, world: int, cfg: C.RunConfig, with_gate: bool) -> None:
    """Base-model answers and distributions of every probe (must run with nothing installed)."""
    from cev11 import experiment as E
    b = stream.base
    b["u_a"], b["u_b"] = exact(base, stream.u_a), exact(base, stream.u_b)
    b["own"] = [exact(base, o) for o in stream.own]
    if "fwl_bpb_base" not in base.layer_cache:
        base.layer_cache["fwl_bpb_base"] = E.bits_per_byte(base, list(base.prose_held[:cfg.bpb_texts]))
    b["bpb"] = base.layer_cache["fwl_bpb_base"]
    w = base.world
    b["shared_old"] = [[sum(1 for it in stream.items[:n] if it["old"] == w.facts[(u, "A")])
                        for u in stream.untouched] for n in range(len(stream.items) + 1)]
    b["never_edited"] = sum(1 for n in w.names if w.partition[n] != "p3")
    if with_gate:
        b["hard"] = hard_negatives(base, stream, world, cfg)
        b["hard_p"] = {k: last_probs(base, v) for k, v in b["hard"].items()}
        b["generic_p"] = last_probs(base, [p for p, _ in stream.u_a])


def evaluate(base, stream: Stream, n: int, with_bpb: bool, cfg: C.RunConfig) -> dict:
    """Stream metrics after the first ``n`` facts (see PROTOCOL §5.2 for the definitions)."""
    from cev11 import experiment as E
    b = stream.base
    ret = exact(base, stream.ret[:n])
    pf = exact(base, [p for row in stream.pf6[:n] for p in row])
    inf = exact(base, [p for row in stream.inf[:n] for p in row])
    u_a, u_b = exact(base, stream.u_a), exact(base, stream.u_b)
    own = exact(base, [p for row in stream.own[:n] for p in row])
    own0 = [x for row in b["own"][:n] for x in row]

    def damage(before, after):
        known = sum(before)
        return (sum(x and not y for x, y in zip(before, after)) / known) if known else 0.0

    dmg_a, dmg_b = damage(b["u_a"], u_a), damage(b["u_b"], u_b)
    by_shared: Dict[str, List[int]] = {}
    for x, y, s in zip(b["u_a"], u_a, b["shared_old"][n]):
        if x:
            cell = by_shared.setdefault(str(min(s, 3)), [0, 0])
            cell[0] += int(not y)
            cell[1] += 1
    retention = sum(ret) / n
    pf6 = sum(pf) / max(len(pf), 1)
    informal = sum(inf) / max(len(inf), 1)
    known_share = sum(b["u_a"]) / max(len(b["u_a"]), 1)
    population = (b["never_edited"] - n) * known_share          # never-edited relation-A facts the base knows
    net = sum(ret) - dmg_a * population
    out = {"retention": retention, "PF6": pf6, "IF": informal, "dmg_A": dmg_a, "dmg_B": dmg_b,
           "known_A": int(sum(b["u_a"])), "known_B": int(sum(b["u_b"])),
           "own_BC": damage(own0, own), "dmg_by_shared_old": by_shared, "net_facts": net,
           "net_per_write": net / n, "composite": (retention + 0.5 * (pf6 + informal)) / 2.0 - dmg_a}
    if with_bpb:
        out["dbpb"] = E.bits_per_byte(base, list(base.prose_held[:cfg.bpb_texts])) - b["bpb"]
    return out


@torch.no_grad()
def score_sentence(base, tok: torch.Tensor, pos: Sequence[int]) -> Tuple[float, bool]:
    """(mean target NLL, greedy reproduction) of a fact sentence under the current model (standard path)."""
    logits = base.model(tok[:-1].unsqueeze(0))[0][0]
    idx = torch.as_tensor(list(pos), dtype=torch.long)
    y = tok[1:][idx]
    return float(F.cross_entropy(logits[idx].float(), y)), bool((logits[idx].argmax(-1) == y).all())


def criterion_met(install: str, nll: float, greedy: bool, stop_nll: float = 0.05) -> bool:
    return nll < stop_nll if install == "nll" else greedy


# ---------------------------------------------------------------------------------------------------------- arms
class Arm:
    """One stream learner. ``write`` returns {"installed", "mu", "n_forward", "flags", "write_id"}."""
    name = ""
    learner = None

    def write(self, item: dict) -> dict:
        raise NotImplementedError

    def info(self) -> dict:
        return {}

    def close(self) -> None:
        pass


def key_statistics(base, world: int, cfg: C.RunConfig) -> dict:
    """Per-block key second moment and native FFN RMS from CEV-11's C0 pool (entities that are never edited or
    probed, plus training prose), cached on the base."""
    cache = base.layer_cache.get("fwl_key_stats")
    if cache is not None:
        return cache
    from cev11 import experiment as E
    from tokenizer_management.byte_bpe import PAD_ID
    FL = C.engine()
    seqs = E.c0_sequences(base, seed=world + 3, n_tokens=cfg.c0_tokens)
    probe = FL.ForwardLearner(base.model, FL.ForwardLearningConfig(n0=cfg.n0, xi=cfg.xi, guard="off"))
    try:
        probe.calibrate(seqs, exclude_ids=(PAD_ID,))
        stats = {b: (st.key_moment.clone(), st.native_rms) for b, st in probe._blocks.items()}
    finally:
        probe.close()
    base.layer_cache["fwl_key_stats"] = stats
    base.layer_cache["fwl_c0_sequences"] = seqs
    return stats


def new_learner(base, world: int, cfg: C.RunConfig, **kw):
    FL = C.engine()
    conf = dict(n0=cfg.n0, xi=cfg.xi, guard="off", install=cfg.stream_install, rms_cap=cfg.guard_rms_cap)
    if cfg.stream_grid is not None:
        conf["step_grid"] = tuple(cfg.stream_grid)
    conf.update(kw)
    learner = FL.ForwardLearner(base.model, FL.ForwardLearningConfig(**conf))
    for b, (km, rms) in key_statistics(base, world, cfg).items():
        learner.set_key_statistics(b, km, rms)
    return learner


class LearnerArm(Arm):
    def __init__(self, base, name: str, world: int, cfg: C.RunConfig):
        spec = dict(LEARNER_ARMS[name])
        self.exact = bool(spec.pop("exact", False))
        self.base, self.name = base, name
        self.learner = new_learner(base, world, cfg, mu_cap=cfg.caps.get("stream", {}).get(name), **spec)

    def write(self, item: dict) -> dict:
        FL = C.engine()
        tok, pos = fact_tokens(self.base, item)
        if self.exact:
            ref = FL.backprop_fields(self.base.model, tok, pos)
            reading = self.learner.reading_from_fields(tok, pos, ref["fields"], ref["keys"], rule="BP-1")
        else:
            reading = self.learner.read(tok, pos)
        r = self.learner.write(reading, tag=item["edit_id"])
        return {"installed": r.installed, "mu": r.mu, "n_forward": r.n_forward, "flags": list(r.flags),
                "write_id": r.write_id, "evicted": list(r.evicted)}

    def info(self) -> dict:
        L = self.learner
        ratios = {str(b): L.generic_rms_ratio(b) for b in range(len(self.base.model.layers)) if L.n_keys(b)}
        return {"rms_ratio": ratios, "evictions": L.evictions, "ledger": len(L),
                "n_keys": {str(b): L.n_keys(b) for b in range(len(self.base.model.layers)) if L.n_keys(b)}}

    def close(self) -> None:
        self.learner.close()


class FTSeqArm(Arm):
    """Sequential FT-L: CEV-11's frozen FT-L configuration applied fact after fact on the edited model."""
    name = "FT-L-seq"

    def __init__(self, base, world: int, cfg: C.RunConfig):
        from cev11 import experiment as E
        self.base, self.world, self.cfg = base, world, cfg
        self.inst = E.Installer(base)
        w = base.model.layers[C.FT_L["layer"]].ffn.fc2.weight
        self.M = torch.zeros(tuple(w.shape), dtype=torch.float64)
        self.steps = 0

    def write(self, item: dict) -> dict:
        from cev11 import experiment as E
        w = E.compute_write(self.base, item, "FT-L", C.FT_L["layer"], C.FT_L["hp"], self.world)
        self.M = self.M + w["M"].double()
        self.inst.set_M(C.FT_L["layer"], self.M)
        self.steps += int(w["diag"].get("steps", 0))
        tok, pos = fact_tokens(self.base, item)
        nll, greedy = score_sentence(self.base, tok, pos)
        return {"installed": criterion_met(self.cfg.stream_install, nll, greedy), "mu": None,
                "n_forward": int(w["diag"].get("steps", 0)), "flags": [], "write_id": None, "evicted": []}

    def info(self) -> dict:
        return {"adam_steps": self.steps}

    def close(self) -> None:
        release(self.inst)


class RLSEditArm(Arm):
    """RLSEdit-style sequential rank-one editor: the value comes from CEV-11's v* search at the ROOT site of the
    current model, the write is r g^T with the recursive-least-squares gain g = P k / (1/xi + k^T P k) under the
    same prior as the forward-learning arms. If the plain RLS write (scale 1) misses the install criterion, larger
    scales from the common step grid are tried, so the comparator gets the same forward-only search."""
    name = "RLSEdit"

    def __init__(self, base, world: int, cfg: C.RunConfig):
        from cev11.vstar import median_fc2_output_norm
        from tokenizer_management.byte_bpe import PAD_ID
        self.base, self.world, self.cfg = base, world, cfg
        self.layer = C.ROOT_SITE["layer"]
        self.learner = new_learner(base, world, cfg)
        seqs = base.layer_cache["fwl_c0_sequences"]
        if "fwl_root_median" not in base.layer_cache:
            base.layer_cache["fwl_root_median"] = median_fc2_output_norm(
                base.model, self.layer, [[s] for s in seqs], exclude_ids=(PAD_ID,))
        self.cap = C.ROOT_SITE["hp"]["cap_mult"] * base.layer_cache["fwl_root_median"]
        FL = C.engine()
        grid = cfg.stream_grid if cfg.stream_grid is not None else FL.DEFAULT_STEP_GRID
        cap = cfg.caps.get("stream", {}).get(self.name)
        self.scales = [1.0] + [g for g in grid if g > 1.0 and (cap is None or g <= cap)]
        self.search_steps = 0

    def write(self, item: dict) -> dict:
        from cev11 import experiment as E
        from cev11.keysites import SiteUndefined, key_at, site_prompts
        from cev11.vstar import search_delta
        from cev11.world import frame_prefix
        base, hp = self.base, C.ROOT_SITE["hp"]
        tok, pos = fact_tokens(base, item)
        nll, greedy = score_sentence(base, tok, pos)
        if criterion_met(self.cfg.stream_install, nll, greedy):
            return {"installed": True, "mu": 0.0, "n_forward": 1, "flags": ["already_installed"],
                    "write_id": None, "evicted": []}
        prefix = frame_prefix(item["name"], item["rel"], item["lived_case"], "formal", "none")
        try:
            prompts = site_prompts(base.tok, prefix, item["name"], "none", site="ROOT")
        except SiteUndefined:
            return {"installed": False, "mu": None, "n_forward": 1, "flags": ["site_undefined"],
                    "write_id": None, "evicted": []}
        key = key_at(base.model, self.layer, base.tok, prefix, prompts[0][1]).double()
        delta, info = search_delta(base.model, self.layer, prompts, E._target_ids(base, item["new"]),
                                   E._kl_ids(base, item), hp["step"], self.cap, max_steps=int(hp["max_steps"]))
        self.search_steps += int(info["steps"])
        gain = self.learner.rls_gain(self.layer, key)
        n_forward = 1
        for scale in self.scales:
            wid = self.learner.write_terms(
                {self.layer: (scale * delta.double().unsqueeze(0), gain.unsqueeze(0), key.unsqueeze(0))},
                tag=item["edit_id"], rule="RLSEdit")
            nll, greedy = score_sentence(base, tok, pos)
            n_forward += 1
            if criterion_met(self.cfg.stream_install, nll, greedy):
                return {"installed": True, "mu": scale, "n_forward": n_forward, "flags": [], "write_id": wid,
                        "evicted": []}
            self.learner.retract(wid)
        return {"installed": False, "mu": None, "n_forward": n_forward, "flags": ["not_installed"],
                "write_id": None, "evicted": []}

    def info(self) -> dict:
        return {"rms_ratio": {str(self.layer): self.learner.generic_rms_ratio(self.layer)},
                "search_steps": self.search_steps, "ledger": len(self.learner)}

    def close(self) -> None:
        self.learner.close()


def make_arm(base, name: str, world: int, cfg: C.RunConfig) -> Arm:
    if name in LEARNER_ARMS:
        return LearnerArm(base, name, world, cfg)
    if name == "FT-L-seq":
        return FTSeqArm(base, world, cfg)
    if name == "RLSEdit":
        return RLSEditArm(base, world, cfg)
    raise ValueError(f"unknown stream arm {name!r}")


# ----------------------------------------------------------------------------------------------- gate (§5.3)
def gate_metrics(base, stream: Stream, n: int, cfg: C.RunConfig, with_bpb: bool) -> dict:
    """Recall and leak under the learner's current read setting (theta, gain)."""
    from cev11 import experiment as E
    b = stream.base
    ret = exact(base, stream.ret[:n])
    pf = exact(base, [p for row in stream.pf6[:n] for p in row])
    u_a = exact(base, stream.u_a)
    known = sum(b["u_a"])
    out = {"retention": sum(ret) / n, "recall_other": sum(pf) / max(len(pf), 1),
           "dmg_A": (sum(x and not y for x, y in zip(b["u_a"], u_a)) / known) if known else 0.0}
    events, total, by_type = 0, 0, {}
    for kind, prefixes in b["hard"].items():
        if not prefixes:
            by_type[kind] = None
            continue
        tv = 0.5 * (last_probs(base, prefixes) - b["hard_p"][kind]).abs().sum(-1)
        hits = int((tv > cfg.leak_tv).sum())
        by_type[kind] = hits / len(prefixes)
        events += hits
        total += len(prefixes)
    tv = 0.5 * (last_probs(base, [p for p, _ in stream.u_a]) - b["generic_p"]).abs().sum(-1)
    out.update({"leak_hard": events / max(total, 1), "leak_by_type": by_type, "n_hard": total,
                "leak_generic": float((tv > cfg.leak_tv).float().mean())})
    if with_bpb:
        out["dbpb"] = E.bits_per_byte(base, list(base.prose_held[:cfg.bpb_texts])) - b["bpb"]
    return out


def gate_sweep(base, arm: LearnerArm, stream: Stream, n: int, cfg: C.RunConfig, run: str, world: int,
               writer: C.RecordWriter) -> None:
    """theta sweep of the gated read and gain sweep of the ungated read (the 'no gate' comparator)."""
    L = arm.learner
    settings = [("theta", t, 1.0) for t in cfg.thetas] + [("gain", 0.0, g) for g in cfg.gains if g != 1.0]
    try:
        for kind, theta, gain in settings:
            label = f"theta={theta:g}" if kind == "theta" else f"gain={gain:g}"
            key = f"gate|{stream.index}|{arm.name}|{label}"
            if writer.has(key):
                continue
            t0 = time.time()
            L.theta, L.read_gain = float(theta), float(gain)
            L._clear_cache()
            m = gate_metrics(base, stream, n, cfg, cfg.gate_bpb)
            writer.write(C.make_record("gate", run, world, key, stream=stream.index, arm=arm.name, setting=label,
                                       theta=float(theta), gain=float(gain), n=n,
                                       seconds=round(time.time() - t0, 3), **m))
    finally:
        L.theta, L.read_gain = 0.0, 1.0
        L._clear_cache()


# ------------------------------------------------------------------------------------------- capacity (§5.4)
def reference_rows(base, world: int, partitions: Sequence[str], block: int) -> List[torch.Tensor]:
    """Keys of ``block`` at the object-span positions of reference fact sentences: entities of the given
    partitions (never written, never used as capacity probes), relation A, the case frames the base was trained
    on, with a seeded counterfactual object (the stream also reads counterfactual facts)."""
    from cev11.morph_tr import CASES
    from cev11.world import frame_prefix
    from tokenizer_management.byte_bpe import BOS_ID
    w = base.world
    held = set(tuple(x) for x in w.p3_held)
    towns = list(w.objects["A"])
    seqs, span = [], []
    for n in sorted(m for m, p in w.partition.items() if p in partitions):
        for c in CASES:
            if (n, c) in held:
                continue
            obj = random.Random(f"{world}|{n}|{c}|ref").choice([o for o in towns if o != w.facts[(n, "A")]])
            pid = [BOS_ID] + base.tok.encode(frame_prefix(n, "A", c))
            tid = base.tok.encode(" " + obj + ".")
            seqs.append((pid + tid)[:-1])
            span.append(list(range(len(pid) - 1, len(pid) + len(tid) - 1)))
    return block_keys(base, seqs, span, block)


def capacity_setup(base, stream: Stream, world: int, cfg: C.RunConfig, learner) -> dict:
    """Law inputs, all from the base model before any write.

    Registered form (PROTOCOL §5.4): reference entities = the C0 pool and the P3 entities (every entity outside
    the editable pool); stored-key classes = the prediction site (first object position) and the later object
    positions, with their shares; queries = unit-normalised prediction-site keys of the reference entities.
    The two-population form of the DEFTER v2 proposal (one class, C0-pool entities only) is computed next to it.
    The measured queries are the stream's untouched probe-pool entities (disjoint from the reference)."""
    from tokenizer_management.byte_bpe import BOS_ID
    FL = C.engine()
    w = base.world
    top = len(base.model.layers) - 1
    C0 = learner.prior(top)
    rows = reference_rows(base, world, ("c0", "p3"), top)
    site = torch.stack([r[0] for r in rows])
    later = torch.cat([r[1:] for r in rows])
    shares = [site.shape[0] / (site.shape[0] + later.shape[0]), later.shape[0] / (site.shape[0] + later.shape[0])]
    classes = [FL.whitened_second_moment(site, C0), FL.whitened_second_moment(later, C0)]
    S_q = FL.whitened_second_moment(site, C0, unit=True)
    rows_c0 = reference_rows(base, world, ("c0",), top)
    S_w_v2 = FL.whitened_second_moment(torch.cat(rows_c0), C0)
    S_q_v2 = FL.whitened_second_moment(torch.stack([r[0] for r in rows_c0]), C0, unit=True)
    probes = [[BOS_ID] + base.tok.encode(p) for (p, _), u in zip(stream.u_a, stream.untouched)
              if w.partition[u] == "probe"]
    q = torch.cat(block_keys(base, probes, [[len(s) - 1] for s in probes], top))
    pooled = shares[0] * classes[0] + shares[1] * classes[1]
    return {"block": top, "classes": classes, "shares": shares, "S_q": S_q, "S_w_v2": S_w_v2, "S_q_v2": S_q_v2,
            "q": q, "n_queries": len(probes), "n_reference_facts": len(rows),
            "keys_per_fact_ref": (site.shape[0] + later.shape[0]) / len(rows),
            "d_ww": float(torch.trace(pooled) ** 2 / torch.trace(pooled @ pooled)), "p_w": float(torch.trace(pooled)),
            "p_site": float(torch.trace(classes[0]))}


def _interp(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    if x <= xs[0]:
        return ys[0] * x / xs[0] if xs[0] > 0 else ys[0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


def _crossing(xs: Sequence[float], ys: Sequence[float], theta: float) -> Optional[float]:
    """First x at which the piecewise-linear curve (0, 0), (xs, ys) reaches theta."""
    px, py = 0.0, 0.0
    for x, y in zip(xs, ys):
        if py < theta <= y:
            return px + (theta - py) / (y - py) * (x - px) if y > py else x
        px, py = x, y
    return None


def capacity_record(setup: dict, trace: List[List[float]], cfg: C.RunConfig, run: str, world: int,
                    stream: Stream, arm: str) -> dict:
    """Predicted against measured familiarity of unrelated probes along the stream, and the knees (in keys)."""
    FL = C.engine()
    xi = cfg.xi
    grid = [2.0 ** (j / 2.0) for j in range(2, 25)]             # 2 .. 4096 stored keys
    curve, state = [], None
    for n_keys in grid:
        v, state = FL.predicted_familiarity(setup["classes"], setup["S_q"], xi, n_keys, shares=setup["shares"],
                                            warm=state, return_state=True)
        curve.append(v)
    curve_v2 = [FL.predicted_familiarity(setup["S_w_v2"], setup["S_q_v2"], xi, n_keys) for n_keys in grid]
    xs = [t[0] for t in trace]
    predicted = [_interp(grid, curve, x) for x in xs]
    predicted_v2 = [_interp(grid, curve_v2, x) for x in xs]
    knees, knees_v2 = [], []
    for theta in cfg.capacity_thetas:
        meas = _crossing(xs, [t[2] for t in trace], theta)
        for out, cv in ((knees, curve), (knees_v2, curve_v2)):
            pred = _crossing(grid, cv, theta)
            out.append({"theta": theta, "predicted_keys": pred, "measured_keys": meas,
                        "ratio": (meas / pred) if (meas and pred) else None})
    end = trace[-1] if trace else [0.0, 0.0, 0.0]
    return C.make_record(
        "capacity", run, world, f"capacity|{stream.index}|{arm}", stream=stream.index, arm=arm,
        block=setup["block"], trace=trace, predicted=predicted, predicted_v2=predicted_v2, knees=knees,
        knees_v2=knees_v2, law={"grid": grid, "mixture": curve, "v2": curve_v2}, shares=setup["shares"], xi=xi,
        n0=cfg.n0, keys_end=end[0], nu_mean_end=end[1], nu_median_end=end[2],
        predicted_end=predicted[-1] if predicted else 0.0,
        predicted_v2_end=predicted_v2[-1] if predicted_v2 else 0.0, d_ww=setup["d_ww"], p_w=setup["p_w"],
        p_site=setup["p_site"], keys_per_fact_ref=setup["keys_per_fact_ref"],
        n_reference_facts=setup["n_reference_facts"], n_queries=setup["n_queries"])


# ------------------------------------------------------------------------------------------ retraction (§5.5)
def _first_token_probs(base, stream: Stream, i: int) -> Tuple[torch.Tensor, List[str]]:
    """Next-token distributions of fact ``i`` at its lived frame and at one other formal frame."""
    from cev11.world import frame_prefix
    it = stream.items[i]
    other = "gen" if it["lived_case"] != "gen" else "nom"
    prefixes = [frame_prefix(it["name"], "A", it["lived_case"]), frame_prefix(it["name"], "A", other)]
    return last_probs(base, prefixes), prefixes


def retraction_records(base, arm: LearnerArm, stream: Stream, snaps: Dict[int, dict], write_ids: Dict[int, int],
                       n: int, cfg: C.RunConfig, run: str, world: int, writer: C.RecordWriter) -> None:
    """For each selected fact: ledger retraction against a replay of the stream without it (the 'never seen'
    reference). KL(replay || retract) at the fact's own queries, and the other facts' retention under both."""
    L = arm.learner
    final = L.snapshot()
    for i in cfg.retract_indices:
        key = f"retract|{stream.index}|{arm.name}|{i}"
        if i >= n or writer.has(key) or i not in snaps or write_ids.get(i) is None:
            continue
        t0 = time.time()
        it = stream.items[i]
        first_tok = base.tok.encode(" " + it["new"] + ".")[0]
        others = [j for j in range(n) if j != i]
        p_full, _ = _first_token_probs(base, stream, i)
        L.retract(write_ids[i])
        p_ret, prefixes = _first_token_probs(base, stream, i)
        ret_others = exact(base, [stream.ret[j] for j in others])
        L.restore(snaps[i])                                   # replay: the stream without fact i
        replay_installed = 0
        for j in range(i + 1, n):
            replay_installed += int(arm.write(stream.items[j])["installed"])
        p_rep, _ = _first_token_probs(base, stream, i)
        rep_others = exact(base, [stream.ret[j] for j in others])
        L.restore(final)
        eps = 1e-12
        kl = (p_rep * ((p_rep + eps).log() - (p_ret + eps).log())).sum(-1)
        tv = 0.5 * (p_rep - p_ret).abs().sum(-1)
        writer.write(C.make_record(
            "retract", run, world, key, stream=stream.index, arm=arm.name, index=i, write_id=int(write_ids[i]),
            p_before=float(p_full[0, first_tok]), p_retract=float(p_ret[0, first_tok]),
            p_replay=float(p_rep[0, first_tok]), kl_replay_retract=[float(x) for x in kl],
            tv_replay_retract=[float(x) for x in tv], others_retract=sum(ret_others) / max(len(others), 1),
            others_replay=sum(rep_others) / max(len(others), 1), replayed=n - 1 - i,
            replay_installed=replay_installed, n=n, seconds=round(time.time() - t0, 3)))


# ----------------------------------------------------------------------------------------------------- driver
def run_stream(base, editable, world: int, index: int, run: str, cfg: C.RunConfig, writer: C.RecordWriter,
               log=print) -> dict:
    """All records of one stream (resumable per arm and checkpoint)."""
    from cev11.model import model_sha256
    timing: Dict[str, float] = {}
    side = index == 0                                          # gate, capacity and retraction on stream 0 only
    checkpoints = [c for c in cfg.checkpoints if c <= cfg.stream_n]

    def arm_done(name: str) -> bool:
        lim = min(cfg.arm_limits.get(name, cfg.stream_n), cfg.stream_n)
        cps = [c for c in checkpoints if c <= lim] or [lim]
        done = all(writer.has(f"stream|{index}|{name}|{c}") for c in cps)
        if side and name in cfg.gate_arms:
            done = done and all(writer.has(f"gate|{index}|{name}|theta={t:g}") for t in cfg.thetas)
            done = done and all(writer.has(f"gate|{index}|{name}|gain={g:g}") for g in cfg.gains if g != 1.0)
        if side and name == "FEC-TOP+RLS":
            # retraction records exist only for facts that were written, so they are not a completion criterion;
            # the "side" marker is written last
            done = done and writer.has(f"capacity|{index}|{name}") and writer.has(f"side|{index}|{name}")
        return done

    todo = [a for a in cfg.stream_arms if not arm_done(a)]
    if not todo:
        return timing
    t0 = time.time()
    stream = build_stream(base, editable, world, index, cfg)
    key_statistics(base, world, cfg)
    prepare_stream(base, stream, world, cfg, with_gate=side)
    timing["prepare"] = time.time() - t0
    for name in todo:
        t0 = time.time()
        arm = make_arm(base, name, world, cfg)
        limit = min(cfg.arm_limits.get(name, cfg.stream_n), cfg.stream_n)
        cps = [c for c in checkpoints if c <= limit] or [limit]
        track = side and name == "FEC-TOP+RLS"
        setup = capacity_setup(base, stream, world, cfg, arm.learner) if track else None
        trace: List[List[float]] = []
        snaps: Dict[int, dict] = {}
        write_ids: Dict[int, int] = {}
        mus: List[float] = []
        installed = n_forward = 0
        flags: Dict[str, int] = {}
        try:
            for i, item in enumerate(stream.items[:limit]):
                if track and i in cfg.retract_indices:
                    snaps[i] = arm.learner.snapshot()
                r = arm.write(item)
                installed += int(bool(r["installed"]))
                n_forward += int(r["n_forward"])
                if r["installed"] and r["mu"]:
                    mus.append(float(r["mu"]))
                for f_ in r["flags"]:
                    flags[f_] = flags.get(f_, 0) + 1
                if r.get("write_id") is not None:
                    write_ids[i] = r["write_id"]
                if track:
                    nu = arm.learner.familiarity(setup["block"], setup["q"])
                    trace.append([float(arm.learner.n_keys(setup["block"])), float(nu.mean()),
                                  float(nu.median())])
                n = i + 1
                if n in cps and not writer.has(f"stream|{index}|{name}|{n}"):
                    m = evaluate(base, stream, n, with_bpb=(n == cps[-1]), cfg=cfg)
                    writer.write(C.make_record(
                        "stream", run, world, f"stream|{index}|{name}|{n}", stream=index, arm=name, n=n,
                        written=n, installed=installed, install=cfg.stream_install,
                        truncated=bool(limit < cfg.stream_n), mu_median=C.median(mus),
                        mu_max=max(mus) if mus else None, n_forward=n_forward, flags=flags,
                        seconds=round(time.time() - t0, 3), **arm.info(), **m))
            if side and name in cfg.gate_arms and isinstance(arm, LearnerArm):
                gate_sweep(base, arm, stream, limit, cfg, run, world, writer)
            if track:
                if not writer.has(f"capacity|{index}|{name}"):
                    writer.write(capacity_record(setup, trace, cfg, run, world, stream, name))
                retraction_records(base, arm, stream, snaps, write_ids, limit, cfg, run, world, writer)
                if not writer.has(f"side|{index}|{name}"):
                    writer.write(C.make_record(
                        "side", run, world, f"side|{index}|{name}", stream=index, arm=name,
                        retracted=sorted(i for i in cfg.retract_indices if i < limit and i in write_ids)))
        finally:
            arm.close()
        timing[name] = time.time() - t0
        log(f"[{run} w{world}] stream {index} {name}: {installed}/{limit} installed ({timing[name]:.0f}s)")
        if model_sha256(base.model) != base.base_sha:
            raise RuntimeError(f"the base model changed during stream arm {name}")
    return timing
