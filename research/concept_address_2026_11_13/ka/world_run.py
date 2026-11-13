# -*- coding: utf-8 -*-
"""Per-world runner of the KA study: setup, Part A, episodic family, FT streams, hooked checks (PROTOCOL §5, §8).

Output layout (``<out>/w<world>/``):

    setup.json         partitions, hashes, c0 fits, gate thresholds, laws computed before any write
    part_a.jsonl       one line per (edit, address, configuration); an edit is complete when all are present
    episodic.jsonl     one line per (address, read rule, N) of the output-side memory, plus frontier lines
    ft_stream.jsonl    one line per (FT arm, checkpoint)
    checks.json        hooked-memory checks: cached evaluator vs hooked model vs true greedy decoding, drift
    DONE.json          written last, and only when every phase the protocol requires for this run is complete in
                       the directory; lists those phases, the required phases, the hash of every record file
                       present, and the timings
    EXCLUDED.json      written instead of any record when the base is not a valid unit (it failed CEV-11's gates)

DONE.json describes the directory, not the last invocation: a run restricted with ``phases`` never shortens the
phase list or the file list of a finished world, and a world that lacks a required phase has no DONE.json (the
selection, the analysis and the record validation accept only worlds with DONE.json). Timings live in DONE.json
only, so every record file is a deterministic function of the base, the configuration and the code.

Resume: setup is deterministic and recomputed (it must agree with an existing setup.json); Part A resumes after
the last complete edit; the episodic family, each FT arm and the checks are written atomically and skipped when
present.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from . import KA_VERSION, REPO
from . import gates as G
from . import part_a as PA
from . import streams as ST
from .codes import Address, capture, fit_address, lda_fit, theta_n
from .episodic import EpisodeLedger, EpisodicCortex, ReadRule
from .ft import FT_ARMS, cev11_ft_configs, ft_stream
from cev11.keysites import VARIANTS, sites
from cev11.morph_tr import INFLECTED
from cev11.world import CONTEXTS, RELATIONS, frame_prefix
from tokenizer_management.byte_bpe import BOS_ID

PRED_LAYER_CANDIDATES = (2, 3, 4, 5)
PHASES = ("parta", "episodic", "ft", "checks")
RECORD_FILES = {"parta": "part_a.jsonl", "episodic": "episodic.jsonl", "ft": "ft_stream.jsonl", "checks": "checks.json"}
RUNS = ("discovery", "confirm")
THRESHOLD_SEQS = 2000        # c0 sequences whose token positions calibrate the gate thresholds (seed w+15)
CHECK_SHARE = 0.05           # share of probes re-scored with the hooked model and true greedy decoding
UNIT_KEYS = ("lived", "para", "fused", "leadin", "Leak_U", "Leak_C", "Leak_RN", "PX", "OBJ", "OBJ1", "ece")


def required_phases(run: str, h1_confirmatory: bool = True) -> Tuple[str, ...]:
    """Phases the protocol requires of one world: all four on a discovery world (the selection reads every one of
    them); on a confirmatory world Part A only if KA-H1 passed the discovery screen (PROTOCOL §8)."""
    if run not in RUNS:
        raise ValueError(f"unknown run {run!r}")
    if run == "confirm" and not h1_confirmatory:
        return tuple(p for p in PHASES if p != "parta")
    return PHASES


@dataclass
class RunConfig:
    """Protocol values by default; ``protocol`` is False for any reduced (smoke) configuration."""
    protocol: bool = True
    n_test: int = PA.N_TEST
    steps: Tuple[float, ...] = G.STEPS
    qs: Tuple[float, ...] = G.QS
    addresses: Tuple[str, ...] = G.ADDRESSES
    ft_refs: Tuple[str, ...] = PA.FT_REFS
    gate_layer: int = G.GATE_LAYER
    search_max_steps: int = 120
    n_u: int = ST.N_U
    n_stream: int = ST.N_STREAM
    checkpoints: Tuple[int, ...] = ST.CHECKPOINTS
    ft_checkpoints: Tuple[int, ...] = ST.FT_CHECKPOINTS
    n_primary: int = ST.N_PRIMARY
    pred_layers: Tuple[int, ...] = PRED_LAYER_CANDIDATES
    pred_att: bool = True
    c0_entities: int = 0                 # 0 = all
    threshold_seqs: int = THRESHOLD_SEQS
    geometry_entities: int = 0           # 0 = all
    bpb_paragraphs: int = 0              # 0 = all held-out paragraphs

    @classmethod
    def smoke(cls) -> "RunConfig":
        """Every code path of the protocol at a reduced size (2 edits, a stream of 120, two candidate layers,
        half of c0): a few minutes on one core. Its records are refused by the selection and the analysis."""
        return cls(protocol=False, n_test=2, n_stream=120, checkpoints=(25, 50, 100), ft_checkpoints=(25, 50),
                   n_primary=50, pred_layers=(3, 4), c0_entities=30, threshold_seqs=600, geometry_entities=40,
                   bpb_paragraphs=20)


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json_atomic(path: pathlib.Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl_atomic(path: pathlib.Path, records: Sequence[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: pathlib.Path) -> List[dict]:
    """Complete lines only: a torn last line (interrupted write) is ignored."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return out


def qkey(q: float) -> str:
    return f"{q:g}"


# ---------------------------------------------------------------------------------------------------- completeness
def _is_unit(v) -> bool:
    return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool) and -1e-12 <= v <= 1 + 1e-12)


def stream_checkpoints(setup: Mapping, key: str = "checkpoints") -> List[int]:
    """Checkpoints of the configuration that the world's stream reaches."""
    cfg = setup["config"]
    n_stream = int(setup.get("partitions", {}).get("n_stream", cfg["n_stream"]))
    return [int(n) for n in cfg[key] if n <= n_stream]


def phase_problems(path: pathlib.Path, setup: Mapping, phase: str) -> List[str]:
    """Completeness and range problems of one phase's record file in the world directory ``path``.

    An empty list means the phase is complete: every record the configuration in ``setup`` calls for is there.
    One implementation for the runner (which phases DONE.json may list) and for the record validation."""
    if phase not in RECORD_FILES:
        raise ValueError(f"unknown phase {phase!r}")
    path = pathlib.Path(path)
    name, cfg = path.name, setup["config"]
    f = path / RECORD_FILES[phase]
    if not f.exists():
        return [f"{name}: {RECORD_FILES[phase]} is missing"]
    problems: List[str] = []
    if phase == "parta":
        expected = set(PA.expected_configs(cfg["addresses"], cfg["steps"], cfg["qs"], cfg["ft_refs"]))
        by_edit: Dict[str, List[str]] = {}
        for r in read_jsonl(f):
            by_edit.setdefault(r["edit_id"], []).append(r["config"])
            where = f"({r['edit_id']} {r['config']}"
            for fam in ("WF", "PF6", "FUSED", "IDENT", "CTX"):
                if not 0 <= r[fam]["ok"] <= r[fam]["n"]:
                    problems.append(f"{name}: part A count out of range {where} {fam})")
            for fam in PA.LEAK_FAMILIES:
                if not 0 <= r[fam]["changed"] <= r[fam]["n"]:
                    problems.append(f"{name}: part A count out of range {where} {fam})")
            if r["address"] in G.GATE_ADDRESSES:
                for fam in PA.IN_BALL_FAMILIES:
                    rows = r.get(f"in_{fam}")
                    # one [in the ball on the scored sequence, outcome, in the ball on the prefix] row per probe
                    if rows is None or len(rows) != r[fam]["n"] or any(
                            len(x) != 3 or any(v not in (0, 1) for v in x) or x[2] > x[0] for x in rows):
                        problems.append(f"{name}: part A in-ball rows malformed {where} {fam})")
        n_test = setup["partitions"]["n_test"]
        if len(by_edit) != n_test:
            problems.append(f"{name}: {len(by_edit)} edits, expected {n_test}")
        for e, confs in sorted(by_edit.items()):
            if set(confs) != expected or len(confs) != len(expected):
                problems.append(f"{name}: edit {e} has {len(set(confs))} of {len(expected)} configurations"
                                + (" (with repeats)" if len(confs) != len(set(confs)) else ""))
    elif phase == "episodic":
        recs = read_jsonl(f)
        seen = {(r["address"], r["read"], r.get("N")) for r in recs}
        cps = stream_checkpoints(setup)
        nonull = [n for n in cps if n in (cfg["checkpoints"][0], cfg["n_primary"], cfg["checkpoints"][-1])]
        for a in setup["addresses"]:
            for read, at in (("rule", cps), ("fixed", cps), ("nonull", nonull)):
                for n in at:
                    if (a, read, n) not in seen:
                        problems.append(f"{name}: episodic record missing ({a}, {read}, N={n})")
            if not any(r["address"] == a and r["read"] == "frontier" for r in recs):
                problems.append(f"{name}: frontier record missing ({a})")
        for r in recs:
            for k in UNIT_KEYS:
                if k in r and not _is_unit(r[k]):
                    problems.append(f"{name}: episodic {k} out of range ({r['address']}, {r['read']}, {r.get('N')})")
    elif phase == "ft":
        recs = read_jsonl(f)
        seen = {(r["arm"], r["N"]) for r in recs}
        finished = {r["arm"] for r in recs if r.get("arm_done")}
        for arm in FT_ARMS:
            for n in stream_checkpoints(setup, "ft_checkpoints"):
                if (arm, n) not in seen:
                    problems.append(f"{name}: FT record missing ({arm}, N={n})")
            if arm not in finished:
                problems.append(f"{name}: the {arm} stream is not finished")
    else:
        chk = json.loads(f.read_text(encoding="utf-8"))
        if not chk.get("model_unchanged"):
            problems.append(f"{name}: the base model changed during the checks")
        for a in setup["addresses"]:
            if a.startswith("PRED-RES") and a not in chk.get("addresses", {}):
                problems.append(f"{name}: hooked check missing ({a})")
    return problems


def write_excluded(out_dir: pathlib.Path, world: int, run: str, reason: str) -> pathlib.Path:
    """Mark a world as not a valid unit (no record is produced for it); returns the marker path."""
    d = pathlib.Path(out_dir) / f"w{int(world)}"
    d.mkdir(parents=True, exist_ok=True)
    write_json_atomic(d / "EXCLUDED.json", {"world": int(world), "run": run, "ka_version": KA_VERSION,
                                            "reason": reason})
    return d / "EXCLUDED.json"


# ---------------------------------------------------------------------------------------------------- setup
class WorldRun:
    def __init__(self, base, world: int, run: str, cfg: RunConfig, out_dir: pathlib.Path,
                 pred_layer: Optional[int] = None, log=print, required: Optional[Sequence[str]] = None):
        self.base, self.world, self.run, self.cfg = base, int(world), run, cfg
        self.out = pathlib.Path(out_dir) / f"w{world}"
        self.out.mkdir(parents=True, exist_ok=True)
        if (self.out / "EXCLUDED.json").exists():
            raise RuntimeError(f"{self.out} is marked EXCLUDED (not a valid unit): remove the marker to run it")
        unknown = [p for p in (required or ()) if p not in PHASES]
        if unknown:
            raise ValueError(f"unknown phases {unknown}")
        # phases the protocol requires of this world; DONE.json is written only when all of them are complete
        self.required = tuple(p for p in PHASES if required is None or p in required)
        self.done = False
        self.log = log
        self.pred_layer = pred_layer
        self.timings: Dict[str, float] = {}
        self.addresses: Dict[str, Address] = {}
        self.thresholds: Dict[str, Dict[float, float]] = {}
        self.sup_proj = None
        self.setup: dict = {}

    # -------------------------------------------------------------- c0 statistics (write-free)
    def _c0_items(self) -> List[dict]:
        c0 = sorted(self.base.world.members("c0"))
        if self.cfg.c0_entities:
            c0 = c0[:self.cfg.c0_entities]
        return [{"name": n, "rel": r, "case": c, "orth": o, "ctx": x}
                for n in c0 for r in RELATIONS for (c, o) in VARIANTS for x in CONTEXTS]

    def prepare(self) -> dict:
        t0 = time.time()
        base, cfg, w = self.base, self.cfg, self.world
        tok, model = base.tok, base.model
        from cev11.model import model_sha256
        if model_sha256(model) != base.base_sha:
            raise RuntimeError("the base model is not in its base state")
        layers = (self.pred_layer,) if self.pred_layer is not None else tuple(cfg.pred_layers)
        gl = cfg.gate_layer
        items = self._c0_items()
        seqs, pick = [], []
        for it in items:
            pre = frame_prefix(it["name"], it["rel"], it["case"], it["orth"], it["ctx"])
            ids = [BOS_ID] + list(tok.encode(pre))
            std = sites(tok, pre, it["name"], it["ctx"], strict=False)["STD"]
            seqs.append(ids)
            pick.append([len(ids) - 1, int(std)])
        specs = [("post", l) for l in layers] + ([("att", l) for l in layers] if cfg.pred_att else []) + [("att", gl)]
        recs = capture(model, seqs, specs, pick=pick)
        for l in layers:
            kinds = [("PRED-RES", "post")] + ([("PRED-ATT", "att")] if cfg.pred_att else [])
            for nm, kind in kinds:
                X = torch.stack([r[f"{kind}{l}"][0] for r in recs])
                a = fit_address(X, items, f"{nm}@{l}", kind, l, seed=w + 16)
                self.addresses[a.name] = a
        # supervised ceiling: LDA of the attention stream at the key site, classes = c0 entities
        if "A-SUP" in cfg.addresses:
            Xs = torch.stack([r[f"att{gl}"][1] for r in recs])
            mu, P = lda_fit(Xs, [it["name"] for it in items], G.SUP_DIM)
            self.sup_proj = (mu, P)
        # radius thresholds at matched c0 false-match rates (per code, per world, before any write)
        rnd = random.Random(w + 15)
        sub = sorted(rnd.sample(range(len(seqs)), min(cfg.threshold_seqs, len(seqs))))
        key_items = [(n, c) for n in sorted({it["name"] for it in items}) for c in INFLECTED]
        key_seqs, key_pos = [], []
        for n, c in key_items:
            pre = frame_prefix(n, "A", c, "formal", "none")
            key_seqs.append([BOS_ID] + list(tok.encode(pre)))
            key_pos.append([int(sites(tok, pre, n, "none", strict=False)["STD"])])
        gate_codes = sorted({G.GATE_CODE[a] for a in cfg.addresses if a in G.GATE_ADDRESSES})
        if gate_codes:
            gspecs = G.code_specs(gate_codes, gl)
            pos_recs = capture(model, [seqs[i] for i in sub], gspecs, dtype=torch.float32)
            key_recs = capture(model, key_seqs, gspecs, pick=key_pos, dtype=torch.float32)
            pos_ent = [items[i]["name"] for i, r in zip(sub, pos_recs) for _ in range(len(seqs[i]) - 1)]
            proj = None if self.sup_proj is None else (self.sup_proj[0].float(), self.sup_proj[1].float())
            for code in gate_codes:
                Pc = torch.cat([G.codes_from_tap(r, code, gl, proj)[1:] for r in pos_recs])     # BOS excluded
                Kc = torch.cat([G.codes_from_tap(r, code, gl, proj) for r in key_recs])
                self.thresholds[code] = G.gate_thresholds(Pc, pos_ent, Kc, [n for n, _ in key_items], cfg.qs)
        # partitions and stream
        self.U, self.pool = ST.split_probe(base.world, w + 14, cfg.n_u)
        self.facts = ST.make_stream(base.world, self.pool, w + 12, cfg.n_stream)
        gates_path = base.base_dir / "gates.json"
        editable = json.loads(gates_path.read_text(encoding="utf-8"))["editable"] if gates_path.exists() \
            else sorted(base.world.members("probe"))
        self.items = PA.test_items(base, editable, w, cfg.n_test)
        self.ft_cfg = cev11_ft_configs()
        setup = {
            "ka_version": KA_VERSION, "world": w, "run": self.run, "config": asdict(cfg),
            "base": {"dir": base.base_dir.name, "model_sha256": base.base_sha,
                     "world_hash": base.world.name_hash(), "tokenizer_identity": base.tok.identity()
                     if hasattr(base.tok, "identity") else None},
            "torch": torch.__version__, "threads": torch.get_num_threads(),
            "partitions": {"c0": len({it["name"] for it in items}), "U": len(self.U), "pool": len(self.pool),
                           "U_sha256": sha256_json(self.U), "stream_sha256": sha256_json([asdict(f) for f in self.facts]),
                           "test_items_sha256": sha256_json(self.items), "n_test": len(self.items),
                           "n_stream": len(self.facts)},
            "addresses": {k: a.summary() for k, a in sorted(self.addresses.items())},
            "thresholds": {c: {qkey(q): th for q, th in v.items()} for c, v in sorted(self.thresholds.items())},
            "ft_configs": self.ft_cfg,
            "geometry": self._geometry(),
            "laws": self._laws(),
        }
        self.setup = setup
        path = self.out / "setup.json"
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            for key in ("ka_version", "run", "config", "base", "partitions", "thresholds"):
                if json.dumps(old.get(key), sort_keys=True) != json.dumps(json.loads(json.dumps(setup[key])), sort_keys=True):
                    raise RuntimeError(f"setup.json exists and differs in {key!r}: refusing to mix runs in {self.out}")
        write_json_atomic(path, setup)
        self.timings["setup"] = round(time.time() - t0, 1)
        self.log(f"[ka w{w}] setup {self.timings['setup']}s: {len(items)} c0 codes, "
                 f"{len(self.addresses)} addresses, {len(self.items)} TEST edits, stream {len(self.facts)}")
        return setup

    def _geometry(self) -> dict:
        """Pre-edit geometry of the gate codes at the key site (CAM, per-entity windows); descriptive."""
        from cev11.tokenizers import classify_informal, last_root_token
        base, tok, gl = self.base, self.base.tok, self.cfg.gate_layer
        ents = PA.eligible_entities(base, sorted(base.world.members("probe")))
        if self.cfg.geometry_entities:
            ents = ents[:self.cfg.geometry_entities]
        if not ents:
            return {}
        lrt = {n: last_root_token(tok, n) for n in ents}
        rows, seqs, pos = [], [], []

        def add(e, kind, prefix, ctx):
            rows.append((e, kind))
            seqs.append([BOS_ID] + list(tok.encode(prefix)))
            pos.append([int(sites(tok, prefix, e, ctx, strict=False)["STD"])])

        for e in ents:
            add(e, "F0", frame_prefix(e, "A", "nom"), "none")
            for cx in ("P1", "P2"):
                add(e, "CTX", frame_prefix(e, "A", "nom", "formal", cx), cx)
            for c in INFLECTED:
                if classify_informal(tok, e, c) == "FUSED":
                    add(e, "FUSED", frame_prefix(e, "A", c, "informal"), "none")
        recs = capture(base.model, seqs, [("pre", gl), ("res", gl), ("key", gl), ("att", gl)], pick=pos)
        rn = {e: [m for m in ents if m != e and lrt[m] == lrt[e]] for e in ents}
        out = {}
        for code in ("pre", "res", "key", "att"):
            F0 = {e: r[f"{code}{gl}"][0] for (e, k), r in zip(rows, recs) if k == "F0"}
            forms: Dict[str, list] = {}
            ctx: Dict[str, list] = {}
            for (e, k), r in zip(rows, recs):
                if k == "FUSED":
                    forms.setdefault(e, []).append(r[f"{code}{gl}"][0])
                elif k == "CTX":
                    ctx.setdefault(e, []).append(r[f"{code}{gl}"][0])
            out[code] = G.entity_windows(F0, forms, ctx, rn)
        return out

    def _laws(self) -> dict:
        """Laws of the calibrated read, computed from c0 before any write (PROTOCOL §4.5)."""
        out = {}
        for name, a in sorted(self.addresses.items()):
            m = -a.stats["h0_mean"]
            rows = []
            for n in self.cfg.checkpoints:
                th = theta_n(n)
                rows.append({"N": n, "theta_N": th, "false_fire_gaussian": ST.false_fire_law(n, m, th),
                             "markov_bound": ST.markov_bound(),
                             "G_formal": ST.generalisation_law(a.stats.get("h1_mean_formal", 0.0), th),
                             "G_informal": ST.generalisation_law(a.stats.get("h1_mean_informal", 0.0), th)})
            out[name] = {"m_h0": m, "var_h0_over_2m": (a.stats["h0_var"] / (2 * m)) if m > 0 else None, "by_N": rows}
        return out

    # -------------------------------------------------------------- Part A
    def run_part_a(self) -> None:
        t0 = time.time()
        cfg = self.cfg
        path = self.out / "part_a.jsonl"
        expected = set(PA.expected_configs(cfg.addresses, cfg.steps, cfg.qs, cfg.ft_refs))
        by_edit: Dict[str, List[dict]] = {}
        for r in read_jsonl(path):
            by_edit.setdefault(r["edit_id"], []).append(r)
        done = {e: rs for e, rs in by_edit.items() if {r["config"] for r in rs} == expected}
        if path.exists():      # keep complete edits only (drops a torn or partial last edit)
            write_jsonl_atomic(path, [r for e in sorted(done) for r in done[e]])
        pa = PA.PartA(self.base, self.world, self.thresholds, self.sup_proj, self.ft_cfg, layer=cfg.gate_layer,
                      steps=cfg.steps, qs=cfg.qs, addresses=cfg.addresses, refs=cfg.ft_refs,
                      max_steps=cfg.search_max_steps)
        try:
            with path.open("a", encoding="utf-8") as f:
                for item in self.items:
                    if item["edit_id"] in done:
                        continue
                    te = time.time()
                    recs = pa.run_edit(item)
                    f.write("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in recs))
                    f.flush()
                    os.fsync(f.fileno())
                    self.log(f"[ka w{self.world}] part A {item['edit_id']} ({item['lived_case']}): "
                             f"{len(recs)} cells, {time.time() - te:.0f}s")
        finally:
            pa.close()
        self.timings["parta"] = round(self.timings.get("parta", 0.0) + time.time() - t0, 1)

    # -------------------------------------------------------------- episodic family
    def _battery(self) -> ST.CachedBattery:
        if getattr(self, "_cb", None) is None:
            t0 = time.time()
            bat = ST.build_battery(self.base, self.facts, self.U, self.pool, self.world + 14)
            kmax = max(len(self.base.tok.encode(" " + f.new + ".")) for f in self.facts)
            self._cb = ST.CachedBattery(self.base, bat["probes"], list(self.addresses.values()), kmax)
            self._dropped = bat["dropped"]
            pos = torch.tensor([k in ST.POS_KINDS for k in self._cb.kind])
            self._usable = self._cb.base_ok | pos
            self._fused_rows = [i for i, k in enumerate(self._cb.kind) if k == "fused"]
            self._new_first = [self._cb.gold[self._cb.lived_row(f.idx)][0] for f in self.facts]
            self.log(f"[ka w{self.world}] battery: {len(bat['probes'])} probes, dropped {bat['dropped']}, "
                     f"base-inconsistent negatives {int((~self._usable).sum())} ({time.time() - t0:.0f}s)")
        return self._cb

    def _metrics(self, ok: torch.Tensor, n: int) -> dict:
        cb = self._cb
        return ST.battery_metrics(cb.kind, cb.ci, cb.first_gold, ok, n, self._new_first[:n], self._usable)

    def _fused_facts(self, ok: torch.Tensor, n: int) -> List[List[int]]:
        """``[fact index, 1 if the new object is produced]`` for the FUSED probe of every live fact that has one:
        G_F per fact (its mean is the record's ``fused``), kept so that intervals over facts can be computed
        from the records without another run."""
        cb = self._cb
        return [[int(cb.ci[i]), int(bool(ok[i]))] for i in self._fused_rows
                if int(cb.ci[i]) < n and bool(self._usable[i])]

    def _counted(self, n: int) -> torch.Tensor:
        cb = self._cb
        is_u = torch.tensor([k == "U" for k in cb.kind])
        return self._usable & (is_u | ((cb.ci >= 0) & (cb.ci < n)))

    def run_episodic(self) -> None:
        path = self.out / "episodic.jsonl"
        if path.exists():
            return
        t0 = time.time()
        cb = self._battery()
        cfg = self.cfg
        pos = torch.tensor([k in ST.POS_KINDS for k in cb.kind])
        is_u = torch.tensor([k == "U" for k in cb.kind])
        fused = torch.tensor([k == "fused" for k in cb.kind])
        records: List[dict] = []
        for name, addr in sorted(self.addresses.items()):
            ledger = EpisodeLedger()
            eps = cb.episodes(self.facts, addr)
            for e in eps:
                ledger.insert(e)
            cap_hits = [e.cap_hits for e in eps]
            m_h0 = -addr.stats["h0_mean"]

            def evaluate(n, rule):
                pack = ledger.pack(addr, [f.fact_id for f in self.facts[:n]])
                return pack, cb.evaluate(pack, addr, rule)

            for n in cfg.checkpoints:
                if n > len(self.facts):
                    continue
                reads = [("rule", ReadRule("rule")), ("fixed", ReadRule("fixed", theta_n(cfg.checkpoints[0])))]
                if n in (cfg.checkpoints[0], cfg.n_primary, cfg.checkpoints[-1]):
                    reads.append(("nonull", ReadRule("nonull")))
                for rname, rule in reads:
                    pack, res = evaluate(n, rule)
                    rec = {"world": self.world, "address": name, "read": rname, **self._metrics(res["ok"], n)}
                    if rname == "rule":
                        rec["fused_facts"] = self._fused_facts(res["ok"], n)
                    cnt = self._counted(n)
                    rec["ece"] = ST.ece(res["mass"][cnt], pos[cnt])
                    negs = cnt & ~pos
                    rec["auc_leak"] = ST.auc(res["mass"][negs].tolist(), (~res["ok"][negs]).tolist())
                    fz = cnt & fused
                    own = res["rho"][fz, cb.ci[fz]] if int(fz.sum()) else torch.zeros(0)
                    rec["auc_fused"] = ST.auc(own.tolist(), res["ok"][fz].tolist())
                    um = is_u & self._usable
                    th = rule.threshold(n)
                    rec["U_fire_share"] = float((res["mass"][um] > 0.5).double().mean()) if bool(um.any()) else None
                    if th is not None and bool(um.any()):
                        rec["U_any_over_theta"] = float((res["smax"][um] > th).double().mean())
                        rec["U_markov_obs"] = float((res["mass"][um] / (1 - res["mass"][um]).clamp_min(1e-12)).mean())
                        rec["law_false_fire"] = ST.false_fire_law(n, m_h0, th)
                        rec["theta"] = th
                    rec["cap_hit_share"] = sum(1 for c in cap_hits[:n] if c) / n
                    records.append(rec)
            # frontier of fixed thresholds at the primary N: matched Leak_U points (S1a, S1b)
            n = min(cfg.n_primary, len(self.facts))
            pack, res0 = evaluate(n, ReadRule("rule"))
            um = is_u & self._usable
            pts = []
            for q in (ST.FRONTIER_QS if bool(um.any()) else ()):
                thr = float(torch.quantile(res0["smax"][um], q))
                _, res = evaluate(n, ReadRule("fixed", thr))
                m_ = self._metrics(res["ok"], n)
                pts.append({"q": q, "theta": thr, "Leak_U": m_["Leak_U"], "fused": m_["fused"], "PX": m_["PX"],
                            "lived": m_["lived"], "structured": m_["structured"]})
            records.append({"world": self.world, "address": name, "read": "frontier", "N": n, "points": pts,
                            "G_F@2": ST.interp_at(pts, "Leak_U", "fused", ST.MATCHED_LEAK_U),
                            "PX@2": ST.interp_at(pts, "Leak_U", "PX", ST.MATCHED_LEAK_U)})
        write_jsonl_atomic(path, records)
        self.timings["episodic"] = round(time.time() - t0, 1)
        self.log(f"[ka w{self.world}] episodic family: {len(records)} records, {self.timings['episodic']}s")

    # -------------------------------------------------------------- FT streams
    def _bpb_texts(self) -> List[str]:
        texts = list(self.base.prose_held)
        return texts[:self.cfg.bpb_paragraphs] if self.cfg.bpb_paragraphs else texts

    def _base_bpb(self) -> float:
        if getattr(self, "_bpb0", None) is None:
            from cev11.experiment import bits_per_byte
            self._bpb0 = bits_per_byte(self.base, self._bpb_texts())
        return self._bpb0

    def _model_ok(self, n: int) -> torch.Tensor:
        """Exact check with the model as it is, on the probes that count at checkpoint n."""
        from cev11.decode import exact_check_ids
        cb = self._cb
        cnt = self._counted(n).nonzero().flatten().tolist()
        pre, tgt = [], []
        for i in cnt:
            p, t = ST.probe_ids(self.base.tok, cb.probes[i])
            pre.append(p)
            tgt.append(t)
        res = exact_check_ids(self.base.model, pre, tgt) if cnt else []
        ok = torch.zeros(len(cb.probes), dtype=torch.bool)
        ok[torch.tensor(cnt, dtype=torch.long)] = torch.tensor(res, dtype=torch.bool)
        return ok

    def run_ft(self) -> None:
        from cev11.experiment import bits_per_byte
        from cev11.model import model_sha256
        path = self.out / "ft_stream.jsonl"
        have = read_jsonl(path)
        done_arms = {r["arm"] for r in have if r.get("arm_done")}
        cfg = self.cfg
        cps = [c for c in cfg.ft_checkpoints if c <= len(self.facts)]
        if set(FT_ARMS) <= done_arms:
            return
        t0 = time.time()
        self._battery()
        bpb0 = self._base_bpb()
        records = [r for r in have if r["arm"] in done_arms]
        for arm in FT_ARMS:
            if arm in done_arms:
                continue
            ta = time.time()

            def at_checkpoint(n):
                ok = self._model_ok(n)
                rec = {"world": self.world, **self._metrics(ok, n)}
                rec["fused_facts"] = self._fused_facts(ok, n)
                newest = self._cb.lived_row(n - 1)
                rec["newest_lived"] = bool(ok[newest])
                if n == max(cps):
                    rec["bpb"] = bits_per_byte(self.base, self._bpb_texts())
                    rec["bpb_base"] = bpb0
                    rec["dbpb"] = rec["bpb"] - bpb0
                return rec

            recs = ft_stream(self.base.model, self.base.tok, self.facts, arm, self.ft_cfg[arm], cps, at_checkpoint)
            if model_sha256(self.base.model) != self.base.base_sha:
                raise RuntimeError(f"{arm} stream left the base model changed")
            if recs:
                recs[-1]["arm_done"] = True
            records += recs
            write_jsonl_atomic(path, records)
            self.timings[f"ft_{arm}"] = round(time.time() - ta, 1)      # wall-clock time goes to DONE.json only
            self.log(f"[ka w{self.world}] {arm} stream ({self.ft_cfg[arm]['config']}): {time.time() - ta:.0f}s")
        self.timings["ft"] = round(time.time() - t0, 1)

    # -------------------------------------------------------------- hooked checks
    def run_checks(self) -> None:
        from cev11.decode import exact_check_ids, greedy_reproduces
        from cev11.experiment import bits_per_byte
        from cev11.model import model_sha256
        path = self.out / "checks.json"
        if path.exists():
            return
        t0 = time.time()
        cb = self._battery()
        cfg = self.cfg
        n = min(cfg.n_primary, len(self.facts))
        out = {"world": self.world, "N": n, "addresses": {}}
        bpb0 = self._base_bpb()
        for name, addr in sorted(self.addresses.items()):
            if not name.startswith("PRED-RES"):
                continue
            cortex = EpisodicCortex(self.base.model, addr, ReadRule("rule"))
            for e in cb.episodes(self.facts[:n], addr):
                cortex.ledger.insert(e)
            pack = cortex.ledger.pack(addr)
            cached = cb.evaluate(pack, addr, ReadRule("rule"))["ok"]
            cnt = self._counted(n).nonzero().flatten().tolist()
            rnd = random.Random(self.world + 14)
            sample = sorted(rnd.sample(cnt, max(1, int(round(CHECK_SHARE * len(cnt))))))
            cortex.attach()
            try:
                pre, tgt = zip(*[ST.probe_ids(self.base.tok, cb.probes[i]) for i in sample])
                hooked = exact_check_ids(self.base.model, list(pre), list(tgt))
                greedy = [greedy_reproduces(self.base.model, self.base.tok, cb.probes[i]["prefix"],
                                            cb.probes[i]["gold"]) for i in sample]
                bpb = bits_per_byte(self.base, self._bpb_texts())
            finally:
                cortex.detach()
            out["addresses"][name] = {
                "n_sample": len(sample),
                "cached_vs_hooked_disagree": int(sum(bool(a) != bool(cached[i]) for a, i in zip(hooked, sample))),
                "hooked_vs_greedy_disagree": int(sum(bool(a) != bool(b) for a, b in zip(hooked, greedy))),
                "bpb": bpb, "bpb_base": bpb0, "dbpb": bpb - bpb0,
                "ledger_sha256": cortex.ledger.sha256(),
            }
        out["model_unchanged"] = model_sha256(self.base.model) == self.base.base_sha
        if not out["model_unchanged"]:
            raise RuntimeError("the base model changed during the checks")
        write_json_atomic(path, out)
        self.timings["checks"] = round(time.time() - t0, 1)
        self.log(f"[ka w{self.world}] hooked checks: {self.timings['checks']}s")

    # -------------------------------------------------------------- completion
    def complete_phases(self) -> List[str]:
        """Phases whose record file is in the directory and complete, whichever invocation wrote it."""
        setup = json.loads((self.out / "setup.json").read_text(encoding="utf-8"))
        return [p for p in PHASES if (self.out / RECORD_FILES[p]).exists() and not phase_problems(self.out, setup, p)]

    def complete(self) -> bool:
        """Write DONE.json iff every required phase is complete in the directory; returns whether it was written.

        DONE.json describes the directory: it lists every complete phase and hashes every record file present,
        whatever phases this invocation ran. A world that lacks a required phase gets no DONE.json, and a stale
        one is removed, so a partial world can never pass as a finished one."""
        done_path = self.out / "DONE.json"
        seconds = dict(self.timings)
        if done_path.exists():
            for k, v in json.loads(done_path.read_text(encoding="utf-8")).get("seconds", {}).items():
                seconds.setdefault(k, v)
        phases = self.complete_phases()
        if not set(self.required) <= set(phases):
            if done_path.exists():
                done_path.unlink()
            return False
        files = ["setup.json"] + [RECORD_FILES[p] for p in PHASES if (self.out / RECORD_FILES[p]).exists()]
        write_json_atomic(done_path, {
            "world": self.world, "run": self.run, "ka_version": KA_VERSION, "phases": phases,
            "required_phases": list(self.required), "protocol_config": bool(self.cfg.protocol),
            "files": {f: sha256_file(self.out / f) for f in files}, "seconds": seconds})
        return True


def run_world(base, world: int, run: str, cfg: RunConfig, out_dir, phases: Sequence[str] = PHASES,
              pred_layer: Optional[int] = None, log=print, required: Optional[Sequence[str]] = None) -> WorldRun:
    """Run ``phases`` of one world (resuming what is there). ``required`` names the phases the protocol requires
    of this world (default: all four); DONE.json is written only when all of them are complete."""
    from cev11.model import model_sha256
    unknown = [p for p in phases if p not in PHASES]
    if unknown:
        raise ValueError(f"unknown phases {unknown}")
    wr = WorldRun(base, world, run, cfg, pathlib.Path(out_dir), pred_layer=pred_layer, log=log, required=required)
    wr.prepare()
    if "parta" in phases:
        wr.run_part_a()
    if "episodic" in phases:
        wr.run_episodic()
    if "ft" in phases:
        wr.run_ft()
    if "checks" in phases:
        wr.run_checks()
    if model_sha256(base.model) != base.base_sha:
        raise RuntimeError("the base model changed during the run")
    wr.done = wr.complete()
    return wr


__all__ = ["PRED_LAYER_CANDIDATES", "PHASES", "RECORD_FILES", "RUNS", "RunConfig", "WorldRun", "run_world",
           "required_phases", "phase_problems", "stream_checkpoints", "write_excluded", "read_jsonl", "sha256_file",
           "sha256_json", "write_json_atomic", "write_jsonl_atomic", "qkey"]
