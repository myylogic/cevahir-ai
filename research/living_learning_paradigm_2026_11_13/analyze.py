# -*- coding: utf-8 -*-
"""CEV-11 "Paradigma" — confirmatory analysis and decisions (PROTOCOL §9–§11, §13, §14).

Usage (from anywhere; OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1):
    python research/living_learning_paradigm_2026_11_13/analyze.py [--results DIR] [--worlds W ...]
        [--bootstrap 10000] [--perm 10000] [--seed 0] [--map-only] [--precondition-failed H ...] [--no-figures]

Reads   <results>/edits/**/*.jsonl  (PER-EDIT records, schema below)
        <results>/lives/**/*.jsonl  (LIFE records, S3)
        <results>/tau/**/*.jsonl    (TAU-BY-LAYER records, S4)
Writes  <results>/decisions.json, <results>/tables/*.csv, <results>/analysis_env.json and, only when matplotlib is
        installed, <results>/figures/*.png. decisions.json and the tables are a deterministic function of the
        inputs and the recorded options (no timestamps), so verify.py --rerun can recompute and compare them.

=====================================================================================================================
PER-EDIT RECORD SCHEMA  —  schema id "cev11.edit.v1"; JSONL, UTF-8, one object per line, one line per
(world, arm, edit). Suggested file name: <results>/edits/<run>_<world>_<arm>.jsonl.
=====================================================================================================================
Required fields
  schema      "cev11.edit.v1"
  run         "confirm" | "select" | "map" | "ffreq" | "pilot". ONLY run == "confirm" on the analysed confirmatory
              worlds (default 1000–1007; reserves 1008/1009 only via --worlds) enters any decision. Discovery
              worlds 0 and 1 can never be analysed confirmatorily (rejected in --worlds, filtered on load), and
              F_freq / map / selection rows are descriptive only.
  world       int, the world seed.
  arm         "STD" | "ROOT" | "LAST" | "PARA-ORTH" | "FT-L" | "FT-ALL" | "NORM" | "BASE"
              (run == "map" may also use the site "PRED"; for map rows, arm = binding site).
  layer       int (fc2 block written) or null (only FT-ALL and BASE). For run == "confirm" one arm has one layer
              in every confirmatory world (the discovery-selected layer); a mismatch is an error.
  edit_id     str, from probes.edit_items; unique per (run=confirm, world, arm). All arms of a confirmatory world
              carry the same edit ids, entities, lived cases, objects and target cases (checked), and every
              confirmatory (world, arm) holds the full PROTOCOL §7 item set: 70 edits on distinct entities,
              EDITS_PER_LIVED_CASE = 10 per lived case (checked; a partial file from a crashed run is an error).
  name        str, the edited entity (bare name).
  rel         "A".
  lived_case  "nom" | "gen" | "dat" | "acc" | "loc" | "abl" | "ins". Hypotheses use the inflected-lived edits
              only; NOM edits feed V3 and the descriptive tables (PROTOCOL §7).
  old, new    str, old object (base fact) and new object (old != new).
  installed   bool. WF: exact greedy production of the new object in the lived frame.
  targets     list of target cases (PROTOCOL §7: inflected cases other than the lived case where the BASE model
              produces the old object in both orthographies; possibly fewer than 2 — the analysis excludes and
              counts such edits for H1, H2 and S1), each an object
                {"case": str, "cls": "IDENT" | "RESEG" | "FUSED", "formal": 0|1, "informal": 0|1}
              formal   = the PF5 probe of that case (formal orthography, no context) produces the new object;
              informal = the IF5 probe (same case without apostrophe) produces the new object;
              cls      = tokenizers.classify_informal(tok, name, case); NONIDENT = RESEG or FUSED.
              PARA-ORTH: "formal"/"informal" are scored under the leave-one-variant-out write that excludes exactly
              that (case, orthography) variant. NORM: "informal" is scored on the normalised informal probe.
              Optional per target (descriptive τ, PROTOCOL §9): "tau_formal", "tau_informal" (τ at the arm's key
              position) and "tau_formal_max", "tau_informal_max" (maximum over prompt positions), floats.
  PF6         {case: 0|1} for exactly the 6 formal frames other than the lived one: new object produced.
  CF14        {"P1": {case: 0|1, all 7 cases}, "P2": {case: 0|1, all 7 cases}}: new object produced in the formal
              frames after the left contexts P1 and P2.
  leak        changed-answer counts against the base model's own greedy answer (computed once by true greedy):
                {"RN": {"changed": int, "n": int, "entities": int},  n = 7 * entities, entities <= 8 (|RN|)
                 "RD": {"changed": int, "n": int, "entities": int},  n = 7 * entities, entities <= 10
                 "SS": {"changed": int, "n": int, "entities": int},  n = entities (lived frame only), <= 20
                 "RB": {"changed": int, "n": 7},  "RC": {"changed": int, "n": 7},
                 optional "PX": {"changed": int, "n": int, "entities": int}, n = 7 * entities}
              optional in every family: "adopted": int <= n, probes whose answer became the edit's new object.
Optional fields
  config        str, the arm configuration label (e.g. "step=2.0"); constant per arm for run == "confirm".
  site          str, binding site of a map row ("ROOT" | "LAST" | "PRED").
  tau           object of finite numbers / lists of numbers, descriptive τ values, e.g.
                {"WF_site": 1.0, "RN": [..per RN probe..], "RD": [..]}.
  OLD           {"count": int, "n": int}: probes (formal frames) that still produce the old object.
  bpb           {"base": float, "edited": float}: held-out bits per byte (PROTOCOL §9, 20 edits per arm).
  tok_frequent  bool, F_freq rows: the entity belongs to the tokenizer-frequent subset.
  diag          object, free-form search diagnostics (nll, steps, delta_norm, ...); not analysed.
Example (one line):
  {"schema":"cev11.edit.v1","run":"confirm","world":1000,"arm":"STD","layer":2,"edit_id":"w1000-e017",
   "name":"Ayten","rel":"A","lived_case":"abl","old":"Kemerköy","new":"Sarıpınar","installed":true,
   "targets":[{"case":"gen","cls":"IDENT","formal":1,"informal":1},
              {"case":"dat","cls":"FUSED","formal":1,"informal":0}],
   "PF6":{"nom":1,"gen":1,"dat":1,"acc":1,"loc":0,"ins":1},
   "CF14":{"P1":{"nom":1,"gen":1,"dat":0,"acc":1,"loc":1,"abl":1,"ins":1},
           "P2":{"nom":1,"gen":1,"dat":1,"acc":1,"loc":1,"abl":1,"ins":0}},
   "leak":{"RN":{"changed":3,"n":21,"entities":3},"RD":{"changed":0,"n":70,"entities":10},
           "SS":{"changed":1,"n":20,"entities":20},"RB":{"changed":0,"n":7},"RC":{"changed":0,"n":7}}}
Runners can assemble and validate a record with make_edit_record(item, ...) (item = one probes.edit_items dict);
validate_edit_record / validate_life_record / validate_tau_record check single records; load_results checks files.

LIFE RECORD SCHEMA (S3) — "cev11.life.v1"; one line per (world, variant, order), <results>/lives/*.jsonl
  schema "cev11.life.v1"; run (as above); world int; variant "J" | "I" | "Q"; order int (J and I: exactly one
  record per confirmatory world, a world with only one of them is an error; Q: orders 0..3); layer int;
  lam float | null (ledger λ, J only);
  facts: list, one entry per fact BELIEVED AT THE END of the life:
    {"fact_id": str, "entity": str, "rel": "A" | "C", "believed": str,
     "correct": int, "n": 7,            # formal frames (no context) producing the believed object
     "max_tau": float | null}           # max τ of this fact's key to the other live keys (needed for variant I)
  J and I of one world must hold the same believed facts (same stream). Optional: "retraction_residue",
  "correction_residue" (float), "leak": {"RN": {...}, "RD": {...}} as in the per-edit schema, "curve": list.
  E_end(variant) = mean over believed facts of correct / n (= mean over facts x 7 frames).

TAU-BY-LAYER RECORD SCHEMA (S4) — "cev11.tau.v1"; one line per (world, layer, entity), <results>/tau/*.jsonl
  schema "cev11.tau.v1"; run; world int; layer int; entity str; site "ROOT";
  tau_RN: list of float, τ(k_neighbour, k_entity) for each RN neighbour of the entity, keys at the root-final
  token of the sentence-initial NOM prefix, with the layer-specific C0. The same neighbours at every layer; layers
  0 and 4 must cover the same entities (checked).

=====================================================================================================================
HYPOTHESES (PROTOCOL §10; effects in shares, i.e. 0.15 = 15 pp; one unit = one qualifying edit/fact/entity)
=====================================================================================================================
Per-edit shares: PE5_e = mean_c formal_c and IE5_e = mean_c informal_c over the edit's target cases;
CE14_e = mean of the 14 CF14 outcomes; Leak_X_e = changed_X / n_X. World-level metrics are means of per-edit
shares over the qualifying edits (each edit weighs equally; PROTOCOL §9 "mean over all qualifying edits").
  H1  unit: STD inflected-lived edit with >= 2 targets.        d_w = mean_e (PE5_e - IE5_e)          m = 0.15
  H2  unit: as H1, installed, >= 1 IDENT and >= 1 NONIDENT.     d_w = mean_e g_e,
      g_e = mean_{NONIDENT c}(F_c - I_c) - mean_{IDENT c}(F_c - I_c)                                  m = 0.30
  H3  unit: STD inflected-lived, installed, |RN| >= 1.          d_w = mean_e (Leak_RN_e - Leak_RD_e)  m = 0.05
  S1  unit: as H1, paired PARA-ORTH/STD by edit id.             d_w = mean_e (IE5_e(PARA) - IE5_e(STD)) m = 0.10
      plus non-inferiority over the S1 worlds: upper bound of the two-sided 95% t-interval of the per-world
      Leak_RD difference (PARA - STD, same edits) <= 0.01 and of the Leak_RN difference (edits with |RN| >= 1)
      <= 0.05. Holm-rejected without non-inferiority -> "not supported (locality cost)"; otherwise the usual rule
      decides and the non-inferiority result is reported alongside.
  S2  unit: STD/ROOT inflected-lived edit pair.                 d_w = mean_e (CE14_e(STD) - CE14_e(ROOT)) m = 0.10
  S3  unit: fact believed at the end of the life (J/I pair).    d_w = E_end(J) - E_end(I)              m = 0.10
      plus the pooled within-world Spearman rho between max_tau and failure under I (failure = 1 - correct/n),
      over the S3 worlds, must be > 0 with within-world permutation p <= 0.05; Holm-rejected without it ->
      "inconclusive (Spearman condition not met)".
  S4  unit: entity with RN neighbours at layers 0 and 4.        d_w = median(pooled τ_RN, layer 0)
                                                                     - median(pooled τ_RN, layer 4)   m = 0.15
Decision rules: cev11/stats.py (min 15 units per world, min 7 worlds, exact sign-flip test, Holm within each
family with m = 3 and m = 4 — untested members enter Holm with p = 1 —, t-interval refutation). Manipulation
check (PROTOCOL §3): mean over confirmatory worlds of the NONIDENT share among the informal target forms of the
H1 units >= 0.25, otherwise H1 and H2 cannot be refuted. --map-only marks H1–H3 "not testable (map-only switch)";
--precondition-failed marks hypotheses "inconclusive (precondition)" (gate failure with reserves exhausted).
"""
from __future__ import annotations

import sys
import pathlib

STUDY = pathlib.Path(__file__).resolve().parent
REPO = STUDY.parents[1]
sys.path[:0] = [str(STUDY), str(REPO)]

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Any, Callable, Iterable  # noqa: E402

import numpy as np  # noqa: E402

from cev11 import stats as S  # noqa: E402
from cev11.morph_tr import CASES, INFLECTED  # noqa: E402
# INTERFACES.md lists these under analyze.py; they live in cev11/stats.py and are re-exported here.
from cev11.stats import decide, hierarchical_bootstrap, holm, signflip_pvalue, t_interval  # noqa: E402,F401

# --- schema and protocol constants ----------------------------------------------------------------------------
EDIT_SCHEMA = "cev11.edit.v1"
LIFE_SCHEMA = "cev11.life.v1"
TAU_SCHEMA = "cev11.tau.v1"
DECISIONS_SCHEMA = "cev11.decisions.v1"

RUNS = ("confirm", "select", "map", "ffreq", "pilot")
CONFIRM = "confirm"
ARMS = ("STD", "ROOT", "LAST", "PARA-ORTH", "FT-L", "FT-ALL", "NORM", "BASE")
MAP_ONLY_ARMS = ("PRED",)
LAYERLESS_ARMS = ("FT-ALL", "BASE")
CLASSES = ("IDENT", "RESEG", "FUSED")
CONTEXT_KEYS = ("P1", "P2")
LEAK_REQUIRED = ("RN", "RD", "SS", "RB", "RC")
LEAK_OPTIONAL = ("PX",)
LEAK_MAX_ENTITIES = {"RN": 8, "RD": 10, "SS": 20}
LIFE_VARIANTS = ("J", "I", "Q")
N_FORMAL_FRAMES = len(CASES)          # 7
# PROTOCOL §7: per base 70 edits on distinct entities, 10 per lived case. Every confirmatory (world, arm) must hold
# exactly this item set, so a run that crashed part-way (even if every arm stopped at the same edit) stops the
# analysis instead of silently entering with fewer edits.
EDITS_PER_LIVED_CASE = 10

DISCOVERY_WORLDS = (0, 1)
CONFIRMATORY_WORLDS = tuple(range(1000, 1008))
RESERVE_WORLDS = (1008, 1009)
N_CONFIRMATORY = len(CONFIRMATORY_WORLDS)

S4_LAYERS = (0, 4)
MC_THRESHOLD = 0.25                   # manipulation check: mean NONIDENT share >= 25%
NI_BOUNDS = {"RD": 0.01, "RN": 0.05}  # S1 non-inferiority upper bounds
SPEARMAN_ALPHA = 0.05
PERM_N = 10_000

FAMILIES = {"primary": ("H1", "H2", "H3"), "secondary": ("S1", "S2", "S3", "S4")}
HYPOTHESES: dict[str, dict[str, Any]] = {
    "H1": {"family": "primary", "margin": 0.15, "predicted": 0.25,
           "title": "subjects without apostrophes: PE5(STD) - IE5(STD)",
           "unit": "STD inflected-lived edit with >= 2 target cases"},
    "H2": {"family": "primary", "margin": 0.30, "predicted": 0.50,
           "title": "identical-key position necessary: NONIDENT minus IDENT formal-informal gap",
           "unit": "installed STD inflected-lived edit with >= 1 IDENT and >= 1 NONIDENT target"},
    "H3": {"family": "primary", "margin": 0.05, "predicted": 0.10,
           "title": "specificity cost of the root site: Leak_RN(STD) - Leak_RD(STD)",
           "unit": "installed STD inflected-lived edit with |RN| >= 1"},
    "S1": {"family": "secondary", "margin": 0.10, "predicted": None,
           "title": "variant-averaged keys: IE5(PARA-ORTH, leave-one-variant-out) - IE5(STD)",
           "unit": "PARA-ORTH/STD inflected-lived edit pair with >= 2 target cases"},
    "S2": {"family": "secondary", "margin": 0.10, "predicted": None,
           "title": "prefix averaging buys context robustness: CE14(STD) - CE14(ROOT)",
           "unit": "STD/ROOT inflected-lived edit pair"},
    "S3": {"family": "secondary", "margin": 0.10, "predicted": None,
           "title": "joint vs independent state in a life: E_end(J) - E_end(I)",
           "unit": "fact believed at the end of the life (J/I pair)"},
    "S4": {"family": "secondary", "margin": 0.15, "predicted": None,
           "title": "identity assembled with depth: median tau_RN layer 0 - layer 4",
           "unit": "entity with RN neighbours at layers 0 and 4"},
}
HYP_ORDER = ("H1", "H2", "H3", "S1", "S2", "S3", "S4")
ANALYSIS_CODE = ("analyze.py", "cev11/stats.py")


class RecordError(ValueError):
    """A record violates the schema or the cross-record integrity rules."""


# --- validation ---------------------------------------------------------------------------------------------------
def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _bit(v: Any, what: str) -> int:
    if isinstance(v, bool):
        return int(v)
    if _is_int(v) and v in (0, 1):
        return int(v)
    raise RecordError(f"{what}: expected 0/1 or a bool, got {v!r}")


def _str(rec: dict, key: str, what: str) -> str:
    v = rec.get(key)
    if not isinstance(v, str) or not v:
        raise RecordError(f"{what}: field {key!r} must be a non-empty string, got {v!r}")
    return v


def _int(rec: dict, key: str, what: str, minimum: int | None = None) -> int:
    v = rec.get(key)
    if not _is_int(v):
        raise RecordError(f"{what}: field {key!r} must be an int, got {v!r}")
    if minimum is not None and v < minimum:
        raise RecordError(f"{what}: field {key!r} must be >= {minimum}, got {v}")
    return v


def _finite(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise RecordError(f"{what}: expected a finite number, got {v!r}")
    return float(v)


def _finite_tree(v: Any, what: str) -> None:
    if isinstance(v, dict):
        for k, x in v.items():
            _finite_tree(x, f"{what}.{k}")
    elif isinstance(v, list):
        for i, x in enumerate(v):
            _finite_tree(x, f"{what}[{i}]")
    else:
        _finite(v, what)


def _leak_family(fam: str, entry: Any, what: str, per_edit: bool = True) -> dict:
    """One leak family. per_edit=True applies the probe-family structure of PROTOCOL §9 (n = 7 x entities, ...);
    life records (per_edit=False) only need changed <= n."""
    if not isinstance(entry, dict):
        raise RecordError(f"{what}: leak[{fam!r}] must be an object")
    changed = _int(entry, "changed", f"{what} leak.{fam}", 0)
    n = _int(entry, "n", f"{what} leak.{fam}", 0)
    if changed > n:
        raise RecordError(f"{what}: leak.{fam} changed={changed} > n={n}")
    out = {"changed": changed, "n": n}
    if not per_edit:
        if "entities" in entry:
            out["entities"] = _int(entry, "entities", f"{what} leak.{fam}", 0)
    elif fam in ("RB", "RC"):
        if n != N_FORMAL_FRAMES:
            raise RecordError(f"{what}: leak.{fam} must cover the 7 formal frames (n=7), got n={n}")
    else:
        ent = _int(entry, "entities", f"{what} leak.{fam}", 0)
        expect = ent if fam == "SS" else N_FORMAL_FRAMES * ent
        if n != expect:
            raise RecordError(f"{what}: leak.{fam} n={n} inconsistent with entities={ent} (expected n={expect})")
        if fam in LEAK_MAX_ENTITIES and ent > LEAK_MAX_ENTITIES[fam]:
            raise RecordError(f"{what}: leak.{fam} entities={ent} exceeds the protocol maximum "
                              f"{LEAK_MAX_ENTITIES[fam]}")
        out["entities"] = ent
    if "adopted" in entry:
        adopted = _int(entry, "adopted", f"{what} leak.{fam}", 0)
        if adopted > n:
            raise RecordError(f"{what}: leak.{fam} adopted={adopted} > n={n}")
        out["adopted"] = adopted
    return out


def _leak_block(rec: dict, what: str, required: Iterable[str], optional: Iterable[str],
                per_edit: bool = True) -> dict:
    leak = rec.get("leak")
    if not isinstance(leak, dict):
        raise RecordError(f"{what}: field 'leak' must be an object")
    required, optional = tuple(required), tuple(optional)
    unknown = set(leak) - set(required) - set(optional)
    if unknown:
        raise RecordError(f"{what}: unknown leak families {sorted(unknown)}")
    out = {}
    for fam in required:
        if fam not in leak:
            raise RecordError(f"{what}: leak family {fam!r} missing")
        out[fam] = _leak_family(fam, leak[fam], what, per_edit)
    for fam in optional:
        if fam in leak:
            out[fam] = _leak_family(fam, leak[fam], what, per_edit)
    return out


def _run_world(rec: dict, schema: str, what: str) -> tuple[str, int]:
    if not isinstance(rec, dict):
        raise RecordError(f"{what}: a record must be a JSON object")
    if rec.get("schema") != schema:
        raise RecordError(f"{what}: schema must be {schema!r}, got {rec.get('schema')!r}")
    run = rec.get("run")
    if run not in RUNS:
        raise RecordError(f"{what}: run must be one of {RUNS}, got {run!r}")
    return run, _int(rec, "world", what)


def validate_edit_record(rec: dict, where: str = "edit record") -> dict:
    """Check one per-edit record against the schema; return a normalised copy (0/1 outcomes as ints)."""
    run, world = _run_world(rec, EDIT_SCHEMA, where)
    arm = rec.get("arm")
    allowed = ARMS + (MAP_ONLY_ARMS if run == "map" else ())
    if arm not in allowed:
        raise RecordError(f"{where}: arm must be one of {allowed}, got {arm!r}")
    layer = rec.get("layer")
    if layer is None:
        if arm not in LAYERLESS_ARMS:
            raise RecordError(f"{where}: arm {arm} needs an int layer")
    elif not _is_int(layer) or layer < 0:
        raise RecordError(f"{where}: layer must be a non-negative int or null, got {layer!r}")
    what = f"{where} (world {world}, arm {arm}, edit {rec.get('edit_id')!r})"
    edit_id = _str(rec, "edit_id", what)
    name = _str(rec, "name", what)
    if rec.get("rel") != "A":
        raise RecordError(f"{what}: rel must be 'A', got {rec.get('rel')!r}")
    lived = rec.get("lived_case")
    if lived not in CASES:
        raise RecordError(f"{what}: lived_case must be one of {CASES}, got {lived!r}")
    old, new = _str(rec, "old", what), _str(rec, "new", what)
    if old == new:
        raise RecordError(f"{what}: new object equals the old one")
    if not isinstance(rec.get("installed"), bool):
        raise RecordError(f"{what}: installed must be a bool, got {rec.get('installed')!r}")

    targets_in = rec.get("targets")
    if not isinstance(targets_in, list):
        raise RecordError(f"{what}: targets must be a list")
    targets, seen = [], set()
    for i, t in enumerate(targets_in):
        tw = f"{what} targets[{i}]"
        if not isinstance(t, dict):
            raise RecordError(f"{tw}: must be an object")
        case = t.get("case")
        if case not in INFLECTED or case == lived:
            raise RecordError(f"{tw}: case must be an inflected case other than the lived case, got {case!r}")
        if case in seen:
            raise RecordError(f"{tw}: duplicate target case {case!r}")
        seen.add(case)
        cls = t.get("cls")
        if cls not in CLASSES:
            raise RecordError(f"{tw}: cls must be one of {CLASSES}, got {cls!r}")
        tt = {"case": case, "cls": cls, "formal": _bit(t.get("formal"), f"{tw} formal"),
              "informal": _bit(t.get("informal"), f"{tw} informal")}
        for key in ("tau_formal", "tau_informal", "tau_formal_max", "tau_informal_max"):
            if t.get(key) is not None:
                tt[key] = _finite(t[key], f"{tw} {key}")
        targets.append(tt)
    targets.sort(key=lambda t: CASES.index(t["case"]))

    pf6 = rec.get("PF6")
    want = [c for c in CASES if c != lived]
    if not isinstance(pf6, dict) or sorted(pf6) != sorted(want):
        raise RecordError(f"{what}: PF6 must have exactly the keys {want}")
    cf14 = rec.get("CF14")
    if not isinstance(cf14, dict) or sorted(cf14) != sorted(CONTEXT_KEYS):
        raise RecordError(f"{what}: CF14 must have exactly the keys {CONTEXT_KEYS}")
    cf_out = {}
    for ctx in CONTEXT_KEYS:
        block = cf14[ctx]
        if not isinstance(block, dict) or sorted(block) != sorted(CASES):
            raise RecordError(f"{what}: CF14[{ctx!r}] must have exactly the 7 case keys")
        cf_out[ctx] = {c: _bit(block[c], f"{what} CF14.{ctx}.{c}") for c in CASES}

    out = dict(rec)
    out.update(
        run=run, world=world, arm=arm, layer=layer, edit_id=edit_id, name=name, lived_case=lived, old=old, new=new,
        installed=bool(rec["installed"]), targets=targets,
        PF6={c: _bit(pf6[c], f"{what} PF6.{c}") for c in want}, CF14=cf_out,
        leak=_leak_block(rec, what, LEAK_REQUIRED, LEAK_OPTIONAL),
        config="" if rec.get("config") is None else rec["config"],
    )
    if not isinstance(out["config"], str):
        raise RecordError(f"{what}: config must be a string")
    if rec.get("tau") is not None:
        if not isinstance(rec["tau"], dict):
            raise RecordError(f"{what}: tau must be an object")
        _finite_tree(rec["tau"], f"{what} tau")
    if rec.get("OLD") is not None:
        o = rec["OLD"]
        if not isinstance(o, dict):
            raise RecordError(f"{what}: OLD must be an object")
        cnt, n = _int(o, "count", f"{what} OLD", 0), _int(o, "n", f"{what} OLD", 1)
        if cnt > n:
            raise RecordError(f"{what}: OLD count > n")
    if rec.get("bpb") is not None:
        b = rec["bpb"]
        if not isinstance(b, dict) or set(b) != {"base", "edited"}:
            raise RecordError(f"{what}: bpb must be {{'base': float, 'edited': float}}")
        _finite(b["base"], f"{what} bpb.base")
        _finite(b["edited"], f"{what} bpb.edited")
    if rec.get("tok_frequent") is not None and not isinstance(rec["tok_frequent"], bool):
        raise RecordError(f"{what}: tok_frequent must be a bool")
    return out


OPTIONAL_EDIT_FIELDS = frozenset({"config", "site", "tau", "OLD", "bpb", "tok_frequent", "diag"})


def make_edit_record(item: dict, *, run: str, world: int, arm: str, layer: int | None, installed: bool,
                     targets: list[dict], pf6: dict[str, int], cf14: dict[str, dict[str, int]], leak: dict,
                     **optional: Any) -> dict:
    """Assemble one per-edit record from a probes.edit_items item (edit_id, name, rel, lived_case, old, new) and
    the probe outcomes; validates it (RecordError on any violation) and returns the record to write as JSON.
    ``optional`` takes only the optional schema fields (config, site, tau, OLD, bpb, tok_frequent, diag)."""
    unknown = set(optional) - OPTIONAL_EDIT_FIELDS
    if unknown:
        raise RecordError(f"make_edit_record: unknown optional fields {sorted(unknown)}")
    rec = {"schema": EDIT_SCHEMA, "run": run, "world": world, "arm": arm, "layer": layer,
           "edit_id": item["edit_id"], "name": item["name"], "rel": item["rel"], "lived_case": item["lived_case"],
           "old": item["old"], "new": item["new"], "installed": installed, "targets": targets, "PF6": pf6,
           "CF14": cf14, "leak": leak}
    rec.update(optional)
    validate_edit_record(rec, "make_edit_record")
    return rec


def validate_life_record(rec: dict, where: str = "life record") -> dict:
    """Check one life record (S3) against the schema; return a normalised copy."""
    run, world = _run_world(rec, LIFE_SCHEMA, where)
    variant = rec.get("variant")
    if variant not in LIFE_VARIANTS:
        raise RecordError(f"{where}: variant must be one of {LIFE_VARIANTS}, got {variant!r}")
    what = f"{where} (world {world}, variant {variant})"
    order = _int(rec, "order", what, 0)
    layer = _int(rec, "layer", what, 0)
    lam = rec.get("lam")
    if lam is not None:
        _finite(lam, f"{what} lam")
    facts_in = rec.get("facts")
    if not isinstance(facts_in, list) or not facts_in:
        raise RecordError(f"{what}: facts must be a non-empty list")
    facts, seen = [], set()
    for i, f in enumerate(facts_in):
        fw = f"{what} facts[{i}]"
        if not isinstance(f, dict):
            raise RecordError(f"{fw}: must be an object")
        fid = _str(f, "fact_id", fw)
        if fid in seen:
            raise RecordError(f"{fw}: duplicate fact_id {fid!r}")
        seen.add(fid)
        if f.get("rel") not in ("A", "C"):
            raise RecordError(f"{fw}: rel must be 'A' or 'C', got {f.get('rel')!r}")
        n = _int(f, "n", fw, 1)
        if n != N_FORMAL_FRAMES:
            raise RecordError(f"{fw}: n must be 7 (the formal frames), got {n}")
        correct = _int(f, "correct", fw, 0)
        if correct > n:
            raise RecordError(f"{fw}: correct={correct} > n={n}")
        mt = f.get("max_tau")
        facts.append({"fact_id": fid, "entity": _str(f, "entity", fw), "rel": f["rel"],
                      "believed": _str(f, "believed", fw), "correct": correct, "n": n,
                      "max_tau": None if mt is None else _finite(mt, f"{fw} max_tau")})
    facts.sort(key=lambda f: f["fact_id"])
    out = dict(rec)
    out.update(run=run, world=world, variant=variant, order=order, layer=layer, facts=facts)
    for key in ("retraction_residue", "correction_residue"):
        if rec.get(key) is not None:
            out[key] = _finite(rec[key], f"{what} {key}")
    if rec.get("leak") is not None:
        out["leak"] = _leak_block(rec, what, (), ("RN", "RD"), per_edit=False)
    return out


def validate_tau_record(rec: dict, where: str = "tau record") -> dict:
    """Check one τ-by-layer record (S4) against the schema; return a normalised copy."""
    run, world = _run_world(rec, TAU_SCHEMA, where)
    what = f"{where} (world {world}, layer {rec.get('layer')!r}, entity {rec.get('entity')!r})"
    layer = _int(rec, "layer", what, 0)
    entity = _str(rec, "entity", what)
    if rec.get("site") != "ROOT":
        raise RecordError(f"{what}: site must be 'ROOT', got {rec.get('site')!r}")
    vals = rec.get("tau_RN")
    if not isinstance(vals, list):
        raise RecordError(f"{what}: tau_RN must be a list")
    out = dict(rec)
    out.update(run=run, world=world, layer=layer, entity=entity,
               tau_RN=[_finite(v, f"{what} tau_RN[{i}]") for i, v in enumerate(vals)])
    return out


# --- loading --------------------------------------------------------------------------------------------------------
@dataclass
class Data:
    edits: list[dict] = field(default_factory=list)
    lives: list[dict] = field(default_factory=list)
    taus: list[dict] = field(default_factory=list)
    inputs: dict[str, list[dict]] = field(default_factory=dict)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_dir(results_dir: pathlib.Path, sub: str, validator: Callable[[dict, str], dict]) -> tuple[list, list]:
    directory = results_dir / sub
    records, inputs = [], []
    if not directory.exists():
        return records, inputs
    for path in sorted(directory.rglob("*.jsonl"), key=lambda p: p.relative_to(results_dir).as_posix()):
        raw = path.read_bytes()
        rel = path.relative_to(results_dir).as_posix()
        n = 0
        # split on "\n" only: str.splitlines() also breaks at U+2028/U+2029/U+0085, which json.dumps(...,
        # ensure_ascii=False) leaves unescaped inside strings, and would reject valid JSONL. A trailing "\r"
        # (CRLF files) is JSON whitespace.
        for lineno, line in enumerate(raw.decode("utf-8").split("\n"), 1):
            if not line.strip():
                continue
            where = f"{rel}:{lineno}"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RecordError(f"{where}: invalid JSON ({exc})") from exc
            records.append(validator(obj, where))
            n += 1
        inputs.append({"path": rel, "sha256": sha256_bytes(raw), "records": n})
    return records, inputs


def load_results(results_dir: pathlib.Path) -> Data:
    """Load and validate every input record under results_dir (edits/, lives/, tau/). An existing but empty
    directory yields no records (every hypothesis then reads 'not run'); a missing directory is an error."""
    results_dir = pathlib.Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results directory not found: {results_dir}")
    edits, e_in = _load_dir(results_dir, "edits", validate_edit_record)
    lives, l_in = _load_dir(results_dir, "lives", validate_life_record)
    taus, t_in = _load_dir(results_dir, "tau", validate_tau_record)
    return Data(edits=edits, lives=lives, taus=taus, inputs={"edits": e_in, "lives": l_in, "tau": t_in})


# --- confirmatory selection and integrity ---------------------------------------------------------------------------
def confirmatory_worlds(worlds: Iterable[int] | None = None) -> tuple[int, ...]:
    """The analysed confirmatory world set. Discovery worlds can never be part of it (by construction)."""
    ws = tuple(sorted(set(CONFIRMATORY_WORLDS if worlds is None else (int(w) for w in worlds))))
    if not ws:
        raise ValueError("no confirmatory worlds given")
    bad = [w for w in ws if w in DISCOVERY_WORLDS]
    if bad:
        raise ValueError(f"discovery worlds {bad} can never be analysed confirmatorily")
    unknown = [w for w in ws if w not in CONFIRMATORY_WORLDS + RESERVE_WORLDS]
    if unknown:
        raise ValueError(f"worlds {unknown} are neither confirmatory (1000-1007) nor reserves (1008, 1009)")
    if len(ws) > N_CONFIRMATORY:
        raise ValueError(f"at most {N_CONFIRMATORY} confirmatory worlds (reserves replace failed bases), got {len(ws)}")
    if RESERVE_WORLDS[1] in ws and RESERVE_WORLDS[0] not in ws:
        raise ValueError("reserve 1009 may only be used after reserve 1008")
    return ws


def _split_confirmatory(records: list[dict], worlds: tuple[int, ...]) -> tuple[list[dict], dict[str, int]]:
    keep, excluded = [], {}
    for r in records:
        if r["run"] == CONFIRM and r["world"] in worlds:
            keep.append(r)
            continue
        if r["world"] in DISCOVERY_WORLDS:
            reason = "discovery world"
        elif r["run"] != CONFIRM:
            reason = f"run={r['run']}"
        else:
            reason = "world not in the analysed confirmatory set"
        key = f"{reason} | world {r['world']}"
        excluded[key] = excluded.get(key, 0) + 1
    assert not any(r["world"] in DISCOVERY_WORLDS for r in keep)
    return keep, dict(sorted(excluded.items()))


def _target_signature(rec: dict) -> tuple:
    return tuple((t["case"], t["cls"]) for t in rec["targets"])


def index_confirmatory_edits(edits: list[dict]) -> dict[tuple[int, str], dict[str, dict]]:
    """{(world, arm): {edit_id: record}} with the integrity checks of the per-edit schema."""
    idx: dict[tuple[int, str], dict[str, dict]] = {}
    arm_layer: dict[str, tuple] = {}
    for r in edits:
        key = (r["world"], r["arm"])
        bucket = idx.setdefault(key, {})
        if r["edit_id"] in bucket:
            raise RecordError(f"duplicate confirmatory record: world {r['world']}, arm {r['arm']}, edit {r['edit_id']}")
        bucket[r["edit_id"]] = r
        cfg = (r["layer"], r["config"])
        if arm_layer.setdefault(r["arm"], cfg) != cfg:
            raise RecordError(f"arm {r['arm']}: layer/config {cfg} differs from {arm_layer[r['arm']]} — one "
                              "discovery-selected configuration per arm is required in all confirmatory worlds")
    worlds = sorted({w for w, _ in idx})
    for w in worlds:
        arms = sorted(a for ww, a in idx if ww == w)
        ref_arm = arms[0]
        ref = idx[(w, ref_arm)]
        for a in arms[1:]:
            cur = idx[(w, a)]
            if set(cur) != set(ref):
                raise RecordError(f"world {w}: arms {ref_arm} and {a} cover different edits "
                                  f"({len(set(ref) ^ set(cur))} ids differ) — all arms use the same items")
            for eid, rec in cur.items():
                r0 = ref[eid]
                same = (rec["name"], rec["lived_case"], rec["old"], rec["new"], _target_signature(rec)) == \
                       (r0["name"], r0["lived_case"], r0["old"], r0["new"], _target_signature(r0))
                if not same:
                    raise RecordError(f"world {w} edit {eid}: arms {ref_arm} and {a} disagree on the item "
                                      "(entity, lived case, objects or target cases/classes)")
    # PROTOCOL §7 item set: distinct entities, EDITS_PER_LIVED_CASE edits for every lived case. The cross-arm check
    # above cannot see a crash that truncated every arm at the same edit (or a world with a single arm).
    for (w, a), bucket in sorted(idx.items()):
        names = [r["name"] for r in bucket.values()]
        if len(set(names)) != len(names):
            raise RecordError(f"world {w}, arm {a}: an entity is edited more than once (PROTOCOL §7: distinct "
                              "entities)")
        counts = {c: 0 for c in CASES}
        for r in bucket.values():
            counts[r["lived_case"]] += 1
        if any(v != EDITS_PER_LIVED_CASE for v in counts.values()):
            raise RecordError(f"world {w}, arm {a}: edits per lived case {counts}, expected {EDITS_PER_LIVED_CASE} "
                              "for every case (PROTOCOL §7) — incomplete or crashed run; rerun it from scratch")
    return idx


def index_confirmatory_lives(lives: list[dict]) -> dict[tuple[int, str], list[dict]]:
    idx: dict[tuple[int, str], list[dict]] = {}
    for r in lives:
        bucket = idx.setdefault((r["world"], r["variant"]), [])
        if any(x["order"] == r["order"] for x in bucket):
            raise RecordError(f"duplicate life record: world {r['world']}, variant {r['variant']}, order {r['order']}")
        bucket.append(r)
    for (w, v), bucket in idx.items():
        bucket.sort(key=lambda x: x["order"])
        if v in ("J", "I") and len(bucket) != 1:
            raise RecordError(f"world {w}: variant {v} must have exactly one life record, got {len(bucket)}")
    # J and I run the same stream (PROTOCOL §13): a world with only one of them is an incomplete run, which must not
    # silently drop the world from S3.
    for w in sorted({w for w, _ in idx}):
        have = [v for v in ("J", "I") if (w, v) in idx]
        if len(have) == 1:
            raise RecordError(f"world {w}: life variant {have[0]} present without its J/I partner — incomplete run")
    return idx


def index_confirmatory_taus(taus: list[dict]) -> dict[tuple[int, int], dict[str, dict]]:
    idx: dict[tuple[int, int], dict[str, dict]] = {}
    for r in taus:
        bucket = idx.setdefault((r["world"], r["layer"]), {})
        if r["entity"] in bucket:
            raise RecordError(f"duplicate tau record: world {r['world']}, layer {r['layer']}, entity {r['entity']}")
        bucket[r["entity"]] = r
    return idx


# --- per-edit quantities -------------------------------------------------------------------------------------------
def _avg(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else None


def pe5(rec: dict) -> float:
    return sum(t["formal"] for t in rec["targets"]) / len(rec["targets"])


def ie5(rec: dict) -> float:
    return sum(t["informal"] for t in rec["targets"]) / len(rec["targets"])


def pe6(rec: dict) -> float:
    return sum(rec["PF6"].values()) / len(rec["PF6"])


def ce14(rec: dict) -> float:
    vals = [v for ctx in CONTEXT_KEYS for v in rec["CF14"][ctx].values()]
    return sum(vals) / len(vals)


def leak_share(rec: dict, fam: str) -> float | None:
    entry = rec["leak"].get(fam)
    if entry is None or entry["n"] == 0:
        return None
    return entry["changed"] / entry["n"]


def h2_gap(rec: dict) -> float | None:
    """g_e for an installed edit with >= 1 IDENT and >= 1 NONIDENT target; None when it does not qualify."""
    ident = [t["formal"] - t["informal"] for t in rec["targets"] if t["cls"] == "IDENT"]
    non = [t["formal"] - t["informal"] for t in rec["targets"] if t["cls"] != "IDENT"]
    if not rec["installed"] or not ident or not non:
        return None
    return sum(non) / len(non) - sum(ident) / len(ident)


def _inflected(bucket: dict[str, dict]) -> list[dict]:
    return [bucket[k] for k in sorted(bucket) if bucket[k]["lived_case"] != "nom"]


@dataclass
class WorldResult:
    units: list
    d: float | None
    components: dict

    @property
    def n(self) -> int:
        return len(self.units)


def _world(units: list[float], **components: Any) -> WorldResult:
    return WorldResult(units=units, d=_avg(units), components=components)


# --- per-world effects, one function per hypothesis ---------------------------------------------------------------
def per_world_h1(idx: dict, worlds: tuple[int, ...]) -> dict[int, WorldResult]:
    out = {}
    for w in worlds:
        std = idx.get((w, "STD"))
        if std is None:
            continue
        units, pes, ies, n_infl, n_few = [], [], [], 0, 0
        for r in _inflected(std):
            n_infl += 1
            if len(r["targets"]) < 2:
                n_few += 1
                continue
            pes.append(pe5(r))
            ies.append(ie5(r))
            units.append(pes[-1] - ies[-1])
        out[w] = _world(units, PE5=_avg(pes), IE5=_avg(ies), n_inflected=n_infl, excluded_lt2_targets=n_few)
    return out


def per_world_h2(idx: dict, worlds: tuple[int, ...]) -> dict[int, WorldResult]:
    out = {}
    for w in worlds:
        std = idx.get((w, "STD"))
        if std is None:
            continue
        units, gap_non, gap_id, n_few, n_not_inst, n_no_mix = [], [], [], 0, 0, 0
        for r in _inflected(std):
            if len(r["targets"]) < 2:
                n_few += 1
                continue
            if not r["installed"]:
                n_not_inst += 1
                continue
            g = h2_gap(r)
            if g is None:
                n_no_mix += 1
                continue
            non = [t["formal"] - t["informal"] for t in r["targets"] if t["cls"] != "IDENT"]
            ide = [t["formal"] - t["informal"] for t in r["targets"] if t["cls"] == "IDENT"]
            gap_non.append(sum(non) / len(non))
            gap_id.append(sum(ide) / len(ide))
            units.append(g)
        out[w] = _world(units, gap_NONIDENT=_avg(gap_non), gap_IDENT=_avg(gap_id), excluded_lt2_targets=n_few,
                        excluded_not_installed=n_not_inst, excluded_no_ident_nonident_mix=n_no_mix)
    return out


def per_world_h3(idx: dict, worlds: tuple[int, ...]) -> dict[int, WorldResult]:
    out = {}
    for w in worlds:
        std = idx.get((w, "STD"))
        if std is None:
            continue
        units, lrn, lrd, n_not_inst, n_no_rn = [], [], [], 0, 0
        for r in _inflected(std):
            if not r["installed"]:
                n_not_inst += 1
                continue
            if r["leak"]["RN"]["entities"] < 1:
                n_no_rn += 1
                continue
            if r["leak"]["RD"]["n"] < 1:
                raise RecordError(f"world {w} edit {r['edit_id']}: RD has no probes; Leak_RD is undefined")
            lrn.append(leak_share(r, "RN"))
            lrd.append(leak_share(r, "RD"))
            units.append(lrn[-1] - lrd[-1])
        out[w] = _world(units, Leak_RN=_avg(lrn), Leak_RD=_avg(lrd), excluded_not_installed=n_not_inst,
                        excluded_no_RN=n_no_rn)
    return out


def per_world_s1(idx: dict, worlds: tuple[int, ...]) -> dict[int, WorldResult]:
    out = {}
    for w in worlds:
        std, para = idx.get((w, "STD")), idx.get((w, "PARA-ORTH"))
        if std is None or para is None:
            continue
        units, ie_p, ie_s, d_rd, d_rn, n_few = [], [], [], [], [], 0
        split: dict[str, list[int]] = {k: [] for k in ("IDENT_PARA", "IDENT_STD", "NONIDENT_PARA", "NONIDENT_STD")}
        for s in _inflected(std):
            p = para[s["edit_id"]]
            if len(s["targets"]) < 2:
                n_few += 1
                continue
            ie_p.append(ie5(p))
            ie_s.append(ie5(s))
            units.append(ie_p[-1] - ie_s[-1])
            # PROTOCOL §10.2: the τ-account predicts failure on NONIDENT targets; report both target classes
            for tp, ts in zip(p["targets"], s["targets"]):     # same cases in the same order (checked on load)
                cls = "IDENT" if ts["cls"] == "IDENT" else "NONIDENT"
                split[f"{cls}_PARA"].append(tp["informal"])
                split[f"{cls}_STD"].append(ts["informal"])
            for fam, sink in (("RD", d_rd), ("RN", d_rn)):
                if s["leak"][fam]["n"] != p["leak"][fam]["n"]:
                    raise RecordError(f"world {w} edit {s['edit_id']}: {fam} probe sets differ between STD and "
                                      "PARA-ORTH")
                if s["leak"][fam]["n"] > 0:
                    sink.append(leak_share(p, fam) - leak_share(s, fam))
                elif fam == "RD":
                    raise RecordError(f"world {w} edit {s['edit_id']}: RD has no probes; Leak_RD is undefined")
        out[w] = _world(units, IE5_PARA=_avg(ie_p), IE5_STD=_avg(ie_s), Leak_RD_diff=_avg(d_rd),
                        Leak_RN_diff=_avg(d_rn), n_RN_edits=len(d_rn), excluded_lt2_targets=n_few,
                        **{f"IE_{k}_pooled": _avg(v) for k, v in split.items()})
    return out


def per_world_s2(idx: dict, worlds: tuple[int, ...]) -> dict[int, WorldResult]:
    out = {}
    for w in worlds:
        std, root = idx.get((w, "STD")), idx.get((w, "ROOT"))
        if std is None or root is None:
            continue
        units, c_std, c_root = [], [], []
        for s in _inflected(std):
            c_std.append(ce14(s))
            c_root.append(ce14(root[s["edit_id"]]))
            units.append(c_std[-1] - c_root[-1])
        out[w] = _world(units, CE14_STD=_avg(c_std), CE14_ROOT=_avg(c_root))
    return out


def per_world_s3(lives_idx: dict, worlds: tuple[int, ...]) -> dict[int, WorldResult]:
    out = {}
    for w in worlds:
        j, i = lives_idx.get((w, "J")), lives_idx.get((w, "I"))
        if j is None or i is None:
            continue
        fj = {f["fact_id"]: f for f in j[0]["facts"]}
        fi = {f["fact_id"]: f for f in i[0]["facts"]}
        if set(fj) != set(fi):
            raise RecordError(f"world {w}: lives J and I end with different believed facts — same stream required")
        units, ej, ei, taus, fails = [], [], [], [], []
        for fid in sorted(fj):
            a, b = fj[fid], fi[fid]
            if (a["entity"], a["rel"], a["believed"]) != (b["entity"], b["rel"], b["believed"]):
                raise RecordError(f"world {w} fact {fid}: J and I disagree on the believed fact")
            ej.append(a["correct"] / a["n"])
            ei.append(b["correct"] / b["n"])
            units.append(ej[-1] - ei[-1])
            taus.append(b["max_tau"])
            fails.append(1.0 - ei[-1])
        out[w] = _world(units, E_end_J=_avg(ej), E_end_I=_avg(ei), spearman_x=taus, spearman_y=fails,
                        retraction_residue_J=j[0].get("retraction_residue"),
                        retraction_residue_I=i[0].get("retraction_residue"))
    return out


def _s4_stat(items: list) -> float:
    a0 = np.concatenate([np.asarray(x[0], dtype=np.float64) for x in items])
    a4 = np.concatenate([np.asarray(x[1], dtype=np.float64) for x in items])
    return float(np.median(a0) - np.median(a4))


def per_world_s4(tau_idx: dict, worlds: tuple[int, ...]) -> dict[int, WorldResult]:
    out = {}
    lo, hi = S4_LAYERS
    for w in worlds:
        if not any(ww == w for ww, _ in tau_idx):
            continue
        e0, e4 = tau_idx.get((w, lo), {}), tau_idx.get((w, hi), {})
        if set(e0) != set(e4):   # a partial τ file must not silently shrink the entity set
            raise RecordError(f"world {w}: τ records at layers {lo} and {hi} cover different entities "
                              f"({len(set(e0) ^ set(e4))} differ) — incomplete run")
        units = []
        for ent in sorted(set(e0) & set(e4)):
            a0, a4 = e0[ent]["tau_RN"], e4[ent]["tau_RN"]
            if len(a0) != len(a4):
                raise RecordError(f"world {w} entity {ent}: different RN neighbour counts at layers {lo} and {hi}")
            if a0:
                units.append((tuple(a0), tuple(a4)))
        res = WorldResult(units=units, d=_s4_stat(units) if units else None, components={})
        if units:
            res.components = {"median_tau_RN_L0": float(np.median(np.concatenate([u[0] for u in units]))),
                              "median_tau_RN_L4": float(np.median(np.concatenate([u[1] for u in units]))),
                              "n_pairs": int(sum(len(u[0]) for u in units))}
        out[w] = res
    return out


# --- evaluation -----------------------------------------------------------------------------------------------------
def manipulation_check(idx: dict, worlds: tuple[int, ...]) -> dict:
    """NONIDENT share among the informal target forms of the H1 units (STD), per world and averaged."""
    per_world = {}
    for w in worlds:
        std = idx.get((w, "STD"))
        if std is None:
            continue
        n = non = 0
        for r in _inflected(std):
            if len(r["targets"]) < 2:
                continue
            n += len(r["targets"])
            non += sum(t["cls"] != "IDENT" for t in r["targets"])
        if n:
            per_world[w] = {"nonident_share": non / n, "n_forms": n}
    shares = [v["nonident_share"] for v in per_world.values()]
    mean_share = _avg(shares)
    return {"threshold": MC_THRESHOLD, "per_world": per_world, "mean_nonident_share": mean_share,
            "passed": mean_share is not None and mean_share >= MC_THRESHOLD,
            "computable": mean_share is not None}


def _noninferiority(per_world: dict[int, WorldResult], qual: list[int]) -> dict:
    out: dict[str, Any] = {}
    met = True
    for fam, key in (("RD", "Leak_RD_diff"), ("RN", "Leak_RN_diff")):
        vals = {w: per_world[w].components[key] for w in qual if per_world[w].components[key] is not None}
        lo, hi = t_interval(list(vals.values())) if len(vals) >= 2 else (math.nan, math.nan)
        ok = math.isfinite(hi) and hi <= NI_BOUNDS[fam]
        out[fam] = {"bound": NI_BOUNDS[fam], "per_world": vals, "mean": _avg(vals.values()),
                    "t_interval_95": [lo, hi], "met": ok,
                    "note": None if len(vals) >= 2 else "fewer than 2 worlds with values: not computable"}
        met = met and ok
    out["met"] = met
    return out


def _spearman(per_world: dict[int, WorldResult], qual: list[int], perm_n: int, seed: int) -> dict:
    groups, missing, present = {}, 0, 0
    for w in qual:
        xs, ys = per_world[w].components["spearman_x"], per_world[w].components["spearman_y"]
        missing += sum(x is None for x in xs)
        present += sum(x is not None for x in xs)
        groups[w] = (xs, ys)
    if present == 0:
        return {"rho": None, "p": None, "met": False, "alpha": SPEARMAN_ALPHA,
                "note": "max_tau not recorded for variant I: Spearman condition cannot be evaluated"}
    if missing:
        raise RecordError(f"S3: max_tau missing for {missing} believed facts under I (present for {present})")
    res = S.spearman_within_world(groups, n_perm=perm_n, seed=seed)
    res["alpha"] = SPEARMAN_ALPHA
    res["met"] = res["rho"] is not None and res["rho"] > 0 and res["p"] <= SPEARMAN_ALPHA
    return res


def _hyp_seed(seed: int, h: str) -> int:
    return seed * 1000 + HYP_ORDER.index(h) + 1


def compute_decisions(data: Data, *, worlds: Iterable[int] | None = None, bootstrap_n: int = S.BOOTSTRAP_N,
                      perm_n: int = PERM_N, seed: int = 0, map_only: bool = False,
                      precondition_failed: Iterable[str] = ()) -> dict:
    """All confirmatory decisions as a JSON-ready dict (deterministic in the inputs and options)."""
    ws = confirmatory_worlds(worlds)
    pre = sorted(set(precondition_failed))
    unknown = [h for h in pre if h not in HYPOTHESES]
    if unknown:
        raise ValueError(f"unknown hypotheses in precondition_failed: {unknown}")
    overrides: dict[str, str] = {h: S.PRECONDITION for h in pre}
    if map_only:
        clash = [h for h in FAMILIES["primary"] if h in overrides]
        if clash:
            raise ValueError(f"{clash} cannot be both 'precondition' and 'map-only'")
        overrides.update({h: S.NOT_TESTABLE for h in FAMILIES["primary"]})

    edits, ex_e = _split_confirmatory(data.edits, ws)
    lives, ex_l = _split_confirmatory(data.lives, ws)
    taus, ex_t = _split_confirmatory(data.taus, ws)
    idx = index_confirmatory_edits(edits)
    lives_idx = index_confirmatory_lives(lives)
    tau_idx = index_confirmatory_taus(taus)

    per_world = {
        "H1": per_world_h1(idx, ws), "H2": per_world_h2(idx, ws), "H3": per_world_h3(idx, ws),
        "S1": per_world_s1(idx, ws), "S2": per_world_s2(idx, ws), "S3": per_world_s3(lives_idx, ws),
        "S4": per_world_s4(tau_idx, ws),
    }
    mc = manipulation_check(idx, ws)

    hyp: dict[str, dict] = {}
    for h in HYP_ORDER:
        spec = HYPOTHESES[h]
        pw = per_world[h]
        qual = [w for w in sorted(pw) if pw[w].n >= S.MIN_EDITS_PER_WORLD]
        d = [pw[w].d for w in qual]
        sufficient = len(qual) >= S.MIN_WORLDS
        p = signflip_pvalue(d, spec["margin"]) if (sufficient and h not in overrides) else None
        lo, hi = t_interval(d) if len(d) >= 2 else (math.nan, math.nan)
        boot = None
        if qual and bootstrap_n > 0:
            units = {w: pw[w].units for w in qual}
            boot = list(hierarchical_bootstrap(units, n=bootstrap_n, seed=_hyp_seed(seed, h),
                                               stat=_s4_stat if h == "S4" else None))
        hyp[h] = {
            "family": spec["family"], "title": spec["title"], "unit": spec["unit"], "margin": spec["margin"],
            "predicted": spec["predicted"], "worlds_with_data": sorted(pw), "qualifying_worlds": qual,
            "n_qualifying_worlds": len(qual), "d_qualifying": d, "mean_d": _avg(d), "t_interval_95": [lo, hi],
            "p_signflip": p, "bootstrap_95": boot,
            "per_world": {w: {"n_units": pw[w].n, "qualifies": pw[w].n >= S.MIN_EDITS_PER_WORLD, "d": pw[w].d,
                              "components": {k: v for k, v in pw[w].components.items()
                                             if k not in ("spearman_x", "spearman_y")}}
                          for w in sorted(pw)},
        }
        if h == "S1":
            hyp[h]["noninferiority"] = _noninferiority(pw, qual) if qual else {"met": False, "note": "no S1 worlds"}
        if h == "S3":
            hyp[h]["spearman"] = (_spearman(pw, qual, perm_n, seed * 1000 + 100) if qual
                                  else {"rho": None, "p": None, "met": False, "note": "no S3 worlds"})

    families = {}
    for fam, members in FAMILIES.items():
        pv = {h: (hyp[h]["p_signflip"] if hyp[h]["p_signflip"] is not None else 1.0) for h in members}
        rej = holm(pv, S.ALPHA)
        adj = S.holm_adjusted(pv)
        families[fam] = {"m": len(members), "alpha": S.ALPHA, "p_for_holm": pv, "holm_rejected": rej,
                         "holm_adjusted_p": adj}
        for h in members:
            hyp[h]["p_for_holm"] = pv[h]
            hyp[h]["holm_rejected"] = rej[h]
            hyp[h]["holm_adjusted_p"] = adj[h]

    for h in HYP_ORDER:
        e = hyp[h]
        reasons = []
        if h in overrides:
            label = overrides[h]
            reasons.append("override: " + ("map-only switch" if label == S.NOT_TESTABLE else "precondition failed"))
        elif not e["worlds_with_data"]:
            label = S.NOT_RUN
            reasons.append("no confirmatory records for this hypothesis")
        else:
            can_refute = mc["passed"] if h in ("H1", "H2") else True
            label = decide(e["d_qualifying"], e["margin"], e["holm_rejected"], can_refute=can_refute)
            if label == S.INSUFFICIENT:
                reasons.append(f"{e['n_qualifying_worlds']} worlds with >= {S.MIN_EDITS_PER_WORLD} units "
                               f"(< {S.MIN_WORLDS})")
            if label == S.WEAK_MANIPULATION:
                reasons.append("manipulation check failed: H1/H2 cannot be refuted")
            if h == "S1" and label == S.SUPPORTED and not e["noninferiority"]["met"]:
                label = S.LOCALITY_COST
                reasons.append("Holm-rejected but the non-inferiority bounds are not met")
            if h == "S3" and label == S.SUPPORTED and not e["spearman"]["met"]:
                label = S.SPEARMAN_NOT_MET
                reasons.append("Holm-rejected but the within-world Spearman condition is not met")
        e["decision"] = label
        e["reasons"] = reasons

    decisions = {
        "schema": DECISIONS_SCHEMA,
        "protocol": "CEV-11 Paradigma v1.0 — PROTOCOL.md §10–§11 (decision rules in cev11/stats.py)",
        "analysis_code": analysis_code_hashes(),
        "options": {"worlds": list(ws), "bootstrap_n": bootstrap_n, "perm_n": perm_n, "seed": seed,
                    "map_only": map_only, "precondition_failed": pre},
        "constants": {"alpha": S.ALPHA, "min_units_per_world": S.MIN_EDITS_PER_WORLD, "min_worlds": S.MIN_WORLDS,
                      "edits_per_lived_case": EDITS_PER_LIVED_CASE,
                      "t_conf": S.T_CONF, "margins": {h: HYPOTHESES[h]["margin"] for h in HYP_ORDER},
                      "families": {k: list(v) for k, v in FAMILIES.items()}, "mc_threshold": MC_THRESHOLD,
                      "ni_bounds": NI_BOUNDS, "spearman_alpha": SPEARMAN_ALPHA, "s4_layers": list(S4_LAYERS),
                      "discovery_worlds": list(DISCOVERY_WORLDS), "confirmatory_worlds": list(CONFIRMATORY_WORLDS),
                      "reserve_worlds": list(RESERVE_WORLDS)},
        "inputs": data.inputs,
        "excluded_records": {"edits": ex_e, "lives": ex_l, "tau": ex_t},
        "manipulation_check": mc,
        "families": families,
        "hypotheses": hyp,
        "summary": {h: hyp[h]["decision"] for h in HYP_ORDER},
    }
    return jsonable(decisions)


def jsonable(obj: Any) -> Any:
    """Plain-JSON copy: numpy scalars to Python, NaN/inf to None, tuples to lists, dict keys to str."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if obj is None or isinstance(obj, str):
        return obj
    raise TypeError(f"not JSON-serialisable: {type(obj).__name__}")


def analysis_code_hashes() -> dict[str, str]:
    return {rel: sha256_bytes((STUDY / rel).read_bytes()) for rel in ANALYSIS_CODE}


# --- descriptive tables -------------------------------------------------------------------------------------------
def _group_metrics(recs: list[dict]) -> dict:
    infl = [r for r in recs if r["lived_case"] != "nom"]
    h1u = [r for r in infl if len(r["targets"]) >= 2]
    inst = [r for r in recs if r["installed"]]
    inst_h1u = [r for r in h1u if r["installed"]]
    tf = {(cls, o): [] for cls in ("IDENT", "NONIDENT") for o in ("formal", "informal")}
    for r in h1u:
        for t in r["targets"]:
            cls = "IDENT" if t["cls"] == "IDENT" else "NONIDENT"
            tf[(cls, "formal")].append(t["formal"])
            tf[(cls, "informal")].append(t["informal"])
    m: dict[str, Any] = {
        "n_edits": len(recs), "n_inflected": len(infl), "n_h1_units": len(h1u),
        "excluded_lt2_targets": len(infl) - len(h1u),
        "WE": _avg(r["installed"] for r in recs), "WE_inflected": _avg(r["installed"] for r in infl),
        "PE6": _avg(pe6(r) for r in recs), "PE5": _avg(pe5(r) for r in h1u), "IE5": _avg(ie5(r) for r in h1u),
        "CE14": _avg(ce14(r) for r in infl),
        "TPE6": _avg(pe6(r) for r in inst), "TPE5": _avg(pe5(r) for r in inst_h1u),
        "TIE5": _avg(ie5(r) for r in inst_h1u), "TCE14": _avg(ce14(r) for r in inst if r["lived_case"] != "nom"),
        "n_forms_IDENT": len(tf[("IDENT", "informal")]), "n_forms_NONIDENT": len(tf[("NONIDENT", "informal")]),
        "PE_IDENT_pooled": _avg(tf[("IDENT", "formal")]), "IE_IDENT_pooled": _avg(tf[("IDENT", "informal")]),
        "PE_NONIDENT_pooled": _avg(tf[("NONIDENT", "formal")]),
        "IE_NONIDENT_pooled": _avg(tf[("NONIDENT", "informal")]),
    }
    for fam in LEAK_REQUIRED + LEAK_OPTIONAL:
        shares = [leak_share(r, fam) for r in recs]
        m[f"Leak_{fam}"] = _avg(s for s in shares if s is not None)
        adopt = [r["leak"][fam]["adopted"] / r["leak"][fam]["n"] for r in recs
                 if fam in r["leak"] and "adopted" in r["leak"][fam] and r["leak"][fam]["n"] > 0]
        m[f"Adopt_{fam}"] = _avg(adopt)
    m["OLD_residue"] = _avg(r["OLD"]["count"] / r["OLD"]["n"] for r in recs if r.get("OLD"))
    m["bpb_drift"] = _avg(r["bpb"]["edited"] - r["bpb"]["base"] for r in recs if r.get("bpb"))
    return m


EDIT_METRIC_COLUMNS = (
    "n_edits", "n_inflected", "n_h1_units", "excluded_lt2_targets", "WE", "WE_inflected", "PE6", "PE5", "IE5",
    "CE14", "TPE6", "TPE5", "TIE5", "TCE14", "n_forms_IDENT", "n_forms_NONIDENT", "PE_IDENT_pooled",
    "IE_IDENT_pooled", "PE_NONIDENT_pooled", "IE_NONIDENT_pooled",
    *[f"Leak_{f}" for f in LEAK_REQUIRED + LEAK_OPTIONAL], *[f"Adopt_{f}" for f in LEAK_REQUIRED + LEAK_OPTIONAL],
    "OLD_residue", "bpb_drift",
)


def edit_metric_rows(edits: list[dict], analysed_worlds: tuple[int, ...]) -> list[dict]:
    """Per (run, world, arm, layer, config[, tok_frequent]) metrics over ALL edit records (descriptive)."""
    groups: dict[tuple, list[dict]] = {}
    for r in edits:
        tf = r.get("tok_frequent")
        key = (r["run"], r["world"], r["arm"], -1 if r["layer"] is None else r["layer"], r["config"],
               "" if tf is None else str(bool(tf)).lower())
        groups.setdefault(key, []).append(r)
    rows = []
    for key in sorted(groups):
        run, world, arm, layer, config, tf = key
        row = {"run": run, "world": world, "arm": arm, "layer": None if layer < 0 else layer, "config": config,
               "tok_frequent": tf, "used_in_decisions": run == CONFIRM and world in analysed_worlds}
        row.update(_group_metrics(groups[key]))
        rows.append(row)
    return rows


def life_metric_rows(lives: list[dict], analysed_worlds: tuple[int, ...]) -> list[dict]:
    rows = []
    for r in sorted(lives, key=lambda x: (x["run"], x["world"], x["variant"], x["order"])):
        row = {"run": r["run"], "world": r["world"], "variant": r["variant"], "order": r["order"],
               "layer": r["layer"], "lam": r.get("lam"), "n_facts": len(r["facts"]),
               "E_end": _avg(f["correct"] / f["n"] for f in r["facts"]),
               "retraction_residue": r.get("retraction_residue"), "correction_residue": r.get("correction_residue"),
               "used_in_decisions": (r["run"] == CONFIRM and r["world"] in analysed_worlds
                                     and r["variant"] in ("J", "I"))}
        for fam in ("RN", "RD"):
            entry = (r.get("leak") or {}).get(fam)
            row[f"Leak_{fam}"] = entry["changed"] / entry["n"] if entry and entry["n"] else None
        rows.append(row)
    return rows


def tau_layer_rows(taus: list[dict], analysed_worlds: tuple[int, ...]) -> list[dict]:
    groups: dict[tuple, list[float]] = {}
    ents: dict[tuple, int] = {}
    for r in taus:
        key = (r["run"], r["world"], r["layer"])
        groups.setdefault(key, []).extend(r["tau_RN"])
        ents[key] = ents.get(key, 0) + (1 if r["tau_RN"] else 0)
    rows = []
    for key in sorted(groups):
        vals = groups[key]
        rows.append({"run": key[0], "world": key[1], "layer": key[2], "n_entities_with_RN": ents[key],
                     "n_pairs": len(vals), "median_tau_RN": float(np.median(vals)) if vals else None,
                     "mean_tau_RN": _avg(vals),
                     "used_in_decisions": key[0] == CONFIRM and key[1] in analysed_worlds and key[2] in S4_LAYERS})
    return rows


# --- rendering and writing ---------------------------------------------------------------------------------------
def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return "" if not math.isfinite(v) else f"{v:.6f}"
    return str(v)


def _csv(header: list[str], rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow([_fmt(r.get(h)) for h in header])
    return buf.getvalue()


def render_outputs(decisions: dict, data: Data) -> dict[str, str]:
    """Every deterministic output file as {path relative to the results dir: text}."""
    out = {"decisions.json": json.dumps(decisions, indent=1, sort_keys=True, ensure_ascii=False,
                                        allow_nan=False) + "\n"}
    for h in HYP_ORDER:
        pw = decisions["hypotheses"][h]["per_world"]
        comp_keys = sorted({k for v in pw.values() for k in v["components"]})
        rows = []
        for w in sorted(pw, key=int):
            row = {"world": int(w), "n_units": pw[w]["n_units"], "qualifies": pw[w]["qualifies"], "d_w": pw[w]["d"]}
            row.update(pw[w]["components"])
            rows.append(row)
        out[f"tables/per_world_{h}.csv"] = _csv(["world", "n_units", "qualifies", "d_w", *comp_keys], rows)
    ws = tuple(decisions["options"]["worlds"])
    out["tables/edit_metrics.csv"] = _csv(
        ["run", "world", "arm", "layer", "config", "tok_frequent", "used_in_decisions", *EDIT_METRIC_COLUMNS],
        edit_metric_rows(data.edits, ws))
    out["tables/life_metrics.csv"] = _csv(
        ["run", "world", "variant", "order", "layer", "lam", "n_facts", "E_end", "retraction_residue",
         "correction_residue", "Leak_RN", "Leak_RD", "used_in_decisions"], life_metric_rows(data.lives, ws))
    out["tables/tau_by_layer.csv"] = _csv(
        ["run", "world", "layer", "n_entities_with_RN", "n_pairs", "median_tau_RN", "mean_tau_RN",
         "used_in_decisions"], tau_layer_rows(data.taus, ws))
    summary_rows = []
    for h in HYP_ORDER:
        e = decisions["hypotheses"][h]
        summary_rows.append({"hypothesis": h, "family": e["family"], "margin": e["margin"],
                             "n_qualifying_worlds": e["n_qualifying_worlds"], "mean_d": e["mean_d"],
                             "t_lo": e["t_interval_95"][0], "t_hi": e["t_interval_95"][1],
                             "p_signflip": e["p_signflip"], "holm_rejected": e["holm_rejected"],
                             "boot_lo": (e["bootstrap_95"] or [None, None])[0],
                             "boot_hi": (e["bootstrap_95"] or [None, None])[1], "decision": e["decision"]})
    out["tables/decisions_summary.csv"] = _csv(
        ["hypothesis", "family", "margin", "n_qualifying_worlds", "mean_d", "t_lo", "t_hi", "p_signflip",
         "holm_rejected", "boot_lo", "boot_hi", "decision"], summary_rows)
    return out


# Figure colours: the dataviz reference palette (light surface). One series per panel -> no legend box; the panel
# title names the hypothesis; the per-world numbers are in tables/per_world_<H>.csv (table view).
_FIG = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "axis": "#b5b4ae", "zero": "#dddcd6",
        "series": "#2a78d6"}


def write_figures(decisions: dict, results_dir: pathlib.Path) -> list[str]:
    """Small multiples of the per-world effects d_w (one panel and one y-scale per hypothesis), with the margin m
    as a labelled reference line and mean +- two-sided 95% t-interval. Only runs when matplotlib is installed;
    a plotting failure propagates (decisions.json and the tables are written before)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(HYP_ORDER), figsize=(2.3 * len(HYP_ORDER), 3.0), facecolor=_FIG["surface"])
    for ax, h in zip(axes, HYP_ORDER):
        e = decisions["hypotheses"][h]
        ws = e["qualifying_worlds"]
        ax.set_facecolor(_FIG["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_FIG["axis"])
            ax.spines[side].set_linewidth(0.8)
        ax.axhline(0.0, color=_FIG["zero"], linewidth=0.8, zorder=1)
        ax.axhline(e["margin"], color=_FIG["ink2"], linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
        ax.annotate(f"m = {e['margin']:g}", xy=(1.0, e["margin"]), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", ha="right", va="bottom", fontsize=6,
                    color=_FIG["ink2"])
        if ws:
            ax.plot(range(len(ws)), e["d_qualifying"], linestyle="none", marker="o", markersize=6,
                    color=_FIG["series"], markeredgecolor=_FIG["surface"], markeredgewidth=1.5, zorder=3)
            ticks, labels = list(range(len(ws))), [str(w) for w in ws]
            lo, hi = e["t_interval_95"]
            if e["mean_d"] is not None and lo is not None and hi is not None:
                x_mean = len(ws) + 0.6
                ax.errorbar([x_mean], [e["mean_d"]], yerr=[[e["mean_d"] - lo], [hi - e["mean_d"]]], fmt="s",
                            color=_FIG["ink"], ecolor=_FIG["ink"], elinewidth=1.5, capsize=3, markersize=5,
                            zorder=3)
                ticks.append(x_mean)
                labels.append("mean")
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, fontsize=5, rotation=90, color=_FIG["ink2"])
        else:
            ax.set_xticks([])
            ax.text(0.5, 0.5, "no qualifying worlds", transform=ax.transAxes, ha="center", va="center",
                    fontsize=6, color=_FIG["ink2"])
        ax.set_title(f"{h}\n{e['decision']}", fontsize=7, color=_FIG["ink"])
        ax.tick_params(axis="both", colors=_FIG["axis"], labelcolor=_FIG["ink2"], labelsize=6, length=2)
    fig.supylabel("per-world effect d_w", fontsize=7, color=_FIG["ink2"])
    fig.tight_layout()
    path = fig_dir / "per_world_effects.png"
    fig.savefig(path, dpi=150, facecolor=_FIG["surface"], metadata={"Software": None})
    plt.close(fig)
    return [path.relative_to(results_dir).as_posix()]


def environment_info() -> dict:
    try:
        import matplotlib
        mpl = matplotlib.__version__
    except ImportError:
        mpl = None
    return {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
            "matplotlib": mpl}


def run_analysis(results_dir: pathlib.Path | str | None = None, *, worlds: Iterable[int] | None = None,
                 bootstrap_n: int = S.BOOTSTRAP_N, perm_n: int = PERM_N, seed: int = 0, map_only: bool = False,
                 precondition_failed: Iterable[str] = (), write: bool = True, figures: bool = True,
                 log: Callable[[str], None] = print) -> dict:
    """Load, validate, decide; optionally write decisions.json, tables, analysis_env.json and figures."""
    results_dir = pathlib.Path(results_dir) if results_dir is not None else STUDY / "results"
    data = load_results(results_dir)
    decisions = compute_decisions(data, worlds=worlds, bootstrap_n=bootstrap_n, perm_n=perm_n, seed=seed,
                                  map_only=map_only, precondition_failed=precondition_failed)
    if not write:
        return decisions
    for rel, text in render_outputs(decisions, data).items():
        path = results_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    env = environment_info()
    if figures and env["matplotlib"] is not None:
        env["figures"] = write_figures(decisions, results_dir)
    else:
        env["figures"] = []
        env["figures_note"] = "matplotlib not installed; figures skipped" if figures else "disabled (--no-figures)"
        log(f"figures: {env['figures_note']}")
    (results_dir / "analysis_env.json").write_text(json.dumps(env, indent=1, sort_keys=True) + "\n",
                                                   encoding="utf-8")
    for h in HYP_ORDER:
        e = decisions["hypotheses"][h]
        md = "n/a" if e["mean_d"] is None else f"{e['mean_d']:+.4f}"
        p = "n/a" if e["p_signflip"] is None else f"{e['p_signflip']:.5f}"
        log(f"{h}: {e['decision']:<52s} worlds={e['n_qualifying_worlds']} mean_d={md} p={p} m={e['margin']}")
    return decisions


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=pathlib.Path, default=STUDY / "results")
    ap.add_argument("--worlds", type=int, nargs="+", default=None,
                    help="analysed confirmatory worlds (default 1000-1007; reserves replace failed bases)")
    ap.add_argument("--bootstrap", type=int, default=S.BOOTSTRAP_N)
    ap.add_argument("--perm", type=int, default=PERM_N)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map-only", action="store_true", help="STD best discovery WE < 50%%: H1-H3 not testable")
    ap.add_argument("--precondition-failed", nargs="+", default=[], metavar="H",
                    help="hypotheses whose preconditions failed with reserves exhausted")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    run_analysis(args.results, worlds=args.worlds, bootstrap_n=args.bootstrap, perm_n=args.perm, seed=args.seed,
                 map_only=args.map_only, precondition_failed=args.precondition_failed, figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
