# -*- coding: utf-8 -*-
"""Paths, imports of the frozen CEV-11 engine and of the forward-learning engine, records, run configuration."""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import sys
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

STUDY = pathlib.Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
CEV11 = REPO / "research" / "living_learning_paradigm_2026_11_13"
ENGINE_FILE = REPO / "src" / "neural_network_module" / "ortak_katman_module" / "forward_learning.py"

for _p in (str(REPO), str(CEV11)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DISCOVERY_WORLDS: Tuple[int, ...] = (0, 1)
CONFIRMATORY_WORLDS: Tuple[int, ...] = tuple(range(1000, 1008))
RECORD_SCHEMA = "fwl.record.v1"
KINDS = ("meta", "single", "stream", "gate", "capacity", "retract", "side")

# ----------------------------------------------------------------------------------------------- protocol constants
SINGLE_ONE_STEP = ("FEC-TOP", "FEC-0", "FEC-1", "FEC-A", "HEBB-TOP", "HEBB-0", "RAND-0", "BP-1", "BP-1b", "BP-pred")
SINGLE_FT = ("FT-L", "FT-ALL")
INSTALLS = ("nll", "greedy")
FAMILIES = ("WF", "PF6", "PF5", "IF5", "RN", "RD", "RB", "RC")
STREAM_ARMS = ("FEC-TOP+RLS", "FEC-A+RLS", "FEC-TOP", "FEC-TOP+RLS+G", "HEBB-TOP+RLS", "HEBB-0+RLS", "BP-1+RLS",
               "FT-L-seq", "RLSEdit")
FEC_STREAM = ("FEC-TOP+RLS", "FEC-A+RLS")
# stream arms whose step is the Frobenius norm of a unit-normalised write (they share one step cap)
NORMALISED_STREAM_ARMS = ("FEC-TOP+RLS", "FEC-A+RLS", "FEC-TOP", "FEC-TOP+RLS+G", "HEBB-TOP+RLS", "HEBB-0+RLS",
                          "BP-1+RLS")
COMPARATORS = ("FT-L-seq", "BP-1+RLS", "RLSEdit", "HEBB-TOP+RLS", "HEBB-0+RLS")
GATE_ARMS = ("FEC-TOP+RLS", "FEC-A+RLS")
THETAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
GAINS = (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1)
# CEV-11's frozen discovery selection for the fine-tuning comparators and the ROOT site (results/selection.json)
FT_L = {"layer": 0, "hp": {"eps": 0.02, "lr": 0.001, "steps": 60}, "config": "eps=0.02,lr=0.001,steps=60"}
FT_ALL = {"layer": None, "hp": {"lr": 0.0001, "steps": 40}, "config": "lr=0.0001,steps=40"}
ROOT_SITE = {"layer": 0, "hp": {"cap_mult": 8, "step": 0.5, "max_steps": 120}, "config": "cap_mult=8,step=0.5"}
PILOT_SINGLE_GRID = (0.0,) + tuple(0.25 * 2.0 ** (i / 2.0) for i in range(20))
PILOT_STREAM_GRID = (0.0,) + tuple(2.0 ** (i / 2.0 - 4.0) for i in range(28))


@dataclass(frozen=True)
class RunConfig:
    """Everything a world run depends on besides the base model. ``protocol_config`` is the registered setting;
    ``smoke_config`` is a reduced setting that mirrors the v1 pilots and is not protocol data."""
    label: str = "protocol"
    # Part A
    per_case: int = 10                 # CEV-11's confirmatory item list (edit_items(..., per_case=10))
    edits_per_case: int = 6            # the first 6 per lived case in list order: 42 edits per world
    max_edits: Optional[int] = None
    single_arms: Tuple[str, ...] = SINGLE_ONE_STEP + SINGLE_FT
    installs: Tuple[str, ...] = INSTALLS
    single_grid: Optional[Tuple[float, ...]] = None       # None = the engine's default grid
    bpb_edits: int = 12
    bpb_arms: Tuple[str, ...] = ("FEC-A", "FEC-TOP", "FT-L")
    bpb_texts: int = 40
    so_entities: int = 6
    # Part B
    streams: int = 2
    stream_n: int = 100
    checkpoints: Tuple[int, ...] = (10, 30, 100)
    stream_arms: Tuple[str, ...] = STREAM_ARMS
    stream_install: str = "nll"
    stream_grid: Optional[Tuple[float, ...]] = None
    arm_limits: Dict[str, int] = field(default_factory=dict)   # smoke only: stop an arm after this many facts
    untouched: int = 120
    n0: float = 50.0
    xi: float = 1.0
    c0_tokens: int = 12000
    guard_rms_cap: float = 0.25
    # gate, capacity, retraction (stream 0 only)
    gate_arms: Tuple[str, ...] = GATE_ARMS
    thetas: Tuple[float, ...] = THETAS
    gains: Tuple[float, ...] = GAINS
    gate_bpb: bool = True
    hard_negatives: int = 240          # per type, at most
    leak_tv: float = 0.10
    retract_indices: Tuple[int, ...] = (5, 20, 35, 50, 65, 80, 95)
    capacity_thetas: Tuple[float, ...] = (0.3, 0.5, 0.7)
    # caps from the discovery selection: {"single": {"ARM|install": mu}, "stream": {"ARM": mu}}
    caps: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_json(self) -> dict:
        return json.loads(json.dumps(asdict(self)))


def protocol_config(selection: Optional[dict] = None) -> RunConfig:
    cfg = RunConfig()
    if selection:
        cfg = replace(cfg, caps={"single": dict(selection.get("caps", {}).get("single", {})),
                                 "stream": dict(selection.get("caps", {}).get("stream", {}))})
    return cfg


def smoke_config() -> RunConfig:
    """Reduced run for a pipeline proof and for a comparison with the v1 pilots (same items, grids and install
    criterion as ``fec_pilot_cev11.py`` and ``fec_stream_pilot2.py``). Not protocol data."""
    return RunConfig(
        label="smoke", per_case=3, edits_per_case=10 ** 6, max_edits=6,
        single_arms=("FEC-TOP", "FEC-0", "FEC-1", "FEC-A", "HEBB-0", "RAND-0", "BP-1", "BP-pred", "FT-L"),
        installs=("greedy",), single_grid=PILOT_SINGLE_GRID, bpb_edits=2, bpb_arms=("FEC-A",), bpb_texts=40,
        streams=1, stream_n=30, checkpoints=(10, 30),
        stream_arms=("FEC-TOP+RLS", "FEC-A+RLS", "FEC-TOP", "FEC-TOP+RLS+G", "HEBB-TOP+RLS", "BP-1+RLS",
                     "FT-L-seq", "RLSEdit"),
        stream_install="greedy", stream_grid=PILOT_STREAM_GRID,
        arm_limits={"HEBB-TOP+RLS": 10, "BP-1+RLS": 10, "FT-L-seq": 10, "RLSEdit": 6, "FEC-TOP+RLS+G": 10},
        gate_arms=("FEC-TOP+RLS",), thetas=(0.0, 0.3, 0.6, 0.9), gains=(1.0, 0.5, 0.25), gate_bpb=False,
        hard_negatives=60, retract_indices=(5, 20), c0_tokens=12000)


# -------------------------------------------------------------------------------------------------------- imports
def engine():
    """The forward-learning engine module (src/neural_network_module/ortak_katman_module/forward_learning.py)."""
    from src.neural_network_module.ortak_katman_module import forward_learning
    return forward_learning


def bases_dir() -> pathlib.Path:
    """Folder that holds the CEV-11 bases: $CEV_BASES, else the CEV-11 results folder of this checkout."""
    env = os.environ.get("CEV_BASES")
    return pathlib.Path(env) if env else CEV11 / "results" / "bases"


def base_name(world: int) -> str:
    if world in DISCOVERY_WORLDS:
        return f"discovery_w{world}_K20"
    if world in CONFIRMATORY_WORLDS:
        return f"confirm_w{world}_K20"
    raise ValueError(f"world {world} is neither a discovery world {DISCOVERY_WORLDS} nor a confirmatory world "
                     f"{CONFIRMATORY_WORLDS[0]}..{CONFIRMATORY_WORLDS[-1]}")


def load_base(world: int, base_dir: Optional[pathlib.Path] = None):
    """A ``cev11.experiment.Base`` for ``world`` plus its editable entity list; refuses a base whose gates failed."""
    from cev11 import experiment as E
    path = pathlib.Path(base_dir) if base_dir else bases_dir() / base_name(world)
    if not (path / "model.pt").exists():
        raise FileNotFoundError(f"{path / 'model.pt'} not found: set CEV_BASES to the folder that holds the "
                                f"CEV-11 bases (the weights are not in the repository)")
    base = E.load_base(path, REPO)
    check_base_world(base, world, path)
    gates = json.loads((path / "gates.json").read_text(encoding="utf-8"))
    if not gates.get("all_pass", False):
        raise RuntimeError(f"{path.name}: the CEV-11 base gates did not pass")
    editable = gates.get("editable") or gates.get("G7", {}).get("editable")
    if not editable:
        raise RuntimeError(f"{path.name}: no editable entity list in gates.json")
    return base, sorted(editable)


def check_base_world(base, world: int, where="base") -> None:
    """A base belongs to exactly one world (the seed of its world file). Without this check ``--base`` could put
    a confirmatory base under a discovery or smoke run, which would break the blinding."""
    seed = getattr(base.world, "seed", None)
    if seed != world:
        raise RuntimeError(f"{where}: this base was pre-trained for world {seed!r}, not for world {world}; "
                           f"a run may only use the base of its own world")


# -------------------------------------------------------------------------------------------------------- records
def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _finite_tree(v: Any, where: str) -> None:
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(f"{where}: non-finite number")
    elif isinstance(v, dict):
        for k, x in v.items():
            if not isinstance(k, str):
                raise ValueError(f"{where}: non-string key {k!r}")
            _finite_tree(x, f"{where}.{k}")
    elif isinstance(v, (list, tuple)):
        for i, x in enumerate(v):
            _finite_tree(x, f"{where}[{i}]")
    elif not (v is None or isinstance(v, (str, int, bool))):
        raise ValueError(f"{where}: unsupported value {type(v).__name__}")


_REQUIRED = {
    "meta": ("config", "config_sha256", "base_sha256", "engine_sha256", "items_sha256"),
    "single": ("edit_id", "arm", "install", "lived_case", "installed", "PF6", "targets", "leak", "seconds"),
    "stream": ("stream", "arm", "n", "written", "installed", "retention", "PF6", "IF", "dmg_A", "dmg_B",
               "net_per_write", "composite", "seconds"),
    "gate": ("stream", "arm", "setting", "theta", "gain", "retention", "recall_other", "leak_hard", "leak_generic"),
    "capacity": ("stream", "arm", "block", "trace", "predicted", "knees"),
    "retract": ("stream", "arm", "index", "write_id", "p_before", "kl_replay_retract", "others_retract",
                "others_replay"),
    "side": ("stream", "arm", "retracted"),      # written last: gate, capacity and retraction records are complete
}


def make_record(kind: str, run: str, world: int, key: str, **fields) -> dict:
    rec = {"schema": RECORD_SCHEMA, "kind": kind, "run": run, "world": int(world), "key": key}
    rec.update(fields)
    return validate_record(rec)


def validate_record(rec: dict, where: str = "record") -> dict:
    if not isinstance(rec, dict) or rec.get("schema") != RECORD_SCHEMA:
        raise ValueError(f"{where}: not a {RECORD_SCHEMA} record")
    kind = rec.get("kind")
    if kind not in KINDS:
        raise ValueError(f"{where}: unknown kind {kind!r}")
    if rec.get("run") not in ("discovery", "confirm", "smoke"):
        raise ValueError(f"{where}: unknown run {rec.get('run')!r}")
    if not isinstance(rec.get("world"), int) or isinstance(rec.get("world"), bool):
        raise ValueError(f"{where}: world must be an int")
    if not isinstance(rec.get("key"), str) or not rec["key"]:
        raise ValueError(f"{where}: missing key")
    missing = [f for f in _REQUIRED[kind] if f not in rec]
    if missing:
        raise ValueError(f"{where} ({kind} {rec['key']}): missing fields {missing}")
    _finite_tree(rec, where)
    if kind == "single":
        if rec["install"] not in INSTALLS + ("ft",):
            raise ValueError(f"{where}: unknown install {rec['install']!r}")
        if not isinstance(rec["installed"], bool):
            raise ValueError(f"{where}: installed must be a bool")
        for fam, entry in rec["leak"].items():
            if not (isinstance(entry, dict) and 0 <= entry.get("changed", -1) <= entry.get("n", -1)):
                raise ValueError(f"{where}: bad leak entry {fam}")
    if kind == "stream":
        for f in ("retention", "PF6", "IF", "dmg_A", "dmg_B"):
            if not 0.0 <= rec[f] <= 1.0:
                raise ValueError(f"{where}: {f} outside [0, 1]")
    return rec


def read_records(path: pathlib.Path) -> List[dict]:
    out = []
    if not pathlib.Path(path).exists():
        return out
    for i, line in enumerate(pathlib.Path(path).read_text(encoding="utf-8").split("\n")):
        if line.strip():
            out.append(validate_record(json.loads(line), f"{pathlib.Path(path).name}:{i + 1}"))
    return out


def repair_tail(path: pathlib.Path) -> Optional[str]:
    """A process killed in the middle of a write leaves a last line without its newline. If that line is a
    complete record the newline is added; otherwise the fragment is cut off (its record will be made again).
    Returns what was done ("completed" / "dropped") or None when the file ends cleanly. Lines that end with a
    newline are never touched: a damaged line in the middle of a file stays an error."""
    path = pathlib.Path(path)
    if not path.exists():
        return None
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return None
    cut = data.rfind(b"\n") + 1
    try:
        validate_record(json.loads(data[cut:].decode("utf-8")))
    except (ValueError, UnicodeDecodeError):
        with path.open("r+b") as f:
            f.truncate(cut)
        return "dropped"
    with path.open("ab") as f:
        f.write(b"\n")
    return "completed"


class RecordWriter:
    """Append-only JSONL with resume: a key that is already in the file is not written again. A last line that
    a killed process left unfinished is repaired first (``repair_tail``)."""

    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.repaired = repair_tail(self.path)
        self.done = {r["key"] for r in read_records(self.path)}

    def has(self, key: str) -> bool:
        return key in self.done

    def write(self, rec: dict) -> None:
        validate_record(rec)
        if rec["key"] in self.done:
            raise ValueError(f"duplicate record key {rec['key']!r}")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.done.add(rec["key"])


def results_path(out_dir: pathlib.Path, world: int) -> pathlib.Path:
    return pathlib.Path(out_dir) / f"w{world}.jsonl"


def mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else None


def median(values: Sequence[float]) -> Optional[float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
