"""Fact providers of the compatibility gate, and the inspection report built from them.

``inspect_checkpoint`` answers, for any checkpoint file: is it intact, which tokenizer and
architecture does it hold, is it compatible with the current code (and with a given target),
and if not, why. Levels:

  file     torch-free: file key, sidecar, safety sniff of the pickle
  weights  loads with ``weights_only=True``: envelope, keys of weights, architecture, tokenizer ...
  compat   also derives the layout the *current code* builds for the declared architecture and
           compares against the target (model config, tokenizer, optimizer, scheduler) if given
"""
from dataclasses import dataclass, field
from pathlib import Path

from cevahir_ai.domain.verdicts import CheckResult
from cevahir_ai.keys.gate import check_compatibility
from cevahir_ai.keys.grammar import Key
from cevahir_ai.keys.schemes.file import file_bytes_key, read_sidecar

LEVELS = ("file", "weights", "compat")
FLOAT_DTYPES = ("float32", "float16", "bfloat16", "float64")


@dataclass
class Target:
    """What the checkpoint is to be loaded into. Every field is optional."""
    config: dict = None            # architecture config of the target model
    model: object = None           # or the live model itself
    tokenizer: object = None       # live TokenizerCore or ByteLevelBPE
    optimizer: object = None
    scheduler: object = None
    dtype: str = "float32"
    max_seq_length: int = None     # serving context length
    allow_cast: bool = False       # policy: allow up-casting stored weights to ``dtype``
    allow_migrations: tuple = ()   # ids of function-preserving migrations that may be applied in memory


@dataclass
class Report:
    path: str
    level: str
    purpose: str
    size: int = 0
    envelope: str = None
    keys: dict = field(default_factory=dict)
    aliases: list = field(default_factory=list)
    explain: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    verdict: object = None
    table: list = None             # per-tensor table (weights level)
    state: dict = None             # loaded tensors (not serialised)
    view: object = None

    @property
    def intact(self):
        """"yes" when a digest was verified, "no" when one failed, else "unknown"."""
        statuses = [self.facts[c].status for c in ("C1", "C3") if c in self.facts]
        if "fail" in statuses:
            return "no"
        return "yes" if "pass" in statuses else "unknown"

    def to_dict(self, tensors=False):
        data = {"path": self.path, "size": self.size, "level": self.level, "envelope": self.envelope,
                "intact": self.intact, "keys": {name: str(key) for name, key in self.keys.items()},
                "aliases": [str(key) for key in self.aliases], "explain": self.explain, "notes": list(self.notes),
                "verdict": self.verdict.to_dict() if self.verdict else None}
        if tensors and self.table is not None:
            data["tensors"] = self.table
        return data


def _fact(report, check_id, status, detail, *evidence):
    report.facts[check_id] = CheckResult(check_id, status, detail, tuple(str(item) for item in evidence if item))


def _as_hex(value):
    text = str(value)
    return text.rsplit(":", 1)[-1] if text.startswith("cev:") else text.lower()


def _file_facts(report, path, expect_file):
    from cevahir_ai.adapters.envelopes import sniff_pickle_globals
    file_key = file_bytes_key(path)
    report.keys["file"] = file_key
    sidecar = read_sidecar(path)
    if expect_file is not None:
        ok = _as_hex(expect_file) == file_key.digest
        _fact(report, "C1", "pass" if ok else "fail",
              "file bytes equal the expected key" if ok else "file bytes differ from the expected key "
              "(tampered, truncated or another file)", file_key)
    elif sidecar is not None:
        ok = sidecar == file_key.digest
        _fact(report, "C1", "pass" if ok else "fail",
              "file bytes equal the .sha256 sidecar" if ok else "file bytes differ from the .sha256 sidecar "
              "(tampered or truncated file, or a stale sidecar)", file_key)
    else:
        _fact(report, "C1", "unknown", "no .sha256 sidecar and no expected file key: integrity cannot be established",
              file_key)
    sniff = sniff_pickle_globals(path)
    if sniff["container"] != "torch-zip":
        _fact(report, "C12", "unsupported", "not a zip-format torch checkpoint")
    elif sniff["unsafe"]:
        report.envelope = "module"
        _fact(report, "C12", "fail", "loading needs code execution (pickled module or arbitrary object); "
              "globals outside the weights-only safe set: " + ", ".join(sniff["unsafe"][:6]))
    else:
        _fact(report, "C12", "pass", f"weights-only loadable ({len(sniff['globals'])} pickle globals, all in the safe set)")


def _arch_facts(report, view):
    """Declared architecture -> keys. Returns (arch key, fields, normalised) or (None, None, None)."""
    from cevahir_ai.adapters.model_build import arch_of_config
    from cevahir_ai.keys.schemes.arch import arch_ctor_v0_key
    if not view.model_config:
        return None, None, None
    try:
        key, fields, normalized = arch_of_config(view.model_config)
    except Exception as error:
        report.notes.append(f"stored architecture config ({view.config_source}) is rejected by the current "
                            f"normaliser: {type(error).__name__}: {error}")
        return None, None, None
    report.keys["arch"] = key
    report.explain["arch"] = fields
    report.explain["arch_source"] = view.config_source
    report.explain["family_version"] = normalized["architecture_version"]
    report.explain["trained_max_seq_length"] = normalized.get("max_seq_length")
    if view.config_source in ("model_config", "config", "additional_info.config"):
        try:
            report.aliases.append(arch_ctor_v0_key(view.model_config))
        except Exception:
            pass
    return key, fields, normalized


def _tokenizer_facts(report, view, path):
    """Tokenizer identity carried by the artefact. Returns the artefact's tokenizer key or None."""
    from cevahir_ai.adapters.tokenizers import lookup_legacy
    if view.tokenizer is None:
        report.explain["tokenizer"] = {"identity": None}
        return None
    scheme, digest = view.tokenizer
    key = Key("tok", scheme, digest)
    report.keys["tok"] = key
    info = {"scheme": scheme}
    if scheme == "legacy-v3":
        tracked = lookup_legacy(digest)
        if tracked:
            info.update(tracked=tracked["name"], canonical=tracked["key"], how=tracked["how"])
            report.keys["tok_canonical"] = Key.parse(tracked["key"])
    else:
        info["canonical"] = str(key)
        report.keys["tok_canonical"] = key
        if view.tokenizer_file:
            info["file"] = view.tokenizer_file
    report.explain["tokenizer"] = info
    return key


def _weights_facts(report, view, expect_weights):
    from cevahir_ai.errors import UnsupportedTensor
    from cevahir_ai.keys.torchkeys.tensors import layout_of_table
    from cevahir_ai.keys.torchkeys.weights import dtype_summary, weights_cev11_fp32_digest, weights_key
    from cevahir_ai.keys.torchkeys.wschema import layout_keys
    try:
        key, table = weights_key(view.state)
    except UnsupportedTensor as error:
        _fact(report, "C9", "unsupported", str(error))
        _fact(report, "C3", "unknown", "weights are outside every weight scheme")
        return None
    report.keys["weights"], report.table = key, table
    layout = layout_of_table(table)
    report.keys["wschema"], report.keys["wschema_sd"] = layout_keys(layout)
    dtypes, n_params = dtype_summary(table)
    report.explain["weights"] = {"tensors": len(table), "parameters": n_params, "dtypes": dtypes,
                                 "aliases": layout["aliases"]}
    declared = view.declared.get("weights:cev11-fp32-v1")
    if declared:
        legacy = weights_cev11_fp32_digest(view.state)
        alias = Key("weights", "cev11-fp32-v1", legacy)
        report.aliases.append(alias)
        ok = legacy == declared
        _fact(report, "C3", "pass" if ok else "fail",
              "recomputed weight digest equals the declared CEV-11 model_sha256" if ok else
              f"weights differ from the declared CEV-11 model_sha256 ({declared[:12]}…)", key, alias)
    elif expect_weights is not None:
        ok = _as_hex(expect_weights) == key.digest
        _fact(report, "C3", "pass" if ok else "fail",
              "weights equal the expected key" if ok else "weights differ from the expected key", key)
    else:
        _fact(report, "C3", "unknown", "the envelope declares no weight digest and none was supplied", key)
    return layout


def _dtype_fact(report, target):
    dtypes = report.explain.get("weights", {}).get("dtypes", [])
    wanted = target.dtype
    other = [name for name in dtypes if name != wanted]
    if not other:
        _fact(report, "C9", "pass", f"floating tensors are {wanted}")
    elif target.allow_cast and all(_rank(name) <= _rank(wanted) for name in other):
        _fact(report, "C9", "pass", f"stored {', '.join(other)}; up-cast to {wanted} allowed by policy")
    else:
        _fact(report, "C9", "fail", f"stored {', '.join(other)}, target {wanted}: loading would cast silently "
              "(casts are not allowed by the built-in policy)")


def _rank(name):
    return {"float16": 1, "bfloat16": 1, "float32": 2, "float64": 3}.get(name, 9)


def _signature_facts(report, view):
    from cevahir_ai.keys.schemes.ckpt import optim_sig_key, sched_sig_key
    from cevahir_ai.keys.torchkeys.optim import optim_content_key, sched_content_key, tstate_content_key
    if isinstance(view.optimizer_state, dict) and "param_groups" in view.optimizer_state:
        report.keys["optim_sig"], sig = optim_sig_key(view.optimizer_state)
        report.keys["optim"] = optim_content_key(view.optimizer_state)
        report.explain["optimizer"] = sig
    stored = view.scheduler_state
    if isinstance(stored, dict) and stored:
        wrapper = "TrainingScheduler" if "scheduler_type" in stored and "scheduler_state" in stored else None
        inner = stored.get("scheduler_state") or {} if wrapper else stored
        report.keys["sched_sig"], sig = sched_sig_key(inner, stored.get("scheduler_type") if wrapper else None, wrapper)
        report.keys["sched"] = sched_content_key(stored)
        report.explain["scheduler"] = sig
    if view.extra_state:
        rest = {name: value for name, value in view.extra_state.items() if name not in ("scheduler", "tokenizer_identity")}
        report.keys["tstate"] = tstate_content_key(rest)


def _member_keys(report):
    from cevahir_ai.keys.schemes.ckpt import ckpt_members_key, model_triple_key
    keys = report.keys
    if "weights" not in keys:
        return
    members = {"weights": keys["weights"], "wschema": keys["wschema"], "arch": keys.get("arch"),
               "tok": keys.get("tok_canonical") or keys.get("tok"), "optim": keys.get("optim"),
               "sched": keys.get("sched"), "tstate": keys.get("tstate")}
    keys["ckpt"] = ckpt_members_key(members)
    if keys.get("arch") and keys.get("tok_canonical"):
        keys["model"] = model_triple_key(keys["weights"], keys["arch"], keys["tok_canonical"])
    else:
        report.notes.append("no model key: a servable identity needs a known architecture and a canonical tokenizer key")


def _layout_fact(report, view, instance, target, target_arch):
    """C4: the stored layout against the layout the current code builds."""
    from cevahir_ai.adapters.model_build import expected_layout
    from cevahir_ai.keys import schemas
    from cevahir_ai.keys.torchkeys.migrate import apply_migration
    from cevahir_ai.keys.torchkeys.wschema import broken_ties, compare_layouts, layout_keys, layout_of_module, layout_of_state
    source = None
    try:
        if "arch" in report.keys:
            expected, source = expected_layout(view.model_config), "the declared architecture"
        elif target.model is not None:
            expected, source = layout_of_module(target.model), "the target model (no declared architecture)"
        elif target.config is not None:
            expected, source = expected_layout(target.config), "the target architecture (no declared architecture)"
        else:
            _fact(report, "C4", "unknown", "no declared architecture and no target: the expected layout is unknown")
            return
    except Exception as error:
        _fact(report, "C4", "fail", f"the current code cannot build the architecture: {type(error).__name__}: {error}")
        return
    expected_key = layout_keys(expected)[0]
    report.explain["expected_layout"] = {"key": str(expected_key), "source": source,
                                         "preset": schemas.preset_of_layout_key(expected_key)}
    findings = compare_layouts(instance, expected)
    if findings:
        for migration_id in schemas.migration_ids():
            migrated, applied = apply_migration(view.state, schemas.load_migration(migration_id))
            if applied and not compare_layouts(layout_of_state(migrated), expected):
                if migration_id in target.allow_migrations:
                    view.state = migrated
                    report.notes.append(f"migration {migration_id} applied in memory")
                    _fact(report, "C4", "pass", f"layout matches {source} after migration {migration_id}", expected_key)
                    return
                findings.append(f"the layout matches after migration {migration_id}: convert the file or allow it")
                break
        _fact(report, "C4", "fail", f"layout differs from {source}: " + "; ".join(findings), expected_key)
        return
    ties = broken_ties(view.state, expected)
    if ties:
        _fact(report, "C4", "fail", "tie broken in file: " + ", ".join(f"{a} != {b}" for a, b in ties), expected_key)
        return
    _fact(report, "C4", "pass", f"names, shapes and ties match {source}", expected_key)


def _arch_fact(report, view, fields, target_arch):
    from cevahir_ai.keys.schemes.arch import diff_fields
    if "arch" not in report.keys:
        _fact(report, "C5", "unknown", f"architecture unknown (envelope {report.envelope} declares none): "
              "identical shapes do not prove an identical function")
        return
    if target_arch is None:
        _fact(report, "C5", "pass", "declared architecture is accepted by the current code "
              f"({report.explain['family_version']}); no target model given", report.keys["arch"])
        return
    target_key, target_fields = target_arch
    if target_key == report.keys["arch"]:
        _fact(report, "C5", "pass", "architecture key equals the target", target_key)
        return
    differences = "; ".join(f"{name}: checkpoint={left!r}, target={right!r}"
                            for name, (left, right) in diff_fields(fields, target_fields).items())
    _fact(report, "C5", "fail", f"architecture differs from the target: {differences}", report.keys["arch"], target_key)


def _tokenizer_fact(report, view, path, target):
    from cevahir_ai.adapters import tokenizers
    from cevahir_ai.keys.schemes.tok import tok_bytebpe_key
    if view.tokenizer is None:
        _fact(report, "C7", "unknown", "the checkpoint carries no tokenizer identity: token semantics cannot be verified")
        return
    scheme, digest = view.tokenizer
    key = report.keys["tok"]
    live = target.tokenizer
    if live is not None:
        is_byte = hasattr(live, "identity") and callable(live.identity)
        if scheme == "bytebpe-v1":
            ok = is_byte and live.identity() == digest
            _fact(report, "C7", "pass" if ok else "fail", "byte-BPE identity equals the target tokenizer" if ok else
                  "tokenizer differs from the target (another byte-BPE, or another tokenizer family)", key)
        elif is_byte:
            _fact(report, "C7", "fail", "the checkpoint was trained with the legacy BPE, the target is a byte-BPE", key)
        else:
            status, detail = tokenizers.reconcile(digest, live)
            canonical, _ = tokenizers.canonical_key(live)
            if status != "mismatch":
                report.keys["tok_canonical"] = canonical
                report.explain["tokenizer"]["canonical"] = str(canonical)
            _fact(report, "C7", "fail" if status == "mismatch" else "pass", detail, key, canonical)
        return
    if scheme == "bytebpe-v1":
        sibling = Path(path).parent / view.tokenizer_file if view.tokenizer_file else None
        if sibling is None or not sibling.is_file():
            _fact(report, "C7", "unknown", "tokenizer file next to the checkpoint is missing; supply the tokenizer", key)
            return
        try:
            ok = tok_bytebpe_key(sibling) == key
        except ValueError as error:
            _fact(report, "C7", "fail", str(error), key)
            return
        _fact(report, "C7", "pass" if ok else "fail",
              f"tokenizer file {sibling.name} has the declared identity" if ok else
              f"tokenizer file {sibling.name} is not the tokenizer the model was trained with", key)
        return
    tracked = report.explain["tokenizer"].get("tracked")
    if tracked:
        _fact(report, "C7", "pass", f"legacy digest is the tracked tokenizer '{tracked}' "
              f"({report.explain['tokenizer']['how']})", key, report.keys.get("tok_canonical"))
    else:
        _fact(report, "C7", "unknown", "legacy tokenizer digest is not a tracked tokenizer; supply the tokenizer to verify it", key)


def _resume_fact(report, view, target):
    from cevahir_ai.keys.schemes.ckpt import optimizer_signature, scheduler_signature
    if "optim_sig" not in report.keys:
        _fact(report, "C10", "unknown", "no optimizer state: a weights-only export cannot resume training")
        return
    problems = []
    if target.optimizer is not None:
        live, stored = optimizer_signature(target.optimizer.state_dict()), report.explain["optimizer"]
        if [g["n_params"] for g in live["groups"]] != [g["n_params"] for g in stored["groups"]]:
            problems.append("optimizer parameter groups differ")
        elif [g["hyper"] for g in live["groups"]] != [g["hyper"] for g in stored["groups"]]:
            problems.append("optimizer type differs (hyperparameter names)")
    if target.scheduler is not None:
        live_state = target.scheduler.state_dict()
        live_type = live_state.get("scheduler_type") if "scheduler_state" in live_state else None
        live_keys = scheduler_signature(live_state.get("scheduler_state") or {} if "scheduler_state" in live_state else live_state)
        stored = report.explain.get("scheduler")
        if stored is None:
            problems.append("the checkpoint has no scheduler state")
        elif stored["type_keys"] != live_keys or (stored["type"] and live_type and stored["type"] != live_type):
            problems.append(f"scheduler type differs: checkpoint {stored['type'] or stored['type_keys']}, "
                            f"target {live_type or live_keys}")
    if problems:
        _fact(report, "C10", "fail", "; ".join(problems), report.keys["optim_sig"])
    else:
        compared = "compared with the target" if (target.optimizer is not None or target.scheduler is not None) \
            else "present; no target optimizer or scheduler given"
        _fact(report, "C10", "pass", f"optimizer and scheduler state {compared}", report.keys["optim_sig"],
              report.keys.get("sched_sig"))


def _context_fact(report, normalized, target):
    if target.max_seq_length is None or normalized is None:
        _fact(report, "C11", "skipped", "no serving context length given" if normalized else "trained context length unknown")
        return
    trained = normalized.get("max_seq_length")
    scaled = normalized.get("pe_mode") == "rope" and normalized.get("rope_scaling_type") not in (None, "none")
    ok = target.max_seq_length <= trained or scaled
    _fact(report, "C11", "pass" if ok else "fail",
          f"serving context {target.max_seq_length} vs trained {trained}" + (" (RoPE scaling declared)" if scaled else ""))


def inspect_checkpoint(path, *, level="compat", purpose="evaluate", target=None, expect_file=None,
                       expect_weights=None, gate=None):
    """Inspect ``path`` and return a ``Report`` whose ``verdict`` is the gate's answer for ``purpose``."""
    if level not in LEVELS:
        raise ValueError(f"unknown level: {level!r}; expected one of {LEVELS}")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    target = target or Target()
    report = Report(path=str(path), level=level, purpose=purpose, size=path.stat().st_size)
    _file_facts(report, path, expect_file)
    if level != "file" and report.facts["C12"].status == "pass":
        _load_facts(report, path, level, target, expect_weights)
    report.verdict = check_compatibility(report.facts, purpose, gate)
    return report


def _load_facts(report, path, level, target, expect_weights):
    from cevahir_ai.adapters.envelopes import read_envelope
    from cevahir_ai.adapters.model_build import arch_of_config, arch_of_model
    from cevahir_ai.errors import EnvelopeError
    try:
        view = read_envelope(path)
    except EnvelopeError as error:
        _fact(report, "C12", "fail", str(error))
        return
    report.view, report.envelope = view, view.envelope
    report.notes.extend(view.notes)
    report.explain["epoch"], report.explain["metric"] = view.epoch, view.metric
    if view.declared:
        report.explain["declared"] = {name: value for name, value in view.declared.items() if name != "arch_readout"}
    if view.state is None:
        return
    instance = _weights_facts(report, view, expect_weights)
    _, fields, normalized = _arch_facts(report, view)
    _tokenizer_facts(report, view, path)
    _signature_facts(report, view)
    if instance is not None:
        _dtype_fact(report, target)
    if level == "compat" and instance is not None:
        target_arch = None
        if target.model is not None:
            key, target_fields, _ = arch_of_model(target.model)
            target_arch = (key, target_fields) if key else None
        elif target.config is not None:
            key, target_fields, _ = arch_of_config(target.config)
            target_arch = (key, target_fields)
        _layout_fact(report, view, instance, target, target_arch)
        _arch_fact(report, view, fields, target_arch)
        _tokenizer_fact(report, view, path, target)
        _resume_fact(report, view, target)
        _context_fact(report, normalized, target)
    _member_keys(report)
    report.state = view.state
