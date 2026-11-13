"""Fact providers of the compatibility gate, and the inspection report built from them.

``inspect_checkpoint`` answers, for any checkpoint file: is it intact, which tokenizer and
architecture does it hold, is it compatible with the current code (and with a given target),
and if not, why. Levels:

  file     torch-free: file key, sidecar, safety sniff of the pickle
  weights  loads with ``weights_only=True``: envelope, keys of weights, architecture, tokenizer ...
  compat   also derives the layout the *current code* builds for the declared architecture and
           compares against the target (model, tokenizer, optimizer, scheduler) if given

The providers fail closed. A stored config that leaves function-defining fields to defaults is no
declared architecture; a target model whose architecture cannot be read is an unknown target, not
an absent one; a tokenizer identity that nothing verified is a failed check, not a warning.
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
        # A static scan of the pickle: it names what the file would import. Whether torch's own
        # weights-only reader accepts the stream is established when the file is read (level weights).
        _fact(report, "C12", "pass", f"the pickle imports {len(sniff['globals'])} globals, all in the weights-only safe set")


def _arch_facts(report, view):
    """Declared architecture -> keys. Returns (arch key, fields, normalised) or (None, None, None).

    A stored config is a declaration only when it is complete (``model_build.undeclared_fields``).
    The normaliser would fill a missing ``num_heads`` with today's default, and a default says
    nothing about the weights in the file: such a config leaves the architecture unknown.
    """
    from cevahir_ai.adapters.model_build import arch_of_config, undeclared_fields
    from cevahir_ai.keys.schemes.arch import arch_ctor_v0_key
    if not view.model_config:
        return None, None, None
    try:
        key, fields, normalized = arch_of_config(view.model_config)
        gaps = undeclared_fields(view.model_config, fields, strict=view.envelope == "tm3")
    except Exception as error:
        gaps = [f"the current normaliser rejects it: {type(error).__name__}: {error}"]
    if gaps:
        reason = f"the stored config ({view.config_source}) declares no architecture: {'; '.join(gaps)}"
        report.explain["arch_undeclared"] = reason
        report.notes.append(reason + " (defaults are not a declaration)")
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
        if view.tokenizer_problem:
            report.explain["tokenizer"]["unverifiable"] = view.tokenizer_problem
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
              f"weights differ from the declared CEV-11 model_sha256 ({str(declared)[:12]}…)", key, alias)
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
    from cevahir_ai.adapters.envelopes import unwrap_scheduler_state
    from cevahir_ai.keys.schemes.ckpt import optim_sig_key, sched_sig_key
    from cevahir_ai.keys.torchkeys.optim import optim_content_key, sched_content_key, tstate_content_key
    if isinstance(view.optimizer_state, dict) and "param_groups" in view.optimizer_state:
        report.keys["optim_sig"], sig = optim_sig_key(view.optimizer_state)
        report.keys["optim"] = optim_content_key(view.optimizer_state)
        report.explain["optimizer"] = sig
    stored = view.scheduler_state
    if isinstance(stored, dict) and stored:
        inner, scheduler_type, wrapper = unwrap_scheduler_state(stored)
        report.keys["sched_sig"], sig = sched_sig_key(inner, scheduler_type, wrapper)
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


def _layout_fact(report, view, instance, target):
    """C4: the stored layout against every layout it has to fit.

    The layout the current code builds for the declared architecture says whether the file is what
    it claims to be. The layout of the live target model says what ``load_state_dict`` will be
    asked to do: both are compared, so a load that passes C4 cannot fail half-way on a shape.
    """
    from cevahir_ai.adapters.model_build import expected_layout
    from cevahir_ai.keys import schemas
    from cevahir_ai.keys.torchkeys.wschema import layout_keys, layout_of_module
    expectations = []
    try:
        if "arch" in report.keys:
            expectations.append((expected_layout(view.model_config), "the declared architecture"))
        if target.model is not None:
            expectations.append((layout_of_module(target.model), "the target model"))
        elif not expectations and target.config is not None:
            expectations.append((expected_layout(target.config), "the target architecture (no declared architecture)"))
    except Exception as error:
        _fact(report, "C4", "fail", f"the current code cannot build the architecture: {type(error).__name__}: {error}")
        return
    if not expectations:
        _fact(report, "C4", "unknown", "no declared architecture and no target: the expected layout is unknown")
        return
    expected_key = layout_keys(expectations[0][0])[0]
    report.explain["expected_layout"] = {"key": str(expected_key), "source": expectations[0][1],
                                         "preset": schemas.preset_of_layout_key(expected_key)}
    migrated = []
    for expected, source in expectations:
        problem, instance = _layout_problem(report, view, instance, expected, source, target, migrated)
        if problem:
            _fact(report, "C4", "fail", problem, expected_key)
            return
    detail = "names, shapes and ties match " + " and ".join(source for _, source in expectations)
    _fact(report, "C4", "pass", detail + "".join(f" after migration {name}" for name in migrated), expected_key)


def _layout_problem(report, view, instance, expected, source, target, migrated):
    """Why ``view.state`` does not fit ``expected`` (None when it does), and its possibly migrated layout."""
    from cevahir_ai.keys import schemas
    from cevahir_ai.keys.torchkeys.migrate import apply_migration
    from cevahir_ai.keys.torchkeys.wschema import broken_ties, compare_layouts, layout_of_state
    findings = compare_layouts(instance, expected)
    if findings:
        for migration_id in schemas.migration_ids():
            state, applied = apply_migration(view.state, schemas.load_migration(migration_id))
            if applied and not compare_layouts(layout_of_state(state), expected):
                if migration_id in target.allow_migrations:
                    view.state, instance, findings = state, layout_of_state(state), []
                    migrated.append(migration_id)
                    report.notes.append(f"migration {migration_id} applied in memory")
                else:
                    findings.append(f"the layout matches after migration {migration_id}: convert the file or allow it")
                break
    if findings:
        return f"layout differs from {source}: " + "; ".join(findings), instance
    ties = broken_ties(view.state, expected)
    if ties:
        return f"tie of {source} broken in file: " + ", ".join(f"{a} != {b}" for a, b in ties), instance
    return None, instance


def _arch_fact(report, view, fields, target_arch, target_problem):
    from cevahir_ai.keys.schemes.arch import diff_fields
    if "arch" not in report.keys:
        why = report.explain.get("arch_undeclared") or f"envelope {report.envelope} declares none"
        _fact(report, "C5", "unknown", f"architecture unknown ({why}): identical shapes do not prove an identical function")
        return
    if target_problem:
        _fact(report, "C5", "unknown", f"the architecture of the target model is unknown ({target_problem}): "
              "identical shapes do not prove an identical function", report.keys["arch"])
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


_UNVERIFIED = ("a tokenizer identity that nothing verified is refused, as ModelManager.load refuses it: "
               "supply the tokenizer")


def _tokenizer_fact(report, view, path, target):
    """C7. ``unknown`` only when the file carries no identity at all (a legacy file).

    A file that does carry one is bound to a tokenizer. The binding is verified against the
    tokenizer supplied, the tokenizer file next to a CEV-11 checkpoint, or the tracked registry;
    when none of them applies the check fails. The legacy contract has the same rule
    (``checkpoint_contract.validate_tokenizer_identity``).
    """
    from cevahir_ai.adapters import tokenizers
    from cevahir_ai.keys.schemes.tok import tok_bytebpe_key
    live = target.tokenizer
    if view.tokenizer is None:
        if not view.tokenizer_problem:
            _fact(report, "C7", "unknown", "the checkpoint carries no tokenizer identity: token semantics cannot be verified")
            return
        declared = view.declared
        same = (live is not None and callable(getattr(live, "identity", None))
                and type(live).__name__ == declared.get("tokenizer_class")
                and live.identity() == declared.get("tokenizer_identity"))
        if same:
            _fact(report, "C7", "pass", f"identity equals the target tokenizer ({view.tokenizer_problem}: no tokenizer key)")
        else:
            _fact(report, "C7", "fail", f"{view.tokenizer_problem}: " +
                  ("the supplied tokenizer does not have the stored identity" if live is not None else _UNVERIFIED))
        return
    scheme, digest = view.tokenizer
    key = report.keys["tok"]
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
            _fact(report, "C7", "fail", f"the tokenizer file next to the checkpoint is missing; {_UNVERIFIED}", key)
            return
        try:
            ok = tok_bytebpe_key(sibling) == key
        except (ValueError, KeyError, TypeError) as error:
            _fact(report, "C7", "fail", f"tokenizer file {sibling.name} is unreadable: {error}", key)
            return
        _fact(report, "C7", "pass" if ok else "fail",
              f"tokenizer file {sibling.name} has the declared identity" if ok else
              f"tokenizer file {sibling.name} is not the tokenizer the model was trained with", key)
        return
    tracked = report.explain["tokenizer"].get("tracked")
    if tracked:
        changed = tokenizers.tracked_files_changed(tracked)
        if changed:
            _fact(report, "C7", "fail", f"legacy digest names the tracked tokenizer '{tracked}', but its files in this "
                  f"checkout are not the recorded ones: {', '.join(changed)}", key, report.keys.get("tok_canonical"))
        else:
            _fact(report, "C7", "pass", f"legacy digest is the tracked tokenizer '{tracked}' "
                  f"({report.explain['tokenizer']['how']}); its files in this checkout are the recorded ones",
                  key, report.keys.get("tok_canonical"))
    else:
        _fact(report, "C7", "fail", f"the legacy tokenizer digest is not a tracked tokenizer; {_UNVERIFIED}", key)


def _resume_fact(report, view, target):
    """C10: the stored optimizer and scheduler state against the objects that would take it."""
    from cevahir_ai.adapters.envelopes import unwrap_scheduler_state
    from cevahir_ai.keys.schemes.ckpt import optimizer_signature, scheduler_signature
    if "optim_sig" not in report.keys:
        _fact(report, "C10", "unknown", "no optimizer state: a weights-only export cannot resume training")
        return
    problems, compared = [], []
    if target.optimizer is not None:
        live, stored = optimizer_signature(target.optimizer.state_dict()), report.explain["optimizer"]
        if [g["n_params"] for g in live["groups"]] != [g["n_params"] for g in stored["groups"]]:
            problems.append("optimizer parameter groups differ")
        elif [g["hyper"] for g in live["groups"]] != [g["hyper"] for g in stored["groups"]]:
            problems.append("optimizer type differs (hyperparameter names)")
        compared.append("optimizer")
    stored = report.explain.get("scheduler")
    if target.scheduler is not None:
        inner, live_type, _ = unwrap_scheduler_state(target.scheduler.state_dict())
        live_keys = scheduler_signature(inner)
        if stored is None:
            problems.append("the checkpoint has no scheduler state")
        elif stored["type_keys"] != live_keys or (stored["type"] and live_type and stored["type"] != live_type):
            problems.append(f"scheduler type differs: checkpoint {stored['type'] or stored['type_keys']}, "
                            f"target {live_type or live_keys}")
        compared.append("scheduler")
    if problems:
        _fact(report, "C10", "fail", "; ".join(problems), report.keys["optim_sig"])
        return
    detail = (f"{' and '.join(compared)} state fits the target" if compared
              else "optimizer state present; no target optimizer or scheduler given")
    if stored is not None and target.scheduler is None and target.optimizer is not None:
        detail += "; the stored scheduler state has no target and will not be restored"
    _fact(report, "C10", "pass", detail, report.keys["optim_sig"], report.keys.get("sched_sig"))


def _context_fact(report, normalized, target, target_normalized):
    """C11. The serving context is the one requested, else the target's own, else the trained one."""
    if normalized is None:
        _fact(report, "C11", "unknown", "trained context length unknown (no declared architecture)")
        return
    trained = normalized.get("max_seq_length")
    if target.max_seq_length is not None:
        serving, origin = target.max_seq_length, "requested"
    elif target_normalized is not None:
        serving, origin = target_normalized.get("max_seq_length"), "of the target"
    else:
        serving, origin = trained, "none requested: the trained context applies"
    scaled = normalized.get("pe_mode") == "rope" and normalized.get("rope_scaling_type") not in (None, "none")
    if not isinstance(serving, int) or not isinstance(trained, int):
        _fact(report, "C11", "unknown", f"context lengths are not comparable (serving {serving!r}, trained {trained!r})")
        return
    ok = serving <= trained or scaled
    _fact(report, "C11", "pass" if ok else "fail",
          f"serving context {serving} ({origin}) vs trained {trained}" + (" (RoPE scaling declared)" if scaled else ""))


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


def _target_arch(target):
    """``(key, fields)``, the normalised target config and, for an unreadable target, the reason."""
    from cevahir_ai.adapters.model_build import arch_of_config, arch_of_model
    if target.model is not None:
        try:
            key, fields, normalized = arch_of_model(target.model)
        except Exception as error:
            return None, None, str(error) or type(error).__name__
        return (key, fields), normalized, None
    if target.config is not None:
        key, fields, normalized = arch_of_config(target.config)
        return (key, fields), normalized, None
    return None, None, None


def _load_facts(report, path, level, target, expect_weights):
    from cevahir_ai.adapters.envelopes import read_envelope
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
        target_arch, target_normalized, target_problem = _target_arch(target)
        _layout_fact(report, view, instance, target)
        _arch_fact(report, view, fields, target_arch, target_problem)
        _tokenizer_fact(report, view, path, target)
        _resume_fact(report, view, target)
        _context_fact(report, normalized, target, target_normalized)
    _member_keys(report)
    report.state = view.state
