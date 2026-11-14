"""Checkpoint, model, tokenizer and cache operations over the registry.

Listing is torch-free: everything shown comes from manifests. ``test_checkpoint`` runs the
compatibility gate (``cevahir_ai.adapters.inspection``); level ``File`` needs no torch, the other
levels import it on first use. A registered checkpoint is tested against its manifest: the bytes
on disk must be the registered file (C1) and the tensors the registered weights (C3), so a file
that changed after registration is reported, where an unregistered file can only say "unknown".
"""
import json
import os
from pathlib import Path

from cevahir_ai.domain.records import Record
from cevahir_ai.domain.verdicts import IMPLEMENTED_CHECKS, PURPOSES
from cevahir_ai.errors import NotFound
from cevahir_ai.keys.grammar import KEY_RE
from cevahir_ai.ops import artifacts
from cevahir_ai.ops.governed import record as record_event
from cevahir_ai.store import locations

LEVELS = ("File", "Weights", "Compat")


def _first(values):
    return values[0] if values else None


def checkpoint_record(store, artifact, known):
    document = artifact.latest
    attributes, members = document.get("attributes", {}), document.get("members", {})
    paths = artifacts.paths_of(store, artifact)
    return Record("Cevahir.Checkpoint", {
        "Key": artifact.key, "Short": artifact.short(), "Epoch": attributes.get("epoch"),
        "Metric": attributes.get("metric"), "Envelope": attributes.get("envelope"),
        "Arch": artifacts.short(members.get("arch")), "Tok": artifacts.short(members.get("tok")),
        "Weights": members.get("weights"), "Model": document.get("model"),
        "Dtype": ", ".join(attributes.get("dtypes") or ()), "Parameters": attributes.get("parameters"),
        "State": artifacts.state_of(artifact, known), "Provenance": document["provenance"],
        "Path": _first(paths), "Locations": paths, "File": _first(artifact.files),
        "Created": artifact.created, "Members": dict(members)})


def _written(record):
    try:
        return os.stat(record.Path).st_mtime_ns if record.Path else 0
    except OSError:
        return 0


def get_checkpoints(store, reference=None, latest=False, best=False, path=None):
    """Registered checkpoints. ``latest``: the one whose file was written last; ``best``: lowest metric."""
    catalog, known = store.catalog(), artifacts.states(store)
    if path is not None:
        found = [artifacts.resolve(store, path)]
    elif reference is not None:
        found = [artifacts.resolve(store, reference)]
    else:
        found = catalog.of_kind("ckpt")
    found = [artifact for artifact in found if artifact.kind == "ckpt"]
    if not found and (reference is not None or path is not None):
        raise NotFound(f"{reference or path} is registered, but it is not a checkpoint")
    records = [checkpoint_record(store, artifact, known) for artifact in found]
    if best:
        scored = [record for record in records if isinstance(record.Metric, (int, float))]
        records = sorted(scored, key=lambda record: record.Metric)[:1]
    if latest:
        records = sorted(records, key=lambda record: (_written(record), record.Epoch or -1, record.Created))[-1:]
    return records


def get_models(store, reference=None):
    catalog, known = store.catalog(), artifacts.states(store)
    found = [artifacts.resolve(store, reference)] if reference is not None else catalog.of_kind("model")
    records = []
    for artifact in found:
        if artifact.kind != "model":
            continue
        document = artifact.latest
        members, attributes = document.get("members", {}), document.get("attributes", {})
        records.append(Record("Cevahir.Model", {
            "Key": artifact.key, "Short": artifact.short(), "Weights": members.get("weights"),
            "Arch": artifacts.short(members.get("arch")), "Tok": artifacts.short(members.get("tok")),
            "Checkpoints": [item["key"] for item in artifact.inputs if item["role"] == "checkpoint"],
            "Parameters": attributes.get("parameters"), "Dtype": ", ".join(attributes.get("dtypes") or ()),
            "Published": False, "State": artifacts.state_of(artifact, known), "Created": artifact.created,
            "Members": dict(members)}))
    return records


def get_tokenizers(store, reference=None, tracked=False):
    """Registered tokenizers, or (``tracked``) the tokenizers pinned in ``governance/registry/tokenizers.json``."""
    if tracked:
        from cevahir_ai.adapters.tokenizers import tracked_registry
        registered = store.catalog().artifacts
        return [Record("Cevahir.Tokenizer", {
            "Key": entry["key"], "Short": artifacts.short(entry["key"]), "Family": entry.get("family"),
            "VocabSize": entry.get("vocab_size"), "Merges": entry.get("merges"), "Tracked": entry["name"],
            "Aliases": [alias["key"] for alias in entry.get("aliases", ())], "Registered": entry["key"] in registered,
            "Locations": [], "State": "tracked"}) for entry in tracked_registry()["tokenizers"]]
    catalog, known = store.catalog(), artifacts.states(store)
    found = [artifacts.resolve(store, reference)] if reference is not None else catalog.of_kind("tok")
    records = []
    for artifact in found:
        if artifact.kind != "tok":
            continue
        attributes = artifact.latest.get("attributes", {})
        records.append(Record("Cevahir.Tokenizer", {
            "Key": artifact.key, "Short": artifact.short(), "Family": attributes.get("family"),
            "VocabSize": attributes.get("vocab_size"), "Merges": attributes.get("merges"),
            "Tracked": attributes.get("tracked"), "Aliases": artifact.aliases, "Registered": True,
            "Locations": artifacts.paths_of(store, artifact), "State": artifacts.state_of(artifact, known)}))
    return records


def get_caches(store, reference=None):
    catalog, known = store.catalog(), artifacts.states(store)
    found = [artifacts.resolve(store, reference)] if reference is not None else catalog.of_kind("cache")
    records = []
    for artifact in found:
        if artifact.kind != "cache":
            continue
        attributes = artifact.latest.get("attributes", {})
        meta = attributes.get("meta") or {}
        records.append(Record("Cevahir.Cache", {
            "Key": artifact.key, "Short": artifact.short(),
            "Corpus": _first([item["key"] for item in artifact.inputs if item["role"] == "corpus"]),
            "Sidecar": attributes.get("sidecar"), "Samples": meta.get("sample_count"),
            "MaxSeqLength": meta.get("max_seq_length"), "EncodeMode": meta.get("encode_mode"),
            "Locations": artifacts.paths_of(store, artifact), "State": artifacts.state_of(artifact, known)}))
    return records


# ------------------------------------------------------------------------------------------- test
def build_target(target=None, vocab=None, merges=None, byte_bpe=None, dtype="float32", max_seq_length=None,
                 allow_migration=()):
    """What a checkpoint is tested against: a preset name or a JSON config file, and a tokenizer."""
    from cevahir_ai.adapters.inspection import Target
    result = Target(dtype=dtype or "float32", max_seq_length=max_seq_length, allow_migrations=tuple(allow_migration or ()))
    if target:
        from cevahir_ai.keys import schemas
        if target in schemas.PRESETS:
            result.config = schemas.preset_config(target)
        else:
            result.config = json.loads(Path(target).read_text(encoding="utf-8"))
    if byte_bpe:
        from cevahir_ai._compat.repo import import_legacy
        result.tokenizer = import_legacy("tokenizer_management.byte_bpe").ByteLevelBPE.load(str(byte_bpe))
    elif vocab or merges:
        if not (vocab and merges):
            raise ValueError("-Vocab and -Merges go together")
        from cevahir_ai._compat.repo import import_legacy
        core = import_legacy("tokenizer_management.core.tokenizer_core").TokenizerCore
        result.tokenizer = core({"vocab_path": str(vocab), "merges_path": str(merges), "read_only": True, "use_gpu": False})
    return result


def verdict_record(report, key, target_kind):
    verdict = report.verdict
    checks = [Record("Cevahir.Check", Id=check.id, Name=check.name, Status=check.status,
                     Required=check.id in verdict.required, Detail=check.detail) for check in verdict.checks]
    props = {"Key": key, "Short": artifacts.short(key), "Kind": target_kind, "Path": report.path,
             "Level": report.level.capitalize(), "Purpose": verdict.purpose, "Verdict": verdict.verdict,
             "Ok": verdict.ok, "Intact": report.intact, "Envelope": report.envelope,
             "Failing": list(verdict.failing), "Warnings": list(verdict.warnings),
             "Summary": verdict.summary_en, "SummaryTr": verdict.summary_tr, "Checks": checks,
             "Notes": list(report.notes)}
    for check_id in IMPLEMENTED_CHECKS:
        fact = report.facts.get(check_id)
        props[check_id] = fact.status if fact is not None else "-"
    return Record("Cevahir.CheckpointVerdict", props)


def test_checkpoint(store, reference=None, path=None, level="File", purpose="inspect", target=None, record=True):
    """Run the compatibility gate on a registered checkpoint or on a file. Returns a verdict record.

    Never raises for an incompatible checkpoint: the verdict says so (``Ok`` is false). With
    ``record`` the verdict is appended to the audit log as ``artifact.verified``.
    """
    from cevahir_ai.adapters.inspection import inspect_checkpoint
    wanted = {name.lower(): name for name in LEVELS}
    if str(level).lower() not in wanted:
        raise ValueError(f"unknown level: {level!r}; expected one of {LEVELS}")
    if purpose not in PURPOSES:
        raise ValueError(f"unknown purpose: {purpose!r}; expected one of {PURPOSES}")
    artifact, expect_file, expect_weights = None, None, None
    if path is None and reference is not None and not KEY_RE.match(str(reference)) and os.path.isfile(str(reference)):
        path, reference = str(reference), None          # a file where a key is expected: test the file
    if path is None:
        if reference is None:
            raise ValueError("give a checkpoint key (or short form) or -Path")
        artifact = artifacts.resolve(store, reference)
        if artifact.kind not in ("ckpt", "file"):
            raise NotFound(f"{artifact.short()} is a {artifact.kind}, not a checkpoint")
        for file_key in artifact.files:
            found = locations.find_file(store, file_key)
            if found is not None:
                path, expect_file = found, file_key
                break
        if path is None:
            raise NotFound(f"{artifact.short()} has no file on this machine; copy the file here and run "
                           "Register-Artifact on it (its bytes are recognised by their file key)")
    else:
        from cevahir_ai.keys.schemes.file import file_bytes_key
        file_key = str(file_bytes_key(path))
        owners = [owner for owner in store.catalog().by_file(file_key) if owner.kind in ("ckpt", "file")]
        if owners:
            artifact, expect_file = owners[0], file_key
    if artifact is not None and artifact.kind == "ckpt":
        expect_weights = artifact.latest.get("members", {}).get("weights")
    report = inspect_checkpoint(path, level=wanted[str(level).lower()].lower(), purpose=purpose,
                                target=target, expect_file=expect_file, expect_weights=expect_weights)
    if artifact is not None:
        key, kind = artifact.key, artifact.kind
    else:
        key, kind = str(report.keys.get("ckpt") or report.keys["file"]), "ckpt" if "ckpt" in report.keys else "file"
    result = verdict_record(report, key, kind)
    result.set("Registered", artifact is not None)
    if record:
        event = record_event(store, "artifact.verified", "Test-Checkpoint", targets=[{"kind": kind, "key": key}],
                             params={"Level": result.Level, "Purpose": purpose},
                             outcome={"status": "ok" if report.verdict.ok else "failed", "verdict": report.verdict.verdict,
                                      "failing": list(report.verdict.failing), "warnings": list(report.verdict.warnings),
                                      "intact": report.intact, "envelope_safety": report.facts["C12"].status})
        result.set("AuditSeq", event["seq"])
    return result
