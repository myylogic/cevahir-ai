"""Key operations: the keys of a file, what a key means, and what differs between two artefacts."""
import os

from cevahir_ai.domain.records import Record
from cevahir_ai.errors import NotFound
from cevahir_ai.keys.grammar import KEY_RE, Key
from cevahir_ai.keys.registry import SCHEMES
from cevahir_ai.ops import artifacts


def scheme_records(kind=None, status=None):
    return [Record("Cevahir.KeyScheme", Name=scheme.name, Kind=scheme.kind, Scheme=scheme.scheme, Status=scheme.status,
                   Torch=scheme.torch, Newline=scheme.newline, Summary=scheme.summary)
            for scheme in SCHEMES.values()
            if (kind is None or scheme.kind == kind) and (status is None or scheme.status == status)]


def _info(store, name, key, source):
    key = Key.parse(str(key))
    scheme = SCHEMES[(key.kind, key.scheme)]
    owners = store.catalog().find(str(key)) if store is not None else []
    return Record("Cevahir.KeyInfo", {
        "Name": name, "Key": str(key), "Short": key.display(), "Kind": key.kind, "Scheme": key.scheme,
        "Status": scheme.status, "Meaning": scheme.summary, "Source": source,
        "Artifact": owners[0].key if len(owners) == 1 else None})


def keys_of(store, path=None, key=None, explain=False):
    """``Cevahir.KeyInfo`` records: every key derivable from a file, or the description of one key.

    For a file that is a registered payload the keys come from its manifest (no torch); an
    unregistered checkpoint is read (torch).
    """
    if key is not None:
        text = str(key)
        if not KEY_RE.match(text):
            text = artifacts.resolve(store, text).key
        record = _info(store, f"{Key.parse(text).kind}:{Key.parse(text).scheme}", text, "key")
        if explain:
            found = store.catalog().find(text)
            if len(found) == 1:
                document = found[0].latest
                record.set("Explain", {"kind": document["kind"], "provenance": document["provenance"],
                                       "members": document.get("members", {}), "inputs": document.get("inputs", []),
                                       "aliases": found[0].aliases, **document.get("explain", {})})
            elif (Key.parse(text).kind, Key.parse(text).scheme) == ("tok", "legacy-v3"):
                from cevahir_ai.adapters.tokenizers import lookup_legacy
                record.set("Explain", lookup_legacy(Key.parse(text).digest) or {"tracked": None})
        return [record]
    if path is None or not os.path.isfile(path):
        raise NotFound(f"no such file: {path}")
    from cevahir_ai.store import locations
    file_key = locations.file_key_of(store, path)
    records = [_info(store, "file:bytes-v1", file_key, "file")]
    owners = store.catalog().by_file(file_key)
    if owners:
        for owner in owners:
            document = owner.latest
            records.append(_info(store, owner.kind, owner.key, "manifest"))
            for name, value in document.get("members", {}).items():
                records.append(_info(store, name, value, "manifest"))
            if document.get("model"):
                records.append(_info(store, "model", document["model"], "manifest"))
        return records
    from cevahir_ai.keys.cli import _keys_of_file
    for name, value in sorted(_keys_of_file(path).items()):
        if name != "file:bytes-v1":
            records.append(_info(store, name, value, "computed"))
    return records


def _side(store, reference):
    """``(label, {member: key}, explain, tensor table or None)`` of one side of a comparison."""
    text = str(reference)
    try:
        artifact = artifacts.resolve(store, text)
    except NotFound:
        if KEY_RE.match(text):
            return text, {"key": text}, {}, None
        if not os.path.isfile(text):
            raise
        from cevahir_ai.adapters.inspection import inspect_checkpoint
        report = inspect_checkpoint(text, level="weights", purpose="inspect")
        names = ("ckpt", "weights", "wschema", "arch", "tok", "optim", "sched", "tstate", "model")
        members = {name: str(report.keys[name]) for name in names if report.keys.get(name)}
        members["key"] = members.pop("ckpt", str(report.keys["file"]))
        return text, members, dict(report.explain), report.table
    document = artifact.latest
    members = {"key": artifact.key, **document.get("members", {})}
    if document.get("model"):
        members["model"] = document["model"]
    table, detail = None, document.get("attributes", {}).get("detail")
    if detail and document["kind"] == "ckpt":
        try:
            table = store.get(detail, "tables").get("tensors")
        except NotFound:
            table = None
    return artifact.short(), members, document.get("explain", {}), table


def _tensor_detail(left, right):
    a = {row[0]: row for row in left}
    b = {row[0]: row for row in right}
    changed = sorted(name for name in a.keys() & b.keys() if a[name][1:] != b[name][1:])
    only_a, only_b = sorted(a.keys() - b.keys()), sorted(b.keys() - a.keys())
    parts = []
    for label, names in (("changed", changed), ("only in reference", only_a), ("only in difference", only_b)):
        if names:
            parts.append(f"{len(names)} {label}: {', '.join(names[:4])}{' …' if len(names) > 4 else ''}")
    return "; ".join(parts) or "same tensors", changed, only_a, only_b


def compare(store, reference, difference):
    """Member-by-member comparison, as ``Cevahir.KeyDiff`` records (one per member)."""
    left_name, left, left_explain, left_table = _side(store, reference)
    right_name, right, right_explain, right_table = _side(store, difference)
    order = ("key", "model", "weights", "wschema", "arch", "tok", "optim", "sched", "tstate")
    rows = []
    for member in [name for name in order if name in left or name in right]:
        a, b = left.get(member), right.get(member)
        detail, extra = "", {}
        if a != b and member == "arch" and left_explain.get("arch") and right_explain.get("arch"):
            fields = sorted(name for name in set(left_explain["arch"]) | set(right_explain["arch"])
                            if left_explain["arch"].get(name) != right_explain["arch"].get(name))
            detail = "; ".join(f"{name}: {left_explain['arch'].get(name)!r} -> {right_explain['arch'].get(name)!r}"
                               for name in fields)
            extra["Fields"] = fields
        elif a != b and member == "weights" and left_table and right_table:
            detail, changed, only_a, only_b = _tensor_detail(left_table, right_table)
            extra.update(Changed=changed, OnlyReference=only_a, OnlyDifference=only_b)
        elif a != b and (a is None or b is None):
            detail = "absent on one side"
        rows.append(Record("Cevahir.KeyDiff", dict({
            "Member": member, "Equal": a == b, "Reference": artifacts.short(a), "Difference": artifacts.short(b),
            "Detail": detail, "ReferenceKey": a, "DifferenceKey": b, "ReferenceName": left_name,
            "DifferenceName": right_name}, **extra)))
    return rows
