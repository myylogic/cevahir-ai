"""Artefact operations: register, list, resolve, lineage, export and import."""
import os
from pathlib import Path

from cevahir_ai.domain.records import Record
from cevahir_ai.errors import NotFound
from cevahir_ai.keys.grammar import KEY_RE, Key
from cevahir_ai.ops.governed import governed
from cevahir_ai.store import bundles, lineage, locations
from cevahir_ai.store import register as registration
from cevahir_ai.store.core import Store, inconsistent, new_manifest, statement, tool_name

STATES = ("registered", "imported", "verified", "quarantined")


def open_store(home=None, environ=None):
    return Store(home, environ=environ)


def short(key):
    """Short display form of a key string; anything that is not a key is returned unchanged."""
    if key is None:
        return None
    try:
        return Key.parse(str(key)).display()
    except ValueError:
        return str(key)


# ------------------------------------------------------------------------------------------ states
def states(store):
    """``{artefact key: state}`` derived from this host's audit events (never stored).

    ``verified`` when the last ``Test-Checkpoint`` of it established its digests; ``quarantined``
    when that test found other bytes than the registered ones or an unsafe envelope; ``imported``
    when it came from a bundle and was never registered or verified here; else ``registered``.
    A verdict of "incompatible with this target" is not a state: it depends on the question asked.
    """
    from cevahir_ai.store.audit import AuditLog
    found = {}
    for event in AuditLog(store).events():
        name = event.get("event")
        for target in event.get("targets") or ():
            key = target.get("key")
            if name == "artifact.verified":
                outcome = event.get("outcome") or {}
                if outcome.get("intact") == "no" or outcome.get("envelope_safety") in ("fail", "unsupported"):
                    found[key] = "quarantined"
                elif outcome.get("intact") == "yes":
                    found[key] = "verified"
            elif name == "artifact.imported":
                found.setdefault(key, "imported")
            elif name in ("artifact.registered", "artifact.derived", "artifact.located"):
                if found.get(key) in (None, "imported"):
                    found[key] = "registered"
    return found


def state_of(artifact, known):
    if artifact.latest.get("attributes", {}).get("quarantined") and known.get(artifact.key) != "verified":
        return "quarantined"
    return known.get(artifact.key, "registered")


def paths_of(store, artifact):
    """Paths on this host that hold the artefact's payload files (existing files only)."""
    found = []
    for file_key in artifact.files:
        found += [entry["path"] for entry in locations.live_locations(store, file_key)]
    return found


def artifact_record(store, artifact, known=None, new=None):
    document = artifact.latest
    record = Record("Cevahir.Artifact", {
        "Key": artifact.key, "Short": artifact.short(), "Kind": artifact.kind,
        "State": state_of(artifact, states(store) if known is None else known),
        "Provenance": document["provenance"], "Created": artifact.created,
        "Locations": paths_of(store, artifact), "Aliases": artifact.aliases,
        "Inputs": [Record("Cevahir.Input", Role=item["role"], Key=item["key"]) for item in artifact.inputs],
        "Members": dict(document.get("members", {})), "Model": document.get("model"),
        "Files": artifact.files, "Manifests": [f"cev:manifest:sha256-v1:{digest}" for digest, _ in artifact.manifests],
        "Attributes": {name: value for name, value in document.get("attributes", {}).items() if name != "detail"},
    })
    if new is not None:
        record.set("New", new)
    return record


# ----------------------------------------------------------------------------------------- resolve
def resolve(store, reference):
    """The artefact a reference names: a key, an alias, a short form, or the path of a registered file."""
    catalog = store.catalog()
    text = str(reference)
    found = catalog.find(text)
    if not found and not KEY_RE.match(text) and os.path.isfile(text):
        found = catalog.by_file(locations.file_key_of(store, text))
        if not found:
            raise NotFound(f"{text} is not a registered artefact; register it first: Register-Artifact '{text}'")
    if len(found) == 1:
        return found[0]
    if not found and KEY_RE.match(text):
        holders = [artifact.short() for artifact in catalog.artifacts.values()
                   if text in artifact.latest.get("members", {}).values() or artifact.latest.get("model") == text]
        if holders:
            raise NotFound(f"{short(text)} is not an artefact of its own; it is a member of {', '.join(holders[:5])}")
    return catalog.resolve(text)             # raises NotFound or AmbiguousReference with the right message


def get(store, reference=None, kind=None, state=None):
    catalog, known = store.catalog(), states(store)
    artifacts = [resolve(store, reference)] if reference is not None else catalog.of_kind()
    records = [artifact_record(store, artifact, known) for artifact in artifacts
               if kind is None or artifact.kind == kind]
    return [record for record in records if state is None or record.State == state]


# ---------------------------------------------------------------------------------------- register
def _record(store, description, *, action, event, provenance, params, activity=None):
    """Store one description as a manifest with its locations and one audit event. Returns ``(key, new)``."""
    attributes = dict(description.attributes)
    if description.detail:
        attributes["detail"] = store.put(description.detail, "tables")
    document = new_manifest(
        description.kind, description.key, members=description.members, model=description.model,
        aliases=description.aliases, inputs=description.inputs, payload=description.payload, attributes=attributes,
        explain=description.explain, provenance=provenance, created=store.clock(),
        producer={"action": action, "tool": tool_name(), "host": store.host_id(create=True)})
    problem = inconsistent(document)
    if problem:
        raise ValueError(f"{description.kind} {document['key']}: {problem}")
    existing = store.catalog().artifacts.get(document["key"])
    same = existing is not None and any(statement(old) == statement(document) for _, old in existing.manifests)
    recorded = {entry["path"] for file_key, _ in description.files for entry in locations.locations_of(store, file_key)}
    fresh = [(file_key, path) for file_key, path in description.files if locations.absolute(path) not in recorded]
    if same and not fresh:
        for file_key, path in description.files:           # refresh size and mtime of what is already known
            locations.add_location(store, file_key, path)
        return document["key"], False
    with governed(store, action, event="artifact.located" if same else event, params=params) as handle:
        handle.targets = [{"kind": description.kind, "key": document["key"]}]
        if not same:
            manifest_id = store.put(document)
            handle.outputs = [{"manifest": f"cev:manifest:sha256-v1:{manifest_id}"}]
            handle.lineage = {"output": document["key"], "inputs": document["inputs"]}
            if activity:
                handle.lineage["activity"] = activity
        for file_key, path in description.files:
            locations.add_location(store, file_key, path)
        handle.data = {"files": [file_key for file_key, _ in fresh]} if fresh else None
    store.invalidate()
    return document["key"], not same


def register(store, path, kind="auto", recurse=False, warn=None):
    """Register what is under ``path``. Returns artefact records, each with ``New``.

    A file whose bytes are already the payload of a registered artefact is only given a new
    location: that needs no torch, and it is how a checkpoint copied from another machine is
    recognised after its manifest was imported.
    """
    warn = warn or (lambda message: None)
    candidates, skipped = registration.scan(path, recurse, kind)
    params = {"Path": str(path), "Kind": kind, "Recurse": bool(recurse)}
    results = []
    for candidate in candidates:
        if len(candidate.paths) == 1 and candidate.kind in ("ckpt", "cache", "tok-byte", "file"):
            file_key = locations.file_key_of(store, candidate.paths[0])
            owners = store.catalog().by_file(file_key)
            if owners:
                new = locations.absolute(candidate.paths[0]) not in \
                    {entry["path"] for entry in locations.locations_of(store, file_key)}
                if new:
                    with governed(store, "Register-Artifact", event="artifact.located", params=params) as handle:
                        handle.targets = [{"kind": owner.kind, "key": owner.key} for owner in owners]
                        handle.data = {"files": [file_key]}
                        locations.add_location(store, file_key, candidate.paths[0])
                else:
                    locations.add_location(store, file_key, candidate.paths[0])
                results += [(owner.key, False) for owner in owners]
                continue
        try:
            descriptions = registration.describe(candidate)
        except (ValueError, KeyError, OSError) as error:
            warn(f"{candidate.paths[0]}: not registered: {error}")
            continue
        for description in descriptions:
            for note in description.notes:
                warn(f"{candidate.paths[0]}: {note}")
            results.append(_record(store, description, action="Register-Artifact", event="artifact.registered",
                                   provenance="legacy-import", params=params))
    if skipped and Path(path).is_file():
        warn(f"{path}: not a recognised artefact (use -Kind file to register it as a plain file)")
    catalog, known = store.catalog(), states(store)
    seen, records = set(), []
    for key, new in results:
        if key not in seen:
            seen.add(key)
            records.append(artifact_record(store, catalog.artifacts[key], known, new=new))
    return records


def derive(store, kind, key, *, inputs=(), activity=None, attributes=None, members=None, model=None, files=(),
           action="Record-Derivation"):
    """Record an artefact that an operation produced from ``inputs`` (provenance ``native``).

    ``files`` are local paths of its payload. This is the entry point of governed operations
    (training, cache preparation) that state their own lineage; ``activity`` names the run.
    """
    payload, located = [], []
    for path in files:
        entry = registration._payload("payload", path)
        payload.append(entry)
        located.append((entry["file"], Path(path)))
    description = registration.Description(kind, str(key), members=dict(members or {}), model=model, inputs=list(inputs),
                                       payload=payload, attributes=dict(attributes or {}), files=located)
    found, new = _record(store, description, action=action, event="artifact.derived", provenance="native",
                         params={"Kind": kind, "Key": str(key)}, activity=activity)
    return artifact_record(store, store.catalog().artifacts[found], new=new)


# ----------------------------------------------------------------------------------------- lineage
def lineage_of(store, reference, direction="Up", depth=None):
    """Lineage edges around a reference, as ``Cevahir.LineageEdge`` records (see ``store.lineage``)."""
    try:
        start = resolve(store, reference).key
    except NotFound:
        if not KEY_RE.match(str(reference)):
            raise
        start = str(reference)               # a key that only appears as someone's input still has lineage
    catalog = store.catalog()
    rows = []
    for level, way, edge in lineage.walk(store, start, direction, depth):
        far = edge.source if way == "Up" else edge.target
        arrow = "<-" if way == "Up" else "->"
        rows.append(Record("Cevahir.LineageEdge", {
            "Level": level, "Direction": way, "Source": edge.source, "Target": edge.target, "Role": edge.role,
            "How": edge.how, "Registered": far in catalog.artifacts, "Host": edge.host, "Seq": edge.seq,
            "Time": edge.ts, "Tree": f"{'  ' * (level - 1)}{arrow} {short(far)}  [{edge.role}]"}))
    return rows


# ------------------------------------------------------------------------------- export and import
def export(store, references, destination, include_payload=False, include_audit=False):
    """Write artefacts and their registered ancestors to a ``.cvb`` bundle."""
    catalog = store.catalog()
    chosen, order = {}, []
    for reference in references:
        artifact = resolve(store, reference)
        if artifact.key not in chosen:
            chosen[artifact.key] = artifact
            order.append(artifact.key)
    requested = list(order)
    for key in requested:                                  # manifests of registered ancestors travel too
        for _, _, edge in lineage.walk(store, key, "Up"):
            if edge.source in catalog.artifacts and edge.source not in chosen:
                chosen[edge.source] = catalog.artifacts[edge.source]
                order.append(edge.source)
    params = {"Key": requested, "To": str(destination), "IncludePayload": bool(include_payload),
              "IncludeAudit": bool(include_audit)}
    with governed(store, "Export-Artifact", event="artifact.exported", params=params) as handle:
        handle.targets = [{"kind": chosen[key].kind, "key": key} for key in order]
        description = bundles.export_bundle(
            store, [chosen[key] for key in order], destination, include_audit=include_audit,
            payload_of=requested if include_payload else (),       # ancestors travel as manifests only
            find_file=lambda file_key: locations.find_file(store, file_key))
        handle.data = {"bundle": description["id"], "manifests": len(description["manifests"]),
                       "payloads": len(description["payloads"]), "chains": len(description["chains"])}
    return Record("Cevahir.Bundle", {
        "Path": description["path"], "Id": description["id"], "Artifacts": order,
        "Manifests": len(description["manifests"]), "Payloads": len(description["payloads"]),
        "Chains": [chain["host"] for chain in description["chains"]], "Host": description["host"]})


def import_bundle(store, path, destination=None):
    """Import a ``.cvb`` bundle after verifying all of it. Returns artefact records, each with ``New``."""
    content = bundles.read_bundle(path)                    # raises BundleError; nothing is written before this passes
    description = content["description"]
    blobs = Path(destination) if destination else store.home / "blobs"
    held = set(store.object_ids())
    by_key = {}
    for digest, document in content["manifests"].items():
        by_key.setdefault(document["key"], []).append((digest, document))
    results = []
    for chain in content["chains"]:
        bundles.merge_chain(store, chain)
    for digest, data in content["details"].items():
        target = store.object_path(digest, "tables")
        if not target.is_file():
            from cevahir_ai._compat.fsio import atomic_write
            atomic_write(target, data)
    placed = {}
    for payload in content["payloads"]:
        placed[payload["file"]] = bundles.extract_payload(content["path"], payload, blobs)
    params = {"Path": str(path)}
    for key, manifests in by_key.items():
        fresh = [(digest, document) for digest, document in manifests if digest not in held]
        files = {item["file"] for _, document in manifests for item in document.get("payload", ())} & set(placed)
        recorded = {entry["path"] for file_key in files for entry in locations.locations_of(store, file_key)}
        new_files = [file_key for file_key in files if locations.absolute(placed[file_key]) not in recorded]
        if not fresh and not new_files:
            results.append((key, False))
            continue
        with governed(store, "Import-Artifact", event="artifact.imported", params=params) as handle:
            handle.targets = [{"kind": manifests[0][1]["kind"], "key": key}]
            inputs = []
            for digest, document in manifests:
                store.put(document)
                inputs += [item for item in document.get("inputs", ()) if item not in inputs]
            handle.outputs = [{"manifest": f"cev:manifest:sha256-v1:{digest}"} for digest, _ in fresh]
            handle.lineage = {"output": key, "inputs": inputs}
            handle.data = {"bundle": content["id"], "source_host": description.get("host"), "files": sorted(new_files)}
            for file_key in files:
                locations.add_location(store, file_key, placed[file_key])
        results.append((key, bool(fresh)))
    store.invalidate()
    catalog, known = store.catalog(), states(store)
    return [artifact_record(store, catalog.artifacts[key], known, new=new) for key, new in results]
