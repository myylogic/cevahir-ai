"""``.cvb`` bundles: manifests (and optionally payloads and the audit chain) in one zip file.

    bundle.json                  what the bundle holds, with the digest of every member
    objects/<id>.json            manifests, named by their content address
    tables/<id>.json             their companion documents (per-tensor tables, file lists)
    payloads/<file sha256>/<name>  payload bytes, stored without compression
    audit/<host-id>/seg-*.jsonl  the exporting host's audit chain (only when asked for), chain.json

Import verifies everything before it writes anything: every manifest must hash to its name, every
payload to its file key, and a foreign chain must verify and must not contradict a copy of the
same chain already held. Member names are never taken from the archive: they are computed from
validated digests, so a crafted bundle cannot write outside the home.

A bundle is not signed. Import proves that it is internally consistent, not who made it: an
imported artefact is in state ``imported`` until it is verified here.
"""
import hashlib
import json
import re
import zipfile
from pathlib import Path

from cevahir_ai._compat.fsio import CHUNK, atomic_write
from cevahir_ai.errors import BundleError
from cevahir_ai.keys.canonical import cjson
from cevahir_ai.store.audit import HOST_RE, SEGMENT_RE, AuditLog, verify_chain
from cevahir_ai.store.core import MANIFEST_FORMAT, inconsistent, tool_name
from cevahir_ai.store.register import safe_name

BUNDLE_FORMAT, BUNDLE_VERSION = "cevahir-bundle", 1
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_FILE_KEY = re.compile(r"^cev:file:bytes-v1:([0-9a-f]{64})$")
_STORED_SUFFIXES = (".pth", ".pt", ".ckpt", ".pkl", ".safetensors")
MAX_JSON_BYTES = 256 * 1024 * 1024


def _sha256_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(CHUNK), b""):
        digest.update(chunk)
    return digest.hexdigest()


def export_bundle(store, artifacts, destination, *, payload_of=(), include_audit=False, find_file=None):
    """Write ``artifacts`` (``Artifact`` objects) to ``destination``. Returns the bundle description.

    ``payload_of`` lists the artefact keys whose payload bytes go into the bundle; the others travel
    as manifests only. ``find_file(file key) -> path or None`` locates payload bytes on this host; a
    payload that cannot be found, or whose bytes are no longer the registered ones, fails the export.
    """
    destination = Path(destination)
    manifests, details, payloads, seen_files = [], [], [], set()
    for artifact in artifacts:
        for digest, document in artifact.manifests:
            manifests.append({"id": digest, "key": artifact.key})
            detail = document.get("attributes", {}).get("detail")
            if detail and detail not in details and store.object_path(detail, "tables").is_file():
                details.append(detail)
        if artifact.key in payload_of:
            for item in artifact.payload:
                match = _FILE_KEY.match(item["file"])
                if match and match.group(1) not in seen_files:
                    seen_files.add(match.group(1))
                    payloads.append({"file": item["file"], "name": safe_name(item.get("name_hint") or "payload"),
                                     "size": item.get("size")})
    description = {"format": BUNDLE_FORMAT, "version": BUNDLE_VERSION, "created": store.clock(),
                   "host": store.host_id(), "tool": tool_name(), "manifests": manifests, "details": details,
                   "payloads": payloads, "chains": []}
    temporary = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for item in manifests:
                archive.write(store.object_path(item["id"]), f"objects/{item['id']}.json")
            for digest in details:
                archive.write(store.object_path(digest, "tables"), f"tables/{digest}.json")
            for item in payloads:
                digest = _FILE_KEY.match(item["file"]).group(1)
                path = find_file(item["file"]) if find_file else None
                if path is None:
                    raise BundleError(f"payload {item['name']} ({item['file'][:32]}…) is not on this host")
                with open(path, "rb") as stream:
                    if _sha256_stream(stream) != digest:
                        raise BundleError(f"{path} no longer holds the registered bytes of {item['name']}")
                item["size"] = Path(path).stat().st_size
                method = zipfile.ZIP_STORED if item["name"].lower().endswith(_STORED_SUFFIXES) else zipfile.ZIP_DEFLATED
                archive.write(path, f"payloads/{digest}/{item['name']}", compress_type=method)
            host = store.host_id()
            if include_audit and host:
                log = AuditLog(store)
                with store.lock(f"audit-{host}"):                 # a consistent copy: no append in between
                    segments = []
                    for path in log.segments(host):
                        data = path.read_bytes()
                        segments.append({"name": path.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
                        archive.writestr(f"audit/{host}/{path.name}", data)
                    head = log.head(host)
                if segments and head:
                    archive.writestr(f"audit/{host}/chain.json", json.dumps(head, sort_keys=True))
                    description["chains"].append({"host": host, "segments": segments})
            data = json.dumps(description, indent=1, sort_keys=True, ensure_ascii=False).encode("utf-8")
            archive.writestr("bundle.json", data)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    description["id"] = hashlib.sha256(data).hexdigest()
    description["path"] = str(destination)
    return description


def bundle_key(path):
    """``cev:bundle:sha256-v1`` of a bundle file: the sha256 of its ``bundle.json``."""
    from cevahir_ai.keys.grammar import Key
    with zipfile.ZipFile(path) as archive:
        return Key("bundle", "sha256-v1", hashlib.sha256(_member(archive, "bundle.json")).hexdigest())


def _member(archive, name, limit=MAX_JSON_BYTES):
    try:
        info = archive.getinfo(name)
    except KeyError:
        raise BundleError(f"the bundle lists {name} but does not contain it") from None
    if info.file_size > limit:
        raise BundleError(f"{name} is larger than {limit} bytes")
    return archive.read(info)


def read_bundle(path):
    """Open and fully verify a bundle without writing anything. Returns the verified content."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    if not zipfile.is_zipfile(path):
        raise BundleError(f"{path} is not a .cvb bundle (not a zip file)")
    with zipfile.ZipFile(path) as archive:
        raw = _member(archive, "bundle.json")
        try:
            description = json.loads(raw)
        except ValueError:
            raise BundleError("bundle.json is not JSON") from None
        if not isinstance(description, dict) or description.get("format") != BUNDLE_FORMAT:
            raise BundleError("bundle.json is not a cevahir-bundle description")
        if description.get("version") != BUNDLE_VERSION:
            raise BundleError(f"unsupported bundle version: {description.get('version')!r}")
        manifests, details, chains = {}, {}, []
        for item in description.get("manifests", ()):
            digest = str(item.get("id", ""))
            if not _HEX64.match(digest):
                raise BundleError(f"bundle.json names a manifest that is not a sha256: {digest[:40]!r}")
            data = _member(archive, f"objects/{digest}.json")
            if hashlib.sha256(data).hexdigest() != digest:
                raise BundleError(f"manifest {digest[:12]}… does not hash to its name: the bundle was modified")
            document = json.loads(data)
            if document.get("format") != MANIFEST_FORMAT or document.get("key") != item.get("key"):
                raise BundleError(f"object {digest[:12]}… is not the manifest of {item.get('key')}")
            if cjson(document) != data:
                raise BundleError(f"manifest {digest[:12]}… is not in canonical form")
            problem = inconsistent(document)
            if problem:
                raise BundleError(f"manifest {digest[:12]}… of {document['key'][:40]}… contradicts itself: {problem}")
            manifests[digest] = document
        for digest in description.get("details", ()):
            if not _HEX64.match(str(digest)):
                raise BundleError("bundle.json names a detail document that is not a sha256")
            data = _member(archive, f"tables/{digest}.json")
            if hashlib.sha256(data).hexdigest() != digest:
                raise BundleError(f"detail document {digest[:12]}… does not hash to its name")
            details[digest] = data
        payloads = []
        for item in description.get("payloads", ()):
            match = _FILE_KEY.match(str(item.get("file", "")))
            if not match:
                raise BundleError("bundle.json names a payload without a file key")
            name = safe_name(item.get("name") or "payload")
            member = f"payloads/{match.group(1)}/{name}"
            try:
                with archive.open(member) as stream:
                    actual = _sha256_stream(stream)
            except KeyError:
                raise BundleError(f"the bundle lists {member} but does not contain it") from None
            except zipfile.BadZipFile as error:
                raise BundleError(f"payload {name} is corrupted: {error}") from None
            if actual != match.group(1):
                raise BundleError(f"payload {name} is corrupted: its bytes are not {item['file'][:32]}…")
            payloads.append({"file": item["file"], "digest": match.group(1), "name": name, "member": member})
        for chain in description.get("chains", ()):
            host = str(chain.get("host", ""))
            if not HOST_RE.match(host):
                raise BundleError(f"bundle.json names a chain of an invalid host id: {host[:40]!r}")
            segments = []
            for segment in chain.get("segments", ()):
                name = str(segment.get("name", ""))
                if not SEGMENT_RE.match(name):
                    raise BundleError(f"invalid audit segment name in the bundle: {name[:40]!r}")
                data = _member(archive, f"audit/{host}/{name}")
                if hashlib.sha256(data).hexdigest() != segment.get("sha256"):
                    raise BundleError(f"audit segment {name} of {host} does not match its digest")
                segments.append((name, data))
            head = json.loads(_member(archive, f"audit/{host}/chain.json"))
            verdict = verify_chain(host, sorted(segments), head)
            if not verdict["ok"]:
                first = verdict["findings"][0]
                raise BundleError(f"the audit chain of {host} in the bundle does not verify: {first['problem']} "
                                  f"({first['segment']}, line {first['line']})")
            chains.append({"host": host, "segments": sorted(segments), "head": head, "last": verdict["last"]})
    return {"description": description, "id": hashlib.sha256(raw).hexdigest(), "manifests": manifests,
            "details": details, "payloads": payloads, "chains": chains, "path": path}


def _hashes_by_seq(segments):
    table = {}
    for _, data in segments:
        for line in data.split(b"\n"):
            if line:
                event = json.loads(line)
                table[event["seq"]] = event["hash"]
    return table


def merge_chain(store, chain):
    """Store a verified foreign chain. Returns ``"imported"``, ``"extended"``, ``"known"`` or ``"own"``.

    A copy already held must be a prefix of the incoming chain, or the other way round. Two chains
    of one host that disagree at any sequence number are a fork: the import is refused.
    """
    host = chain["host"]
    if host == store.host_id():
        return "own"
    log = AuditLog(store)
    held = [(path.name, path.read_bytes()) for path in log.segments(host)]
    state = "imported"
    if held:
        ours, theirs = _hashes_by_seq(held), _hashes_by_seq(chain["segments"])
        common = set(ours) & set(theirs)
        if any(ours[seq] != theirs[seq] for seq in common) or not common:
            raise BundleError(f"the audit chain of {host} in the bundle contradicts the copy already held (a fork)")
        if max(theirs) <= max(ours):
            return "known"
        state = "extended"
    directory = log.directory(host)
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in chain["segments"]:
        atomic_write(directory / name, data)
    atomic_write(directory / "chain.json", json.dumps(chain["head"], indent=1, sort_keys=True).encode("utf-8") + b"\n")
    return state


def extract_payload(bundle_path, payload, directory):
    """Copy one verified payload out of the bundle to ``directory``; returns its path.

    The file is named by its digest plus the suffix of its name hint, and hashed again while it is
    written: what lands on disk is what was verified.
    """
    suffix = "".join(Path(payload["name"]).suffixes[-2:])[:24]
    target = Path(directory) / payload["digest"][:2] / (payload["digest"][2:] + suffix)
    if target.is_file() and target.stat().st_size > 0:
        with open(target, "rb") as stream:
            if _sha256_stream(stream) == payload["digest"]:
                return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    try:
        with zipfile.ZipFile(bundle_path) as archive, archive.open(payload["member"]) as source, \
                open(temporary, "wb") as sink:
            for chunk in iter(lambda: source.read(CHUNK), b""):
                digest.update(chunk)
                sink.write(chunk)
        if digest.hexdigest() != payload["digest"]:
            raise BundleError(f"payload {payload['name']} changed while it was being unpacked")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
