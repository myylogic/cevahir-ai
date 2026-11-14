"""``Store``: one ``CEVAHIR_HOME``, its host identity and its content-addressed objects.

An object is a JSON document stored under the sha256 of its CJSON-1 bytes. Writing the same
document twice gives the same file, so writers on different machines cannot disagree, and a file
that no longer hashes to its name is reported, never used.
"""
import hashlib
import json
import math
import os
import platform
from pathlib import Path

from cevahir_ai._compat.clock import utc_now
from cevahir_ai._compat.fsio import atomic_write
from cevahir_ai._compat.home import cevahir_home, host_file
from cevahir_ai._compat.locks import FileLock, lock_path
from cevahir_ai.errors import NotFound, StoreError
from cevahir_ai.keys.canonical import cjson
from cevahir_ai.keys.grammar import Key

MANIFEST_FORMAT, MANIFEST_VERSION = "cevahir-manifest", 1
PROVENANCE = ("native", "legacy-import", "imported", "migrated", "asserted")
AREAS = ("objects", "tables")
_HEX = set("0123456789abcdef")


def tool_name():
    from cevahir_ai import __version__
    return f"cevahir-ai {__version__}"


def sanitize(value):
    """Data as CJSON-1 can encode it: a NaN metric or a 70-bit integer becomes text instead of an error."""
    if isinstance(value, dict):
        return {str(name): sanitize(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, int) and not isinstance(value, bool) and not -2**63 <= value < 2**63:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def manifest_key(document):
    """``cev:manifest:sha256-v1`` of a manifest document (sha256 of its CJSON-1 bytes)."""
    return Key("manifest", "sha256-v1", hashlib.sha256(cjson(document)).hexdigest())


def new_manifest(kind, key, *, members=None, model=None, aliases=(), inputs=(), payload=(), attributes=None,
                 explain=None, provenance="legacy-import", producer=None, created=None):
    """A manifest: the portable statement about one artefact key. Paths never appear in it.

    ``inputs`` are the keys that produced the artefact (``[{"role", "key"}]``): they are its
    lineage and travel with it. ``payload`` lists the files that hold its bytes, by file key.
    """
    key = Key.parse(str(key))
    if provenance not in PROVENANCE:
        raise ValueError(f"unknown provenance: {provenance!r}")
    document = {
        "format": MANIFEST_FORMAT, "version": MANIFEST_VERSION, "kind": str(kind), "key": str(key),
        "members": {name: str(Key.parse(str(value))) for name, value in sorted((members or {}).items()) if value},
        "model": None if model is None else str(Key.parse(str(model))),
        "aliases": sorted({str(Key.parse(str(alias))) for alias in aliases} - {str(key)}),
        "inputs": [{"role": str(item["role"]), "key": str(Key.parse(str(item["key"])))} for item in inputs],
        "payload": [sanitize(dict(item, file=str(Key.parse(str(item["file"]))))) for item in payload],
        "attributes": sanitize(attributes or {}), "explain": sanitize(explain or {}),
        "provenance": provenance, "producer": dict(producer or {}), "created": created or utc_now(),
    }
    cjson(document)                      # a manifest must be canonical data; raises CanonicalError otherwise
    return document


def inconsistent(document):
    """Why a manifest's key cannot be the key of what it states, or None.

    The key of a checkpoint and of a model is derived from its members, so a manifest that names
    one key and states other weights is a forgery (or a defect) and is refused. Keys of other
    kinds are derived from payload bytes and cannot be recomputed from the manifest alone.
    """
    from cevahir_ai.keys.schemes.ckpt import CKPT_MEMBERS, ckpt_members_key, model_triple_key
    members, key = document.get("members") or {}, document.get("key")
    try:
        if document.get("kind") == "ckpt" and str(key).startswith("cev:ckpt:members-v1:"):
            expected = ckpt_members_key({name: members.get(name) for name in CKPT_MEMBERS})
        elif document.get("kind") == "model" and str(key).startswith("cev:model:triple-v1:"):
            expected = model_triple_key(members.get("weights"), members.get("arch"), members.get("tok"))
        else:
            return None
    except (ValueError, TypeError) as error:
        return f"its members do not form a key ({error})"
    return None if str(expected) == key else "its key is not the key of its members"


def statement(document):
    """The manifest without who recorded it and when: equal statements are the same registration."""
    return {name: value for name, value in document.items() if name not in ("created", "producer")}


class Store:
    def __init__(self, home=None, *, environ=None, clock=None):
        self.environ = os.environ if environ is None else environ
        self.home = Path(home).expanduser().resolve() if home is not None else cevahir_home(self.environ)
        self.clock = clock or utc_now
        self._host = None
        self._catalog = None

    # ------------------------------------------------------------------------------------ host
    def lock(self, name, timeout=30.0):
        return FileLock(lock_path(self.home, name), timeout=timeout)

    def host_id(self, create=False):
        """This machine's id (``h-<16 hex>``), or None when the home has none yet and ``create`` is false."""
        if self._host:
            return self._host
        path = host_file(self.home, self.environ)
        if not path.is_file():
            if not create:
                return None
            with self.lock("host"):
                if not path.is_file():
                    document = {"format": "cevahir-host", "version": 1, "host_id": "h-" + os.urandom(8).hex(),
                                "created": self.clock(), "platform": platform.system()}
                    atomic_write(path, json.dumps(document, indent=1, sort_keys=True).encode("utf-8") + b"\n")
        host = json.loads(path.read_text(encoding="utf-8")).get("host_id", "")
        if not (host.startswith("h-") and len(host) == 18 and set(host[2:]) <= _HEX):
            raise StoreError(f"{path} does not hold a host id")
        self._host = host
        return host

    # --------------------------------------------------------------------------------- objects
    def object_path(self, digest, area="objects"):
        if area not in AREAS or len(digest) != 64 or not set(digest) <= _HEX:
            raise ValueError(f"not an object id: {area}/{digest}")
        return self.home / area / digest[:2] / f"{digest[2:]}.json"

    def put(self, document, area="objects"):
        """Store ``document`` under the sha256 of its canonical bytes; returns the hex id."""
        data = cjson(document)
        digest = hashlib.sha256(data).hexdigest()
        path = self.object_path(digest, area)
        if not path.is_file():
            atomic_write(path, data)
            self._catalog = None
        return digest

    def get(self, digest, area="objects"):
        path = self.object_path(digest, area)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise NotFound(f"no object {digest[:12]}… in {area}/ of {self.home}") from None
        if hashlib.sha256(data).hexdigest() != digest:
            raise StoreError(f"object {area}/{digest[:12]}… does not hash to its name: the file was modified")
        return json.loads(data)

    def object_ids(self, area="objects"):
        base = self.home / area
        if not base.is_dir():
            return []
        return sorted(folder.name + path.stem for folder in base.iterdir() if folder.is_dir() and len(folder.name) == 2
                      for path in folder.glob("*.json") if len(path.stem) == 62)

    # --------------------------------------------------------------------------------- catalog
    def catalog(self):
        """The artefacts of this home, read from the manifests (cached until the next write)."""
        if self._catalog is None:
            from cevahir_ai.store.catalog import Catalog
            self._catalog = Catalog(self)
        return self._catalog

    def invalidate(self):
        self._catalog = None
