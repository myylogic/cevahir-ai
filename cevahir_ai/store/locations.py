"""Locations: where the payload files of artefacts are, per host.

A manifest names a payload by its file key and never by a path. Paths are host-local facts kept
in ``locations/<host-id>.json``: ``{file key: [{path, size, mtime_ns, seen}]}``. Two byte-identical
copies (``last.pth`` and ``best.pth``) have one file key, hence one artefact with two locations.
Each host writes only its own file, so a home shared through a synchronised drive never has two
writers on one file.
"""
import json
import os
from pathlib import Path

from cevahir_ai._compat.fsio import atomic_write
from cevahir_ai.keys.schemes.file import file_bytes_key

FORMAT = "cevahir-locations"


def _path(store, host):
    return store.home / "locations" / f"{host}.json"


def _load(store, host):
    path = _path(store, host)
    if not path.is_file():
        return {"format": FORMAT, "version": 1, "host": host, "files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def absolute(path):
    """The spelling under which a path is recorded: absolute, without resolving symlinks or drives."""
    return os.path.abspath(os.fspath(path))


def add_location(store, file_key, path):
    """Record that ``path`` on this host holds the file ``file_key``. Returns True when it is new."""
    host, text = store.host_id(create=True), absolute(path)
    stat = os.stat(text)
    entry = {"path": text, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "seen": store.clock()}
    with store.lock(f"locations-{host}"):
        document = _load(store, host)
        new = not any(item["path"] == text for item in document["files"].get(str(file_key), ()))
        for files in document["files"].values():       # a path holds one file: drop what it held before
            files[:] = [item for item in files if item["path"] != text]
        document["files"].setdefault(str(file_key), []).append(entry)
        document["files"] = {key: value for key, value in document["files"].items() if value}
        atomic_write(_path(store, host), json.dumps(document, indent=1, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return new


def locations_of(store, file_key, host=None):
    """Recorded locations of a file on ``host`` (default: this host)."""
    host = host or store.host_id()
    if host is None:
        return []
    return list(_load(store, host)["files"].get(str(file_key), ()))


def known_file_key(store, path):
    """The recorded file key of ``path`` when size and mtime are unchanged since it was hashed, else None.

    Only a shortcut for listing and re-registration: integrity checks always hash the bytes.
    """
    host = store.host_id()
    if host is None:
        return None
    text = absolute(path)
    try:
        stat = os.stat(text)
    except OSError:
        return None
    for file_key, entries in _load(store, host)["files"].items():
        for entry in entries:
            if entry["path"] == text and entry["size"] == stat.st_size and entry["mtime_ns"] == stat.st_mtime_ns:
                return file_key
    return None


def file_key_of(store, path):
    """File key of ``path``: the recorded one when the file is unchanged, else computed."""
    return known_file_key(store, path) or str(file_bytes_key(path))


def live_locations(store, file_key):
    """Recorded locations whose file still exists, each with ``changed`` (size or mtime differs)."""
    live = []
    for entry in locations_of(store, file_key):
        try:
            stat = os.stat(entry["path"])
        except OSError:
            continue
        changed = stat.st_size != entry["size"] or stat.st_mtime_ns != entry["mtime_ns"]
        live.append(dict(entry, changed=changed))
    return live


def find_file(store, file_key):
    """A path on this host that holds ``file_key`` (unchanged entries first), or None."""
    live = sorted(live_locations(store, file_key), key=lambda entry: entry["changed"])
    return Path(live[0]["path"]) if live else None
