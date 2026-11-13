"""File keys. Deliberately not domain-separated, so they equal ``sha256sum``, the ``.sha256``
sidecars of the checkpoint saver and the book's ``utf8-lf`` fingerprints."""
from pathlib import Path

from cevahir_ai._compat.fsio import sha256_file
from cevahir_ai.keys.grammar import Key
from cevahir_ai.keys.text import is_textual, utf8lf_digest


def file_bytes_key(path):
    return Key("file", "bytes-v1", sha256_file(path))


def file_utf8lf_key(path):
    return Key("file", "utf8lf-v1", utf8lf_digest(path))


def file_key(path, scheme=None):
    """Key of a file: text suffixes use TEXT-LF-1 (raw bytes when the file is not UTF-8), others bytes."""
    if scheme == "bytes-v1" or (scheme is None and not is_textual(path)):
        return file_bytes_key(path)
    if scheme not in (None, "utf8lf-v1"):
        raise ValueError(f"unknown file scheme: {scheme}")
    try:
        return file_utf8lf_key(path)
    except UnicodeDecodeError:
        if scheme is None:
            return file_bytes_key(path)
        raise


def read_sidecar(path):
    """The digest stored in ``<path>.sha256`` (first token), or None when there is no sidecar."""
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        return None
    parts = sidecar.read_text(encoding="utf-8", errors="replace").split()
    return parts[0].lower() if parts else ""
