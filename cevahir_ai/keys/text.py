"""TEXT-LF-1: the newline policy of text keys, and path names inside payloads.

A text file is decoded as ``utf-8-sig`` (a leading BOM is dropped), ``\\r\\n`` and a lone ``\\r``
become ``\\n``, and the result is encoded as UTF-8. This is exactly ``scripts/check_book.py``'s
``fingerprint`` for its textual suffixes, so a Windows CRLF checkout and a Linux LF checkout of the
same file have the same key.
"""
import hashlib
import unicodedata
from pathlib import Path, PurePosixPath

TEXTUAL_SUFFIXES = frozenset({".py", ".md", ".json", ".yaml", ".yml", ".txt"})


def text_lf(data):
    """TEXT-LF-1 bytes of raw file bytes. Raises ``UnicodeDecodeError`` when the file is not UTF-8."""
    text = data.decode("utf-8-sig")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def utf8lf_digest(path):
    return hashlib.sha256(text_lf(Path(path).read_bytes())).hexdigest()


def is_textual(path):
    return Path(path).suffix.lower() in TEXTUAL_SUFFIXES


def nfc_posix_relpath(path, root):
    """Path of ``path`` relative to ``root``: POSIX separators, NFC (macOS returns NFD names)."""
    relative = Path(path).resolve().relative_to(Path(root).resolve())
    return unicodedata.normalize("NFC", PurePosixPath(*relative.parts).as_posix())
