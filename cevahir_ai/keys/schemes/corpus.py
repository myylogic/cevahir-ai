"""Corpus and evaluation-set keys: the sorted list of (relative path, file key).

Text documents are hashed as TEXT-LF-1, so a corpus prepared on Windows (CRLF) and the same corpus
on Colab (LF) have one key; paths are relative to the corpus root, so the working directory and
the absolute location do not matter. An empty or missing directory is an error (the legacy digest
returned the empty string and accepted it as a key component).
"""
from pathlib import Path

from cevahir_ai.keys.canonical import structured_digest
from cevahir_ai.keys.grammar import Key
from cevahir_ai.keys.schemes.file import file_bytes_key, file_key
from cevahir_ai.keys.text import nfc_posix_relpath

CORPUS_SUFFIXES = (".docx", ".json", ".txt")


def _files_payload(directory, suffixes):
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus directory does not exist: {root}")
    wanted = {suffix.lower() for suffix in suffixes}
    files = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in wanted:
            key = file_bytes_key(path) if path.suffix.lower() == ".docx" else file_key(path)
            files.append([nfc_posix_relpath(path, root), str(key)])
    if not files:
        raise ValueError(f"corpus directory holds no {sorted(wanted)} file: {root}")
    return {"selector": {"suffixes": sorted(wanted)}, "files": sorted(files)}


def corpus_files_key(directory, suffixes=CORPUS_SUFFIXES):
    """Returns ``(key, payload)``; the payload lists every file with its own key."""
    payload = _files_payload(directory, suffixes)
    return Key("corpus", "files-v1", structured_digest("corpus", "files-v1", payload)), payload


def evalset_files_key(directory, suffixes=CORPUS_SUFFIXES):
    payload = _files_payload(directory, suffixes)
    return Key("evalset", "files-v1", structured_digest("evalset", "files-v1", payload)), payload
