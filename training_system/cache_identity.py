"""Shared content identities for cache preparation and training consumption."""
import hashlib
import json
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_directory_digest(directory):
    root = Path(directory)
    if not root.exists():
        return ""
    entries = [(path.relative_to(root).as_posix(), file_digest(path))
               for path in sorted(root.rglob("*"))
               if path.is_file() and path.suffix.lower() in {".json", ".txt", ".docx"}]
    if not entries:
        return ""
    return hashlib.sha256(json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def tokenizer_digest(tokenizer_core):
    manager = getattr(tokenizer_core, "tokenizer", None)
    merges_path = getattr(tokenizer_core, "merges_path", None) or getattr(manager, "merges_file", None)
    payload = {
        "identity_version": 2,
        "vocab": tokenizer_core.get_vocab(),
        "merges": file_digest(merges_path) if merges_path else None,
        "bpe_config": getattr(manager, "config", {}),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()
