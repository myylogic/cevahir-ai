"""Atomic writes and streaming digests."""
import hashlib
import os
import time
from pathlib import Path

CHUNK = 1024 * 1024


def sha256_file(path):
    """sha256 of the raw bytes, streamed in 1 MiB chunks (equal to ``sha256sum``)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, data):
    """Write bytes through a sibling temp file, fsync and ``os.replace`` (retried on Windows locks)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.part")
    with tmp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(6):
        try:
            os.replace(tmp, path)
            return path
        except PermissionError:
            if attempt == 5:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.025 * (attempt + 1))
    return path
