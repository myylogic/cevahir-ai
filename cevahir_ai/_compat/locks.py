"""Cross-process file locks: ``msvcrt.locking`` on Windows, ``fcntl.flock`` elsewhere.

Lock files live on local disk (``<temp>/cevahir-locks/<hash of the home>/``), never inside
``CEVAHIR_HOME``: a home on a synchronised or network drive (Google Drive, OneDrive) does not give
reliable locks. Each host appends only to its own audit chain, so no lock ever spans machines.
"""
import hashlib
import os
import tempfile
import time
from pathlib import Path

try:
    import msvcrt
except ImportError:                      # POSIX
    msvcrt = None
    import fcntl


def lock_path(home, name):
    """Lock file for ``name`` of the home at ``home``. The directory is created on demand."""
    token = hashlib.sha256(str(Path(home)).encode("utf-8")).hexdigest()[:16]
    directory = Path(os.environ.get("CEVAHIR_LOCK_DIR") or Path(tempfile.gettempdir()) / "cevahir-locks") / token
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}.lock"


class FileLock:
    """Exclusive lock on a lock file, held for the ``with`` block. Waits up to ``timeout`` seconds.

    Two ``FileLock`` objects on one path exclude each other across processes and across threads
    (each holds its own file handle). The lock file itself is never removed.
    """

    def __init__(self, path, timeout=30.0, poll=0.01):
        self.path, self.timeout, self.poll = Path(path), timeout, poll
        self._handle = None

    def _try(self):
        if msvcrt is not None:
            self._handle.seek(0)
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a+b")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self._try()
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    self._handle.close()
                    self._handle = None
                    raise TimeoutError(f"could not lock {self.path} within {self.timeout} s") from None
                time.sleep(self.poll)

    def __exit__(self, *exc):
        try:
            if msvcrt is not None:
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None
        return False
