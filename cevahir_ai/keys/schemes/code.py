"""Code and environment keys."""
import platform
import sys
from pathlib import Path

from cevahir_ai.keys.canonical import structured_digest
from cevahir_ai.keys.grammar import Key
from cevahir_ai.keys.text import utf8lf_digest

ENV_PACKAGES = ("torch", "numpy", "psutil", "tqdm", "python-docx")


def code_tree_key(files):
    """``files``: iterable of ``[posix path, utf8-lf sha256]``; order does not matter."""
    payload = {"files": sorted([str(path), str(digest)] for path, digest in files)}
    return Key("code", "tree-v1", structured_digest("code", "tree-v1", payload))


def code_tree_key_of(root, paths):
    """Key over files under ``root`` (paths relative to it). CRLF and LF checkouts give one key."""
    root = Path(root)
    files = [[Path(path).as_posix(), utf8lf_digest(root / path)] for path in paths]
    return code_tree_key(files), sorted(files)


def env_rt_key(packages=ENV_PACKAGES):
    """Runtime identity from ``importlib.metadata`` (torch is not imported). No host or user names."""
    from importlib import metadata
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    payload = {"python": ".".join(str(part) for part in sys.version_info[:3]),
               "implementation": platform.python_implementation(), "platform": platform.system(),
               "machine": platform.machine(), "packages": versions}
    return Key("env", "rt-v1", structured_digest("env", "rt-v1", payload)), payload
