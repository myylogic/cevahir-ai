"""Legacy tokenizer identities: ``tokenizer_digest`` (identity_version 3) mapped to canonical keys.

The legacy value is always computed by the original function. It wrongly depends on the
``read_only`` flag and on CRLF vs LF bytes of ``merges.txt``; ``legacy_candidates`` lets the
original function compute every such variant, so a checkpoint saved by a writable core on Windows
is recognised by a read-only core on Linux instead of being called a mismatch.
"""
import json
import os
import tempfile
from pathlib import Path

from cevahir_ai._compat.repo import import_legacy, repo_root
from cevahir_ai.keys.grammar import Key
from cevahir_ai.keys.schemes.tok import tok_bpe_sem_key, tok_bytebpe_key

REGISTRY_PATH = ("governance", "registry", "tokenizers.json")


def _digest(core):
    return import_legacy("training_system.cache_identity").tokenizer_digest(core)


def tok_legacy_v3_key(core):
    """``tokenizer_digest(core)`` of a live ``TokenizerCore``, called unchanged."""
    return Key("tok", "legacy-v3", _digest(core))


def _parts(core):
    manager = getattr(core, "tokenizer", None)
    merges = getattr(core, "merges_path", None) or getattr(manager, "merges_file", None)
    return manager, merges, dict(getattr(manager, "config", None) or {})


def canonical_key(tokenizer):
    """Canonical key of a live tokenizer: ``tok:bytebpe-v1`` or ``tok:bpe-sem-v1``. Returns ``(key, explain)``."""
    if hasattr(tokenizer, "identity") and callable(tokenizer.identity) and hasattr(tokenizer, "merges"):
        return tok_bytebpe_key(tokenizer), {"family": "cevahir-byte-bpe", "vocab_size": int(tokenizer.vocab_size)}
    _, merges, config = _parts(tokenizer)
    if not merges:
        raise ValueError("tokenizer has no merges file; it is not a legacy BPE core")
    return tok_bpe_sem_key(tokenizer.get_vocab(), merges, config)


class _Variant:
    """Stand-in handed to the original ``tokenizer_digest``: same vocab, one flag or newline form changed."""

    class _Manager:
        def __init__(self, config, merges_file):
            self.config, self.merges_file = config, merges_file

    def __init__(self, vocab, config, merges_path):
        self._vocab, self.merges_path = vocab, merges_path
        self.tokenizer = self._Manager(config, merges_path)

    def get_vocab(self):
        return self._vocab


def legacy_candidates(core):
    """Legacy digests of the same tokenizer under read_only in {True, False, absent} x merges as {LF, CRLF}.

    A writable core built without the flag has no ``read_only`` key at all in its manager config
    (that is the production training case), which hashes differently from ``read_only=False``.
    Returns ``{hex: "read_only=…, merges=…"}``. Every value is produced by the original function.
    """
    _, merges, config = _parts(core)
    if not merges:
        return {_digest(core): "as is"}
    vocab = core.get_vocab()
    # Only the newline form varies. A BOM stays where it is: the legacy reader keeps it in the first
    # merge, so the same merges without it are another tokenizer, not a variant of this one.
    lf = Path(merges).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    forms = {"LF": lf, "CRLF": lf.replace(b"\n", b"\r\n")}
    candidates = {}
    with tempfile.TemporaryDirectory(prefix="cevahir-tok-") as directory:
        for newline, data in forms.items():
            path = os.path.join(directory, f"merges-{newline}.txt")
            with open(path, "wb") as stream:
                stream.write(data)
            for read_only in (True, False, "absent"):
                variant_config = {key: value for key, value in config.items() if key != "read_only"}
                if read_only != "absent":
                    variant_config["read_only"] = read_only
                variant = _Variant(vocab, variant_config, path)
                candidates.setdefault(_digest(variant), f"read_only={read_only}, merges={newline}")
    return candidates


def reconcile(saved_hex, core):
    """Compare a saved legacy digest with a live core.

    Returns ``(status, detail)``: ``"exact"`` (the live digest), ``"drift"`` (same tokenizer, the
    digest differs only by read_only or newline form), or ``"mismatch"``.
    """
    live = _digest(core)
    if saved_hex == live:
        return "exact", "tokenizer_digest equal"
    candidates = legacy_candidates(core)
    if saved_hex in candidates:
        return "drift", f"legacy digest drift ({candidates[saved_hex]}); same vocabulary, merges and semantics"
    return "mismatch", "no variant of the current tokenizer reproduces the saved digest"


def tracked_entry(name, vocab_path, merges_path, root=None):
    """Registry entry of a tracked legacy tokenizer, computed from the files with the live code.

    Records the canonical key, the file keys of both files and every legacy digest the original
    function gives for this tokenizer (read_only x newline form), as aliases of the canonical key.
    """
    from cevahir_ai.keys.schemes.file import file_utf8lf_key
    root = Path(root or repo_root())
    core_class = import_legacy("tokenizer_management.core.tokenizer_core").TokenizerCore
    core = core_class({"vocab_path": str(root / vocab_path), "merges_path": str(root / merges_path),
                       "read_only": True, "use_gpu": False})
    key, explain = canonical_key(core)
    return {
        "name": name, "family": explain["family"], "key": str(key),
        "vocab_size": explain["vocab_size"], "merges": explain["merges"],
        "files": {Path(vocab_path).as_posix(): str(file_utf8lf_key(root / vocab_path)),
                  Path(merges_path).as_posix(): str(file_utf8lf_key(root / merges_path))},
        "aliases": [{"key": f"cev:tok:legacy-v3:{digest}", "how": how}
                    for digest, how in sorted(legacy_candidates(core).items(), key=lambda item: item[1])],
    }


def tracked_registry(root=None):
    """``governance/registry/tokenizers.json``: tracked tokenizers with their canonical key and legacy aliases."""
    path = Path(root or repo_root()).joinpath(*REGISTRY_PATH)
    return json.loads(path.read_text(encoding="utf-8"))


_UNCHANGED = {}     # (path, size, mtime_ns) -> utf8-lf key; the tracked vocabulary is several megabytes


def tracked_files_changed(name, root=None):
    """Files of the tracked tokenizer ``name`` that are no longer the files the registry recorded.

    The registry pins each file by its utf8-lf key, which ignores line endings (they do not change
    the tokenizer) and a BOM (which does), so a BOM counts as a change of its own. An empty list
    means: the tokenizer files in this checkout are the tracked tokenizer.
    """
    from cevahir_ai.keys.schemes.file import file_utf8lf_key
    root = Path(root or repo_root())
    entry = next((entry for entry in tracked_registry(root)["tokenizers"] if entry["name"] == name), None)
    if entry is None:
        raise KeyError(f"no tracked tokenizer named {name!r}")
    changed = []
    for relative, recorded in entry["files"].items():
        path = root / relative
        if not path.is_file():
            changed.append(relative)
            continue
        stat = path.stat()
        token = (str(path), stat.st_size, stat.st_mtime_ns)
        if token not in _UNCHANGED:
            with open(path, "rb") as stream:
                bom = stream.read(3) == b"\xef\xbb\xbf"
            _UNCHANGED[token] = None if bom else str(file_utf8lf_key(path))
        if _UNCHANGED[token] != recorded:
            changed.append(relative)
    return sorted(changed)


def lookup_legacy(saved_hex, root=None):
    """The tracked tokenizer whose legacy aliases contain ``saved_hex``, or None."""
    try:
        registry = tracked_registry(root)
    except FileNotFoundError:
        return None
    for entry in registry["tokenizers"]:
        for alias in entry.get("aliases", ()):
            if alias["key"] == f"cev:tok:legacy-v3:{saved_hex}":
                return {"name": entry["name"], "key": entry["key"], "how": alias.get("how", "")}
    return None
