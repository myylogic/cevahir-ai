# -*- coding: utf-8 -*-
"""Keeping a mind between sessions: save and load the whole substance with integrity checks.

A saved mind is two files: ``<name>.zihin`` (the pickled Mind: tensors, indices, alphabet, ledger, working state)
and ``<name>.zihin.json`` (manifest: sha256 of the file, digest of the substance tensors, sizes, moments lived).
``load`` refuses a file whose bytes or substance digest do not match the manifest. Pickle executes code when
loaded: load only minds you saved yourself.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Dict

from .mind import Mind

FORMAT = "zihin-mind-v1"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(mind: Mind, path: str) -> Dict[str, object]:
    """Write the mind atomically and its manifest; returns the manifest."""
    if not path.endswith(".zihin"):
        path += ".zihin"
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(mind, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    manifest = {"format": FORMAT, "file": os.path.basename(path), "sha256": _sha256(path),
                "substance_digest": mind.S.digest(), "moments": mind.t, **mind.size()}
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load(path: str) -> Mind:
    """Read a mind saved by ``save``; verifies the file hash and the substance digest."""
    if not path.endswith(".zihin"):
        path += ".zihin"
    with open(path + ".json", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format") != FORMAT:
        raise ValueError("not a %s manifest" % FORMAT)
    if _sha256(path) != manifest["sha256"]:
        raise ValueError("file does not match its manifest (sha256)")
    with open(path, "rb") as f:
        mind = pickle.load(f)
    if mind.S.digest() != manifest["substance_digest"]:
        raise ValueError("substance digest does not match the manifest")
    return mind
