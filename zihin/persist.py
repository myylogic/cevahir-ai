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
import shutil
import time
from typing import Dict

from .mind import Mind
from .store import DiskLog

FORMAT = "zihin-mind-v1"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(mind: Mind, path: str) -> Dict[str, object]:
    """Write the mind atomically and its manifest; returns the manifest. A disk log is copied beside it."""
    if not path.endswith(".zihin"):
        path += ".zihin"
    log = mind.S.log if isinstance(mind.S.log, DiskLog) else None
    live_path = log.path if log else None
    log_sha = None
    if log:
        if os.path.exists(live_path):
            shutil.copyfile(live_path, path + ".log")
        else:
            open(path + ".log", "wb").close()
        log_sha = _sha256(path + ".log")
        log.path = os.path.basename(path) + ".log"          # stored relative to the saved mind
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(mind, f, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        if log:
            log.path = live_path
    os.replace(tmp, path)
    manifest = {"format": FORMAT, "file": os.path.basename(path), "sha256": _sha256(path),
                "substance_digest": mind.S.digest(), "moments": mind.t, "log_sha256": log_sha, **mind.size()}
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def export_portable(mind: Mind, path: str) -> Dict[str, object]:
    """A copy for conversation only: the same substance without the experience ledger (much smaller file).

    It learns while you talk to it like any mind, but single experiences can no longer be retracted (the ledger
    that makes retraction exact stays with the original mind)."""
    twin = mind.clone()                                    # clone drops the ledger
    twin.S.log, twin.S.keep_log = {}, False
    twin.cfg.keep_log, twin.cfg.log_path = False, None
    return save(twin, path)


def load(path: str, live_log: str = None, ledger: bool = True) -> Mind:
    """Read a mind saved by ``save``; verifies the file hash and the substance digest.

    ``ledger`` False: the mind is loaded for conversation without its experience ledger (nothing is copied;
    single experiences cannot be retracted in this session). Otherwise the saved ledger is copied to a live file
    that is unique to this load, so two loads of the same saved mind never write into the same file."""
    if path.endswith(".gz"):                               # the compressed file itself was named
        path = path[:-3]
    if not path.endswith(".zihin"):
        path += ".zihin"
    if not os.path.exists(path) and os.path.exists(path + ".gz"):   # a compressed mind: unpack it once
        import gzip
        with gzip.open(path + ".gz", "rb") as src, open(path + ".tmp", "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        os.replace(path + ".tmp", path)
    manifest = {}
    if os.path.exists(path + ".json"):
        with open(path + ".json", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("format") != FORMAT:
            raise ValueError("not a %s manifest" % FORMAT)
        if _sha256(path) != manifest["sha256"]:
            raise ValueError("file does not match its manifest (sha256)")
    else:                                                  # a mind file that travelled without its manifest
        print("  (uyarı: %s.json bulunamadı; dosyanın bütünlüğü doğrulanamadı)" % os.path.basename(path), flush=True)
    with open(path, "rb") as f:
        mind = pickle.load(f)
    if manifest and mind.S.digest() != manifest["substance_digest"]:
        raise ValueError("substance digest does not match the manifest")
    for name, default in (("trace", list), ("value_sign", dict), ("dreams", list)):   # minds saved before these existed
        if not hasattr(mind, name):
            setattr(mind, name, default())
    import dataclasses
    for f in dataclasses.fields(type(mind.cfg)):                      # settings added after the mind was saved
        if f.name not in mind.cfg.__dict__:
            value = f.default if f.default is not dataclasses.MISSING else f.default_factory()
            setattr(mind.cfg, f.name, value)
    if mind.cfg.support == "members" and mind.cfg.bind_repeats:      # a speaking mind reads with today's rules
        mind.cfg.backoff = True
        mind.cfg.resemblance = True
        mind.cfg.race = True
        mind.cfg.situations = True
    if not ledger:
        mind.S.log, mind.S.keep_log = {}, False
        mind.cfg.keep_log, mind.cfg.log_path = False, None
        return mind
    if isinstance(mind.S.log, DiskLog):
        saved_log = os.path.join(os.path.dirname(os.path.abspath(path)), mind.S.log.path)
        if manifest.get("log_sha256") and _sha256(saved_log) != manifest["log_sha256"]:
            raise ValueError("experience log does not match the manifest")
        live = live_log or "%s.live-%d-%d" % (saved_log, os.getpid(), time.time_ns())
        shutil.copyfile(saved_log, live)                    # the saved ledger is never written to
        mind.S.log.path = live
    return mind
