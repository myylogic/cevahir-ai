# -*- coding: utf-8 -*-
"""Keeping a mind between sessions: save and load the whole substance with integrity checks.

A saved mind is two files: ``<name>.zihin`` (the pickled Mind: tensors, indices, alphabet, ledger, working state)
and ``<name>.zihin.json`` (manifest: sha256 of the file, digest of the substance tensors, sizes, moments lived).
``load`` refuses a file whose bytes or substance digest do not match the manifest. Pickle executes code when
loaded: load only minds you saved yourself.

A save never leaves a mind that does not load: its new files are written beside the old ones (``.tmp``), the
manifest last, and only then put in place. Stopped before its manifest was whole (Ctrl+C, a crash, a killed thread),
the save leaves the mind saved before untouched, and its own files are removed (at once when it is stopped by an
exception, else by the next ``load`` or ``save`` of that mind); stopped while its files were being put in place, it is
finished by the next ``load`` or ``save`` of that mind (``_finish_save``).
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


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _finish_save(path: str) -> None:
    """Make what an earlier save of ``path`` left whole again, before the mind is read or saved.

    A save stopped while its files were being put in place left its whole manifest as ``.json.tmp``: when the mind
    file (and the ledger) that manifest names are there, still as ``.tmp`` or already in place, they are put in place
    and the manifest last. Anything else a save left (a ``.tmp`` without a whole manifest, or files that are not the
    ones its manifest names) belongs to a save that did not get that far: it is removed, and the mind saved before
    stays as it was."""
    pending = path + ".json.tmp"
    leftovers = (path + ".tmp", path + ".log.tmp", pending)
    if not os.path.exists(pending):
        for leftover in leftovers[:2]:
            _remove(leftover)
        return
    try:
        with open(pending, encoding="utf-8") as f:
            manifest = json.load(f)
    except ValueError:
        manifest = None

    def named(new: str, final: str, key: str) -> bool:      # the file that manifest names: new, or already in place
        found = new if os.path.exists(new) else final
        return os.path.exists(found) and _sha256(found) == manifest.get(key)
    whole = isinstance(manifest, dict) and manifest.get("format") == FORMAT and named(path + ".tmp", path, "sha256") \
        and (not manifest.get("log_sha256") or named(path + ".log.tmp", path + ".log", "log_sha256"))
    if not whole:
        for leftover in leftovers:
            _remove(leftover)
        return
    for tmp, final in ((path + ".log.tmp", path + ".log"), (path + ".tmp", path)):
        if os.path.exists(tmp):
            os.replace(tmp, final)
    os.replace(pending, path + ".json")


def save(mind: Mind, path: str) -> Dict[str, object]:
    """Write the mind and its manifest; returns the manifest. A disk log is copied beside it. The new files are
    written as ``.tmp`` beside the old ones, the manifest last, and then put in place (see the module's docstring):
    a save that does not finish leaves the mind saved before as it was, and no file of its own."""
    if not path.endswith(".zihin"):
        path += ".zihin"
    _finish_save(path)                                     # a save of this mind stopped earlier is finished first
    log = mind.S.log if isinstance(mind.S.log, DiskLog) else None
    live_path = log.path if log else None
    log_sha = None
    tmp, manifest_tmp = path + ".tmp", path + ".json.tmp"
    try:
        if log:
            if os.path.exists(live_path):
                shutil.copyfile(live_path, path + ".log.tmp")
            else:
                open(path + ".log.tmp", "wb").close()
            log_sha = _sha256(path + ".log.tmp")
            log.path = os.path.basename(path) + ".log"      # stored relative to the saved mind
        try:
            with open(tmp, "wb") as f:
                pickle.dump(mind, f, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            if log:
                log.path = live_path
        manifest = {"format": FORMAT, "file": os.path.basename(path), "sha256": _sha256(tmp),
                    "substance_digest": mind.S.digest(), "moments": mind.t, "log_sha256": log_sha, **mind.size()}
        with open(manifest_tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except BaseException:                                  # stopped before its files were whole: nothing is left
        for leftover in (tmp, manifest_tmp, path + ".log.tmp"):
            _remove(leftover)
        raise
    if log:                                                # whole: put in place, the manifest last
        os.replace(path + ".log.tmp", path + ".log")
    os.replace(tmp, path)
    os.replace(manifest_tmp, path + ".json")
    return manifest


def export_portable(mind: Mind, path: str) -> Dict[str, object]:
    """A copy for conversation only: the same substance without the experience ledger (much smaller file).

    It learns while you talk to it like any mind, but single experiences can no longer be retracted (the ledger
    that makes retraction exact stays with the original mind)."""
    held = (mind.S.log, mind.S.keep_log, mind.cfg.keep_log, mind.cfg.log_path)
    mind.S.log, mind.S.keep_log = {}, False                # the ledger is set aside while the substance is written:
    mind.cfg.keep_log, mind.cfg.log_path = False, None     # no copy of a large mind is made (memory stays flat)
    try:
        return save(mind, path)
    finally:
        mind.S.log, mind.S.keep_log, mind.cfg.keep_log, mind.cfg.log_path = held


def load(path: str, live_log: str = None, ledger: bool = True) -> Mind:
    """Read a mind saved by ``save``; verifies the file hash and the substance digest.

    ``ledger`` False: the mind is loaded for conversation without its experience ledger (nothing is copied;
    single experiences cannot be retracted in this session). Otherwise the saved ledger is copied to a live file
    that is unique to this load, so two loads of the same saved mind never write into the same file."""
    if path.endswith(".gz"):                               # the compressed file itself was named
        path = path[:-3]
    if not path.endswith(".zihin"):
        path += ".zihin"
    _finish_save(path)                                     # a save stopped while its files were put in place
    if not os.path.exists(path) and os.path.exists(path + ".gz"):   # a compressed mind: unpack it once
        import gzip
        try:
            with gzip.open(path + ".gz", "rb") as src, open(path + ".gz.tmp", "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
        except BaseException:
            _remove(path + ".gz.tmp")
            raise
        os.replace(path + ".gz.tmp", path)
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
    for name, default in (("trace", list), ("value_sign", dict), ("dreams", list), ("unknown", float),
                          ("voice", dict), ("dil", lambda: "tr"), ("witnesses", list)):  # minds saved before these existed (their text
        if not hasattr(mind, name):                                 # was lower-cased by Turkish rules)
            setattr(mind, name, default())
    import dataclasses
    for f in dataclasses.fields(type(mind.cfg)):                      # settings added after the mind was saved
        if f.name not in mind.cfg.__dict__:
            value = f.default if f.default is not dataclasses.MISSING else f.default_factory()
            setattr(mind.cfg, f.name, value)
    # A mind keeps the configuration it was raised with: nothing is switched on at load. Settings added after it was
    # saved take their defaults above (off); the program turns a newer mechanism on only when asked to (--ac).
    if not ledger:
        mind.S.log, mind.S.keep_log = {}, False
        mind.cfg.keep_log, mind.cfg.log_path = False, None
        return mind
    if isinstance(mind.S.log, DiskLog):
        saved_log = os.path.join(os.path.dirname(os.path.abspath(path)), mind.S.log.path)
        if manifest.get("log_sha256") and _sha256(saved_log) != manifest["log_sha256"]:
            raise ValueError("experience log does not match the manifest")
        live = live_log or "%s.live-%d-%d" % (saved_log, os.getpid(), time.time_ns())
        try:
            shutil.copyfile(saved_log, live)                # the saved ledger is never written to
        except BaseException:                              # a copy that did not finish is not left behind
            _remove(live)
            raise
        mind.S.log.path = live
    return mind
