"""The registry under ``CEVAHIR_HOME``: manifests, locations, the audit chain and lineage.

    CEVAHIR_HOME/
      host.json                        identity of this machine (h-<16 hex>)
      objects/ab/<62 hex>.json         manifests, content-addressed (cev:manifest:sha256-v1)
      tables/ab/<62 hex>.json          per-tensor tables of registered weights
      locations/<host-id>.json         where each payload file is on that host (never in a manifest)
      audit/<host-id>/seg-000001.jsonl one append-only, hash-chained log per machine; chain.json; quarantine/
      trust/hmac/<host-id>.bin         the host's audit MAC key (mode 0600)
      blobs/ab/<62 hex><suffix>        payloads unpacked from bundles

Everything is plain files (CJSON-1), written atomically; nothing here needs torch. A manifest
states facts about one artefact key and never contains a path, so it means the same on Windows
and on Colab. Lineage is a projection of the audit events, so the two cannot diverge.
"""
import importlib

_EXPORTS = {
    "Store": "core", "new_manifest": "core", "manifest_key": "core",
    "Catalog": "catalog", "Artifact": "catalog",
    "AuditLog": "audit", "event_hash": "audit",
    "redact_params": "redact",
}
_SUBMODULES = ("core", "catalog", "locations", "audit", "redact", "lineage", "register", "bundles")

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(_SUBMODULES))
