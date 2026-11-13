"""The central key system.

Every artefact has a typed, content-derived key ``cev:<kind>:<scheme>:<digest>``: tokenizer,
corpus, cache, architecture, weight schema, weights, checkpoint, model, run and evaluation.
Existing identities (``tokenizer_digest`` v3, md5 cache names, CEV-11 ``model_sha256``, ``.sha256``
sidecars, the book's fingerprints) are registered as schemes of their own, so nothing persisted is
recomputed or invalidated. One pure function, ``check_compatibility``, decides every governed load.

Importing this package loads no torch and no legacy package.
"""
import importlib

_EXPORTS = {
    "Key": "grammar", "parse_key": "grammar", "is_key": "grammar", "KEY_RE": "grammar",
    "cjson": "canonical", "structured_digest": "canonical",
    "text_lf": "text", "utf8lf_digest": "text",
    "SCHEMES": "registry", "derive": "registry",
    "check_compatibility": "gate", "DEFAULT_GATE_POLICY": "gate",
}
_SUBMODULES = ("canonical", "text", "grammar", "registry", "classify", "versions", "gate", "cli",
               "schemes", "schemas", "torchkeys")

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(_SUBMODULES))
