"""Operations: one function per user-level action, under every front end.

The console cmdlets are thin adapters over these functions, so a Python caller gets the same
behaviour and the same audit records. Every operation takes the ``Store`` it works on and returns
``Record`` objects. Acting operations run inside ``governed`` (``cevahir_ai.ops.governed``).
Nothing here imports torch at module level; checkpoint inspection imports it on first use.
"""
import importlib

_SUBMODULES = ("governed", "artifacts", "checkpoints", "audit", "keys", "env")


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(_SUBMODULES)
