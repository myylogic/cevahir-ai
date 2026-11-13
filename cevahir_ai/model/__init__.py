"""Unversioned facade over ``model_management``; ``config`` and ``checkpoint`` are sub-facades.

``cevahir_ai.model.checkpoint.open_checkpoint`` is the governed load path: every load goes through
the compatibility gate and is attested afterwards.
"""
import importlib

from cevahir_ai._lazy import lazy_exports
from cevahir_ai._roof import EXPORTS

__all__ = sorted([*EXPORTS["model"], "config", "checkpoint"])
_exports_getattr, _exports_dir = lazy_exports(__name__, EXPORTS["model"])


def __getattr__(name):
    if name in ("config", "checkpoint"):
        return importlib.import_module(f"{__name__}.{name}")
    return _exports_getattr(name)


def __dir__():
    return sorted(set(_exports_dir()) | {"config", "checkpoint"})
