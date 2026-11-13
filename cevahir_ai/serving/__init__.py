"""Unversioned facade: the public names of ``cevahir_ai.serving`` are the legacy objects themselves.

The table is ``cevahir_ai._roof.EXPORTS["serving"]``. Nothing is imported until a name is read.
"""
from cevahir_ai._lazy import lazy_exports
from cevahir_ai._roof import EXPORTS

__all__ = sorted(EXPORTS["serving"])
__getattr__, __dir__ = lazy_exports(__name__, EXPORTS["serving"])
