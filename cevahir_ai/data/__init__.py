"""Unversioned facade: the public names of ``cevahir_ai.data`` are the legacy objects themselves.

The table is ``cevahir_ai._roof.EXPORTS["data"]``. Nothing is imported until a name is read.
"""
from cevahir_ai._lazy import lazy_exports
from cevahir_ai._roof import EXPORTS

__all__ = sorted(EXPORTS["data"])
__getattr__, __dir__ = lazy_exports(__name__, EXPORTS["data"])
