"""Unversioned tokenizer facade.

``LegacyBPETokenizer`` is ``tokenizer_management.core.tokenizer_core.TokenizerCore`` and
``ByteLevelBPE`` is ``tokenizer_management.byte_bpe.ByteLevelBPE``: the same objects, not copies.
``tokenizer_key`` gives the canonical key of either.
"""
from cevahir_ai._lazy import lazy_exports
from cevahir_ai._roof import EXPORTS

__all__ = sorted([*EXPORTS["tokenizer"], "tokenizer_key"])
__getattr__, _legacy_dir = lazy_exports(__name__, EXPORTS["tokenizer"])


def __dir__():
    return sorted(set(_legacy_dir()) | set(__all__))


def tokenizer_key(tokenizer):
    """Canonical key (``tok:bytebpe-v1`` or ``tok:bpe-sem-v1``) of a live tokenizer."""
    from cevahir_ai.adapters.tokenizers import canonical_key
    return canonical_key(tokenizer)[0]
