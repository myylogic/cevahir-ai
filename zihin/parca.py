# -*- coding: utf-8 -*-
"""Pieces of a word from the owner's own tokenizer infrastructure (tokenizer_management): its root and suffixes
(morphology) and its last syllable (syllabifier), as facets that accompany the word.

Turkish builds a word from a root and a chain of suffixes, so "yapıyorsun", "gidiyorsun" and "düşünüyorsun" share
"ıyor"-like and "sun" pieces, and "yapıyorsun" and "yaptım" share the root "yap". As facets (they accompany the word,
they are not words of the utterance) they let words that are built alike resemble each other (F18): a new
inflection of a known root, or a new root with a known suffix chain, finds its place among what was lived. The
pieces are symbols the mind lives with; no grammar rule enters the core.
"""
from __future__ import annotations

import logging
from typing import Dict, List


class OwnPieces:
    """The word's pieces: "kök:<root>" when the root differs from the word, "ek:<suffix>" for each suffix the
    morphology splits off, "son:<last syllable>" when the word has more than one syllable."""

    def __init__(self) -> None:
        level = logging.getLogger().level
        logging.disable(logging.WARNING)                  # the tokenizer logs at import; the session stays quiet
        try:
            from tokenizer_management.bpe.tokenization.morphology import Morphology
            from tokenizer_management.bpe.tokenization.syllabifier import Syllabifier
            self.morphology, self.syllabifier = Morphology(), Syllabifier()
        finally:
            logging.disable(logging.NOTSET)
            logging.getLogger().setLevel(level)
        self.cache: Dict[str, List[str]] = {}

    def tokenize(self, word: str) -> List[str]:
        if word in self.cache:
            return self.cache[word]
        out: List[str] = []
        if word.isalpha() and len(word) > 2:
            try:
                parts = self.morphology._split_morpheme(word)
                root = self.morphology.find_root(word)
            except Exception:
                parts, root = [word], word
            if root and root != word:
                out.append("kök:" + root)
            out += ["ek:" + p for p in parts[1:] if p]
            try:
                syllables = self.syllabifier.split([word])
            except Exception:
                syllables = []
            if len(syllables) > 1:
                out.append("son:" + syllables[-1])
        self.cache[word] = out
        return out


def own_pieces():
    """The owner's tokenizer pieces when its package is importable (the repository root on the path), else None."""
    try:
        return OwnPieces()
    except Exception:
        return None
