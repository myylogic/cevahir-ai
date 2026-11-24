# -*- coding: utf-8 -*-
"""The lexicon: words kept as paths of letters, numbered by the mind in the order it met them (MODEL.md, 3.3).

There is no list of words and no vector per word. A word is a path from the root, one letter per step. Each step the
mind took for the first time became a node, with the next free number. A word's number is the number of the node where
it ends. The path regenerates the word from its number (``spell``): the store is a generator. Words that share a
beginning share nodes ("ev", "evler", "evlerimiz"), so the store grows by the new part of a new word only.

Each node counts the words that passed through it and the words that ended at it. The belief that a path is a whole
word is the rule on that record (``word_belief``).
"""
from __future__ import annotations

import re

from .evidence import belief

WORD = re.compile(r"[^\W_]+")       # a maximal run of letters and digits, in any script


def words(text: str) -> list[tuple[int, int]]:
    """Spans (start, end) of the words of ``text``."""
    return [m.span() for m in WORD.finditer(text)]


class Lexicon:
    def __init__(self) -> None:
        self.parent = [-1]
        self.letter = [""]
        self.passes = [0]
        self.ends = [0]
        self.kids: list[dict[str, int]] = [{}]

    def live(self, word: str) -> int:
        """Meet ``word``: walk (and grow) its path, count the meeting; return its number."""
        node = 0
        kids = self.kids
        for ch in word:
            nxt = kids[node].get(ch)
            if nxt is None:
                nxt = len(self.parent)
                self.parent.append(node)
                self.letter.append(ch)
                self.passes.append(0)
                self.ends.append(0)
                kids.append({})
                kids[node][ch] = nxt
            self.passes[nxt] += 1
            node = nxt
        self.ends[node] += 1
        return node

    def number(self, word: str) -> int | None:
        """The number of ``word`` if the mind has met it as a whole word, else None."""
        node = 0
        for ch in word:
            node = self.kids[node].get(ch)
            if node is None:
                return None
        return node if self.ends[node] else None

    def spell(self, number: int) -> str:
        """Regenerate a word from its number."""
        out = []
        while number > 0:
            out.append(self.letter[number])
            number = self.parent[number]
        return "".join(reversed(out))

    def word_belief(self, number: int) -> float:
        """Belief that the path ending at ``number`` is a whole word: W+ = ends there, W = passes through it."""
        return belief(self.ends[number], self.passes[number])

    def size(self) -> dict:
        n_words = sum(1 for e in self.ends if e)
        once = sum(1 for e in self.ends if e == 1)
        return {"nodes": len(self.parent) - 1, "words": n_words, "words_met_once": once}
