"""Turkish-aware casing and keyword matching for the rule-based cognitive features.

Python's ``str.lower`` is not Turkish-aware: ``"KISMEN".lower()`` gives ``"kismen"``
(not ``"kısmen"``) and ``"İLAÇ".lower()`` keeps a combining dot above (U+0307) after
the ``i``, so a keyword written in correct Turkish never matches capitalised input.
Raw substring tests have the opposite problem in an agglutinative language:
``"yaz" in "yazılım"``, ``"hak" in "hakkında"``, ``"kaç" in "kaçırdığım"`` and
``"gen" in "genel"`` are all true although none of these words means write, right,
how many or gene.

:func:`keyword_in_text` therefore matches keywords against word tokens:

* Text is lowercased with Turkish rules (``I`` -> ``ı``, ``İ`` -> ``i``). The plain
  Unicode lowering is tried as well, so an English acronym written in capitals
  (``"API"``) still matches ``"api"``.
* Apostrophes separate a name from its suffix (``"Python'da"`` -> ``python``, ``da``).
* A token matches a keyword when it equals the keyword or starts with it. For
  keywords shorter than :data:`FREE_PREFIX_MIN_LEN` letters the rest of the token
  must be a common inflectional ending (case, plural, possessive, copula, person,
  a few verb request forms, English ``-s``); derivational extensions are rejected.
  Longer keywords match any continuation (``"hesapla"`` -> ``"hesaplar mısın"``).
  A final ``p/ç/t/k`` may soften before a vowel (``"ilaç"`` -> ``"ilacı"``).
* Multi-word keywords match consecutive tokens; only the last word may inflect.
* Keywords without letters (``"%"``) or with symbols (``"c++"``) are plain substrings.

This is a heuristic word matcher, not a morphological analyser. Homographs remain
(``"yazı"`` is still a form of ``"yaz"``) and are handled only by the explicit
tables below.
"""
from __future__ import annotations

from bisect import bisect_left
from functools import lru_cache
import re
import unicodedata
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

__all__ = [
    "FREE_PREFIX_MIN_LEN",
    "ascii_fold",
    "count_keyword_matches",
    "is_inflectional_ending",
    "keyword_in_text",
    "turkish_lower",
    "turkish_upper",
    "word_tokens",
]

#: Keywords with at least this many letters match any continuation of a token.
FREE_PREFIX_MIN_LEN = 5

# Short English keywords that collide with Turkish stems ("git"ar/gitti, "ok"ul/oku,
# "hi"le, "hey"ecan): only the whole token matches.
_EXACT_ONLY = frozenset({"git", "hi", "hey", "ok"})

# Valid inflections of a keyword that are, in practice, another word.
_FALSE_FRIENDS = {
    "böl": ("bölüm",),                          # chapter / department, not "divide"
    "tanı": ("tanım",),                         # definition, recognising, getting to know
    "kaç": ("kaçm",),                           # kaçmak / kaçma: to escape
    "tamam": ("tamamen", "tamamı", "tamamla"),  # completely / all of it / complete
}

_COMBINING_DOT = "\u0307"
_SOFTENING = {"p": "b", "ç": "c", "t": "d", "k": "ğ"}
_VOWELS = frozenset("aeıioöuüâîû")
_WORD = re.compile(r"[^\W_]+")
_ASCII = str.maketrans("çğıöşüâîû", "cgiosuaiu")

_V4 = "[ıiuü]"
_V2 = "[ae]"
_D = "[dt]"
_NOMINAL = (
    f"(?:l{_V2}r)?"                                                   # plural
    f"(?:{_V4}?m{_V4}z|{_V4}?n{_V4}z|l{_V2}r{_V4}|{_V4}?m|{_V4}?n|s?{_V4})?"  # possessive
    f"(?:y?{_V4}|n{_V4}|y?{_V2}|n{_V2}|n?{_D}{_V2}n?|n?{_V4}n|y?l{_V2}"  # case
    f"|{_D}{_V2}k{_V4}|k{_V4})?"                                      # -daki / -ki
    f"(?:{_D}{_V4}r)?"                                                # copula -dır
    f"(?:y?{_V4}m|s{_V4}n{_V4}z|s{_V4}n|y?{_V4}z)?"                   # person: -(y)ım, -sın
)
_VERBAL = (
    f"y?{_V4}n(?:{_V4}z)?"                    # yazın, yazınız, söyleyin
    f"|{_V2}?r|{_V4}r"                        # aorist: yazar (mısın), söyler, verir
    f"|y?{_V2}bil(?:{_V4}r)?"                 # yazabilir
    f"|m{_V2}(?:k|y{_V4}|n{_V4}|s{_V4})?"     # yazma, yazmak, yazmayı, yazmanı
    f"|s{_V2}n{_V2}|y?{_V2}l{_V4}m|y?{_V2}y{_V4}m"  # yazsana, yazalım, yazayım
)
_INFLECTION = re.compile(f"(?:{_NOMINAL})|(?:{_VERBAL})|(?:e?s)")


def turkish_lower(text) -> str:
    """Lowercase with Turkish dotted/dotless ``i`` rules and no combining dot."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    return text.replace("İ", "i").replace("I", "ı").lower().replace(_COMBINING_DOT, "")


def turkish_upper(text) -> str:
    """Uppercase with Turkish rules: ``i`` -> ``İ``, ``ı`` -> ``I``."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    return text.replace("i", "İ").replace("ı", "I").upper()


def ascii_fold(text) -> str:
    """Turkish lowercase without diacritics (``"DOĞRU DEĞİL"`` -> ``"dogru degil"``)."""
    return turkish_lower(text).translate(_ASCII)


def _plain_lower(text) -> str:
    return unicodedata.normalize("NFC", str(text or "")).lower().replace(_COMBINING_DOT, "")


def word_tokens(text, *, lower=turkish_lower) -> List[str]:
    """Letter/digit tokens of ``lower(text)``; apostrophes and punctuation separate."""
    return _WORD.findall(lower(text))


def is_inflectional_ending(rest: str) -> bool:
    """True when ``rest`` (already lowercase) is an accepted inflectional ending."""
    return bool(_INFLECTION.fullmatch(rest))


@lru_cache(maxsize=4096)
def _compile(keyword: str):
    lowered = turkish_lower(keyword).strip()
    words = tuple(_WORD.findall(lowered))
    if not words or " ".join(words) != " ".join(lowered.split()):
        return ("substring", lowered)  # "%", "c++", or empty
    last = words[-1]
    stems = [last]
    if len(last) >= 4 and last[-1] in _SOFTENING:
        stems.append(last[:-1] + _SOFTENING[last[-1]])
    elif len(last) >= 4 and last.endswith("nk"):
        stems.append(last[:-1] + "g")
    return ("words", words, tuple(stems))


def _token_matches(token: str, keyword: str, stems: Sequence[str]) -> bool:
    if token == keyword:
        return True
    if keyword in _EXACT_ONLY:
        return False
    if any(token.startswith(prefix) for prefix in _FALSE_FRIENDS.get(keyword, ())):
        return False
    for index, stem in enumerate(stems):
        if not token.startswith(stem) or len(token) == len(stem):
            continue
        rest = token[len(stem):]
        if index > 0 and rest[0] not in _VOWELS:
            continue  # softened stems only before a vowel: ilacı, not ilacla
        if len(keyword) >= FREE_PREFIX_MIN_LEN or is_inflectional_ending(rest):
            return True
    return False


class _TokenIndex:
    """Tokens of one lowering of a text, indexed for exact and prefix lookups."""

    __slots__ = ("tokens", "vocabulary", "positions")

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.vocabulary = sorted(set(tokens))
        self.positions: Dict[str, List[int]] = {}
        for position, token in enumerate(tokens):
            self.positions.setdefault(token, []).append(position)

    def starting_with(self, prefix: str) -> Iterator[str]:
        for position in range(bisect_left(self.vocabulary, prefix), len(self.vocabulary)):
            token = self.vocabulary[position]
            if not token.startswith(prefix):
                return
            yield token

    def matches(self, compiled) -> bool:
        _, words, stems = compiled
        last = words[-1]
        if len(words) == 1:
            if last in self.positions:
                return True
            return any(_token_matches(token, last, stems)
                       for stem in stems for token in self.starting_with(stem))
        head = words[:-1]
        for start in self.positions.get(head[0], ()):
            end = start + len(head)
            if end < len(self.tokens) and \
                    all(self.tokens[start + i] == word for i, word in enumerate(head)) and \
                    _token_matches(self.tokens[end], last, stems):
                return True
        return False


@lru_cache(maxsize=16)
def _views(text: str) -> Tuple[Tuple[str, ...], Tuple[_TokenIndex, ...]]:
    """(lowered strings, token indexes) for the Turkish and the plain lowering.
    Cached because build_features matches one message against ~15 keyword lists."""
    turkish, plain = turkish_lower(text), _plain_lower(text)
    lowered = (turkish,) if plain == turkish else (turkish, plain)
    return lowered, tuple(_TokenIndex(_WORD.findall(variant)) for variant in lowered)


def _matches(text: str, keyword: str) -> bool:
    compiled = _compile(keyword)
    lowered, indexes = _views(text)
    if compiled[0] == "substring":
        return bool(compiled[1]) and any(compiled[1] in variant for variant in lowered)
    return any(index.matches(compiled) for index in indexes)


def keyword_in_text(text, keyword: str) -> bool:
    """True when ``keyword`` occurs in ``text`` as a word (see module notes)."""
    if not text or not keyword:
        return False
    return _matches(str(text), keyword)


def count_keyword_matches(text, keywords: Iterable[str]) -> int:
    """Number of distinct ``keywords`` that occur in ``text`` as words."""
    if not text or not keywords:
        return 0
    text = str(text)
    return sum(1 for keyword in keywords if keyword and _matches(text, keyword))
