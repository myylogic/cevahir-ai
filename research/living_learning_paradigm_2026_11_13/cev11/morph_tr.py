# -*- coding: utf-8 -*-
"""Türkçe özel ad hâl ekleri (CEV-11 Paradigma).

Kapsam bilerek dardır: tek sözcüklü özel adlar, yalın + altı hâl. İki ve dört yönlü ünlü uyumu, kaynaştırma
harfleri (ilgi hâlinde n; yönelme, belirtme ve vasıta hâlinde y), -DA/-DAn'da ünsüz benzeşmesi (fçşhpstk'den sonra
t) uygulanır. Özel adlarda yazıda ünsüz yumuşaması yapılmaz (Ahmet'e, Burak'ın — TDK yazım kuralı).

Biçimsel yazım ekten önce kesme işareti (U+0027) koyar: ``Ayten'den``. Gündelik (kesme işaretsiz) yazım aynı
dizgedir, yalnız kesme işareti yoktur: ``Aytenden``. Alıntı sözcüklerdeki uyum istisnaları kapsam dışıdır; dünya
üreteci yalnız Türkçe ses düzenine uyan uydurma adlar üretir.
"""
from __future__ import annotations

CASES: tuple[str, ...] = ("nom", "gen", "dat", "acc", "loc", "abl", "ins")
INFLECTED: tuple[str, ...] = CASES[1:]

BACK_VOWELS = frozenset("aıouAIOU")
FRONT_VOWELS = frozenset("eiöüEİÖÜ")
VOWELS = BACK_VOWELS | FRONT_VOWELS
VOICELESS = frozenset("fstkçşhpFSTKÇŞHP")
APOSTROPHE = "'"

_UPPER_MAP = {"i": "İ", "ı": "I"}
_LOWER_MAP = {"İ": "i", "I": "ı"}


def tr_upper(s: str) -> str:
    return "".join(_UPPER_MAP.get(ch, ch.upper()) for ch in s)


def tr_lower(s: str) -> str:
    return "".join(_LOWER_MAP.get(ch, ch.lower()) for ch in s)


def tr_capitalize(s: str) -> str:
    return tr_upper(s[:1]) + tr_lower(s[1:]) if s else s


def _last_vowel(word: str) -> str:
    for ch in reversed(word):
        if ch in VOWELS:
            return tr_lower(ch)
    raise ValueError(f"no vowel in {word!r}")


def harmony2(word: str) -> str:
    """İki yönlü uyum: kalın ünlüden sonra 'a', ince ünlüden sonra 'e'."""
    return "a" if _last_vowel(word) in "aıou" else "e"


def harmony4(word: str) -> str:
    """Dört yönlü uyum: a/ı→ı, e/i→i, o/u→u, ö/ü→ü."""
    return {"a": "ı", "ı": "ı", "e": "i", "i": "i", "o": "u", "u": "u", "ö": "ü", "ü": "ü"}[_last_vowel(word)]


def suffix(root: str, case: str) -> str:
    """Hâl ekinin bu köke uyan biçimi (kesme işaretsiz); yalın için boş dizge."""
    if case == "nom":
        return ""
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}")
    vowel_final = root[-1] in VOWELS
    a, i = harmony2(root), harmony4(root)
    d = "t" if root[-1] in VOICELESS else "d"
    if case == "gen":
        return ("n" if vowel_final else "") + i + "n"
    if case == "dat":
        return ("y" if vowel_final else "") + a
    if case == "acc":
        return ("y" if vowel_final else "") + i
    if case == "loc":
        return d + a
    if case == "abl":
        return d + a + "n"
    return ("y" if vowel_final else "") + "l" + a    # ins


def formal(root: str, case: str) -> str:
    """Biçimsel yazım: ``Ayten`` (yalın) veya ``Ayten'den``."""
    s = suffix(root, case)
    return root if not s else f"{root}{APOSTROPHE}{s}"


def informal(root: str, case: str) -> str:
    """Kesme işaretsiz gündelik yazım: ``Ayten`` veya ``Aytenden``."""
    return root + suffix(root, case)
