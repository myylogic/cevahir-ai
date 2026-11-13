# -*- coding: utf-8 -*-
"""Düzyazı girişleri (lead-in): dünya cümlelerinin bir kısmının önüne eğitim düzyazısından bir cümle eklenir.

Neden (PROTOCOL ön-dondurma düzeltmesi): ROME/MEMIT standardı (STD kolu), anahtarı yaşanan cümle ile 10 sabit
düzyazı-önekli kopyası üzerinden ortalar. İlk tasarımda her dünya cümlesi kendi dizisinde tek başına ([BOS] cümle
[EOS]) eğitiliyordu; model bir olgu cümlesini hiç bir düzyazı cümlesinin ardından görmüyordu. Bellek grubunun
denetçisinin pilotunda, düzyazı öneki sonrasında temel model eski nesneyi yalnız %0,9 oranında üretti ve STD'nin v*
araması 72 koşunun hiçbirinde yakınsamadı. Gerçek metinde olgular metnin ortasında da geçer; bu yüzden dünya
cümlelerinin ``LEADIN_P`` kadarının önüne eğitim düzyazısından kısa bir cümle eklenir.

Ayrım: girişler eğitim düzyazısından seçilir; STD'nin ortalama için kullandığı ``keysites.PROSE_PREFIXES`` elle
yazılmış ve bu havuzda bulunmayan cümlelerdir (dağılım olarak benzer, dizge olarak görülmemiş). Bağlam genellemesi
yoklamaları (CF14) ise ayrı, eğitilmiş P1/P2 bağlamlarını kullanır.
"""
from __future__ import annotations

import random
import re
import unicodedata
from typing import Callable, Sequence

LEADIN_P = 0.20
MIN_WORDS, MAX_WORDS, MAX_TOKENS = 3, 16, 20
_SENT_END = re.compile(r"(?<=[.!?])\s+")
_OK = re.compile(r"^[^\d\[\]{}<>|#*_`$\\/]+[.!?]$")


def leadin_pool(prose_train: Sequence[str], tok, exclude: Sequence[str] = ()) -> list[str]:
    """Eğitim düzyazısından aday giriş cümleleri: 3–16 sözcük, ≤ 20 token, rakam ve işaretleme içermeyen,
    büyük harfle başlayan, nokta/ünlem/soru ile biten; tekrarsız ve sıralı (deterministik)."""
    excluded = {unicodedata.normalize("NFC", e).strip() for e in exclude}
    seen = set()
    for para in prose_train:
        for sent in _SENT_END.split(unicodedata.normalize("NFC", para)):
            s = " ".join(sent.split())
            if not s or s in seen or s in excluded:
                continue
            words = s.split()
            if not (MIN_WORDS <= len(words) <= MAX_WORDS) or not _OK.match(s) or not s[0].isupper():
                continue
            if len(tok.encode(s)) > MAX_TOKENS:
                continue
            seen.add(s)
    pool = sorted(seen)
    if len(pool) < 100:
        raise ValueError(f"lead-in pool too small: {len(pool)} sentences")
    return pool


def make_sentence_fn(pool: Sequence[str], p: float = LEADIN_P) -> Callable:
    """``world.pass_sentences`` ile aynı imzayı taşıyan kaynak: her cümleye bağımsız olarak p olasılıkla giriş ekler.

    Meta'ya ``leadin: True`` eklenir (ExposureLedger yalnız kendi alanlarını kullanır). Rastgelelik çağıranın rng'sidir.
    """
    pool = tuple(pool)

    def sentence_fn(world, pass_idx: int, rng: random.Random):
        from .world import pass_sentences
        out = []
        for text, meta in pass_sentences(world, pass_idx, rng):
            if rng.random() < p:
                text = pool[rng.randrange(len(pool))] + " " + text
                meta = {**meta, "leadin": True}
            out.append((text, meta))
        return out

    return sentence_fn
