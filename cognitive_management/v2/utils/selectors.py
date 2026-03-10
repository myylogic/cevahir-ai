# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: selectors.py
Modül: cognitive_management/v2/utils
Görev: V2 Selectors - Thought candidate selection utilities. V1'den taşındı
       ve V2'ye özelleştirildi. pick_best_by_score, select_topk, select_diverse
       ve diğer selection fonksiyonlarını içerir. Thought candidate selection,
       top-k selection ve diverse selection işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (selection utilities)
- Design Patterns: Selector Pattern (candidate selection)
- Endüstri Standartları: Selection algorithms best practices

KULLANIM:
- Thought candidate selection için
- Top-k selection için
- Diverse selection için

BAĞIMLILIKLAR:
- CognitiveTypes: ThoughtCandidate tip tanımları

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import List, Iterable

from cognitive_management.cognitive_types import ThoughtCandidate


# === Basit seçimler ===========================================================

def pick_best_by_score(cands: List[ThoughtCandidate]) -> ThoughtCandidate:
    """
    En yüksek skorlu adayı döndürür.
    Hiç aday yoksa skor=-inf benzeri bir boş aday döndürür.
    """
    if not cands:
        return ThoughtCandidate(text="", score=float("-inf"))
    return max(cands, key=lambda c: (c.score, len(c.text)))


def select_topk(cands: List[ThoughtCandidate], k: int = 2) -> List[ThoughtCandidate]:
    """
    Skora göre azalan sıralı ilk k adayı döndürür.
    """
    k = max(0, int(k))
    if k == 0 or not cands:
        return []
    return sorted(cands, key=lambda c: (c.score, len(c.text)), reverse=True)[:k]


# === Rerank Yardımcıları =====================================================

def rerank_with_length_penalty(cands: List[ThoughtCandidate], alpha: float = 0.0) -> List[ThoughtCandidate]:
    """
    Uzunluk cezası ile yeniden sıralama.
    final_score = score - alpha * log(1 + len(text))
    alpha=0 → etkisiz.
    """
    if not cands or alpha <= 0.0:
        return list(sorted(cands, key=lambda c: (c.score, len(c.text)), reverse=True))
    def _pen(c: ThoughtCandidate) -> float:
        # çok uzun iç sesleri hafifçe cezalandır
        import math
        return c.score - float(alpha) * math.log1p(len(c.text) or 0.0)
    return sorted(cands, key=lambda c: (_pen(c), len(c.text)), reverse=True)


def diversify_topk(
    cands: List[ThoughtCandidate],
    k: int = 2,
    *,
    jaccard_threshold: float = 0.75,
) -> List[ThoughtCandidate]:
    """
    Basit çeşitlilik seçimi:
    - Skora göre sırala.
    - Yüksek benzerlik gösteren (Jaccard >= eşik) adayları eler.
    """
    ordered = sorted(cands, key=lambda c: (c.score, len(c.text)), reverse=True)
    chosen: List[ThoughtCandidate] = []
    for c in ordered:
        if len(chosen) >= k:
            break
        if not chosen:
            chosen.append(c)
            continue
        tb = _token_bag(c.text)
        if all(_jaccard(tb, _token_bag(x.text)) < jaccard_threshold for x in chosen):
            chosen.append(c)
    return chosen[:k]


# === Yardımcı Metin Fonksiyonları ============================================

def _token_bag(text: str) -> set:
    """
    Çok basit bir token torbası (harf/rakam+altçizgi). 2+ uzunluk.
    """
    out = []
    cur = []
    for ch in (text or "").lower():
        if ch.isalnum() or ch in ("_", "-"):
            cur.append(ch)
        else:
            if len(cur) >= 2:
                out.append("".join(cur))
            cur = []
    if len(cur) >= 2:
        out.append("".join(cur))
    return set(out)


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / float(union or 1)


__all__ = [
    "pick_best_by_score",
    "select_topk",
    "rerank_with_length_penalty",
    "diversify_topk",
]

