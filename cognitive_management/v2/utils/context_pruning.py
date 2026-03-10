# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: context_pruning.py
Modül: cognitive_management/v2/utils
Görev: V2 Context Pruning - Context building ve pruning utilities. V1'den taşındı
       ve V2'ye özelleştirildi. Context building, context pruning, token estimation
       ve role-based context management işlemlerini yapar. ROLE_USER, ROLE_ASSISTANT,
       ROLE_SYSTEM_SUMMARY role tanımlarını içerir.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (context pruning)
- Design Patterns: Pruning Pattern (context pruning)
- Endüstri Standartları: Context management best practices

KULLANIM:
- Context building için
- Context pruning için
- Token estimation için
- Role-based context management için

BAĞIMLILIKLAR:
- CognitiveExceptions: Exception tanımları
- Config: Yapılandırma

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple

from cognitive_management.exceptions import ContextBuildError, ValidationError
from cognitive_management.config import CognitiveManagerConfig


# ====== Yardımcı / Ortak ======================================================

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM_SUMMARY = "system_summary"

DEFAULT_CHARS_PER_TOKEN = 4.0  # Byte/WordPiece yaklaşık katsayı

def estimate_tokens(text: str, *, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Basit yaklaşık token sayacı (dil ve tokenizer'a göre değişebilir)."""
    if not text:
        return 0
    # whitespace yoğun metinlerde aşırı sapmayı önlemek için sıkıştır
    n_chars = len(" ".join(text.split()))
    return max(1, int(round(n_chars / max(1e-6, chars_per_token))))

def _render_turn(role: str, content: str) -> str:
    role = (role or "").strip().lower()
    tag = {
        ROLE_USER: "[USER]",
        ROLE_ASSISTANT: "[ASSISTANT]",
        ROLE_SYSTEM_SUMMARY: "[SYSTEM_SUMMARY]",
    }.get(role, f"[{role.upper() or 'UNKNOWN'}]")
    return f"{tag}\n{content.strip()}"

def _keyword_bag(text: str) -> List[str]:
    # Çok basit bir anahtar kelime torbası: harf/rakam, 2+ uzunluk, küçük harf
    out: List[str] = []
    cur = []
    for ch in text.lower():
        if ch.isalnum() or ch in ("_", "-"):
            cur.append(ch)
        else:
            if len(cur) >= 2:
                out.append("".join(cur))
            cur = []
    if len(cur) >= 2:
        out.append("".join(cur))
    return out

def _overlap_score(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    if not sa:
        return 0.0
    sb = set(b)
    return len(sa & sb) / float(len(sa))


# ====== Önem (Saliency) Hesabı ================================================

def score_history_items(history: List[Dict[str, Any]], *, query_text: str, recency_bias: float = 0.15) -> List[Tuple[int, float]]:
    """
    Her bir geçmiş turu (index, skor) şeklinde puanlar.
    Skor = anahtar kelime örtüşmesi + zayıf bir yakınlık (recency) bileşeni.
    """
    q_bag = _keyword_bag(query_text)
    scores: List[Tuple[int, float]] = []
    n = len(history)
    if n == 0:
        return scores

    for i, item in enumerate(history):
        content = str(item.get("content", "") or "")
        role = str(item.get("role", "") or "")
        # sistem özetlerini hafifçe ödüllendir (bilgi yoğun)
        base = 0.05 if role == ROLE_SYSTEM_SUMMARY else 0.0
        c_bag = _keyword_bag(content)
        overlap = _overlap_score(q_bag, c_bag)
        # recency: sona yakın olanlar biraz daha yüksek
        # i:0 en eski → rec_idx artınca skor artsın
        rec_w = (i + 1) / float(n)
        score = base + overlap + recency_bias * rec_w
        scores.append((i, score))
    return scores

def select_salient(history: List[Dict[str, Any]], *, topk: int, query_text: str) -> List[Dict[str, Any]]:
    """
    En önemli 'topk' turu seçer (stabil sıralama).
    """
    if topk <= 0 or not history:
        return []
    scored = score_history_items(history, query_text=query_text)
    # Skora göre azalan, eşitse daha yeni olan öne
    scored.sort(key=lambda t: (t[1], t[0]), reverse=True)
    chosen_idx = sorted([idx for idx, _ in scored[:topk]])
    return [history[i] for i in chosen_idx]


# ====== Token Bütçesine Göre Kırpma ==========================================

def truncate_to_budget(chunks: List[str], *, max_tokens: int, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> List[str]:
    """
    Sıralı parçaları token bütçesine sığacak şekilde kırpar. Baştan itibaren ekler.
    Son parça bütçeyi aşıyorsa, makul bir yerde keser.
    """
    kept: List[str] = []
    used = 0
    for part in chunks:
        t = estimate_tokens(part, chars_per_token=chars_per_token)
        if used + t <= max_tokens:
            kept.append(part)
            used += t
        else:
            # Parçayı kısmen al
            remain = max_tokens - used
            if remain <= 0:
                break
            # karaktersel kestirme
            # not: remain * cpt ~ karakter sayısı
            approx_chars = int(remain * chars_per_token)
            if approx_chars > 20:
                kept.append(part[:approx_chars] + " …")
            # Bütçe doldu
            break
    return kept


# ====== Bağlam İnşası ========================================================

def build_context(
    cfg: CognitiveManagerConfig,
    *,
    user_message: str,
    history: Optional[List[Dict[str, Any]]] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Sistem prompt'u + seçilmiş geçmiş + güncel kullanıcı mesajından yekpare
    bir bağlam metni oluşturur. Token bütçesi MemoryConfig.max_history_tokens
    ile sınırlandırılır.
    """
    if not isinstance(cfg, CognitiveManagerConfig):
        raise ValidationError("cfg tipi geçersiz (CognitiveManagerConfig bekleniyor).")

    history = history or []
    system_prompt = system_prompt or cfg.default_system_prompt

    # 1) Ön eleme: boş/çok kısa turları düşür
    filtered: List[Dict[str, Any]] = []
    for item in history:
        try:
            role = str(item.get("role", "") or "").strip().lower()
            content = str(item.get("content", "") or "").strip()
        except Exception:
            # kötü formatlı girişler atılır
            continue
        if not content:
            continue
        if role not in {ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM_SUMMARY}:
            role = "other"
        filtered.append({"role": role, "content": content})

    # 2) Önem tabanlı seçim (opsiyonel) + son turların korunması
    sel: List[Dict[str, Any]] = filtered
    if cfg.memory.enable_salient_pruning and filtered:
        # son 2 turu koşulsuz tut, geri kalan için saliency uygula
        tail = filtered[-2:] if len(filtered) >= 2 else filtered[:]
        head = filtered[:-2] if len(filtered) > 2 else []
        picked = select_salient(head, topk=max(0, cfg.memory.salient_topk - len(tail)), query_text=user_message)
        sel = picked + tail

    # 3) Parçaları oluştur - sistem promptu gizle
    parts: List[str] = []
    # Sistem promptu artık context'e dahil edilmiyor - modelin "düşüncesi" olarak kalacak
    
    for item in sel:
        parts.append(_render_turn(item.get("role", ""), item.get("content", "")))

    parts.append(_render_turn(ROLE_USER, user_message))

    # 4) Token bütçesi uygulama
    budget = max(128, int(cfg.memory.max_history_tokens))  # güvenli alt sınır
    try:
        pruned = truncate_to_budget(parts, max_tokens=budget)
    except Exception as e:
        raise ContextBuildError("Token bütçesine göre kırpma başarısız.") from e

    # 5) Nihai metin
    context = "\n\n".join(pruned).strip()
    if not context:
        raise ContextBuildError("Bağlam oluşturulamadı (boş çıktı).")
    return context


__all__ = [
    "estimate_tokens",
    "select_salient",
    "truncate_to_budget",
    "build_context",
    "ROLE_USER",
    "ROLE_ASSISTANT",
    "ROLE_SYSTEM_SUMMARY",
]

