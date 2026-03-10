# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: heuristics.py
Modül: cognitive_management/v2/utils
Görev: V2 Heuristics - Policy routing ve tool selection için heuristic yardımcılar.
       V1'den taşındı ve V2'ye özelleştirildi. estimate_risk, should_tool, mode_gates,
       build_features ve diğer heuristic fonksiyonlarını içerir. Policy routing
       ve tool selection için heuristic hesaplamaları yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (heuristic utilities)
- Design Patterns: Utility Pattern (heuristic functions)
- Endüstri Standartları: Heuristic calculation best practices

KULLANIM:
- Policy routing heuristics için
- Tool selection heuristics için
- Risk estimation için
- Feature building için

BAĞIMLILIKLAR:
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
from typing import Dict, Any, Tuple

from cognitive_management.config import CognitiveManagerConfig


# =============================================================================
# Metin Yardımcıları
# =============================================================================

def _lower(s: str | None) -> str:
    return (s or "").strip().lower()

def _contains_any(text: str, needles: Tuple[str, ...]) -> bool:
    if not text or not needles:
        return False
    t = _lower(text)
    return any(n in t for n in needles)

def _token_len(text: str) -> int:
    # kabaca kelime sayısı
    return len((text or "").split())


# =============================================================================
# Heuristik Özellikler
# =============================================================================

def build_features(
    cfg: CognitiveManagerConfig,
    *,
    user_message: str,
    entropy_est: float | None = None,
) -> Dict[str, Any]:
    """
    Kullanıcı mesajından ve opsiyonel entropi kestiriminden temel sinyalleri üretir.
    """
    msg = _lower(user_message)
    f_rules = cfg.features.tool_rules
    s_rules = cfg.safety

    # Araç tetikleyicileri
    needs_recent = _contains_any(msg, tuple(f_rules.get("needs_recent_info_triggers", ())))
    needs_calc = _contains_any(msg, tuple(f_rules.get("needs_calc_or_parse_triggers", ())))

    # Risk/iddia sinyalleri
    has_claim_marker = _contains_any(msg, s_rules.claim_markers)
    is_sensitive = _contains_any(msg, s_rules.risk_keywords_sensitive)

    # Uzunluk & yoğunluk sinyalleri
    tokens = _token_len(msg)

    return {
        "input_length": tokens,
        "entropy_est": float(entropy_est or 0.0),
        "needs_recent_info": needs_recent,
        "needs_calc_or_parse": needs_calc,
        "has_claims": has_claim_marker,
        "is_sensitive_domain": is_sensitive,
    }


def estimate_risk(cfg: CognitiveManagerConfig, features: Dict[str, Any]) -> float:
    """
    Basit risk skoru (0..1). İddia ve hassas alanlara ağırlık verir.
    """
    r = cfg.features.risk_rules
    base = float(r.get("base_risk", 0.0))
    score = base

    if features.get("has_claims"):
        score += float(r.get("claim_bonus", 0.5))

    if features.get("is_sensitive_domain"):
        score += float(r.get("sensitive_bonus", 0.5))

    # Entropi yüksekliği belirsizlik riski olarak küçük katkı
    try:
        ent = float(features.get("entropy_est", 0.0))
        score += 0.1 * max(0.0, min(ent, 3.0)) / 3.0
    except Exception:
        pass

    # Üst sınırla
    return max(0.0, min(float(r.get("max_risk", 1.0)), score))


def should_tool(cfg: CognitiveManagerConfig, features: Dict[str, Any]) -> str:
    """
    Araç kullanımına yönelik öneri: "none" | "maybe" | "must"
    - Güncel bilgi gerekiyorsa "must"
    - Hesap/parsing gerekiyorsa "maybe"
    """
    if not cfg.tools.enable_tools:
        return "none"

    if features.get("needs_recent_info"):
        return "must"
    if features.get("needs_calc_or_parse"):
        return "maybe"

    return str(cfg.features.tool_rules.get("default_decision", "none"))


# =============================================================================
# Mod Seçimi İçin Basit Kapılar (opsiyonel sinyal)
# =============================================================================

def mode_gates(cfg: CognitiveManagerConfig, features: Dict[str, Any]) -> Dict[str, bool]:
    """
    PolicyRouter'ın kullanabileceği basit kapılar:
      - think_gate: iç ses ihtimali
      - debate_gate: çift aday denemesi için kapı
      - tot_gate: Tree of Thoughts için kapı (Phase 4)
    """
    ent = float(features.get("entropy_est", 0.0))
    length = int(features.get("input_length", 0))

    return {
        "think_gate": ent >= cfg.policy.entropy_gate_think,
        "debate_gate": (cfg.policy.debate_enabled and
                        (ent >= cfg.policy.entropy_gate_debate or
                         length >= cfg.policy.length_gate_debate)),
        # Phase 4: Tree of Thoughts gate
        "tot_gate": (cfg.policy.tot_enabled and
                     (ent >= cfg.policy.entropy_gate_tot or
                      length >= cfg.policy.length_gate_tot)),
    }


__all__ = [
    "build_features",
    "estimate_risk",
    "should_tool",
    "mode_gates",
]

