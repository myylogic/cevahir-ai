# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: policy_router_v2.py
Modül: cognitive_management/v2/components
Görev: V3 Policy Router — QueryType ve DomainType farkındası, çok-faktörlü
       bilişsel mod seçimi ve domain-aware decoding parametresi üretimi.

       V3 Yenilikleri:
         • QueryType farkındası mod seçimi:
             math/code   → think1 (her zaman, CoT şart)
             creative    → direct (yüksek temperature, ToT/debate yok)
             conversational → direct (düşük token, düşük temp)
             reasoning   → debate2 veya tot (karmaşıklığa göre)
         • DomainType farkındası decoding:
             math        → temperature=0.45 (deterministik)
             creative    → temperature=0.85 (çeşitlilik)
             code        → temperature=0.50 (tutarlı)
             medical/law → temperature=0.55 (güvenli ve kesin)
         • self_consistency_gate desteği
         • complexity_score ağırlıklı karar hiyerarşisi

       Akademik Referanslar:
         • Wei et al. 2022 — CoT aktivasyonu için karmaşıklık eşiği
         • Wang et al. 2022 — Self-Consistency için entropy >= 1.5 önerisi
         • Yao et al. 2023 — ToT için complexity >= 0.80 eşiği

MİMARİ:
- SOLID Prensipleri: SRP, Dependency Inversion (interface'e bağımlı)
- Design Patterns: Router Pattern

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
================================================================================
"""

from __future__ import annotations

from typing import Dict, Any, Tuple

from cognitive_management.cognitive_types import (
    PolicyOutput,
    CognitiveState,
    DecodingConfig,
    Mode,
    QueryType,
    DomainType,
)
from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.v2.interfaces.component_protocols import PolicyRouter as IPolicyRouter
from cognitive_management.v2.utils.heuristics import estimate_risk, should_tool, mode_gates


class PolicyRouterV2(IPolicyRouter):
    """
    V3 Policy Router — QueryType + DomainType + complexity farkındası.

    Mod Seçim Hiyerarşisi:
        1. allow_inner_steps=False      → direkt
        2. query_type=conversational    → direkt (kısa, sohbet)
        3. query_type=creative          → direkt (yüksek temp, debate yok)
        4. tot_gate açık                → tot
        5. debate_gate açık             → debate2
        6. self_consistency_gate açık  → self_consistency (gelecek sürüm)
        7. think_gate açık OR high risk → think1
        8. continuity heuristiği        → think1
        9. Aksi hâlde                   → direkt
    """

    def __init__(self, cfg: CognitiveManagerConfig):
        if not isinstance(cfg, CognitiveManagerConfig):
            raise TypeError("cfg tipi geçersiz (CognitiveManagerConfig bekleniyor).")
        self.cfg = cfg

    # =========================================================================
    # Ana Yönlendirme
    # =========================================================================

    def route(
        self,
        features: Dict[str, Any],
        state: CognitiveState,
    ) -> PolicyOutput:
        """
        Özellik vektörüne ve duruma göre PolicyOutput üretir.

        Args:
            features: build_features() çıktısı (V3 zenginleştirilmiş).
            state:    Mevcut oturum durumu.

        Returns:
            PolicyOutput — mode, tool, decoding, query_type, domain, complexity.
        """
        f = _normalize_features(features)

        risk_score = estimate_risk(self.cfg, f)
        gates      = mode_gates(self.cfg, f)
        tool       = should_tool(self.cfg, f)

        query_type = f.get("query_type", "unknown")
        domain     = f.get("domain", "general")
        complexity = float(f.get("complexity_score", 0.0))

        mode, inner_steps = self._select_mode(f, gates, risk_score, state)
        decoding          = self._decoding_from_features(
            f, risk_score=risk_score, mode=mode
        )

        return PolicyOutput(
            mode=mode,
            tool=tool,
            decoding=decoding,
            inner_steps=inner_steps,
            query_type=query_type,
            domain=domain,
            complexity=complexity,
        )

    # =========================================================================
    # Mod Seçimi
    # =========================================================================

    def _select_mode(
        self,
        features: Dict[str, Any],
        gates: Dict[str, bool],
        risk_score: float,
        state: CognitiveState,
    ) -> Tuple[Mode, int]:
        """
        Çok-faktörlü mod seçim hiyerarşisi.

        Args:
            features:   Normalize edilmiş özellik vektörü.
            gates:      mode_gates() çıktısı (think/debate/tot/sc kapıları).
            risk_score: Birleşik risk puanı (0.0–1.0).
            state:      Oturum durumu (last_mode, turn_count…).

        Returns:
            (mode, inner_steps)
        """
        if not self.cfg.policy.allow_inner_steps:
            return "direct", 0

        query_type = features.get("query_type", "unknown")
        domain     = features.get("domain", "general")
        complexity = float(features.get("complexity_score", 0.0))
        entropy    = float(features.get("entropy_est", 0.0))
        length     = int(features.get("input_length", 0))
        last_mode  = getattr(state, "last_mode", None)

        # --- Kural 1: Basit sohbet → direkt ---
        if query_type == "conversational" and length <= 15:
            return "direct", 0

        # --- Kural 2: Yaratıcı içerik → direkt (debate/tot anlamsız) ---
        if query_type == "creative":
            return "direct", 0

        # --- Kural 3: ToT kapısı açık → tot ---
        if self.cfg.policy.tot_enabled and gates.get("tot_gate", False):
            return "tot", self.cfg.policy.tot_max_depth

        # --- Kural 4: Math/code → her zaman CoT (en az think1) ---
        if query_type in ("math", "code"):
            # Çok karmaşık matematik → debate2 (iki farklı çözüm yolu dene)
            if complexity >= 0.60 and self.cfg.policy.debate_enabled:
                return "debate2", 2
            return "think1", 1

        # --- Kural 5: Debate kapısı açık → debate2 ---
        if self.cfg.policy.debate_enabled and gates.get("debate_gate", False):
            return "debate2", 2

        # --- Kural 6: Think kapısı açık veya yüksek risk → think1 ---
        high_risk = 0.65
        if gates.get("think_gate", False) or risk_score >= high_risk:
            return "think1", 1

        # --- Kural 7: Tıp / hukuk alan riski → think1 ---
        if domain in ("medical", "law") and risk_score >= 0.40:
            return "think1", 1

        # --- Kural 8: Süreklilik heuristiği ---
        # Önceki tur düşünme modundaydı ve entropy hâlâ görece yüksek → devam
        if last_mode in ("think1", "debate2", "tot") and entropy >= self.cfg.policy.entropy_gate_think * 0.75:
            return "think1", 1

        # --- Varsayılan ---
        return "direct", 0

    # =========================================================================
    # Domain-Aware Decoding
    # =========================================================================

    def _decoding_from_features(
        self,
        features: Dict[str, Any],
        *,
        risk_score: float,
        mode: Mode = "direct",
    ) -> DecodingConfig:
        """
        Domain + QueryType + Mode farkındası decoding parametreleri.

        Akademik Temeller:
            • Matematik/kod: düşük temperature → deterministik, tekrarlanabilir
              (Cobbe et al. 2021 — GSM8K'de 0.5 altı temp önerilir)
            • Yaratıcı yazı: yüksek temperature → çeşitli, özgün
            • Risk yüksek: düşük temperature → muhafazakâr yanıtlar
            • ToT: orta temp → keşif-sömürü dengesi

        Args:
            features:   Normalize edilmiş özellik vektörü.
            risk_score: 0.0–1.0 risk puanı.
            mode:       Seçilen bilişsel mod.

        Returns:
            DecodingConfig
        """
        bounds     = self.cfg.decoding_bounds
        mlo, mhi   = bounds.max_new_tokens_bounds
        tlo, thi   = bounds.temperature_bounds

        query_type = features.get("query_type", "unknown")
        domain     = features.get("domain", "general")
        entropy    = float(features.get("entropy_est", 0.0))
        length     = int(features.get("input_length", 0))
        complexity = float(features.get("complexity_score", 0.0))

        # ---- Token Sayısı ----
        if mode == "tot":
            max_new = int(mlo + (mhi - mlo) * 0.90)    # ToT: maksimum alan
        elif mode == "debate2":
            max_new = int(mlo + (mhi - mlo) * 0.75)    # Debate: iki yol gerekir
        elif mode == "think1":
            max_new = int(mlo + (mhi - mlo) * 0.60)    # CoT: orta
        elif mode == "self_consistency":
            max_new = int(mlo + (mhi - mlo) * 0.65)
        else:
            # Direkt mod: entropi + uzunluk + karmaşıklık ağırlıklı
            ent_f  = min(1.0, entropy  / 3.0)
            len_f  = min(1.0, length   / 80.0)
            cmp_f  = min(1.0, complexity)
            factor = (ent_f * 0.40 + len_f * 0.30 + cmp_f * 0.30)
            max_new = int(mlo + (mhi - mlo) * factor)

        # ---- Sıcaklık ----
        temperature = self._select_temperature(
            query_type, domain, mode, risk_score, entropy, tlo, thi, bounds
        )

        # Sınır kontrolü
        max_new     = int(max(mlo, min(mhi, max_new)))
        temperature = round(max(tlo, min(thi, temperature)), 3)

        return DecodingConfig(
            max_new_tokens=max_new,
            temperature=temperature,
            top_p=bounds.top_p_default,
            repetition_penalty=bounds.repetition_penalty_default,
        )

    def _select_temperature(
        self,
        query_type: str,
        domain: str,
        mode: Mode,
        risk_score: float,
        entropy: float,
        tlo: float,
        thi: float,
        bounds,
    ) -> float:
        """
        Sıcaklık seçim mantığı — domain > mode > risk sırasıyla öncelikli.

        Domain Öncelikleri:
            math / code   → düşük sıcaklık (deterministik)
            creative      → yüksek sıcaklık (çeşitlilik)
            medical / law → orta-düşük (güvenli ve kesin)
            science       → orta (dengeli)
            general       → risk ve entropy tabanlı

        Mode Düzeltmeleri (domain üstüne):
            tot           → +0.05 (keşif için hafif artış)
            think1        → -0.05 (odak için hafif düşüş)
        """
        # --- Domain-based base temperature ---
        if query_type in ("math",) or domain == "math":
            base = getattr(bounds, "math_temperature", 0.45)
        elif query_type in ("code",) or domain == "technology":
            base = getattr(bounds, "code_temperature", 0.50)
        elif query_type in ("creative",) or domain == "creative":
            base = getattr(bounds, "creative_temperature", 0.85)
        elif domain in ("medical", "law"):
            base = tlo + (thi - tlo) * 0.35      # 0.40 + 0.18 ≈ 0.58 (güvenli)
        elif domain == "science":
            base = tlo + (thi - tlo) * 0.45      # Dengeli
        else:
            # Genel: risk baskın, entropy hafif katkı
            risk_factor    = 1.0 - risk_score
            entropy_factor = min(1.0, entropy / 3.0)
            factor         = risk_factor * 0.65 + entropy_factor * 0.35
            base           = tlo + (thi - tlo) * factor

        # --- Mode düzeltmesi ---
        if mode == "tot":
            base += 0.05     # Keşif bonusu
        elif mode in ("think1", "debate2"):
            base -= 0.05     # Odak indirimi
        elif mode == "direct" and query_type == "conversational":
            base -= 0.08     # Konuşma: çok sıcaklık akıcılığı bozar

        return float(base)


# =============================================================================
# Yardımcı — Normalize Edilmiş Özellik Vektörü
# =============================================================================

def _normalize_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Eksik anahtarları güvenli varsayılan değerlerle doldurur.
    V3: query_type, domain, complexity_score, is_* kısayollar eklendi.
    """
    return {
        "input_length":        int(raw.get("input_length",        0)     or 0),
        "entropy_est":         float(raw.get("entropy_est",       0.0)   or 0.0),
        "needs_recent_info":   bool(raw.get("needs_recent_info",  False)),
        "needs_calc_or_parse": bool(raw.get("needs_calc_or_parse",False)),
        "has_claims":          bool(raw.get("has_claims",         False)),
        "is_sensitive_domain": bool(raw.get("is_sensitive_domain",False)),
        # V3 yeni alanlar
        "query_type":          raw.get("query_type",          "unknown"),
        "domain":              raw.get("domain",              "general"),
        "complexity_score":    float(raw.get("complexity_score", 0.0) or 0.0),
        "is_math_query":       bool(raw.get("is_math_query",     False)),
        "is_creative_query":   bool(raw.get("is_creative_query", False)),
        "is_conversational":   bool(raw.get("is_conversational", False)),
        "is_code_query":       bool(raw.get("is_code_query",     False)),
        "n_question_marks":    int(raw.get("n_question_marks",   0)     or 0),
    }


__all__ = ["PolicyRouterV2"]
