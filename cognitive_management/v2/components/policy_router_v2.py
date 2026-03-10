# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: policy_router_v2.py
Modül: cognitive_management/v2/components
Görev: V2 Policy Router - Bağımsız implementasyon. V1'e bağımlı değil. Strateji
       seçimi (direct/think/debate/tot), mode gates, risk estimation ve tool
       decision işlemlerini yapar. Heuristics kullanarak policy routing kararları
       verir.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (policy routing),
                     Dependency Inversion (interface'lere bağımlı)
- Design Patterns: Router Pattern (policy routing)
- Endüstri Standartları: Policy routing best practices

KULLANIM:
- Strateji seçimi için
- Mode gates için
- Risk estimation için
- Tool decision için

BAĞIMLILIKLAR:
- Heuristics: Heuristic fonksiyonlar
- Component Protocols: PolicyRouter interface

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

from cognitive_management.cognitive_types import PolicyOutput, CognitiveState, DecodingConfig, Mode
from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.v2.interfaces.component_protocols import PolicyRouter as IPolicyRouter
from cognitive_management.v2.utils.heuristics import estimate_risk, should_tool, mode_gates


class PolicyRouterV2(IPolicyRouter):
    """
    V2 Policy Router - Bağımsız implementasyon.
    """
    
    def __init__(self, cfg: CognitiveManagerConfig):
        """
        Initialize V2 Policy Router.
        
        Args:
            cfg: Cognitive manager configuration
        """
        if not isinstance(cfg, CognitiveManagerConfig):
            raise TypeError("cfg tipi geçersiz (CognitiveManagerConfig bekleniyor).")
        self.cfg = cfg
    
    def route(
        self,
        features: Dict[str, Any],
        state: CognitiveState
    ) -> PolicyOutput:
        """
        Route to appropriate policy based on features.
        
        Advanced routing with entropy-based and context-aware decision making.
        Supports think1 (Chain of Thought) and debate2 (Self-Consistency) modes.
        
        Args:
            features: Extracted features from input
            state: Current cognitive state
            
        Returns:
            PolicyOutput with selected strategy
        """
        # Varsayılan eksik anahtarları doldur
        f = _normalize_features(features)

        # Risk & kapılar
        risk_score = estimate_risk(self.cfg, f)
        gates = mode_gates(self.cfg, f)

        # Mod kararı - Advanced routing with think/debate support
        mode, inner_steps = self._select_mode(f, gates, risk_score, state)

        # Araç kararı (öneri)
        tool = should_tool(self.cfg, f)

        # Decoding düğmeleri - Context-aware decoding
        decoding = self._decoding_from_features(f, risk_score=risk_score, mode=mode)

        return PolicyOutput(
            mode=mode,
            tool=tool,
            decoding=decoding,
            inner_steps=inner_steps,
        )
    
    def _select_mode(
        self,
        features: Dict[str, Any],
        gates: Dict[str, bool],
        risk_score: float,
        state: CognitiveState
    ) -> tuple[Mode, int]:
        """
        Select reasoning mode based on features, gates, risk, and context.
        
        Phase 4: Enhanced with Tree of Thoughts (ToT) support.
        
        Decision hierarchy:
        1. If ToT enabled and tot_gate open → tot (highest complexity)
        2. If debate enabled and debate_gate open → debate2
        3. If think enabled and think_gate open → think1
        4. If high risk → think1 (safer reasoning)
        5. Otherwise → direct
        
        Args:
            features: Normalized features
            gates: Mode gates (think_gate, debate_gate, tot_gate)
            risk_score: Risk score (0.0-1.0)
            state: Current cognitive state
            
        Returns:
            Tuple of (mode, inner_steps)
        """
        # Check if inner steps are allowed
        if not self.cfg.policy.allow_inner_steps:
            return "direct", 0
        
        # Context-aware decision: Check if previous mode was think/debate/tot
        # (continuity heuristic - if we were thinking, continue thinking)
        last_mode = getattr(state, 'last_mode', None)
        entropy = float(features.get("entropy_est", 0.0))
        length = int(features.get("input_length", 0))
        
        # Phase 4: ToT mode: Highest complexity, requires ToT enabled
        if self.cfg.policy.tot_enabled and gates.get("tot_gate", False):
            # ToT mode: Tree-based reasoning
            return "tot", self.cfg.policy.tot_max_depth
        
        # Debate mode: High complexity, requires debate enabled
        if self.cfg.policy.debate_enabled and gates.get("debate_gate", False):
            # Debate mode: Generate 2 alternative thoughts
            return "debate2", 2
        
        # Think mode: Medium complexity, requires think gate or high risk
        think_threshold = self.cfg.policy.entropy_gate_think
        high_risk_threshold = 0.7  # High risk threshold
        
        if gates.get("think_gate", False) or risk_score >= high_risk_threshold:
            # Think mode: Generate 1 thought (Chain of Thought)
            return "think1", 1
        
        # Continuity heuristic: If last mode was think/debate/tot and entropy still high
        if last_mode in ("think1", "debate2", "tot") and entropy >= think_threshold * 0.8:
            # Continue with think mode for consistency
            return "think1", 1
        
        # Direct mode: Default for simple queries
        return "direct", 0

    def _decoding_from_features(
        self, 
        features: Dict[str, Any], 
        *, 
        risk_score: float,
        mode: Mode = "direct"
    ) -> DecodingConfig:
        """
        Entropi/uzunluğa göre max_new_tokens ve temperature ayarla.
        Context-aware decoding: mode, risk, entropy, and length considered.
        
        Args:
            features: Normalized features
            risk_score: Risk score (0.0-1.0)
            mode: Selected reasoning mode
            
        Returns:
            DecodingConfig optimized for the context
        """
        bounds = self.cfg.decoding_bounds
        mlo, mhi = bounds.max_new_tokens_bounds
        tlo, thi = bounds.temperature_bounds

        length = int(features.get("input_length", 0))
        ent = float(features.get("entropy_est", 0.0))

        # Mode-aware token allocation
        if mode == "tot":
            # Phase 4: ToT mode: Need more tokens for tree expansion
            # Each node needs tokens for generation
            max_new = int(mlo + (mhi - mlo) * 0.8)  # 80% of range (high for tree expansion)
        elif mode == "debate2":
            # Debate mode: Need more tokens for 2 alternative thoughts
            max_new = int(mlo + (mhi - mlo) * 0.7)  # 70% of range
        elif mode == "think1":
            # Think mode: Moderate tokens for CoT reasoning
            max_new = int(mlo + (mhi - mlo) * 0.5)  # 50% of range
        else:
            # Direct mode: Adaptive based on entropy and length
            # Higher entropy or longer input → more tokens needed
            entropy_factor = min(1.0, ent / 3.0)  # Normalize entropy to 0-1
            length_factor = min(1.0, length / 100.0)  # Normalize length
            combined_factor = (entropy_factor + length_factor) / 2.0
            max_new = int(mlo + (mhi - mlo) * combined_factor)
        
        # Temperature: Risk-aware and mode-aware
        if mode == "tot":
            # Phase 4: ToT mode: Moderate temperature for balanced exploration/exploitation
            # Need exploration for diverse branches, but also focused reasoning
            base_temp = tlo + (thi - tlo) * 0.5  # 50% of range
        elif mode in ("think1", "debate2"):
            # Reasoning modes: Lower temperature for more focused thinking
            # But not too low to allow exploration
            base_temp = tlo + (thi - tlo) * 0.4  # 40% of range
        else:
            # Direct mode: Adaptive based on risk and entropy
            # Higher risk → lower temperature (more conservative)
            # Higher entropy → slightly higher temperature (more exploration)
            risk_factor = 1.0 - risk_score  # Higher risk → lower temp
            entropy_factor = min(1.0, ent / 3.0)  # Higher entropy → higher temp
            # Balance: risk dominates, entropy adds slight exploration
            temp_factor = (risk_factor * 0.7) + (entropy_factor * 0.3)
            base_temp = tlo + (thi - tlo) * temp_factor

        # Ensure bounds
        max_new = int(max(mlo, min(mhi, max_new)))
        temperature = round(max(tlo, min(thi, base_temp)), 3)

        return DecodingConfig(
            max_new_tokens=max_new,
            temperature=temperature,
            top_p=bounds.top_p_default,
            repetition_penalty=bounds.repetition_penalty_default,
        )


# ---------------------------------------------------------------------- #
# Dahili yardımcı fonksiyonlar
# ---------------------------------------------------------------------- #

def _normalize_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Eksik anahtarları güvenli değerlerle doldurur.
    """
    out = {
        "input_length": int(raw.get("input_length", 0) or 0),
        "entropy_est": float(raw.get("entropy_est", 0.0) or 0.0),
        "needs_recent_info": bool(raw.get("needs_recent_info", False)),
        "needs_calc_or_parse": bool(raw.get("needs_calc_or_parse", False)),
        "has_claims": bool(raw.get("has_claims", False)),
        "is_sensitive_domain": bool(raw.get("is_sensitive_domain", False)),
    }
    return out


__all__ = ["PolicyRouterV2"]
