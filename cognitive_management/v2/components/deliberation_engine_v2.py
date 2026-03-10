# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: deliberation_engine_v2.py
Modül: cognitive_management/v2/components
Görev: V2 Deliberation Engine - Bağımsız implementasyon. V1'e bağımlı değil.
       İç ses üretimi, thought candidate generation, scoring ve selection
       işlemlerini yapar. Model API interface kullanarak deliberation işlemlerini
       gerçekleştirir.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (deliberation işlemleri),
                     Dependency Inversion (ModelAPI interface'e bağımlı)
- Design Patterns: Engine Pattern (deliberation engine)
- Endüstri Standartları: Deliberation best practices

KULLANIM:
- İç ses üretimi için
- Thought candidate generation için
- Scoring ve selection için

BAĞIMLILIKLAR:
- ModelAPI: Model interface
- Component Protocols: DeliberationEngine interface

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import List, Optional, Protocol

from cognitive_management.cognitive_types import ThoughtCandidate, DecodingConfig
from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.exceptions import DeliberationError, ValidationError
from cognitive_management.v2.interfaces.component_protocols import DeliberationEngine as IDeliberationEngine


# === Model API Protocol ======================================================

class ModelAPI(Protocol):
    """Model API interface for Deliberation."""
    def generate(self, prompt: str, decoding_cfg: DecodingConfig) -> str: ...
    def score(self, prompt: str, candidate: str) -> float: ...


class DeliberationEngineV2(IDeliberationEngine):
    """
    V2 Deliberation Engine - Bağımsız implementasyon.
    """
    
    def __init__(self, cfg: CognitiveManagerConfig, model_api: ModelAPI):
        """
        Initialize V2 Deliberation Engine.
        
        Args:
            cfg: Cognitive manager configuration
            model_api: Model API for generation
        """
        if not isinstance(cfg, CognitiveManagerConfig):
            raise ValidationError("cfg tipi geçersiz (CognitiveManagerConfig bekleniyor).")
        for fn in ("generate", "score"):
            if not hasattr(model_api, fn):
                raise ValidationError(f"ModelAPI '{fn}' metodunu sağlamıyor.")
        self.cfg = cfg
        self.mm = model_api
    
    def generate_thoughts(
        self,
        prompt: str,
        num_thoughts: int = 1,
        decoding_config: Optional[DecodingConfig] = None
    ) -> List[ThoughtCandidate]:
        """
        Generate internal thoughts using Chain of Thought (CoT) pattern.
        
        Phase 3: Advanced CoT reasoning with self-consistency support.
        
        Args:
            prompt: Input prompt (user message)
            num_thoughts: Number of thoughts to generate (1 for think1, 2 for debate2)
            decoding_config: Optional decoding configuration
            
        Returns:
            List of thought candidates with scores
        """
        try:
            if not prompt or not isinstance(prompt, str):
                raise ValidationError("prompt boş ya da geçersiz.")

            if not (1 <= int(num_thoughts) <= 4):
                raise ValidationError("num_thoughts 1 ile 4 arasında olmalı.")
            num_thoughts = int(num_thoughts)

            inner_cfg = self._derive_inner_decoding(decoding_config)
            system_prompt = self.cfg.default_system_prompt

            out: List[ThoughtCandidate] = []
            
            # Generate thoughts with different strategies
            for i in range(num_thoughts):
                # For debate mode (num_thoughts=2), use different perspectives
                if num_thoughts == 2:
                    prompt_text = self._build_debate_prompt(system_prompt, prompt, i)
                else:
                    # For think mode (num_thoughts=1), use standard CoT
                    prompt_text = self._build_inner_prompt(system_prompt, prompt)
                
                # Generate thought
                text = (self.mm.generate(prompt_text, inner_cfg) or "").strip()
                if not text:
                    # Boş üretimleri zayıf skorla geçir
                    out.append(ThoughtCandidate(text="", score=-1e9))
                    continue

                # Güvenli kırpma - CoT thoughts can be longer
                max_thought_chars = 800 if num_thoughts == 1 else 600  # Debate thoughts shorter
                if len(text) > max_thought_chars:
                    text = text[:max_thought_chars].rstrip() + " …"

                # Score the thought
                try:
                    score = float(self.mm.score(prompt_text, text))
                except Exception:
                    # Score metodu yoksa/başarısızsa, heuristik scoring
                    score = self._heuristic_score(text, prompt)

                out.append(ThoughtCandidate(text=text, score=score))

            return out

        except Exception as e:
            raise DeliberationError("İç düşünce üretimi sırasında hata.", cause=e)
    
    def _heuristic_score(self, thought: str, original_prompt: str) -> float:
        """
        Heuristic scoring for thoughts when model scoring is unavailable.
        
        Factors:
        - Length (reasonable length is better)
        - Structure (step-by-step indicators)
        - Relevance (keyword overlap with prompt)
        
        Args:
            thought: Generated thought text
            original_prompt: Original user prompt
            
        Returns:
            Heuristic score (higher is better)
        """
        if not thought:
            return -1e9
        
        score = 0.0
        
        # Length factor: Reasonable length (200-600 chars) is good
        length = len(thought)
        if 200 <= length <= 600:
            score += 0.3
        elif 100 <= length < 200 or 600 < length <= 800:
            score += 0.1
        
        # Structure factor: Step-by-step indicators
        step_indicators = ["adım", "step", "1.", "2.", "3.", "önce", "sonra", "sonuç"]
        step_count = sum(1 for indicator in step_indicators if indicator.lower() in thought.lower())
        if step_count >= 2:
            score += 0.4
        elif step_count == 1:
            score += 0.2
        
        # Relevance factor: Keyword overlap
        prompt_words = set(original_prompt.lower().split())
        thought_words = set(thought.lower().split())
        overlap = len(prompt_words & thought_words)
        if overlap > 0:
            score += 0.3 * min(1.0, overlap / max(1, len(prompt_words)))
        
        return score

    def _derive_inner_decoding(self, base: Optional[DecodingConfig]) -> DecodingConfig:
        """
        İç düşünce için temkinli üretim ayarları.
        """
        bounds = self.cfg.decoding_bounds
        tlo, thi = bounds.temperature_bounds
        mlo, mhi = bounds.max_new_tokens_bounds

        # Dışarıdan geldiyse kopyalayalım
        if isinstance(base, DecodingConfig):
            max_new = max(mlo, min(mhi, base.max_new_tokens))
            temp = max(tlo, min(thi, min(base.temperature, 0.8)))  # iç ses daha düşük sıcaklık
            top_p = base.top_p
            rep = max(1.0, min(2.0, base.repetition_penalty))
        else:
            max_new = (mlo + mhi) // 2
            temp = max(tlo, min(thi, 0.6))
            top_p = bounds.top_p_default
            rep = bounds.repetition_penalty_default

        return DecodingConfig(
            max_new_tokens=max_new,
            temperature=round(temp, 3),
            top_p=top_p,
            repetition_penalty=rep,
        )

    def _build_inner_prompt(self, system_prompt: str, user_message: str) -> str:
        """
        İç düşünce için Chain of Thought (CoT) pattern prompt.
        
        Phase 3: Advanced CoT prompting based on Wei et al. (2022).
        Encourages step-by-step reasoning.
        """
        return (
            f"[SYSTEM]\n{system_prompt}\n\n"
            "[CHAIN OF THOUGHT REASONING]\n"
            "Aşağıdaki kullanıcı isteğini çözmek için adım adım düşün:\n\n"
            "1. Önce problemi anla ve ana noktaları belirle.\n"
            "2. Gerekli bilgileri ve adımları düşün.\n"
            "3. Mantıklı bir çözüm yolu öner.\n"
            "4. Olası zorlukları veya dikkat edilmesi gerekenleri not et.\n"
            "5. Sonuç olarak net bir yaklaşım özetle.\n\n"
            f"[USER REQUEST]\n{user_message}\n\n"
            "[YOUR REASONING]\n"
            "Adım adım düşün ve her adımı açıkla:\n"
        )
    
    def _build_debate_prompt(self, system_prompt: str, user_message: str, thought_index: int) -> str:
        """
        Debate mode için alternatif düşünce prompt'u.
        
        Phase 3: Self-consistency sampling (Wang et al. 2022).
        Her alternatif farklı bir perspektiften düşünür.
        """
        perspective = "konservatif ve güvenli" if thought_index == 0 else "yaratıcı ve esnek"
        
        return (
            f"[SYSTEM]\n{system_prompt}\n\n"
            f"[ALTERNATIVE THOUGHT {thought_index + 1} - {perspective.upper()} PERSPECTIVE]\n"
            "Aşağıdaki kullanıcı isteğini çözmek için {perspective} bir yaklaşım düşün:\n\n"
            f"[USER REQUEST]\n{user_message}\n\n"
            f"[YOUR {perspective.upper()} REASONING]\n"
            "Bu perspektiften adım adım düşün:\n"
        )


__all__ = ["DeliberationEngineV2", "ModelAPI"]
