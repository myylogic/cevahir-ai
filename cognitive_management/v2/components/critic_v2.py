# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: critic_v2.py
Modül: cognitive_management/v2/components
Görev: V2 Critic - Bağımsız implementasyon. V1'e bağımlı değil. Çıktı sonrası
       eleştirel kontrol katmanı. Fact checking, claim extraction, validation
       ve constitutional principles kontrolü işlemlerini yapar. Advanced critic
       system enhancement ile genişletilmiş.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (critic işlemleri),
                     Dependency Inversion (interface'lere bağımlı)
- Design Patterns: Critic Pattern (eleştirel kontrol)
- Endüstri Standartları: Output validation best practices

KULLANIM:
- Çıktı validasyonu için
- Fact checking için
- Claim extraction için
- Constitutional principles kontrolü için

BAĞIMLILIKLAR:
- FactCheckers: Fact checking işlemleri
- ClaimExtraction: Claim extraction işlemleri
- Component Protocols: Critic interface

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Optional, Tuple, Protocol, Dict, List

from cognitive_management.cognitive_types import DecodingConfig
from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.exceptions import ValidationError
from cognitive_management.v2.interfaces.component_protocols import Critic as ICritic

# Phase 7.2: Advanced Critic System Enhancement
from cognitive_management.v2.components.fact_checkers import (
    create_fact_checkers,
    FactChecker,
    FactCheckResult,
)
from cognitive_management.v2.utils.claim_extraction import (
    extract_claims,
    ExtractedClaim,
)


# === Model API Protocol ======================================================

class ModelAPI(Protocol):
    """Model API interface for Critic."""
    def generate(self, prompt: str, decoding_cfg: DecodingConfig) -> str: ...
    def score(self, prompt: str, candidate: str) -> float: ...


class CriticV2(ICritic):
    """
    V2 Critic - Bağımsız implementasyon.
    
    Phase 7.2: Enhanced with:
    - External fact-checking (Wikipedia, Google, etc.)
    - Constitutional AI patterns
    - LLM-based fact verification
    """
    
    def __init__(self, cfg: CognitiveManagerConfig, model_api: ModelAPI):
        """
        Initialize V2 Critic.
        
        Phase 7.2: Enhanced with external fact-checking and Constitutional AI.
        
        Args:
            cfg: Cognitive manager configuration
            model_api: Model API for generation
        """
        if not isinstance(cfg, CognitiveManagerConfig):
            raise ValidationError("cfg tipi geçersiz (CognitiveManagerConfig bekleniyor).")
        # Basit arayüz doğrulaması
        for fn in ("generate", "score"):
            if not hasattr(model_api, fn):
                raise ValidationError(f"ModelAPI '{fn}' metodunu sağlamıyor.")
        self.cfg = cfg
        self.mm = model_api
        
        # Phase 7.2: External fact-checking services
        self._fact_checkers: List[FactChecker] = []
        if cfg.critic.enable_external_fact_checking:
            try:
                self._fact_checkers = create_fact_checkers(cfg)
            except Exception as e:
                # Log warning but don't fail initialization
                import logging
                logging.warning(f"Fact checker initialization başarısız: {e}")
                self._fact_checkers = []
        
        # Phase 7.2: Constitutional AI (lazy initialization)
        self._constitutional_critic = None
    
    def review(
        self,
        user_message: str,
        draft_text: str,
        context: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Review and revise text using multi-aspect evaluation.
        
        Phase 7.2: Enhanced with:
        - External fact-checking (Wikipedia, etc.)
        - Constitutional AI patterns
        - LLM-based fact verification
        - Multi-aspect scoring
        
        Args:
            user_message: Original user message
            draft_text: Draft response text
            context: Optional context
            
        Returns:
            Tuple of (final_text, was_revised)
        """
        if not self.cfg.critic.enabled:
            return draft_text, False

        if not draft_text or not draft_text.strip():
            return draft_text, False

        # Phase 7.2: Apply Constitutional AI review first (if enabled)
        if self.cfg.critic.enable_constitutional_ai:
            try:
                # Lazy initialize constitutional critic
                if self._constitutional_critic is None:
                    from cognitive_management.v2.components.constitutional_critic import ConstitutionalCritic
                    self._constitutional_critic = ConstitutionalCritic(self.cfg, self.mm)
                
                # Review with constitutional principles
                constitutional_result, violations_dict = self._constitutional_critic.review_with_constitution(
                    user_message=user_message,
                    draft_text=draft_text,
                )
                
                # Use constitutionally reviewed text for further evaluation
                if violations_dict.get("revised", False):
                    draft_text = constitutional_result
            except Exception as e:
                # Constitutional review failed - continue with standard review
                import logging
                logging.warning(f"Constitutional review başarısız, standart review kullanılıyor: {e}")

        # Multi-aspect evaluation
        scores = self._evaluate_aspects(user_message, draft_text, context)
        
        # Determine if revision is needed
        needs_revision = self._should_revise(scores)
        
        if not needs_revision:
            return draft_text, False
        
        # Perform revision
        revised_text = self._revise_text(user_message, draft_text, scores, context)
        
        return revised_text, True
    
    def _evaluate_aspects(
        self,
        user_message: str,
        draft_text: str,
        context: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate draft text across multiple aspects.
        
        Returns:
            Dictionary with aspect scores (0.0-1.0, higher is better)
        """
        scores = {}
        
        # 1. Length check (0.0-1.0)
        scores["length"] = self._check_length(draft_text)
        
        # 2. Coherence check (0.0-1.0)
        scores["coherence"] = self._check_coherence(draft_text)
        
        # 3. Safety check (0.0-1.0)
        scores["safety"] = self._check_safety(user_message, draft_text)
        
        # 4. Fact-checking (basic) (0.0-1.0)
        scores["factuality"] = self._check_facts(draft_text)
        
        # 5. Style consistency (0.0-1.0)
        scores["style"] = self._check_style(draft_text)
        
        # 6. Relevance (0.0-1.0)
        scores["relevance"] = self._check_relevance(user_message, draft_text)
        
        return scores
    
    def _check_length(self, text: str) -> float:
        """
        Check if text length is appropriate.
        
        Returns:
            Score: 1.0 if good length, decreases for too short/long
        """
        length = len(text)
        
        # Optimal range: 50-500 chars
        if 50 <= length <= 500:
            return 1.0
        elif 20 <= length < 50:
            # Too short
            return 0.5 + 0.5 * (length - 20) / 30
        elif 500 < length <= 1000:
            # Too long but acceptable
            return 1.0 - 0.3 * (length - 500) / 500
        elif length > 1000:
            # Very long
            return 0.2
        else:
            # Very short
            return 0.1
    
    def _check_coherence(self, text: str) -> float:
        """
        Check text coherence (flow, structure, readability).
        
        Returns:
            Score based on coherence indicators
        """
        if not text or len(text) < 10:
            return 0.3
        
        score = 0.5  # Base score
        
        # Positive indicators
        sentence_endings = text.count('.') + text.count('!') + text.count('?')
        if sentence_endings >= 1:
            score += 0.2
        
        # Check for basic structure
        has_capitalization = any(c.isupper() for c in text[:10])
        if has_capitalization:
            score += 0.1
        
        # Check for reasonable word count
        words = text.split()
        if 5 <= len(words) <= 100:
            score += 0.2
        
        return min(1.0, score)
    
    def _check_safety(self, user_message: str, draft_text: str) -> float:
        """
        Basic safety check (harmful content detection).
        
        Returns:
            Score: 1.0 if safe, lower if potentially harmful
        """
        # Safety keywords (basic - can be expanded)
        harmful_keywords = [
            "zararlı", "tehlikeli", "yasadışı", "şiddet", "nefret",
            "discrimination", "violence", "illegal", "harmful"
        ]
        
        text_lower = draft_text.lower()
        user_lower = user_message.lower()
        
        # Check for harmful patterns
        harmful_count = sum(1 for keyword in harmful_keywords if keyword in text_lower)
        
        if harmful_count == 0:
            return 1.0
        elif harmful_count == 1:
            return 0.7  # Warning but not critical
        else:
            return 0.3  # Multiple harmful indicators
    
    def _check_facts(self, text: str) -> float:
        """
        Enhanced fact-checking with external APIs and claim extraction.
        
        Phase 7.2: Enhanced with:
        - Claim extraction
        - External fact-checking (Wikipedia, etc.)
        - LLM-based verification
        
        Returns:
            Score based on factuality (0.0-1.0, higher is better)
        """
        # Phase 7.2: Extract claims from text
        if self.cfg.critic.enable_external_fact_checking and self._fact_checkers:
            return self._check_facts_enhanced(text)
        
        # Fallback to basic keyword-based fact-checking
        return self._check_facts_basic(text)
    
    def _check_facts_basic(self, text: str) -> float:
        """
        Basic keyword-based fact-checking (fallback).
        
        Returns:
            Score based on factuality indicators
        """
        # Factual claim markers
        claim_markers = ["%", "oran", "istatistik", "kanıt", "araştırma", "çalışma", "kesinlikle"]
        uncertain_markers = ["belki", "muhtemelen", "sanırım", "olabilir", "maybe", "perhaps"]
        unverified_markers = ["kaynak yok", "kaynak", "doğrulanmamış", "unverified", "no source"]
        
        text_lower = text.lower()
        
        # Check for unverified claims markers (strong indicator of low factuality)
        has_unverified = any(marker in text_lower for marker in unverified_markers)
        if has_unverified:
            # Explicitly unverified claims - low factuality score
            return 0.5
        
        # If contains claims but no uncertainty markers, might need fact-checking
        has_claims = any(marker in text_lower for marker in claim_markers)
        has_uncertainty = any(marker in text_lower for marker in uncertain_markers)
        
        if has_claims and not has_uncertainty:
            # Claims without uncertainty - might be factual but unverified
            return 0.7
        elif has_claims and has_uncertainty:
            # Claims with uncertainty - appropriate hedging
            return 0.9
        else:
            # No strong claims
            return 1.0
    
    def _check_facts_enhanced(self, text: str) -> float:
        """
        Enhanced fact-checking with external services.
        
        Phase 7.2: Uses claim extraction and external fact-checkers.
        
        Args:
            text: Text to fact-check
            
        Returns:
            Factuality score (0.0-1.0)
        """
        # Extract claims from text
        min_confidence = self.cfg.critic.claim_extraction_min_confidence
        claims = extract_claims(text, min_confidence=min_confidence)
        
        if not claims:
            # No claims found - default to high score
            return 1.0
        
        # Verify claims with external fact-checkers
        verified_count = 0
        total_claims = len(claims)
        
        for claim_obj in claims:
            claim_text = claim_obj.claim
            
            # Try each fact checker
            verified = False
            for checker in self._fact_checkers:
                try:
                    result = checker.verify(claim_text)
                    if result and result.is_verified is True:
                        # Claim verified
                        verified = True
                        break
                except Exception:
                    # Fact checker error - continue to next checker
                    continue
            
            if verified:
                verified_count += 1
        
        # Calculate factuality score
        if total_claims == 0:
            return 1.0
        
        verification_rate = verified_count / total_claims
        
        # If some claims verified, increase score
        # If no claims verified but we tried, decrease score slightly
        if verified_count > 0:
            # At least some claims verified
            base_score = 0.7 + (verification_rate * 0.3)  # 0.7-1.0
        else:
            # No claims verified - might be unverified or uncertain
            base_score = 0.6
        
        # Phase 7.2: Also use LLM-based verification if enabled
        if self.cfg.critic.enable_llm_fact_verification:
            llm_score = self._check_facts_llm(text, claims)
            # Combine scores (weighted average)
            combined_score = (base_score * 0.6) + (llm_score * 0.4)
            return combined_score
        
        return base_score
    
    def _check_facts_llm(self, text: str, claims: List[ExtractedClaim]) -> float:
        """
        LLM-based fact verification.
        
        Uses the model itself to verify factual claims.
        
        Args:
            text: Text to verify
            claims: Extracted claims
            
        Returns:
            Factuality score from LLM verification
        """
        if not claims:
            return 1.0
        
        try:
            # Build verification prompt
            claims_text = "\n".join([f"- {c.claim}" for c in claims[:3]])  # Limit to top 3
            
            verification_prompt = f"""Aşağıdaki metinde geçen iddiaların doğruluğunu değerlendir:

Metin:
{text[:500]}

İddialar:
{claims_text}

Bu iddiaların genel olarak doğru, yanlış veya belirsiz olduğunu değerlendir. 
Yanıtını "DOĞRU", "YANLIŞ", "BELİRSİZ" veya "KISMEN DOĞRU" olarak ver."""
            
            # Use model to verify
            from cognitive_management.cognitive_types import DecodingConfig
            response = self.mm.generate(
                verification_prompt,
                DecodingConfig(
                    max_new_tokens=50,
                    temperature=0.3,  # Low temperature for factual assessment
                    top_p=0.8,
                    repetition_penalty=1.1,
                )
            )
            
            response_lower = response.lower()
            
            # Score based on response
            if "doğru" in response_lower or "correct" in response_lower:
                return 0.9
            elif "kısmen" in response_lower or "partial" in response_lower:
                return 0.7
            elif "belirsiz" in response_lower or "uncertain" in response_lower or "unknown" in response_lower:
                return 0.5
            elif "yanlış" in response_lower or "false" in response_lower or "incorrect" in response_lower:
                return 0.2
            else:
                # Default to medium score if unclear
                return 0.6
                
        except Exception as e:
            # LLM verification failed - return neutral score
            import logging
            logging.debug(f"LLM fact verification failed: {e}")
            return 0.6
    
    def _check_style(self, text: str) -> float:
        """
        Check style consistency (tone, formality, etc.).
        
        Returns:
            Score based on style consistency
        """
        if not text or len(text) < 10:
            return 0.5
        
        score = 0.7  # Base score
        
        # Check for consistent punctuation
        has_punctuation = any(c in text for c in '.!?,')
        if has_punctuation:
            score += 0.2
        
        # Check for reasonable sentence structure
        sentences = text.split('.')
        if 1 <= len(sentences) <= 10:
            score += 0.1
        
        return min(1.0, score)
    
    def _check_relevance(self, user_message: str, draft_text: str) -> float:
        """
        Check if response is relevant to user message.
        
        Returns:
            Score based on keyword overlap and semantic relevance
        """
        if not user_message or not draft_text:
            return 0.5
        
        # Simple keyword overlap
        user_words = set(user_message.lower().split())
        draft_words = set(draft_text.lower().split())
        
        if not user_words:
            return 0.5
        
        overlap = len(user_words & draft_words)
        overlap_ratio = overlap / len(user_words)
        
        # Relevance score based on overlap
        if overlap_ratio >= 0.3:
            return 1.0
        elif overlap_ratio >= 0.1:
            return 0.5 + 0.5 * (overlap_ratio - 0.1) / 0.2
        else:
            return 0.3
    
    def _should_revise(self, scores: Dict[str, float]) -> bool:
        """
        Determine if revision is needed based on scores.
        
        Args:
            scores: Dictionary of aspect scores
            
        Returns:
            True if revision is needed
        """
        # Critical aspects (must be above threshold)
        critical_threshold = 0.5
        critical_aspects = ["safety", "coherence"]
        
        for aspect in critical_aspects:
            if scores.get(aspect, 1.0) < critical_threshold:
                return True
        
        # Factuality threshold: if factuality is low, revision is needed
        factuality_score = scores.get("factuality", 1.0)
        if factuality_score < 0.8:
            return True
        
        # Overall score threshold
        overall_score = sum(scores.values()) / len(scores)
        if overall_score < 0.6:
            return True
        
        return False
    
    def _revise_text(
        self,
        user_message: str,
        draft_text: str,
        scores: Dict[str, float],
        context: Optional[str] = None
    ) -> str:
        """
        Revise text based on evaluation scores.
        
        Args:
            user_message: Original user message
            draft_text: Draft text to revise
            scores: Evaluation scores
            context: Optional context
            
        Returns:
            Revised text
        """
        revised = draft_text
        
        # Length revision
        if scores.get("length", 1.0) < 0.5:
            if len(revised) > 500:
                # Too long - truncate intelligently
                revised = revised[:500].rsplit('.', 1)[0] + '.'
            elif len(revised) < 20:
                # Too short - add context
                revised = f"{revised} (Bu konuda daha fazla bilgi verebilirim.)"
        
        # Safety revision
        if scores.get("safety", 1.0) < 0.7:
            # Add disclaimer for potentially harmful content
            revised = f"[Not: Bu konuda dikkatli olunmalıdır.] {revised}"
        
        # Coherence revision
        if scores.get("coherence", 1.0) < 0.6:
            # Ensure proper sentence structure
            if not revised.endswith(('.', '!', '?')):
                revised += '.'
        
        # Factuality revision
        if scores.get("factuality", 1.0) < 0.8:
            # Add uncertainty markers for unverified claims
            if not any(word in revised.lower() for word in ["belki", "muhtemelen", "sanırım"]):
                revised = f"Muhtemelen {revised.lower()}"
        
        return revised


__all__ = ["CriticV2", "ModelAPI"]
