# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: constitutional_critic.py
Modül: cognitive_management/v2/components
Görev: Constitutional AI Critic - Constitutional AI patterns for self-improvement
       through principles. Phase 7.2: Advanced Critic System Enhancement.
       Constitutional principles kontrolü, self-improvement ve harmlessness
       işlemlerini yapar. Akademik referans: Anthropic (2023).

MİMARİ:
- SOLID Prensipleri: Single Responsibility (constitutional critic),
                     Dependency Inversion (interface'lere bağımlı)
- Design Patterns: Critic Pattern (constitutional critic)
- Endüstri Standartları: Constitutional AI best practices

KULLANIM:
- Constitutional principles kontrolü için
- Self-improvement için
- Harmlessness için

BAĞIMLILIKLAR:
- ConstitutionalPrinciples: Constitutional principles
- ModelAPI: Model interface
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
from typing import List, Dict, Any, Optional, Tuple, Protocol
from dataclasses import dataclass

from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.exceptions import ValidationError
from cognitive_management.cognitive_types import DecodingConfig
from cognitive_management.v2.config.constitutional_principles import get_principles


class ModelAPI(Protocol):
    """Model API interface for Constitutional Critic."""
    def generate(self, prompt: str, decoding_cfg: DecodingConfig) -> str: ...
    def score(self, prompt: str, candidate: str) -> float: ...


@dataclass
class PrincipleViolation:
    """
    Constitutional principle violation.
    
    Attributes:
        principle: The violated principle
        score: Violation score (0.0-1.0, higher = more severe)
        suggestion: Suggested fix for the violation
    """
    principle: str
    score: float
    suggestion: str


class ConstitutionalCritic:
    """
    Constitutional AI critic - self-improvement through principles.
    
    Reviews text against constitutional principles and suggests
    improvements to align with those principles.
    """
    
    def __init__(
        self,
        cfg: CognitiveManagerConfig,
        model_api: ModelAPI
    ):
        """
        Initialize Constitutional Critic.
        
        Args:
            cfg: Cognitive manager configuration
            model_api: Model API for generation
        """
        if not isinstance(cfg, CognitiveManagerConfig):
            raise ValidationError("cfg tipi geçersiz (CognitiveManagerConfig bekleniyor).")
        
        self.cfg = cfg
        self.mm = model_api
        
        # Get constitutional principles
        self.principles = get_principles(cfg.critic.custom_principles)
        self.strictness = cfg.critic.constitutional_strictness
    
    def review_with_constitution(
        self,
        user_message: str,
        draft_text: str,
        principles: Optional[List[str]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Review text against constitutional principles.
        
        Args:
            user_message: Original user message
            draft_text: Draft text to review
            principles: Optional custom principles (defaults to config)
            
        Returns:
            Tuple of (revised_text, violations_dict)
        """
        if not draft_text or not draft_text.strip():
            return draft_text, {"violations": [], "revised": False}
        
        principles = principles or self.principles
        
        # Evaluate against principles
        violations = self._evaluate_principles(draft_text, principles, user_message)
        
        if not violations:
            # No violations - return original text
            return draft_text, {"violations": [], "revised": False}
        
        # Apply constitutional revision
        revised_text = self._apply_constitutional_revision(
            draft_text,
            violations,
            user_message
        )
        
        return revised_text, {
            "violations": [
                {
                    "principle": v.principle,
                    "score": v.score,
                    "suggestion": v.suggestion,
                }
                for v in violations
            ],
            "revised": True,
        }
    
    def _evaluate_principles(
        self,
        text: str,
        principles: List[str],
        user_message: str
    ) -> List[PrincipleViolation]:
        """
        Evaluate text against constitutional principles.
        
        Args:
            text: Text to evaluate
            principles: List of principles to check
            user_message: Original user message
            
        Returns:
            List of violations
        """
        violations = []
        
        for principle in principles:
            violation_score = self._evaluate_principle(text, principle, user_message)
            
            # Apply strictness threshold
            threshold = 1.0 - self.strictness  # Higher strictness = lower threshold
            
            if violation_score > threshold:
                # Violation detected
                suggestion = self._generate_suggestion(text, principle, violation_score)
                violations.append(
                    PrincipleViolation(
                        principle=principle,
                        score=violation_score,
                        suggestion=suggestion,
                    )
                )
        
        # Sort by violation score (most severe first)
        violations.sort(key=lambda v: v.score, reverse=True)
        
        return violations
    
    def _evaluate_principle(
        self,
        text: str,
        principle: str,
        user_message: str
    ) -> float:
        """
        Evaluate text against a single principle.
        
        Returns violation score (0.0-1.0, higher = more severe violation).
        
        Args:
            text: Text to evaluate
            principle: Principle to check
            user_message: Original user message
            
        Returns:
            Violation score (0.0 = no violation, 1.0 = severe violation)
        """
        try:
            # Use model to evaluate principle adherence
            evaluation_prompt = self._build_evaluation_prompt(
                text, principle, user_message
            )
            
            from cognitive_management.cognitive_types import DecodingConfig
            response = self.mm.generate(
                evaluation_prompt,
                DecodingConfig(
                    max_new_tokens=100,
                    temperature=0.3,  # Low temperature for consistent evaluation
                    top_p=0.8,
                    repetition_penalty=1.1,
                )
            )
            
            # Parse response to get violation score
            violation_score = self._parse_violation_score(response)
            return violation_score
            
        except Exception as e:
            # Evaluation failed - use heuristic fallback
            import logging
            logging.debug(f"Constitutional evaluation failed: {e}")
            return self._evaluate_principle_heuristic(text, principle)
    
    def _build_evaluation_prompt(
        self,
        text: str,
        principle: str,
        user_message: str
    ) -> str:
        """
        Build prompt for principle evaluation.
        
        Args:
            text: Text to evaluate
            principle: Principle to check
            user_message: Original user message
            
        Returns:
            Evaluation prompt
        """
        return f"""Aşağıdaki metnin şu prensibe ne kadar uyduğunu değerlendir:

PRENSİP: {principle}

KULLANICI SORUSU:
{user_message[:300]}

CEVAP METNİ:
{text[:500]}

Bu cevap metni yukarıdaki prensibe ne kadar uygun? 
0.0-1.0 arası bir skor ver (0.0 = tamamen uygun, 1.0 = ciddi ihlal).
Sadece sayıyı yaz:"""
    
    def _parse_violation_score(self, response: str) -> float:
        """
        Parse violation score from model response.
        
        Args:
            response: Model response
            
        Returns:
            Violation score (0.0-1.0)
        """
        # Extract number from response
        import re
        numbers = re.findall(r'\d+\.?\d*', response)
        
        if numbers:
            try:
                score = float(numbers[0])
                # Normalize to 0.0-1.0 range
                if score > 1.0:
                    score = score / 10.0  # Assume scale of 10
                return min(1.0, max(0.0, score))
            except ValueError:
                pass
        
        # Fallback: parse keywords
        response_lower = response.lower()
        if "uygun" in response_lower or "compliant" in response_lower or "good" in response_lower:
            return 0.2
        elif "kısmen" in response_lower or "partial" in response_lower or "moderate" in response_lower:
            return 0.5
        elif "ihlal" in response_lower or "violation" in response_lower or "bad" in response_lower:
            return 0.8
        else:
            return 0.5  # Default: moderate
    
    def _evaluate_principle_heuristic(
        self,
        text: str,
        principle: str
    ) -> float:
        """
        Heuristic evaluation of principle adherence (fallback).
        
        Args:
            text: Text to evaluate
            principle: Principle to check
            
        Returns:
            Violation score (0.0-1.0)
        """
        text_lower = text.lower()
        principle_lower = principle.lower()
        
        # Simple keyword-based evaluation
        # Check for harmful keywords
        harmful_keywords = ["zararlı", "tehlikeli", "yasadışı", "harmful", "illegal", "dangerous"]
        if any(keyword in principle_lower for keyword in ["harmless", "zararsız", "safe", "güvenli"]):
            # Safety principle
            if any(keyword in text_lower for keyword in harmful_keywords):
                return 0.7  # Moderate violation
        
        # Check for honesty keywords
        if any(keyword in principle_lower for keyword in ["honest", "dürüst", "accurate", "doğru"]):
            # Accuracy/honesty principle
            uncertain_markers = ["belki", "muhtemelen", "sanırım", "maybe", "perhaps"]
            if not any(marker in text_lower for marker in uncertain_markers):
                # No uncertainty markers - might be too confident
                return 0.3  # Mild violation
        
        return 0.1  # Low violation score
    
    def _generate_suggestion(
        self,
        text: str,
        principle: str,
        violation_score: float
    ) -> str:
        """
        Generate suggestion for fixing principle violation.
        
        Args:
            text: Original text
            principle: Violated principle
            violation_score: Severity of violation
            
        Returns:
            Suggestion text
        """
        try:
            # Use model to generate suggestion
            suggestion_prompt = f"""Aşağıdaki metni şu prensibe uygun hale getirmek için nasıl düzeltmeliyim?

PRENSİP: {principle}

METİN:
{text[:500]}

Sadece düzeltme önerisini kısa ve net bir şekilde yaz:"""
            
            from cognitive_management.cognitive_types import DecodingConfig
            suggestion = self.mm.generate(
                suggestion_prompt,
                DecodingConfig(
                    max_new_tokens=100,
                    temperature=0.5,
                    top_p=0.9,
                    repetition_penalty=1.1,
                )
            )
            
            return suggestion.strip()
            
        except Exception:
            # Fallback to generic suggestion
            return f"Metni '{principle}' prensibine uygun hale getir."
    
    def _apply_constitutional_revision(
        self,
        draft_text: str,
        violations: List[PrincipleViolation],
        user_message: str
    ) -> str:
        """
        Apply constitutional revision based on violations.
        
        Args:
            draft_text: Original draft text
            violations: List of violations
            user_message: Original user message
            
        Returns:
            Revised text
        """
        if not violations:
            return draft_text
        
        # Get most severe violations (top 2)
        top_violations = violations[:2]
        
        try:
            # Use model to revise text
            violation_descriptions = "\n".join([
                f"- {v.principle}: {v.suggestion}"
                for v in top_violations
            ])
            
            revision_prompt = f"""Aşağıdaki metni şu prensiplere uygun hale getir:

KULLANICI SORUSU:
{user_message[:300]}

ORİJİNAL CEVAP:
{draft_text[:500]}

İHLAL EDİLEN PRENSİPLER:
{violation_descriptions}

Metni bu prensiplere uygun şekilde düzelt, ama cevabın içeriğini ve anlamını koru:"""
            
            from cognitive_management.cognitive_types import DecodingConfig
            revised = self.mm.generate(
                revision_prompt,
                DecodingConfig(
                    max_new_tokens=len(draft_text) + 100,  # Allow some expansion
                    temperature=0.4,
                    top_p=0.9,
                    repetition_penalty=1.2,
                )
            )
            
            return revised.strip()
            
        except Exception as e:
            # Revision failed - return original with warning
            import logging
            logging.warning(f"Constitutional revision failed: {e}")
            # Add simple warning prefix
            warning = "[Not: Bu cevap prensiplerimize uygun olmayabilir.] "
            return warning + draft_text


__all__ = [
    "ConstitutionalCritic",
    "PrincipleViolation",
    "ModelAPI",
]

