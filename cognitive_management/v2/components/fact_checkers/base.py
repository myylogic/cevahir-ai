# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: base.py
Modül: cognitive_management/v2/components/fact_checkers
Görev: Fact Checker Base Interface - Base interface and abstract implementation
       for fact checkers. Phase 7.2: Advanced Critic System Enhancement.
       FactCheckResult, BaseFactChecker ve FactChecker interface tanımlarını
       içerir. Akademik referans: Thorne et al. (2018).

MİMARİ:
- SOLID Prensipleri: Single Responsibility (fact checker interface),
                     Dependency Inversion (interface'lere bağımlı)
- Design Patterns: Interface Pattern (fact checker interface)
- Endüstri Standartları: Fact checking best practices

KULLANIM:
- Fact checker interface tanımları için
- Base fact checker implementation için
- Fact check result için

BAĞIMLILIKLAR:
- abc: Abstract base classes
- dataclasses: Dataclass tanımları

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Protocol, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class FactCheckResult:
    """
    Result from fact-checking a claim.
    
    Attributes:
        claim: The claim that was checked
        is_verified: Whether the claim was verified (True/False/None for uncertain)
        confidence: Confidence score (0.0-1.0)
        evidence: Supporting or contradicting evidence (if available)
        source: Source of the fact-check (e.g., "wikipedia", "google")
    """
    claim: str
    is_verified: Optional[bool]  # True = verified, False = refuted, None = uncertain
    confidence: float
    evidence: Optional[str] = None
    source: str = "unknown"


class FactChecker(Protocol):
    """
    Protocol for fact checker implementations.
    
    Fact checkers verify factual claims using external sources.
    """
    
    def verify(self, claim: str) -> Optional[FactCheckResult]:
        """
        Verify a factual claim.
        
        Args:
            claim: The claim to verify
            
        Returns:
            FactCheckResult or None if verification failed/unavailable
        """
        ...
    
    def verify_batch(self, claims: List[str]) -> List[Optional[FactCheckResult]]:
        """
        Verify multiple claims in batch.
        
        Args:
            claims: List of claims to verify
            
        Returns:
            List of FactCheckResult (None for failed/unavailable)
        """
        ...


class BaseFactChecker(ABC):
    """
    Abstract base class for fact checker implementations.
    
    Provides common functionality and enforces interface.
    """
    
    def __init__(self, source_name: str):
        """
        Initialize base fact checker.
        
        Args:
            source_name: Name of the fact-checking source
        """
        self.source_name = source_name
    
    @abstractmethod
    def verify(self, claim: str) -> Optional[FactCheckResult]:
        """
        Verify a factual claim.
        
        Args:
            claim: The claim to verify
            
        Returns:
            FactCheckResult or None
        """
        pass
    
    def verify_batch(self, claims: List[str]) -> List[Optional[FactCheckResult]]:
        """
        Verify multiple claims (default: sequential verification).
        
        Args:
            claims: List of claims to verify
            
        Returns:
            List of FactCheckResult
        """
        results = []
        for claim in claims:
            result = self.verify(claim)
            results.append(result)
        return results
    
    def _normalize_claim(self, claim: str) -> str:
        """
        Normalize claim text for verification.
        
        Args:
            claim: Raw claim text
            
        Returns:
            Normalized claim text
        """
        # Remove extra whitespace
        claim = ' '.join(claim.split())
        # Remove leading/trailing punctuation
        claim = claim.strip('.,;:!?')
        return claim


__all__ = [
    "FactChecker",
    "FactCheckResult",
    "BaseFactChecker",
]

