# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: claim_extraction.py
Modül: cognitive_management/v2/utils
Görev: Claim Extraction Utility - Extract factual claims from text for fact-checking.
       Phase 7.2: Advanced Critic System Enhancement. ExtractedClaim, extract_claims
       fonksiyonunu içerir. Factual claim extraction, claim type classification
       ve claim confidence scoring işlemlerini yapar. Akademik referans: Thorne
       et al. (2018).

MİMARİ:
- SOLID Prensipleri: Single Responsibility (claim extraction)
- Design Patterns: Extractor Pattern (claim extraction)
- Endüstri Standartları: Claim extraction best practices

KULLANIM:
- Claim extraction için
- Fact-checking için
- Claim type classification için

BAĞIMLILIKLAR:
- re: Regular expression işlemleri
- dataclasses: Claim data structures

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass


@dataclass
class ExtractedClaim:
    """
    Extracted factual claim from text.
    
    Attributes:
        claim: The factual claim text
        confidence: Confidence score (0.0-1.0)
        claim_type: Type of claim ("statistical", "factual", "temporal", "comparative", etc.)
        context: Surrounding context
    """
    claim: str
    confidence: float
    claim_type: str
    context: Optional[str] = None


class ClaimExtractor:
    """
    Extract factual claims from text.
    
    Uses pattern-based extraction with confidence scoring.
    Can be enhanced with NLP models in the future.
    """
    
    # Claim markers (patterns that indicate factual claims)
    STATISTICAL_MARKERS = [
        r'\d+%', r'\d+\s*percent', r'%', r'oran', r'istatistik', r'istatistiksel',
        r'\d+/\d+', r'out of \d+', r'\d+ out of \d+',
        r'\d+\.\d+', r'\d+,\d+',  # Decimal numbers
        r'increase', r'decrease', r'growth', r'artış', r'azalış',
    ]
    
    FACTUAL_MARKERS = [
        r'is', r'are', r'was', r'were', r'becomes', r'became',
        r'tür', r'olmak', r'dır', r'dir', r'dur', r'dür',
        r'according to', r'göre', r'gösteriyor',
        r'study', r'araştırma', r'çalışma', r'research',
        r'found', r'bulundu', r'gösterdi',
        r'evidence', r'kanıt', r'delil',
    ]
    
    TEMPORAL_MARKERS = [
        r'in \d{4}', r'\d{4} yılında', r'\d{4}',
        r'last year', r'geçen yıl', r'bu yıl', r'this year',
        r'recently', r'yakın zamanda', r'son zamanlarda',
        r'now', r'şimdi', r'currently', r'şu anda',
    ]
    
    COMPARATIVE_MARKERS = [
        r'more than', r'less than', r'daha fazla', r'daha az',
        r'compared to', r'karşılaştırıldığında', r'kıyasla',
        r'higher', r'lower', r'daha yüksek', r'daha düşük',
        r'better', r'worse', r'daha iyi', r'daha kötü',
    ]
    
    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize claim extractor.
        
        Args:
            min_confidence: Minimum confidence score for extracted claims
        """
        self.min_confidence = min_confidence
        # Compile regex patterns
        self._statistical_patterns = [re.compile(marker, re.IGNORECASE) for marker in self.STATISTICAL_MARKERS]
        self._factual_patterns = [re.compile(marker, re.IGNORECASE) for marker in self.FACTUAL_MARKERS]
        self._temporal_patterns = [re.compile(marker, re.IGNORECASE) for marker in self.TEMPORAL_MARKERS]
        self._comparative_patterns = [re.compile(marker, re.IGNORECASE) for marker in self.COMPARATIVE_MARKERS]
    
    def extract_claims(self, text: str) -> List[ExtractedClaim]:
        """
        Extract factual claims from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted claims with confidence scores
        """
        if not text or not text.strip():
            return []
        
        claims = []
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            sentence_claims = self._extract_from_sentence(sentence, text)
            claims.extend(sentence_claims)
        
        # Filter by confidence threshold
        claims = [c for c in claims if c.confidence >= self.min_confidence]
        
        # Remove duplicates (by claim text)
        seen = set()
        unique_claims = []
        for claim in claims:
            claim_key = claim.claim.lower().strip()
            if claim_key not in seen:
                seen.add(claim_key)
                unique_claims.append(claim)
        
        return unique_claims
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be enhanced with NLTK/spaCy)
        # Split by sentence endings
        sentences = re.split(r'[.!?]+', text)
        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _extract_from_sentence(self, sentence: str, full_text: str) -> List[ExtractedClaim]:
        """Extract claims from a single sentence."""
        claims = []
        
        # Check for statistical claims
        statistical_claims = self._extract_statistical_claims(sentence, full_text)
        claims.extend(statistical_claims)
        
        # Check for factual claims
        factual_claims = self._extract_factual_claims(sentence, full_text)
        claims.extend(factual_claims)
        
        # Check for temporal claims
        temporal_claims = self._extract_temporal_claims(sentence, full_text)
        claims.extend(temporal_claims)
        
        # Check for comparative claims
        comparative_claims = self._extract_comparative_claims(sentence, full_text)
        claims.extend(comparative_claims)
        
        return claims
    
    def _extract_statistical_claims(self, sentence: str, full_text: str) -> List[ExtractedClaim]:
        """Extract statistical claims."""
        claims = []
        
        # Check for statistical markers
        for pattern in self._statistical_patterns:
            if pattern.search(sentence):
                # Found statistical marker
                confidence = 0.8  # High confidence for statistical claims
                claim_type = "statistical"
                
                # Extract the claim (full sentence or relevant part)
                claim_text = sentence.strip()
                
                claims.append(ExtractedClaim(
                    claim=claim_text,
                    confidence=confidence,
                    claim_type=claim_type,
                    context=full_text[:200] if len(full_text) > 200 else full_text,
                ))
                break  # Only extract one statistical claim per sentence
        
        return claims
    
    def _extract_factual_claims(self, sentence: str, full_text: str) -> List[ExtractedClaim]:
        """Extract factual claims."""
        claims = []
        
        # Check for factual markers
        matches = []
        for pattern in self._factual_patterns:
            if pattern.search(sentence):
                matches.append(pattern)
        
        if matches:
            # Found factual markers
            confidence = 0.7  # Medium-high confidence
            claim_type = "factual"
            
            claim_text = sentence.strip()
            
            claims.append(ExtractedClaim(
                claim=claim_text,
                confidence=confidence,
                claim_type=claim_type,
                context=full_text[:200] if len(full_text) > 200 else full_text,
            ))
        
        return claims
    
    def _extract_temporal_claims(self, sentence: str, full_text: str) -> List[ExtractedClaim]:
        """Extract temporal claims."""
        claims = []
        
        # Check for temporal markers
        for pattern in self._temporal_patterns:
            if pattern.search(sentence):
                confidence = 0.6  # Medium confidence
                claim_type = "temporal"
                
                claim_text = sentence.strip()
                
                claims.append(ExtractedClaim(
                    claim=claim_text,
                    confidence=confidence,
                    claim_type=claim_type,
                    context=full_text[:200] if len(full_text) > 200 else full_text,
                ))
                break
        
        return claims
    
    def _extract_comparative_claims(self, sentence: str, full_text: str) -> List[ExtractedClaim]:
        """Extract comparative claims."""
        claims = []
        
        # Check for comparative markers
        for pattern in self._comparative_patterns:
            if pattern.search(sentence):
                confidence = 0.65  # Medium confidence
                claim_type = "comparative"
                
                claim_text = sentence.strip()
                
                claims.append(ExtractedClaim(
                    claim=claim_text,
                    confidence=confidence,
                    claim_type=claim_type,
                    context=full_text[:200] if len(full_text) > 200 else full_text,
                ))
                break
        
        return claims


def extract_claims(text: str, min_confidence: float = 0.5) -> List[ExtractedClaim]:
    """
    Convenience function to extract claims from text.
    
    Args:
        text: Input text
        min_confidence: Minimum confidence score
        
    Returns:
        List of extracted claims
    """
    extractor = ClaimExtractor(min_confidence=min_confidence)
    return extractor.extract_claims(text)


__all__ = [
    "ExtractedClaim",
    "ClaimExtractor",
    "extract_claims",
]

