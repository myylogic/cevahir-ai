# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: wikipedia_checker.py
Modül: cognitive_management/v2/components/fact_checkers
Görev: Wikipedia Fact Checker - Wikipedia API integration for fact-checking.
       Phase 7.2: Advanced Critic System Enhancement. Uses Wikipedia API to
       search for relevant articles and verify claims through cross-reference
       checking. Wikipedia-based fact verification sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (Wikipedia fact checking),
                     Dependency Inversion (BaseFactChecker interface'e bağımlı)
- Design Patterns: Checker Pattern (Wikipedia fact checking)
- Endüstri Standartları: Fact checking best practices

KULLANIM:
- Wikipedia-based fact checking için
- Claim verification için
- Cross-reference checking için

BAĞIMLILIKLAR:
- BaseFactChecker: Base fact checker interface
- urllib: Wikipedia API işlemleri
- json: JSON işlemleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
import urllib.parse
import urllib.request
import json

from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.exceptions import ValidationError
from .base import BaseFactChecker, FactCheckResult


class WikipediaFactChecker(BaseFactChecker):
    """
    Wikipedia-based fact checker.
    
    Uses Wikipedia API to search for relevant articles and verify claims.
    Cross-references claims with Wikipedia content to assess factuality.
    """
    
    WIKIPEDIA_API_BASE = "https://en.wikipedia.org/api/rest_v1"
    WIKIPEDIA_SEARCH_API = f"{WIKIPEDIA_API_BASE}/page/summary"
    
    def __init__(self, cfg: CognitiveManagerConfig):
        """
        Initialize Wikipedia fact checker.
        
        Args:
            cfg: Cognitive manager configuration
        """
        super().__init__(source_name="wikipedia")
        self.cfg = cfg
        self.max_results = cfg.critic.wikipedia_max_results or 3
        self.timeout = 5.0  # Request timeout in seconds
    
    def verify(self, claim: str) -> Optional[FactCheckResult]:
        """
        Verify claim using Wikipedia.
        
        Searches Wikipedia for relevant articles and checks if claim
        can be verified or contradicted based on article content.
        
        Args:
            claim: The claim to verify
            
        Returns:
            FactCheckResult or None if verification unavailable
        """
        if not claim or not claim.strip():
            return None
        
        normalized_claim = self._normalize_claim(claim)
        
        try:
            # Extract key terms from claim for Wikipedia search
            search_terms = self._extract_search_terms(normalized_claim)
            
            if not search_terms:
                return None
            
            # Search Wikipedia for relevant pages
            relevant_pages = self._search_wikipedia(search_terms)
            
            if not relevant_pages:
                # No relevant pages found
                return FactCheckResult(
                    claim=normalized_claim,
                    is_verified=None,
                    confidence=0.3,  # Low confidence - no evidence found
                    evidence=None,
                    source=self.source_name,
                )
            
            # Verify claim against Wikipedia content
            verification = self._verify_against_pages(normalized_claim, relevant_pages)
            
            return verification
            
        except Exception as e:
            # Error during verification - return uncertain result
            import logging
            logging.warning(f"Wikipedia fact-checking error: {e}")
            return FactCheckResult(
                claim=normalized_claim,
                is_verified=None,
                confidence=0.2,  # Very low confidence due to error
                evidence=None,
                source=self.source_name,
            )
    
    def _extract_search_terms(self, claim: str) -> List[str]:
        """
        Extract key search terms from claim.
        
        Args:
            claim: The claim text
            
        Returns:
            List of search terms
        """
        # Simple extraction: take important words (nouns, proper nouns)
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "and", "or", "but", "in", "on", "at", "to", "for", "of",
            "with", "by", "from", "as", "this", "that", "these", "those",
            "bir", "bu", "şu", "o", "ve", "ile", "için", "gibi", "kadar",
        }
        
        # Split into words and filter
        words = claim.lower().split()
        # Remove punctuation and filter
        words = [w.strip('.,;:!?()[]{}"\'') for w in words]
        search_terms = [w for w in words if w and w not in stop_words and len(w) > 2]
        
        # Limit to top 3 terms
        return search_terms[:3]
    
    def _search_wikipedia(self, search_terms: List[str]) -> List[Dict[str, Any]]:
        """
        Search Wikipedia for relevant pages.
        
        Args:
            search_terms: List of search terms
            
        Returns:
            List of relevant Wikipedia page summaries
        """
        if not search_terms:
            return []
        
        # Use first term for primary search
        primary_term = search_terms[0]
        
        # URL encode the search term
        encoded_term = urllib.parse.quote(primary_term)
        
        try:
            # Search Wikipedia using search API
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/search/{encoded_term}?limit={self.max_results}"
            
            # Make request
            request = urllib.request.Request(
                search_url,
                headers={"User-Agent": "CognitiveManagement/1.0"}
            )
            
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode())
            
            # Extract relevant pages
            pages = []
            if isinstance(data, dict) and "pages" in data:
                pages = data["pages"]
            elif isinstance(data, list):
                pages = data
            
            # Limit to max_results
            return pages[:self.max_results]
            
        except Exception as e:
            # Search failed - return empty
            import logging
            logging.debug(f"Wikipedia search failed: {e}")
            return []
    
    def _verify_against_pages(
        self,
        claim: str,
        pages: List[Dict[str, Any]]
    ) -> FactCheckResult:
        """
        Verify claim against Wikipedia page content.
        
        Args:
            claim: The claim to verify
            pages: List of Wikipedia page summaries
            
        Returns:
            FactCheckResult
        """
        if not pages:
            return FactCheckResult(
                claim=claim,
                is_verified=None,
                confidence=0.3,
                evidence=None,
                source=self.source_name,
            )
        
        # Extract text from page summaries
        page_texts = []
        for page in pages:
            extract = page.get("extract", "")
            title = page.get("title", "")
            if extract:
                page_texts.append(f"{title}: {extract}")
        
        if not page_texts:
            return FactCheckResult(
                claim=claim,
                is_verified=None,
                confidence=0.3,
                evidence=None,
                source=self.source_name,
            )
        
        # Combine all page texts
        combined_text = " ".join(page_texts).lower()
        claim_lower = claim.lower()
        
        # Extract key terms from claim
        claim_terms = set(self._extract_search_terms(claim))
        
        # Check for term matches in Wikipedia content
        matching_terms = sum(1 for term in claim_terms if term in combined_text)
        
        # Calculate verification score
        if matching_terms == 0:
            # No matching terms - uncertain
            is_verified = None
            confidence = 0.4
        elif matching_terms >= len(claim_terms) * 0.5:
            # Good match - likely verified
            is_verified = True
            confidence = 0.7 + (matching_terms / len(claim_terms)) * 0.2
            confidence = min(0.95, confidence)  # Cap at 0.95
        else:
            # Partial match - uncertain
            is_verified = None
            confidence = 0.5
        
        # Extract evidence snippet
        evidence = None
        if page_texts:
            # Use first page extract as evidence
            first_extract = page_texts[0]
            if len(first_extract) > 200:
                evidence = first_extract[:200] + "..."
            else:
                evidence = first_extract
        
        return FactCheckResult(
            claim=claim,
            is_verified=is_verified,
            confidence=confidence,
            evidence=evidence,
            source=self.source_name,
        )


__all__ = ["WikipediaFactChecker"]

