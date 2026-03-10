# -*- coding: utf-8 -*-
"""
Fact Checker Components
=======================
External fact-checking integrations for enhanced critic system.

Phase 7.2: Advanced Critic System Enhancement
"""

from typing import List
from cognitive_management.config import CognitiveManagerConfig

# Import base interface and implementations
from .base import FactChecker, FactCheckResult
from .wikipedia_checker import WikipediaFactChecker

__all__ = [
    "FactChecker",
    "FactCheckResult",
    "WikipediaFactChecker",
    "create_fact_checkers",
]


def create_fact_checkers(cfg: CognitiveManagerConfig) -> List[FactChecker]:
    """
    Factory function to create fact checkers based on configuration.
    
    Args:
        cfg: Cognitive manager configuration
        
    Returns:
        List of fact checker instances
    """
    checkers = []
    
    if not cfg.critic.enable_external_fact_checking:
        return checkers
    
    providers = cfg.critic.fact_checking_providers
    
    if "wikipedia" in providers and cfg.critic.enable_wikipedia:
        checkers.append(WikipediaFactChecker(cfg))
    
    if "google" in providers and cfg.critic.enable_google_fact_check:
        # TODO: Phase 7.2 - Google Fact Check API implementation
        pass
    
    if "wolfram" in providers and cfg.critic.enable_wolfram:
        # TODO: Phase 7.2 - Wolfram Alpha implementation
        pass
    
    return checkers

