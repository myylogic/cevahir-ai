# -*- coding: utf-8 -*-
"""
Database Helper Functions
=========================

Utility functions for database operations.
"""

import uuid
from typing import Optional


def generate_uuid() -> str:
    """
    Generate UUID string for entity IDs.
    
    Returns:
        UUID string (36 characters)
    """
    return str(uuid.uuid4())


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text (Turkish-optimized).
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
        
    Note:
        Rough estimation: ~4 characters per token for Turkish
    """
    if not text:
        return 0
    # Turkish-optimized: ~4 characters per token
    return len(text) // 4

