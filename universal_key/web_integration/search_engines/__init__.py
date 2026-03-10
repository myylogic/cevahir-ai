# -*- coding: utf-8 -*-
"""
Search Engines Module
=====================

Arama motorları entegrasyonu için modül.
"""

from .google_search import GoogleSearch
from .duckduckgo_search import DuckDuckGoSearch
from .bing_search import BingSearch
from .academic_search import AcademicSearch

__all__ = [
    "GoogleSearch",
    "DuckDuckGoSearch",
    "BingSearch", 
    "AcademicSearch"
]
