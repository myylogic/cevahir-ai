# -*- coding: utf-8 -*-
"""
V2 Configuration Package
========================
Advanced configuration management for V2 cognitive management.
"""

from .config_manager import ConfigManager, ConfigChangeEvent

# Phase 7.2: Constitutional AI Principles
from .constitutional_principles import (
    DEFAULT_CONSTITUTIONAL_PRINCIPLES,
    get_principles,
)

__all__ = [
    "ConfigManager",
    "ConfigChangeEvent",
    # Phase 7.2: Constitutional AI
    "DEFAULT_CONSTITUTIONAL_PRINCIPLES",
    "get_principles",
]

