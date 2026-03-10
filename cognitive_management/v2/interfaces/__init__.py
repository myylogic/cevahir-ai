# -*- coding: utf-8 -*-
"""
V2 Interface Definitions
========================
Tüm protocol ve interface tanımları burada.
"""

from .backend_protocols import (
    TextGenerationBackend,
    StreamingBackend,
    CachedBackend,
    ScoringBackend,
    EntropyBackend,
    MultimodalBackend,
    FullModelBackend,
)

from .component_protocols import (
    PolicyRouter,
    MemoryService,
    Critic,
    DeliberationEngine,
    ToolExecutor,
)

__all__ = [
    # Backend protocols
    "TextGenerationBackend",
    "StreamingBackend",
    "CachedBackend",
    "ScoringBackend",
    "EntropyBackend",
    "MultimodalBackend",
    "FullModelBackend",
    # Component protocols
    "PolicyRouter",
    "MemoryService",
    "Critic",
    "DeliberationEngine",
    "ToolExecutor",
]

