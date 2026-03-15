# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: async_handlers.py
Modül: cognitive_management/v2/processing
Görev: Async Processing Handlers - Async Chain of Responsibility pattern handler
       implementations. Her handler tek bir sorumluluğa sahip (SRP) - async
       versiyonlar. Async handler implementations, async feature extraction,
       async policy routing ve async deliberation işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (her handler tek sorumluluk),
                     Chain of Responsibility Pattern
- Design Patterns: Async Chain of Responsibility Pattern (async processing handlers)
- Endüstri Standartları: Async handler pattern best practices

KULLANIM:
- Async request processing handlers için
- Async feature extraction için
- Async policy routing için
- Async deliberation için

BAĞIMLILIKLAR:
- AsyncProcessingPipeline: Async processing pipeline
- ProcessingHandlers: Sync processing handlers
- asyncio: Async işlemler

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Optional
import asyncio

# V1'den import
import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cognitive_management.cognitive_types import DecodingConfig, PolicyOutput, ThoughtCandidate
from cognitive_management.v2.utils.heuristics import build_features

from .async_pipeline import BaseAsyncProcessingHandler, ProcessingContext
from .handlers import (
    FeatureExtractionHandler,
    PolicyRoutingHandler,
    DeliberationHandler,
    ContextBuildingHandler,
    GenerationHandler,
    SelfConsistencyHandler,
    CriticHandler,
    MemoryUpdateHandler,
)


# =============================================================================
# Async Feature Extraction Handler
# =============================================================================

class AsyncFeatureExtractionHandler(BaseAsyncProcessingHandler):
    """
    Async feature extraction handler.
    SOLID: SRP - Sadece feature extraction yapar (async).
    """

    def __init__(self, memory_service, backend=None):
        super().__init__("AsyncFeatureExtraction")
        self.memory_service = memory_service
        # Sync handler for delegation (Phase 9: pass backend for logit entropy)
        self._sync_handler = FeatureExtractionHandler(memory_service, backend=backend)
    
    async def _process_async(self, context: ProcessingContext) -> ProcessingContext:
        """
        Extract features from input (async).
        
        Phase 5.1: Async support with concurrent memory retrieval.
        """
        # Run sync handler in thread pool for I/O-bound operations
        return await asyncio.to_thread(self._sync_handler._process, context)


# =============================================================================
# Async Policy Routing Handler
# =============================================================================

class AsyncPolicyRoutingHandler(BaseAsyncProcessingHandler):
    """
    Async policy routing handler.
    SOLID: SRP - Sadece policy routing yapar (async).
    """
    
    def __init__(self, policy_router):
        super().__init__("AsyncPolicyRouting")
        self.policy_router = policy_router
        self._sync_handler = PolicyRoutingHandler(policy_router)
    
    async def _process_async(self, context: ProcessingContext) -> ProcessingContext:
        """Route to appropriate policy (async)"""
        # Policy routing is CPU-bound, run in thread pool
        return await asyncio.to_thread(self._sync_handler._process, context)


# =============================================================================
# Async Deliberation Handler
# =============================================================================

class AsyncDeliberationHandler(BaseAsyncProcessingHandler):
    """
    Async deliberation handler.
    SOLID: SRP - Sadece deliberation yapar (async).
    
    Phase 4: Enhanced with Tree of Thoughts (ToT) support.
    """
    
    def __init__(self, engine, backend, cfg=None, tree_of_thoughts=None):
        super().__init__("AsyncDeliberation")
        self.engine = engine
        self.backend = backend
        # Phase 4: Pass ToT support to sync handler
        self._sync_handler = DeliberationHandler(engine, backend, cfg=cfg, tree_of_thoughts=tree_of_thoughts)
    
    async def _process_async(self, context: ProcessingContext) -> ProcessingContext:
        """Generate internal thoughts if needed (async)"""
        # Deliberation involves model calls, run in thread pool
        return await asyncio.to_thread(self._sync_handler._process, context)


# =============================================================================
# Async Context Building Handler
# =============================================================================

class AsyncContextBuildingHandler(BaseAsyncProcessingHandler):
    """
    Async context building handler.
    SOLID: SRP - Sadece context building yapar (async).
    """
    
    def __init__(self, memory_service, tool_policy=None):
        super().__init__("AsyncContextBuilding")
        self.memory_service = memory_service
        self.tool_policy = tool_policy
        self._sync_handler = ContextBuildingHandler(memory_service, tool_policy)
    
    async def _process_async(self, context: ProcessingContext) -> ProcessingContext:
        """
        Build context for generation (async).
        
        Phase 5.1: Async support with concurrent context building.
        """
        # Context building involves memory operations, run in thread pool
        return await asyncio.to_thread(self._sync_handler._process, context)


# =============================================================================
# Async Generation Handler
# =============================================================================

class AsyncGenerationHandler(BaseAsyncProcessingHandler):
    """
    Async generation handler.
    SOLID: SRP - Sadece text generation yapar (async).
    """
    
    def __init__(self, backend):
        super().__init__("AsyncGeneration")
        self.backend = backend
        self._sync_handler = GenerationHandler(backend)
    
    async def _process_async(self, context: ProcessingContext) -> ProcessingContext:
        """Generate text using backend (async)"""
        # Generation involves model calls, run in thread pool
        return await asyncio.to_thread(self._sync_handler._process, context)


# =============================================================================
# Async Self-Consistency Handler  (Wang et al. 2022)
# =============================================================================

class AsyncSelfConsistencyHandler(BaseAsyncProcessingHandler):
    """
    Async Self-Consistency Decoding — Wang et al. 2022.

    N farklı yanıt örnekler (yüksek temperature) ve en çok desteklenen
    yanıtı çoğunluk oyu veya model skoru ile seçer.

    Senkron SelfConsistencyHandler'ı thread pool üzerinde çalıştırır
    — model generate() çağrıları I/O-bound kabul edilir.

    SOLID: SRP — yalnızca async self-consistency örneklemesi.
    """

    def __init__(self, backend, cfg=None):
        super().__init__("AsyncSelfConsistency")
        self.backend = backend
        self.cfg = cfg
        self._sync_handler = SelfConsistencyHandler(backend=backend, cfg=cfg)

    async def _process_async(self, context: ProcessingContext) -> ProcessingContext:
        """N örneklem → çoğunluk seçimi (async, thread pool üzerinden)."""
        return await asyncio.to_thread(self._sync_handler._process, context)


# =============================================================================
# Async Critic Handler
# =============================================================================

class AsyncCriticHandler(BaseAsyncProcessingHandler):
    """
    Async critic handler.
    SOLID: SRP - Sadece critic review yapar (async).
    """
    
    def __init__(self, critic):
        super().__init__("AsyncCritic")
        self.critic = critic
        self._sync_handler = CriticHandler(critic)
    
    async def _process_async(self, context: ProcessingContext) -> ProcessingContext:
        """Review and revise text (async)"""
        # Critic review involves model calls, run in thread pool
        return await asyncio.to_thread(self._sync_handler._process, context)


# =============================================================================
# Async Memory Update Handler
# =============================================================================

class AsyncMemoryUpdateHandler(BaseAsyncProcessingHandler):
    """
    Async memory update handler.
    SOLID: SRP - Sadece memory update yapar (async).
    """
    
    def __init__(self, memory_service):
        super().__init__("AsyncMemoryUpdate")
        self.memory_service = memory_service
        self._sync_handler = MemoryUpdateHandler(memory_service)
    
    async def _process_async(self, context: ProcessingContext) -> ProcessingContext:
        """Update memory with conversation turn (async)"""
        # Memory update involves I/O, run in thread pool
        return await asyncio.to_thread(self._sync_handler._process, context)


__all__ = [
    "AsyncFeatureExtractionHandler",
    "AsyncPolicyRoutingHandler",
    "AsyncDeliberationHandler",
    "AsyncContextBuildingHandler",
    "AsyncGenerationHandler",
    "AsyncSelfConsistencyHandler",
    "AsyncCriticHandler",
    "AsyncMemoryUpdateHandler",
]

