# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: async_pipeline.py
Modül: cognitive_management/v2/processing
Görev: Async Processing Pipeline - Async Chain of Responsibility pattern ile
       request processing. AsyncProcessingHandler, BaseAsyncProcessingHandler,
       AsyncProcessingPipeline sınıflarını içerir. Async request processing,
       async handler chain execution ve async pipeline orchestration işlemlerini
       yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (async processing pipeline),
                     Chain of Responsibility Pattern
- Design Patterns: Async Chain of Responsibility Pattern (async processing pipeline)
- Endüstri Standartları: Async processing pipeline best practices

KULLANIM:
- Async request processing için
- Async handler chain execution için
- Async pipeline orchestration için

BAĞIMLILIKLAR:
- ProcessingPipeline: Sync processing pipeline
- asyncio: Async işlemler
- CognitiveTypes: Tip tanımları

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Optional, List, Protocol
from dataclasses import dataclass, field

# V1'den import
import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cognitive_management.cognitive_types import (
    CognitiveState,
    CognitiveInput,
    CognitiveOutput,
    DecodingConfig,
    PolicyOutput,
    ThoughtCandidate,
)

from .pipeline import ProcessingContext


# =============================================================================
# Async Processing Handler Protocol
# =============================================================================

class AsyncProcessingHandler(Protocol):
    """
    Async processing handler interface.
    Chain of Responsibility pattern için (async).
    """
    async def handle_async(
        self,
        context: ProcessingContext
    ) -> Optional[ProcessingContext]:
        """
        Handle processing step (async).
        
        Args:
            context: Processing context
            
        Returns:
            Updated context (None if chain should stop)
        """
        ...
    
    def set_next(self, handler: AsyncProcessingHandler) -> None:
        """
        Set next handler in chain.
        
        Args:
            handler: Next handler
        """
        ...


# =============================================================================
# Base Async Handler Implementation
# =============================================================================

class BaseAsyncProcessingHandler:
    """
    Base async processing handler.
    Chain of Responsibility pattern base implementation (async).
    """
    
    def __init__(self, name: str):
        self.name = name
        self._next: Optional[AsyncProcessingHandler] = None
    
    def set_next(self, handler: AsyncProcessingHandler) -> None:
        """Set next handler"""
        self._next = handler
    
    async def handle_async(self, context: ProcessingContext) -> Optional[ProcessingContext]:
        """
        Handle processing step and pass to next handler (async).
        
        Args:
            context: Processing context
            
        Returns:
            Updated context
        """
        # Process this step
        context = await self._process_async(context)
        
        # Add to processing steps
        if context:
            context.processing_steps.append(self.name)
        
        # Pass to next handler
        if self._next and context:
            return await self._next.handle_async(context)
        
        return context
    
    async def _process_async(self, context: ProcessingContext) -> ProcessingContext:
        """
        Process this step (async).
        Subclasses implement this.
        
        Args:
            context: Processing context
            
        Returns:
            Updated context
        """
        # Base implementation: just pass through
        return context


# =============================================================================
# Async Processing Pipeline
# =============================================================================

class AsyncProcessingPipeline:
    """
    Async processing pipeline.
    Chain of Responsibility pattern ile async request processing.
    
    SOLID: Single Responsibility Principle
    - Pipeline sadece async chain yönetimi yapar
    - Her handler tek bir iş yapar (async)
    """
    
    def __init__(self, handlers: List[AsyncProcessingHandler]):
        """
        Initialize async pipeline with handlers.
        
        Args:
            handlers: List of async handlers in order
        """
        if not handlers:
            raise ValueError("AsyncPipeline requires at least one handler")
        
        # Build chain
        self._first_handler = handlers[0]
        for i in range(len(handlers) - 1):
            handlers[i].set_next(handlers[i + 1])
    
    async def process_async(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        decoding_config: Optional[DecodingConfig] = None
    ) -> CognitiveOutput:
        """
        Process request through async pipeline.
        
        Args:
            state: Cognitive state
            request: Cognitive input
            decoding_config: Optional decoding config
            
        Returns:
            Cognitive output
        """
        # Create context
        context = ProcessingContext(
            state=state,
            request=request,
            decoding_config=decoding_config,
        )
        
        # Process through async chain
        result_context = await self._first_handler.handle_async(context)
        
        if not result_context:
            # Pipeline failed
            return CognitiveOutput(
                text="İşleme sırasında bir hata oluştu.",
                used_mode="direct",
                tool_used=None,
                revised_by_critic=False,
            )
        
        # Build output
        return CognitiveOutput(
            text=result_context.final_text or result_context.draft_text or "",
            used_mode=result_context.policy_output.mode if result_context.policy_output else "direct",
            tool_used=result_context.tool_name,
            revised_by_critic=result_context.revised,
        )


__all__ = [
    "AsyncProcessingPipeline",
    "AsyncProcessingHandler",
    "BaseAsyncProcessingHandler",
]

