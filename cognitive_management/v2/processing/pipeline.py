# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: pipeline.py
Modül: cognitive_management/v2/processing
Görev: Processing Pipeline - Chain of Responsibility pattern ile request processing.
       ProcessingContext, ProcessingHandler, BaseProcessingHandler, ProcessingPipeline
       sınıflarını içerir. Request processing, handler chain execution ve pipeline
       orchestration işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (processing pipeline),
                     Chain of Responsibility Pattern
- Design Patterns: Chain of Responsibility Pattern (processing pipeline)
- Endüstri Standartları: Processing pipeline best practices

KULLANIM:
- Request processing için
- Handler chain execution için
- Pipeline orchestration için

BAĞIMLILIKLAR:
- CognitiveTypes: Tip tanımları
- dataclasses: Processing context data structures

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


# =============================================================================
# Processing Context
# =============================================================================

@dataclass
class ProcessingContext:
    """
    Processing context.
    Pipeline boyunca taşınan veri.
    
    Phase 3: Enhanced with retrieved contexts and advanced metadata.
    """
    state: CognitiveState
    request: CognitiveInput
    decoding_config: Optional[DecodingConfig] = None
    
    # Pipeline boyunca değişen veriler
    features: dict = field(default_factory=dict)
    policy_output: Optional[PolicyOutput] = None
    selected_thought: Optional[ThoughtCandidate] = None
    tool_name: Optional[str] = None
    context_text: Optional[str] = None
    draft_text: Optional[str] = None
    final_text: Optional[str] = None
    revised: bool = False
    
    # Phase 3: Memory retrieval
    retrieved_contexts: List[dict] = field(default_factory=list)
    
    # Metadata
    processing_steps: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# =============================================================================
# Processing Handler Protocol
# =============================================================================

class ProcessingHandler(Protocol):
    """
    Processing handler interface.
    Chain of Responsibility pattern için.
    """
    def handle(
        self,
        context: ProcessingContext
    ) -> Optional[ProcessingContext]:
        """
        Handle processing step.
        
        Args:
            context: Processing context
            
        Returns:
            Updated context (None if chain should stop)
        """
        ...
    
    def set_next(self, handler: ProcessingHandler) -> None:
        """
        Set next handler in chain.
        
        Args:
            handler: Next handler
        """
        ...


# =============================================================================
# Base Handler Implementation
# =============================================================================

class BaseProcessingHandler:
    """
    Base processing handler.
    Chain of Responsibility pattern base implementation.
    """
    
    def __init__(self, name: str):
        self.name = name
        self._next: Optional[ProcessingHandler] = None
    
    def set_next(self, handler: ProcessingHandler) -> None:
        """Set next handler"""
        self._next = handler
    
    def handle(self, context: ProcessingContext) -> Optional[ProcessingContext]:
        """
        Handle processing step and pass to next handler.
        
        Args:
            context: Processing context
            
        Returns:
            Updated context
        """
        # Process this step
        context = self._process(context)
        
        # Add to processing steps
        if context:
            context.processing_steps.append(self.name)
        
        # Pass to next handler
        if self._next and context:
            return self._next.handle(context)
        
        return context
    
    def _process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Process this step.
        Subclasses implement this.
        
        Args:
            context: Processing context
            
        Returns:
            Updated context
        """
        # Base implementation: just pass through
        return context


# =============================================================================
# Processing Pipeline
# =============================================================================

class ProcessingPipeline:
    """
    Processing pipeline.
    Chain of Responsibility pattern ile request processing.
    
    SOLID: Single Responsibility Principle
    - Pipeline sadece chain yönetimi yapar
    - Her handler tek bir iş yapar
    """
    
    def __init__(self, handlers: List[ProcessingHandler]):
        """
        Initialize pipeline with handlers.
        
        Args:
            handlers: List of handlers in order
        """
        if not handlers:
            raise ValueError("Pipeline requires at least one handler")
        
        # Build chain
        self._first_handler = handlers[0]
        for i in range(len(handlers) - 1):
            handlers[i].set_next(handlers[i + 1])
    
    def process(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        decoding_config: Optional[DecodingConfig] = None
    ) -> CognitiveOutput:
        """
        Process request through pipeline.
        
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
        
        # Process through chain
        result_context = self._first_handler.handle(context)
        
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
    "ProcessingPipeline",
    "ProcessingContext",
    "ProcessingHandler",
    "BaseProcessingHandler",
]

