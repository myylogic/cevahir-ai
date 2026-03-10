# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: tracing.py
Modül: cognitive_management/v2/middleware
Görev: Tracing Middleware - Distributed tracing middleware for V2 Cognitive
       Management. Phase 5.3: Full observability with distributed tracing.
       Industry standard: OpenTelemetry compatible. Span creation, trace context
       propagation, trace correlation ve distributed tracing işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (tracing middleware),
                     Dependency Inversion (BaseMiddleware interface'e bağımlı)
- Design Patterns: Middleware Pattern (tracing middleware)
- Endüstri Standartları: OpenTelemetry compatible tracing

KULLANIM:
- Distributed tracing için
- Span creation için
- Trace context propagation için

BAĞIMLILIKLAR:
- BaseMiddleware: Base middleware
- Tracing utilities: Tracing işlemleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import time

from cognitive_management.cognitive_types import (
    CognitiveState,
    CognitiveInput,
    CognitiveOutput,
)
from ..utils.tracing import (
    SpanContext,
    Span,
    Trace,
    TraceStatus,
    generate_trace_id,
    create_span_context,
    get_current_trace_context,
    set_trace_context,
    clear_trace_context,
    create_child_span_context,
)
from .base import BaseMiddleware


class TracingMiddleware(BaseMiddleware):
    """
    Tracing middleware for distributed tracing.
    
    Phase 5.3: Full observability with distributed tracing.
    
    Features:
    - Request ID generation
    - Span creation and tracking
    - Context propagation
    - Trace export
    - OpenTelemetry compatible
    """
    
    def __init__(
        self,
        trace_storage=None,
        enabled: bool = True,
        sample_rate: float = 1.0,  # 1.0 = 100% sampling
    ):
        """
        Initialize tracing middleware.
        
        Args:
            trace_storage: TraceStorage instance (None = use default)
            enabled: Enable/disable tracing
            sample_rate: Sampling rate (0.0-1.0)
        """
        super().__init__("Tracing")
        self.trace_storage = trace_storage
        self.enabled = enabled
        self.sample_rate = sample_rate
        
        # Current trace per request
        self._current_trace: Optional[Trace] = None
        self._current_span: Optional[Span] = None
    
    def _should_sample(self) -> bool:
        """Determine if request should be sampled"""
        if not self.enabled:
            return False
        import random
        return random.random() < self.sample_rate
    
    def _before(
        self,
        state: CognitiveState,
        request: CognitiveInput,
    ) -> tuple[CognitiveState, CognitiveInput]:
        """Start trace before processing"""
        if not self._should_sample():
            return state, request
        
        # Check for existing trace context (from upstream)
        existing_context = get_current_trace_context()
        
        if existing_context:
            # Use existing trace
            trace_id = existing_context.trace_id
            parent_span_id = existing_context.span_id
        else:
            # Create new trace
            trace_id = generate_trace_id()
            parent_span_id = None
        
        # Create root span for this request
        span_context = SpanContext(
            trace_id=trace_id,
            span_id=create_span_context().span_id,
            parent_span_id=parent_span_id,
        )
        
        # Create span
        span = Span(
            name="cognitive_request",
            context=span_context,
            kind="server",
        )
        
        # Set attributes
        span.set_attribute("user_message", request.user_message or "")
        span.set_attribute("system_prompt", request.system_prompt or "")
        span.set_attribute("step", state.step)
        span.set_attribute("last_mode", state.last_mode or "unknown")
        
        # Store trace
        self._current_trace = Trace(trace_id=trace_id)
        self._current_trace.add_span(span)
        self._current_span = span
        
        # Set trace context
        set_trace_context(span_context)
        
        # Store trace ID in request metadata
        request.metadata["_trace_id"] = trace_id
        request.metadata["_span_id"] = span_context.span_id
        
        return state, request
    
    def _after(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        response: CognitiveOutput,
    ) -> CognitiveOutput:
        """End trace after processing"""
        if not self._current_span:
            return response
        
        # End span
        status = TraceStatus.ERROR if response.revised_by_critic else TraceStatus.OK
        self._current_span.end(status=status)
        
        # Add response attributes
        self._current_span.set_attribute("response_mode", response.used_mode or "unknown")
        self._current_span.set_attribute("tool_used", response.tool_used or "none")
        self._current_span.set_attribute("revised", response.revised_by_critic)
        self._current_span.set_attribute("response_length", len(response.text or ""))
        
        # Store trace if storage available
        if self.trace_storage and self._current_trace:
            self.trace_storage.store_trace(self._current_trace)
        
        # Clear context
        clear_trace_context()
        
        # Store trace ID in response metadata
        response.metadata = getattr(response, 'metadata', {})
        response.metadata["_trace_id"] = self._current_trace.trace_id
        response.metadata["_span_id"] = self._current_span.context.span_id
        
        return response
    
    def _on_error(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        error: Exception,
    ) -> Optional[CognitiveOutput]:
        """Handle error in trace"""
        if not self._current_span:
            return None
        
        # Mark span as error
        self._current_span.end(TraceStatus.ERROR)
        self._current_span.set_attribute("error_type", type(error).__name__)
        self._current_span.set_attribute("error_message", str(error))
        
        # Add error event
        self._current_span.add_event(
            "error",
            attributes={
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        
        # Store trace
        if self.trace_storage and self._current_trace:
            self.trace_storage.store_trace(self._current_trace)
        
        # Clear context
        clear_trace_context()
        
        return None
    
    def create_child_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[Span]:
        """
        Create child span for nested operations.
        
        Args:
            name: Span name
            attributes: Optional span attributes
            
        Returns:
            Span or None if tracing disabled
        """
        if not self._current_span or not self.enabled:
            return None
        
        # Create child context
        child_context = create_child_span_context(name)
        
        # Create child span
        child_span = Span(
            name=name,
            context=child_context,
            kind="internal",
        )
        
        # Set attributes
        if attributes:
            for key, value in attributes.items():
                child_span.set_attribute(key, value)
        
        # Add to trace
        if self._current_trace:
            self._current_trace.add_span(child_span)
        
        return child_span
    
    def get_current_trace(self) -> Optional[Trace]:
        """Get current trace"""
        return self._current_trace
    
    def get_current_span(self) -> Optional[Span]:
        """Get current span"""
        return self._current_span


__all__ = ["TracingMiddleware"]

