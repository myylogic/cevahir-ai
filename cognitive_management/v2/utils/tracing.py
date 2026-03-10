# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: tracing.py
Modül: cognitive_management/v2/utils
Görev: Distributed Tracing Utilities - Industry-standard distributed tracing for
       V2 Cognitive Management. Phase 5.3: Full observability with distributed
       tracing. Based on OpenTelemetry concepts and industry best practices.
       TraceStatus, SpanContext, Span, Trace ve tracing utility fonksiyonlarını
       içerir. Distributed tracing, span creation, trace context propagation
       ve trace correlation işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (distributed tracing)
- Design Patterns: Tracing Pattern (distributed tracing)
- Endüstri Standartları: OpenTelemetry compatible tracing

KULLANIM:
- Distributed tracing için
- Span creation için
- Trace context propagation için
- Trace correlation için

BAĞIMLILIKLAR:
- uuid: Trace ID generation
- threading: Thread-safe işlemler
- dataclasses: Tracing data structures

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import time
import threading
from collections import deque


# =============================================================================
# Trace Context
# =============================================================================

class TraceStatus(Enum):
    """Trace status (OpenTelemetry compatible)"""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """
    Span context for distributed tracing.
    
    Industry standard: OpenTelemetry TraceContext
    """
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    flags: int = 0  # Trace flags (sampling, etc.)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "flags": self.flags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SpanContext:
        """Create from dictionary"""
        return cls(
            trace_id=data.get("trace_id", ""),
            span_id=data.get("span_id", ""),
            parent_span_id=data.get("parent_span_id"),
            flags=data.get("flags", 0),
        )


@dataclass
class Span:
    """
    Tracing span (OpenTelemetry compatible).
    
    Represents a single operation in a trace.
    """
    name: str
    context: SpanContext
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: TraceStatus = TraceStatus.UNSET
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    links: List[SpanContext] = field(default_factory=list)
    kind: str = "internal"  # internal, server, client, producer, consumer
    
    def end(self, status: Optional[TraceStatus] = None) -> None:
        """End the span"""
        self.end_time = time.time()
        if status is not None:
            self.status = status
    
    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Add event to span"""
        self.events.append({
            "name": name,
            "attributes": attributes or {},
            "timestamp": timestamp or time.time(),
        })
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute"""
        self.attributes[key] = value
    
    def duration(self) -> float:
        """Get span duration in seconds"""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary"""
        return {
            "name": self.name,
            "context": self.context.to_dict(),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration(),
            "status": self.status.value,
            "attributes": self.attributes,
            "events": self.events,
            "kind": self.kind,
        }


@dataclass
class Trace:
    """
    Complete trace (collection of spans).
    
    Industry standard: OpenTelemetry Trace
    """
    trace_id: str
    spans: List[Span] = field(default_factory=list)
    root_span: Optional[Span] = None
    
    def add_span(self, span: Span) -> None:
        """Add span to trace"""
        self.spans.append(span)
        if span.context.parent_span_id is None:
            self.root_span = span
    
    def get_span(self, span_id: str) -> Optional[Span]:
        """Get span by ID"""
        for span in self.spans:
            if span.context.span_id == span_id:
                return span
        return None
    
    def get_spans_by_name(self, name: str) -> List[Span]:
        """Get spans by name"""
        return [span for span in self.spans if span.name == name]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary"""
        return {
            "trace_id": self.trace_id,
            "spans": [span.to_dict() for span in self.spans],
            "root_span": self.root_span.to_dict() if self.root_span else None,
            "total_spans": len(self.spans),
            "duration": self._calculate_duration(),
        }
    
    def _calculate_duration(self) -> float:
        """Calculate total trace duration"""
        if not self.spans:
            return 0.0
        
        start = min(span.start_time for span in self.spans)
        end = max(
            span.end_time if span.end_time else time.time()
            for span in self.spans
        )
        return end - start


# =============================================================================
# Trace ID Generation
# =============================================================================

def generate_trace_id() -> str:
    """
    Generate trace ID (128-bit, hex encoded).
    
    Industry standard: OpenTelemetry TraceID format
    """
    # Generate 128-bit UUID and convert to hex (32 chars)
    trace_uuid = uuid.uuid4()
    return trace_uuid.hex


def generate_span_id() -> str:
    """
    Generate span ID (64-bit, hex encoded).
    
    Industry standard: OpenTelemetry SpanID format
    """
    # Generate 64-bit ID (16 hex chars)
    span_uuid = uuid.uuid4()
    return span_uuid.hex[:16]


def create_span_context(
    trace_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
) -> SpanContext:
    """
    Create span context.
    
    Args:
        trace_id: Trace ID (None = generate new)
        parent_span_id: Parent span ID (None = root span)
        
    Returns:
        SpanContext
    """
    return SpanContext(
        trace_id=trace_id or generate_trace_id(),
        span_id=generate_span_id(),
        parent_span_id=parent_span_id,
    )


# =============================================================================
# Trace Storage
# =============================================================================

class TraceStorage:
    """
    Thread-safe trace storage.
    
    Stores traces in memory with configurable retention.
    """
    
    def __init__(self, max_traces: int = 1000):
        """
        Initialize trace storage.
        
        Args:
            max_traces: Maximum number of traces to store
        """
        self.max_traces = max_traces
        self._traces: Dict[str, Trace] = {}
        self._lock = threading.RLock()
    
    def store_trace(self, trace: Trace) -> None:
        """Store trace"""
        with self._lock:
            self._traces[trace.trace_id] = trace
            
            # Evict oldest if over limit
            if len(self._traces) > self.max_traces:
                # Remove oldest (simple FIFO)
                oldest_id = next(iter(self._traces))
                del self._traces[oldest_id]
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get trace by ID"""
        with self._lock:
            return self._traces.get(trace_id)
    
    def get_all_traces(self) -> List[Trace]:
        """Get all traces"""
        with self._lock:
            return list(self._traces.values())
    
    def clear(self) -> None:
        """Clear all traces"""
        with self._lock:
            self._traces.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        with self._lock:
            total_spans = sum(len(trace.spans) for trace in self._traces.values())
            return {
                "total_traces": len(self._traces),
                "total_spans": total_spans,
                "max_traces": self.max_traces,
            }


# =============================================================================
# Trace Context Manager
# =============================================================================

class TraceContextManager:
    """
    Thread-local trace context manager.
    
    Manages active trace context per thread.
    """
    
    def __init__(self):
        self._local = threading.local()
    
    def get_context(self) -> Optional[SpanContext]:
        """Get current trace context"""
        return getattr(self._local, 'context', None)
    
    def set_context(self, context: SpanContext) -> None:
        """Set current trace context"""
        self._local.context = context
    
    def clear_context(self) -> None:
        """Clear current trace context"""
        if hasattr(self._local, 'context'):
            delattr(self._local, 'context')
    
    def create_child_context(self, name: str) -> SpanContext:
        """
        Create child span context from current context.
        
        Args:
            name: Span name
            
        Returns:
            Child SpanContext
        """
        parent = self.get_context()
        if parent:
            return SpanContext(
                trace_id=parent.trace_id,
                span_id=generate_span_id(),
                parent_span_id=parent.span_id,
                flags=parent.flags,
            )
        else:
            return create_span_context()


# Global trace context manager
_trace_context_manager = TraceContextManager()


def get_current_trace_context() -> Optional[SpanContext]:
    """Get current trace context"""
    return _trace_context_manager.get_context()


def set_trace_context(context: SpanContext) -> None:
    """Set trace context"""
    _trace_context_manager.set_context(context)


def clear_trace_context() -> None:
    """Clear trace context"""
    _trace_context_manager.clear_context()


def create_child_span_context(name: str) -> SpanContext:
    """Create child span context"""
    return _trace_context_manager.create_child_context(name)


__all__ = [
    "TraceStatus",
    "SpanContext",
    "Span",
    "Trace",
    "TraceStorage",
    "TraceContextManager",
    "generate_trace_id",
    "generate_span_id",
    "create_span_context",
    "get_current_trace_context",
    "set_trace_context",
    "clear_trace_context",
    "create_child_span_context",
]

