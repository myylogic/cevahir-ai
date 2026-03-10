# -*- coding: utf-8 -*-
"""
V2 Utilities
============
V2 için yardımcı fonksiyonlar ve utilities.
"""

from .heuristics import build_features, estimate_risk, should_tool, mode_gates
from .context_pruning import build_context, ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM_SUMMARY
from .selectors import pick_best_by_score
from .cache import (
    Cache,
    CacheEntry,
    InMemoryCache,
    generate_cache_key,
    cached,
)
from .tracing import (
    TraceStatus,
    SpanContext,
    Span,
    Trace,
    TraceStorage,
    TraceContextManager,
    generate_trace_id,
    generate_span_id,
    create_span_context,
    get_current_trace_context,
    set_trace_context,
    clear_trace_context,
    create_child_span_context,
)

# Phase 6: Performance Optimization utilities
from .semantic_cache import SemanticCache, SemanticCacheEntry
from .connection_pool import ConnectionPool, PooledConnection
from .request_batcher import RequestBatcher, BatchRequest
from .cache_warming import CacheWarmer, WarmingStrategy
from .performance_profiler import PerformanceProfiler, ProfileEntry

__all__ = [
    "build_features",
    "estimate_risk",
    "should_tool",
    "mode_gates",
    "build_context",
    "ROLE_USER",
    "ROLE_ASSISTANT",
    "ROLE_SYSTEM_SUMMARY",
    "pick_best_by_score",
    "Cache",
    "CacheEntry",
    "InMemoryCache",
    "generate_cache_key",
    "cached",
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
    # Phase 6: Performance Optimization
    "SemanticCache",
    "SemanticCacheEntry",
    "ConnectionPool",
    "PooledConnection",
    "RequestBatcher",
    "BatchRequest",
    "CacheWarmer",
    "WarmingStrategy",
    "PerformanceProfiler",
    "ProfileEntry",
]

