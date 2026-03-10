# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: cache.py
Modül: cognitive_management/v2/middleware
Görev: Cache Middleware - Response caching middleware for V2 Cognitive Management.
       Phase 5.2: Advanced caching with multi-layer support. Phase 8: Enhanced
       with SemanticCache integration. Response caching, cache key generation,
       cache invalidation ve multi-layer cache desteği sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (cache middleware),
                     Dependency Inversion (BaseMiddleware interface'e bağımlı)
- Design Patterns: Middleware Pattern (cache middleware)
- Endüstri Standartları: Response caching best practices

KULLANIM:
- Response caching için
- Cache key generation için
- Cache invalidation için

BAĞIMLILIKLAR:
- BaseMiddleware: Base middleware
- Cache utilities: Cache işlemleri

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
import hashlib
import json

from cognitive_management.cognitive_types import (
    CognitiveState,
    CognitiveInput,
    CognitiveOutput,
)
from ..utils.cache import Cache, InMemoryCache, generate_cache_key
from .base import BaseMiddleware


class CacheMiddleware(BaseMiddleware):
    """
    Cache middleware for response caching.
    
    Phase 5.2: Advanced caching with multi-layer support.
    Phase 8: Enhanced with SemanticCache integration.
    
    Features:
    - Response caching
    - Context caching
    - Thought caching
    - Semantic cache (similar query matching)
    - Cache invalidation
    - TTL management
    """
    
    def __init__(
        self,
        cache: Optional[Cache] = None,
        response_ttl: float = 3600.0,  # 1 hour default
        context_ttl: float = 1800.0,    # 30 minutes default
        thought_ttl: float = 900.0,     # 15 minutes default
        enabled: bool = True,
        enable_semantic_cache: bool = False,
        semantic_cache: Optional[Any] = None,
        similarity_threshold: float = 0.85,
    ):
        """
        Initialize cache middleware.
        
        Args:
            cache: Cache instance (None = use default InMemoryCache)
            response_ttl: Response cache TTL in seconds
            context_ttl: Context cache TTL in seconds
            thought_ttl: Thought cache TTL in seconds
            enabled: Enable/disable caching
            enable_semantic_cache: Enable semantic cache for similar queries
            semantic_cache: SemanticCache instance (optional, will be created if None and enabled)
            similarity_threshold: Minimum similarity for semantic cache hit
        """
        super().__init__("Cache")
        self.cache = cache or InMemoryCache(max_size=1000, default_ttl=response_ttl)
        self.response_ttl = response_ttl
        self.context_ttl = context_ttl
        self.thought_ttl = thought_ttl
        self.enabled = enabled
        
        # Phase 8: SemanticCache integration
        self.enable_semantic_cache = enable_semantic_cache
        self._semantic_cache = semantic_cache
        self.similarity_threshold = similarity_threshold
    
    def _before(
        self,
        state: CognitiveState,
        request: CognitiveInput,
    ) -> tuple[CognitiveState, CognitiveInput]:
        """Check cache before processing"""
        if not self.enabled:
            return state, request
        
        # Generate cache key for response
        cache_key = self._generate_response_key(state, request)
        
        # Try exact cache first
        cached_response = self.cache.get(cache_key)
        if cached_response is not None:
            # Exact cache hit - mark in metadata and store response
            request.metadata["_cache_hit"] = True
            request.metadata["_cache_key"] = cache_key
            request.metadata["_cache_type"] = "exact"
            request.metadata["_cached_response"] = cached_response  # Store for orchestrator
            return state, request
        
        # Phase 8: Try semantic cache if enabled
        if self.enable_semantic_cache and self._semantic_cache is not None:
            try:
                query_text = request.user_message or ""
                semantic_result = self._semantic_cache.get(query_text, exact_key=cache_key)
                if semantic_result:
                    cached_value, similarity = semantic_result
                    # Store in exact cache for faster lookup next time
                    self.cache.set(cache_key, cached_value, ttl=self.response_ttl)
                    request.metadata["_cache_hit"] = True
                    request.metadata["_cache_key"] = cache_key
                    request.metadata["_cache_type"] = "semantic"
                    request.metadata["_cache_similarity"] = similarity
                    request.metadata["_cached_response"] = cached_value  # Store for orchestrator
                    return state, request
            except Exception:
                # Semantic cache error - fall back to regular cache
                pass
        
        # Cache miss
        request.metadata["_cache_hit"] = False
        request.metadata["_cache_key"] = cache_key
        
        return state, request
    
    def _after(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        response: CognitiveOutput,
    ) -> CognitiveOutput:
        """Cache response after processing"""
        if not self.enabled:
            return response
        
        # Skip if cache hit
        if request.metadata.get("_cache_hit", False):
            return response
        
        # Generate cache key
        cache_key = request.metadata.get("_cache_key")
        if not cache_key:
            cache_key = self._generate_response_key(state, request)
        
        # Cache response in exact cache
        self.cache.set(cache_key, response, ttl=self.response_ttl)
        
        # Phase 8: Also cache in semantic cache if enabled
        if self.enable_semantic_cache and self._semantic_cache is not None:
            try:
                query_text = request.user_message or ""
                if query_text:
                    self._semantic_cache.set(
                        key=cache_key,
                        value=response,
                        query_text=query_text,
                        ttl=self.response_ttl
                    )
            except Exception:
                # Semantic cache error - log but don't fail
                pass
        
        return response
    
    def _generate_response_key(
        self,
        state: CognitiveState,
        request: CognitiveInput,
    ) -> str:
        """
        Generate cache key for response.
        
        Args:
            state: Cognitive state
            request: Cognitive input
            
        Returns:
            Cache key string
        """
        # Include relevant state info
        state_hash = self._hash_state(state)
        user_message = request.user_message or ""
        system_prompt = request.system_prompt or ""
        
        return generate_cache_key(
            "response",
            state_hash=state_hash,
            user_message=user_message,
            system_prompt=system_prompt,
        )
    
    def _hash_state(self, state: CognitiveState) -> str:
        """Generate hash from state"""
        # Include last few turns for context
        recent_history = state.history[-3:] if len(state.history) > 3 else state.history
        state_data = {
            "step": state.step,
            "last_mode": state.last_mode,
            "recent_history": recent_history,
        }
        state_str = json.dumps(state_data, sort_keys=True)
        return hashlib.md5(state_str.encode()).hexdigest()[:16]
    
    def cache_context(
        self,
        context_key: str,
        context_data: Dict[str, Any],
    ) -> None:
        """
        Cache context data.
        
        Args:
            context_key: Context key
            context_data: Context data to cache
        """
        if not self.enabled:
            return
        
        cache_key = f"context:{context_key}"
        self.cache.set(cache_key, context_data, ttl=self.context_ttl)
    
    def get_cached_context(
        self,
        context_key: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached context data.
        
        Args:
            context_key: Context key
            
        Returns:
            Cached context data or None
        """
        if not self.enabled:
            return None
        
        cache_key = f"context:{context_key}"
        return self.cache.get(cache_key)
    
    def cache_thought(
        self,
        thought_key: str,
        thought_data: Dict[str, Any],
    ) -> None:
        """
        Cache thought data.
        
        Args:
            thought_key: Thought key
            thought_data: Thought data to cache
        """
        if not self.enabled:
            return
        
        cache_key = f"thought:{thought_key}"
        self.cache.set(cache_key, thought_data, ttl=self.thought_ttl)
    
    def get_cached_thought(
        self,
        thought_key: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached thought data.
        
        Args:
            thought_key: Thought key
            
        Returns:
            Cached thought data or None
        """
        if not self.enabled:
            return None
        
        cache_key = f"thought:{thought_key}"
        return self.cache.get(cache_key)
    
    def invalidate(
        self,
        pattern: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries.
        
        Args:
            pattern: Key pattern to invalidate (None = all)
            
        Returns:
            Number of invalidated entries
        """
        if pattern is None:
            # Clear all
            self.cache.clear()
            return -1  # Unknown count
        
        # Pattern-based invalidation (simple prefix matching)
        # Note: Full pattern matching requires cache iteration
        # For now, we'll use prefix-based invalidation
        if isinstance(self.cache, InMemoryCache):
            count = 0
            keys_to_delete = []
            with self.cache._lock:
                for key in self.cache._cache.keys():
                    if key.startswith(pattern):
                        keys_to_delete.append(key)
            
            for key in keys_to_delete:
                if self.cache.delete(key):
                    count += 1
            
            return count
        
        return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if isinstance(self.cache, InMemoryCache):
            return self.cache.get_stats()
        return {}


__all__ = ["CacheMiddleware"]

