# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: semantic_cache.py
Modül: cognitive_management/v2/utils
Görev: Semantic Cache - Semantic similarity-based caching for similar queries.
       Phase 6: Performance Optimization & Caching Enhancement. SemanticCacheEntry,
       SemanticCache sınıflarını içerir. Semantic similarity-based caching,
       embedding-based similarity search ve cache retrieval işlemlerini yapar.
       Akademik referans: CacheQL (Chen et al., 2023), Embedding-based similarity
       search for cache retrieval.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (semantic caching)
- Design Patterns: Cache Pattern (semantic caching)
- Endüstri Standartları: Semantic caching best practices

KULLANIM:
- Semantic caching için
- Embedding-based similarity search için
- Cache retrieval için

BAĞIMLILIKLAR:
- Cache: Base cache interface
- Config: Yapılandırma

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time
import threading

from .cache import InMemoryCache, CacheEntry
from cognitive_management.config import CognitiveManagerConfig


@dataclass
class SemanticCacheEntry:
    """
    Semantic cache entry with embedding.
    
    Attributes:
        key: Original cache key
        value: Cached value
        embedding: Query embedding vector
        query_text: Original query text
        created_at: Creation timestamp
        access_count: Access count
        similarity_threshold: Minimum similarity for cache hit
    """
    key: str
    value: Any
    embedding: List[float]
    query_text: str
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    similarity_threshold: float = 0.85  # Default threshold for semantic match


class SemanticCache:
    """
    Semantic cache for similar query matching.
    
    Uses embedding-based similarity search to find cached results
    for semantically similar queries, even if exact match doesn't exist.
    
    Academic Reference:
    - CacheQL: Semantic Caching for Language Models (Chen et al., 2023)
    """
    
    def __init__(
        self,
        cfg: CognitiveManagerConfig,
        embedding_adapter=None,
        similarity_threshold: float = 0.85,
        max_size: int = 1000,
    ):
        """
        Initialize semantic cache.
        
        Args:
            cfg: Cognitive manager configuration
            embedding_adapter: Embedding adapter for generating embeddings
            similarity_threshold: Minimum similarity score for cache hit (0.0-1.0)
            max_size: Maximum number of cache entries
        """
        self.cfg = cfg
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        
        # Embedding adapter (lazy initialization)
        self._embedding_adapter = embedding_adapter
        self._lock = threading.RLock()
        
        # Storage
        self._entries: Dict[str, SemanticCacheEntry] = {}
        self._embedding_index: List[Tuple[str, List[float]]] = []  # (key, embedding) pairs
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._semantic_hits = 0  # Semantic similarity hits
    
    def get(
        self,
        query_text: str,
        exact_key: Optional[str] = None
    ) -> Optional[Tuple[Any, float]]:
        """
        Get cached value for query (exact or semantic match).
        
        Args:
            query_text: Query text for semantic matching
            exact_key: Optional exact cache key for direct lookup
            
        Returns:
            Tuple of (cached_value, similarity_score) or None if not found
            similarity_score: 1.0 for exact match, 0.85-0.99 for semantic match
        """
        with self._lock:
            # Try exact match first if key provided
            if exact_key and exact_key in self._entries:
                entry = self._entries[exact_key]
                entry.access_count += 1
                self._hits += 1
                return (entry.value, 1.0)
            
            # Try semantic match
            if not self._embedding_adapter or not query_text:
                self._misses += 1
                return None
            
            try:
                # Generate query embedding
                query_embedding = self._embedding_adapter.encode_single(query_text)
                
                # Find most similar entry
                best_match = self._find_similar_entry(query_embedding)
                
                if best_match:
                    entry_key, similarity = best_match
                    if similarity >= self.similarity_threshold:
                        entry = self._entries[entry_key]
                        entry.access_count += 1
                        self._semantic_hits += 1
                        return (entry.value, similarity)
                
                self._misses += 1
                return None
                
            except Exception:
                # Embedding generation failed
                self._misses += 1
                return None
    
    def set(
        self,
        key: str,
        value: Any,
        query_text: str,
        ttl: Optional[float] = None
    ) -> None:
        """
        Store value in semantic cache.
        
        Args:
            key: Cache key
            value: Value to cache
            query_text: Query text for embedding generation
            ttl: Time to live in seconds (optional)
        """
        with self._lock:
            if not self._embedding_adapter or not query_text:
                # Fallback: store without embedding
                return
            
            try:
                # Generate embedding
                embedding = self._embedding_adapter.encode_single(query_text)
                
                # Create entry
                entry = SemanticCacheEntry(
                    key=key,
                    value=value,
                    embedding=embedding,
                    query_text=query_text,
                    similarity_threshold=self.similarity_threshold,
                )
                
                # Check if need to evict
                if len(self._entries) >= self.max_size:
                    self._evict_lru()
                
                # Store entry
                self._entries[key] = entry
                
                # Update embedding index
                self._update_embedding_index(key, embedding)
                
            except Exception:
                # Embedding generation failed - skip caching
                pass
    
    def _find_similar_entry(
        self,
        query_embedding: List[float]
    ) -> Optional[Tuple[str, float]]:
        """
        Find most similar cache entry using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            
        Returns:
            Tuple of (entry_key, similarity_score) or None
        """
        if not self._embedding_index:
            return None
        
        best_similarity = 0.0
        best_key = None
        
        # Calculate cosine similarity with all entries
        for entry_key, entry_embedding in self._embedding_index:
            similarity = self._cosine_similarity(query_embedding, entry_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_key = entry_key
        
        if best_similarity >= self.similarity_threshold:
            return (best_key, best_similarity)
        
        return None
    
    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0.0-1.0)
        """
        if len(vec1) != len(vec2):
            return 0.0
        
        # Dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Magnitudes
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0
        
        # Cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)
        return max(0.0, min(1.0, similarity))
    
    def _update_embedding_index(
        self,
        key: str,
        embedding: List[float]
    ) -> None:
        """Update embedding index with new entry."""
        # Remove old entry if exists
        self._embedding_index = [
            (k, emb) for k, emb in self._embedding_index if k != key
        ]
        
        # Add new entry
        self._embedding_index.append((key, embedding))
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._entries:
            return
        
        # Find LRU entry (lowest access_count, oldest created_at)
        lru_key = min(
            self._entries.keys(),
            key=lambda k: (
                self._entries[k].access_count,
                self._entries[k].created_at
            )
        )
        
        # Remove from entries
        del self._entries[lru_key]
        
        # Remove from embedding index
        self._embedding_index = [
            (k, emb) for k, emb in self._embedding_index if k != lru_key
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            semantic_hit_rate = (
                self._semantic_hits / total_requests if total_requests > 0 else 0.0
            )
            
            return {
                "size": len(self._entries),
                "max_size": self.max_size,
                "hits": self._hits,
                "semantic_hits": self._semantic_hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "semantic_hit_rate": semantic_hit_rate,
                "similarity_threshold": self.similarity_threshold,
            }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._entries.clear()
            self._embedding_index.clear()
            self._hits = 0
            self._misses = 0
            self._semantic_hits = 0


__all__ = [
    "SemanticCache",
    "SemanticCacheEntry",
]

