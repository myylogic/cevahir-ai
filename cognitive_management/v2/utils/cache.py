# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: cache.py
Modül: cognitive_management/v2/utils
Görev: Cache Utilities - Multi-layer caching system for V2 Cognitive Management.
       CacheEntry, Cache, InMemoryCache, LRUCache ve cache key generation
       sınıflarını içerir. Multi-layer caching, cache expiration, cache
       invalidation ve thread-safe cache operations işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (cache utilities)
- Design Patterns: Cache Pattern (multi-layer caching)
- Endüstri Standartları: Caching best practices

KULLANIM:
- Multi-layer caching için
- Cache expiration için
- Cache invalidation için
- Thread-safe cache operations için

BAĞIMLILIKLAR:
- threading: Thread-safe işlemler
- hashlib: Cache key generation
- json: Cache serialization

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Callable, Tuple, Protocol
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import hashlib
import json
import threading
import time


# =============================================================================
# Cache Entry
# =============================================================================

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self) -> None:
        """Update access metadata"""
        self.access_count += 1
        self.last_accessed = time.time()


# =============================================================================
# Cache Interface
# =============================================================================

class Cache(Protocol):
    """Cache interface"""
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        ...
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache"""
        ...
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        ...
    
    def clear(self) -> None:
        """Clear all cache"""
        ...
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        ...


# =============================================================================
# In-Memory Cache (LRU)
# =============================================================================

class InMemoryCache:
    """
    Thread-safe in-memory LRU cache.
    
    Features:
    - LRU eviction
    - TTL support
    - Thread-safe
    - Statistics
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
    ):
        """
        Initialize in-memory cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds (None = no expiration)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """Set value in cache"""
        with self._lock:
            # Remove if exists
            if key in self._cache:
                del self._cache[key]
            
            # Calculate expiration
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl
            elif self.default_ttl is not None:
                expires_at = time.time() + self.default_ttl
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
            )
            
            # Add to cache
            self._cache[key] = entry
            
            # Evict if needed (LRU)
            if len(self._cache) > self.max_size:
                # Remove oldest (first) entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._evictions += 1
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
            }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)


# =============================================================================
# Cache Key Generation
# =============================================================================

def generate_cache_key(
    prefix: str,
    **kwargs: Any
) -> str:
    """
    Generate cache key from prefix and parameters.
    
    Args:
        prefix: Key prefix
        **kwargs: Parameters to include in key
        
    Returns:
        Cache key string
    """
    # Sort kwargs for consistent keys
    sorted_kwargs = sorted(kwargs.items())
    
    # Create key string
    key_parts = [prefix]
    for k, v in sorted_kwargs:
        if v is not None:
            # Convert to string representation
            if isinstance(v, (dict, list)):
                v_str = json.dumps(v, sort_keys=True)
            else:
                v_str = str(v)
            key_parts.append(f"{k}:{v_str}")
    
    key_string = "|".join(key_parts)
    
    # Hash if too long
    if len(key_string) > 200:
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    return key_string


# =============================================================================
# Cache Decorator
# =============================================================================

def cached(
    cache: Cache,
    key_prefix: str,
    ttl: Optional[float] = None,
    key_func: Optional[Callable[..., str]] = None,
):
    """
    Cache decorator for functions.
    
    Args:
        cache: Cache instance
        key_prefix: Key prefix
        ttl: TTL in seconds
        key_func: Custom key generation function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = generate_cache_key(
                    key_prefix,
                    args=args,
                    kwargs=kwargs,
                )
            
            # Try cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Call function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


__all__ = [
    "Cache",
    "CacheEntry",
    "InMemoryCache",
    "generate_cache_key",
    "cached",
]

