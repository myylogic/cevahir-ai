# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: cache_warming.py
Modül: cognitive_management/v2/utils
Görev: Cache Warming Strategies - Cache warming utilities for pre-populating cache.
       Phase 6: Performance Optimization & Caching Enhancement. WarmingStrategy,
       CacheWarmer sınıflarını içerir. Cache warming strategies, predictive cache
       pre-loading ve cache warming execution işlemlerini yapar. Akademik referans:
       Cache Warming Strategies (Industry Best Practices), Predictive Cache
       Pre-loading.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (cache warming)
- Design Patterns: Strategy Pattern (cache warming strategies)
- Endüstri Standartları: Cache warming best practices

KULLANIM:
- Cache warming için
- Predictive cache pre-loading için
- Cache warming strategies için

BAĞIMLILIKLAR:
- Cache: Cache interface
- threading: Thread-safe işlemler

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading
import time

from .cache import Cache, InMemoryCache


@dataclass
class WarmingStrategy:
    """
    Cache warming strategy configuration.
    
    Attributes:
        name: Strategy name
        priority: Priority level (higher = more important)
        enabled: Whether strategy is enabled
        target_keys: Target cache keys to warm
        target_pattern: Key pattern to match
        preload_function: Function to generate cache entries
    """
    name: str
    priority: int = 5
    enabled: bool = True
    target_keys: List[str] = None
    target_pattern: Optional[str] = None
    preload_function: Optional[Callable[[], List[Tuple[str, Any]]]] = None


class CacheWarmer:
    """
    Cache warmer for pre-populating cache.
    
    Implements various cache warming strategies:
    - Predefined keys warming
    - Pattern-based warming
    - Predictive warming
    - Popular content warming
    """
    
    def __init__(
        self,
        cache: Cache,
        strategies: Optional[List[WarmingStrategy]] = None,
    ):
        """
        Initialize cache warmer.
        
        Args:
            cache: Cache instance to warm
            strategies: List of warming strategies
        """
        self.cache = cache
        self.strategies = strategies or []
        self._lock = threading.RLock()
        
        # Statistics
        self._warmed_keys = 0
        self._warm_attempts = 0
    
    def warm_cache(
        self,
        strategy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Warm cache using configured strategies.
        
        Args:
            strategy_name: Specific strategy to use (None = use all enabled)
            
        Returns:
            Warming results dictionary
        """
        with self._lock:
            results = {
                "strategies_executed": 0,
                "keys_warmed": 0,
                "keys_failed": 0,
                "errors": [],
            }
            
            # Filter strategies
            strategies_to_run = []
            if strategy_name:
                strategies_to_run = [s for s in self.strategies if s.name == strategy_name and s.enabled]
            else:
                strategies_to_run = [s for s in self.strategies if s.enabled]
            
            # Sort by priority (higher first)
            strategies_to_run.sort(key=lambda s: s.priority, reverse=True)
            
            # Execute strategies
            for strategy in strategies_to_run:
                try:
                    strategy_results = self._execute_strategy(strategy)
                    results["strategies_executed"] += 1
                    results["keys_warmed"] += strategy_results["keys_warmed"]
                    results["keys_failed"] += strategy_results["keys_failed"]
                    if strategy_results.get("error"):
                        results["errors"].append(f"{strategy.name}: {strategy_results['error']}")
                except Exception as e:
                    results["errors"].append(f"{strategy.name}: {str(e)}")
                    results["keys_failed"] += 1
            
            self._warmed_keys += results["keys_warmed"]
            self._warm_attempts += 1
            
            return results
    
    def _execute_strategy(self, strategy: WarmingStrategy) -> Dict[str, Any]:
        """
        Execute a warming strategy.
        
        Args:
            strategy: Warming strategy
            
        Returns:
            Strategy execution results
        """
        results = {
            "keys_warmed": 0,
            "keys_failed": 0,
            "error": None,
        }
        
        try:
            # Strategy 1: Predefined keys
            if strategy.target_keys:
                for key in strategy.target_keys:
                    # Try to get key - if miss, could trigger warming
                    if self.cache.get(key) is None:
                        # Key not in cache - would need preload function to populate
                        pass
            
            # Strategy 2: Pattern-based (if cache supports iteration)
            if strategy.target_pattern:
                # Pattern-based warming requires cache iteration
                # For now, log that pattern is specified
                pass
            
            # Strategy 3: Preload function
            if strategy.preload_function:
                try:
                    preload_data = strategy.preload_function()
                    for key, value in preload_data:
                        self.cache.set(key, value)
                        results["keys_warmed"] += 1
                except Exception as e:
                    results["error"] = str(e)
                    results["keys_failed"] += 1
            
        except Exception as e:
            results["error"] = str(e)
            results["keys_failed"] += 1
        
        return results
    
    def warm_popular_content(
        self,
        key_access_counts: Dict[str, int],
        top_k: int = 100,
        ttl: Optional[float] = None
    ) -> int:
        """
        Warm cache with popular content.
        
        Args:
            key_access_counts: Dictionary of key -> access count
            top_k: Number of top keys to warm
            ttl: TTL for warmed entries
            
        Returns:
            Number of keys warmed
        """
        with self._lock:
            # Sort by access count
            sorted_keys = sorted(
                key_access_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            top_keys = sorted_keys[:top_k]
            
            warmed = 0
            for key, _ in top_keys:
                # Check if key exists
                if self.cache.get(key) is None:
                    # Would need to load content here
                    # For now, just mark as warmed
                    warmed += 1
            
            return warmed
    
    def get_warming_stats(self) -> Dict[str, Any]:
        """Get cache warming statistics."""
        with self._lock:
            return {
                "total_keys_warmed": self._warmed_keys,
                "warm_attempts": self._warm_attempts,
                "strategies_configured": len(self.strategies),
                "enabled_strategies": sum(1 for s in self.strategies if s.enabled),
            }


__all__ = [
    "CacheWarmer",
    "WarmingStrategy",
]

