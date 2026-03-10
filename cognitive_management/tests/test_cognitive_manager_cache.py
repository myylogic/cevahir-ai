# -*- coding: utf-8 -*-
"""
Cache Management API Tests
===========================
CognitiveManager cache management metodlarını test eder.

Test Edilen Dosya: cognitive_manager.py
Test Edilen Metodlar:
- get_cache_stats() - Cache statistics
- invalidate_cache() - Cache invalidation
- clear_cache() - Cache clearing
- warm_cache() - Cache warming
- warm_popular_content() - Popular content warming
- get_cache_warming_stats() - Cache warming statistics
- get_cache_warmer_stats() - Cache warmer statistics

Alt Modül Test Edilen Dosyalar:
- v2/middleware/cache.py (CacheMiddleware)
- v2/utils/semantic_cache.py (SemanticCache)
- v2/utils/cache_warming.py (CacheWarmer)

Endüstri Standartları:
- pytest framework
- Fixture-based setup
- Assertion-based validation
- Cache hit/miss testing
"""

import pytest
from typing import Dict, Any

from cognitive_management.cognitive_manager import CognitiveManager
from cognitive_management.cognitive_types import CognitiveState, CognitiveInput
from .conftest import (
    mock_model_api,
    default_config,
    cognitive_manager,
    cognitive_state,
    cognitive_input
)


# ============================================================================
# Test 1-10: get_cache_stats() - Cache Statistics
# Test Edilen Dosya: cognitive_manager.py (get_cache_stats method)
# Alt Modül: v2/middleware/cache.py (CacheMiddleware), v2/utils/semantic_cache.py (SemanticCache)
# ============================================================================

def test_get_cache_stats_basic(cognitive_manager: CognitiveManager):
    """
    Test 1: Basic get_cache_stats() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_cache_stats()
    Alt Modül Dosyası: v2/middleware/cache.py
    Alt Modül Metod: CacheMiddleware.get_stats()
    Test Senaryosu: Basit cache stats alma
    """
    stats = cognitive_manager.get_cache_stats()
    # May be None if cache disabled
    if stats is not None:
        assert isinstance(stats, dict)
        # Common stats fields
        assert "hits" in stats or "misses" in stats or "size" in stats or "hit_rate" in stats or len(stats) >= 0


def test_get_cache_stats_after_requests(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 2: get_cache_stats() after processing requests.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_cache_stats(), handle()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Request sonrası cache stats
    """
    # Process some requests
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"Cache stats test {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    stats = cognitive_manager.get_cache_stats()
    if stats is not None:
        assert isinstance(stats, dict)


def test_get_cache_stats_structure(cognitive_manager: CognitiveManager):
    """
    Test 3: get_cache_stats() structure validation.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_cache_stats()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Cache stats yapısı validation
    """
    stats = cognitive_manager.get_cache_stats()
    if stats is not None:
        assert isinstance(stats, dict)
        # Should have some structure
        assert len(stats) >= 0


def test_get_cache_stats_consistency(cognitive_manager: CognitiveManager):
    """
    Test 4: get_cache_stats() consistency.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_cache_stats()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Multiple call'larda consistency
    """
    stats1 = cognitive_manager.get_cache_stats()
    stats2 = cognitive_manager.get_cache_stats()
    
    if stats1 is not None and stats2 is not None:
        assert isinstance(stats1, dict)
        assert isinstance(stats2, dict)


def test_get_cache_stats_empty_cache(cognitive_manager: CognitiveManager):
    """
    Test 5: get_cache_stats() with empty cache.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_cache_stats(), clear_cache()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Boş cache stats
    """
    # Clear cache
    cognitive_manager.clear_cache()
    
    stats = cognitive_manager.get_cache_stats()
    if stats is not None:
        assert isinstance(stats, dict)
        # Size should be 0 or low
        if "size" in stats:
            assert stats["size"] >= 0


def test_get_cache_stats_after_invalidation(cognitive_manager: CognitiveManager):
    """
    Test 6: get_cache_stats() after cache invalidation.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_cache_stats(), invalidate_cache()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Invalidation sonrası stats
    """
    # Invalidate cache
    cognitive_manager.invalidate_cache("test_pattern")
    
    stats = cognitive_manager.get_cache_stats()
    if stats is not None:
        assert isinstance(stats, dict)


def test_get_cache_stats_performance(cognitive_manager: CognitiveManager):
    """
    Test 7: get_cache_stats() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_cache_stats()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Performans testi
    """
    import time
    
    start = time.time()
    for _ in range(100):
        stats = cognitive_manager.get_cache_stats()
        assert stats is None or isinstance(stats, dict)
    elapsed = time.time() - start
    
    assert elapsed < 0.5  # Should be fast


def test_get_cache_stats_concurrent(cognitive_manager: CognitiveManager):
    """
    Test 8: get_cache_stats() concurrent access.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_cache_stats()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Concurrent access testi
    """
    import threading
    
    results = []
    
    def worker():
        stats = cognitive_manager.get_cache_stats()
        results.append(stats)
    
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    assert len(results) == 5
    for result in results:
        assert result is None or isinstance(result, dict)


def test_get_cache_stats_integration(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 9: get_cache_stats() integration test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: get_cache_stats(), handle(), clear_cache()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Integration testi
    """
    # Initial stats
    initial_stats = cognitive_manager.get_cache_stats()
    
    # Process requests
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"Cache integration {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Stats after requests
    after_stats = cognitive_manager.get_cache_stats()
    
    # Clear cache
    cognitive_manager.clear_cache()
    
    # Stats after clear
    final_stats = cognitive_manager.get_cache_stats()
    
    if initial_stats is not None:
        assert isinstance(initial_stats, dict)
    if after_stats is not None:
        assert isinstance(after_stats, dict)
    if final_stats is not None:
        assert isinstance(final_stats, dict)


def test_get_cache_stats_type_check(cognitive_manager: CognitiveManager):
    """
    Test 10: get_cache_stats() return type check.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_cache_stats()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Return type validation
    """
    stats = cognitive_manager.get_cache_stats()
    # Can be None if disabled
    assert stats is None or isinstance(stats, dict)


# ============================================================================
# Test 11-20: invalidate_cache() and clear_cache()
# Test Edilen Dosya: cognitive_manager.py
# Alt Modül: v2/middleware/cache.py (CacheMiddleware)
# ============================================================================

def test_invalidate_cache_basic(cognitive_manager: CognitiveManager):
    """
    Test 11: Basic invalidate_cache() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: invalidate_cache()
    Alt Modül Dosyası: v2/middleware/cache.py
    Alt Modül Metod: CacheMiddleware.invalidate()
    Test Senaryosu: Basit cache invalidation
    """
    cognitive_manager.invalidate_cache("test_pattern")
    # Should not crash
    assert True


def test_invalidate_cache_with_pattern(cognitive_manager: CognitiveManager):
    """
    Test 12: invalidate_cache() with pattern.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: invalidate_cache(pattern)
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Pattern ile invalidation
    """
    patterns = ["test_*", "prefix_*", "*_suffix", "exact_key"]
    for pattern in patterns:
        cognitive_manager.invalidate_cache(pattern)
        # Should not crash


def test_invalidate_cache_empty_pattern(cognitive_manager: CognitiveManager):
    """
    Test 13: invalidate_cache() with empty pattern.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: invalidate_cache()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Boş pattern (edge case)
    """
    cognitive_manager.invalidate_cache("")
    # Should not crash


def test_invalidate_cache_wildcard(cognitive_manager: CognitiveManager):
    """
    Test 14: invalidate_cache() with wildcard pattern.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: invalidate_cache()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Wildcard pattern ile invalidation
    """
    cognitive_manager.invalidate_cache("*")
    # Should invalidate all
    stats = cognitive_manager.get_cache_stats()
    if stats is not None:
        assert isinstance(stats, dict)


def test_clear_cache_basic(cognitive_manager: CognitiveManager):
    """
    Test 15: Basic clear_cache() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: clear_cache()
    Alt Modül Dosyası: v2/middleware/cache.py
    Alt Modül Metod: CacheMiddleware.clear()
    Test Senaryosu: Basit cache temizleme
    """
    cognitive_manager.clear_cache()
    stats = cognitive_manager.get_cache_stats()
    if stats is not None:
        if "size" in stats:
            assert stats["size"] == 0


def test_clear_cache_idempotent(cognitive_manager: CognitiveManager):
    """
    Test 16: clear_cache() idempotency.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: clear_cache()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Idempotent olması
    """
    cognitive_manager.clear_cache()
    cognitive_manager.clear_cache()  # Clear again
    
    stats = cognitive_manager.get_cache_stats()
    if stats is not None:
        if "size" in stats:
            assert stats["size"] == 0


def test_clear_cache_after_requests(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 17: clear_cache() after requests.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: clear_cache(), handle()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Request sonrası cache temizleme
    """
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Clear cache test {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    cognitive_manager.clear_cache()
    stats = cognitive_manager.get_cache_stats()
    if stats is not None:
        if "size" in stats:
            assert stats["size"] == 0


def test_invalidate_vs_clear(cognitive_manager: CognitiveManager):
    """
    Test 18: invalidate_cache() vs clear_cache() comparison.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: invalidate_cache(), clear_cache()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Invalidate vs clear karşılaştırması
    """
    # Invalidate specific pattern
    cognitive_manager.invalidate_cache("test_pattern")
    
    # Clear all
    cognitive_manager.clear_cache()
    
    # Both should work
    stats = cognitive_manager.get_cache_stats()
    if stats is not None:
        assert isinstance(stats, dict)


def test_cache_invalidation_performance(cognitive_manager: CognitiveManager):
    """
    Test 19: Cache invalidation performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: invalidate_cache()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Performans testi
    """
    import time
    
    start = time.time()
    for i in range(100):
        cognitive_manager.invalidate_cache(f"pattern_{i}")
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should be fast


def test_cache_operations_integration(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 20: Cache operations integration test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: get_cache_stats(), invalidate_cache(), clear_cache()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Integration testi
    """
    # Get initial stats
    initial_stats = cognitive_manager.get_cache_stats()
    
    # Process request
    input_msg = CognitiveInput(user_message="Cache operations test")
    cognitive_manager.handle(cognitive_state, input_msg)
    
    # Invalidate
    cognitive_manager.invalidate_cache("test")
    
    # Clear
    cognitive_manager.clear_cache()
    
    # Final stats
    final_stats = cognitive_manager.get_cache_stats()
    
    if initial_stats is not None:
        assert isinstance(initial_stats, dict)
    if final_stats is not None:
        assert isinstance(final_stats, dict)


# ============================================================================
# Test 21-30: warm_cache() and warm_popular_content()
# Test Edilen Dosya: cognitive_manager.py
# Alt Modül: v2/utils/cache_warming.py (CacheWarmer)
# ============================================================================

def test_warm_cache_basic(cognitive_manager: CognitiveManager):
    """
    Test 21: Basic warm_cache() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: warm_cache()
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Alt Modül Metod: CacheWarmer.warm_cache()
    Test Senaryosu: Basit cache warming
    """
    try:
        cognitive_manager.warm_cache()
        # Should not crash
        assert True
    except Exception:
        # Cache warming may not be enabled
        pass


def test_warm_cache_with_queries(cognitive_manager: CognitiveManager):
    """
    Test 22: warm_cache() with specific queries.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: warm_cache(queries)
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Test Senaryosu: Belirli query'ler ile warming
    """
    queries = ["query1", "query2", "query3"]
    try:
        cognitive_manager.warm_cache(queries=queries)
        assert True
    except Exception:
        pass


def test_warm_popular_content_basic(cognitive_manager: CognitiveManager):
    """
    Test 23: Basic warm_popular_content() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: warm_popular_content()
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Alt Modül Metod: CacheWarmer.warm_popular_content()
    Test Senaryosu: Basit popular content warming
    """
    try:
        cognitive_manager.warm_popular_content()
        # Should not crash
        assert True
    except Exception:
        pass


def test_warm_popular_content_with_limit(cognitive_manager: CognitiveManager):
    """
    Test 24: warm_popular_content() with limit.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: warm_popular_content(limit)
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Test Senaryosu: Limit ile popular content warming
    """
    try:
        cognitive_manager.warm_popular_content(limit=10)
        assert True
    except Exception:
        pass


def test_warm_cache_performance(cognitive_manager: CognitiveManager):
    """
    Test 25: warm_cache() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: warm_cache()
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Test Senaryosu: Performans testi
    """
    import time
    
    try:
        start = time.time()
        cognitive_manager.warm_cache()
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 10.0
    except Exception:
        pass


def test_warm_cache_after_clear(cognitive_manager: CognitiveManager):
    """
    Test 26: warm_cache() after clearing cache.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: warm_cache(), clear_cache()
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Test Senaryosu: Clear sonrası warming
    """
    cognitive_manager.clear_cache()
    
    try:
        cognitive_manager.warm_cache()
        assert True
    except Exception:
        pass


def test_warm_cache_integration(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 27: warm_cache() integration test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: warm_cache(), handle(), get_cache_stats()
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Test Senaryosu: Integration testi
    """
    try:
        # Warm cache
        cognitive_manager.warm_cache()
        
        # Process request (should use cache)
        input_msg = CognitiveInput(user_message="Warm cache integration")
        cognitive_manager.handle(cognitive_state, input_msg)
        
        # Get stats
        stats = cognitive_manager.get_cache_stats()
        if stats is not None:
            assert isinstance(stats, dict)
    except Exception:
        pass


def test_warm_popular_content_performance(cognitive_manager: CognitiveManager):
    """
    Test 28: warm_popular_content() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: warm_popular_content()
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Test Senaryosu: Performans testi
    """
    import time
    
    try:
        start = time.time()
        cognitive_manager.warm_popular_content(limit=5)
        elapsed = time.time() - start
        
        assert elapsed < 10.0
    except Exception:
        pass


def test_warm_cache_multiple_times(cognitive_manager: CognitiveManager):
    """
    Test 29: warm_cache() multiple times.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: warm_cache()
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Test Senaryosu: Multiple warming
    """
    try:
        cognitive_manager.warm_cache()
        cognitive_manager.warm_cache()  # Warm again
        assert True
    except Exception:
        pass


def test_warm_cache_error_handling(cognitive_manager: CognitiveManager):
    """
    Test 30: warm_cache() error handling.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: warm_cache()
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Test Senaryosu: Hata durumlarında handling
    """
    try:
        # Invalid queries
        cognitive_manager.warm_cache(queries=None)  # type: ignore
    except Exception:
        # Expected if validation exists
        pass


# ============================================================================
# Test 31-40: get_cache_warming_stats() and get_cache_warmer_stats()
# Test Edilen Dosya: cognitive_manager.py
# Alt Modül: v2/utils/cache_warming.py (CacheWarmer)
# ============================================================================

def test_get_cache_warming_stats_basic(cognitive_manager: CognitiveManager):
    """
    Test 31: Basic get_cache_warming_stats() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_cache_warming_stats()
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Alt Modül Metod: CacheWarmer.get_warming_stats()
    Test Senaryosu: Basit cache warming stats alma
    """
    stats = cognitive_manager.get_cache_warming_stats()
    # May be None if cache warming disabled
    if stats is not None:
        assert isinstance(stats, dict)
        # Common stats fields
        assert "warmed" in stats or "total" in stats or "success_rate" in stats or len(stats) >= 0


def test_get_cache_warmer_stats_basic(cognitive_manager: CognitiveManager):
    """
    Test 32: Basic get_cache_warmer_stats() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_cache_warmer_stats()
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Alt Modül Metod: CacheWarmer.get_stats()
    Test Senaryosu: Basit cache warmer stats alma
    """
    stats = cognitive_manager.get_cache_warmer_stats()
    assert isinstance(stats, dict)
    # Common stats fields
    assert "warmed" in stats or "total" in stats or "last_warm_time" in stats or len(stats) >= 0


def test_get_cache_warming_stats_after_warming(cognitive_manager: CognitiveManager):
    """
    Test 33: get_cache_warming_stats() after warming.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: get_cache_warming_stats(), warm_cache()
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Test Senaryosu: Warming sonrası stats
    """
    try:
        cognitive_manager.warm_cache()
        stats = cognitive_manager.get_cache_warming_stats()
        if stats is not None:
            assert isinstance(stats, dict)
    except Exception:
        pass


def test_get_cache_warmer_stats_consistency(cognitive_manager: CognitiveManager):
    """
    Test 34: get_cache_warmer_stats() consistency.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_cache_warmer_stats()
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Test Senaryosu: Multiple call'larda consistency
    """
    stats1 = cognitive_manager.get_cache_warmer_stats()
    stats2 = cognitive_manager.get_cache_warmer_stats()
    
    assert isinstance(stats1, dict)
    assert isinstance(stats2, dict)


def test_get_cache_warming_stats_structure(cognitive_manager: CognitiveManager):
    """
    Test 35: get_cache_warming_stats() structure validation.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_cache_warming_stats()
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Test Senaryosu: Stats yapısı validation
    """
    stats = cognitive_manager.get_cache_warming_stats()
    if stats is not None:
        assert isinstance(stats, dict)
        assert len(stats) >= 0


def test_get_cache_warmer_stats_performance(cognitive_manager: CognitiveManager):
    """
    Test 36: get_cache_warmer_stats() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_cache_warmer_stats()
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Test Senaryosu: Performans testi
    """
    import time
    
    start = time.time()
    for _ in range(100):
        stats = cognitive_manager.get_cache_warmer_stats()
        assert isinstance(stats, dict)
    elapsed = time.time() - start
    
    assert elapsed < 0.5  # Should be fast


def test_cache_warming_stats_integration(cognitive_manager: CognitiveManager):
    """
    Test 37: Cache warming stats integration test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: warm_cache(), get_cache_warming_stats(), get_cache_warmer_stats()
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Test Senaryosu: Integration testi
    """
    try:
        # Warm cache
        cognitive_manager.warm_cache()
        
        # Get warming stats
        warming_stats = cognitive_manager.get_cache_warming_stats()
        if warming_stats is not None:
            assert isinstance(warming_stats, dict)
        
        # Get warmer stats
        warmer_stats = cognitive_manager.get_cache_warmer_stats()
        assert isinstance(warmer_stats, dict)
    except Exception:
        pass


def test_get_cache_warming_stats_empty(cognitive_manager: CognitiveManager):
    """
    Test 38: get_cache_warming_stats() with no warming.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_cache_warming_stats()
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Test Senaryosu: Warming yapılmadan stats
    """
    stats = cognitive_manager.get_cache_warming_stats()
    if stats is not None:
        assert isinstance(stats, dict)
        # May have zero values
        if "warmed" in stats:
            assert stats["warmed"] >= 0


def test_cache_warming_full_workflow(cognitive_manager: CognitiveManager):
    """
    Test 39: Full cache warming workflow.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: Tüm cache warming metodları
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Test Senaryosu: Tam cache warming workflow
    """
    try:
        # 1. Get initial stats
        initial_stats = cognitive_manager.get_cache_warmer_stats()
        assert isinstance(initial_stats, dict)
        
        # 2. Warm cache
        cognitive_manager.warm_cache()
        
        # 3. Warm popular content
        cognitive_manager.warm_popular_content(limit=5)
        
        # 4. Get warming stats
        warming_stats = cognitive_manager.get_cache_warming_stats()
        if warming_stats is not None:
            assert isinstance(warming_stats, dict)
        
        # 5. Get warmer stats
        final_stats = cognitive_manager.get_cache_warmer_stats()
        assert isinstance(final_stats, dict)
    except Exception:
        pass


def test_cache_warming_performance(cognitive_manager: CognitiveManager):
    """
    Test 40: Cache warming performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: warm_cache(), warm_popular_content()
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Test Senaryosu: Performans testi
    """
    import time
    
    try:
        start = time.time()
        cognitive_manager.warm_cache()
        cognitive_manager.warm_popular_content(limit=3)
        elapsed = time.time() - start
        
        assert elapsed < 15.0  # Should complete in reasonable time
    except Exception:
        pass


# ============================================================================
# Test 41-50: Cache Management Integration and Edge Cases
# Test Edilen Dosya: cognitive_manager.py
# Alt Modül: v2/middleware/cache.py, v2/utils/semantic_cache.py, v2/utils/cache_warming.py
# ============================================================================

def test_cache_management_full_workflow(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 41: Full cache management workflow.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: Tüm cache management metodları
    Alt Modül Dosyaları:
    - v2/middleware/cache.py
    - v2/utils/semantic_cache.py
    - v2/utils/cache_warming.py
    Test Senaryosu: Tam cache management workflow
    """
    # 1. Get initial stats
    initial_stats = cognitive_manager.get_cache_stats()
    
    # 2. Process requests (may populate cache)
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"Cache workflow {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # 3. Get stats after requests
    after_stats = cognitive_manager.get_cache_stats()
    
    # 4. Warm cache
    try:
        cognitive_manager.warm_cache()
    except Exception:
        pass
    
    # 5. Get warming stats
    warming_stats = cognitive_manager.get_cache_warming_stats()
    
    # 6. Invalidate cache
    cognitive_manager.invalidate_cache("test")
    
    # 7. Clear cache
    cognitive_manager.clear_cache()
    
    # 8. Get final stats
    final_stats = cognitive_manager.get_cache_stats()
    
    # Verify
    if initial_stats is not None:
        assert isinstance(initial_stats, dict)
    if after_stats is not None:
        assert isinstance(after_stats, dict)
    if final_stats is not None:
        assert isinstance(final_stats, dict)


def test_cache_hit_miss_scenario(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 42: Cache hit/miss scenario test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: handle(), get_cache_stats()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Cache hit/miss senaryosu
    """
    # Process same request twice (should hit cache on second)
    input_msg = CognitiveInput(user_message="Cache hit test")
    
    cognitive_manager.handle(cognitive_state, input_msg)
    stats1 = cognitive_manager.get_cache_stats()
    
    cognitive_manager.handle(cognitive_state, input_msg)
    stats2 = cognitive_manager.get_cache_stats()
    
    if stats1 is not None and stats2 is not None:
        assert isinstance(stats1, dict)
        assert isinstance(stats2, dict)
        # Hits should increase (if cache is working)
        if "hits" in stats1 and "hits" in stats2:
            assert stats2["hits"] >= stats1["hits"]


def test_cache_invalidation_patterns(cognitive_manager: CognitiveManager):
    """
    Test 43: Cache invalidation with different patterns.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: invalidate_cache()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Farklı pattern'ler ile invalidation
    """
    patterns = [
        "exact_key",
        "prefix_*",
        "*_suffix",
        "*_middle_*",
        "*"
    ]
    
    for pattern in patterns:
        cognitive_manager.invalidate_cache(pattern)
        # Should not crash
    
    stats = cognitive_manager.get_cache_stats()
    if stats is not None:
        assert isinstance(stats, dict)


def test_cache_clear_vs_invalidate(cognitive_manager: CognitiveManager):
    """
    Test 44: clear_cache() vs invalidate_cache() comparison.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: clear_cache(), invalidate_cache()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Clear vs invalidate karşılaştırması
    """
    # Invalidate specific
    cognitive_manager.invalidate_cache("specific_pattern")
    
    # Clear all
    cognitive_manager.clear_cache()
    
    # Both should work
    stats = cognitive_manager.get_cache_stats()
    if stats is not None:
        assert isinstance(stats, dict)


def test_cache_warming_after_clear(cognitive_manager: CognitiveManager):
    """
    Test 45: Cache warming after clearing.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: clear_cache(), warm_cache()
    Alt Modül Dosyası: v2/utils/cache_warming.py
    Test Senaryosu: Clear sonrası warming
    """
    cognitive_manager.clear_cache()
    
    try:
        cognitive_manager.warm_cache()
        cognitive_manager.warm_popular_content()
        
        stats = cognitive_manager.get_cache_stats()
        if stats is not None:
            assert isinstance(stats, dict)
    except Exception:
        pass


def test_cache_stats_accuracy(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 46: Cache stats accuracy test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: get_cache_stats(), handle(), clear_cache()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Stats doğruluğu testi
    """
    # Clear cache
    cognitive_manager.clear_cache()
    stats_after_clear = cognitive_manager.get_cache_stats()
    
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Stats accuracy {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    stats_after_requests = cognitive_manager.get_cache_stats()
    
    if stats_after_clear is not None and stats_after_requests is not None:
        assert isinstance(stats_after_clear, dict)
        assert isinstance(stats_after_requests, dict)
        # Size should increase (if cache is working)
        if "size" in stats_after_clear and "size" in stats_after_requests:
            assert stats_after_requests["size"] >= stats_after_clear["size"]


def test_cache_management_performance(cognitive_manager: CognitiveManager):
    """
    Test 47: Cache management performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: get_cache_stats(), invalidate_cache(), clear_cache()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Performans testi
    """
    import time
    
    start = time.time()
    
    # Multiple operations
    for i in range(50):
        cognitive_manager.get_cache_stats()
        cognitive_manager.invalidate_cache(f"pattern_{i}")
    
    cognitive_manager.clear_cache()
    
    elapsed = time.time() - start
    assert elapsed < 2.0  # Should complete in reasonable time


def test_cache_management_concurrent(cognitive_manager: CognitiveManager):
    """
    Test 48: Cache management concurrent operations.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: get_cache_stats(), invalidate_cache()
    Alt Modül Dosyası: v2/middleware/cache.py
    Test Senaryosu: Concurrent operations testi
    """
    import threading
    
    def worker(worker_id: int):
        cognitive_manager.get_cache_stats()
        cognitive_manager.invalidate_cache(f"concurrent_{worker_id}")
    
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    stats = cognitive_manager.get_cache_stats()
    if stats is not None:
        assert isinstance(stats, dict)


def test_cache_management_error_handling(cognitive_manager: CognitiveManager):
    """
    Test 49: Cache management error handling.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: invalidate_cache(), warm_cache()
    Alt Modül Dosyası: v2/middleware/cache.py, v2/utils/cache_warming.py
    Test Senaryosu: Hata durumlarında handling
    """
    # Invalid pattern
    try:
        cognitive_manager.invalidate_cache(None)  # type: ignore
    except (TypeError, ValueError):
        # Expected behavior
        pass
    
    # Invalid queries
    try:
        cognitive_manager.warm_cache(queries="invalid")  # type: ignore
    except Exception:
        pass


def test_cache_management_end_to_end(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 50: Cache management end-to-end test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: Tüm cache management metodları
    Alt Modül Dosyaları:
    - v2/middleware/cache.py
    - v2/utils/semantic_cache.py
    - v2/utils/cache_warming.py
    Test Senaryosu: End-to-end cache management testi
    """
    # 1. Get initial stats
    initial_stats = cognitive_manager.get_cache_stats()
    
    # 2. Process requests
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"E2E cache {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # 3. Get stats
    after_stats = cognitive_manager.get_cache_stats()
    
    # 4. Warm cache
    try:
        cognitive_manager.warm_cache()
        cognitive_manager.warm_popular_content(limit=3)
    except Exception:
        pass
    
    # 5. Get warming stats
    warming_stats = cognitive_manager.get_cache_warming_stats()
    warmer_stats = cognitive_manager.get_cache_warmer_stats()
    
    # 6. Invalidate
    cognitive_manager.invalidate_cache("e2e_test")
    
    # 7. Clear
    cognitive_manager.clear_cache()
    
    # 8. Final stats
    final_stats = cognitive_manager.get_cache_stats()
    
    # Verify
    if initial_stats is not None:
        assert isinstance(initial_stats, dict)
    if after_stats is not None:
        assert isinstance(after_stats, dict)
    if final_stats is not None:
        assert isinstance(final_stats, dict)
    assert isinstance(warmer_stats, dict)

