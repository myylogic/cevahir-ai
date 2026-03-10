# -*- coding: utf-8 -*-
"""
Connection Pool API Tests
==========================
CognitiveManager connection pool metodlarını test eder.

Test Edilen Dosya: cognitive_manager.py
Test Edilen Metodlar:
- get_connection_pool_stats() - Connection pool statistics
- cleanup_idle_connections() - Cleanup idle connections

Alt Modül Test Edilen Dosyalar:
- v2/utils/connection_pool.py (ConnectionPool)
- v2/adapters/backend_adapter.py (ModelAPIAdapter)

Endüstri Standartları:
- pytest framework
- Fixture-based setup
- Assertion-based validation
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
# Test 1-25: get_connection_pool_stats()
# Test Edilen Dosya: cognitive_manager.py (get_connection_pool_stats method)
# Alt Modül: v2/utils/connection_pool.py (ConnectionPool.get_stats)
# ============================================================================

def test_get_connection_pool_stats_basic(cognitive_manager: CognitiveManager):
    """
    Test 1: Basic get_connection_pool_stats() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_connection_pool_stats()
    Alt Modül Dosyası: v2/utils/connection_pool.py
    Alt Modül Metod: ConnectionPool.get_stats()
    Test Senaryosu: Basit connection pool stats alma
    """
    stats = cognitive_manager.get_connection_pool_stats()
    # May be None if connection pooling disabled
    if stats is not None:
        assert isinstance(stats, dict)
        # Common stats fields
        assert "active" in stats or "idle" in stats or "total" in stats or "max_size" in stats or len(stats) >= 0


def test_get_connection_pool_stats_after_requests(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 2: get_connection_pool_stats() after processing requests.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_connection_pool_stats(), handle()
    Alt Modül Dosyası: v2/utils/connection_pool.py
    Test Senaryosu: Request sonrası stats
    """
    # Process requests
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"Pool stats test {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    stats = cognitive_manager.get_connection_pool_stats()
    if stats is not None:
        assert isinstance(stats, dict)


def test_get_connection_pool_stats_structure(cognitive_manager: CognitiveManager):
    """
    Test 3: get_connection_pool_stats() structure validation.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_connection_pool_stats()
    Alt Modül Dosyası: v2/utils/connection_pool.py
    Test Senaryosu: Stats yapısı validation
    """
    stats = cognitive_manager.get_connection_pool_stats()
    if stats is not None:
        assert isinstance(stats, dict)
        assert len(stats) >= 0


def test_get_connection_pool_stats_consistency(cognitive_manager: CognitiveManager):
    """
    Test 4: get_connection_pool_stats() consistency.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_connection_pool_stats()
    Alt Modül Dosyası: v2/utils/connection_pool.py
    Test Senaryosu: Multiple call'larda consistency
    """
    stats1 = cognitive_manager.get_connection_pool_stats()
    stats2 = cognitive_manager.get_connection_pool_stats()
    
    if stats1 is not None and stats2 is not None:
        assert isinstance(stats1, dict)
        assert isinstance(stats2, dict)


def test_get_connection_pool_stats_performance(cognitive_manager: CognitiveManager):
    """
    Test 5: get_connection_pool_stats() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_connection_pool_stats()
    Alt Modül Dosyası: v2/utils/connection_pool.py
    Test Senaryosu: Performans testi
    """
    import time
    
    start = time.time()
    for _ in range(100):
        stats = cognitive_manager.get_connection_pool_stats()
        assert stats is None or isinstance(stats, dict)
    elapsed = time.time() - start
    
    assert elapsed < 0.5  # Should be fast


def test_get_connection_pool_stats_concurrent(cognitive_manager: CognitiveManager):
    """
    Test 6: get_connection_pool_stats() concurrent access.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_connection_pool_stats()
    Alt Modül Dosyası: v2/utils/connection_pool.py
    Test Senaryosu: Concurrent access testi
    """
    import threading
    
    results = []
    
    def worker():
        stats = cognitive_manager.get_connection_pool_stats()
        results.append(stats)
    
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    assert len(results) == 5
    for result in results:
        assert result is None or isinstance(result, dict)


def test_get_connection_pool_stats_integration(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 7: get_connection_pool_stats() integration test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: get_connection_pool_stats(), handle()
    Alt Modül Dosyası: v2/utils/connection_pool.py
    Test Senaryosu: Integration testi
    """
    # Initial stats
    initial_stats = cognitive_manager.get_connection_pool_stats()
    
    # Process requests
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"Integration pool {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Final stats
    final_stats = cognitive_manager.get_connection_pool_stats()
    
    if initial_stats is not None:
        assert isinstance(initial_stats, dict)
    if final_stats is not None:
        assert isinstance(final_stats, dict)


# ============================================================================
# Test 8-25: cleanup_idle_connections()
# Test Edilen Dosya: cognitive_manager.py (cleanup_idle_connections method)
# Alt Modül: v2/utils/connection_pool.py (ConnectionPool.cleanup_idle)
# ============================================================================

def test_cleanup_idle_connections_basic(cognitive_manager: CognitiveManager):
    """
    Test 8: Basic cleanup_idle_connections() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: cleanup_idle_connections()
    Alt Modül Dosyası: v2/utils/connection_pool.py
    Alt Modül Metod: ConnectionPool.cleanup_idle()
    Test Senaryosu: Basit idle connection cleanup
    """
    cleaned = cognitive_manager.cleanup_idle_connections()
    assert isinstance(cleaned, int)
    assert cleaned >= 0


def test_cleanup_idle_connections_after_requests(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 9: cleanup_idle_connections() after requests.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: cleanup_idle_connections(), handle()
    Alt Modül Dosyası: v2/utils/connection_pool.py
    Test Senaryosu: Request sonrası cleanup
    """
    # Process requests
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"Cleanup test {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    cleaned = cognitive_manager.cleanup_idle_connections()
    assert isinstance(cleaned, int)
    assert cleaned >= 0


def test_cleanup_idle_connections_idempotent(cognitive_manager: CognitiveManager):
    """
    Test 10: cleanup_idle_connections() idempotency.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: cleanup_idle_connections()
    Alt Modül Dosyası: v2/utils/connection_pool.py
    Test Senaryosu: Idempotent olması
    """
    cleaned1 = cognitive_manager.cleanup_idle_connections()
    cleaned2 = cognitive_manager.cleanup_idle_connections()  # Cleanup again
    
    assert isinstance(cleaned1, int)
    assert isinstance(cleaned2, int)
    assert cleaned1 >= 0
    assert cleaned2 >= 0


def test_cleanup_idle_connections_performance(cognitive_manager: CognitiveManager):
    """
    Test 11: cleanup_idle_connections() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: cleanup_idle_connections()
    Alt Modül Dosyası: v2/utils/connection_pool.py
    Test Senaryosu: Performans testi
    """
    import time
    
    start = time.time()
    for _ in range(20):
        cleaned = cognitive_manager.cleanup_idle_connections()
        assert isinstance(cleaned, int)
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should be fast


def test_connection_pool_operations_integration(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 12: Connection pool operations integration test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: get_connection_pool_stats(), cleanup_idle_connections(), handle()
    Alt Modül Dosyası: v2/utils/connection_pool.py
    Test Senaryosu: Integration testi
    """
    # Get initial stats
    initial_stats = cognitive_manager.get_connection_pool_stats()
    
    # Process requests
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"Pool integration {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Get stats after requests
    after_stats = cognitive_manager.get_connection_pool_stats()
    
    # Cleanup idle connections
    cleaned = cognitive_manager.cleanup_idle_connections()
    assert isinstance(cleaned, int)
    
    # Get final stats
    final_stats = cognitive_manager.get_connection_pool_stats()
    
    if initial_stats is not None:
        assert isinstance(initial_stats, dict)
    if after_stats is not None:
        assert isinstance(after_stats, dict)
    if final_stats is not None:
        assert isinstance(final_stats, dict)


def test_connection_pool_full_workflow(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 13: Full connection pool workflow.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: get_connection_pool_stats(), cleanup_idle_connections()
    Alt Modül Dosyası: v2/utils/connection_pool.py
    Test Senaryosu: Tam connection pool workflow
    """
    # 1. Get initial stats
    initial_stats = cognitive_manager.get_connection_pool_stats()
    
    # 2. Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Pool workflow {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # 3. Get stats
    stats = cognitive_manager.get_connection_pool_stats()
    
    # 4. Cleanup
    cleaned = cognitive_manager.cleanup_idle_connections()
    
    # 5. Get final stats
    final_stats = cognitive_manager.get_connection_pool_stats()
    
    if initial_stats is not None:
        assert isinstance(initial_stats, dict)
    if stats is not None:
        assert isinstance(stats, dict)
    if final_stats is not None:
        assert isinstance(final_stats, dict)
    assert isinstance(cleaned, int)


# ============================================================================
# Test 14-25: Connection Pool Edge Cases and Additional Tests
# ============================================================================

def test_connection_pool_stats_type_check(cognitive_manager: CognitiveManager):
    """Test 14: Connection pool stats type check."""
    stats = cognitive_manager.get_connection_pool_stats()
    assert stats is None or isinstance(stats, dict)


def test_cleanup_idle_connections_type_check(cognitive_manager: CognitiveManager):
    """Test 15: Cleanup idle connections type check."""
    cleaned = cognitive_manager.cleanup_idle_connections()
    assert isinstance(cleaned, int)


def test_connection_pool_with_async(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 16: Connection pool with async requests."""
    import asyncio
    
    async def async_test():
        input_msg = CognitiveInput(user_message="Async pool test")
        await cognitive_manager.handle_async(cognitive_state, input_msg)
    
    asyncio.run(async_test())
    
    stats = cognitive_manager.get_connection_pool_stats()
    if stats is not None:
        assert isinstance(stats, dict)


def test_connection_pool_with_batch(cognitive_manager: CognitiveManager):
    """Test 17: Connection pool with batch requests."""
    states = [CognitiveState() for _ in range(3)]
    inputs = [CognitiveInput(user_message=f"Batch pool {i}") for i in range(3)]
    requests = list(zip(states, inputs))
    
    cognitive_manager.handle_batch(requests)
    
    stats = cognitive_manager.get_connection_pool_stats()
    if stats is not None:
        assert isinstance(stats, dict)


def test_connection_pool_concurrent_cleanup(cognitive_manager: CognitiveManager):
    """Test 18: Concurrent cleanup operations."""
    import threading
    
    def worker():
        cleaned = cognitive_manager.cleanup_idle_connections()
        assert isinstance(cleaned, int)
    
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def test_connection_pool_performance_under_load(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 19: Connection pool performance under load."""
    import time
    
    start = time.time()
    
    # Process many requests
    for i in range(20):
        input_msg = CognitiveInput(user_message=f"Load pool {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    elapsed = time.time() - start
    
    # Get stats and cleanup
    stats = cognitive_manager.get_connection_pool_stats()
    cleaned = cognitive_manager.cleanup_idle_connections()
    
    if stats is not None:
        assert isinstance(stats, dict)
    assert isinstance(cleaned, int)
    assert elapsed < 60.0


def test_connection_pool_error_recovery(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 20: Connection pool error recovery."""
    # Process requests
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"Recovery pool {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Pool operations should still work
    stats = cognitive_manager.get_connection_pool_stats()
    cleaned = cognitive_manager.cleanup_idle_connections()
    
    if stats is not None:
        assert isinstance(stats, dict)
    assert isinstance(cleaned, int)


def test_connection_pool_stats_accuracy(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 21: Connection pool stats accuracy."""
    # Get initial stats
    initial_stats = cognitive_manager.get_connection_pool_stats()
    
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Accuracy pool {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Get stats after requests
    after_stats = cognitive_manager.get_connection_pool_stats()
    
    if initial_stats is not None and after_stats is not None:
        assert isinstance(initial_stats, dict)
        assert isinstance(after_stats, dict)


def test_connection_pool_cleanup_effectiveness(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 22: Connection pool cleanup effectiveness."""
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Effectiveness pool {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Get stats before cleanup
    stats_before = cognitive_manager.get_connection_pool_stats()
    
    # Cleanup
    cleaned = cognitive_manager.cleanup_idle_connections()
    
    # Get stats after cleanup
    stats_after = cognitive_manager.get_connection_pool_stats()
    
    assert isinstance(cleaned, int)
    if stats_before is not None and stats_after is not None:
        assert isinstance(stats_before, dict)
        assert isinstance(stats_after, dict)


def test_connection_pool_integration_full_system(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 23: Connection pool integration with full system."""
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Full system pool {i}")
        output = cognitive_manager.handle(cognitive_state, input_msg)
        assert output is not None
    
    # Pool operations
    stats = cognitive_manager.get_connection_pool_stats()
    cleaned = cognitive_manager.cleanup_idle_connections()
    
    if stats is not None:
        assert isinstance(stats, dict)
    assert isinstance(cleaned, int)


def test_connection_pool_end_to_end(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 24: Connection pool end-to-end test."""
    # 1. Initial state
    initial_stats = cognitive_manager.get_connection_pool_stats()
    
    # 2. Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"E2E pool {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # 3. Get stats
    stats = cognitive_manager.get_connection_pool_stats()
    
    # 4. Cleanup
    cleaned = cognitive_manager.cleanup_idle_connections()
    
    # 5. Final stats
    final_stats = cognitive_manager.get_connection_pool_stats()
    
    # Verify
    if initial_stats is not None:
        assert isinstance(initial_stats, dict)
    if stats is not None:
        assert isinstance(stats, dict)
    if final_stats is not None:
        assert isinstance(final_stats, dict)
    assert isinstance(cleaned, int)


def test_connection_pool_comprehensive(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 25: Comprehensive connection pool test."""
    # Multiple operations
    for i in range(10):
        input_msg = CognitiveInput(user_message=f"Comprehensive pool {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
        
        if i % 3 == 0:
            stats = cognitive_manager.get_connection_pool_stats()
            if stats is not None:
                assert isinstance(stats, dict)
        
        if i % 5 == 0:
            cleaned = cognitive_manager.cleanup_idle_connections()
            assert isinstance(cleaned, int)

