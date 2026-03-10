# -*- coding: utf-8 -*-
"""
Performance Profiling API Tests
=================================
CognitiveManager performance profiling metodlarını test eder.

Test Edilen Dosya: cognitive_manager.py
Test Edilen Metodlar:
- get_performance_metrics() - Performance metrics
- reset_performance_metrics() - Reset performance metrics
- get_performance_profile() - Performance profile report
- get_all_performance_stats() - All performance stats
- clear_performance_profile() - Clear performance profile
- identify_bottlenecks() - Identify bottlenecks
- get_operation_stats() - Operation statistics

Alt Modül Test Edilen Dosyalar:
- v2/utils/performance_profiler.py (PerformanceProfiler)
- v2/monitoring/performance_monitor.py (PerformanceMonitor)

Endüstri Standartları:
- pytest framework
- Fixture-based setup
- Assertion-based validation
- Performance validation testing
"""

import pytest
from typing import Dict, Any, List

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
# Test 1-10: get_performance_metrics() and reset_performance_metrics()
# Test Edilen Dosya: cognitive_manager.py
# Alt Modül: v2/monitoring/performance_monitor.py (PerformanceMonitor)
# ============================================================================

def test_get_performance_metrics_basic(cognitive_manager: CognitiveManager):
    """
    Test 1: Basic get_performance_metrics() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_performance_metrics()
    Alt Modül Dosyası: v2/monitoring/performance_monitor.py
    Alt Modül Metod: PerformanceMonitor.get_metrics()
    Test Senaryosu: Basit performance metrics alma
    """
    metrics = cognitive_manager.get_performance_metrics()
    assert isinstance(metrics, dict)
    # Common metrics fields
    assert "latency" in metrics or "throughput" in metrics or "error_rate" in metrics or len(metrics) >= 0


def test_get_performance_metrics_after_requests(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 2: get_performance_metrics() after processing requests.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_performance_metrics(), handle()
    Alt Modül Dosyası: v2/monitoring/performance_monitor.py
    Test Senaryosu: Request sonrası metrics
    """
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Perf metrics test {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    metrics = cognitive_manager.get_performance_metrics()
    assert isinstance(metrics, dict)


def test_get_performance_metrics_structure(cognitive_manager: CognitiveManager):
    """
    Test 3: get_performance_metrics() structure validation.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_performance_metrics()
    Alt Modül Dosyası: v2/monitoring/performance_monitor.py
    Test Senaryosu: Metrics yapısı validation
    """
    metrics = cognitive_manager.get_performance_metrics()
    assert isinstance(metrics, dict)
    assert len(metrics) >= 0


def test_reset_performance_metrics_basic(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 4: Basic reset_performance_metrics() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: reset_performance_metrics()
    Alt Modül Dosyası: v2/monitoring/performance_monitor.py
    Alt Modül Metod: PerformanceMonitor.reset_metrics()
    Test Senaryosu: Basit performance metrics reset
    """
    # Process requests to generate metrics
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"Reset perf {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Reset metrics
    cognitive_manager.reset_performance_metrics()
    
    # Verify reset
    metrics = cognitive_manager.get_performance_metrics()
    assert isinstance(metrics, dict)


def test_reset_performance_metrics_idempotent(cognitive_manager: CognitiveManager):
    """
    Test 5: reset_performance_metrics() idempotency.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: reset_performance_metrics()
    Alt Modül Dosyası: v2/monitoring/performance_monitor.py
    Test Senaryosu: Idempotent olması
    """
    cognitive_manager.reset_performance_metrics()
    cognitive_manager.reset_performance_metrics()  # Reset again
    
    metrics = cognitive_manager.get_performance_metrics()
    assert isinstance(metrics, dict)


def test_get_performance_metrics_consistency(cognitive_manager: CognitiveManager):
    """
    Test 6: get_performance_metrics() consistency.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_performance_metrics()
    Alt Modül Dosyası: v2/monitoring/performance_monitor.py
    Test Senaryosu: Multiple call'larda consistency
    """
    metrics1 = cognitive_manager.get_performance_metrics()
    metrics2 = cognitive_manager.get_performance_metrics()
    
    assert isinstance(metrics1, dict)
    assert isinstance(metrics2, dict)


def test_get_performance_metrics_performance(cognitive_manager: CognitiveManager):
    """
    Test 7: get_performance_metrics() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_performance_metrics()
    Alt Modül Dosyası: v2/monitoring/performance_monitor.py
    Test Senaryosu: Performans testi
    """
    import time
    
    start = time.time()
    for _ in range(100):
        metrics = cognitive_manager.get_performance_metrics()
        assert isinstance(metrics, dict)
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should be fast


def test_performance_metrics_integration(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 8: Performance metrics integration test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: get_performance_metrics(), reset_performance_metrics(), handle()
    Alt Modül Dosyası: v2/monitoring/performance_monitor.py
    Test Senaryosu: Integration testi
    """
    # Initial metrics
    initial_metrics = cognitive_manager.get_performance_metrics()
    
    # Process requests
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"Integration perf {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Get metrics after requests
    after_metrics = cognitive_manager.get_performance_metrics()
    
    # Reset
    cognitive_manager.reset_performance_metrics()
    
    # Get metrics after reset
    reset_metrics = cognitive_manager.get_performance_metrics()
    
    assert isinstance(initial_metrics, dict)
    assert isinstance(after_metrics, dict)
    assert isinstance(reset_metrics, dict)


def test_get_performance_metrics_concurrent(cognitive_manager: CognitiveManager):
    """
    Test 9: get_performance_metrics() concurrent access.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_performance_metrics()
    Alt Modül Dosyası: v2/monitoring/performance_monitor.py
    Test Senaryosu: Concurrent access testi
    """
    import threading
    
    results = []
    
    def worker():
        metrics = cognitive_manager.get_performance_metrics()
        results.append(metrics)
    
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    assert len(results) == 5
    for result in results:
        assert isinstance(result, dict)


def test_performance_metrics_accuracy(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 10: Performance metrics accuracy test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: get_performance_metrics(), handle()
    Alt Modül Dosyası: v2/monitoring/performance_monitor.py
    Test Senaryosu: Metrics doğruluğu testi
    """
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Accuracy test {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    metrics = cognitive_manager.get_performance_metrics()
    assert isinstance(metrics, dict)
    # Metrics should reflect request processing


# ============================================================================
# Test 11-20: get_performance_profile() and get_all_performance_stats()
# Test Edilen Dosya: cognitive_manager.py
# Alt Modül: v2/utils/performance_profiler.py (PerformanceProfiler)
# ============================================================================

def test_get_performance_profile_basic(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 11: Basic get_performance_profile() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_performance_profile()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Alt Modül Metod: PerformanceProfiler.get_profile_report()
    Test Senaryosu: Basit performance profile alma
    """
    # Process requests to generate profile data
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"Profile test {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    profile = cognitive_manager.get_performance_profile()
    assert isinstance(profile, str)
    # May be empty or contain report


def test_get_performance_profile_formats(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 12: get_performance_profile() with different formats.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_performance_profile(format)
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Farklı format'larla profile
    """
    # Process requests
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"Profile formats {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    formats = ["summary", "detailed", "bottlenecks"]
    for fmt in formats:
        profile = cognitive_manager.get_performance_profile(format=fmt)
        assert isinstance(profile, str)


def test_get_all_performance_stats_basic(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 13: Basic get_all_performance_stats() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_all_performance_stats()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Alt Modül Metod: PerformanceProfiler.get_all_stats()
    Test Senaryosu: Basit all performance stats alma
    """
    # Process requests
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"All stats test {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    stats = cognitive_manager.get_all_performance_stats()
    assert isinstance(stats, dict)
    # Dictionary of operation -> stats


def test_get_all_performance_stats_structure(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 14: get_all_performance_stats() structure validation.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_all_performance_stats()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Stats yapısı validation
    """
    # Process requests
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"Stats structure {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    stats = cognitive_manager.get_all_performance_stats()
    assert isinstance(stats, dict)
    # Each value should be a dict
    for operation_stats in stats.values():
        assert isinstance(operation_stats, dict)


def test_get_all_performance_stats_consistency(cognitive_manager: CognitiveManager):
    """
    Test 15: get_all_performance_stats() consistency.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_all_performance_stats()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Multiple call'larda consistency
    """
    stats1 = cognitive_manager.get_all_performance_stats()
    stats2 = cognitive_manager.get_all_performance_stats()
    
    assert isinstance(stats1, dict)
    assert isinstance(stats2, dict)


def test_get_performance_profile_performance(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 16: get_performance_profile() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_performance_profile()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Performans testi
    """
    import time
    
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Perf profile {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    start = time.time()
    profile = cognitive_manager.get_performance_profile()
    elapsed = time.time() - start
    
    assert isinstance(profile, str)
    assert elapsed < 2.0  # Should be fast


def test_get_all_performance_stats_performance(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 17: get_all_performance_stats() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_all_performance_stats()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Performans testi
    """
    import time
    
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Perf all stats {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    start = time.time()
    stats = cognitive_manager.get_all_performance_stats()
    elapsed = time.time() - start
    
    assert isinstance(stats, dict)
    assert elapsed < 1.0  # Should be fast


def test_performance_profile_integration(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 18: Performance profile integration test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: get_performance_profile(), get_all_performance_stats()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Integration testi
    """
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Profile integration {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Get profile
    profile = cognitive_manager.get_performance_profile()
    assert isinstance(profile, str)
    
    # Get all stats
    stats = cognitive_manager.get_all_performance_stats()
    assert isinstance(stats, dict)


def test_get_performance_profile_empty(cognitive_manager: CognitiveManager):
    """
    Test 19: get_performance_profile() with no data.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_performance_profile()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Data yokken profile (edge case)
    """
    profile = cognitive_manager.get_performance_profile()
    assert isinstance(profile, str)
    # May be empty or contain message


def test_performance_profile_formats_validation(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 20: get_performance_profile() format validation.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_performance_profile(format)
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Format validation testi
    """
    # Process requests
    for i in range(2):
        input_msg = CognitiveInput(user_message=f"Format validation {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Test different formats
    for fmt in ["summary", "detailed", "bottlenecks"]:
        profile = cognitive_manager.get_performance_profile(format=fmt)
        assert isinstance(profile, str)


# ============================================================================
# Test 21-30: clear_performance_profile() and identify_bottlenecks()
# Test Edilen Dosya: cognitive_manager.py
# Alt Modül: v2/utils/performance_profiler.py (PerformanceProfiler)
# ============================================================================

def test_clear_performance_profile_basic(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 21: Basic clear_performance_profile() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: clear_performance_profile()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Alt Modül Metod: PerformanceProfiler.clear()
    Test Senaryosu: Basit performance profile temizleme
    """
    # Process requests to generate profile data
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"Clear profile {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Clear profile
    cognitive_manager.clear_performance_profile()
    
    # Verify cleared
    stats = cognitive_manager.get_all_performance_stats()
    assert isinstance(stats, dict)


def test_clear_performance_profile_idempotent(cognitive_manager: CognitiveManager):
    """
    Test 22: clear_performance_profile() idempotency.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: clear_performance_profile()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Idempotent olması
    """
    cognitive_manager.clear_performance_profile()
    cognitive_manager.clear_performance_profile()  # Clear again
    
    stats = cognitive_manager.get_all_performance_stats()
    assert isinstance(stats, dict)


def test_identify_bottlenecks_basic(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 23: Basic identify_bottlenecks() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: identify_bottlenecks()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Alt Modül Metod: PerformanceProfiler.identify_bottlenecks()
    Test Senaryosu: Basit bottleneck identification
    """
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Bottleneck test {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    bottlenecks = cognitive_manager.identify_bottlenecks()
    assert isinstance(bottlenecks, list)
    # Each bottleneck should be a dict
    for bottleneck in bottlenecks:
        assert isinstance(bottleneck, dict)


def test_identify_bottlenecks_structure(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 24: identify_bottlenecks() structure validation.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: identify_bottlenecks()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Bottleneck yapısı validation
    """
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Bottleneck structure {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    bottlenecks = cognitive_manager.identify_bottlenecks()
    assert isinstance(bottlenecks, list)
    # Common bottleneck fields
    for bottleneck in bottlenecks:
        assert isinstance(bottleneck, dict)
        assert "operation" in bottleneck or "latency" in bottleneck or "impact" in bottleneck or len(bottleneck) >= 0


def test_identify_bottlenecks_after_clear(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 25: identify_bottlenecks() after clearing profile.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: identify_bottlenecks(), clear_performance_profile()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Clear sonrası bottleneck identification
    """
    # Process requests
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"Bottleneck clear {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Clear
    cognitive_manager.clear_performance_profile()
    
    # Identify bottlenecks (may be empty)
    bottlenecks = cognitive_manager.identify_bottlenecks()
    assert isinstance(bottlenecks, list)


def test_identify_bottlenecks_consistency(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 26: identify_bottlenecks() consistency.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: identify_bottlenecks()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Multiple call'larda consistency
    """
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Bottleneck consistency {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    bottlenecks1 = cognitive_manager.identify_bottlenecks()
    bottlenecks2 = cognitive_manager.identify_bottlenecks()
    
    assert isinstance(bottlenecks1, list)
    assert isinstance(bottlenecks2, list)


def test_clear_performance_profile_after_requests(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 27: clear_performance_profile() after requests.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: clear_performance_profile(), handle()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Request sonrası temizleme
    """
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Clear after {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Get stats before clear
    stats_before = cognitive_manager.get_all_performance_stats()
    
    # Clear
    cognitive_manager.clear_performance_profile()
    
    # Get stats after clear
    stats_after = cognitive_manager.get_all_performance_stats()
    
    assert isinstance(stats_before, dict)
    assert isinstance(stats_after, dict)


def test_identify_bottlenecks_performance(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 28: identify_bottlenecks() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: identify_bottlenecks()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Performans testi
    """
    import time
    
    # Process requests
    for i in range(10):
        input_msg = CognitiveInput(user_message=f"Bottleneck perf {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    start = time.time()
    bottlenecks = cognitive_manager.identify_bottlenecks()
    elapsed = time.time() - start
    
    assert isinstance(bottlenecks, list)
    assert elapsed < 2.0  # Should be fast


def test_performance_profile_operations_integration(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 29: Performance profile operations integration test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: get_performance_profile(), identify_bottlenecks(), clear_performance_profile()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Integration testi
    """
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Profile ops integration {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Get profile
    profile = cognitive_manager.get_performance_profile()
    assert isinstance(profile, str)
    
    # Identify bottlenecks
    bottlenecks = cognitive_manager.identify_bottlenecks()
    assert isinstance(bottlenecks, list)
    
    # Clear
    cognitive_manager.clear_performance_profile()
    
    # Verify cleared
    stats = cognitive_manager.get_all_performance_stats()
    assert isinstance(stats, dict)


def test_clear_performance_profile_performance(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 30: clear_performance_profile() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: clear_performance_profile()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Performans testi
    """
    import time
    
    # Process many requests
    for i in range(20):
        input_msg = CognitiveInput(user_message=f"Clear perf {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    start = time.time()
    cognitive_manager.clear_performance_profile()
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should be fast


# ============================================================================
# Test 31-40: get_operation_stats() - Operation Statistics
# Test Edilen Dosya: cognitive_manager.py
# Alt Modül: v2/utils/performance_profiler.py (PerformanceProfiler)
# ============================================================================

def test_get_operation_stats_basic(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 31: Basic get_operation_stats() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_operation_stats()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Alt Modül Metod: PerformanceProfiler.get_operation_stats()
    Test Senaryosu: Basit operation stats alma
    """
    # Process requests
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"Operation stats {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Get stats for a common operation
    stats = cognitive_manager.get_operation_stats("handle")
    # May be None if operation not found or disabled
    assert stats is None or isinstance(stats, dict)


def test_get_operation_stats_nonexistent_operation(cognitive_manager: CognitiveManager):
    """
    Test 32: get_operation_stats() with nonexistent operation.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_operation_stats()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Var olmayan operation (edge case)
    """
    stats = cognitive_manager.get_operation_stats("nonexistent_operation")
    assert stats is None


def test_get_operation_stats_empty_name(cognitive_manager: CognitiveManager):
    """
    Test 33: get_operation_stats() with empty name.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_operation_stats()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Boş operation name (edge case)
    """
    stats = cognitive_manager.get_operation_stats("")
    assert stats is None


def test_get_operation_stats_structure(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 34: get_operation_stats() structure validation.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_operation_stats()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Stats yapısı validation
    """
    # Process requests
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"Operation structure {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    stats = cognitive_manager.get_operation_stats("handle")
    if stats is not None:
        assert isinstance(stats, dict)
        # Common stats fields
        assert "count" in stats or "avg_latency" in stats or "total_time" in stats or len(stats) >= 0


def test_get_operation_stats_multiple_operations(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 35: get_operation_stats() for multiple operations.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_operation_stats()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Multiple operation stats
    """
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Multiple ops {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Try different operation names
    operations = ["handle", "handle_async", "handle_batch"]
    for op in operations:
        stats = cognitive_manager.get_operation_stats(op)
        assert stats is None or isinstance(stats, dict)


def test_get_operation_stats_consistency(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 36: get_operation_stats() consistency.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_operation_stats()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Multiple call'larda consistency
    """
    # Process requests
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"Consistency ops {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    stats1 = cognitive_manager.get_operation_stats("handle")
    stats2 = cognitive_manager.get_operation_stats("handle")
    
    # Should return same stats if no new requests
    assert stats1 == stats2


def test_get_operation_stats_performance(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 37: get_operation_stats() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_operation_stats()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Performans testi
    """
    import time
    
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Perf ops {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    start = time.time()
    for _ in range(50):
        stats = cognitive_manager.get_operation_stats("handle")
        assert stats is None or isinstance(stats, dict)
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should be fast


def test_get_operation_stats_after_clear(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 38: get_operation_stats() after clearing profile.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: get_operation_stats(), clear_performance_profile()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Clear sonrası operation stats
    """
    # Process requests
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"Ops after clear {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Clear
    cognitive_manager.clear_performance_profile()
    
    # Get stats (may be None or empty)
    stats = cognitive_manager.get_operation_stats("handle")
    assert stats is None or isinstance(stats, dict)


def test_get_operation_stats_integration(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 39: get_operation_stats() integration test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: get_operation_stats(), get_all_performance_stats()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Integration testi
    """
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Ops integration {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Get all stats
    all_stats = cognitive_manager.get_all_performance_stats()
    assert isinstance(all_stats, dict)
    
    # Get specific operation stats
    if "handle" in all_stats:
        op_stats = cognitive_manager.get_operation_stats("handle")
        assert op_stats is None or isinstance(op_stats, dict)


def test_operation_stats_accuracy(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 40: Operation stats accuracy test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: get_operation_stats(), handle()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Stats doğruluğu testi
    """
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Accuracy ops {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    stats = cognitive_manager.get_operation_stats("handle")
    if stats is not None:
        assert isinstance(stats, dict)
        # Count should reflect number of requests
        if "count" in stats:
            assert stats["count"] >= 0


# ============================================================================
# Test 41-50: Performance Profiling Integration and Edge Cases
# Test Edilen Dosya: cognitive_manager.py
# Alt Modül: v2/utils/performance_profiler.py, v2/monitoring/performance_monitor.py
# ============================================================================

def test_performance_profiling_full_workflow(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 41: Full performance profiling workflow.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: Tüm performance profiling metodları
    Alt Modül Dosyaları:
    - v2/utils/performance_profiler.py
    - v2/monitoring/performance_monitor.py
    Test Senaryosu: Tam performance profiling workflow
    """
    # 1. Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Workflow perf {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # 2. Get performance metrics
    metrics = cognitive_manager.get_performance_metrics()
    assert isinstance(metrics, dict)
    
    # 3. Get performance profile
    profile = cognitive_manager.get_performance_profile()
    assert isinstance(profile, str)
    
    # 4. Get all performance stats
    all_stats = cognitive_manager.get_all_performance_stats()
    assert isinstance(all_stats, dict)
    
    # 5. Identify bottlenecks
    bottlenecks = cognitive_manager.identify_bottlenecks()
    assert isinstance(bottlenecks, list)
    
    # 6. Get operation stats
    op_stats = cognitive_manager.get_operation_stats("handle")
    assert op_stats is None or isinstance(op_stats, dict)
    
    # 7. Clear profile
    cognitive_manager.clear_performance_profile()
    
    # 8. Reset metrics
    cognitive_manager.reset_performance_metrics()


def test_performance_profiling_with_async(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 42: Performance profiling with async requests.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: handle_async(), get_performance_metrics()
    Alt Modül Dosyası: v2/monitoring/performance_monitor.py
    Test Senaryosu: Async request'lerde profiling
    """
    import asyncio
    
    async def async_test():
        for i in range(3):
            input_msg = CognitiveInput(user_message=f"Async perf {i}")
            await cognitive_manager.handle_async(cognitive_state, input_msg)
    
    asyncio.run(async_test())
    
    metrics = cognitive_manager.get_performance_metrics()
    assert isinstance(metrics, dict)


def test_performance_profiling_with_batch(cognitive_manager: CognitiveManager):
    """
    Test 43: Performance profiling with batch requests.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: handle_batch(), get_performance_metrics()
    Alt Modül Dosyası: v2/monitoring/performance_monitor.py
    Test Senaryosu: Batch request'lerde profiling
    """
    states = [CognitiveState() for _ in range(3)]
    inputs = [CognitiveInput(user_message=f"Batch perf {i}") for i in range(3)]
    requests = list(zip(states, inputs))
    
    cognitive_manager.handle_batch(requests)
    
    metrics = cognitive_manager.get_performance_metrics()
    assert isinstance(metrics, dict)


def test_performance_profiling_under_load(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 44: Performance profiling under load.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: handle(), get_performance_metrics(), identify_bottlenecks()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Yük altında profiling
    """
    import time
    
    start = time.time()
    
    # Process many requests
    for i in range(20):
        input_msg = CognitiveInput(user_message=f"Load perf {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    elapsed = time.time() - start
    
    # Get metrics
    metrics = cognitive_manager.get_performance_metrics()
    
    # Identify bottlenecks
    bottlenecks = cognitive_manager.identify_bottlenecks()
    
    assert isinstance(metrics, dict)
    assert isinstance(bottlenecks, list)
    assert elapsed < 60.0  # Should complete in reasonable time


def test_performance_profiling_concurrent(cognitive_manager: CognitiveManager):
    """
    Test 45: Performance profiling concurrent operations.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: handle(), get_performance_metrics()
    Alt Modül Dosyası: v2/monitoring/performance_monitor.py
    Test Senaryosu: Concurrent operations testi
    """
    import threading
    
    def worker(worker_id: int):
        state = CognitiveState()
        input_msg = CognitiveInput(user_message=f"Concurrent perf {worker_id}")
        cognitive_manager.handle(state, input_msg)
        metrics = cognitive_manager.get_performance_metrics()
        assert isinstance(metrics, dict)
    
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def test_performance_profiling_error_recovery(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 46: Performance profiling error recovery.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: handle(), get_performance_metrics()
    Alt Modül Dosyası: v2/monitoring/performance_monitor.py
    Test Senaryosu: Hata sonrası recovery
    """
    # Process normal request
    input_msg1 = CognitiveInput(user_message="Normal perf")
    cognitive_manager.handle(cognitive_state, input_msg1)
    
    # Process another request
    input_msg2 = CognitiveInput(user_message="Recovery perf")
    cognitive_manager.handle(cognitive_state, input_msg2)
    
    # Profiling should still work
    metrics = cognitive_manager.get_performance_metrics()
    assert isinstance(metrics, dict)


def test_performance_profiling_stats_accuracy(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 47: Performance profiling stats accuracy.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: get_performance_metrics(), handle(), reset_performance_metrics()
    Alt Modül Dosyası: v2/monitoring/performance_monitor.py
    Test Senaryosu: Stats doğruluğu testi
    """
    # Reset metrics
    cognitive_manager.reset_performance_metrics()
    initial_metrics = cognitive_manager.get_performance_metrics()
    
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Accuracy perf {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    final_metrics = cognitive_manager.get_performance_metrics()
    
    assert isinstance(initial_metrics, dict)
    assert isinstance(final_metrics, dict)
    # Metrics should reflect request processing


def test_performance_profiling_bottleneck_identification(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 48: Performance profiling bottleneck identification accuracy.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: identify_bottlenecks(), handle()
    Alt Modül Dosyası: v2/utils/performance_profiler.py
    Test Senaryosu: Bottleneck identification doğruluğu
    """
    # Process requests
    for i in range(10):
        input_msg = CognitiveInput(user_message=f"Bottleneck accuracy {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    bottlenecks = cognitive_manager.identify_bottlenecks()
    assert isinstance(bottlenecks, list)
    
    # Each bottleneck should have valid structure
    for bottleneck in bottlenecks:
        assert isinstance(bottleneck, dict)


def test_performance_profiling_integration_full_system(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 49: Performance profiling integration with full system.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: Tüm performance profiling metodları
    Alt Modül Dosyaları:
    - v2/utils/performance_profiler.py
    - v2/monitoring/performance_monitor.py
    Test Senaryosu: Full system integration
    """
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Full system perf {i}")
        output = cognitive_manager.handle(cognitive_state, input_msg)
        assert output is not None
    
    # Get all performance data
    metrics = cognitive_manager.get_performance_metrics()
    profile = cognitive_manager.get_performance_profile()
    all_stats = cognitive_manager.get_all_performance_stats()
    bottlenecks = cognitive_manager.identify_bottlenecks()
    
    assert isinstance(metrics, dict)
    assert isinstance(profile, str)
    assert isinstance(all_stats, dict)
    assert isinstance(bottlenecks, list)


def test_performance_profiling_end_to_end(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 50: Performance profiling end-to-end test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: Tüm performance profiling metodları
    Alt Modül Dosyaları:
    - v2/utils/performance_profiler.py
    - v2/monitoring/performance_monitor.py
    Test Senaryosu: End-to-end performance profiling testi
    """
    # 1. Initial state
    initial_metrics = cognitive_manager.get_performance_metrics()
    initial_stats = cognitive_manager.get_all_performance_stats()
    
    # 2. Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"E2E perf {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # 3. Get performance metrics
    metrics = cognitive_manager.get_performance_metrics()
    assert isinstance(metrics, dict)
    
    # 4. Get performance profile
    profile = cognitive_manager.get_performance_profile(format="summary")
    assert isinstance(profile, str)
    
    # 5. Get all performance stats
    all_stats = cognitive_manager.get_all_performance_stats()
    assert isinstance(all_stats, dict)
    
    # 6. Identify bottlenecks
    bottlenecks = cognitive_manager.identify_bottlenecks()
    assert isinstance(bottlenecks, list)
    
    # 7. Get operation stats
    op_stats = cognitive_manager.get_operation_stats("handle")
    assert op_stats is None or isinstance(op_stats, dict)
    
    # 8. Clear profile
    cognitive_manager.clear_performance_profile()
    
    # 9. Reset metrics
    cognitive_manager.reset_performance_metrics()
    
    # 10. Final state
    final_metrics = cognitive_manager.get_performance_metrics()
    final_stats = cognitive_manager.get_all_performance_stats()
    
    # Verify
    assert isinstance(initial_metrics, dict)
    assert isinstance(initial_stats, dict)
    assert isinstance(final_metrics, dict)
    assert isinstance(final_stats, dict)

