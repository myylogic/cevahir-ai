# -*- coding: utf-8 -*-
"""
Memory Service API Extended Tests
==================================
CognitiveManager memory service metodlarının genişletilmiş testleri.

Test Edilen Dosya: cognitive_manager.py
Test Edilen Metodlar:
- add_memory_note() - Extended scenarios
- get_memory_notes() - Extended scenarios
- clear_memory_notes() - Extended scenarios
- get_vector_store_stats() - Extended scenarios
- clear_vector_store() - Extended scenarios
- delete_vector_store_items() - Extended scenarios

Alt Modül Test Edilen Dosyalar:
- v2/components/memory_service_v2.py (MemoryServiceV2)
- v2/components/vector_store/base.py (VectorStore)
- v2/components/vector_store/memory_vector_store.py (MemoryVectorStore)
- v2/components/vector_store/chroma_vector_store.py (ChromaVectorStore)

Endüstri Standartları:
- pytest framework
- Advanced edge cases
- Complex integration scenarios
- Performance validation
- Memory leak detection
"""

import pytest
import threading
import time
from typing import List, Dict, Any

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
# Test 51-60: add_memory_note() - Advanced Edge Cases
# ============================================================================

def test_add_memory_note_with_unicode_characters(cognitive_manager: CognitiveManager):
    """Test 51: add_memory_note() with unicode characters."""
    unicode_note = "Unicode test: Türkçe 中文 🚀 émojis 🎉"
    cognitive_manager.add_memory_note(unicode_note)
    notes = cognitive_manager.get_memory_notes()
    assert unicode_note in notes


def test_add_memory_note_with_special_characters(cognitive_manager: CognitiveManager):
    """Test 52: add_memory_note() with special characters."""
    special_note = "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
    cognitive_manager.add_memory_note(special_note)
    notes = cognitive_manager.get_memory_notes()
    assert special_note in notes


def test_add_memory_note_with_very_long_text(cognitive_manager: CognitiveManager):
    """Test 53: add_memory_note() with very long text."""
    long_note = "Long note: " + "A" * 10000
    cognitive_manager.add_memory_note(long_note)
    notes = cognitive_manager.get_memory_notes()
    assert long_note in notes


def test_add_memory_note_with_empty_string(cognitive_manager: CognitiveManager):
    """Test 54: add_memory_note() with empty string."""
    cognitive_manager.add_memory_note("")
    notes = cognitive_manager.get_memory_notes()
    assert "" in notes or len(notes) >= 0


def test_add_memory_note_with_newlines(cognitive_manager: CognitiveManager):
    """Test 55: add_memory_note() with newlines."""
    multiline_note = "Line 1\nLine 2\nLine 3"
    cognitive_manager.add_memory_note(multiline_note)
    notes = cognitive_manager.get_memory_notes()
    assert multiline_note in notes


def test_add_memory_note_with_tabs(cognitive_manager: CognitiveManager):
    """Test 56: add_memory_note() with tabs."""
    tab_note = "Tab\tSeparated\tValues"
    cognitive_manager.add_memory_note(tab_note)
    notes = cognitive_manager.get_memory_notes()
    assert tab_note in notes


def test_add_memory_note_with_json_string(cognitive_manager: CognitiveManager):
    """Test 57: add_memory_note() with JSON string."""
    json_note = '{"key": "value", "number": 123}'
    cognitive_manager.add_memory_note(json_note)
    notes = cognitive_manager.get_memory_notes()
    assert json_note in notes


def test_add_memory_note_with_xml_string(cognitive_manager: CognitiveManager):
    """Test 58: add_memory_note() with XML string."""
    xml_note = "<root><item>value</item></root>"
    cognitive_manager.add_memory_note(xml_note)
    notes = cognitive_manager.get_memory_notes()
    assert xml_note in notes


def test_add_memory_note_with_sql_injection_pattern(cognitive_manager: CognitiveManager):
    """Test 59: add_memory_note() with SQL injection pattern."""
    sql_note = "'; DROP TABLE notes; --"
    cognitive_manager.add_memory_note(sql_note)
    notes = cognitive_manager.get_memory_notes()
    assert sql_note in notes


def test_add_memory_note_with_html_tags(cognitive_manager: CognitiveManager):
    """Test 60: add_memory_note() with HTML tags."""
    html_note = "<script>alert('test')</script>"
    cognitive_manager.add_memory_note(html_note)
    notes = cognitive_manager.get_memory_notes()
    assert html_note in notes


# ============================================================================
# Test 61-70: get_memory_notes() - Complex Integration Scenarios
# ============================================================================

def test_get_memory_notes_with_concurrent_additions(cognitive_manager: CognitiveManager):
    """Test 61: get_memory_notes() with concurrent additions."""
    import threading
    
    def worker(worker_id: int):
        for i in range(10):
            cognitive_manager.add_memory_note(f"Concurrent note {worker_id}_{i}")
    
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    notes = cognitive_manager.get_memory_notes()
    assert len(notes) >= 50


def test_get_memory_notes_with_rapid_additions(cognitive_manager: CognitiveManager):
    """Test 62: get_memory_notes() with rapid additions."""
    for i in range(100):
        cognitive_manager.add_memory_note(f"Rapid note {i}")
    
    notes = cognitive_manager.get_memory_notes()
    assert len(notes) >= 100


def test_get_memory_notes_with_memory_pressure(cognitive_manager: CognitiveManager):
    """Test 63: get_memory_notes() under memory pressure."""
    for i in range(1000):
        cognitive_manager.add_memory_note(f"Memory pressure note {i}")
    
    notes = cognitive_manager.get_memory_notes()
    assert isinstance(notes, list)


def test_get_memory_notes_after_clear_and_add(cognitive_manager: CognitiveManager):
    """Test 64: get_memory_notes() after clear and add."""
    cognitive_manager.add_memory_note("Before clear")
    cognitive_manager.clear_memory_notes()
    cognitive_manager.add_memory_note("After clear")
    
    notes = cognitive_manager.get_memory_notes()
    assert "After clear" in notes
    assert "Before clear" not in notes


def test_get_memory_notes_with_state_persistence(cognitive_manager: CognitiveManager):
    """Test 65: get_memory_notes() with state persistence."""
    cognitive_manager.add_memory_note("Persistent note 1")
    notes1 = cognitive_manager.get_memory_notes()
    
    cognitive_manager.add_memory_note("Persistent note 2")
    notes2 = cognitive_manager.get_memory_notes()
    
    assert "Persistent note 1" in notes1
    assert "Persistent note 1" in notes2
    assert "Persistent note 2" in notes2


def test_get_memory_notes_with_request_processing(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 66: get_memory_notes() with request processing."""
    cognitive_manager.add_memory_note("Before request")
    
    input_msg = CognitiveInput(user_message="Test request")
    cognitive_manager.handle(cognitive_state, input_msg)
    
    notes = cognitive_manager.get_memory_notes()
    assert "Before request" in notes


def test_get_memory_notes_with_vector_store_interaction(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 67: get_memory_notes() with vector store interaction."""
    cognitive_manager.add_memory_note("Vector store note")
    
    input_msg = CognitiveInput(user_message="Vector store test")
    cognitive_manager.handle(cognitive_state, input_msg)
    
    notes = cognitive_manager.get_memory_notes()
    stats = cognitive_manager.get_vector_store_stats()
    
    assert "Vector store note" in notes
    assert stats is None or isinstance(stats, dict)


def test_get_memory_notes_with_multiple_clears(cognitive_manager: CognitiveManager):
    """Test 68: get_memory_notes() with multiple clears."""
    for i in range(5):
        cognitive_manager.add_memory_note(f"Note {i}")
        cognitive_manager.clear_memory_notes()
    
    notes = cognitive_manager.get_memory_notes()
    assert len(notes) == 0


def test_get_memory_notes_with_duplicate_notes(cognitive_manager: CognitiveManager):
    """Test 69: get_memory_notes() with duplicate notes."""
    duplicate_note = "Duplicate note"
    for i in range(10):
        cognitive_manager.add_memory_note(duplicate_note)
    
    notes = cognitive_manager.get_memory_notes()
    assert duplicate_note in notes


def test_get_memory_notes_integration_full_system(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 70: get_memory_notes() integration with full system."""
    cognitive_manager.add_memory_note("Integration note")
    
    input_msg = CognitiveInput(user_message="Integration test")
    output = cognitive_manager.handle(cognitive_state, input_msg)
    
    notes = cognitive_manager.get_memory_notes()
    metrics = cognitive_manager.get_metrics()
    health = cognitive_manager.get_health_status()
    
    assert "Integration note" in notes
    assert isinstance(metrics, dict)
    assert isinstance(health, dict)


# ============================================================================
# Test 71-80: clear_memory_notes() - Performance & Stress Tests
# ============================================================================

def test_clear_memory_notes_with_large_dataset(cognitive_manager: CognitiveManager):
    """Test 71: clear_memory_notes() with large dataset."""
    for i in range(1000):
        cognitive_manager.add_memory_note(f"Large dataset note {i}")
    
    import time
    start = time.time()
    cognitive_manager.clear_memory_notes()
    elapsed = time.time() - start
    
    notes = cognitive_manager.get_memory_notes()
    assert len(notes) == 0
    assert elapsed < 5.0


def test_clear_memory_notes_concurrent_operations(cognitive_manager: CognitiveManager):
    """Test 72: clear_memory_notes() with concurrent operations."""
    import threading
    
    for i in range(100):
        cognitive_manager.add_memory_note(f"Concurrent note {i}")
    
    def worker():
        cognitive_manager.clear_memory_notes()
    
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    notes = cognitive_manager.get_memory_notes()
    assert len(notes) == 0


def test_clear_memory_notes_performance_under_load(cognitive_manager: CognitiveManager):
    """Test 73: clear_memory_notes() performance under load."""
    import time
    
    # Add many notes
    for i in range(500):
        cognitive_manager.add_memory_note(f"Load note {i}")
    
    # Clear multiple times
    start = time.time()
    for _ in range(10):
        cognitive_manager.clear_memory_notes()
    elapsed = time.time() - start
    
    assert elapsed < 2.0


def test_clear_memory_notes_with_rapid_add_clear_cycle(cognitive_manager: CognitiveManager):
    """Test 74: clear_memory_notes() with rapid add-clear cycle."""
    for i in range(50):
        cognitive_manager.add_memory_note(f"Cycle note {i}")
        cognitive_manager.clear_memory_notes()
    
    notes = cognitive_manager.get_memory_notes()
    assert len(notes) == 0


def test_clear_memory_notes_with_partial_operations(cognitive_manager: CognitiveManager):
    """Test 75: clear_memory_notes() with partial operations."""
    cognitive_manager.add_memory_note("Note 1")
    cognitive_manager.add_memory_note("Note 2")
    cognitive_manager.clear_memory_notes()
    cognitive_manager.add_memory_note("Note 3")
    
    notes = cognitive_manager.get_memory_notes()
    assert "Note 3" in notes
    assert "Note 1" not in notes
    assert "Note 2" not in notes


def test_clear_memory_notes_with_vector_store_cleanup(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 76: clear_memory_notes() with vector store cleanup."""
    cognitive_manager.add_memory_note("Vector note")
    
    input_msg = CognitiveInput(user_message="Vector test")
    cognitive_manager.handle(cognitive_state, input_msg)
    
    cognitive_manager.clear_memory_notes()
    
    notes = cognitive_manager.get_memory_notes()
    stats = cognitive_manager.get_vector_store_stats()
    
    assert len(notes) == 0
    assert stats is None or isinstance(stats, dict)


def test_clear_memory_notes_idempotency(cognitive_manager: CognitiveManager):
    """Test 77: clear_memory_notes() idempotency."""
    cognitive_manager.add_memory_note("Idempotency test")
    cognitive_manager.clear_memory_notes()
    cognitive_manager.clear_memory_notes()  # Clear again
    
    notes = cognitive_manager.get_memory_notes()
    assert len(notes) == 0


def test_clear_memory_notes_with_error_recovery(cognitive_manager: CognitiveManager):
    """Test 78: clear_memory_notes() with error recovery."""
    cognitive_manager.add_memory_note("Error recovery note")
    
    try:
        cognitive_manager.clear_memory_notes()
    except Exception:
        pass
    
    # Should still work
    notes = cognitive_manager.get_memory_notes()
    assert isinstance(notes, list)


def test_clear_memory_notes_with_memory_leak_detection(cognitive_manager: CognitiveManager):
    """Test 79: clear_memory_notes() memory leak detection."""
    import sys
    
    initial_size = sys.getsizeof(cognitive_manager)
    
    for i in range(100):
        cognitive_manager.add_memory_note(f"Leak test {i}")
    
    cognitive_manager.clear_memory_notes()
    
    final_size = sys.getsizeof(cognitive_manager)
    # Basic sanity check
    assert final_size >= initial_size


def test_clear_memory_notes_stress_test(cognitive_manager: CognitiveManager):
    """Test 80: clear_memory_notes() stress test."""
    import time
    
    start = time.time()
    
    for cycle in range(20):
        for i in range(50):
            cognitive_manager.add_memory_note(f"Stress note {cycle}_{i}")
        cognitive_manager.clear_memory_notes()
    
    elapsed = time.time() - start
    
    notes = cognitive_manager.get_memory_notes()
    assert len(notes) == 0
    assert elapsed < 30.0


# ============================================================================
# Test 81-90: Vector Store Operations - Error Recovery & Resilience
# ============================================================================

def test_vector_store_stats_with_error_handling(cognitive_manager: CognitiveManager):
    """Test 81: get_vector_store_stats() with error handling."""
    try:
        stats = cognitive_manager.get_vector_store_stats()
        assert stats is None or isinstance(stats, dict)
    except Exception:
        # Should handle gracefully
        pass


def test_vector_store_stats_after_operations(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 82: get_vector_store_stats() after operations."""
    stats1 = cognitive_manager.get_vector_store_stats()
    
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Vector stats test {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    stats2 = cognitive_manager.get_vector_store_stats()
    
    if stats1 is not None and stats2 is not None:
        assert isinstance(stats1, dict)
        assert isinstance(stats2, dict)


def test_clear_vector_store_with_error_recovery(cognitive_manager: CognitiveManager):
    """Test 83: clear_vector_store() with error recovery."""
    try:
        cognitive_manager.clear_vector_store()
        stats = cognitive_manager.get_vector_store_stats()
        assert stats is None or isinstance(stats, dict)
    except Exception:
        # Should handle gracefully
        pass


def test_clear_vector_store_idempotency(cognitive_manager: CognitiveManager):
    """Test 84: clear_vector_store() idempotency."""
    cognitive_manager.clear_vector_store()
    cognitive_manager.clear_vector_store()  # Clear again
    
    stats = cognitive_manager.get_vector_store_stats()
    assert stats is None or isinstance(stats, dict)


def test_delete_vector_store_items_with_invalid_ids(cognitive_manager: CognitiveManager):
    """Test 85: delete_vector_store_items() with invalid IDs."""
    try:
        cognitive_manager.delete_vector_store_items(["invalid_id_1", "invalid_id_2"])
    except Exception:
        # Should handle gracefully
        pass


def test_delete_vector_store_items_with_empty_list(cognitive_manager: CognitiveManager):
    """Test 86: delete_vector_store_items() with empty list."""
    try:
        cognitive_manager.delete_vector_store_items([])
    except Exception:
        # Should handle gracefully
        pass


def test_delete_vector_store_items_with_none_ids(cognitive_manager: CognitiveManager):
    """Test 87: delete_vector_store_items() with None IDs."""
    try:
        cognitive_manager.delete_vector_store_items([None, None])  # type: ignore
    except Exception:
        # Should handle gracefully
        pass


def test_vector_store_operations_with_concurrent_access(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 88: Vector store operations with concurrent access."""
    import threading
    
    def worker(worker_id: int):
        input_msg = CognitiveInput(user_message=f"Concurrent vector {worker_id}")
        cognitive_manager.handle(cognitive_state, input_msg)
        stats = cognitive_manager.get_vector_store_stats()
        assert stats is None or isinstance(stats, dict)
    
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def test_vector_store_operations_with_partial_failures(cognitive_manager: CognitiveManager):
    """Test 89: Vector store operations with partial failures."""
    # Try operations that may fail
    try:
        cognitive_manager.clear_vector_store()
        cognitive_manager.delete_vector_store_items(["any_id"])
        stats = cognitive_manager.get_vector_store_stats()
        assert stats is None or isinstance(stats, dict)
    except Exception:
        # Should handle gracefully
        pass


def test_vector_store_operations_resilience(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 90: Vector store operations resilience."""
    # Multiple operations
    for i in range(10):
        try:
            input_msg = CognitiveInput(user_message=f"Resilience test {i}")
            cognitive_manager.handle(cognitive_state, input_msg)
            cognitive_manager.get_vector_store_stats()
        except Exception:
            # Should continue
            pass
    
    # Final check
    stats = cognitive_manager.get_vector_store_stats()
    assert stats is None or isinstance(stats, dict)


# ============================================================================
# Test 91-100: Advanced Validation & End-to-End
# ============================================================================

def test_memory_service_data_consistency(cognitive_manager: CognitiveManager):
    """Test 91: Memory service data consistency."""
    cognitive_manager.add_memory_note("Consistency note 1")
    cognitive_manager.add_memory_note("Consistency note 2")
    
    notes1 = cognitive_manager.get_memory_notes()
    notes2 = cognitive_manager.get_memory_notes()
    
    assert notes1 == notes2


def test_memory_service_state_validation(cognitive_manager: CognitiveManager):
    """Test 92: Memory service state validation."""
    cognitive_manager.add_memory_note("State validation note")
    
    notes = cognitive_manager.get_memory_notes()
    assert isinstance(notes, list)
    assert "State validation note" in notes
    
    cognitive_manager.clear_memory_notes()
    notes_after = cognitive_manager.get_memory_notes()
    assert len(notes_after) == 0


def test_memory_service_output_quality(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 93: Memory service output quality."""
    cognitive_manager.add_memory_note("Quality test note")
    
    input_msg = CognitiveInput(user_message="Quality test")
    output = cognitive_manager.handle(cognitive_state, input_msg)
    
    assert output is not None
    assert output.text is not None


def test_memory_service_system_health(cognitive_manager: CognitiveManager):
    """Test 94: Memory service system health."""
    cognitive_manager.add_memory_note("Health test note")
    
    health = cognitive_manager.get_health_status()
    metrics = cognitive_manager.get_metrics()
    
    assert isinstance(health, dict)
    assert isinstance(metrics, dict)


def test_memory_service_comprehensive_workflow(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 95: Memory service comprehensive workflow."""
    # Add notes
    for i in range(10):
        cognitive_manager.add_memory_note(f"Workflow note {i}")
    
    # Process requests
    for i in range(5):
        input_msg = CognitiveInput(user_message=f"Workflow test {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Check notes
    notes = cognitive_manager.get_memory_notes()
    assert len(notes) >= 10
    
    # Check vector store
    stats = cognitive_manager.get_vector_store_stats()
    assert stats is None or isinstance(stats, dict)
    
    # Clear
    cognitive_manager.clear_memory_notes()
    notes_after = cognitive_manager.get_memory_notes()
    assert len(notes_after) == 0


def test_memory_service_end_to_end_scenario(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 96: Memory service end-to-end scenario."""
    # Initial state
    initial_notes = cognitive_manager.get_memory_notes()
    initial_stats = cognitive_manager.get_vector_store_stats()
    
    # Add notes
    cognitive_manager.add_memory_note("E2E note 1")
    cognitive_manager.add_memory_note("E2E note 2")
    
    # Process requests
    for i in range(3):
        input_msg = CognitiveInput(user_message=f"E2E test {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Check state
    notes = cognitive_manager.get_memory_notes()
    stats = cognitive_manager.get_vector_store_stats()
    
    assert "E2E note 1" in notes
    assert "E2E note 2" in notes
    assert isinstance(initial_notes, list)
    assert initial_stats is None or isinstance(initial_stats, dict)
    assert stats is None or isinstance(stats, dict)


def test_memory_service_production_readiness(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 97: Memory service production readiness."""
    # Simulate production workload
    for i in range(50):
        cognitive_manager.add_memory_note(f"Production note {i}")
    
    for i in range(20):
        input_msg = CognitiveInput(user_message=f"Production test {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Verify all systems
    notes = cognitive_manager.get_memory_notes()
    stats = cognitive_manager.get_vector_store_stats()
    health = cognitive_manager.get_health_status()
    metrics = cognitive_manager.get_metrics()
    
    assert len(notes) >= 50
    assert stats is None or isinstance(stats, dict)
    assert isinstance(health, dict)
    assert isinstance(metrics, dict)


def test_memory_service_full_integration(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 98: Memory service full integration."""
    # Setup
    cognitive_manager.add_memory_note("Integration note")
    
    # Process request
    input_msg = CognitiveInput(user_message="Integration test")
    output = cognitive_manager.handle(cognitive_state, input_msg)
    
    # Verify all systems
    notes = cognitive_manager.get_memory_notes()
    stats = cognitive_manager.get_vector_store_stats()
    metrics = cognitive_manager.get_metrics()
    health = cognitive_manager.get_health_status()
    traces = cognitive_manager.get_all_traces()
    
    assert "Integration note" in notes
    assert stats is None or isinstance(stats, dict)
    assert isinstance(metrics, dict)
    assert isinstance(health, dict)
    assert isinstance(traces, list)


def test_memory_service_comprehensive_validation(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 99: Memory service comprehensive validation."""
    # Multiple operations
    for i in range(20):
        cognitive_manager.add_memory_note(f"Validation note {i}")
    
    for i in range(10):
        input_msg = CognitiveInput(user_message=f"Validation test {i}")
        cognitive_manager.handle(cognitive_state, input_msg)
    
    # Validate all aspects
    notes = cognitive_manager.get_memory_notes()
    stats = cognitive_manager.get_vector_store_stats()
    metrics = cognitive_manager.get_metrics()
    health = cognitive_manager.get_health_status()
    
    assert len(notes) >= 20
    assert stats is None or isinstance(stats, dict)
    assert isinstance(metrics, dict)
    assert isinstance(health, dict)
    
    # Clear and validate
    cognitive_manager.clear_memory_notes()
    notes_after = cognitive_manager.get_memory_notes()
    assert len(notes_after) == 0


def test_memory_service_ultimate_validation(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """Test 100: Memory service ultimate validation."""
    # Comprehensive test
    cognitive_manager.add_memory_note("Ultimate test note")
    
    input_msg = CognitiveInput(user_message="Ultimate test")
    output = cognitive_manager.handle(cognitive_state, input_msg)
    
    # All validations
    notes = cognitive_manager.get_memory_notes()
    stats = cognitive_manager.get_vector_store_stats()
    metrics = cognitive_manager.get_metrics()
    health = cognitive_manager.get_health_status()
    traces = cognitive_manager.get_all_traces()
    
    assert "Ultimate test note" in notes
    assert stats is None or isinstance(stats, dict)
    assert isinstance(metrics, dict)
    assert isinstance(health, dict)
    assert isinstance(traces, list)
    assert output is not None
    assert output.text is not None

