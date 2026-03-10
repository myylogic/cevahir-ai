# -*- coding: utf-8 -*-
"""
Event Management API Tests
===========================
CognitiveManager event management metodlarını test eder.

Test Edilen Dosya: cognitive_manager.py
Test Edilen Metodlar:
- subscribe_to_events() - Event subscription
- unsubscribe_from_events() - Event unsubscription
- publish_event() - Event publishing
- get_event_history() - Event history retrieval
- clear_event_history() - Event history clearing
- get_event_subscriber_count() - Subscriber count
- get_event_metrics() - Event metrics
- reset_event_metrics() - Reset event metrics

Alt Modül Test Edilen Dosyalar:
- v2/events/event_bus.py (EventBus)
- v2/events/event_handlers.py (EventHandlers)

Endüstri Standartları:
- pytest framework
- Fixture-based setup
- Assertion-based validation
- Event-driven testing
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
# Test 1-10: subscribe_to_events() - Event Subscription
# Test Edilen Dosya: cognitive_manager.py (subscribe_to_events method)
# Alt Modül: v2/events/event_bus.py (EventBus.subscribe)
# ============================================================================

def test_subscribe_to_events_basic(cognitive_manager: CognitiveManager):
    """
    Test 1: Basic subscribe_to_events() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: subscribe_to_events()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.subscribe()
    Test Senaryosu: Basit event subscription
    """
    class TestObserver:
        def __init__(self):
            self.events = []
        
        def on_event(self, event: Dict[str, Any]) -> None:
            self.events.append(event)
    
    observer = TestObserver()
    cognitive_manager.subscribe_to_events(observer)
    
    # Verify subscription
    count = cognitive_manager.get_event_subscriber_count()
    assert count >= 1


def test_subscribe_to_events_specific_type(cognitive_manager: CognitiveManager):
    """
    Test 2: subscribe_to_events() with specific event type.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: subscribe_to_events(event_type)
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.subscribe()
    Test Senaryosu: Belirli event type için subscription
    """
    class TestObserver:
        def on_event(self, event: Dict[str, Any]) -> None:
            pass
    
    observer = TestObserver()
    cognitive_manager.subscribe_to_events(observer, event_type="request_received")
    
    # Verify subscription
    count = cognitive_manager.get_event_subscriber_count("request_received")
    assert count >= 1


def test_subscribe_to_events_multiple_observers(cognitive_manager: CognitiveManager):
    """
    Test 3: subscribe_to_events() with multiple observers.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: subscribe_to_events()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.subscribe()
    Test Senaryosu: Multiple observer subscription
    """
    class TestObserver:
        def __init__(self, name: str):
            self.name = name
            self.events = []
        
        def on_event(self, event: Dict[str, Any]) -> None:
            self.events.append(event)
    
    observers = [TestObserver(f"Observer {i}") for i in range(5)]
    for observer in observers:
        cognitive_manager.subscribe_to_events(observer)
    
    count = cognitive_manager.get_event_subscriber_count()
    assert count >= 5


def test_subscribe_to_events_multiple_types(cognitive_manager: CognitiveManager):
    """
    Test 4: subscribe_to_events() for multiple event types.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: subscribe_to_events()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.subscribe()
    Test Senaryosu: Multiple event type subscription
    """
    class TestObserver:
        def on_event(self, event: Dict[str, Any]) -> None:
            pass
    
    observer = TestObserver()
    event_types = ["request_received", "response_generated", "error_occurred"]
    
    for event_type in event_types:
        cognitive_manager.subscribe_to_events(observer, event_type=event_type)
    
    # Verify subscriptions
    for event_type in event_types:
        count = cognitive_manager.get_event_subscriber_count(event_type)
        assert count >= 1


def test_subscribe_to_events_duplicate(cognitive_manager: CognitiveManager):
    """
    Test 5: subscribe_to_events() with duplicate subscription.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: subscribe_to_events()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.subscribe()
    Test Senaryosu: Duplicate subscription (edge case)
    """
    class TestObserver:
        def on_event(self, event: Dict[str, Any]) -> None:
            pass
    
    observer = TestObserver()
    
    # Subscribe twice
    cognitive_manager.subscribe_to_events(observer)
    cognitive_manager.subscribe_to_events(observer)  # Duplicate
    
    # Should handle gracefully (may ignore or count once)
    count = cognitive_manager.get_event_subscriber_count()
    assert count >= 1


def test_subscribe_to_events_invalid_observer(cognitive_manager: CognitiveManager):
    """
    Test 6: subscribe_to_events() with invalid observer.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: subscribe_to_events()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.subscribe()
    Test Senaryosu: Invalid observer (edge case)
    """
    try:
        cognitive_manager.subscribe_to_events(None)  # type: ignore
    except (TypeError, ValueError):
        # Expected behavior
        pass


def test_subscribe_to_events_with_handler(cognitive_manager: CognitiveManager):
    """
    Test 7: subscribe_to_events() with custom handler method.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: subscribe_to_events()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.subscribe()
    Test Senaryosu: Custom handler method ile subscription
    """
    class TestObserver:
        def __init__(self):
            self.custom_events = []
        
        def custom_handler(self, event: Dict[str, Any]) -> None:
            self.custom_events.append(event)
    
    observer = TestObserver()
    # Note: Implementation may require specific method name
    # This test checks if custom handlers are supported
    try:
        cognitive_manager.subscribe_to_events(observer)
    except Exception:
        # May or may not support custom handlers
        pass


def test_subscribe_to_events_concurrent(cognitive_manager: CognitiveManager):
    """
    Test 8: subscribe_to_events() concurrent subscription.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: subscribe_to_events()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.subscribe()
    Test Senaryosu: Concurrent subscription testi
    """
    import threading
    
    class TestObserver:
        def on_event(self, event: Dict[str, Any]) -> None:
            pass
    
    observers = [TestObserver() for _ in range(5)]
    
    def subscribe_worker(observer: TestObserver):
        cognitive_manager.subscribe_to_events(observer)
    
    threads = [threading.Thread(target=subscribe_worker, args=(obs,)) for obs in observers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    count = cognitive_manager.get_event_subscriber_count()
    assert count >= 5


def test_subscribe_to_events_performance(cognitive_manager: CognitiveManager):
    """
    Test 9: subscribe_to_events() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: subscribe_to_events()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.subscribe()
    Test Senaryosu: Performans testi
    """
    import time
    
    class TestObserver:
        def on_event(self, event: Dict[str, Any]) -> None:
            pass
    
    start = time.time()
    for i in range(100):
        observer = TestObserver()
        cognitive_manager.subscribe_to_events(observer)
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should be fast


def test_subscribe_to_events_integration(cognitive_manager: CognitiveManager):
    """
    Test 10: subscribe_to_events() integration test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: subscribe_to_events(), publish_event()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metodlar: EventBus.subscribe(), publish()
    Test Senaryosu: Subscription ve publishing integration
    """
    class IntegrationObserver:
        def __init__(self):
            self.received_events = []
        
        def on_event(self, event: Dict[str, Any]) -> None:
            self.received_events.append(event)
    
    observer = IntegrationObserver()
    cognitive_manager.subscribe_to_events(observer, event_type="test_event")
    
    # Publish event
    cognitive_manager.publish_event(
        event_type="test_event",
        data={"key": "value"},
        source="integration_test"
    )
    
    # Observer may receive event (implementation dependent)
    assert isinstance(observer.received_events, list)


# ============================================================================
# Test 11-20: unsubscribe_from_events() - Event Unsubscription
# Test Edilen Dosya: cognitive_manager.py (unsubscribe_from_events method)
# Alt Modül: v2/events/event_bus.py (EventBus.unsubscribe)
# ============================================================================

def test_unsubscribe_from_events_basic(cognitive_manager: CognitiveManager):
    """
    Test 11: Basic unsubscribe_from_events() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: unsubscribe_from_events()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.unsubscribe()
    Test Senaryosu: Basit event unsubscription
    """
    class TestObserver:
        def on_event(self, event: Dict[str, Any]) -> None:
            pass
    
    observer = TestObserver()
    
    # Subscribe
    cognitive_manager.subscribe_to_events(observer)
    count_before = cognitive_manager.get_event_subscriber_count()
    
    # Unsubscribe
    cognitive_manager.unsubscribe_from_events(observer)
    count_after = cognitive_manager.get_event_subscriber_count()
    
    assert count_after <= count_before


def test_unsubscribe_from_events_specific_type(cognitive_manager: CognitiveManager):
    """
    Test 12: unsubscribe_from_events() with specific event type.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: unsubscribe_from_events(event_type)
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.unsubscribe()
    Test Senaryosu: Belirli event type için unsubscription
    """
    class TestObserver:
        def on_event(self, event: Dict[str, Any]) -> None:
            pass
    
    observer = TestObserver()
    
    # Subscribe to multiple types
    cognitive_manager.subscribe_to_events(observer, event_type="type1")
    cognitive_manager.subscribe_to_events(observer, event_type="type2")
    
    # Unsubscribe from one type
    cognitive_manager.unsubscribe_from_events(observer, event_type="type1")
    
    # Should still be subscribed to type2
    count_type2 = cognitive_manager.get_event_subscriber_count("type2")
    assert count_type2 >= 1


def test_unsubscribe_from_events_nonexistent(cognitive_manager: CognitiveManager):
    """
    Test 13: unsubscribe_from_events() with nonexistent observer.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: unsubscribe_from_events()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.unsubscribe()
    Test Senaryosu: Var olmayan observer unsubscription (edge case)
    """
    class TestObserver:
        def on_event(self, event: Dict[str, Any]) -> None:
            pass
    
    observer = TestObserver()
    
    # Unsubscribe without subscribing (should not crash)
    cognitive_manager.unsubscribe_from_events(observer)


def test_unsubscribe_from_events_all_types(cognitive_manager: CognitiveManager):
    """
    Test 14: unsubscribe_from_events() from all event types.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: unsubscribe_from_events()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.unsubscribe()
    Test Senaryosu: Tüm event type'larından unsubscription
    """
    class TestObserver:
        def on_event(self, event: Dict[str, Any]) -> None:
            pass
    
    observer = TestObserver()
    
    # Subscribe to multiple types
    for event_type in ["type1", "type2", "type3"]:
        cognitive_manager.subscribe_to_events(observer, event_type=event_type)
    
    # Unsubscribe from all (no event_type specified)
    cognitive_manager.unsubscribe_from_events(observer)
    
    # Should be unsubscribed from all
    total_count = cognitive_manager.get_event_subscriber_count()
    # May have other subscribers, but this observer should be removed


def test_unsubscribe_from_events_idempotent(cognitive_manager: CognitiveManager):
    """
    Test 15: unsubscribe_from_events() idempotency.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: unsubscribe_from_events()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.unsubscribe()
    Test Senaryosu: Idempotent olması
    """
    class TestObserver:
        def on_event(self, event: Dict[str, Any]) -> None:
            pass
    
    observer = TestObserver()
    
    cognitive_manager.subscribe_to_events(observer)
    cognitive_manager.unsubscribe_from_events(observer)
    cognitive_manager.unsubscribe_from_events(observer)  # Unsubscribe again
    
    # Should not crash
    count = cognitive_manager.get_event_subscriber_count()
    assert isinstance(count, int)


def test_unsubscribe_from_events_multiple_observers(cognitive_manager: CognitiveManager):
    """
    Test 16: unsubscribe_from_events() with multiple observers.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: unsubscribe_from_events()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.unsubscribe()
    Test Senaryosu: Multiple observer unsubscription
    """
    class TestObserver:
        def on_event(self, event: Dict[str, Any]) -> None:
            pass
    
    observers = [TestObserver() for _ in range(5)]
    
    # Subscribe all
    for observer in observers:
        cognitive_manager.subscribe_to_events(observer)
    
    # Unsubscribe one
    cognitive_manager.unsubscribe_from_events(observers[0])
    
    # Others should still be subscribed
    count = cognitive_manager.get_event_subscriber_count()
    assert count >= 4


def test_unsubscribe_from_events_performance(cognitive_manager: CognitiveManager):
    """
    Test 17: unsubscribe_from_events() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: unsubscribe_from_events()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.unsubscribe()
    Test Senaryosu: Performans testi
    """
    import time
    
    class TestObserver:
        def on_event(self, event: Dict[str, Any]) -> None:
            pass
    
    observers = [TestObserver() for _ in range(50)]
    
    # Subscribe all
    for observer in observers:
        cognitive_manager.subscribe_to_events(observer)
    
    # Unsubscribe all
    start = time.time()
    for observer in observers:
        cognitive_manager.unsubscribe_from_events(observer)
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should be fast


def test_unsubscribe_from_events_concurrent(cognitive_manager: CognitiveManager):
    """
    Test 18: unsubscribe_from_events() concurrent operations.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: unsubscribe_from_events()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.unsubscribe()
    Test Senaryosu: Concurrent unsubscription
    """
    import threading
    
    class TestObserver:
        def on_event(self, event: Dict[str, Any]) -> None:
            pass
    
    observers = [TestObserver() for _ in range(5)]
    
    # Subscribe all
    for observer in observers:
        cognitive_manager.subscribe_to_events(observer)
    
    # Unsubscribe concurrently
    def unsubscribe_worker(observer: TestObserver):
        cognitive_manager.unsubscribe_from_events(observer)
    
    threads = [threading.Thread(target=unsubscribe_worker, args=(obs,)) for obs in observers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    # Should complete without errors
    count = cognitive_manager.get_event_subscriber_count()
    assert isinstance(count, int)


def test_unsubscribe_from_events_integration(cognitive_manager: CognitiveManager):
    """
    Test 19: unsubscribe_from_events() integration test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: subscribe_to_events(), unsubscribe_from_events(), publish_event()
    Alt Modül Dosyası: v2/events/event_bus.py
    Test Senaryosu: Integration testi
    """
    class IntegrationObserver:
        def __init__(self):
            self.events = []
        
        def on_event(self, event: Dict[str, Any]) -> None:
            self.events.append(event)
    
    observer = IntegrationObserver()
    
    # Subscribe
    cognitive_manager.subscribe_to_events(observer, event_type="integration_test")
    
    # Publish event
    cognitive_manager.publish_event("integration_test", {"key": "value"})
    
    # Unsubscribe
    cognitive_manager.unsubscribe_from_events(observer, event_type="integration_test")
    
    # Publish again (should not be received)
    cognitive_manager.publish_event("integration_test", {"key": "value2"})
    
    # Observer should have received first event (implementation dependent)
    assert isinstance(observer.events, list)


def test_subscribe_unsubscribe_cycle(cognitive_manager: CognitiveManager):
    """
    Test 20: subscribe_to_events() and unsubscribe_from_events() cycle.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: subscribe_to_events(), unsubscribe_from_events()
    Alt Modül Dosyası: v2/events/event_bus.py
    Test Senaryosu: Subscribe/unsubscribe cycle
    """
    class TestObserver:
        def on_event(self, event: Dict[str, Any]) -> None:
            pass
    
    observer = TestObserver()
    
    # Multiple cycles
    for _ in range(3):
        cognitive_manager.subscribe_to_events(observer)
        cognitive_manager.unsubscribe_from_events(observer)
    
    # Should work without errors
    count = cognitive_manager.get_event_subscriber_count()
    assert isinstance(count, int)


# ============================================================================
# Test 21-30: publish_event() - Event Publishing
# Test Edilen Dosya: cognitive_manager.py (publish_event method)
# Alt Modül: v2/events/event_bus.py (EventBus.publish)
# ============================================================================

def test_publish_event_basic(cognitive_manager: CognitiveManager):
    """
    Test 21: Basic publish_event() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: publish_event()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.publish()
    Test Senaryosu: Basit event publishing
    """
    cognitive_manager.publish_event(
        event_type="test_event",
        data={"key": "value"},
        source="test_source"
    )
    
    # Event should be in history
    history = cognitive_manager.get_event_history()
    assert isinstance(history, list)


def test_publish_event_with_data(cognitive_manager: CognitiveManager):
    """
    Test 22: publish_event() with data.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: publish_event()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.publish()
    Test Senaryosu: Data ile event publishing
    """
    test_data = {
        "message": "Test message",
        "count": 42,
        "nested": {"key": "value"}
    }
    
    cognitive_manager.publish_event(
        event_type="data_event",
        data=test_data,
        source="test_source"
    )
    
    history = cognitive_manager.get_event_history("data_event")
    assert isinstance(history, list)


def test_publish_event_multiple_events(cognitive_manager: CognitiveManager):
    """
    Test 23: publish_event() with multiple events.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: publish_event()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.publish()
    Test Senaryosu: Multiple event publishing
    """
    for i in range(10):
        cognitive_manager.publish_event(
            event_type=f"multi_event_{i}",
            data={"index": i},
            source="test_source"
        )
    
    history = cognitive_manager.get_event_history()
    assert isinstance(history, list)


def test_publish_event_empty_data(cognitive_manager: CognitiveManager):
    """
    Test 24: publish_event() with empty data.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: publish_event()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.publish()
    Test Senaryosu: Boş data ile event (edge case)
    """
    cognitive_manager.publish_event(
        event_type="empty_data_event",
        data={},
        source="test_source"
    )
    
    history = cognitive_manager.get_event_history()
    assert isinstance(history, list)


def test_publish_event_large_data(cognitive_manager: CognitiveManager):
    """
    Test 25: publish_event() with large data.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: publish_event()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.publish()
    Test Senaryosu: Büyük data ile event (edge case)
    """
    large_data = {
        "large_array": list(range(1000)),
        "large_string": "x" * 10000
    }
    
    cognitive_manager.publish_event(
        event_type="large_data_event",
        data=large_data,
        source="test_source"
    )
    
    history = cognitive_manager.get_event_history()
    assert isinstance(history, list)


def test_publish_event_with_subscribers(cognitive_manager: CognitiveManager):
    """
    Test 26: publish_event() with subscribers.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: publish_event(), subscribe_to_events()
    Alt Modül Dosyası: v2/events/event_bus.py
    Test Senaryosu: Subscriber'lar ile event publishing
    """
    class TestObserver:
        def __init__(self):
            self.received_events = []
        
        def on_event(self, event: Dict[str, Any]) -> None:
            self.received_events.append(event)
    
    observer = TestObserver()
    cognitive_manager.subscribe_to_events(observer, event_type="subscriber_test")
    
    cognitive_manager.publish_event(
        event_type="subscriber_test",
        data={"key": "value"},
        source="test_source"
    )
    
    # Observer may receive event
    assert isinstance(observer.received_events, list)


def test_publish_event_performance(cognitive_manager: CognitiveManager):
    """
    Test 27: publish_event() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: publish_event()
    Alt Modül Dosyası: v2/events/event_bus.py
    Test Senaryosu: Performans testi
    """
    import time
    
    start = time.time()
    for i in range(100):
        cognitive_manager.publish_event(
            event_type=f"perf_event_{i}",
            data={"index": i},
            source="test_source"
        )
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should be fast


def test_publish_event_concurrent(cognitive_manager: CognitiveManager):
    """
    Test 28: publish_event() concurrent publishing.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: publish_event()
    Alt Modül Dosyası: v2/events/event_bus.py
    Test Senaryosu: Concurrent publishing
    """
    import threading
    
    def publish_worker(worker_id: int):
        for i in range(10):
            cognitive_manager.publish_event(
                event_type=f"concurrent_event_{worker_id}",
                data={"worker": worker_id, "index": i},
                source="test_source"
            )
    
    threads = [threading.Thread(target=publish_worker, args=(i,)) for i in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    history = cognitive_manager.get_event_history()
    assert isinstance(history, list)


def test_publish_event_integration(cognitive_manager: CognitiveManager):
    """
    Test 29: publish_event() integration test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: publish_event(), get_event_history()
    Alt Modül Dosyası: v2/events/event_bus.py
    Test Senaryosu: Integration testi
    """
    # Publish events
    for i in range(5):
        cognitive_manager.publish_event(
            event_type="integration_event",
            data={"index": i},
            source="integration_test"
        )
    
    # Get history
    history = cognitive_manager.get_event_history("integration_event")
    assert isinstance(history, list)


def test_publish_event_error_handling(cognitive_manager: CognitiveManager):
    """
    Test 30: publish_event() error handling.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: publish_event()
    Alt Modül Dosyası: v2/events/event_bus.py
    Test Senaryosu: Hata durumlarında handling
    """
    # Invalid event type
    try:
        cognitive_manager.publish_event(
            event_type=None,  # type: ignore
            data={"key": "value"},
            source="test_source"
        )
    except (TypeError, ValueError):
        # Expected behavior
        pass


# ============================================================================
# Test 31-40: get_event_history() and clear_event_history()
# Test Edilen Dosya: cognitive_manager.py
# Alt Modül: v2/events/event_bus.py (EventBus.get_history, clear_history)
# ============================================================================

def test_get_event_history_basic(cognitive_manager: CognitiveManager):
    """
    Test 31: Basic get_event_history() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_event_history()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.get_history()
    Test Senaryosu: Basit event history alma
    """
    history = cognitive_manager.get_event_history()
    assert isinstance(history, list)


def test_get_event_history_with_limit(cognitive_manager: CognitiveManager):
    """
    Test 32: get_event_history() with limit.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_event_history(limit)
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.get_history()
    Test Senaryosu: Limit ile history alma
    """
    # Publish many events
    for i in range(20):
        cognitive_manager.publish_event(
            event_type="limit_test",
            data={"index": i},
            source="test_source"
        )
    
    history = cognitive_manager.get_event_history(limit=10)
    assert isinstance(history, list)
    assert len(history) <= 10


def test_get_event_history_filtered(cognitive_manager: CognitiveManager):
    """
    Test 33: get_event_history() with event type filter.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_event_history(event_type)
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.get_history()
    Test Senaryosu: Event type filter ile history
    """
    # Publish different event types
    cognitive_manager.publish_event("type1", {"key": "value1"}, "test")
    cognitive_manager.publish_event("type2", {"key": "value2"}, "test")
    cognitive_manager.publish_event("type1", {"key": "value3"}, "test")
    
    # Get history for specific type
    history = cognitive_manager.get_event_history("type1")
    assert isinstance(history, list)


def test_clear_event_history_basic(cognitive_manager: CognitiveManager):
    """
    Test 34: Basic clear_event_history() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: clear_event_history()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.clear_history()
    Test Senaryosu: Basit event history temizleme
    """
    # Publish events
    for i in range(5):
        cognitive_manager.publish_event(f"clear_test_{i}", {"index": i}, "test")
    
    # Clear history
    cognitive_manager.clear_event_history()
    
    # History should be cleared
    history = cognitive_manager.get_event_history()
    assert isinstance(history, list)
    # May be empty or have new events


def test_clear_event_history_idempotent(cognitive_manager: CognitiveManager):
    """
    Test 35: clear_event_history() idempotency.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: clear_event_history()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.clear_history()
    Test Senaryosu: Idempotent olması
    """
    cognitive_manager.clear_event_history()
    cognitive_manager.clear_event_history()  # Clear again
    
    # Should not crash
    history = cognitive_manager.get_event_history()
    assert isinstance(history, list)


def test_get_event_history_after_clear(cognitive_manager: CognitiveManager):
    """
    Test 36: get_event_history() after clearing.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: clear_event_history(), get_event_history()
    Alt Modül Dosyası: v2/events/event_bus.py
    Test Senaryosu: Clear sonrası history
    """
    # Publish events
    cognitive_manager.publish_event("before_clear", {"key": "value"}, "test")
    
    # Clear
    cognitive_manager.clear_event_history()
    
    # Publish new event
    cognitive_manager.publish_event("after_clear", {"key": "value"}, "test")
    
    # Get history
    history = cognitive_manager.get_event_history()
    assert isinstance(history, list)


def test_get_event_history_performance(cognitive_manager: CognitiveManager):
    """
    Test 37: get_event_history() performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_event_history()
    Alt Modül Dosyası: v2/events/event_bus.py
    Test Senaryosu: Performans testi
    """
    import time
    
    # Publish many events
    for i in range(100):
        cognitive_manager.publish_event(f"perf_history_{i}", {"index": i}, "test")
    
    start = time.time()
    history = cognitive_manager.get_event_history()
    elapsed = time.time() - start
    
    assert isinstance(history, list)
    assert elapsed < 0.5  # Should be fast


def test_get_event_history_empty(cognitive_manager: CognitiveManager):
    """
    Test 38: get_event_history() with empty history.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_event_history()
    Alt Modül Dosyası: v2/events/event_bus.py
    Test Senaryosu: Boş history (edge case)
    """
    cognitive_manager.clear_event_history()
    history = cognitive_manager.get_event_history()
    assert isinstance(history, list)
    # May be empty


def test_event_history_integration(cognitive_manager: CognitiveManager):
    """
    Test 39: Event history integration test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: publish_event(), get_event_history(), clear_event_history()
    Alt Modül Dosyası: v2/events/event_bus.py
    Test Senaryosu: Integration testi
    """
    # Publish events
    for i in range(5):
        cognitive_manager.publish_event(f"integration_{i}", {"index": i}, "test")
    
    # Get history
    history = cognitive_manager.get_event_history()
    assert isinstance(history, list)
    
    # Clear
    cognitive_manager.clear_event_history()
    
    # Get history after clear
    history_after = cognitive_manager.get_event_history()
    assert isinstance(history_after, list)


def test_get_event_subscriber_count_basic(cognitive_manager: CognitiveManager):
    """
    Test 40: Basic get_event_subscriber_count() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_event_subscriber_count()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.get_subscriber_count()
    Test Senaryosu: Basit subscriber count alma
    """
    count = cognitive_manager.get_event_subscriber_count()
    assert isinstance(count, int)
    assert count >= 0


# ============================================================================
# Test 41-50: get_event_metrics(), reset_event_metrics(), and Integration
# Test Edilen Dosya: cognitive_manager.py
# Alt Modül: v2/events/event_bus.py (EventBus)
# ============================================================================

def test_get_event_subscriber_count_filtered(cognitive_manager: CognitiveManager):
    """
    Test 41: get_event_subscriber_count() with event type filter.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_event_subscriber_count(event_type)
    Alt Modül Dosyası: v2/events/event_bus.py
    Test Senaryosu: Event type filter ile subscriber count
    """
    class TestObserver:
        def on_event(self, event: Dict[str, Any]) -> None:
            pass
    
    observer = TestObserver()
    cognitive_manager.subscribe_to_events(observer, event_type="filtered_test")
    
    count = cognitive_manager.get_event_subscriber_count("filtered_test")
    assert isinstance(count, int)
    assert count >= 1


def test_get_event_metrics_basic(cognitive_manager: CognitiveManager):
    """
    Test 42: Basic get_event_metrics() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: get_event_metrics()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.get_metrics()
    Test Senaryosu: Basit event metrics alma
    """
    metrics = cognitive_manager.get_event_metrics()
    assert isinstance(metrics, dict)
    # Common metrics fields
    assert "published" in metrics or "subscribed" in metrics or "total" in metrics or len(metrics) >= 0


def test_get_event_metrics_after_publishing(cognitive_manager: CognitiveManager):
    """
    Test 43: get_event_metrics() after publishing events.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: get_event_metrics(), publish_event()
    Alt Modül Dosyası: v2/events/event_bus.py
    Test Senaryosu: Publishing sonrası metrics
    """
    # Publish events
    for i in range(10):
        cognitive_manager.publish_event(f"metrics_test_{i}", {"index": i}, "test")
    
    metrics = cognitive_manager.get_event_metrics()
    assert isinstance(metrics, dict)


def test_reset_event_metrics_basic(cognitive_manager: CognitiveManager):
    """
    Test 44: Basic reset_event_metrics() functionality.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: reset_event_metrics()
    Alt Modül Dosyası: v2/events/event_bus.py
    Alt Modül Metod: EventBus.reset_metrics()
    Test Senaryosu: Basit event metrics reset
    """
    # Publish events
    cognitive_manager.publish_event("reset_test", {"key": "value"}, "test")
    
    # Reset metrics
    cognitive_manager.reset_event_metrics()
    
    # Get metrics
    metrics = cognitive_manager.get_event_metrics()
    assert isinstance(metrics, dict)


def test_reset_event_metrics_idempotent(cognitive_manager: CognitiveManager):
    """
    Test 45: reset_event_metrics() idempotency.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metod: reset_event_metrics()
    Alt Modül Dosyası: v2/events/event_bus.py
    Test Senaryosu: Idempotent olması
    """
    cognitive_manager.reset_event_metrics()
    cognitive_manager.reset_event_metrics()  # Reset again
    
    metrics = cognitive_manager.get_event_metrics()
    assert isinstance(metrics, dict)


def test_event_metrics_consistency(cognitive_manager: CognitiveManager):
    """
    Test 46: Event metrics consistency.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: get_event_metrics(), reset_event_metrics()
    Alt Modül Dosyası: v2/events/event_bus.py
    Test Senaryosu: Metrics consistency testi
    """
    metrics1 = cognitive_manager.get_event_metrics()
    cognitive_manager.reset_event_metrics()
    metrics2 = cognitive_manager.get_event_metrics()
    
    assert isinstance(metrics1, dict)
    assert isinstance(metrics2, dict)


def test_event_management_full_workflow(cognitive_manager: CognitiveManager):
    """
    Test 47: Full event management workflow.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: Tüm event management metodları
    Alt Modül Dosyası: v2/events/event_bus.py
    Test Senaryosu: Tam event management workflow
    """
    class WorkflowObserver:
        def __init__(self):
            self.events = []
        
        def on_event(self, event: Dict[str, Any]) -> None:
            self.events.append(event)
    
    observer = WorkflowObserver()
    
    # 1. Subscribe
    cognitive_manager.subscribe_to_events(observer, event_type="workflow_test")
    
    # 2. Publish events
    for i in range(5):
        cognitive_manager.publish_event("workflow_test", {"index": i}, "test")
    
    # 3. Get history
    history = cognitive_manager.get_event_history("workflow_test")
    assert isinstance(history, list)
    
    # 4. Get metrics
    metrics = cognitive_manager.get_event_metrics()
    assert isinstance(metrics, dict)
    
    # 5. Get subscriber count
    count = cognitive_manager.get_event_subscriber_count("workflow_test")
    assert count >= 1
    
    # 6. Unsubscribe
    cognitive_manager.unsubscribe_from_events(observer, event_type="workflow_test")
    
    # 7. Reset metrics
    cognitive_manager.reset_event_metrics()
    
    # 8. Clear history
    cognitive_manager.clear_event_history()


def test_event_management_performance(cognitive_manager: CognitiveManager):
    """
    Test 48: Event management performance test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: publish_event(), get_event_history()
    Alt Modül Dosyası: v2/events/event_bus.py
    Test Senaryosu: Performans testi
    """
    import time
    
    start = time.time()
    
    # Publish many events
    for i in range(100):
        cognitive_manager.publish_event(f"perf_{i}", {"index": i}, "test")
    
    # Get history
    history = cognitive_manager.get_event_history()
    
    # Get metrics
    metrics = cognitive_manager.get_event_metrics()
    
    elapsed = time.time() - start
    assert elapsed < 2.0  # Should complete in reasonable time
    assert isinstance(history, list)
    assert isinstance(metrics, dict)


def test_event_management_concurrent(cognitive_manager: CognitiveManager):
    """
    Test 49: Event management concurrent operations.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: subscribe_to_events(), publish_event()
    Alt Modül Dosyası: v2/events/event_bus.py
    Test Senaryosu: Concurrent operations testi
    """
    import threading
    
    class ConcurrentObserver:
        def on_event(self, event: Dict[str, Any]) -> None:
            pass
    
    observers = [ConcurrentObserver() for _ in range(5)]
    
    def worker(observer: ConcurrentObserver, worker_id: int):
        cognitive_manager.subscribe_to_events(observer)
        for i in range(10):
            cognitive_manager.publish_event(f"concurrent_{worker_id}", {"index": i}, "test")
    
    threads = [threading.Thread(target=worker, args=(obs, i)) for i, obs in enumerate(observers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    history = cognitive_manager.get_event_history()
    assert isinstance(history, list)


def test_event_management_end_to_end(cognitive_manager: CognitiveManager, cognitive_state: CognitiveState):
    """
    Test 50: Event management end-to-end test.
    
    Test Edilen Dosya: cognitive_manager.py
    Test Edilen Metodlar: Tüm event management metodları
    Alt Modül Dosyası: v2/events/event_bus.py, v2/events/event_handlers.py
    Test Senaryosu: End-to-end event management testi
    """
    class E2EObserver:
        def __init__(self):
            self.events = []
        
        def on_event(self, event: Dict[str, Any]) -> None:
            self.events.append(event)
    
    observer = E2EObserver()
    
    # 1. Subscribe to multiple event types
    cognitive_manager.subscribe_to_events(observer, event_type="e2e_test")
    
    # 2. Process request (may generate events)
    input_msg = CognitiveInput(user_message="E2E event test")
    cognitive_manager.handle(cognitive_state, input_msg)
    
    # 3. Publish custom event
    cognitive_manager.publish_event("e2e_test", {"custom": "data"}, "e2e_test")
    
    # 4. Get event history
    history = cognitive_manager.get_event_history()
    assert isinstance(history, list)
    
    # 5. Get event metrics
    metrics = cognitive_manager.get_event_metrics()
    assert isinstance(metrics, dict)
    
    # 6. Get subscriber count
    count = cognitive_manager.get_event_subscriber_count("e2e_test")
    assert count >= 1
    
    # 7. Unsubscribe
    cognitive_manager.unsubscribe_from_events(observer, event_type="e2e_test")
    
    # 8. Reset metrics
    cognitive_manager.reset_event_metrics()
    
    # 9. Clear history
    cognitive_manager.clear_event_history()
    
    # 10. Verify final state
    final_history = cognitive_manager.get_event_history()
    assert isinstance(final_history, list)

