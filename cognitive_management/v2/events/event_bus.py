# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: event_bus.py
Modül: cognitive_management/v2/events
Görev: Event Bus - Observer pattern ile event publishing/subscription sistemi.
       CognitiveEvent, EventObserver ve EventBus sınıflarını içerir. Event
       publishing, subscription, unsubscription ve notification işlemlerini
       yapar. Thread-safe event bus implementation.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (event bus),
                     Dependency Inversion (observer pattern)
- Design Patterns: Observer Pattern (event bus), Bus Pattern
- Endüstri Standartları: Event-driven architecture best practices

KULLANIM:
- Event publishing için
- Event subscription için
- Event notification için

BAĞIMLILIKLAR:
- threading: Thread-safe işlemler
- dataclasses: Event data structures

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import threading


@dataclass
class CognitiveEvent:
    """
    Cognitive event data structure.
    """
    event_type: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    data: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    
    def __post_init__(self):
        """Validate event"""
        if not self.event_type:
            raise ValueError("event_type cannot be empty")


class EventObserver:
    """
    Event observer interface.
    Observer pattern için.
    """
    def on_event(self, event: CognitiveEvent) -> None:
        """
        Handle event.
        
        Args:
            event: Cognitive event
        """
        ...


class EventBus:
    """
    Event bus for publishing and subscribing to events.
    Observer pattern implementation.
    
    Thread-safe event bus.
    """
    
    def __init__(self):
        self._subscribers: Dict[str, Set[EventObserver]] = {}
        self._lock = threading.Lock()
        self._event_history: List[CognitiveEvent] = []  # Optional: event history
        self._max_history: int = 1000  # Max events to keep in history
    
    def subscribe(
        self,
        observer: EventObserver,
        event_type: Optional[str] = None
    ) -> None:
        """
        Subscribe observer to events.
        
        Args:
            observer: Event observer
            event_type: Specific event type (None = all events)
        """
        with self._lock:
            if event_type is None:
                event_type = "*"  # All events
            
            if event_type not in self._subscribers:
                self._subscribers[event_type] = set()
            
            self._subscribers[event_type].add(observer)
    
    def unsubscribe(
        self,
        observer: EventObserver,
        event_type: Optional[str] = None
    ) -> None:
        """
        Unsubscribe observer from events.
        
        Args:
            observer: Event observer
            event_type: Specific event type (None = all events)
        """
        with self._lock:
            if event_type is None:
                event_type = "*"
            
            if event_type in self._subscribers:
                self._subscribers[event_type].discard(observer)
    
    def publish(
        self,
        event: CognitiveEvent,
        source: Optional[str] = None
    ) -> None:
        """
        Publish event to all subscribers.
        
        Args:
            event: Cognitive event
            source: Optional source identifier
        """
        if source:
            event.source = source
        
        # Add to history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
        
        # Get subscribers (thread-safe copy)
        with self._lock:
            # Specific event type subscribers
            specific_subscribers = self._subscribers.get(event.event_type, set()).copy()
            # All events subscribers
            all_subscribers = self._subscribers.get("*", set()).copy()
        
        # Notify all subscribers
        all_observers = specific_subscribers | all_subscribers
        
        for observer in all_observers:
            try:
                observer.on_event(event)
            except Exception as e:
                # Observer hataları event bus'ı düşürmesin
                import logging
                logger = logging.getLogger("EventBus")
                logger.error(f"Observer {type(observer).__name__} error: {e}", exc_info=True)
    
    def get_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[CognitiveEvent]:
        """
        Get event history.
        
        Args:
            event_type: Filter by event type (None = all)
            limit: Maximum number of events to return
            
        Returns:
            List of events
        """
        with self._lock:
            if event_type is None:
                return self._event_history[-limit:]
            return [
                e for e in self._event_history[-limit:]
                if e.event_type == event_type
            ]
    
    def clear_history(self) -> None:
        """Clear event history"""
        with self._lock:
            self._event_history.clear()
    
    def get_subscriber_count(self, event_type: Optional[str] = None) -> int:
        """Get number of subscribers for event type"""
        with self._lock:
            if event_type is None:
                return sum(len(subs) for subs in self._subscribers.values())
            return len(self._subscribers.get(event_type, set()))


__all__ = ["EventBus", "CognitiveEvent", "EventObserver"]

