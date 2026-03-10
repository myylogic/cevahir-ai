# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: event_handlers.py
Modül: cognitive_management/v2/events
Görev: Event Handlers - Event observer implementations. EventHandler, LoggingEventHandler
       ve diğer event handler sınıflarını içerir. Callback-based event handling,
       logging event handling ve custom event handling işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (event handling),
                     Dependency Inversion (EventObserver interface'e bağımlı)
- Design Patterns: Handler Pattern (event handling)
- Endüstri Standartları: Event handling best practices

KULLANIM:
- Event handling için
- Callback-based handling için
- Logging event handling için

BAĞIMLILIKLAR:
- EventBus: Event bus interface
- logging: Logging işlemleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Dict, List, Callable, Optional
from .event_bus import EventObserver, CognitiveEvent
import logging


class EventHandler(EventObserver):
    """
    Generic event handler.
    Callback-based event handling.
    """
    
    def __init__(
        self,
        callback: Callable[[CognitiveEvent], None],
        event_type: Optional[str] = None
    ):
        """
        Initialize event handler.
        
        Args:
            callback: Function to call when event occurs
            event_type: Filter by event type (None = all)
        """
        self.callback = callback
        self.event_type = event_type
        self.logger = logging.getLogger(f"{__name__}.EventHandler")
    
    def on_event(self, event: CognitiveEvent) -> None:
        """Handle event"""
        if self.event_type is None or event.event_type == self.event_type:
            try:
                self.callback(event)
            except Exception as e:
                self.logger.error(f"Event handler callback error: {e}", exc_info=True)


class LoggingEventHandler(EventObserver):
    """
    Logging event handler.
    Events'i loglar.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("CognitiveEvents")
    
    def on_event(self, event: CognitiveEvent) -> None:
        """Log event"""
        self.logger.info(
            f"Event: {event.event_type} | "
            f"Source: {event.source or 'unknown'} | "
            f"Data: {len(event.data)} fields"
        )


class MetricsEventHandler(EventObserver):
    """
    Metrics event handler.
    Events'leri metrics olarak toplar.
    """
    
    def __init__(self):
        self.metrics: Dict[str, int] = {}
        self.logger = logging.getLogger(f"{__name__}.MetricsEventHandler")
    
    def on_event(self, event: CognitiveEvent) -> None:
        """Collect metrics"""
        event_type = event.event_type
        self.metrics[event_type] = self.metrics.get(event_type, 0) + 1
    
    def get_metrics(self) -> Dict[str, int]:
        """Get collected metrics"""
        return dict(self.metrics)
    
    def reset_metrics(self) -> None:
        """Reset metrics"""
        self.metrics.clear()


__all__ = [
    "EventHandler",
    "LoggingEventHandler",
    "MetricsEventHandler",
]

