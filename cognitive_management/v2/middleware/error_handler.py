# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: error_handler.py
Modül: cognitive_management/v2/middleware
Görev: Error Handling Middleware - circuit breaker ve hata yanıtı (graceful
       degradation). Pipeline'dan taşan hata devre kesiciye işlenir ve kısa bir
       hata yanıtı döner.

       Bu middleware isteği YENİDEN DENEMEZ ve istek zaman aşımı UYGULAMAZ:
       pipeline oturum durumunu değiştirir, model çağrıları pahalıdır ve senkron
       bir model çağrısı ortasında kesilemez. Eski "retry" döngüsü ilk turda
       hata yanıtı dönüyordu ve 30 s zaman aşımı hiç uygulanmıyordu; bu sahte
       iddialar kaldırıldı. Maliyeti 'lean' profil ve araştırma bütçesi sınırlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (error handling middleware),
                     Dependency Inversion (BaseMiddleware interface'e bağımlı)
- Design Patterns: Circuit Breaker Pattern
- Endüstri Standartları: Error handling best practices

KULLANIM:
- Circuit breaker için
- Graceful degradation (hata yanıtı) için

BAĞIMLILIKLAR:
- BaseMiddleware: Base middleware
- CognitiveExceptions: Exception tanımları

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from collections import deque
from typing import Optional, Callable, Any, Deque
import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from cognitive_management.cognitive_types import (
    CognitiveState,
    CognitiveInput,
    CognitiveOutput,
)
from cognitive_management.exceptions import CognitiveError, CognitiveTimeout
from .base import BaseMiddleware


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Open circuit after N failures
    success_threshold: int = 2  # Close circuit after N successes
    timeout: float = 60.0  # seconds to wait before half-open
    failure_window: float = 60.0  # only failures within the last N seconds count toward the threshold


@dataclass
class CircuitBreakerState:
    """Circuit breaker internal state"""
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0  # failures inside the window since the last success
    successes: int = 0
    last_failure_time: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    failure_times: Deque[datetime] = field(default_factory=deque)


class ErrorHandlingMiddleware(BaseMiddleware):
    """
    Error handling middleware.
    - Circuit breaker: son failure_window saniyede, arada başarı olmadan
      failure_threshold hata birikirse devre açılır ve istekler timeout saniye
      reddedilir (kayan pencere: pencereden çıkan hata sayılmaz)
    - Graceful degradation: hata, kısa bir CognitiveOutput yanıtına çevrilir

    Yeniden deneme ve istek zaman aşımı yoktur (modül açıklamasına bakın).
    """
    
    def __init__(
        self,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    ):
        super().__init__("ErrorHandling")
        self.circuit_config = circuit_breaker_config or CircuitBreakerConfig()
        self.circuit_state = CircuitBreakerState()
    
    def _before(
        self,
        state: CognitiveState,
        request: CognitiveInput,
    ) -> tuple[CognitiveState, CognitiveInput]:
        """Check circuit breaker before processing"""
        # Check circuit breaker
        if self.circuit_state.state == CircuitState.OPEN:
            # Check if timeout passed
            if self.circuit_state.opened_at:
                elapsed = (datetime.now() - self.circuit_state.opened_at).total_seconds()
                if elapsed >= self.circuit_config.timeout:
                    # Transition to half-open
                    self.circuit_state.state = CircuitState.HALF_OPEN
                    self.circuit_state.successes = 0
                    logger.info("Circuit breaker: OPEN -> HALF_OPEN")
                else:
                    # Still open, reject request
                    raise CognitiveError(
                        "Circuit breaker is OPEN. Service temporarily unavailable.",
                        details={
                            "state": self.circuit_state.state.value,
                            "opened_at": self.circuit_state.opened_at.isoformat(),
                        }
                    )
        
        return state, request
    
    def _on_error(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        error: Exception,
    ) -> Optional[CognitiveOutput]:
        """Record the failure in the circuit breaker and answer with an error response.

        The request is not retried: the pipeline is not idempotent (it updates
        session history) and a budget denial must never be retried.
        """
        self._record_failure()
        logger.warning("Cognitive request failed (%s); returning error response", type(error).__name__)
        return self._create_error_response(error)
    
    def _after(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        response: CognitiveOutput,
    ) -> CognitiveOutput:
        """Record success after processing"""
        self._record_success()
        return response
    
    def _record_failure(self) -> None:
        """Record failure in circuit breaker.

        Sliding window: only failures within the last failure_window seconds
        count toward failure_threshold; a success in CLOSED state clears them.
        """
        now = datetime.now()
        times = self.circuit_state.failure_times
        times.append(now)
        while times and (now - times[0]).total_seconds() > self.circuit_config.failure_window:
            times.popleft()
        self.circuit_state.failures = len(times)
        self.circuit_state.last_failure_time = now
        
        if self.circuit_state.state == CircuitState.HALF_OPEN:
            # Half-open -> Open
            self.circuit_state.state = CircuitState.OPEN
            self.circuit_state.opened_at = datetime.now()
            logger.warning("Circuit breaker: HALF_OPEN -> OPEN")
        elif self.circuit_state.state == CircuitState.CLOSED:
            # Check if should open
            if self.circuit_state.failures >= self.circuit_config.failure_threshold:
                self.circuit_state.state = CircuitState.OPEN
                self.circuit_state.opened_at = datetime.now()
                logger.warning("Circuit breaker: CLOSED -> OPEN")
    
    def _record_success(self) -> None:
        """Record success in circuit breaker"""
        if self.circuit_state.state == CircuitState.HALF_OPEN:
            self.circuit_state.successes += 1
            if self.circuit_state.successes >= self.circuit_config.success_threshold:
                # Half-open -> Closed
                self.circuit_state.state = CircuitState.CLOSED
                self.circuit_state.failures = 0
                self.circuit_state.failure_times.clear()
                self.circuit_state.successes = 0
                logger.info("Circuit breaker: HALF_OPEN -> CLOSED")
        elif self.circuit_state.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.circuit_state.failures = 0
            self.circuit_state.failure_times.clear()
    
    def _create_error_response(self, error: Exception) -> CognitiveOutput:
        """Create error response"""
        if isinstance(error, CognitiveTimeout):
            message = "İşlem zaman aşımına uğradı. Lütfen tekrar deneyin."
        elif isinstance(error, CognitiveError):
            message = f"İşlem sırasında bir hata oluştu: {error.message}"
        else:
            message = "Beklenmeyen bir hata oluştu. Lütfen tekrar deneyin."
        
        return CognitiveOutput(
            text=message,
            used_mode="direct",
            tool_used=None,
            revised_by_critic=False,
        )


__all__ = [
    "ErrorHandlingMiddleware",
    "CircuitBreakerConfig",
    "CircuitState",
]

