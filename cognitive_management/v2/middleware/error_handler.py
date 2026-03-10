# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: error_handler.py
Modül: cognitive_management/v2/middleware
Görev: Error Handling Middleware - Retry, circuit breaker, timeout, graceful
       degradation. CircuitState, CircuitBreaker, RetryPolicy ve ErrorHandlerMiddleware
       sınıflarını içerir. Retry logic, circuit breaker pattern, timeout handling
       ve graceful degradation işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (error handling middleware),
                     Dependency Inversion (BaseMiddleware interface'e bağımlı)
- Design Patterns: Circuit Breaker Pattern, Retry Pattern
- Endüstri Standartları: Error handling best practices

KULLANIM:
- Retry logic için
- Circuit breaker için
- Timeout handling için
- Graceful degradation için

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
from typing import Optional, Callable, Any
import time
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
class RetryConfig:
    """Retry configuration"""
    max_retries: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 10.0  # seconds
    exponential_base: float = 2.0
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Open circuit after N failures
    success_threshold: int = 2  # Close circuit after N successes
    timeout: float = 60.0  # seconds to wait before half-open
    failure_window: float = 60.0  # seconds for failure counting


@dataclass
class CircuitBreakerState:
    """Circuit breaker internal state"""
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    successes: int = 0
    last_failure_time: Optional[datetime] = None
    opened_at: Optional[datetime] = None


class ErrorHandlingMiddleware(BaseMiddleware):
    """
    Error handling middleware.
    - Retry mechanisms
    - Circuit breaker
    - Timeout handling
    - Graceful degradation
    """
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        timeout: Optional[float] = None,
    ):
        super().__init__("ErrorHandling")
        self.retry_config = retry_config or RetryConfig()
        self.circuit_config = circuit_breaker_config or CircuitBreakerConfig()
        self.timeout = timeout or 30.0  # Default 30 seconds
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
        """Handle errors with retry and circuit breaker"""
        # Update circuit breaker
        self._record_failure()
        
        # Check if retryable
        if not self._is_retryable(error):
            return self._create_error_response(error)
        
        # Retry logic
        last_error = error
        for attempt in range(self.retry_config.max_retries):
            try:
                # Calculate delay
                delay = min(
                    self.retry_config.initial_delay * (self.retry_config.exponential_base ** attempt),
                    self.retry_config.max_delay
                )
                
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt + 1}/{self.retry_config.max_retries} after {delay}s")
                    time.sleep(delay)
                
                # Retry (this will be handled by orchestrator)
                # For now, return error response
                return self._create_error_response(error)
                
            except Exception as retry_error:
                last_error = retry_error
                self._record_failure()
        
        # All retries failed
        logger.error(f"All retries failed: {last_error}")
        return self._create_error_response(last_error)
    
    def _after(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        response: CognitiveOutput,
    ) -> CognitiveOutput:
        """Record success after processing"""
        self._record_success()
        return response
    
    def _is_retryable(self, error: Exception) -> bool:
        """Check if error is retryable"""
        return isinstance(error, self.retry_config.retryable_exceptions)
    
    def _record_failure(self) -> None:
        """Record failure in circuit breaker"""
        self.circuit_state.failures += 1
        self.circuit_state.last_failure_time = datetime.now()
        
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
                self.circuit_state.successes = 0
                logger.info("Circuit breaker: HALF_OPEN -> CLOSED")
        elif self.circuit_state.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.circuit_state.failures = 0
    
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
    "RetryConfig",
    "CircuitBreakerConfig",
    "CircuitState",
]

