# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: metrics.py
Modül: cognitive_management/v2/middleware
Görev: Metrics Middleware - Performance metrics, health checks, observability.
       Metrics, MetricsMiddleware sınıflarını içerir. Request count, success/error
       count, latency metrics, throughput metrics ve health check işlemlerini
       yapar. Performance monitoring ve observability sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (metrics middleware),
                     Dependency Inversion (BaseMiddleware interface'e bağımlı)
- Design Patterns: Middleware Pattern (metrics middleware)
- Endüstri Standartları: Metrics collection best practices

KULLANIM:
- Performance metrics için
- Health checks için
- Observability için

BAĞIMLILIKLAR:
- BaseMiddleware: Base middleware
- dataclasses: Metrics data structures

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import time
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

from cognitive_management.cognitive_types import (
    CognitiveState,
    CognitiveInput,
    CognitiveOutput,
)
from .base import BaseMiddleware


@dataclass
class Metrics:
    """Performance metrics"""
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency: float = 0.0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    last_request_time: Optional[datetime] = None
    
    @property
    def avg_latency(self) -> float:
        """Average latency"""
        if self.request_count == 0:
            return 0.0
        return self.total_latency / self.request_count
    
    @property
    def success_rate(self) -> float:
        """Success rate"""
        if self.request_count == 0:
            return 0.0
        return self.success_count / self.request_count


class MetricsMiddleware(BaseMiddleware):
    """
    Metrics collection middleware.
    - Request/response metrics
    - Latency tracking
    - Error rate tracking
    - Health checks
    """
    
    def __init__(self):
        super().__init__("Metrics")
        self.metrics = Metrics()
        self._request_start_times: Dict[str, float] = {}
        self._mode_metrics: Dict[str, Metrics] = defaultdict(Metrics)
    
    def _before(
        self,
        state: CognitiveState,
        request: CognitiveInput,
    ) -> tuple[CognitiveState, CognitiveInput]:
        """Record request start"""
        request_id = f"{id(request)}_{time.time()}"
        self._request_start_times[request_id] = time.time()
        
        # Store request_id in request metadata
        request.metadata['_request_id'] = request_id
        request.metadata['_start_time'] = time.time()
        
        self.metrics.request_count += 1
        self.metrics.last_request_time = datetime.now()
        
        return state, request
    
    def _after(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        response: CognitiveOutput,
    ) -> CognitiveOutput:
        """Record request completion"""
        request_id = None
        if hasattr(state, '_request_metadata'):
            request_id = state._request_metadata.get('_request_id')
        
        if request_id and request_id in self._request_start_times:
            start_time = self._request_start_times.pop(request_id)
            latency = time.time() - start_time
            
            # Update global metrics
            self.metrics.success_count += 1
            self.metrics.total_latency += latency
            self.metrics.min_latency = min(self.metrics.min_latency, latency)
            self.metrics.max_latency = max(self.metrics.max_latency, latency)
            
            # Update mode-specific metrics
            mode = response.used_mode
            mode_metrics = self._mode_metrics[mode]
            mode_metrics.request_count += 1
            mode_metrics.success_count += 1
            mode_metrics.total_latency += latency
            mode_metrics.min_latency = min(mode_metrics.min_latency, latency)
            mode_metrics.max_latency = max(mode_metrics.max_latency, latency)
        
        return response
    
    def _on_error(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        error: Exception,
    ) -> Optional[CognitiveOutput]:
        """Record error"""
        request_id = request.metadata.get('_request_id')
        if request_id and request_id in self._request_start_times:
            self._request_start_times.pop(request_id)
        
        self.metrics.error_count += 1
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            "global": {
                "request_count": self.metrics.request_count,
                "success_count": self.metrics.success_count,
                "error_count": self.metrics.error_count,
                "avg_latency": self.metrics.avg_latency,
                "min_latency": self.metrics.min_latency if self.metrics.min_latency != float('inf') else 0.0,
                "max_latency": self.metrics.max_latency,
                "success_rate": self.metrics.success_rate,
            },
            "by_mode": {
                mode: {
                    "request_count": metrics.request_count,
                    "success_count": metrics.success_count,
                    "avg_latency": metrics.avg_latency,
                }
                for mode, metrics in self._mode_metrics.items()
            },
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self.metrics = Metrics()
        self._mode_metrics.clear()
        self._request_start_times.clear()


__all__ = ["MetricsMiddleware", "Metrics"]

