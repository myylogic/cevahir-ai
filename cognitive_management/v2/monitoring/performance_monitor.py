# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: performance_monitor.py
Modül: cognitive_management/v2/monitoring
Görev: Performance Monitor - Enterprise-grade performance monitoring for V2
       Cognitive Management. Phase 5.4: Advanced monitoring with detailed
       performance metrics. PerformanceMetrics, PerformanceMonitor sınıflarını
       içerir. Performance metrics collection, latency tracking, throughput
       monitoring ve performance analysis işlemlerini yapar. Industry standard:
       Prometheus metrics.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (performance monitoring)
- Design Patterns: Monitor Pattern (performance monitoring)
- Endüstri Standartları: Performance monitoring best practices

KULLANIM:
- Performance metrics collection için
- Latency tracking için
- Throughput monitoring için
- Performance analysis için

BAĞIMLILIKLAR:
- threading: Thread-safe işlemler
- dataclasses: Performance metrics data structures

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import time
import threading
from collections import deque, defaultdict


# =============================================================================
# Performance Metrics
# =============================================================================

@dataclass
class PerformanceMetrics:
    """
    Performance metrics for a component or operation.
    
    Industry standard: Prometheus metrics
    """
    name: str
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency: float = 0.0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    last_request_time: Optional[float] = None
    latency_samples: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def record_request(
        self,
        latency: float,
        success: bool = True,
    ) -> None:
        """
        Record a request.
        
        Args:
            latency: Request latency in seconds
            success: Whether request was successful
        """
        self.request_count += 1
        self.last_request_time = time.time()
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        # Update latency statistics
        self.total_latency += latency
        self.min_latency = min(self.min_latency, latency)
        self.max_latency = max(self.max_latency, latency)
        
        # Store sample for percentile calculation
        self.latency_samples.append(latency)
        
        # Calculate percentiles
        if self.latency_samples:
            sorted_samples = sorted(self.latency_samples)
            n = len(sorted_samples)
            self.p50_latency = sorted_samples[int(n * 0.50)]
            self.p95_latency = sorted_samples[int(n * 0.95)]
            self.p99_latency = sorted_samples[int(n * 0.99)]
    
    @property
    def avg_latency(self) -> float:
        """Average latency"""
        if self.request_count == 0:
            return 0.0
        return self.total_latency / self.request_count
    
    @property
    def success_rate(self) -> float:
        """Success rate (0.0-1.0)"""
        if self.request_count == 0:
            return 0.0
        return self.success_count / self.request_count
    
    @property
    def error_rate(self) -> float:
        """Error rate (0.0-1.0)"""
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "request_count": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "avg_latency": self.avg_latency,
            "min_latency": self.min_latency if self.min_latency != float('inf') else 0.0,
            "max_latency": self.max_latency,
            "p50_latency": self.p50_latency,
            "p95_latency": self.p95_latency,
            "p99_latency": self.p99_latency,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "last_request_time": self.last_request_time,
        }
    
    def reset(self) -> None:
        """Reset metrics"""
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        self.min_latency = float('inf')
        self.max_latency = 0.0
        self.p50_latency = 0.0
        self.p95_latency = 0.0
        self.p99_latency = 0.0
        self.last_request_time = None
        self.latency_samples.clear()


# =============================================================================
# Performance Monitor
# =============================================================================

class PerformanceMonitor:
    """
    Performance monitor for cognitive management system.
    
    Phase 5.4: Enterprise-grade performance monitoring.
    
    Features:
    - Component-level metrics
    - Operation-level metrics
    - Latency percentiles
    - Success/error rates
    - Performance history
    """
    
    def __init__(self):
        """Initialize performance monitor"""
        self._metrics: Dict[str, PerformanceMetrics] = {}
        self._lock = threading.RLock()
    
    def get_or_create_metrics(self, name: str) -> PerformanceMetrics:
        """
        Get or create metrics for component/operation.
        
        Args:
            name: Component/operation name
            
        Returns:
            PerformanceMetrics instance
        """
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = PerformanceMetrics(name=name)
            return self._metrics[name]
    
    def record_operation(
        self,
        name: str,
        latency: float,
        success: bool = True,
    ) -> None:
        """
        Record an operation.
        
        Args:
            name: Operation name
            latency: Operation latency in seconds
            success: Whether operation was successful
        """
        metrics = self.get_or_create_metrics(name)
        metrics.record_request(latency, success)
    
    def get_metrics(self, name: str) -> Optional[PerformanceMetrics]:
        """
        Get metrics for component/operation.
        
        Args:
            name: Component/operation name
            
        Returns:
            PerformanceMetrics or None
        """
        with self._lock:
            return self._metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Get all metrics"""
        with self._lock:
            return self._metrics.copy()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary.
        
        Returns:
            Summary dictionary
        """
        with self._lock:
            all_metrics = list(self._metrics.values())
            
            if not all_metrics:
                return {
                    "total_components": 0,
                    "total_requests": 0,
                    "total_success": 0,
                    "total_errors": 0,
                }
            
            total_requests = sum(m.request_count for m in all_metrics)
            total_success = sum(m.success_count for m in all_metrics)
            total_errors = sum(m.error_count for m in all_metrics)
            
            avg_latencies = [m.avg_latency for m in all_metrics if m.request_count > 0]
            overall_avg_latency = sum(avg_latencies) / len(avg_latencies) if avg_latencies else 0.0
            
            return {
                "total_components": len(all_metrics),
                "total_requests": total_requests,
                "total_success": total_success,
                "total_errors": total_errors,
                "overall_success_rate": total_success / total_requests if total_requests > 0 else 0.0,
                "overall_error_rate": total_errors / total_requests if total_requests > 0 else 0.0,
                "overall_avg_latency": overall_avg_latency,
                "components": {
                    name: metrics.to_dict()
                    for name, metrics in self._metrics.items()
                },
            }
    
    def reset_metrics(self, name: Optional[str] = None) -> None:
        """
        Reset metrics.
        
        Args:
            name: Component name (None = reset all)
        """
        with self._lock:
            if name:
                if name in self._metrics:
                    self._metrics[name].reset()
            else:
                for metrics in self._metrics.values():
                    metrics.reset()


__all__ = ["PerformanceMonitor", "PerformanceMetrics"]

