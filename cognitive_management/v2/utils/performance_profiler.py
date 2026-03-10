# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: performance_profiler.py
Modül: cognitive_management/v2/utils
Görev: Performance Profiler - Performance profiling utilities for bottleneck
       identification. Phase 6: Performance Optimization & Caching Enhancement.
       ProfileEntry, PerformanceProfiler sınıflarını içerir. Performance profiling,
       bottleneck identification, cProfile integration ve profiling analysis
       işlemlerini yapar. Akademik referans: Profiling Best Practices, Bottleneck
       Analysis.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (performance profiling)
- Design Patterns: Profiler Pattern (performance profiling)
- Endüstri Standartları: Performance profiling best practices

KULLANIM:
- Performance profiling için
- Bottleneck identification için
- cProfile integration için
- Profiling analysis için

BAĞIMLILIKLAR:
- cProfile: Python profiling
- pstats: Profile statistics
- Cache: Cache işlemleri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
import threading
import cProfile
import pstats
import io
from contextlib import contextmanager

from .cache import Cache


@dataclass
class ProfileEntry:
    """
    Performance profile entry.
    
    Attributes:
        operation_name: Name of the operation
        duration: Duration in seconds
        start_time: Start timestamp
        end_time: End timestamp
        metadata: Optional metadata
    """
    operation_name: str
    duration: float
    start_time: float
    end_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """
    Performance profiler for bottleneck identification.
    
    Features:
    - Operation-level profiling
    - Call graph profiling
    - Bottleneck detection
    - Performance reports
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize performance profiler.
        
        Args:
            enabled: Enable/disable profiling
        """
        self.enabled = enabled
        self._lock = threading.RLock()
        
        # Profile entries
        self._entries: List[ProfileEntry] = []
        self._operation_stats: Dict[str, List[float]] = {}  # operation -> [durations]
        
        # Active profilers (for nested profiling)
        self._active_profilers: Dict[str, cProfile.Profile] = {}
    
    @contextmanager
    def profile_operation(
        self,
        operation_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for profiling an operation.
        
        Args:
            operation_name: Name of the operation
            metadata: Optional metadata
            
        Usage:
            with profiler.profile_operation("my_operation"):
                # code to profile
                pass
        """
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        metadata = metadata or {}
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            entry = ProfileEntry(
                operation_name=operation_name,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                metadata=metadata,
            )
            
            with self._lock:
                self._entries.append(entry)
                
                # Update stats
                if operation_name not in self._operation_stats:
                    self._operation_stats[operation_name] = []
                self._operation_stats[operation_name].append(duration)
    
    def profile_function(
        self,
        operation_name: Optional[str] = None
    ) -> Callable:
        """
        Decorator for profiling functions.
        
        Args:
            operation_name: Operation name (default: function name)
            
        Usage:
            @profiler.profile_function("my_function")
            def my_function():
                pass
        """
        def decorator(func: Callable) -> Callable:
            op_name = operation_name or func.__name__
            
            def wrapper(*args, **kwargs):
                with self.profile_operation(op_name):
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def get_operation_stats(
        self,
        operation_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get statistics for an operation.
        
        Args:
            operation_name: Operation name
            
        Returns:
            Statistics dictionary or None
        """
        with self._lock:
            if operation_name not in self._operation_stats:
                return None
            
            durations = self._operation_stats[operation_name]
            
            if not durations:
                return None
            
            durations_sorted = sorted(durations)
            n = len(durations)
            
            return {
                "operation_name": operation_name,
                "call_count": n,
                "total_time": sum(durations),
                "avg_time": sum(durations) / n,
                "min_time": durations_sorted[0],
                "max_time": durations_sorted[-1],
                "p50_time": durations_sorted[int(n * 0.50)] if n > 0 else 0.0,
                "p95_time": durations_sorted[int(n * 0.95)] if n > 0 else 0.0,
                "p99_time": durations_sorted[int(n * 0.99)] if n > 0 else 0.0,
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all operations."""
        with self._lock:
            return {
                op_name: self.get_operation_stats(op_name)
                for op_name in self._operation_stats.keys()
            }
    
    def identify_bottlenecks(
        self,
        threshold_percentile: float = 0.95
    ) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks.
        
        Args:
            threshold_percentile: Percentile threshold for bottleneck (default: 95th)
            
        Returns:
            List of bottleneck dictionaries
        """
        with self._lock:
            bottlenecks = []
            
            for op_name in self._operation_stats.keys():
                stats = self.get_operation_stats(op_name)
                if not stats:
                    continue
                
                # Check if operation exceeds threshold
                p95 = stats.get("p95_time", 0.0)
                avg = stats.get("avg_time", 0.0)
                
                if p95 > avg * 2.0:  # p95 is 2x average - potential bottleneck
                    bottlenecks.append({
                        "operation": op_name,
                        "p95_time": p95,
                        "avg_time": avg,
                        "ratio": p95 / avg if avg > 0 else 0.0,
                        "severity": "high" if p95 / avg > 3.0 else "medium",
                    })
            
            # Sort by severity
            bottlenecks.sort(key=lambda x: x["ratio"], reverse=True)
            
            return bottlenecks
    
    def get_profile_report(
        self,
        format: str = "summary"  # "summary" | "detailed" | "bottlenecks"
    ) -> str:
        """
        Get profiling report.
        
        Args:
            format: Report format
            
        Returns:
            Report string
        """
        with self._lock:
            if format == "summary":
                return self._generate_summary_report()
            elif format == "detailed":
                return self._generate_detailed_report()
            elif format == "bottlenecks":
                return self._generate_bottlenecks_report()
            else:
                return self._generate_summary_report()
    
    def _generate_summary_report(self) -> str:
        """Generate summary report."""
        all_stats = self.get_all_stats()
        bottlenecks = self.identify_bottlenecks()
        
        lines = [
            "=== Performance Profile Summary ===",
            f"Total Operations: {len(all_stats)}",
            f"Bottlenecks Found: {len(bottlenecks)}",
            "",
            "Top Operations by Avg Time:",
        ]
        
        # Sort by avg time
        sorted_ops = sorted(
            all_stats.items(),
            key=lambda x: x[1]["avg_time"] if x[1] else 0.0,
            reverse=True
        )[:10]
        
        for op_name, stats in sorted_ops:
            if stats:
                lines.append(
                    f"  {op_name}: avg={stats['avg_time']:.3f}s, "
                    f"p95={stats['p95_time']:.3f}s, calls={stats['call_count']}"
                )
        
        return "\n".join(lines)
    
    def _generate_detailed_report(self) -> str:
        """Generate detailed report."""
        all_stats = self.get_all_stats()
        lines = ["=== Detailed Performance Profile ===", ""]
        
        for op_name, stats in sorted(all_stats.items()):
            if stats:
                lines.append(f"Operation: {op_name}")
                lines.append(f"  Calls: {stats['call_count']}")
                lines.append(f"  Total Time: {stats['total_time']:.3f}s")
                lines.append(f"  Avg Time: {stats['avg_time']:.3f}s")
                lines.append(f"  Min Time: {stats['min_time']:.3f}s")
                lines.append(f"  Max Time: {stats['max_time']:.3f}s")
                lines.append(f"  P50: {stats['p50_time']:.3f}s")
                lines.append(f"  P95: {stats['p95_time']:.3f}s")
                lines.append(f"  P99: {stats['p99_time']:.3f}s")
                lines.append("")
        
        return "\n".join(lines)
    
    def _generate_bottlenecks_report(self) -> str:
        """Generate bottlenecks report."""
        bottlenecks = self.identify_bottlenecks()
        lines = ["=== Performance Bottlenecks ===", ""]
        
        if not bottlenecks:
            lines.append("No bottlenecks identified.")
        else:
            for i, bottleneck in enumerate(bottlenecks, 1):
                lines.append(f"{i}. {bottleneck['operation']}")
                lines.append(f"   P95 Time: {bottleneck['p95_time']:.3f}s")
                lines.append(f"   Avg Time: {bottleneck['avg_time']:.3f}s")
                lines.append(f"   Ratio: {bottleneck['ratio']:.2f}x")
                lines.append(f"   Severity: {bottleneck['severity']}")
                lines.append("")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all profile data."""
        with self._lock:
            self._entries.clear()
            self._operation_stats.clear()


__all__ = [
    "PerformanceProfiler",
    "ProfileEntry",
]

