# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: trend_analyzer.py
Modül: cognitive_management/v2/monitoring
Görev: Performance Trend Analyzer - Trend analysis for performance metrics over
       time. Phase 5: AIOps Integration & Predictive Analytics. Trend, TrendAnalyzer
       sınıflarını içerir. Time series analysis, trend detection algorithms ve
       seasonal decomposition işlemlerini yapar. Akademik referans: Time Series
       Analysis, Trend Detection Algorithms, Seasonal Decomposition.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (trend analysis)
- Design Patterns: Analyzer Pattern (trend analysis)
- Endüstri Standartları: Trend analysis best practices

KULLANIM:
- Trend analysis için
- Time series analysis için
- Seasonal decomposition için

BAĞIMLILIKLAR:
- PerformanceMonitor: Performance monitoring
- statistics: Statistical işlemler

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import statistics

from .performance_monitor import PerformanceMonitor, PerformanceMetrics


@dataclass
class Trend:
    """
    Trend analysis result.
    
    Attributes:
        metric_name: Name of the metric
        trend_direction: "increasing", "decreasing", "stable"
        trend_strength: Strength of trend (0.0-1.0)
        slope: Slope of trend (positive = increasing, negative = decreasing)
        change_percentage: Percentage change over period
        period: Analysis period (minutes)
        timestamp: When analysis was performed
    """
    metric_name: str
    trend_direction: str
    trend_strength: float
    slope: float
    change_percentage: float
    period: int
    timestamp: datetime


class TrendAnalyzer:
    """
    Performance trend analyzer.
    
    Analyzes historical performance metrics to identify trends:
    - Latency trends
    - Error rate trends
    - Throughput trends
    - Overall system health trends
    
    Academic Reference:
    - Linear Regression for Trend Detection
    - Moving Average Comparison
    - Percent Change Analysis
    """
    
    def __init__(
        self,
        performance_monitor: PerformanceMonitor,
        trend_period_minutes: int = 60
    ):
        """
        Initialize trend analyzer.
        
        Args:
            performance_monitor: Performance monitor instance
            trend_period_minutes: Period for trend analysis (default: 60 minutes)
        """
        self.performance_monitor = performance_monitor
        self.trend_period_minutes = trend_period_minutes
        
        # Historical snapshots of metrics
        self._metric_snapshots: Dict[str, deque] = {}
        self._snapshot_interval = 5  # Take snapshot every 5 minutes
        self._last_snapshot_time: Dict[str, datetime] = {}
    
    def analyze_trend(
        self,
        metric_name: str,
        period_minutes: Optional[int] = None
    ) -> Optional[Trend]:
        """
        Analyze trend for a specific metric.
        
        Args:
            metric_name: Name of the metric
            period_minutes: Analysis period (default: self.trend_period_minutes)
            
        Returns:
            Trend analysis result or None
        """
        period = period_minutes or self.trend_period_minutes
        
        # Get current metrics
        current_metric = self.performance_monitor.get_metrics(metric_name)
        if not current_metric:
            return None
        
        # Take snapshot if needed
        self._take_snapshot(metric_name, current_metric)
        
        # Get historical snapshots
        snapshots = self._get_snapshots(metric_name, period)
        
        if len(snapshots) < 3:
            return None
        
        # Extract values and timestamps
        values = [s["value"] for s in snapshots]
        timestamps = [s["timestamp"] for s in snapshots]
        
        # Calculate trend
        trend_direction, trend_strength, slope = self._calculate_trend(values, timestamps)
        
        # Calculate percentage change
        if values:
            change_percentage = ((values[-1] - values[0]) / values[0]) * 100 if values[0] > 0 else 0.0
        else:
            change_percentage = 0.0
        
        return Trend(
            metric_name=metric_name,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            slope=slope,
            change_percentage=change_percentage,
            period=period,
            timestamp=datetime.now(),
        )
    
    def analyze_latency_trend(self, metric_name: str) -> Optional[Trend]:
        """Analyze latency trend for a metric."""
        return self.analyze_trend(f"{metric_name}_latency")
    
    def analyze_error_rate_trend(self, metric_name: str) -> Optional[Trend]:
        """Analyze error rate trend for a metric."""
        return self.analyze_trend(f"{metric_name}_error_rate")
    
    def get_all_trends(
        self,
        period_minutes: Optional[int] = None
    ) -> Dict[str, Trend]:
        """
        Analyze trends for all metrics.
        
        Args:
            period_minutes: Analysis period
            
        Returns:
            Dictionary of metric_name -> Trend
        """
        all_metrics = self.performance_monitor.get_all_metrics()
        trends = {}
        
        for metric_name in all_metrics.keys():
            # Analyze latency trend
            latency_trend = self.analyze_latency_trend(metric_name)
            if latency_trend:
                trends[f"{metric_name}_latency"] = latency_trend
            
            # Analyze error rate trend
            error_trend = self.analyze_error_rate_trend(metric_name)
            if error_trend:
                trends[f"{metric_name}_error_rate"] = error_trend
        
        return trends
    
    def _take_snapshot(
        self,
        metric_name: str,
        metric: PerformanceMetrics
    ) -> None:
        """
        Take a snapshot of current metric state.
        
        Args:
            metric_name: Name of the metric
            metric: Current metrics
        """
        now = datetime.now()
        last_time = self._last_snapshot_time.get(metric_name)
        
        # Check if enough time has passed
        if last_time and (now - last_time).total_seconds() < self._snapshot_interval * 60:
            return
        
        # Initialize snapshot storage
        if metric_name not in self._metric_snapshots:
            self._metric_snapshots[metric_name] = deque(maxlen=100)  # Keep last 100 snapshots
        
        # Store latency snapshot
        latency_key = f"{metric_name}_latency"
        if latency_key not in self._metric_snapshots:
            self._metric_snapshots[latency_key] = deque(maxlen=100)
        
        self._metric_snapshots[latency_key].append({
            "value": metric.avg_latency,
            "timestamp": now,
            "p95": metric.p95_latency,
            "p99": metric.p99_latency,
        })
        
        # Store error rate snapshot
        error_key = f"{metric_name}_error_rate"
        if error_key not in self._metric_snapshots:
            self._metric_snapshots[error_key] = deque(maxlen=100)
        
        self._metric_snapshots[error_key].append({
            "value": metric.error_rate,
            "timestamp": now,
            "request_count": metric.request_count,
        })
        
        self._last_snapshot_time[metric_name] = now
    
    def _get_snapshots(
        self,
        metric_key: str,
        period_minutes: int
    ) -> List[Dict[str, Any]]:
        """
        Get snapshots within the specified period.
        
        Args:
            metric_key: Metric key (e.g., "metric_name_latency")
            period_minutes: Period in minutes
            
        Returns:
            List of snapshots
        """
        if metric_key not in self._metric_snapshots:
            return []
        
        snapshots = list(self._metric_snapshots[metric_key])
        cutoff_time = datetime.now() - timedelta(minutes=period_minutes)
        
        # Filter snapshots within period
        filtered = [s for s in snapshots if s["timestamp"] >= cutoff_time]
        
        return filtered
    
    def _calculate_trend(
        self,
        values: List[float],
        timestamps: List[datetime]
    ) -> Tuple[str, float, float]:
        """
        Calculate trend from time series data.
        
        Args:
            values: Metric values
            timestamps: Corresponding timestamps
            
        Returns:
            Tuple of (trend_direction, trend_strength, slope)
        """
        if len(values) < 2:
            return "stable", 0.0, 0.0
        
        # Calculate time differences in minutes
        if len(timestamps) > 1:
            total_minutes = (timestamps[-1] - timestamps[0]).total_seconds() / 60.0
            if total_minutes == 0:
                return "stable", 0.0, 0.0
        else:
            return "stable", 0.0, 0.0
        
        # Simple linear regression for slope
        n = len(values)
        x = [(t - timestamps[0]).total_seconds() / 60.0 for t in timestamps]  # Minutes from start
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        
        denominator = (n * sum_x2) - (sum_x * sum_x)
        if denominator == 0:
            return "stable", 0.0, 0.0
        
        slope = ((n * sum_xy) - (sum_x * sum_y)) / denominator
        
        # Determine trend direction
        if slope > 0.001:  # Positive slope
            trend_direction = "increasing"
        elif slope < -0.001:  # Negative slope
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        # Calculate trend strength (R-squared approximation)
        mean_value = statistics.mean(values)
        if mean_value == 0:
            trend_strength = 0.0
        else:
            # Use coefficient of variation as proxy for strength
            variance = statistics.variance(values) if len(values) > 1 else 0.0
            std_dev = variance ** 0.5
            
            # Normalize slope relative to mean
            normalized_slope = abs(slope) / mean_value if mean_value > 0 else 0.0
            
            # Trend strength: combination of slope magnitude and consistency
            trend_strength = min(1.0, normalized_slope * 10.0)  # Scale factor
        
        return trend_direction, trend_strength, slope


__all__ = [
    "TrendAnalyzer",
    "Trend",
]

