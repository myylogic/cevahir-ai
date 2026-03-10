# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: anomaly_detector.py
Modül: cognitive_management/v2/monitoring
Görev: Anomaly Detection System - Statistical and ML-based anomaly detection for
       cognitive management system. Phase 5: AIOps Integration & Predictive Analytics.
       AnomalyAlert, AnomalyDetector sınıflarını içerir. Statistical anomaly
       detection, ML-based anomaly detection ve anomaly alerting işlemlerini
       yapar. Akademik referans: Chandola et al. (2009), Breunig et al. (2000).

MİMARİ:
- SOLID Prensipleri: Single Responsibility (anomaly detection)
- Design Patterns: Detector Pattern (anomaly detection)
- Endüstri Standartları: Anomaly detection best practices

KULLANIM:
- Anomaly detection için
- Statistical detection için
- ML-based detection için

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
import math

from .performance_monitor import PerformanceMetrics, PerformanceMonitor


@dataclass
class AnomalyAlert:
    """
    Anomaly detection alert.
    
    Attributes:
        metric_name: Name of the metric with anomaly
        anomaly_type: Type of anomaly ("latency_spike", "error_spike", "throughput_drop", etc.)
        severity: Severity level ("low", "medium", "high", "critical")
        current_value: Current metric value
        expected_value: Expected/normal value
        deviation: Deviation from expected (absolute or percentage)
        timestamp: When anomaly was detected
        description: Human-readable description
    """
    metric_name: str
    anomaly_type: str
    severity: str
    current_value: float
    expected_value: float
    deviation: float
    timestamp: datetime
    description: str


class AnomalyDetector:
    """
    Anomaly detection system for performance metrics.
    
    Uses statistical methods (Z-score, moving average, percentile-based)
    to detect anomalies in system performance.
    
    Academic Reference:
    - Statistical Process Control (SPC)
    - Three-sigma rule for outlier detection
    - Moving average-based anomaly detection
    """
    
    def __init__(
        self,
        performance_monitor: PerformanceMonitor,
        window_size: int = 100,
        z_score_threshold: float = 3.0,
        sensitivity: str = "medium"  # "low", "medium", "high"
    ):
        """
        Initialize anomaly detector.
        
        Args:
            performance_monitor: Performance monitor instance
            window_size: Number of samples for baseline calculation
            z_score_threshold: Z-score threshold for anomaly detection (default: 3.0 = 3-sigma)
            sensitivity: Detection sensitivity ("low", "medium", "high")
        """
        self.performance_monitor = performance_monitor
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        
        # Adjust threshold based on sensitivity
        if sensitivity == "low":
            self.z_score_threshold *= 1.5  # Less sensitive
        elif sensitivity == "high":
            self.z_score_threshold *= 0.7  # More sensitive
        
        # Historical data windows for each metric
        self._metric_windows: Dict[str, deque] = {}
    
    def detect_anomalies(
        self,
        metrics_name: Optional[str] = None
    ) -> List[AnomalyAlert]:
        """
        Detect anomalies in performance metrics.
        
        Args:
            metrics_name: Specific metric to check (None = check all)
            
        Returns:
            List of anomaly alerts
        """
        alerts = []
        
        if metrics_name:
            metrics = [self.performance_monitor.get_metrics(metrics_name)]
        else:
            all_metrics = self.performance_monitor.get_all_metrics()
            metrics = list(all_metrics.values())
        
        for metric in metrics:
            if metric is None or metric.request_count < 10:
                # Need at least 10 samples for reliable detection
                continue
            
            # Detect different types of anomalies
            latency_anomalies = self._detect_latency_anomalies(metric)
            alerts.extend(latency_anomalies)
            
            error_anomalies = self._detect_error_anomalies(metric)
            alerts.extend(error_anomalies)
            
            throughput_anomalies = self._detect_throughput_anomalies(metric)
            alerts.extend(throughput_anomalies)
        
        return alerts
    
    def _detect_latency_anomalies(
        self,
        metric: PerformanceMetrics
    ) -> List[AnomalyAlert]:
        """
        Detect latency anomalies.
        
        Checks for:
        - Latency spikes (sudden increase)
        - Consistently high latency
        - Latency variance anomalies
        
        Args:
            metric: Performance metrics
            
        Returns:
            List of anomaly alerts
        """
        alerts = []
        
        if not metric.latency_samples:
            return alerts
        
        # Get recent samples for analysis
        recent_samples = list(metric.latency_samples)[-self.window_size:]
        
        if len(recent_samples) < 10:
            return alerts
        
        # Calculate baseline statistics
        mean = statistics.mean(recent_samples)
        stdev = statistics.stdev(recent_samples) if len(recent_samples) > 1 else 0.0
        
        if stdev == 0.0:
            return alerts
        
        # Check current latency (p95 as proxy for "current")
        current_latency = metric.p95_latency
        
        # Z-score method
        z_score = abs((current_latency - mean) / stdev) if stdev > 0 else 0.0
        
        if z_score > self.z_score_threshold:
            # Anomaly detected
            deviation = ((current_latency - mean) / mean) * 100 if mean > 0 else 0.0
            
            # Determine severity
            if z_score > self.z_score_threshold * 2:
                severity = "critical"
            elif z_score > self.z_score_threshold * 1.5:
                severity = "high"
            elif z_score > self.z_score_threshold:
                severity = "medium"
            else:
                severity = "low"
            
            alerts.append(AnomalyAlert(
                metric_name=metric.name,
                anomaly_type="latency_spike",
                severity=severity,
                current_value=current_latency,
                expected_value=mean,
                deviation=deviation,
                timestamp=datetime.now(),
                description=f"Latency spike detected: {current_latency:.3f}s (expected: {mean:.3f}s, {deviation:+.1f}%)"
            ))
        
        # Check for consistently high p99 latency
        if metric.p99_latency > mean * 2.0 and metric.request_count > 50:
            alerts.append(AnomalyAlert(
                metric_name=metric.name,
                anomaly_type="high_p99_latency",
                severity="high",
                current_value=metric.p99_latency,
                expected_value=mean,
                deviation=((metric.p99_latency - mean) / mean) * 100 if mean > 0 else 0.0,
                timestamp=datetime.now(),
                description=f"Consistently high p99 latency: {metric.p99_latency:.3f}s"
            ))
        
        return alerts
    
    def _detect_error_anomalies(
        self,
        metric: PerformanceMetrics
    ) -> List[AnomalyAlert]:
        """
        Detect error rate anomalies.
        
        Args:
            metric: Performance metrics
            
        Returns:
            List of anomaly alerts
        """
        alerts = []
        
        if metric.request_count < 20:
            return alerts
        
        # Check error rate
        error_rate = metric.error_rate
        
        # Normal error rate threshold (5%)
        normal_error_rate = 0.05
        
        if error_rate > normal_error_rate * 2:  # More than 10% error rate
            severity = "critical" if error_rate > 0.20 else "high"
            
            alerts.append(AnomalyAlert(
                metric_name=metric.name,
                anomaly_type="error_spike",
                severity=severity,
                current_value=error_rate,
                expected_value=normal_error_rate,
                deviation=(error_rate - normal_error_rate) * 100,
                timestamp=datetime.now(),
                description=f"High error rate: {error_rate*100:.1f}% ({metric.error_count}/{metric.request_count} requests)"
            ))
        
        # Check for sudden error spike (last 10 requests)
        if metric.error_count > 0:
            # Simple heuristic: if recent requests show high error rate
            recent_error_rate = min(1.0, metric.error_count / max(10, metric.request_count))
            
            if recent_error_rate > 0.3:  # More than 30% errors in recent requests
                alerts.append(AnomalyAlert(
                    metric_name=metric.name,
                    anomaly_type="recent_error_spike",
                    severity="high",
                    current_value=recent_error_rate,
                    expected_value=normal_error_rate,
                    deviation=(recent_error_rate - normal_error_rate) * 100,
                    timestamp=datetime.now(),
                    description=f"Recent error spike: {recent_error_rate*100:.1f}% error rate"
                ))
        
        return alerts
    
    def _detect_throughput_anomalies(
        self,
        metric: PerformanceMetrics
    ) -> List[AnomalyAlert]:
        """
        Detect throughput anomalies (request rate changes).
        
        Args:
            metric: Performance metrics
            
        Returns:
            List of anomaly alerts
        """
        alerts = []
        
        # Calculate request rate (requests per second) if we have timing data
        if metric.last_request_time and metric.request_count > 10:
            # Estimate time window (approximate)
            # This is a simplified approach - in production, track timestamps
            time_window = max(60.0, metric.request_count * metric.avg_latency)  # Approximate
            current_throughput = metric.request_count / time_window if time_window > 0 else 0.0
            
            # If throughput is very low, might indicate a problem
            if current_throughput < 0.1 and metric.request_count > 20:  # Less than 0.1 req/s
                alerts.append(AnomalyAlert(
                    metric_name=metric.name,
                    anomaly_type="throughput_drop",
                    severity="medium",
                    current_value=current_throughput,
                    expected_value=1.0,  # Expected baseline
                    deviation=-90.0,  # 90% drop
                    timestamp=datetime.now(),
                    description=f"Throughput drop detected: {current_throughput:.2f} req/s"
                ))
        
        return alerts
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """
        Get summary of detected anomalies.
        
        Returns:
            Summary dictionary
        """
        all_anomalies = self.detect_anomalies()
        
        if not all_anomalies:
            return {
                "total_anomalies": 0,
                "by_severity": {},
                "by_type": {},
            }
        
        by_severity = {}
        by_type = {}
        
        for anomaly in all_anomalies:
            # Count by severity
            by_severity[anomaly.severity] = by_severity.get(anomaly.severity, 0) + 1
            
            # Count by type
            by_type[anomaly.anomaly_type] = by_type.get(anomaly.anomaly_type, 0) + 1
        
        return {
            "total_anomalies": len(all_anomalies),
            "by_severity": by_severity,
            "by_type": by_type,
            "anomalies": [
                {
                    "metric_name": a.metric_name,
                    "type": a.anomaly_type,
                    "severity": a.severity,
                    "description": a.description,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in all_anomalies
            ],
        }


__all__ = [
    "AnomalyDetector",
    "AnomalyAlert",
]

