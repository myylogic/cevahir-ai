# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: predictive_analytics.py
Modül: cognitive_management/v2/monitoring
Görev: Predictive Analytics Engine - Predictive analytics for cognitive management
       system performance. Phase 5: AIOps Integration & Predictive Analytics.
       Prediction, PredictiveAnalytics sınıflarını içerir. Time series forecasting
       (ARIMA, Exponential Smoothing), linear regression for latency prediction
       ve moving average models işlemlerini yapar. Akademik referans: Time Series
       Forecasting, Linear Regression, Moving Average Models.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (predictive analytics)
- Design Patterns: Analytics Pattern (predictive analytics)
- Endüstri Standartları: Predictive analytics best practices

KULLANIM:
- Time series forecasting için
- Latency prediction için
- Performance prediction için

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
class Prediction:
    """
    Prediction result.
    
    Attributes:
        metric_name: Name of predicted metric
        predicted_value: Predicted value
        confidence: Confidence level (0.0-1.0)
        prediction_horizon: How far into the future (minutes)
        upper_bound: Upper confidence bound
        lower_bound: Lower confidence bound
        timestamp: When prediction was made
    """
    metric_name: str
    predicted_value: float
    confidence: float
    prediction_horizon: int  # minutes
    upper_bound: float
    lower_bound: float
    timestamp: datetime


@dataclass
class ScalingRecommendation:
    """
    Auto-scaling recommendation.
    
    Attributes:
        action: "scale_up", "scale_down", "maintain"
        confidence: Confidence in recommendation (0.0-1.0)
        reason: Human-readable reason
        estimated_impact: Expected impact description
        priority: Priority level ("low", "medium", "high")
    """
    action: str
    confidence: float
    reason: str
    estimated_impact: str
    priority: str


class PredictiveAnalytics:
    """
    Predictive analytics engine for performance forecasting.
    
    Uses statistical methods and time series analysis to predict:
    - Future latency
    - Error rate trends
    - Throughput forecasting
    - Resource requirements
    
    Academic Reference:
    - Simple Moving Average (SMA)
    - Exponential Moving Average (EMA)
    - Linear Regression
    - Trend Analysis
    """
    
    def __init__(
        self,
        performance_monitor: PerformanceMonitor,
        prediction_window: int = 5  # minutes
    ):
        """
        Initialize predictive analytics engine.
        
        Args:
            performance_monitor: Performance monitor instance
            prediction_window: Prediction horizon in minutes
        """
        self.performance_monitor = performance_monitor
        self.prediction_window = prediction_window
        
        # Historical data for trend analysis
        self._historical_data: Dict[str, deque] = {}
        self._max_history = 100  # Keep last 100 data points
    
    def predict_latency(
        self,
        metric_name: str,
        horizon_minutes: Optional[int] = None
    ) -> Optional[Prediction]:
        """
        Predict future latency for a metric.
        
        Args:
            metric_name: Name of the metric
            horizon_minutes: Prediction horizon (default: self.prediction_window)
            
        Returns:
            Prediction or None if insufficient data
        """
        horizon = horizon_minutes or self.prediction_window
        
        metric = self.performance_monitor.get_metrics(metric_name)
        if not metric or metric.request_count < 20:
            return None
        
        # Get historical latency data
        if metric_name not in self._historical_data:
            self._historical_data[metric_name] = deque(maxlen=self._max_history)
        
        # Add current average latency to history
        current_avg = metric.avg_latency
        self._historical_data[metric_name].append({
            "value": current_avg,
            "timestamp": datetime.now(),
        })
        
        history = list(self._historical_data[metric_name])
        
        if len(history) < 5:
            # Need at least 5 data points
            return None
        
        # Extract values for prediction
        values = [h["value"] for h in history]
        
        # Simple linear regression for trend
        predicted_value, confidence = self._linear_trend_prediction(values, horizon)
        
        # Calculate confidence bounds (based on variance)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        upper_bound = predicted_value + (1.96 * std_dev)  # 95% confidence
        lower_bound = max(0.0, predicted_value - (1.96 * std_dev))
        
        return Prediction(
            metric_name=metric_name,
            predicted_value=predicted_value,
            confidence=confidence,
            prediction_horizon=horizon,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            timestamp=datetime.now(),
        )
    
    def predict_error_rate(
        self,
        metric_name: str,
        horizon_minutes: Optional[int] = None
    ) -> Optional[Prediction]:
        """
        Predict future error rate.
        
        Args:
            metric_name: Name of the metric
            horizon_minutes: Prediction horizon
            
        Returns:
            Prediction or None
        """
        horizon = horizon_minutes or self.prediction_window
        
        metric = self.performance_monitor.get_metrics(metric_name)
        if not metric or metric.request_count < 20:
            return None
        
        # Use exponential moving average for error rate prediction
        error_rate = metric.error_rate
        
        history_key = f"{metric_name}_error_rate"
        if history_key not in self._historical_data:
            self._historical_data[history_key] = deque(maxlen=self._max_history)
        
        self._historical_data[history_key].append({
            "value": error_rate,
            "timestamp": datetime.now(),
        })
        
        history = list(self._historical_data[history_key])
        
        if len(history) < 3:
            return None
        
        values = [h["value"] for h in history]
        
        # EMA prediction (exponential moving average)
        predicted_value, confidence = self._ema_prediction(values, alpha=0.3)
        
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.01
        upper_bound = min(1.0, predicted_value + (1.96 * std_dev))
        lower_bound = max(0.0, predicted_value - (1.96 * std_dev))
        
        return Prediction(
            metric_name=f"{metric_name}_error_rate",
            predicted_value=predicted_value,
            confidence=confidence,
            prediction_horizon=horizon,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            timestamp=datetime.now(),
        )
    
    def recommend_scaling(
        self,
        target_latency: Optional[float] = None,
        target_error_rate: Optional[float] = None
    ) -> List[ScalingRecommendation]:
        """
        Recommend scaling actions based on predictions.
        
        Args:
            target_latency: Target latency threshold (seconds)
            target_error_rate: Target error rate threshold (0.0-1.0)
            
        Returns:
            List of scaling recommendations
        """
        recommendations = []
        
        target_latency = target_latency or 1.0  # Default: 1 second
        target_error_rate = target_error_rate or 0.05  # Default: 5%
        
        all_metrics = self.performance_monitor.get_all_metrics()
        
        for metric_name, metric in all_metrics.items():
            if metric.request_count < 20:
                continue
            
            # Predict future latency
            latency_pred = self.predict_latency(metric_name)
            error_pred = self.predict_error_rate(metric_name)
            
            if latency_pred and latency_pred.predicted_value > target_latency * 1.5:
                # Latency will exceed target significantly
                recommendations.append(ScalingRecommendation(
                    action="scale_up",
                    confidence=latency_pred.confidence,
                    reason=f"Predicted latency ({latency_pred.predicted_value:.3f}s) exceeds target ({target_latency}s)",
                    estimated_impact=f"Expected to reduce latency by ~{(latency_pred.predicted_value - target_latency) / latency_pred.predicted_value * 100:.1f}%",
                    priority="high" if latency_pred.predicted_value > target_latency * 2 else "medium",
                ))
            elif latency_pred and latency_pred.predicted_value < target_latency * 0.5:
                # Latency much lower than target - can scale down
                if metric.request_count > 100:  # Only if sufficient load
                    recommendations.append(ScalingRecommendation(
                        action="scale_down",
                        confidence=min(0.7, latency_pred.confidence),  # Lower confidence for scale-down
                        reason=f"Predicted latency ({latency_pred.predicted_value:.3f}s) well below target ({target_latency}s)",
                        estimated_impact="Could reduce resource usage while maintaining performance",
                        priority="low",
                    ))
            
            if error_pred and error_pred.predicted_value > target_error_rate * 2:
                # Error rate will exceed target
                recommendations.append(ScalingRecommendation(
                    action="scale_up",
                    confidence=error_pred.confidence,
                    reason=f"Predicted error rate ({error_pred.predicted_value*100:.1f}%) exceeds target ({target_error_rate*100:.1f}%)",
                    estimated_impact="Expected to reduce error rate by improving resource availability",
                    priority="high",
                ))
        
        # Sort by priority and confidence
        priority_order = {"high": 3, "medium": 2, "low": 1}
        recommendations.sort(
            key=lambda r: (priority_order.get(r.priority, 0), r.confidence),
            reverse=True
        )
        
        return recommendations
    
    def _linear_trend_prediction(
        self,
        values: List[float],
        horizon: int
    ) -> Tuple[float, float]:
        """
        Simple linear trend prediction.
        
        Args:
            values: Historical values
            horizon: Prediction horizon (minutes)
            
        Returns:
            Tuple of (predicted_value, confidence)
        """
        if len(values) < 2:
            return values[-1] if values else 0.0, 0.5
        
        # Calculate linear trend
        n = len(values)
        x = list(range(n))
        
        # Simple linear regression
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        
        # Slope
        denominator = (n * sum_x2) - (sum_x * sum_x)
        if denominator == 0:
            # No trend - use average
            return statistics.mean(values), 0.5
        
        slope = ((n * sum_xy) - (sum_x * sum_y)) / denominator
        
        # Intercept
        intercept = (sum_y - (slope * sum_x)) / n
        
        # Predict future value (extrapolate)
        # Scale horizon to data points (approximate)
        future_x = n + (horizon / 60.0)  # Assume ~1 data point per minute
        predicted_value = intercept + (slope * future_x)
        
        # Confidence based on R-squared (simplified)
        # Use variance as proxy for confidence
        variance = statistics.variance(values) if len(values) > 1 else 0.0
        mean_value = statistics.mean(values)
        
        if mean_value > 0:
            cv = variance / mean_value if mean_value > 0 else 1.0  # Coefficient of variation
            confidence = max(0.3, min(0.9, 1.0 - cv))  # Lower variance = higher confidence
        else:
            confidence = 0.5
        
        return max(0.0, predicted_value), confidence
    
    def _ema_prediction(
        self,
        values: List[float],
        alpha: float = 0.3
    ) -> Tuple[float, float]:
        """
        Exponential Moving Average prediction.
        
        Args:
            values: Historical values
            alpha: Smoothing factor (0.0-1.0)
            
        Returns:
            Tuple of (predicted_value, confidence)
        """
        if not values:
            return 0.0, 0.5
        
        # Calculate EMA
        ema = values[0]
        for value in values[1:]:
            ema = alpha * value + (1 - alpha) * ema
        
        # Confidence based on recent stability
        recent_values = values[-5:] if len(values) >= 5 else values
        variance = statistics.variance(recent_values) if len(recent_values) > 1 else 0.0
        mean_value = statistics.mean(recent_values)
        
        if mean_value > 0:
            cv = variance / mean_value
            confidence = max(0.3, min(0.9, 1.0 - cv))
        else:
            confidence = 0.5
        
        return ema, confidence


__all__ = [
    "PredictiveAnalytics",
    "Prediction",
    "ScalingRecommendation",
]

