# -*- coding: utf-8 -*-
"""
Metrics Collection
==================

Metrics collection for monitoring.
Endüstri Standardı: Prometheus-style metrics
"""

from flask import Flask, request, g
from typing import Dict, Any
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Metrics collector"""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "requests_total": 0,
            "requests_by_method": {},
            "requests_by_status": {},
            "request_duration_ms": [],
            "errors_total": 0,
        }
    
    def record_request(self, method: str, status_code: int, duration_ms: float):
        """Record HTTP request"""
        self.metrics["requests_total"] += 1
        
        # By method
        if method not in self.metrics["requests_by_method"]:
            self.metrics["requests_by_method"][method] = 0
        self.metrics["requests_by_method"][method] += 1
        
        # By status
        status_class = f"{status_code // 100}xx"
        if status_class not in self.metrics["requests_by_status"]:
            self.metrics["requests_by_status"][status_class] = 0
        self.metrics["requests_by_status"][status_class] += 1
        
        # Duration
        self.metrics["request_duration_ms"].append(duration_ms)
        # Keep only last 1000 durations
        if len(self.metrics["request_duration_ms"]) > 1000:
            self.metrics["request_duration_ms"] = self.metrics["request_duration_ms"][-1000:]
        
        # Errors
        if status_code >= 500:
            self.metrics["errors_total"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        metrics = self.metrics.copy()
        
        # Calculate average duration
        if metrics["request_duration_ms"]:
            metrics["avg_duration_ms"] = sum(metrics["request_duration_ms"]) / len(metrics["request_duration_ms"])
            metrics["max_duration_ms"] = max(metrics["request_duration_ms"])
            metrics["min_duration_ms"] = min(metrics["request_duration_ms"])
        else:
            metrics["avg_duration_ms"] = 0
            metrics["max_duration_ms"] = 0
            metrics["min_duration_ms"] = 0
        
        # Remove raw duration list (too large)
        del metrics["request_duration_ms"]
        
        metrics["timestamp"] = datetime.utcnow().isoformat() + "Z"
        return metrics


# Global metrics collector
_metrics_collector = MetricsCollector()


def register_metrics(app: Flask):
    """
    Register metrics middleware.
    
    Args:
        app: Flask application instance
    """
    @app.before_request
    def record_request_start():
        """Record request start time"""
        g.request_start_time = time.time()
    
    @app.after_request
    def record_request_metrics(response):
        """Record request metrics"""
        if app.config.get("ENABLE_METRICS", True):
            duration_ms = (time.time() - g.request_start_time) * 1000
            _metrics_collector.record_request(
                method=request.method,
                status_code=response.status_code,
                duration_ms=duration_ms
            )
        return response
    
    logger.info("Metrics collection registered")


def get_metrics() -> Dict[str, Any]:
    """
    Get current metrics.
    
    Returns:
        Metrics dictionary
    """
    return _metrics_collector.get_metrics()

