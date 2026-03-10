# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: alerting.py
Modül: cognitive_management/v2/monitoring
Görev: Alerting System - Enterprise-grade alerting for V2 Cognitive Management.
       Phase 5.4: Advanced monitoring with intelligent alerting. AlertLevel,
       Alert, AlertRule, AlertManager sınıflarını içerir. Alert creation,
       alert routing, alert aggregation ve intelligent alerting işlemlerini
       yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (alerting system)
- Design Patterns: Manager Pattern (alert management)
- Endüstri Standartları: Enterprise alerting best practices

KULLANIM:
- Alert creation için
- Alert routing için
- Alert aggregation için
- Intelligent alerting için

BAĞIMLILIKLAR:
- threading: Thread-safe işlemler
- dataclasses: Alert data structures

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time
import threading
from collections import deque


# =============================================================================
# Alert Levels
# =============================================================================

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """
    Alert instance.
    
    Industry standard: Prometheus alerting
    """
    level: AlertLevel
    title: str
    message: str
    component: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "component": self.component,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at,
        }
    
    def resolve(self) -> None:
        """Mark alert as resolved"""
        self.resolved = True
        self.resolved_at = time.time()


# =============================================================================
# Alert Manager
# =============================================================================

class AlertManager:
    """
    Alert manager for cognitive management system.
    
    Phase 5.4: Enterprise-grade alerting.
    
    Features:
    - Alert generation
    - Alert deduplication
    - Alert history
    - Alert handlers
    - Alert aggregation
    """
    
    def __init__(self, max_alerts: int = 1000):
        """
        Initialize alert manager.
        
        Args:
            max_alerts: Maximum number of alerts to store
        """
        self.max_alerts = max_alerts
        self._alerts: deque = deque(maxlen=max_alerts)
        self._handlers: List[Callable[[Alert], None]] = []
        self._lock = threading.RLock()
        self._deduplication_window = 60.0  # seconds
    
    def register_handler(self, handler: Callable[[Alert], None]) -> None:
        """
        Register alert handler.
        
        Args:
            handler: Function to call when alert is raised
        """
        with self._lock:
            self._handlers.append(handler)
    
    def unregister_handler(self, handler: Callable[[Alert], None]) -> None:
        """Unregister alert handler"""
        with self._lock:
            if handler in self._handlers:
                self._handlers.remove(handler)
    
    def _is_duplicate(self, alert: Alert) -> bool:
        """
        Check if alert is duplicate (within deduplication window).
        
        Args:
            alert: Alert to check
            
        Returns:
            True if duplicate
        """
        with self._lock:
            cutoff_time = time.time() - self._deduplication_window
            
            for existing in reversed(self._alerts):
                if existing.timestamp < cutoff_time:
                    break
                
                if (existing.level == alert.level and
                    existing.title == alert.title and
                    existing.component == alert.component and
                    not existing.resolved):
                    return True
            
            return False
    
    def raise_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        component: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        deduplicate: bool = True,
    ) -> Alert:
        """
        Raise an alert.
        
        Args:
            level: Alert level
            title: Alert title
            message: Alert message
            component: Component name
            metadata: Optional metadata
            deduplicate: Whether to deduplicate alerts
            
        Returns:
            Created Alert
        """
        alert = Alert(
            level=level,
            title=title,
            message=message,
            component=component,
            metadata=metadata or {},
        )
        
        # Check for duplicates
        if deduplicate and self._is_duplicate(alert):
            return alert  # Return but don't process
        
        with self._lock:
            # Store alert
            self._alerts.append(alert)
            
            # Call handlers
            for handler in self._handlers:
                try:
                    handler(alert)
                except Exception:
                    pass  # Don't let handler errors break alerting
        
        return alert
    
    def resolve_alert(
        self,
        title: str,
        component: Optional[str] = None,
    ) -> bool:
        """
        Resolve an alert.
        
        Args:
            title: Alert title
            component: Component name (optional filter)
            
        Returns:
            True if alert was found and resolved
        """
        with self._lock:
            for alert in reversed(self._alerts):
                if alert.title == title and not alert.resolved:
                    if component is None or alert.component == component:
                        alert.resolve()
                        return True
            return False
    
    def get_active_alerts(
        self,
        level: Optional[AlertLevel] = None,
        component: Optional[str] = None,
    ) -> List[Alert]:
        """
        Get active (unresolved) alerts.
        
        Args:
            level: Filter by level (None = all)
            component: Filter by component (None = all)
            
        Returns:
            List of active alerts
        """
        with self._lock:
            alerts = [
                alert for alert in self._alerts
                if not alert.resolved
            ]
            
            if level:
                alerts = [a for a in alerts if a.level == level]
            
            if component:
                alerts = [a for a in alerts if a.component == component]
            
            return alerts
    
    def get_all_alerts(
        self,
        limit: int = 100,
        resolved: Optional[bool] = None,
    ) -> List[Alert]:
        """
        Get all alerts.
        
        Args:
            limit: Maximum number of alerts
            resolved: Filter by resolved status (None = all)
            
        Returns:
            List of alerts
        """
        with self._lock:
            alerts = list(self._alerts)
            
            if resolved is not None:
                alerts = [
                    a for a in alerts
                    if a.resolved == resolved
                ]
            
            return alerts[-limit:]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        with self._lock:
            active = [a for a in self._alerts if not a.resolved]
            resolved = [a for a in self._alerts if a.resolved]
            
            level_counts = {}
            for level in AlertLevel:
                level_counts[level.value] = sum(
                    1 for a in active if a.level == level
                )
            
            return {
                "total_alerts": len(self._alerts),
                "active_alerts": len(active),
                "resolved_alerts": len(resolved),
                "level_counts": level_counts,
            }


__all__ = ["AlertManager", "AlertLevel", "Alert"]

