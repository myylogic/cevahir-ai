# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: health_check.py
Modül: cognitive_management/v2/monitoring
Görev: Health Check System - Enterprise-grade health checking for V2 Cognitive
       Management. Phase 5.4: Advanced monitoring with comprehensive health checks.
       HealthStatus, HealthCheck, HealthCheckResult, HealthCheckManager sınıflarını
       içerir. Health check execution, health status monitoring ve health check
       aggregation işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (health checking)
- Design Patterns: Manager Pattern (health check management)
- Endüstri Standartları: Health checking best practices

KULLANIM:
- Health check execution için
- Health status monitoring için
- Health check aggregation için

BAĞIMLILIKLAR:
- threading: Thread-safe işlemler
- dataclasses: Health check data structures

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


# =============================================================================
# Health Status
# =============================================================================

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """
    Component health status.
    
    Industry standard: Kubernetes liveness/readiness probes
    """
    name: str
    status: HealthStatus
    message: str = ""
    last_check: Optional[float] = None
    check_duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check,
            "check_duration": self.check_duration,
            "metadata": self.metadata,
        }


# =============================================================================
# Health Checker
# =============================================================================

class HealthChecker:
    """
    Health checker for cognitive management system.
    
    Phase 5.4: Enterprise-grade health monitoring.
    
    Features:
    - Component health checks
    - Liveness probes
    - Readiness probes
    - Health aggregation
    - Health history
    """
    
    def __init__(self):
        """Initialize health checker"""
        self._checks: Dict[str, Callable[[], ComponentHealth]] = {}
        self._health_history: List[ComponentHealth] = []
        self._lock = threading.RLock()
        self._max_history = 100
    
    def register_check(
        self,
        name: str,
        check_func: Callable[[], ComponentHealth],
    ) -> None:
        """
        Register health check function.
        
        Args:
            name: Component name
            check_func: Health check function
        """
        with self._lock:
            self._checks[name] = check_func
    
    def unregister_check(self, name: str) -> None:
        """Unregister health check"""
        with self._lock:
            self._checks.pop(name, None)
    
    def check_component(self, name: str) -> Optional[ComponentHealth]:
        """
        Check health of specific component.
        
        Args:
            name: Component name
            
        Returns:
            ComponentHealth or None if not found
        """
        with self._lock:
            check_func = self._checks.get(name)
            if not check_func:
                return None
            
            start_time = time.time()
            try:
                health = check_func()
                health.last_check = time.time()
                health.check_duration = time.time() - start_time
                
                # Store in history
                self._health_history.append(health)
                if len(self._health_history) > self._max_history:
                    self._health_history.pop(0)
                
                return health
            except Exception as e:
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(e)}",
                    last_check=time.time(),
                    check_duration=time.time() - start_time,
                )
    
    def check_all(self) -> Dict[str, ComponentHealth]:
        """
        Check health of all registered components.
        
        Returns:
            Dictionary of component health statuses
        """
        with self._lock:
            results = {}
            for name in self._checks.keys():
                health = self.check_component(name)
                if health:
                    results[name] = health
            return results
    
    def get_overall_health(self) -> HealthStatus:
        """
        Get overall system health.
        
        Returns:
            Overall health status
        """
        all_health = self.check_all()
        
        if not all_health:
            return HealthStatus.UNKNOWN
        
        statuses = [health.status for health in all_health.values()]
        
        # If any unhealthy, system is unhealthy
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        
        # If any degraded, system is degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        
        # If all healthy, system is healthy
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        
        return HealthStatus.UNKNOWN
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get health summary.
        
        Returns:
            Health summary dictionary
        """
        all_health = self.check_all()
        overall = self.get_overall_health()
        
        return {
            "overall_status": overall.value,
            "components": {
                name: health.to_dict()
                for name, health in all_health.items()
            },
            "total_components": len(all_health),
            "healthy_count": sum(
                1 for h in all_health.values()
                if h.status == HealthStatus.HEALTHY
            ),
            "degraded_count": sum(
                1 for h in all_health.values()
                if h.status == HealthStatus.DEGRADED
            ),
            "unhealthy_count": sum(
                1 for h in all_health.values()
                if h.status == HealthStatus.UNHEALTHY
            ),
        }
    
    def get_health_history(
        self,
        component_name: Optional[str] = None,
        limit: int = 50
    ) -> List[ComponentHealth]:
        """
        Get health check history.
        
        Args:
            component_name: Filter by component (None = all)
            limit: Maximum number of records
            
        Returns:
            List of ComponentHealth records
        """
        with self._lock:
            history = self._health_history.copy()
            
            if component_name:
                history = [
                    h for h in history
                    if h.name == component_name
                ]
            
            return history[-limit:]


__all__ = ["HealthChecker", "HealthStatus", "ComponentHealth"]

