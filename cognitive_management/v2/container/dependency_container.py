# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: dependency_container.py
Modül: cognitive_management/v2/container
Görev: Dependency Injection Container - SOLID: Dependency Inversion Principle (DIP)
       uygulanmış. Container pattern ile tüm bağımlılıklar merkezi olarak yönetilir.
       Service registration, resolution, lifecycle management ve dependency injection
       işlemlerini yapar.

MİMARİ:
- SOLID Prensipleri: Dependency Inversion Principle (DIP)
- Design Patterns: Container Pattern (dependency injection)
- Endüstri Standartları: Dependency injection best practices

KULLANIM:
- Dependency injection için
- Service registration için
- Service resolution için

BAĞIMLILIKLAR:
- Modül içi bağımlılıklar

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Any, Dict, Type, Callable, Optional, TypeVar
from dataclasses import dataclass

T = TypeVar("T")


@dataclass
class ServiceRegistration:
    """Service registration bilgisi"""
    service_type: Type
    implementation: Any
    is_singleton: bool
    factory: Optional[Callable[[], Any]] = None


class CognitiveContainer:
    """
    Dependency Injection Container.
    
    SOLID: Dependency Inversion Principle
    - High-level modules depend on abstractions (interfaces)
    - Low-level modules implement interfaces
    - Container manages dependencies
    
    Usage:
        container = CognitiveContainer()
        container.register_singleton(ModelBackend, backend_instance)
        container.register_factory(PolicyRouter, lambda: DefaultPolicyRouter())
        
        orchestrator = container.build_orchestrator()
    """
    
    def __init__(self):
        self._services: Dict[Type, ServiceRegistration] = {}
        self._singletons: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable[[], Any]] = {}
    
    # ---------------------------------------------------------------------
    # Service Registration
    # ---------------------------------------------------------------------
    
    def register_singleton(
        self,
        interface: Type[T],
        implementation: T
    ) -> None:
        """
        Register singleton service.
        Aynı instance her zaman döner.
        
        Args:
            interface: Interface/Protocol type
            implementation: Implementation instance
        """
        self._services[interface] = ServiceRegistration(
            service_type=interface,
            implementation=implementation,
            is_singleton=True,
        )
        self._singletons[interface] = implementation
    
    def register_factory(
        self,
        interface: Type[T],
        factory: Callable[[], T]
    ) -> None:
        """
        Register factory-based service.
        Her resolve'da yeni instance oluşturulur.
        
        Args:
            interface: Interface/Protocol type
            factory: Factory function that creates instance
        """
        self._services[interface] = ServiceRegistration(
            service_type=interface,
            implementation=None,
            is_singleton=False,
            factory=factory,
        )
        self._factories[interface] = factory
    
    def register_transient(
        self,
        interface: Type[T],
        implementation_class: Type[T]
    ) -> None:
        """
        Register transient service.
        Her resolve'da yeni instance oluşturulur (no-arg constructor).
        
        Args:
            interface: Interface/Protocol type
            implementation_class: Implementation class
        """
        def factory() -> T:
            return implementation_class()
        
        self.register_factory(interface, factory)
    
    # ---------------------------------------------------------------------
    # Service Resolution
    # ---------------------------------------------------------------------
    
    def resolve(self, interface: Type[T]) -> T:
        """
        Resolve service from container.
        
        Args:
            interface: Interface/Protocol type to resolve
            
        Returns:
            Resolved service instance
            
        Raises:
            ValueError: If service not registered
        """
        if interface not in self._services:
            raise ValueError(
                f"Service {interface.__name__} not registered in container. "
                f"Available services: {list(self._services.keys())}"
            )
        
        registration = self._services[interface]
        
        # Singleton: return cached instance
        if registration.is_singleton:
            if interface in self._singletons:
                return self._singletons[interface]
            # First time: cache it
            instance = registration.implementation
            self._singletons[interface] = instance
            return instance
        
        # Factory: create new instance
        if registration.factory:
            return registration.factory()
        
        # Transient: create new instance from class
        if registration.implementation:
            return registration.implementation()
        
        raise ValueError(f"Cannot resolve {interface.__name__}: invalid registration")
    
    def is_registered(self, interface: Type) -> bool:
        """Check if service is registered"""
        return interface in self._services
    
    def list_services(self) -> list[Type]:
        """List all registered service types"""
        return list(self._services.keys())
    
    # ---------------------------------------------------------------------
    # Component Building
    # ---------------------------------------------------------------------
    
    def build_orchestrator(self):
        """
        Build CognitiveOrchestrator with all dependencies.
        
        Returns:
            CognitiveOrchestrator instance
            
        Raises:
            ValueError: If required services are not registered
        """
        from ..core.orchestrator import CognitiveOrchestrator
        from ..interfaces.backend_protocols import FullModelBackend
        from ..interfaces.component_protocols import (
            PolicyRouter,
            MemoryService,
            Critic,
            DeliberationEngine,
            ToolExecutor,
        )
        from ..events.event_bus import EventBus
        
        # Resolve required services
        backend = self.resolve(FullModelBackend)
        policy_router = self.resolve(PolicyRouter)
        memory_service = self.resolve(MemoryService)
        critic = self.resolve(Critic)
        
        # Optional services
        deliberation_engine = None
        if self.is_registered(DeliberationEngine):
            try:
                deliberation_engine = self.resolve(DeliberationEngine)
            except ValueError:
                pass  # Optional
        
        tool_executor = None
        if self.is_registered(ToolExecutor):
            try:
                tool_executor = self.resolve(ToolExecutor)
            except ValueError:
                pass  # Optional
        
        event_bus = None
        if self.is_registered(EventBus):
            try:
                event_bus = self.resolve(EventBus)
            except ValueError:
                pass  # Optional, will use default
        
        # Get middleware from container if registered
        middleware = None
        if self.is_registered(list):  # Middleware list
            try:
                middleware = self.resolve(list)
            except ValueError:
                pass
        
        # Phase 5.4: Performance monitor (if available)
        performance_monitor = None
        try:
            from ..monitoring.performance_monitor import PerformanceMonitor
            performance_monitor = PerformanceMonitor()
        except ImportError:
            performance_monitor = None
        
        # Build orchestrator
        return CognitiveOrchestrator(
            backend=backend,
            policy_router=policy_router,
            memory_service=memory_service,
            critic=critic,
            deliberation_engine=deliberation_engine,
            event_bus=event_bus,
            middleware=middleware,  # Middleware chain (if provided)
            performance_monitor=performance_monitor,  # Phase 5.4
            tool_executor=tool_executor,  # Phase 6.1: Tool Policy Implementation
        )
    
    # ---------------------------------------------------------------------
    # Container Management
    # ---------------------------------------------------------------------
    
    def clear(self) -> None:
        """Clear all registrations (for testing)"""
        self._services.clear()
        self._singletons.clear()
        self._factories.clear()
    
    def get_registration(self, interface: Type) -> Optional[ServiceRegistration]:
        """Get service registration info"""
        return self._services.get(interface)


__all__ = ["CognitiveContainer", "ServiceRegistration"]

