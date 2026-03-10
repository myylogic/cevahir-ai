# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: cognitive_manager.py
Modül: cognitive_management
Görev: Cognitive Manager - Bilişsel Üst Denetleyici (V2 Implementation). Chat/inference
       akışında strateji seçimi (direct/think/debate), iç ses üretimi, araç kullanımı
       önerisi, eleştirel kontrol (critic) ve bellek/bağlam yönetimini tek bir
       koordinatörde birleştirir. Model katmanına somut sınıfla değil dar bir arayüzle
       (ModelAPI) bağımlıdır. V2 Orchestrator kullanır (SOLID principles).

MİMARİ:
- SOLID Prensipleri: Single Responsibility (bilişsel yönetim koordinasyonu),
                     Dependency Inversion (ModelAPI interface'e bağımlı)
- Design Patterns: Manager Pattern (bilişsel yönetim), Orchestrator Pattern
- Endüstri Standartları: Cognitive management best practices

KULLANIM:
- Chat/inference akışı için
- Strateji seçimi için
- İç ses üretimi için
- Araç kullanımı önerisi için
- Eleştirel kontrol için

BAĞIMLILIKLAR:
- V2 Orchestrator: Orchestration işlemleri
- CognitiveTypes: Tip tanımları
- Config: Yapılandırma

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
import time

from .cognitive_types import (
    CognitiveState,
    CognitiveInput,
    CognitiveOutput,
    PolicyOutput,
    DecodingConfig,
    ThoughtCandidate,
)
from .config import CognitiveManagerConfig
from .exceptions import (
    CognitiveError,
    ConfigError,
    ModelInterfaceError,
    ValidationError,
)
from .utils.logging import get_logger
from .utils.timers import timer

# V2 imports
from .v2.core.orchestrator import CognitiveOrchestrator
from .v2.components import (
    PolicyRouterV2,
    MemoryServiceV2,
    CriticV2,
    DeliberationEngineV2,
)
from .v2.adapters.backend_adapter import ModelAPIAdapter
from .v2.middleware import (
    ErrorHandlingMiddleware,
    MetricsMiddleware,
    ValidationMiddleware,
    RetryConfig,
    CircuitBreakerConfig,
)
from .v2.events.event_bus import EventBus
from .v2.events.event_handlers import LoggingEventHandler
from .v2.container import CognitiveContainer
from .v2.interfaces.backend_protocols import FullModelBackend
from .v2.interfaces.component_protocols import (
    PolicyRouter,
    MemoryService,
    Critic,
    DeliberationEngine,
    ToolExecutor,
)


# === Dar Model Arayüzü (Dependency Inversion) ================================

class ModelAPI(Protocol):
    """
    Cognitive katmanın ihtiyaç duyduğu minimal model arayüzü.
    Somut ModelManager bu protokolü karşılamalıdır.
    """
    def generate(self, prompt: str, decoding_cfg: DecodingConfig) -> str: ... 
    def score(self, prompt: str, candidate: str) -> float: ... 
    def entropy_estimate(self, text: str) -> float: ...  # pragma: no cover
    
    # Multimodal desteği
    def process_audio(self, audio_data: bytes) -> str: ...  # pragma: no cover
    def process_image(self, image_data: bytes) -> str: ...  # pragma: no cover
    def process_multimodal(self, text: str = None, audio: bytes = None, image: bytes = None) -> str: ...  # pragma: no cover


# === CognitiveManager V2 ======================================================

class CognitiveManager:
    """
    Üst denetleyici: V2 Orchestrator kullanır.
    Backward compatibility: V1 interface'i korunur.
    """

    def __init__(
        self,
        model_manager: ModelAPI,
        cfg: Optional[CognitiveManagerConfig] = None,
        config_path: Optional[str] = None,
        enable_hot_reload: bool = False
    ):
        """
        Initialize CognitiveManager.
        
        Args:
            model_manager: Model API interface
            cfg: Optional configuration (if None, uses default)
            config_path: Optional path to config file (JSON/YAML)
            enable_hot_reload: Enable hot reload for config file changes
        """
        # Phase 6.2: Configuration Management
        if config_path:
            from .v2.config import ConfigManager
            self._config_manager = ConfigManager(
                config_path=config_path,
                watch=enable_hot_reload,
                default_config=cfg or CognitiveManagerConfig()
            )
            cfg = self._config_manager.get_config()
        else:
            self._config_manager = None
            if cfg is None:
                cfg = CognitiveManagerConfig()
        
        if not isinstance(cfg, CognitiveManagerConfig):
            raise ConfigError("CognitiveManagerConfig bekleniyor.")
        self.cfg = cfg
        self.cfg.validate()

        # Model arayüz doğrulaması
        required_methods = ["generate", "score"]
        for fn in required_methods:
            if not hasattr(model_manager, fn):
                raise ModelInterfaceError(f"ModelAPI '{fn}' metodunu sağlamıyor.")
        
        # Multimodal metodları kontrol et (opsiyonel)
        self.multimodal_enabled = all(hasattr(model_manager, method) for method in 
                                    ["process_audio", "process_image", "process_multimodal"])
        
        self.mm: ModelAPI = model_manager  # somut sınıf değil, arayüz

        # V2 Container kullanarak DI Pattern uygula
        container = CognitiveContainer()
        
        # Backend adapter
        # Phase 8: ConnectionPool integration (optional)
        from .v2.utils.connection_pool import ConnectionPool
        connection_pool = None
        if getattr(cfg.runtime, 'enable_connection_pool', False):
            try:
                def create_connection():
                    # Return a connection-like object (wrapped model API)
                    # For simple cases, we can pool the model API itself
                    return model_manager
                
                connection_pool = ConnectionPool(
                    factory=create_connection,
                    max_size=getattr(cfg.runtime, 'connection_pool_size', 10),
                    min_size=2,
                )
            except Exception as e:
                import logging
                logging.warning(f"ConnectionPool initialization başarısız: {e}")
                connection_pool = None
        
        backend = ModelAPIAdapter(model_manager, cfg=cfg, connection_pool=connection_pool)
        container.register_singleton(FullModelBackend, backend)
        
        # Phase 8: Store connection pool reference for API access
        self._connection_pool = connection_pool
        
        # Components
        policy_router = PolicyRouterV2(cfg)
        container.register_singleton(PolicyRouter, policy_router)
        
        memory_service = MemoryServiceV2(cfg)
        container.register_singleton(MemoryService, memory_service)
        
        # Phase 8: Store memory service reference for API access
        self._memory_service = memory_service
        
        critic = CriticV2(cfg, model_manager)
        container.register_singleton(Critic, critic)
        
        # Optional: Deliberation engine
        if cfg.policy.allow_inner_steps:
            deliberation_engine = DeliberationEngineV2(cfg, model_manager)
            container.register_singleton(DeliberationEngine, deliberation_engine)
        
        # Phase 6.1: Tool Executor (if tools enabled)
        self._tool_executor = None
        if cfg.tools.enable_tools:
            from .v2.components.tool_executor_v2 import ToolExecutorV2
            tool_executor = ToolExecutorV2(cfg)
            self._tool_executor = tool_executor  # Store reference for API access
            container.register_singleton(ToolExecutor, tool_executor)
        
        # Event Bus ve handlers
        event_bus = EventBus()
        logging_handler = LoggingEventHandler()
        event_bus.subscribe(logging_handler, event_type="*")
        
        # Phase 8: MetricsEventHandler entegrasyonu
        from .v2.events.event_handlers import MetricsEventHandler
        metrics_event_handler = MetricsEventHandler()
        event_bus.subscribe(metrics_event_handler, event_type="*")
        
        container.register_singleton(EventBus, event_bus)
        
        # Middleware chain oluştur (sıra önemli!)
        max_input_length = getattr(cfg.runtime, 'max_input_length', 10000)
        
        # Phase 5.2: Cache middleware (optional, can be enabled via config)
        from .v2.middleware.cache import CacheMiddleware
        cache_enabled = getattr(cfg.runtime, 'enable_caching', True)
        
        # Phase 5.3: Tracing middleware (optional, can be enabled via config)
        from .v2.middleware.tracing import TracingMiddleware
        from .v2.utils.tracing import TraceStorage
        tracing_enabled = getattr(cfg.runtime, 'enable_tracing', True)
        trace_storage = TraceStorage(max_traces=getattr(cfg.runtime, 'max_traces', 1000)) if tracing_enabled else None
        
        middleware_chain = [
            ValidationMiddleware(
                max_message_length=max_input_length,
                min_message_length=1,
            ),
        ]
        
        # Add tracing middleware if enabled (early in chain for full coverage)
        if tracing_enabled:
            middleware_chain.append(
                TracingMiddleware(
                    trace_storage=trace_storage,
                    enabled=True,
                    sample_rate=getattr(cfg.runtime, 'trace_sample_rate', 1.0),
                )
            )
        
        # Add cache middleware if enabled
        if cache_enabled:
            # Phase 8: SemanticCache integration
            semantic_cache = None
            enable_semantic_cache = getattr(cfg.runtime, 'enable_semantic_cache', False)
            if enable_semantic_cache and cfg.memory.enable_vector_memory:
                try:
                    # Get embedding adapter from memory service for semantic cache
                    from .v2.utils.semantic_cache import SemanticCache
                    from .v2.components.embedding_adapter import create_embedding_adapter
                    embedding_adapter = create_embedding_adapter(cfg)
                    if embedding_adapter:
                        semantic_cache = SemanticCache(
                            cfg=cfg,
                            embedding_adapter=embedding_adapter,
                            similarity_threshold=getattr(cfg.runtime, 'semantic_cache_threshold', 0.85),
                            max_size=getattr(cfg.runtime, 'semantic_cache_max_size', 1000),
                        )
                except Exception as e:
                    import logging
                    logging.warning(f"SemanticCache initialization başarısız: {e}")
                    semantic_cache = None
            
            middleware_chain.append(
                CacheMiddleware(
                    response_ttl=getattr(cfg.runtime, 'cache_ttl', 3600.0),
                    enabled=True,
                    enable_semantic_cache=enable_semantic_cache and semantic_cache is not None,
                    semantic_cache=semantic_cache,
                    similarity_threshold=getattr(cfg.runtime, 'semantic_cache_threshold', 0.85),
                )
            )
        
        middleware_chain.extend([
            ErrorHandlingMiddleware(
                retry_config=RetryConfig(
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=10.0,
                ),
                circuit_breaker_config=CircuitBreakerConfig(
                    failure_threshold=5,
                    success_threshold=2,
                    timeout=60.0,
                ),
                timeout=30.0,
            ),
            MetricsMiddleware(),
        ])

        # Middleware'i container'a kaydet (opsiyonel - container middleware'i desteklemiyor)
        # Şimdilik middleware'i orchestrator'a direkt geçeceğiz
        
        # Phase 5.4: Advanced Monitoring (BEFORE orchestrator build - needed for performance monitor)
        from .v2.monitoring import (
            HealthChecker,
            AlertManager,
            PerformanceMonitor,
        )
        self._health_checker = HealthChecker()
        self._alert_manager = AlertManager()
        self._performance_monitor = PerformanceMonitor()
        
        # Phase 8: AlertManager default handler sistemi
        from .v2.monitoring.alerting import AlertLevel
        def _default_alert_handler(alert):
            """Default alert handler - logs critical alerts"""
            if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
                import logging
                logger = logging.getLogger("CognitiveManager.Alerts")
                logger.error(
                    f"ALERT [{alert.level.value.upper()}]: {alert.title} | "
                    f"Component: {alert.component} | {alert.message}",
                    extra={"alert_metadata": alert.metadata}
                )
        
        self._alert_manager.register_handler(_default_alert_handler)
        
        # Store metrics event handler for API access
        self._metrics_event_handler = metrics_event_handler
        
        # Phase 5: AIOps Integration & Predictive Analytics
        from .v2.monitoring import (
            AnomalyDetector,
            PredictiveAnalytics,
            TrendAnalyzer,
        )
        
        # Initialize AIOps components
        aiops_sensitivity = getattr(cfg.runtime, 'aiops_sensitivity', 'medium') if cfg else 'medium'
        self._anomaly_detector = AnomalyDetector(
            performance_monitor=self._performance_monitor,
            sensitivity=aiops_sensitivity,
        )
        self._predictive_analytics = PredictiveAnalytics(
            performance_monitor=self._performance_monitor,
        )
        self._trend_analyzer = TrendAnalyzer(
            performance_monitor=self._performance_monitor,
        )
        
        # V2 Orchestrator'ı container'dan oluştur (DI Pattern)
        # Note: Performance monitor is registered in container, orchestrator will get it via DI
        self._orchestrator = container.build_orchestrator()
        
        # Phase 5: AIOps - Enable auto anomaly checking in orchestrator
        if getattr(cfg.runtime, 'enable_auto_anomaly_check', True):
            self._orchestrator._check_anomalies = True
            # Store anomaly detector reference in orchestrator for auto-checking
            self._orchestrator._anomaly_detector = self._anomaly_detector
        
        # Middleware chain'i orchestrator'a ekle (container build'den sonra)
        # Not: Container build_orchestrator middleware'i desteklemiyor, manuel ekliyoruz
        if middleware_chain:
            # Middleware chain'i orchestrator'a set et
            self._orchestrator.middleware_chain = self._orchestrator._build_middleware_chain(middleware_chain)

        # Middleware'lere erişim için referanslar sakla
        # Order depends on enabled features
        idx = 0
        self._validation_middleware = middleware_chain[idx]
        idx += 1
        
        if tracing_enabled:
            self._tracing_middleware = middleware_chain[idx]
            idx += 1
        else:
            self._tracing_middleware = None
        
        if cache_enabled:
            self._cache_middleware = middleware_chain[idx]
            idx += 1
        else:
            self._cache_middleware = None
        
        self._error_middleware = middleware_chain[idx]
        idx += 1
        self._metrics_middleware = middleware_chain[idx]
        
        # Store trace storage for API access
        self._trace_storage = trace_storage
        
        # Register default health checks (AFTER orchestrator is built)
        self._register_default_health_checks()
        
        # Container'ı sakla (test ve debugging için)
        self._container = container

        # Phase 6: Cache Warmer (if enabled)
        if cache_enabled and getattr(cfg.runtime, 'enable_cache_warming', False):
            from .v2.utils.cache_warming import CacheWarmer
            self._cache_warmer = CacheWarmer(
                cache=self._cache_middleware.cache if self._cache_middleware else None,
            )
            # Warm cache on startup (async, non-blocking)
            import threading
            def warm_cache():
                try:
                    self._cache_warmer.warm_cache()
                except Exception:
                    pass  # Non-critical
            threading.Thread(target=warm_cache, daemon=True).start()
        else:
            self._cache_warmer = None
        
        # Logger
        self.log = get_logger("CognitiveManager", base_fields={"component": "cognitive_manager"})
        
        self.log.info("CognitiveManager V2 initialized with full enterprise features")
        
        # Phase 8: Log component initialization summary
        enabled_features = []
        if cfg.memory.enable_vector_memory:
            enabled_features.append("Vector Memory & RAG")
        if cfg.critic.enable_external_fact_checking:
            enabled_features.append("External Fact-Checking")
        if hasattr(cfg.critic, 'constitutional_ai') and cfg.critic.constitutional_ai.enable_constitutional_ai:
            enabled_features.append("Constitutional AI")
        if cfg.policy.tot_enabled:
            enabled_features.append("Tree of Thoughts")
        if cache_enabled:
            enabled_features.append("Caching")
        if tracing_enabled:
            enabled_features.append("Distributed Tracing")
        
        if enabled_features:
            self.log.info(f"Enabled features: {', '.join(enabled_features)}")
    
    def _register_default_health_checks(self) -> None:
        """Register default health checks"""
        from .v2.monitoring import HealthStatus, ComponentHealth
        
        # Backend health check
        def check_backend() -> ComponentHealth:
            try:
                # Simple check: backend exists and is callable
                if hasattr(self._orchestrator, 'backend') and self._orchestrator.backend:
                    return ComponentHealth(
                        name="backend",
                        status=HealthStatus.HEALTHY,
                        message="Backend is available",
                    )
                else:
                    return ComponentHealth(
                        name="backend",
                        status=HealthStatus.UNHEALTHY,
                        message="Backend is not available",
                    )
            except Exception as e:
                return ComponentHealth(
                    name="backend",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Backend check failed: {str(e)}",
                )
        
        # Memory service health check
        def check_memory() -> ComponentHealth:
            try:
                if hasattr(self._orchestrator, 'memory_service') and self._orchestrator.memory_service:
                    return ComponentHealth(
                        name="memory_service",
                        status=HealthStatus.HEALTHY,
                        message="Memory service is available",
                    )
                else:
                    return ComponentHealth(
                        name="memory_service",
                        status=HealthStatus.UNHEALTHY,
                        message="Memory service is not available",
                    )
            except Exception as e:
                return ComponentHealth(
                    name="memory_service",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Memory service check failed: {str(e)}",
                )
        
        # Register checks
        self._health_checker.register_check("backend", check_backend)
        self._health_checker.register_check("memory_service", check_memory)

    # --------------------------------------------------------------------- #
    # Public API (V1 interface - backward compatibility)
    # --------------------------------------------------------------------- #

    def handle(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        *,
        decoding: Optional[DecodingConfig] = None,  # opsiyonel harici override
    ) -> CognitiveOutput:
        """
        Bir kullanıcı mesajını işleyip nihai yanıtı üretir.
        V2 Orchestrator kullanır (sync).
        """
        if not isinstance(state, CognitiveState):
            raise ValidationError("state tipi geçersiz (CognitiveState bekleniyor).")
        if not isinstance(request, CognitiveInput):
            raise ValidationError("request tipi geçersiz (CognitiveInput bekleniyor).")

        user_message = (request.user_message or "").strip()
        if not user_message:
            # Boş gelen mesajlar için kısa güvenli cevap
            return CognitiveOutput(
                text="Boş bir mesaj aldım. Nasıl yardımcı olabilirim?",
                used_mode="direct",
                tool_used=None,
                revised_by_critic=False,
            )

        with timer("cognitive_handle"):
            # V2 Orchestrator'ı kullan (sync)
            output = self._orchestrator.handle(
                state=state,
                request=request,
                decoding_config=decoding,
            )
            
            return output
    
    async def handle_async(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        *,
        decoding: Optional[DecodingConfig] = None,
    ) -> CognitiveOutput:
        """
        Bir kullanıcı mesajını asenkron olarak işleyip nihai yanıtı üretir.
        V2 Orchestrator async method kullanır.
        
        Phase 5.1: Async support for concurrent processing.
        
        Args:
            state: Cognitive state
            request: Cognitive input
            decoding: Optional decoding config override
            
        Returns:
            Cognitive output
        """
        if not isinstance(state, CognitiveState):
            raise ValidationError("state tipi geçersiz (CognitiveState bekleniyor).")
        if not isinstance(request, CognitiveInput):
            raise ValidationError("request tipi geçersiz (CognitiveInput bekleniyor).")

        user_message = (request.user_message or "").strip()
        if not user_message:
            # Boş gelen mesajlar için kısa güvenli cevap
            return CognitiveOutput(
                text="Boş bir mesaj aldım. Nasıl yardımcı olabilirim?",
                used_mode="direct",
                tool_used=None,
                revised_by_critic=False,
            )

        # V2 Orchestrator async method kullan
        output = await self._orchestrator.handle_async(
            state=state,
            request=request,
            decoding_config=decoding,
        )
        
        return output

    # NOTE: render_prompt() metodu kaldırıldı
    # V2'de context building ContextBuildingHandler tarafından yapılıyor
    # Eğer context'e ihtiyaç varsa, memory_service.build_context() kullanılabilir

    # --------------------------------------------------------------------- #
    # Multimodal API
    # --------------------------------------------------------------------- #
    
    def handle_multimodal(
        self,
        state: CognitiveState,
        text: str = None,
        audio: bytes = None,
        image: bytes = None,
        *,
        decoding: Optional[DecodingConfig] = None,
    ) -> CognitiveOutput:
        """
        Çok modaliteli girdi işleme (metin, ses, görüntü).
        """
        if not self.multimodal_enabled:
            # Fallback to text-only
            if text:
                request = CognitiveInput(user_message=text)
                return self.handle(state, request, decoding=decoding)
            else:
                return CognitiveOutput(
                    text="Multimodal işleme desteklenmiyor.",
                    used_mode="direct",
                    tool_used=None,
                    revised_by_critic=False,
                )
        
        # Multimodal işleme
        try:
            # Önce multimodal veriyi işle
            processed_text = self.mm.process_multimodal(text=text, audio=audio, image=image)
            
            # İşlenmiş metni normal akışa gönder
            request = CognitiveInput(user_message=processed_text)
            return self.handle(state, request, decoding=decoding)
            
        except Exception as e:
            self.log.warning(f"Multimodal işleme hatası: {e}")
            # Fallback to text-only
            if text:
                request = CognitiveInput(user_message=text)
                return self.handle(state, request, decoding=decoding)
            else:
                return CognitiveOutput(
                    text="Multimodal veri işlenemedi.",
                    used_mode="direct",
                    tool_used=None,
                    revised_by_critic=False,
                )

    # --------------------------------------------------------------------- #
    # Enterprise Features API
    # --------------------------------------------------------------------- #
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Metrics dictionary with global and mode-specific metrics
        """
        return self._metrics_middleware.get_metrics()
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self._metrics_middleware.reset_metrics()
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of the cognitive system.
        
        Phase 5.4: Advanced monitoring API.
        Uses HealthChecker for component-level health checks.
        
        Returns:
            Health status dictionary
        """
        # Get health from HealthChecker (Phase 5.4)
        health_summary = self._health_checker.get_health_summary()
        
        # Add metrics-based health info
        metrics = self.get_metrics()
        global_metrics = metrics.get("global", {})
        
        # Circuit breaker status
        circuit_state = "unknown"
        if hasattr(self._error_middleware, 'circuit_state'):
            circuit_state = self._error_middleware.circuit_state.state.value
        
        # Combine health summary with metrics
        return {
            **health_summary,
            "circuit_breaker": circuit_state,
            "metrics": {
                "success_rate": global_metrics.get("success_rate", 0.0),
                "request_count": global_metrics.get("request_count", 0),
                "error_count": global_metrics.get("error_count", 0),
                "avg_latency": global_metrics.get("avg_latency", 0.0),
                "last_request_time": global_metrics.get("last_request_time"),
            },
        }
    
    def get_event_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> list:
        """
        Get event history from event bus.
        
        Args:
            event_type: Filter by event type (None = all)
            limit: Maximum number of events to return
            
        Returns:
            List of events
        """
        return self._orchestrator.event_bus.get_history(event_type=event_type, limit=limit)
    
    def clear_event_history(self) -> None:
        """
        Clear event history from event bus.
        
        Phase 8: EventBus API completion.
        """
        if hasattr(self._orchestrator, 'event_bus'):
            self._orchestrator.event_bus.clear_history()
    
    def get_event_subscriber_count(self, event_type: Optional[str] = None) -> int:
        """
        Get number of event subscribers.
        
        Phase 8: EventBus API completion.
        
        Args:
            event_type: Filter by event type (None = all)
            
        Returns:
            Number of subscribers
        """
        if hasattr(self._orchestrator, 'event_bus'):
            return self._orchestrator.event_bus.get_subscriber_count(event_type)
        return 0
    
    def unsubscribe_from_events(
        self,
        observer: Any,
        event_type: Optional[str] = None
    ) -> None:
        """
        Unsubscribe observer from events.
        
        Phase 8: EventBus API completion.
        
        Args:
            observer: Event observer to unsubscribe
            event_type: Specific event type (None = all)
        """
        if hasattr(self._orchestrator, 'event_bus'):
            self._orchestrator.event_bus.unsubscribe(observer, event_type)
    
    def get_event_metrics(self) -> Dict[str, int]:
        """
        Get event-based metrics from MetricsEventHandler.
        
        Phase 8: MetricsEventHandler integration.
        
        Returns:
            Dictionary of event type -> count
        """
        if hasattr(self, '_metrics_event_handler') and self._metrics_event_handler:
            return self._metrics_event_handler.get_metrics()
        return {}
    
    def reset_event_metrics(self) -> None:
        """
        Reset event-based metrics.
        
        Phase 8: MetricsEventHandler integration.
        """
        if hasattr(self, '_metrics_event_handler') and self._metrics_event_handler:
            self._metrics_event_handler.reset_metrics()
    
    # --------------------------------------------------------------------- #
    # Cache Management API (Phase 5.2)
    # ---------------------------------------------------------------------
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get cache statistics.
        
        Phase 5.2: Cache management API.
        
        Returns:
            Cache statistics dictionary or None if caching disabled
        """
        if self._cache_middleware:
            return self._cache_middleware.get_cache_stats()
        return None
    
    def invalidate_cache(
        self,
        pattern: Optional[str] = None
    ) -> int:
        """
        Invalidate cache entries.
        
        Phase 5.2: Cache management API.
        
        Args:
            pattern: Key pattern to invalidate (None = all)
            
        Returns:
            Number of invalidated entries
        """
        if self._cache_middleware:
            return self._cache_middleware.invalidate(pattern)
        return 0
    
    def clear_cache(self) -> None:
        """
        Clear all cache.
        
        Phase 5.2: Cache management API.
        """
        if self._cache_middleware:
            self._cache_middleware.cache.clear()
    
    # --------------------------------------------------------------------- #
    # Tracing API (Phase 5.3)
    # ---------------------------------------------------------------------
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get trace by ID.
        
        Phase 5.3: Distributed tracing API.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Trace dictionary or None
        """
        if self._trace_storage:
            trace = self._trace_storage.get_trace(trace_id)
            if trace:
                return trace.to_dict()
        return None
    
    def get_all_traces(
        self,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all traces.
        
        Phase 5.3: Distributed tracing API.
        
        Args:
            limit: Maximum number of traces to return
            
        Returns:
            List of trace dictionaries
        """
        if self._trace_storage:
            traces = self._trace_storage.get_all_traces()
            if limit:
                traces = traces[-limit:]  # Most recent
            return [trace.to_dict() for trace in traces]
        return []
    
    def get_trace_stats(self) -> Dict[str, Any]:
        """
        Get trace storage statistics.
        
        Phase 5.3: Distributed tracing API.
        
        Returns:
            Trace statistics dictionary
        """
        if self._trace_storage:
            return self._trace_storage.get_stats()
        return {}
    
    def clear_traces(self) -> None:
        """
        Clear all traces.
        
        Phase 5.3: Distributed tracing API.
        """
        if self._trace_storage:
            self._trace_storage.clear()
    
    def export_trace(self, trace_id: str, format: str = "json") -> Optional[str]:
        """
        Export trace in specified format.
        
        Phase 5.3: Distributed tracing API.
        
        Args:
            trace_id: Trace ID
            format: Export format ("json", "dict")
            
        Returns:
            Exported trace string or None
        """
        if not self._trace_storage:
            return None
        
        trace = self._trace_storage.get_trace(trace_id)
        if not trace:
            return None
        
        if format == "json":
            import json
            return json.dumps(trace.to_dict(), indent=2)
        elif format == "dict":
            return trace.to_dict()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    # --------------------------------------------------------------------- #
    # Monitoring API (Phase 5.4)
    # ---------------------------------------------------------------------
    
    def check_component_health(self, component_name: str) -> Optional[Dict[str, Any]]:
        """
        Check health of specific component.
        
        Phase 5.4: Advanced monitoring API.
        
        Args:
            component_name: Component name
            
        Returns:
            Component health dictionary or None
        """
        health = self._health_checker.check_component(component_name)
        if health:
            return health.to_dict()
        return None
    
    def get_health_history(
        self,
        component_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get health check history.
        
        Phase 8: HealthChecker API integration.
        
        Args:
            component_name: Filter by component name (None = all components)
            limit: Maximum number of history records to return
            
        Returns:
            List of health check history dictionaries
        """
        history = self._health_checker.get_health_history(component_name=component_name, limit=limit)
        return [health.to_dict() for health in history]
    
    def register_health_check(
        self,
        name: str,
        check_func: Callable[[], Dict[str, Any]],
        interval_seconds: Optional[float] = None,
    ) -> None:
        """
        Register a custom health check function.
        
        Phase 8: HealthChecker API integration.
        
        Args:
            name: Component name
            check_func: Health check function that returns ComponentHealth-like dictionary
                       Must return dict with keys: status (str), message (str), etc.
        
        Example:
            def my_health_check():
                return {
                    "status": "healthy",
                    "message": "All good",
                    "last_check": time.time()
                }
            
            cm.register_health_check("custom_component", my_health_check)
        """
        from .v2.monitoring.health_check import ComponentHealth, HealthStatus
        
        def wrapper() -> ComponentHealth:
            result = check_func()
            if isinstance(result, dict):
                # Convert dict to ComponentHealth
                status_str = result.get("status", "unknown")
                status_map = {
                    "healthy": HealthStatus.HEALTHY,
                    "degraded": HealthStatus.DEGRADED,
                    "unhealthy": HealthStatus.UNHEALTHY,
                }
                status = status_map.get(status_str.lower(), HealthStatus.UNHEALTHY)
                return ComponentHealth(
                    name=name,
                    status=status,
                    message=result.get("message", ""),
                    last_check=result.get("last_check", time.time()),
                    check_duration=result.get("check_duration", 0.0),
                )
            elif isinstance(result, ComponentHealth):
                return result
            else:
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message="Invalid health check result",
                )
        
        self._health_checker.register_check(name, wrapper)
    
    def unregister_health_check(self, name: str) -> None:
        """
        Unregister a health check.
        
        Phase 8: HealthChecker API integration.
        
        Args:
            name: Component name to unregister
        """
        self._health_checker.unregister_check(name)
    
    def raise_alert(
        self,
        level: str,
        title: str,
        message: str,
        component: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Raise an alert.
        
        Phase 5.4: Advanced monitoring API.
        
        Args:
            level: Alert level ("info", "warning", "error", "critical")
            title: Alert title
            message: Alert message
            component: Component name
            metadata: Optional metadata
            
        Returns:
            Alert dictionary
        """
        from .v2.monitoring import AlertLevel
        
        level_enum = AlertLevel(level.lower())
        alert = self._alert_manager.raise_alert(
            level=level_enum,
            title=title,
            message=message,
            component=component,
            metadata=metadata,
        )
        return alert.to_dict()
    
    def get_active_alerts(
        self,
        level: Optional[str] = None,
        component: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get active alerts.
        
        Phase 5.4: Advanced monitoring API.
        
        Args:
            level: Filter by level (None = all)
            component: Filter by component (None = all)
            
        Returns:
            List of alert dictionaries
        """
        from .v2.monitoring import AlertLevel
        
        level_enum = AlertLevel(level.lower()) if level else None
        alerts = self._alert_manager.get_active_alerts(
            level=level_enum,
            component=component,
        )
        return [alert.to_dict() for alert in alerts]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """
        Get alert statistics.
        
        Phase 5.4: Advanced monitoring API.
        
        Returns:
            Alert statistics dictionary
        """
        return self._alert_manager.get_alert_stats()
    
    def get_all_alerts(
        self,
        limit: int = 100,
        resolved: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all alerts (active and resolved).
        
        Phase 8: AlertManager API integration.
        
        Args:
            limit: Maximum number of alerts to return
            resolved: Filter by resolved status (None = all, True = resolved only, False = active only)
            
        Returns:
            List of alert dictionaries
        """
        alerts = self._alert_manager.get_all_alerts(limit=limit, resolved=resolved)
        return [alert.to_dict() for alert in alerts]
    
    def resolve_alert(
        self,
        title: str,
        component: Optional[str] = None,
    ) -> bool:
        """
        Resolve an alert.
        
        Phase 8: AlertManager API integration.
        
        Args:
            title: Alert title
            component: Optional component name filter
            
        Returns:
            True if alert was found and resolved, False otherwise
        """
        return self._alert_manager.resolve_alert(title, component)
    
    def register_alert_handler(
        self,
        handler: Callable[[Any], None]
    ) -> None:
        """
        Register custom alert handler.
        
        Phase 8: AlertManager handler system integration.
        
        Args:
            handler: Callback function that receives Alert object
            
        Example:
            def my_alert_handler(alert):
                if alert.level == AlertLevel.CRITICAL:
                    send_email(alert.title, alert.message)
            
            cm.register_alert_handler(my_alert_handler)
        """
        from .v2.monitoring.alerting import Alert
        self._alert_manager.register_handler(handler)
    
    def unregister_alert_handler(
        self,
        handler: Callable[[Any], None]
    ) -> None:
        """
        Unregister alert handler.
        
        Phase 8: AlertManager handler system integration.
        
        Args:
            handler: Callback function to remove
        """
        self._alert_manager.unregister_handler(handler)
    
    def get_performance_metrics(
        self,
        component: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Phase 5.4: Advanced monitoring API.
        
        Args:
            component: Component name (None = all)
            
        Returns:
            Performance metrics dictionary
        """
        if component:
            metrics = self._performance_monitor.get_metrics(component)
            if metrics:
                return metrics.to_dict()
            return {}
        else:
            return self._performance_monitor.get_metrics_summary()
    
    def reset_performance_metrics(self) -> None:
        """
        Reset all performance metrics.
        
        Phase 5.4: Advanced monitoring API.
        """
        self._performance_monitor.reset_metrics()
    
    # --------------------------------------------------------------------- #
    # Phase 5: AIOps Integration & Predictive Analytics API
    # --------------------------------------------------------------------- #
    
    def detect_anomalies(
        self,
        metric_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in performance metrics.
        
        Phase 5: AIOps Integration.
        
        Args:
            metric_name: Specific metric to check (None = check all)
            
        Returns:
            List of anomaly alert dictionaries
        """
        alerts = self._anomaly_detector.detect_anomalies(metric_name)
        return [
            {
                "metric_name": alert.metric_name,
                "anomaly_type": alert.anomaly_type,
                "severity": alert.severity,
                "current_value": alert.current_value,
                "expected_value": alert.expected_value,
                "deviation": alert.deviation,
                "description": alert.description,
                "timestamp": alert.timestamp.isoformat(),
            }
            for alert in alerts
        ]
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """
        Get summary of detected anomalies.
        
        Phase 5: AIOps Integration.
        
        Returns:
            Anomaly summary dictionary
        """
        return self._anomaly_detector.get_anomaly_summary()
    
    def predict_latency(
        self,
        metric_name: str,
        horizon_minutes: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Predict future latency for a metric.
        
        Phase 5: Predictive Analytics.
        
        Args:
            metric_name: Name of the metric
            horizon_minutes: Prediction horizon in minutes (default: 5)
            
        Returns:
            Prediction dictionary or None
        """
        prediction = self._predictive_analytics.predict_latency(metric_name, horizon_minutes)
        if not prediction:
            return None
        
        return {
            "metric_name": prediction.metric_name,
            "predicted_value": prediction.predicted_value,
            "confidence": prediction.confidence,
            "prediction_horizon": prediction.prediction_horizon,
            "upper_bound": prediction.upper_bound,
            "lower_bound": prediction.lower_bound,
            "timestamp": prediction.timestamp.isoformat(),
        }
    
    def predict_error_rate(
        self,
        metric_name: str,
        horizon_minutes: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Predict future error rate.
        
        Phase 5: Predictive Analytics.
        
        Args:
            metric_name: Name of the metric
            horizon_minutes: Prediction horizon in minutes
            
        Returns:
            Prediction dictionary or None
        """
        prediction = self._predictive_analytics.predict_error_rate(metric_name, horizon_minutes)
        if not prediction:
            return None
        
        return {
            "metric_name": prediction.metric_name,
            "predicted_value": prediction.predicted_value,
            "confidence": prediction.confidence,
            "prediction_horizon": prediction.prediction_horizon,
            "upper_bound": prediction.upper_bound,
            "lower_bound": prediction.lower_bound,
            "timestamp": prediction.timestamp.isoformat(),
        }
    
    def get_scaling_recommendations(
        self,
        target_latency: Optional[float] = None,
        target_error_rate: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get auto-scaling recommendations.
        
        Phase 5: Predictive Analytics.
        
        Args:
            target_latency: Target latency threshold (seconds)
            target_error_rate: Target error rate threshold (0.0-1.0)
            
        Returns:
            List of scaling recommendation dictionaries
        """
        recommendations = self._predictive_analytics.recommend_scaling(
            target_latency=target_latency,
            target_error_rate=target_error_rate
        )
        return [
            {
                "action": rec.action,
                "confidence": rec.confidence,
                "reason": rec.reason,
                "estimated_impact": rec.estimated_impact,
                "priority": rec.priority,
            }
            for rec in recommendations
        ]
    
    def analyze_trend(
        self,
        metric_name: str,
        period_minutes: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze performance trend for a metric.
        
        Phase 5: Trend Analysis.
        
        Args:
            metric_name: Name of the metric
            period_minutes: Analysis period in minutes (default: 60)
            
        Returns:
            Trend analysis dictionary or None
        """
        trend = self._trend_analyzer.analyze_trend(metric_name, period_minutes)
        if not trend:
            return None
        
        return {
            "metric_name": trend.metric_name,
            "trend_direction": trend.trend_direction,
            "trend_strength": trend.trend_strength,
            "slope": trend.slope,
            "change_percentage": trend.change_percentage,
            "period": trend.period,
            "timestamp": trend.timestamp.isoformat(),
        }
    
    def get_all_trends(
        self,
        period_minutes: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze trends for all metrics.
        
        Phase 5: Trend Analysis.
        
        Args:
            period_minutes: Analysis period in minutes
            
        Returns:
            Dictionary of metric_name -> trend dictionary
        """
        trends = self._trend_analyzer.get_all_trends(period_minutes)
        return {
            name: {
                "trend_direction": trend.trend_direction,
                "trend_strength": trend.trend_strength,
                "slope": trend.slope,
                "change_percentage": trend.change_percentage,
                "period": trend.period,
                "timestamp": trend.timestamp.isoformat(),
            }
            for name, trend in trends.items()
        }
    
    # --------------------------------------------------------------------- #
    # Configuration Management API (Phase 6.2)
    # ---------------------------------------------------------------------
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get config value using dot notation.
        
        Phase 6.2: Configuration Management API.
        
        Args:
            key: Config key (e.g., "critic.enabled", "tools.enable_tools")
            default: Default value if key not found
            
        Returns:
            Config value
        """
        if hasattr(self, '_config_manager') and self._config_manager:
            return self._config_manager.get(key, default)
        # Fallback to direct access
        keys = key.split('.')
        value = self.cfg
        for k in keys:
            value = getattr(value, k, None)
            if value is None:
                return default
        return value
    
    def set_config_value(self, key: str, value: Any) -> None:
        """
        Set config value (with hot reload if enabled).
        
        Phase 6.2: Configuration Management API.
        
        Args:
            key: Config key (dot notation)
            value: New value
        """
        if hasattr(self, '_config_manager') and self._config_manager:
            self._config_manager.set(key, value)
            # Update local cfg reference
            self.cfg = self._config_manager.get_config()
        else:
            # Direct update (no hot reload)
            keys = key.split('.')
            obj = self.cfg
            for k in keys[:-1]:
                obj = getattr(obj, k)
            setattr(obj, keys[-1], value)
            self.cfg.validate()
    
    def reload_config(self) -> None:
        """
        Reload config from file.
        
        Phase 6.2: Configuration Management API.
        """
        if hasattr(self, '_config_manager') and self._config_manager:
            self._config_manager.reload()
            self.cfg = self._config_manager.get_config()
        else:
            raise ConfigError("ConfigManager not initialized (config_path not provided)")
    
    def register_config_listener(
        self,
        listener: Callable[[Any], None]
    ) -> None:
        """
        Register config change listener.
        
        Phase 6.2: Configuration Management API.
        
        Args:
            listener: Callback function that receives ConfigChangeEvent
        """
        if hasattr(self, '_config_manager') and self._config_manager:
            from .v2.config import ConfigChangeEvent
            self._config_manager.register_listener(listener)
        else:
            raise ConfigError("ConfigManager not initialized (config_path not provided)")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update config with dictionary of changes.
        
        Phase 8: ConfigManager API completion.
        
        Args:
            updates: Dictionary of config updates (can be nested)
            
        Example:
            cm.update_config({
                "critic": {"enabled": False},
                "tools": {"enable_tools": True}
            })
        """
        if hasattr(self, '_config_manager') and self._config_manager:
            self._config_manager.update_config(updates)
            # Update local cfg reference
            self.cfg = self._config_manager.get_config()
        else:
            raise ConfigError("ConfigManager not initialized (config_path not provided)")
    
    def validate_config(self) -> bool:
        """
        Validate current config.
        
        Phase 8: ConfigManager API completion.
        
        Returns:
            True if config is valid
            
        Raises:
            ConfigError: If config is invalid
        """
        if hasattr(self, '_config_manager') and self._config_manager:
            return self._config_manager.validate_config()
        else:
            # Validate local config
            try:
                self.cfg.validate()
                return True
            except Exception as e:
                raise ConfigError(f"Config validation failed: {e}") from e
    
    # --------------------------------------------------------------------- #
    # Phase 6: Performance Optimization API
    # --------------------------------------------------------------------- #
    
    def get_performance_profile(
        self,
        format: str = "summary"
    ) -> str:
        """
        Get performance profiling report.
        
        Phase 6: Performance Optimization.
        
        Args:
            format: Report format ("summary", "detailed", "bottlenecks")
            
        Returns:
            Performance profile report string
        """
        if not hasattr(self._orchestrator, 'performance_profiler') or not self._orchestrator.performance_profiler:
            return "Performance profiling is disabled."
        
        return self._orchestrator.performance_profiler.get_profile_report(format=format)
    
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks.
        
        Phase 6: Performance Optimization.
        
        Returns:
            List of bottleneck dictionaries
        """
        if not hasattr(self._orchestrator, 'performance_profiler') or not self._orchestrator.performance_profiler:
            return []
        
        return self._orchestrator.performance_profiler.identify_bottlenecks()
    
    def get_operation_stats(
        self,
        operation_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific operation.
        
        Phase 6: Performance Optimization.
        
        Args:
            operation_name: Operation name
            
        Returns:
            Operation statistics dictionary or None
        """
        if not hasattr(self._orchestrator, 'performance_profiler') or not self._orchestrator.performance_profiler:
            return None
        
        return self._orchestrator.performance_profiler.get_operation_stats(operation_name)
    
    def warm_cache(
        self,
        strategy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Warm cache using configured strategies.
        
        Phase 6: Performance Optimization.
        
        Args:
            strategy_name: Specific strategy to use (None = use all enabled)
            
        Returns:
            Cache warming results dictionary
        """
        if not self._cache_warmer:
            return {
                "error": "Cache warmer not initialized (enable_cache_warming=True in config)",
                "strategies_executed": 0,
                "keys_warmed": 0,
            }
        
        return self._cache_warmer.warm_cache(strategy_name)
    
    def get_cache_warmer_stats(self) -> Dict[str, Any]:
        """
        Get cache warmer statistics.
        
        Phase 6: Performance Optimization.
        
        Returns:
            Cache warmer statistics dictionary
        """
        if not self._cache_warmer:
            return {"error": "Cache warmer not initialized"}
        
        return self._cache_warmer.get_warming_stats()
    
    def export_config(self, path: Optional[str] = None) -> str:
        """
        Export config to JSON file.
        
        Phase 6.2: Configuration Management API.
        
        Args:
            path: Output path (None = use config_path)
            
        Returns:
            Exported config as JSON string
        """
        if hasattr(self, '_config_manager') and self._config_manager:
            return self._config_manager.export_config(path)
        else:
            # Export current config directly
            import json
            from dataclasses import asdict
            config_dict = asdict(self.cfg)
            json_str = json.dumps(config_dict, indent=2, ensure_ascii=False)
            if path:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(json_str)
            return json_str
    
    # --------------------------------------------------------------------- #
    # Container & Debugging API (Phase 8)
    # --------------------------------------------------------------------- #
    
    def get_registered_services(self) -> List[str]:
        """
        Get list of registered service names in DI container.
        
        Phase 8: Container debugging API.
        
        Returns:
            List of service type names
        """
        if hasattr(self, '_container') and self._container:
            services = self._container.list_services()
            return [s.__name__ if hasattr(s, '__name__') else str(s) for s in services]
        return []
    
    def get_service_registration(self, service_type: Any) -> Optional[Dict[str, Any]]:
        """
        Get service registration information.
        
        Phase 8: Container debugging API.
        
        Args:
            service_type: Service type/interface class
            
        Returns:
            Registration info dictionary or None
        """
        if hasattr(self, '_container') and self._container:
            registration = self._container.get_registration(service_type)
            if registration:
                return {
                    "service_type": registration.service_type.__name__ if hasattr(registration.service_type, '__name__') else str(registration.service_type),
                    "is_singleton": registration.is_singleton,
                    "has_factory": registration.factory is not None,
                    "has_implementation": registration.implementation is not None,
                }
        return None
    
    # --------------------------------------------------------------------- #
    # Tool Management API (Phase 8 - Critical Fix)
    # --------------------------------------------------------------------- #
    
    def register_tool(
        self,
        name: str,
        func: Optional[Callable] = None,
        tool_func: Optional[Callable] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a custom tool.
        
        Phase 8: ToolExecutorV2 API integration.
        
        Args:
            name: Tool name
            func: Tool function (alias for tool_func, for backward compatibility)
            tool_func: Tool function (must accept **kwargs)
            description: Tool description
            parameters: Tool parameters schema (will be converted to schema format)
            schema: Optional tool schema (parameters, description, etc.)
            
        Raises:
            ConfigError: If tools are not enabled
        """
        if not self._tool_executor:
            raise ConfigError("Tools are not enabled. Set cfg.tools.enable_tools = True")
        
        # Support both func and tool_func for backward compatibility
        actual_func = func or tool_func
        if not actual_func:
            raise ValueError("Either 'func' or 'tool_func' must be provided")
        
        # Build schema from parameters if provided
        if schema is None and (description or parameters):
            schema = {}
            if description:
                schema["description"] = description
            if parameters:
                # Validate parameters is a dict
                if not isinstance(parameters, dict):
                    raise ValueError("parameters must be a dictionary")
                schema["parameters"] = {
                    "type": "object",
                    "properties": parameters,
                    "required": [k for k, v in parameters.items() if isinstance(v, dict) and v.get("required", False)]
                }
        
        self._tool_executor.register_tool(name, actual_func, schema)
    
    def list_available_tools(self) -> List[str]:
        """
        List all available tools.
        
        Phase 8: ToolExecutorV2 API integration.
        
        Returns:
            List of available tool names
        """
        if not self._tool_executor:
            return []
        return self._tool_executor.list_available_tools()
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get tool schema.
        
        Phase 8: ToolExecutorV2 API integration.
        
        Args:
            tool_name: Tool name
            
        Returns:
            Tool schema dictionary or None
        """
        if not self._tool_executor:
            return None
        return self._tool_executor.get_tool_schema(tool_name)
    
    def get_tool_metrics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get tool execution metrics.
        
        Phase 8: ToolExecutorV2 API integration.
        
        Args:
            tool_name: Tool name (None = all tools)
            
        Returns:
            Metrics dictionary
        """
        if not self._tool_executor:
            return {}
        return self._tool_executor.get_tool_metrics(tool_name)
    
    # --------------------------------------------------------------------- #
    # ConnectionPool Management API (Phase 8 - Critical Fix)
    # --------------------------------------------------------------------- #
    
    def get_connection_pool_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get connection pool statistics.
        
        Phase 8: ConnectionPool API integration.
        
        Returns:
            Connection pool statistics dictionary or None if pooling disabled
        """
        if not self._connection_pool:
            return None
        return self._connection_pool.get_stats()
    
    def cleanup_idle_connections(self) -> int:
        """
        Clean up idle connections in the pool.
        
        Phase 8: ConnectionPool API integration.
        
        Returns:
            Number of connections cleaned up
        """
        if not self._connection_pool:
            return 0
        return self._connection_pool.cleanup_idle()
    
    # --------------------------------------------------------------------- #
    # PerformanceProfiler API (Phase 8 - Critical Fix)
    # --------------------------------------------------------------------- #
    
    def get_all_performance_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance statistics for all operations.
        
        Phase 8: PerformanceProfiler API integration.
        
        Returns:
            Dictionary of operation name -> statistics
        """
        if not hasattr(self._orchestrator, 'performance_profiler') or not self._orchestrator.performance_profiler:
            return {}
        return self._orchestrator.performance_profiler.get_all_stats()
    
    def clear_performance_profile(self) -> None:
        """
        Clear all performance profile data.
        
        Phase 8: PerformanceProfiler API integration.
        """
        if hasattr(self._orchestrator, 'performance_profiler') and self._orchestrator.performance_profiler:
            self._orchestrator.performance_profiler.clear()
    
    # --------------------------------------------------------------------- #
    # CacheWarmer API (Phase 8 - Critical Fix)
    # --------------------------------------------------------------------- #
    
    def get_cache_warming_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get cache warming statistics.
        
        Phase 8: CacheWarmer API integration.
        
        Returns:
            Cache warming statistics dictionary or None if cache warmer not initialized
        """
        if not self._cache_warmer:
            return None
        return self._cache_warmer.get_warming_stats()
    
    def warm_popular_content(
        self,
        key_access_counts: Dict[str, int],
        top_k: int = 100,
        ttl: Optional[float] = None
    ) -> int:
        """
        Warm cache with popular content based on access counts.
        
        Phase 8: CacheWarmer API integration.
        
        Args:
            key_access_counts: Dictionary of key -> access count
            top_k: Number of top keys to warm
            ttl: Optional TTL for warmed entries
            
        Returns:
            Number of keys warmed
        """
        if not self._cache_warmer:
            return 0
        return self._cache_warmer.warm_popular_content(key_access_counts, top_k=top_k, ttl=ttl)
    
    # --------------------------------------------------------------------- #
    # Memory Management API (Phase 8 - Critical Fix)
    # --------------------------------------------------------------------- #
    
    def add_memory_note(self, text: str) -> None:
        """
        Add a note to episodic memory.
        
        Phase 8: MemoryServiceV2 API integration.
        
        Args:
            text: Note text to add
        """
        if hasattr(self, '_memory_service') and self._memory_service:
            self._memory_service.add_note(text)
    
    def get_memory_notes(self) -> List[str]:
        """
        Get all memory notes.
        
        Phase 8: MemoryServiceV2 API integration.
        
        Returns:
            List of memory notes
        """
        if hasattr(self, '_memory_service') and self._memory_service:
            return self._memory_service.notes()
        return []
    
    def clear_memory_notes(self) -> None:
        """
        Clear all memory notes.
        
        Phase 8: MemoryServiceV2 API integration.
        """
        if hasattr(self, '_memory_service') and self._memory_service:
            self._memory_service.clear_notes()
    
    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from memory using semantic search.
        
        Phase 8: MemoryServiceV2 API integration.
        This method provides a public API for memory retrieval, avoiding
        private attribute access (encapsulation violation).
        
        Args:
            query: Query text for context retrieval
            top_k: Number of relevant items to retrieve (default: 5)
            
        Returns:
            List of relevant context items with metadata and scores.
            Each item contains:
            - content: The text content
            - role: Role of the message (user/assistant)
            - score: Relevance score (if available)
            - metadata: Additional metadata (timestamp, etc.)
        """
        if hasattr(self, '_memory_service') and self._memory_service:
            return self._memory_service.retrieve_context(query, top_k=top_k)
        return []
    
    def get_vector_store_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get vector store statistics.
        
        Phase 8: VectorStore API integration.
        
        Returns:
            Vector store statistics dictionary or None if vector memory disabled
        """
        if hasattr(self, '_memory_service') and self._memory_service:
            # Access vector store through memory service
            if hasattr(self._memory_service, '_vector_store') and self._memory_service._vector_store:
                return self._memory_service._vector_store.get_stats()
        return None
    
    def clear_vector_store(self) -> None:
        """
        Clear all items from vector store.
        
        Phase 8: VectorStore API integration.
        """
        if hasattr(self, '_memory_service') and self._memory_service:
            if hasattr(self._memory_service, '_vector_store') and self._memory_service._vector_store:
                self._memory_service._vector_store.clear()
    
    def delete_vector_store_items(self, ids: List[str]) -> None:
        """
        Delete specific items from vector store by IDs.
        
        Phase 8: VectorStore API integration.
        
        Args:
            ids: List of item IDs to delete
        """
        if hasattr(self, '_memory_service') and self._memory_service:
            if hasattr(self._memory_service, '_vector_store') and self._memory_service._vector_store:
                self._memory_service._vector_store.delete(ids)
    
    # --------------------------------------------------------------------- #
    # Batch Processing API (Phase 8 - Critical Fix)
    # --------------------------------------------------------------------- #
    
    def handle_batch(
        self,
        requests: List[Tuple[CognitiveState, CognitiveInput]],
        decoding: Optional[DecodingConfig] = None
    ) -> List[CognitiveOutput]:
        """
        Handle batch of cognitive requests.
        
        Phase 8: Batch processing API completion.
        
        Args:
            requests: List of (state, request) tuples
            decoding: Optional decoding config override
            
        Returns:
            List of CognitiveOutput
        """
        return self._orchestrator.handle_batch(requests, decoding_config=decoding)
    
    # --------------------------------------------------------------------- #
    # Event Subscription & Publishing API (Phase 8 - Critical Fix)
    # --------------------------------------------------------------------- #
    
    def subscribe_to_events(
        self,
        observer: Any,
        event_type: Optional[str] = None
    ) -> None:
        """
        Subscribe to events from event bus.
        
        Phase 8: EventBus API completion.
        
        Args:
            observer: Event observer (must implement on_event(event: CognitiveEvent) method)
            event_type: Specific event type (None = all events, "*" = all events)
            
        Example:
            class MyObserver:
                def on_event(self, event):
                    print(f"Event: {event.event_type}")
            
            observer = MyObserver()
            cm.subscribe_to_events(observer, "request_received")
        """
        if hasattr(self._orchestrator, 'event_bus'):
            self._orchestrator.event_bus.subscribe(observer, event_type)
    
    def publish_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        source: Optional[str] = None
    ) -> None:
        """
        Publish an event to event bus.
        
        Phase 8: EventBus API completion.
        
        Args:
            event_type: Event type (e.g., "request_received", "response_generated", "custom_event")
            data: Event data dictionary
            source: Optional source identifier (default: "CognitiveManager")
            
        Example:
            cm.publish_event(
                event_type="custom_event",
                data={"message": "Hello", "timestamp": time.time()},
                source="MyApplication"
            )
        """
        if hasattr(self._orchestrator, 'event_bus'):
            from .v2.events.event_bus import CognitiveEvent
            event = CognitiveEvent(
                event_type=event_type,
                data=data,
                source=source or "CognitiveManager"
            )
            self._orchestrator.event_bus.publish(event)


__all__ = ["CognitiveManager", "ModelAPI"]
