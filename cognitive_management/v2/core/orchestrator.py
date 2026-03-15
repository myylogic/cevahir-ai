# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: orchestrator.py
Modül: cognitive_management/v2/core
Görev: Cognitive Orchestrator - V2 Cognitive Management orchestrator. Full enterprise
       integration with monitoring, tracing, and performance tracking. Policy routing,
       deliberation, memory management, critic ve tool execution işlemlerini koordine
       eder. Event bus, processing pipeline ve async pipeline entegrasyonu sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (orchestration), Dependency Inversion
                     (interface'lere bağımlı)
- Design Patterns: Orchestrator Pattern (bilişsel yönetim koordinasyonu)
- Endüstri Standartları: Enterprise orchestration best practices

KULLANIM:
- Cognitive management orchestration için
- Policy routing için
- Deliberation için
- Memory management için

BAĞIMLILIKLAR:
- EventBus: Event yönetimi
- ProcessingPipeline: Processing işlemleri
- Component Protocols: Component interface'leri

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import asyncio

from cognitive_management.cognitive_types import (
    CognitiveState,
    CognitiveInput,
    CognitiveOutput,
    DecodingConfig,
)
from ..events.event_bus import EventBus, CognitiveEvent
from ..interfaces.backend_protocols import FullModelBackend
from ..interfaces.component_protocols import (
    PolicyRouter as IPolicyRouter,
    MemoryService as IMemoryService,
    DeliberationEngine as IDeliberationEngine,
    Critic as ICritic,
    ToolExecutor,
)
from ..processing.pipeline import ProcessingPipeline
from ..processing.async_pipeline import AsyncProcessingPipeline
from ..middleware.base import Middleware, BaseMiddleware
from ..middleware.async_middleware import AsyncMiddleware, BaseAsyncMiddleware

# Phase 5.4: Performance monitoring
try:
    from ..monitoring.performance_monitor import PerformanceMonitor
    _PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    _PERFORMANCE_MONITORING_AVAILABLE = False


class CognitiveOrchestrator:
    """
    Central orchestrator for V2 cognitive management.
    
    Phase 5: Full enterprise integration.
    
    Responsibilities:
    - Request orchestration
    - Middleware chain execution
    - Pipeline processing
    - Event publishing
    - Performance monitoring
    - Error handling
    """
    
    def __init__(
        self,
        backend: FullModelBackend,
        policy_router: IPolicyRouter,
        memory_service: IMemoryService,
        critic: ICritic,
        deliberation_engine: Optional[IDeliberationEngine] = None,
        event_bus: Optional[EventBus] = None,
        middleware: Optional[list[Middleware]] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
        tool_executor: Optional[ToolExecutor] = None,
    ):
        """
        Initialize orchestrator.
        
        Args:
            backend: Model backend
            policy_router: Policy router
            memory_service: Memory service
            critic: Critic component
            deliberation_engine: Deliberation engine (optional)
            event_bus: Event bus (optional)
            middleware: Middleware chain (optional)
            performance_monitor: Performance monitor (optional, Phase 5.4)
        """
        self.backend = backend
        self.policy_router = policy_router
        self.memory_service = memory_service
        self.critic = critic
        self.deliberation_engine = deliberation_engine
        self.event_bus = event_bus or EventBus()
        self.tool_executor = tool_executor  # Phase 6.1: Tool Policy Implementation
        
        # Build middleware chain
        self.middleware_chain = self._build_middleware_chain(middleware or [])
        
        # Build processing pipeline
        self.pipeline = self._build_pipeline()
        
        # Build async pipeline (lazy initialization)
        self._async_pipeline: Optional[AsyncProcessingPipeline] = None
        self._async_middleware_chain: Optional[AsyncMiddleware] = None
        
        # Phase 5.4: Performance monitoring
        self.performance_monitor = performance_monitor
        if _PERFORMANCE_MONITORING_AVAILABLE and not self.performance_monitor:
            self.performance_monitor = PerformanceMonitor()
        
        # Phase 6: Performance Profiler (for bottleneck detection)
        from ..utils.performance_profiler import PerformanceProfiler
        if hasattr(policy_router, 'cfg') and policy_router.cfg:
            enable_profiling = getattr(policy_router.cfg.runtime, 'enable_profiling', False)
        else:
            enable_profiling = False
        self.performance_profiler = PerformanceProfiler(enabled=enable_profiling) if enable_profiling else None
        
        # Phase 5: AIOps - Auto anomaly checking flag
        self._check_anomalies = False  # Will be set by CognitiveManager if anomaly detector available
        self._anomaly_detector = None  # Will be set by CognitiveManager
        
        # Phase 8: RequestBatcher initialization (Critical Fix)
        self._request_batcher = None
        # Note: RequestBatcher callback-based pattern orchestrator'un senkron pattern'i ile uyumsuz
        # Bu yüzden şimdilik None olarak bırakıyoruz - sequential processing kullanıyoruz
        # Gelecekte async batch processing için uygun bir wrapper yazılabilir
    
    def _build_pipeline(self) -> ProcessingPipeline:
        """
        Build processing pipeline.
        Chain of Responsibility pattern.
        
        Returns:
            ProcessingPipeline
        """
        from ..processing.handlers import (
            FeatureExtractionHandler,
            PolicyRoutingHandler,
            DeliberationHandler,
            ContextBuildingHandler,
            GenerationHandler,
            SelfConsistencyHandler,
            CriticHandler,
            MemoryUpdateHandler,
        )
        
        # Build handler chain
        handlers = [
            FeatureExtractionHandler(
                memory_service=self.memory_service,
                backend=self.backend,  # Phase 9: Model logit-based entropy
            ),
            PolicyRoutingHandler(
                policy_router=self.policy_router,
            ),
        ]
        
        # Optional: Deliberation
        if self.deliberation_engine:
            # Phase 4: Tree of Thoughts support (lazy initialization)
            tree_of_thoughts = None
            if hasattr(self.policy_router, 'cfg') and self.policy_router.cfg.policy.tot_enabled:
                try:
                    from ..components.tree_of_thoughts import TreeOfThoughts
                    # Use backend as ModelAPI for ToT
                    tree_of_thoughts = TreeOfThoughts(
                        cfg=self.policy_router.cfg,
                        model_api=self.backend,  # Backend implements ModelAPI
                        max_depth=self.policy_router.cfg.policy.tot_max_depth,
                        branching_factor=self.policy_router.cfg.policy.tot_branching_factor,
                        top_k=self.policy_router.cfg.policy.tot_top_k,
                    )
                except Exception as e:
                    import logging
                    logging.warning(f"TreeOfThoughts initialization failed: {e}")
                    tree_of_thoughts = None
            
            handlers.append(
                DeliberationHandler(
                    engine=self.deliberation_engine,
                    backend=self.backend,
                    cfg=self.policy_router.cfg if hasattr(self.policy_router, 'cfg') else None,
                    tree_of_thoughts=tree_of_thoughts,
                )
            )
        
        # Tool policy (Phase 6.1: Tool Policy Implementation)
        tool_policy = None
        if self.tool_executor:
            from ..components.tool_policy_v2 import ToolPolicyV2
            from ..interfaces.component_protocols import ToolExecutor as IToolExecutor
            # Get config from policy router (it has cfg)
            if hasattr(self.policy_router, 'cfg'):
                cfg = self.policy_router.cfg
            else:
                # Fallback: create default config
                from cognitive_management.config import CognitiveManagerConfig
                cfg = CognitiveManagerConfig()
            tool_policy = ToolPolicyV2(cfg, self.tool_executor)
        
        # Config erişimi (SelfConsistency için gerekli)
        _cfg = self.policy_router.cfg if hasattr(self.policy_router, "cfg") else None

        # Self-Consistency etkin mi?
        _sc_enabled = (
            _cfg is not None
            and getattr(getattr(_cfg, "policy", None), "self_consistency_enabled", False)
        )

        # Core handlers
        core = [
            ContextBuildingHandler(
                memory_service=self.memory_service,
                tool_policy=tool_policy,
            ),
            GenerationHandler(
                backend=self.backend,
            ),
        ]

        # Wang et al. 2022 — Self-Consistency (config'de etkinse pipeline'a eklenir)
        if _sc_enabled:
            core.append(
                SelfConsistencyHandler(
                    backend=self.backend,
                    cfg=_cfg,
                )
            )

        core.extend([
            CriticHandler(
                critic=self.critic,
            ),
            MemoryUpdateHandler(
                memory_service=self.memory_service,
            ),
        ])
        handlers.extend(core)
        
        return ProcessingPipeline(handlers)
    
    def _build_middleware_chain(
        self,
        middleware: list[Middleware]
    ) -> Optional[Middleware]:
        """
        Build middleware chain.
        
        Args:
            middleware: List of middleware instances
            
        Returns:
            First middleware in chain (None if empty)
        """
        if not middleware:
            return None
        
        # Build chain
        for i in range(len(middleware) - 1):
            if isinstance(middleware[i], BaseMiddleware):
                middleware[i].set_next(middleware[i + 1])
        
        return middleware[0] if middleware else None
    
    def _process_batch_requests(
        self,
        batch_requests: list[tuple[CognitiveState, CognitiveInput]]
    ) -> list[CognitiveOutput]:
        """
        Process batch of requests.
        
        Phase 8: Batch processing support.
        
        Args:
            batch_requests: List of (state, request) tuples
            
        Returns:
            List of CognitiveOutput
        """
        results = []
        for state, request in batch_requests:
            try:
                # Process each request through normal pipeline
                response = self.handle(state, request)
                results.append(response)
            except Exception as e:
                # Create error response for failed request
                from cognitive_management.cognitive_types import CognitiveOutput
                error_response = CognitiveOutput(
                    text=f"Error: {str(e)}",
                    used_mode="direct",
                    revised_by_critic=False,
                )
                results.append(error_response)
        return results
    
    def handle_batch(
        self,
        requests: list[tuple[CognitiveState, CognitiveInput]],
        decoding_config: Optional[DecodingConfig] = None,
    ) -> list[CognitiveOutput]:
        """
        Handle batch of cognitive requests.
        
        Phase 8: Batch processing support.
        
        Args:
            requests: List of (state, request) tuples
            decoding_config: Optional decoding config override
            
        Returns:
            List of CognitiveOutput
        """
        # Phase 8: Currently using sequential processing
        # RequestBatcher callback-based pattern orchestrator'un senkron pattern'i ile uyumsuz
        # Sequential processing is more suitable for orchestrator's synchronous design
        return self._process_batch_requests(requests)
    
    def handle(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        *,
        decoding_config: Optional[DecodingConfig] = None,
    ) -> CognitiveOutput:
        """
        Handle cognitive request.
        
        Phase 5: Full enterprise integration with monitoring.
        
        Args:
            state: Cognitive state
            request: Cognitive input
            decoding_config: Optional decoding config override
            
        Returns:
            Cognitive output
        """
        import time
        start_time = time.time()
        operation_name = "cognitive_request"
        
        # Phase 6: Performance profiling (if enabled)
        profiler = getattr(self, 'performance_profiler', None)
        if profiler:
            profiler_context = profiler.profile_operation(operation_name)
            profiler_context.__enter__()
        else:
            profiler_context = None
        
        try:
            # Apply middleware before processing
            if self.middleware_chain:
                try:
                    state, request = self.middleware_chain.before(state, request)
                except Exception as e:
                    # Middleware error handling
                    error_response = self.middleware_chain.on_error(state, request, e)
                    if error_response:
                        # Record error in performance monitor
                        if self.performance_monitor:
                            latency = time.time() - start_time
                            self.performance_monitor.record_operation(
                                operation_name,
                                latency,
                                success=False,
                            )
                        return error_response
                    raise
            
            # Phase 8: Check for cached response (from CacheMiddleware)
            if request.metadata.get("_cache_hit", False) and "_cached_response" in request.metadata:
                cached_response = request.metadata["_cached_response"]
                # Ensure it's a CognitiveOutput
                if isinstance(cached_response, CognitiveOutput):
                    # Record cache hit in performance monitor
                    if self.performance_monitor:
                        latency = time.time() - start_time
                        self.performance_monitor.record_operation(
                            operation_name,
                            latency,
                            success=True,
                        )
                    return cached_response
            
            # Publish request event
            self.event_bus.publish(
                CognitiveEvent(
                    event_type="request_received",
                    data={
                        "user_message": request.user_message,
                        "system_prompt": request.system_prompt,
                    },
                    source="CognitiveOrchestrator",
                )
            )
            
            # Process through pipeline
            try:
                response = self.pipeline.process(
                    state=state,
                    request=request,
                    decoding_config=decoding_config,
                )
            except Exception as e:
                # Pipeline error handling
                if self.middleware_chain:
                    error_response = self.middleware_chain.on_error(state, request, e)
                    if error_response:
                        # Record error in performance monitor
                        if self.performance_monitor:
                            latency = time.time() - start_time
                            self.performance_monitor.record_operation(
                                operation_name,
                                latency,
                                success=False,
                            )
                        return error_response
                raise
            
            # Apply middleware after processing
            if self.middleware_chain:
                response = self.middleware_chain.after(state, request, response)
            
            # Publish response event
            self.event_bus.publish(
                CognitiveEvent(
                    event_type="response_generated",
                    data={
                        "text": response.text,
                        "mode": response.used_mode,
                        "tool_used": response.tool_used,
                        "revised": response.revised_by_critic,
                    },
                    source="CognitiveOrchestrator",
                )
            )
            
            # Record success in performance monitor
            if self.performance_monitor:
                latency = time.time() - start_time
                self.performance_monitor.record_operation(
                    operation_name,
                    latency,
                    success=True,
                )
            
            # Phase 5: AIOps - Automatic anomaly detection (if enabled)
            if getattr(self, '_check_anomalies', False) and hasattr(self, '_anomaly_detector') and self._anomaly_detector:
                try:
                    anomalies = self._anomaly_detector.detect_anomalies(operation_name)
                    if anomalies:
                        # Log critical anomalies
                        critical_anomalies = [a for a in anomalies if a.severity in ("high", "critical")]
                        if critical_anomalies:
                            import logging
                            logging.warning(f"Critical anomalies detected: {[a.description for a in critical_anomalies]}")
                except Exception:
                    pass  # Non-critical - don't fail request
            
            return response
            
        except Exception as e:
            # Record error in performance monitor
            if self.performance_monitor:
                latency = time.time() - start_time
                self.performance_monitor.record_operation(
                    operation_name,
                    latency,
                    success=False,
                )
            raise
        finally:
            # Phase 6: End profiling
            if profiler_context:
                try:
                    profiler_context.__exit__(None, None, None)
                except Exception:
                    pass
    
    def _build_async_pipeline(self) -> AsyncProcessingPipeline:
        """
        Build async processing pipeline.
        Chain of Responsibility pattern (async).
        
        Returns:
            AsyncProcessingPipeline
        """
        from ..processing.async_handlers import (
            AsyncFeatureExtractionHandler,
            AsyncPolicyRoutingHandler,
            AsyncDeliberationHandler,
            AsyncContextBuildingHandler,
            AsyncGenerationHandler,
            AsyncSelfConsistencyHandler,
            AsyncCriticHandler,
            AsyncMemoryUpdateHandler,
        )
        
        # Build async handler chain
        handlers = [
            AsyncFeatureExtractionHandler(
                memory_service=self.memory_service,
                backend=self.backend,  # Phase 9: Model logit-based entropy
            ),
            AsyncPolicyRoutingHandler(
                policy_router=self.policy_router,
            ),
        ]
        
        # Optional: Deliberation
        if self.deliberation_engine:
            handlers.append(
                AsyncDeliberationHandler(
                    engine=self.deliberation_engine,
                    backend=self.backend,
                )
            )
        
        # Tool policy (Phase 6.1: Tool Policy Implementation)
        tool_policy = None
        if self.tool_executor:
            from ..components.tool_policy_v2 import ToolPolicyV2
            # Get config from policy router (it has cfg)
            if hasattr(self.policy_router, 'cfg'):
                cfg = self.policy_router.cfg
            else:
                # Fallback: create default config
                from cognitive_management.config import CognitiveManagerConfig
                cfg = CognitiveManagerConfig()
            tool_policy = ToolPolicyV2(cfg, self.tool_executor)
        
        # Config erişimi
        _cfg_async = self.policy_router.cfg if hasattr(self.policy_router, "cfg") else None
        _sc_enabled_async = (
            _cfg_async is not None
            and getattr(getattr(_cfg_async, "policy", None), "self_consistency_enabled", False)
        )

        # Core async handlers
        async_core = [
            AsyncContextBuildingHandler(
                memory_service=self.memory_service,
                tool_policy=tool_policy,  # Phase 6.1: Tool Policy Implementation
            ),
            AsyncGenerationHandler(
                backend=self.backend,
            ),
        ]

        # Wang et al. 2022 — Async Self-Consistency
        if _sc_enabled_async:
            async_core.append(
                AsyncSelfConsistencyHandler(
                    backend=self.backend,
                    cfg=_cfg_async,
                )
            )

        async_core.extend([
            AsyncCriticHandler(
                critic=self.critic,
            ),
            AsyncMemoryUpdateHandler(
                memory_service=self.memory_service,
            ),
        ])
        handlers.extend(async_core)
        
        return AsyncProcessingPipeline(handlers)
    
    def _build_async_middleware_chain(
        self,
        middleware: list[Middleware]
    ) -> Optional[AsyncMiddleware]:
        """
        Build async middleware chain.
        Converts sync middleware to async if needed.
        
        Args:
            middleware: List of middleware instances
            
        Returns:
            First async middleware in chain (None if empty)
        """
        if not middleware:
            return None
        
        # Convert sync middleware to async wrapper if needed
        async_middleware = []
        for mw in middleware:
            if isinstance(mw, BaseAsyncMiddleware):
                async_middleware.append(mw)
            else:
                # Wrap sync middleware in async adapter
                from ..middleware.async_middleware import SyncToAsyncMiddlewareAdapter
                async_middleware.append(SyncToAsyncMiddlewareAdapter(mw))
        
        # Build chain
        for i in range(len(async_middleware) - 1):
            async_middleware[i].set_next(async_middleware[i + 1])
        
        return async_middleware[0] if async_middleware else None
    
    async def handle_async(
        self,
        state: CognitiveState,
        request: CognitiveInput,
        *,
        decoding_config: Optional[DecodingConfig] = None,
    ) -> CognitiveOutput:
        """
        Handle cognitive request asynchronously.
        
        Phase 5.1: Async support for concurrent processing.
        Phase 5.4: Performance monitoring integration.
        
        Args:
            state: Cognitive state
            request: Cognitive input
            decoding_config: Optional decoding config override
            
        Returns:
            Cognitive output
        """
        import time
        start_time = time.time()
        operation_name = "cognitive_request_async"
        
        # Lazy initialization of async pipeline
        if self._async_pipeline is None:
            self._async_pipeline = self._build_async_pipeline()
        
        # Build async middleware chain if needed
        if self._async_middleware_chain is None:
            # Get middleware from sync chain
            middleware_list = []
            current = self.middleware_chain
            while current:
                middleware_list.append(current)
                current = getattr(current, '_next', None)
            self._async_middleware_chain = self._build_async_middleware_chain(middleware_list)
        
        try:
            # Apply async middleware before processing
            if self._async_middleware_chain:
                try:
                    state, request = await self._async_middleware_chain.before_async(state, request)
                except Exception as e:
                    # Middleware error handling
                    error_response = await self._async_middleware_chain.on_error_async(state, request, e)
                    if error_response:
                        # Record error in performance monitor
                        if self.performance_monitor:
                            latency = time.time() - start_time
                            self.performance_monitor.record_operation(
                                operation_name,
                                latency,
                                success=False,
                            )
                        return error_response
                    raise
            
            # Publish request event (async)
            await asyncio.to_thread(
                self.event_bus.publish,
                CognitiveEvent(
                    event_type="request_received_async",
                    data={
                        "user_message": request.user_message,
                        "system_prompt": request.system_prompt,
                    },
                    source="CognitiveOrchestrator",
                )
            )
            
            # Process through async pipeline
            try:
                response = await self._async_pipeline.process_async(
                    state=state,
                    request=request,
                    decoding_config=decoding_config,
                )
            except Exception as e:
                # Pipeline error handling
                if self._async_middleware_chain:
                    error_response = await self._async_middleware_chain.on_error_async(state, request, e)
                    if error_response:
                        # Record error in performance monitor
                        if self.performance_monitor:
                            latency = time.time() - start_time
                            self.performance_monitor.record_operation(
                                operation_name,
                                latency,
                                success=False,
                            )
                        return error_response
                raise
            
            # Apply async middleware after processing
            if self._async_middleware_chain:
                response = await self._async_middleware_chain.after_async(state, request, response)
            
            # Publish response event (async)
            await asyncio.to_thread(
                self.event_bus.publish,
                CognitiveEvent(
                    event_type="response_generated_async",
                    data={
                        "text": response.text,
                        "mode": response.used_mode,
                        "tool_used": response.tool_used,
                        "revised": response.revised_by_critic,
                    },
                    source="CognitiveOrchestrator",
                )
            )
            
            # Record success in performance monitor
            if self.performance_monitor:
                latency = time.time() - start_time
                self.performance_monitor.record_operation(
                    operation_name,
                    latency,
                    success=True,
                )
            
            return response
            
        except Exception as e:
            # Record error in performance monitor
            if self.performance_monitor:
                latency = time.time() - start_time
                self.performance_monitor.record_operation(
                    operation_name,
                    latency,
                    success=False,
                )
            raise


__all__ = ["CognitiveOrchestrator"]
