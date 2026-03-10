# 🧠 Cognitive Management V2 - Kapsamlı Dokümantasyon

**Versiyon:** 2.0 (Advanced Enterprise)  
**Son Güncelleme:** 2025-01-27  
**Durum:** ✅ Production-Ready | Endüstri Standartları | Enterprise-Grade

---

## 📋 İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Mimari Yapı](#mimari-yapı)
3. [Çalışma Prensibi](#çalışma-prensibi)
4. [Core API](#core-api)
5. [Enterprise Features](#enterprise-features)
6. [V2 Components](#v2-components)
7. [Cognitive Patterns](#cognitive-patterns)
8. [Memory Management](#memory-management)
9. [Tool Management](#tool-management)
10. [Configuration Management](#configuration-management)
11. [Monitoring & Observability](#monitoring--observability)
12. [AIOps & Predictive Analytics](#aiops--predictive-analytics)
13. [Performance Optimization](#performance-optimization)
14. [API Referansı](#api-referansı)
15. [Kullanım Örnekleri](#kullanım-örnekleri)
16. [Best Practices](#best-practices)

---

## 🎯 Genel Bakış

**Cognitive Management**, Cevahir Sinir Sistemi'nin bilişsel üst katmanını yöneten enterprise-grade bir modüldür. Chat/inference akışında strateji seçimi, iç ses üretimi, araç kullanımı, eleştirel kontrol ve bellek yönetimini tek bir koordinatörde birleştirir.

### Temel Özellikler

- ✅ **V2 Orchestrator Pattern:** SOLID principles, Dependency Injection, Clean Architecture
- ✅ **Cognitive Patterns:** Chain of Thought, Tree of Thoughts, Constitutional AI, System 1/2 Thinking
- ✅ **Enterprise Features:** Monitoring, Caching, Tracing, Alerting, Health Checks
- ✅ **Memory Management:** Episodic, Working, Vector Memory (RAG), Memory Pruning
- ✅ **Tool Management:** Dynamic tool registration, execution tracking, metrics
- ✅ **Multimodal Support:** Text, audio, image processing
- ✅ **AIOps Integration:** Anomaly detection, predictive analytics, trend analysis
- ✅ **Performance Optimization:** Cache warming, connection pooling, batch processing
- ✅ **Configuration Management:** Hot reload, config listeners, validation
- ✅ **Event System:** Event bus, subscribers, metrics tracking

### Modül Bileşenleri

1. **CognitiveManager** - Ana orchestrator (~2050 satır)
2. **CognitiveOrchestrator** - V2 core orchestrator
3. **V2 Components:**
   - PolicyRouterV2 - Strateji seçimi
   - MemoryServiceV2 - Bellek yönetimi
   - CriticV2 - Eleştirel kontrol
   - DeliberationEngineV2 - İç ses üretimi
   - ToolExecutorV2 - Araç yönetimi
4. **Middleware Chain:**
   - ValidationMiddleware
   - TracingMiddleware
   - CacheMiddleware
   - ErrorHandlingMiddleware
   - MetricsMiddleware
5. **Monitoring Components:**
   - HealthChecker
   - AlertManager
   - PerformanceMonitor
   - AnomalyDetector
   - PredictiveAnalytics
   - TrendAnalyzer

---

## 🏗️ Mimari Yapı

### Dosya Organizasyonu

```
cognitive_management/
├── __init__.py
├── cognitive_manager.py          # Ana orchestrator (~2050 satır)
├── config.py                     # Configuration management (~414 satır)
├── cognitive_types.py            # Type definitions
├── exceptions.py                 # Exception classes
├── utils/                        # Utility modules
│   ├── logging.py
│   └── timers.py
├── v2/                           # V2 Implementation
│   ├── core/
│   │   └── orchestrator.py       # Core orchestrator
│   ├── components/               # Cognitive components
│   │   ├── policy_router_v2.py
│   │   ├── memory_service_v2.py
│   │   ├── critic_v2.py
│   │   ├── deliberation_engine_v2.py
│   │   ├── tool_executor_v2.py
│   │   ├── tree_of_thoughts.py   # Tree of Thoughts pattern
│   │   ├── constitutional_critic.py  # Constitutional AI
│   │   ├── rag_enhancer.py       # RAG enhancement
│   │   ├── embedding_adapter.py  # Embedding providers
│   │   ├── vector_store/         # Vector store implementations
│   │   └── fact_checkers/        # External fact-checking
│   ├── middleware/               # Middleware components
│   │   ├── validation.py
│   │   ├── tracing.py
│   │   ├── cache.py
│   │   ├── error_handler.py
│   │   └── metrics.py
│   ├── monitoring/               # Monitoring & observability
│   │   ├── health_check.py
│   │   ├── alerting.py
│   │   ├── performance_monitor.py
│   │   ├── anomaly_detector.py
│   │   ├── predictive_analytics.py
│   │   └── trend_analyzer.py
│   ├── events/                   # Event system
│   │   ├── event_bus.py
│   │   └── event_handlers.py
│   ├── processing/               # Processing pipelines
│   │   ├── pipeline.py
│   │   ├── async_pipeline.py
│   │   └── handlers.py
│   ├── config/                   # Configuration management
│   │   ├── config_manager.py
│   │   └── constitutional_principles.py
│   ├── container/                # Dependency injection
│   │   └── dependency_container.py
│   ├── adapters/                 # Adapters
│   │   └── backend_adapter.py
│   ├── interfaces/               # Protocol definitions
│   │   ├── backend_protocols.py
│   │   └── component_protocols.py
│   └── utils/                    # V2 utilities
│       ├── cache_warming.py
│       ├── semantic_cache.py
│       ├── performance_profiler.py
│       ├── connection_pool.py
│       └── tracing.py
└── docs/                         # Documentation (moved to docs/modules/)
```

### Mimari Katmanlar

```
┌─────────────────────────────────────────────────────────────┐
│                  CognitiveManager                           │
│              (Public API - 91+ Methods)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │  CognitiveOrchestrator  │
        │    (V2 Core Engine)     │
        └────────────┬────────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
┌───▼────┐  ┌────────▼────────┐  ┌───▼──────────┐
│Policy  │  │   Memory        │  │   Critic     │
│Router  │  │   Service       │  │              │
└────────┘  └─────────────────┘  └──────────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
┌───▼────┐  ┌────────▼────────┐  ┌───▼──────────┐
│Deliber │  │   Tool          │  │  Vector      │
│Engine  │  │   Executor      │  │  Store       │
└────────┘  └─────────────────┘  └──────────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
┌───▼────┐  ┌────────▼────────┐  ┌───▼──────────┐
│Middle  │  │   Monitoring    │  │   Events     │
│ware    │  │                 │  │              │
└────────┘  └─────────────────┘  └──────────────┘
```

---

## ⚙️ Çalışma Prensibi

### 1. Request Processing Flow

```python
CognitiveManager.handle()
    ↓
1. Validation (ValidationMiddleware)
    ├── Input validation
    ├── Length checks
    └── Type validation
    ↓
2. Tracing (TracingMiddleware - optional)
    ├── Trace ID generation
    ├── Span creation
    └── Context propagation
    ↓
3. Cache Check (CacheMiddleware - optional)
    ├── Semantic cache lookup
    ├── Cache hit/miss
    └── Cache storage
    ↓
4. Error Handling (ErrorHandlingMiddleware)
    ├── Retry logic
    ├── Circuit breaker
    └── Timeout handling
    ↓
5. Orchestrator Processing
    ├── Feature Extraction
    ├── Policy Routing (direct/think/debate/tot)
    ├── Deliberation (if needed)
    ├── Context Building (RAG, Memory)
    ├── Generation
    ├── Critic Review
    └── Memory Update
    ↓
6. Metrics (MetricsMiddleware)
    ├── Latency tracking
    ├── Success/error rates
    └── Mode-specific metrics
    ↓
7. Response Return
```

### 2. Policy Routing Flow

```python
PolicyRouterV2.route()
    ↓
1. Feature Extraction
    ├── Entropy calculation
    ├── Length analysis
    ├── Risk assessment
    └── Tool need detection
    ↓
2. Policy Decision
    ├── Direct Mode (low entropy, simple)
    ├── Think Mode (medium entropy, CoT)
    ├── Debate Mode (high entropy, multi-perspective)
    └── Tree of Thoughts (complex problems)
    ↓
3. Decoding Config Adjustment
    ├── Temperature tuning
    ├── Max tokens adjustment
    └── Sampling parameters
```

### 3. Memory Management Flow

```python
MemoryServiceV2.process()
    ↓
1. Context Retrieval
    ├── Episodic memory search
    ├── Vector memory search (RAG)
    ├── Hybrid search (if enabled)
    └── Relevance scoring
    ↓
2. Context Building
    ├── History pruning
    ├── Summary generation (if needed)
    └── Context assembly
    ↓
3. Memory Update (after generation)
    ├── Episodic memory storage
    ├── Vector embedding (if enabled)
    └── Memory pruning (if needed)
```

---

## 🎯 Core API

### 1. Request Handling

#### `handle()`

Ana senkron request handling metodu.

```python
def handle(
    self,
    state: CognitiveState,
    request: CognitiveInput,
    *,
    decoding: Optional[DecodingConfig] = None,
) -> CognitiveOutput:
    """
    Bir kullanıcı mesajını işleyip nihai yanıtı üretir.
    V2 Orchestrator kullanır (sync).
    
    Args:
        state: Cognitive state (conversation context)
        request: Cognitive input (user message)
        decoding: Optional decoding config override
        
    Returns:
        CognitiveOutput (text, metadata, mode, etc.)
    """
```

**Örnek:**

```python
from cognitive_management import CognitiveManager
from cognitive_management.cognitive_types import CognitiveState, CognitiveInput

cm = CognitiveManager(model_manager)

state = CognitiveState()
request = CognitiveInput(user_message="Merhaba, nasılsın?")

response = cm.handle(state, request)
print(response.text)  # Generated response
print(response.used_mode)  # "direct", "think", "debate", or "tot"
print(response.tool_used)  # Tool name if tool was used
print(response.revised_by_critic)  # True if critic revised
```

#### `handle_async()`

Asenkron request handling metodu.

```python
async def handle_async(
    self,
    state: CognitiveState,
    request: CognitiveInput,
    *,
    decoding: Optional[DecodingConfig] = None,
) -> CognitiveOutput:
    """Asenkron olarak bir kullanıcı mesajını işler."""
```

**Örnek:**

```python
import asyncio

async def main():
    response = await cm.handle_async(state, request)
    print(response.text)

asyncio.run(main())
```

#### `handle_multimodal()`

Çok modaliteli girdi işleme (metin, ses, görüntü).

```python
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
    Çok modaliteli girdi işleme.
    
    Args:
        state: Cognitive state
        text: Optional text input
        audio: Optional audio bytes
        image: Optional image bytes
        decoding: Optional decoding config
        
    Returns:
        CognitiveOutput
    """
```

**Örnek:**

```python
# Text + Image
with open("image.jpg", "rb") as f:
    image_data = f.read()

response = cm.handle_multimodal(
    state,
    text="Bu resimde ne görüyorsun?",
    image=image_data
)

# Audio transcription + processing
with open("audio.wav", "rb") as f:
    audio_data = f.read()

response = cm.handle_multimodal(
    state,
    audio=audio_data
)
```

#### `handle_batch()`

Batch request processing.

```python
def handle_batch(
    self,
    requests: List[Tuple[CognitiveState, CognitiveInput]],
    decoding: Optional[DecodingConfig] = None
) -> List[CognitiveOutput]:
    """
    Handle batch of cognitive requests.
    
    Args:
        requests: List of (state, request) tuples
        decoding: Optional decoding config override
        
    Returns:
        List of CognitiveOutput
    """
```

**Örnek:**

```python
batch_requests = [
    (state1, CognitiveInput(user_message="Sorularım var")),
    (state2, CognitiveInput(user_message="Nasıl çalışır?")),
    (state3, CognitiveInput(user_message="Örnek ver")),
]

responses = cm.handle_batch(batch_requests)
for i, response in enumerate(responses):
    print(f"Response {i+1}: {response.text}")
```

---

## 🏢 Enterprise Features

### 1. Metrics & Monitoring

#### `get_metrics()`

Performance metrics al.

```python
def get_metrics(self) -> Dict[str, Any]:
    """
    Get performance metrics.
    
    Returns:
        Metrics dictionary with global and mode-specific metrics
    """
```

**Örnek:**

```python
metrics = cm.get_metrics()

print(metrics["global"])
# {
#     "request_count": 1000,
#     "success_rate": 0.95,
#     "error_count": 50,
#     "avg_latency": 0.45,
#     "last_request_time": "2025-01-27T10:30:00"
# }

print(metrics["direct"])
# {
#     "count": 700,
#     "avg_latency": 0.3,
#     "success_rate": 0.98
# }

print(metrics["think"])
# {
#     "count": 250,
#     "avg_latency": 1.2,
#     "success_rate": 0.92
# }
```

#### `reset_metrics()`

Tüm metrikleri sıfırla.

```python
def reset_metrics(self) -> None:
    """Reset all metrics"""
```

### 2. Health Checks

#### `get_health_status()`

Kapsamlı sistem sağlık durumu.

```python
def get_health_status(self) -> Dict[str, Any]:
    """
    Get comprehensive health status of the cognitive system.
    
    Returns:
        Health status dictionary
    """
```

**Örnek:**

```python
health = cm.get_health_status()

print(health["overall"])  # "healthy", "degraded", "unhealthy"
print(health["components"])
# {
#     "backend": {"status": "healthy", "message": "..."},
#     "memory_service": {"status": "healthy", "message": "..."},
#     "vector_store": {"status": "degraded", "message": "..."}
# }
print(health["circuit_breaker"])  # "closed", "open", "half_open"
print(health["metrics"])
# {
#     "success_rate": 0.95,
#     "request_count": 1000,
#     "avg_latency": 0.45
# }
```

#### `check_component_health()`

Belirli bir component'in sağlık durumunu kontrol et.

```python
def check_component_health(self, component_name: str) -> Optional[Dict[str, Any]]:
    """
    Check health of specific component.
    
    Args:
        component_name: Component name
        
    Returns:
        Component health dictionary or None
    """
```

**Örnek:**

```python
backend_health = cm.check_component_health("backend")
print(backend_health["status"])  # "healthy"
print(backend_health["message"])  # "Backend is available"
print(backend_health["last_check"])  # Timestamp
```

#### `register_health_check()`

Custom health check kaydet.

```python
def register_health_check(
    self,
    name: str,
    check_func: Callable[[], Dict[str, Any]],
    interval_seconds: Optional[float] = None,
) -> None:
    """
    Register a custom health check function.
    
    Args:
        name: Component name
        check_func: Health check function that returns ComponentHealth-like dictionary
    """
```

**Örnek:**

```python
def my_database_check():
    try:
        # Check database connection
        db.ping()
        return {
            "status": "healthy",
            "message": "Database connection OK",
            "last_check": time.time()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Database error: {str(e)}",
            "last_check": time.time()
        }

cm.register_health_check("database", my_database_check)
```

#### `get_health_history()`

Health check geçmişi al.

```python
def get_health_history(
    self,
    component_name: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Get health check history.
    
    Args:
        component_name: Filter by component name (None = all)
        limit: Maximum number of history records
        
    Returns:
        List of health check history dictionaries
    """
```

### 3. Alerting

#### `raise_alert()`

Alert oluştur.

```python
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
    
    Args:
        level: Alert level ("info", "warning", "error", "critical")
        title: Alert title
        message: Alert message
        component: Component name
        metadata: Optional metadata
        
    Returns:
        Alert dictionary
    """
```

**Örnek:**

```python
alert = cm.raise_alert(
    level="error",
    title="High Error Rate",
    message="Error rate exceeded 10% in last 5 minutes",
    component="backend",
    metadata={"error_rate": 0.12, "threshold": 0.10}
)

print(alert["id"])  # Alert ID
print(alert["level"])  # "error"
print(alert["timestamp"])  # ISO timestamp
```

#### `get_active_alerts()`

Aktif alert'leri al.

```python
def get_active_alerts(
    self,
    level: Optional[str] = None,
    component: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get active alerts.
    
    Args:
        level: Filter by level (None = all)
        component: Filter by component (None = all)
        
    Returns:
        List of alert dictionaries
    """
```

**Örnek:**

```python
# All active alerts
all_alerts = cm.get_active_alerts()

# Only critical alerts
critical_alerts = cm.get_active_alerts(level="critical")

# Alerts for specific component
backend_alerts = cm.get_active_alerts(component="backend")

for alert in critical_alerts:
    print(f"[{alert['level']}] {alert['title']}: {alert['message']}")
```

#### `register_alert_handler()`

Custom alert handler kaydet.

```python
def register_alert_handler(
    self,
    handler: Callable[[Any], None]
) -> None:
    """
    Register custom alert handler.
    
    Args:
        handler: Callback function that receives Alert object
    """
```

**Örnek:**

```python
from cognitive_management.v2.monitoring.alerting import AlertLevel

def email_alert_handler(alert):
    """Send email for critical alerts"""
    if alert.level == AlertLevel.CRITICAL:
        send_email(
            to="admin@example.com",
            subject=f"[CRITICAL] {alert.title}",
            body=alert.message
        )

cm.register_alert_handler(email_alert_handler)
```

#### `resolve_alert()`

Alert'i çöz (resolve et).

```python
def resolve_alert(
    self,
    title: str,
    component: Optional[str] = None,
) -> bool:
    """
    Resolve an alert.
    
    Args:
        title: Alert title
        component: Optional component name filter
        
    Returns:
        True if alert was found and resolved
    """
```

**Örnek:**

```python
# Resolve alert by title
resolved = cm.resolve_alert("High Error Rate", component="backend")
if resolved:
    print("Alert resolved successfully")
```

### 4. Caching

#### `get_cache_stats()`

Cache istatistikleri al.

```python
def get_cache_stats(self) -> Optional[Dict[str, Any]]:
    """
    Get cache statistics.
    
    Returns:
        Cache statistics dictionary or None if caching disabled
    """
```

**Örnek:**

```python
stats = cm.get_cache_stats()

if stats:
    print(f"Cache size: {stats['size']}")
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")
    print(f"Hit rate: {stats['hit_rate']:.2%}")
    print(f"Semantic cache enabled: {stats.get('semantic_cache_enabled', False)}")
```

#### `invalidate_cache()`

Cache entry'lerini geçersiz kıl.

```python
def invalidate_cache(
    self,
    pattern: Optional[str] = None
) -> int:
    """
    Invalidate cache entries.
    
    Args:
        pattern: Key pattern to invalidate (None = all)
        
    Returns:
        Number of invalidated entries
    """
```

**Örnek:**

```python
# Invalidate all cache
invalidated = cm.invalidate_cache()
print(f"Invalidated {invalidated} cache entries")

# Invalidate specific pattern
invalidated = cm.invalidate_cache(pattern="user_*")
print(f"Invalidated {invalidated} user-related cache entries")
```

#### `clear_cache()`

Tüm cache'i temizle.

```python
def clear_cache(self) -> None:
    """Clear all cache."""
```

### 5. Tracing

#### `get_trace()`

Trace ID ile trace al.

```python
def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
    """
    Get trace by ID.
    
    Args:
        trace_id: Trace ID
        
    Returns:
        Trace dictionary or None
    """
```

**Örnek:**

```python
trace = cm.get_trace("trace_abc123")

if trace:
    print(f"Trace ID: {trace['trace_id']}")
    print(f"Status: {trace['status']}")
    print(f"Duration: {trace['duration']:.3f}s")
    print(f"Spans: {len(trace['spans'])}")
    
    for span in trace['spans']:
        print(f"  - {span['name']}: {span['duration']:.3f}s")
```

#### `get_all_traces()`

Tüm trace'leri al.

```python
def get_all_traces(
    self,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get all traces.
    
    Args:
        limit: Maximum number of traces to return
        
    Returns:
        List of trace dictionaries
    """
```

#### `export_trace()`

Trace'i export et.

```python
def export_trace(self, trace_id: str, format: str = "json") -> Optional[str]:
    """
    Export trace in specified format.
    
    Args:
        trace_id: Trace ID
        format: Export format ("json", "dict")
        
    Returns:
        Exported trace string or None
    """
```

**Örnek:**

```python
# Export as JSON
json_trace = cm.export_trace("trace_abc123", format="json")
with open("trace.json", "w") as f:
    f.write(json_trace)

# Export as dict
trace_dict = cm.export_trace("trace_abc123", format="dict")
print(trace_dict)
```

---

## 🧩 V2 Components

### 1. PolicyRouterV2

Strateji seçimi yapar (direct/think/debate/tot).

**Özellikler:**
- Entropy-based routing
- Length-based routing
- Risk assessment
- Tool need detection
- Decoding config adjustment

**Kullanım:**

```python
from cognitive_management.v2.components import PolicyRouterV2

policy_router = PolicyRouterV2(cfg)
policy_output = policy_router.route(state, request)

print(policy_output.mode)  # "direct", "think", "debate", or "tot"
print(policy_output.decoding_config)  # Adjusted decoding config
```

### 2. MemoryServiceV2

Bellek yönetimi sağlar.

**Özellikler:**
- Episodic memory
- Working memory
- Vector memory (RAG)
- Memory pruning
- Context building

**Kullanım:**

```python
from cognitive_management.v2.components import MemoryServiceV2

memory_service = MemoryServiceV2(cfg)

# Add note
memory_service.add_note("User prefers short responses")

# Retrieve context
context = memory_service.retrieve_context("What did we discuss?", top_k=5)

# Build context
full_context = memory_service.build_context(state, request)
```

### 3. CriticV2

Eleştirel kontrol yapar.

**Özellikler:**
- Task match checking
- Safety checking
- Claim density analysis
- External fact-checking
- Constitutional AI

**Kullanım:**

```python
from cognitive_management.v2.components import CriticV2

critic = CriticV2(cfg, model_manager)

review_result = critic.review(
    user_message="What is the capital of France?",
    draft_text="Paris is the capital of France."
)

print(review_result.needs_revision)  # True/False
print(review_result.score)  # 0.0-1.0
print(review_result.issues)  # List of issues
```

### 4. DeliberationEngineV2

İç ses üretimi (Chain of Thought).

**Özellikler:**
- Chain of Thought reasoning
- Inner thought generation
- Multi-step reasoning

**Kullanım:**

```python
from cognitive_management.v2.components import DeliberationEngineV2

deliberation_engine = DeliberationEngineV2(cfg, model_manager)

thoughts = deliberation_engine.generate_thoughts(
    user_message="Solve: 2x + 5 = 15",
    context=context
)

print(thoughts.inner_thought)  # Step-by-step reasoning
print(thoughts.final_answer)  # Final answer
```

### 5. ToolExecutorV2

Araç yönetimi ve çalıştırma.

**Özellikler:**
- Tool registration
- Tool execution
- Tool metrics
- Schema validation

**Kullanım:**

```python
from cognitive_management.v2.components import ToolExecutorV2

tool_executor = ToolExecutorV2(cfg)

# Register tool
def calculator(a: float, b: float, operation: str) -> float:
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    # ...

tool_executor.register_tool(
    name="calculator",
    func=calculator,
    schema={
        "description": "Perform basic calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
                "operation": {"type": "string", "enum": ["add", "multiply"]}
            }
        }
    }
)

# Execute tool
result = tool_executor.execute(
    tool_name="calculator",
    parameters={"a": 5, "b": 3, "operation": "add"}
)
print(result)  # 8
```

---

## 🎭 Cognitive Patterns

### 1. Chain of Thought (CoT)

Step-by-step reasoning pattern.

**Özellikler:**
- Inner thought generation
- Multi-step reasoning
- Explicit reasoning chain

**Kullanım:**

```python
# Automatically used when entropy > entropy_gate_think
request = CognitiveInput(
    user_message="Solve: If a train travels 120 km in 2 hours, what's its speed?"
)

response = cm.handle(state, request)

if response.used_mode == "think":
    print("Used Chain of Thought reasoning")
    # Response includes step-by-step reasoning
```

### 2. Tree of Thoughts (ToT)

Tree-based reasoning for complex problems.

**Özellikler:**
- Multiple reasoning paths
- Path expansion
- Best path selection
- Pruning

**Kullanım:**

```python
# Enable ToT in config
cfg.policy.tot_enabled = True
cfg.policy.tot_max_depth = 3
cfg.policy.tot_branching_factor = 3
cfg.policy.tot_top_k = 5

cm = CognitiveManager(model_manager, cfg=cfg)

# Automatically used for complex problems
request = CognitiveInput(
    user_message="Plan a 7-day trip to Japan with budget constraints"
)

response = cm.handle(state, request)

if response.used_mode == "tot":
    print("Used Tree of Thoughts reasoning")
```

### 3. Constitutional AI

Self-improvement through principles.

**Özellikler:**
- Principle-based review
- Self-correction
- Harmlessness focus

**Kullanım:**

```python
# Enable Constitutional AI in config
cfg.critic.enable_constitutional_ai = True
cfg.critic.constitutional_strictness = 0.7

cm = CognitiveManager(model_manager, cfg=cfg)

# Critic automatically applies constitutional principles
response = cm.handle(state, request)

if response.revised_by_critic:
    print("Response was revised based on constitutional principles")
```

### 4. Debate Mode

Multi-perspective reasoning.

**Özellikler:**
- Multiple viewpoints
- Argument generation
- Synthesis

**Kullanım:**

```python
# Enable debate mode
cfg.policy.debate_enabled = True
cfg.policy.entropy_gate_debate = 2.5

cm = CognitiveManager(model_manager, cfg=cfg)

# Automatically used for high-entropy inputs
request = CognitiveInput(
    user_message="What are the pros and cons of remote work?"
)

response = cm.handle(state, request)

if response.used_mode == "debate":
    print("Used debate mode for multi-perspective reasoning")
```

---

## 💾 Memory Management

### 1. Episodic Memory

Conversation history storage.

**Özellikler:**
- Message history
- Context preservation
- Automatic pruning

**Kullanım:**

```python
# Memory is automatically managed
state = CognitiveState()

# Add conversation
response1 = cm.handle(state, CognitiveInput(user_message="Merhaba"))
response2 = cm.handle(state, CognitiveInput(user_message="Nasılsın?"))

# Memory contains full conversation history
print(state.history)  # List of messages
```

### 2. Vector Memory (RAG)

Semantic search and retrieval.

**Özellikler:**
- Vector embeddings
- Semantic search
- RAG enhancement

**Kullanım:**

```python
# Enable vector memory
cfg.memory.enable_vector_memory = True
cfg.memory.embedding_provider = "sentence-transformers"
cfg.memory.vector_store_provider = "chroma"

cm = CognitiveManager(model_manager, cfg=cfg)

# Automatic RAG enhancement
response = cm.handle(state, request)
# Response is enhanced with retrieved context
```

### 3. Memory API

#### `add_memory_note()`

Not ekle.

```python
def add_memory_note(self, text: str) -> None:
    """Add a note to episodic memory."""
```

**Örnek:**

```python
cm.add_memory_note("User prefers detailed explanations")
cm.add_memory_note("User is interested in machine learning")
```

#### `get_memory_notes()`

Tüm notları al.

```python
def get_memory_notes(self) -> List[str]:
    """Get all memory notes."""
```

#### `retrieve_context()`

Context retrieval (semantic search).

```python
def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant context from memory using semantic search.
    
    Args:
        query: Query text
        top_k: Number of relevant items to retrieve
        
    Returns:
        List of relevant context items
    """
```

**Örnek:**

```python
context_items = cm.retrieve_context("What did we discuss about Python?", top_k=5)

for item in context_items:
    print(f"Score: {item['score']:.3f}")
    print(f"Content: {item['content']}")
    print(f"Role: {item['role']}")
    print("---")
```

#### `clear_memory_notes()`

Tüm notları temizle.

```python
def clear_memory_notes(self) -> None:
    """Clear all memory notes."""
```

### 4. Vector Store Management

#### `get_vector_store_stats()`

Vector store istatistikleri.

```python
def get_vector_store_stats(self) -> Optional[Dict[str, Any]]:
    """
    Get vector store statistics.
    
    Returns:
        Vector store statistics dictionary or None if disabled
    """
```

**Örnek:**

```python
stats = cm.get_vector_store_stats()

if stats:
    print(f"Total items: {stats['total_items']}")
    print(f"Collection name: {stats['collection_name']}")
    print(f"Embedding dimension: {stats['embedding_dim']}")
```

#### `clear_vector_store()`

Vector store'u temizle.

```python
def clear_vector_store(self) -> None:
    """Clear all items from vector store."""
```

#### `delete_vector_store_items()`

Belirli item'ları sil.

```python
def delete_vector_store_items(self, ids: List[str]) -> None:
    """
    Delete specific items from vector store by IDs.
    
    Args:
        ids: List of item IDs to delete
    """
```

---

## 🛠️ Tool Management

### 1. Tool Registration

#### `register_tool()`

Tool kaydet.

```python
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
    
    Args:
        name: Tool name
        func: Tool function (alias for tool_func)
        tool_func: Tool function (must accept **kwargs)
        description: Tool description
        parameters: Tool parameters schema
        schema: Optional tool schema
    """
```

**Örnek:**

```python
# Simple tool
def get_weather(city: str) -> str:
    # Weather API call
    return f"Weather in {city}: Sunny, 25°C"

cm.register_tool(
    name="get_weather",
    func=get_weather,
    description="Get current weather for a city",
    parameters={
        "city": {"type": "string", "description": "City name", "required": True}
    }
)

# Complex tool with schema
def calculator(**kwargs) -> float:
    a = kwargs.get("a")
    b = kwargs.get("b")
    operation = kwargs.get("operation")
    # ...
    return result

cm.register_tool(
    name="calculator",
    tool_func=calculator,
    schema={
        "description": "Perform basic calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "Operation to perform"
                }
            },
            "required": ["a", "b", "operation"]
        }
    }
)
```

### 2. Tool Listing

#### `list_available_tools()`

Mevcut tool'ları listele.

```python
def list_available_tools(self) -> List[str]:
    """List all available tools."""
```

**Örnek:**

```python
tools = cm.list_available_tools()
print(tools)  # ["calculator", "get_weather", "search", ...]
```

#### `get_tool_schema()`

Tool schema al.

```python
def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
    """
    Get tool schema.
    
    Args:
        tool_name: Tool name
        
    Returns:
        Tool schema dictionary or None
    """
```

**Örnek:**

```python
schema = cm.get_tool_schema("calculator")
print(schema["description"])
print(schema["parameters"])
```

### 3. Tool Metrics

#### `get_tool_metrics()`

Tool execution metrikleri.

```python
def get_tool_metrics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get tool execution metrics.
    
    Args:
        tool_name: Tool name (None = all tools)
        
    Returns:
        Metrics dictionary
    """
```

**Örnek:**

```python
# All tools
all_metrics = cm.get_tool_metrics()

# Specific tool
calc_metrics = cm.get_tool_metrics("calculator")

print(calc_metrics["execution_count"])  # Number of executions
print(calc_metrics["success_count"])  # Number of successes
print(calc_metrics["error_count"])  # Number of errors
print(calc_metrics["avg_execution_time"])  # Average execution time
```

---

## ⚙️ Configuration Management

### 1. Config File Usage

#### Hot Reload ile Config

```python
# config.json
{
  "critic": {
    "enabled": true,
    "strictness": 0.7,
    "enable_constitutional_ai": true
  },
  "memory": {
    "enable_vector_memory": true,
    "embedding_provider": "sentence-transformers"
  },
  "policy": {
    "tot_enabled": true,
    "debate_enabled": true
  }
}

# Python
cm = CognitiveManager(
    model_manager=model_manager,
    config_path="config.json",
    enable_hot_reload=True  # Auto-reload on file changes
)
```

### 2. Config API

#### `get_config_value()`

Config değeri al.

```python
def get_config_value(self, key: str, default: Any = None) -> Any:
    """
    Get config value using dot notation.
    
    Args:
        key: Config key (e.g., "critic.enabled", "tools.enable_tools")
        default: Default value if key not found
        
    Returns:
        Config value
    """
```

**Örnek:**

```python
critic_enabled = cm.get_config_value("critic.enabled", default=True)
strictness = cm.get_config_value("critic.strictness", default=0.5)
tot_enabled = cm.get_config_value("policy.tot_enabled", default=False)
```

#### `set_config_value()`

Config değeri set et.

```python
def set_config_value(self, key: str, value: Any) -> None:
    """
    Set config value (with hot reload if enabled).
    
    Args:
        key: Config key (dot notation)
        value: New value
    """
```

**Örnek:**

```python
# Update config
cm.set_config_value("critic.strictness", 0.8)
cm.set_config_value("policy.debate_enabled", False)

# Config is automatically reloaded if hot reload is enabled
```

#### `update_config()`

Toplu config güncelleme.

```python
def update_config(self, updates: Dict[str, Any]) -> None:
    """
    Update config with dictionary of changes.
    
    Args:
        updates: Dictionary of config updates (can be nested)
    """
```

**Örnek:**

```python
cm.update_config({
    "critic": {
        "enabled": False,
        "strictness": 0.9
    },
    "tools": {
        "enable_tools": True
    },
    "policy": {
        "tot_enabled": True,
        "tot_max_depth": 5
    }
})
```

#### `reload_config()`

Config'i yeniden yükle.

```python
def reload_config(self) -> None:
    """Reload config from file."""
```

#### `validate_config()`

Config'i validate et.

```python
def validate_config(self) -> bool:
    """
    Validate current config.
    
    Returns:
        True if config is valid
        
    Raises:
        ConfigError: If config is invalid
    """
```

**Örnek:**

```python
try:
    is_valid = cm.validate_config()
    if is_valid:
        print("Config is valid")
except ConfigError as e:
    print(f"Config validation failed: {e}")
```

#### `export_config()`

Config'i export et.

```python
def export_config(self, path: Optional[str] = None) -> str:
    """
    Export config to JSON file.
    
    Args:
        path: Output path (None = use config_path)
        
    Returns:
        Exported config as JSON string
    """
```

**Örnek:**

```python
# Export to file
json_config = cm.export_config("backup_config.json")

# Export as string
json_str = cm.export_config()
print(json_str)
```

### 3. Config Listeners

#### `register_config_listener()`

Config değişikliği dinleyicisi kaydet.

```python
def register_config_listener(
    self,
    listener: Callable[[Any], None]
) -> None:
    """
    Register config change listener.
    
    Args:
        listener: Callback function that receives ConfigChangeEvent
    """
```

**Örnek:**

```python
from cognitive_management.v2.config import ConfigChangeEvent

def on_config_change(event: ConfigChangeEvent):
    print(f"Config changed: {event.key} = {event.new_value}")
    if event.key == "critic.strictness":
        # React to strictness change
        print(f"Critic strictness updated to {event.new_value}")

cm.register_config_listener(on_config_change)

# Now any config change will trigger the listener
cm.set_config_value("critic.strictness", 0.9)  # Listener called
```

---

## 📊 Monitoring & Observability

### 1. Performance Monitoring

#### `get_performance_metrics()`

Performance metrikleri al.

```python
def get_performance_metrics(
    self,
    component: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get performance metrics.
    
    Args:
        component: Component name (None = all)
        
    Returns:
        Performance metrics dictionary
    """
```

**Örnek:**

```python
# All components
all_perf = cm.get_performance_metrics()

# Specific component
backend_perf = cm.get_performance_metrics("backend")

print(backend_perf["avg_latency"])
print(backend_perf["p95_latency"])
print(backend_perf["p99_latency"])
print(backend_perf["throughput"])
```

#### `reset_performance_metrics()`

Performance metriklerini sıfırla.

```python
def reset_performance_metrics(self) -> None:
    """Reset all performance metrics."""
```

### 2. Event System

#### `subscribe_to_events()`

Event'lere subscribe ol.

```python
def subscribe_to_events(
    self,
    observer: Any,
    event_type: Optional[str] = None
) -> None:
    """
    Subscribe to events from event bus.
    
    Args:
        observer: Event observer (must implement on_event method)
        event_type: Specific event type (None = all events)
    """
```

**Örnek:**

```python
class MyEventObserver:
    def on_event(self, event):
        print(f"Event received: {event.event_type}")
        print(f"Data: {event.data}")
        print(f"Source: {event.source}")
        print(f"Timestamp: {event.timestamp}")

observer = MyEventObserver()

# Subscribe to all events
cm.subscribe_to_events(observer)

# Subscribe to specific event type
cm.subscribe_to_events(observer, event_type="request_received")
```

#### `publish_event()`

Event yayınla.

```python
def publish_event(
    self,
    event_type: str,
    data: Dict[str, Any],
    source: Optional[str] = None
) -> None:
    """
    Publish an event to event bus.
    
    Args:
        event_type: Event type
        data: Event data dictionary
        source: Optional source identifier
    """
```

**Örnek:**

```python
cm.publish_event(
    event_type="custom_event",
    data={
        "message": "User completed onboarding",
        "user_id": "12345",
        "timestamp": time.time()
    },
    source="OnboardingService"
)
```

#### `get_event_history()`

Event geçmişi al.

```python
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
```

**Örnek:**

```python
# All events
all_events = cm.get_event_history(limit=50)

# Specific event type
request_events = cm.get_event_history(event_type="request_received", limit=20)

for event in request_events:
    print(f"{event.timestamp}: {event.event_type}")
    print(f"  Data: {event.data}")
```

#### `get_event_metrics()`

Event-based metrikler.

```python
def get_event_metrics(self) -> Dict[str, int]:
    """
    Get event-based metrics from MetricsEventHandler.
    
    Returns:
        Dictionary of event type -> count
    """
```

**Örnek:**

```python
metrics = cm.get_event_metrics()

print(metrics)
# {
#     "request_received": 1000,
#     "response_generated": 950,
#     "error_occurred": 50,
#     "tool_executed": 200
# }
```

---

## 🤖 AIOps & Predictive Analytics

### 1. Anomaly Detection

#### `detect_anomalies()`

Anomali tespit et.

```python
def detect_anomalies(
    self,
    metric_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Detect anomalies in performance metrics.
    
    Args:
        metric_name: Specific metric to check (None = check all)
        
    Returns:
        List of anomaly alert dictionaries
    """
```

**Örnek:**

```python
# Check all metrics
all_anomalies = cm.detect_anomalies()

# Check specific metric
latency_anomalies = cm.detect_anomalies(metric_name="latency")

for anomaly in latency_anomalies:
    print(f"Metric: {anomaly['metric_name']}")
    print(f"Anomaly Type: {anomaly['anomaly_type']}")
    print(f"Severity: {anomaly['severity']}")
    print(f"Current Value: {anomaly['current_value']}")
    print(f"Expected Value: {anomaly['expected_value']}")
    print(f"Deviation: {anomaly['deviation']}")
    print(f"Description: {anomaly['description']}")
```

#### `get_anomaly_summary()`

Anomali özeti al.

```python
def get_anomaly_summary(self) -> Dict[str, Any]:
    """
    Get summary of detected anomalies.
    
    Returns:
        Anomaly summary dictionary
    """
```

**Örnek:**

```python
summary = cm.get_anomaly_summary()

print(f"Total anomalies: {summary['total_anomalies']}")
print(f"Critical: {summary['critical_count']}")
print(f"Warning: {summary['warning_count']}")
print(f"Info: {summary['info_count']}")
```

### 2. Predictive Analytics

#### `predict_latency()`

Gelecek latency tahmini.

```python
def predict_latency(
    self,
    metric_name: str,
    horizon_minutes: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Predict future latency for a metric.
    
    Args:
        metric_name: Name of the metric
        horizon_minutes: Prediction horizon in minutes (default: 5)
        
    Returns:
        Prediction dictionary or None
    """
```

**Örnek:**

```python
prediction = cm.predict_latency("latency", horizon_minutes=10)

if prediction:
    print(f"Predicted latency: {prediction['predicted_value']:.3f}s")
    print(f"Confidence: {prediction['confidence']:.2%}")
    print(f"Upper bound: {prediction['upper_bound']:.3f}s")
    print(f"Lower bound: {prediction['lower_bound']:.3f}s")
    print(f"Horizon: {prediction['prediction_horizon']} minutes")
```

#### `predict_error_rate()`

Gelecek error rate tahmini.

```python
def predict_error_rate(
    self,
    metric_name: str,
    horizon_minutes: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Predict future error rate.
    
    Args:
        metric_name: Name of the metric
        horizon_minutes: Prediction horizon in minutes
        
    Returns:
        Prediction dictionary or None
    """
```

#### `get_scaling_recommendations()`

Auto-scaling önerileri al.

```python
def get_scaling_recommendations(
    self,
    target_latency: Optional[float] = None,
    target_error_rate: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Get auto-scaling recommendations.
    
    Args:
        target_latency: Target latency threshold (seconds)
        target_error_rate: Target error rate threshold (0.0-1.0)
        
    Returns:
        List of scaling recommendation dictionaries
    """
```

**Örnek:**

```python
recommendations = cm.get_scaling_recommendations(
    target_latency=0.5,  # 500ms target
    target_error_rate=0.01  # 1% target
)

for rec in recommendations:
    print(f"Action: {rec['action']}")  # "scale_up", "scale_down", "no_action"
    print(f"Confidence: {rec['confidence']:.2%}")
    print(f"Reason: {rec['reason']}")
    print(f"Estimated Impact: {rec['estimated_impact']}")
    print(f"Priority: {rec['priority']}")
```

### 3. Trend Analysis

#### `analyze_trend()`

Performance trend analizi.

```python
def analyze_trend(
    self,
    metric_name: str,
    period_minutes: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Analyze performance trend for a metric.
    
    Args:
        metric_name: Name of the metric
        period_minutes: Analysis period in minutes (default: 60)
        
    Returns:
        Trend analysis dictionary or None
    """
```

**Örnek:**

```python
trend = cm.analyze_trend("latency", period_minutes=120)

if trend:
    print(f"Trend Direction: {trend['trend_direction']}")  # "increasing", "decreasing", "stable"
    print(f"Trend Strength: {trend['trend_strength']:.2%}")
    print(f"Slope: {trend['slope']:.6f}")
    print(f"Change Percentage: {trend['change_percentage']:.2%}")
    print(f"Period: {trend['period']} minutes")
```

#### `get_all_trends()`

Tüm metrikler için trend analizi.

```python
def get_all_trends(
    self,
    period_minutes: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze trends for all metrics.
    
    Args:
        period_minutes: Analysis period in minutes
        
    Returns:
        Dictionary of metric_name -> trend dictionary
    """
```

---

## ⚡ Performance Optimization

### 1. Cache Warming

#### `warm_cache()`

Cache'i ısıt.

```python
def warm_cache(
    self,
    strategy_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Warm cache using configured strategies.
    
    Args:
        strategy_name: Specific strategy to use (None = use all enabled)
        
    Returns:
        Cache warming results dictionary
    """
```

**Örnek:**

```python
# Warm cache with all strategies
result = cm.warm_cache()

print(f"Strategies executed: {result['strategies_executed']}")
print(f"Keys warmed: {result['keys_warmed']}")
print(f"Time taken: {result['time_taken']:.3f}s")

# Warm with specific strategy
result = cm.warm_cache(strategy_name="popular_content")
```

#### `get_cache_warmer_stats()`

Cache warmer istatistikleri.

```python
def get_cache_warmer_stats(self) -> Dict[str, Any]:
    """
    Get cache warmer statistics.
    
    Returns:
        Cache warmer statistics dictionary
    """
```

#### `warm_popular_content()`

Popüler içerikle cache'i ısıt.

```python
def warm_popular_content(
    self,
    key_access_counts: Dict[str, int],
    top_k: int = 100,
    ttl: Optional[float] = None
) -> int:
    """
    Warm cache with popular content based on access counts.
    
    Args:
        key_access_counts: Dictionary of key -> access count
        top_k: Number of top keys to warm
        ttl: Optional TTL for warmed entries
        
    Returns:
        Number of keys warmed
    """
```

### 2. Performance Profiling

#### `get_performance_profile()`

Performance profiling raporu.

```python
def get_performance_profile(
    self,
    format: str = "summary"
) -> str:
    """
    Get performance profiling report.
    
    Args:
        format: Report format ("summary", "detailed", "bottlenecks")
        
    Returns:
        Performance profile report string
    """
```

**Örnek:**

```python
# Summary report
summary = cm.get_performance_profile(format="summary")
print(summary)

# Detailed report
detailed = cm.get_performance_profile(format="detailed")
print(detailed)

# Bottlenecks only
bottlenecks = cm.get_performance_profile(format="bottlenecks")
print(bottlenecks)
```

#### `identify_bottlenecks()`

Performance bottleneck'leri tespit et.

```python
def identify_bottlenecks(self) -> List[Dict[str, Any]]:
    """
    Identify performance bottlenecks.
    
    Returns:
        List of bottleneck dictionaries
    """
```

**Örnek:**

```python
bottlenecks = cm.identify_bottlenecks()

for bottleneck in bottlenecks:
    print(f"Operation: {bottleneck['operation']}")
    print(f"Severity: {bottleneck['severity']}")  # "high", "medium", "low"
    print(f"Avg Duration: {bottleneck['avg_duration']:.3f}s")
    print(f"Impact: {bottleneck['impact']:.2%}")
    print(f"Recommendation: {bottleneck['recommendation']}")
```

#### `get_operation_stats()`

Belirli bir operasyonun istatistikleri.

```python
def get_operation_stats(
    self,
    operation_name: str
) -> Optional[Dict[str, Any]]:
    """
    Get statistics for a specific operation.
    
    Args:
        operation_name: Operation name
        
    Returns:
        Operation statistics dictionary or None
    """
```

#### `get_all_performance_stats()`

Tüm operasyonların istatistikleri.

```python
def get_all_performance_stats(self) -> Dict[str, Dict[str, Any]]:
    """
    Get performance statistics for all operations.
    
    Returns:
        Dictionary of operation name -> statistics
    """
```

### 3. Connection Pooling

#### `get_connection_pool_stats()`

Connection pool istatistikleri.

```python
def get_connection_pool_stats(self) -> Optional[Dict[str, Any]]:
    """
    Get connection pool statistics.
    
    Returns:
        Connection pool statistics dictionary or None if pooling disabled
    """
```

**Örnek:**

```python
stats = cm.get_connection_pool_stats()

if stats:
    print(f"Total connections: {stats['total_connections']}")
    print(f"Active connections: {stats['active_connections']}")
    print(f"Idle connections: {stats['idle_connections']}")
    print(f"Max size: {stats['max_size']}")
```

#### `cleanup_idle_connections()`

Idle connection'ları temizle.

```python
def cleanup_idle_connections(self) -> int:
    """
    Clean up idle connections in the pool.
    
    Returns:
        Number of connections cleaned up
    """
```

---

## 📝 Kullanım Örnekleri

### Örnek 1: Temel Kullanım

```python
from cognitive_management import CognitiveManager
from cognitive_management.cognitive_types import CognitiveState, CognitiveInput
from model_management import ModelManager

# Initialize
model_manager = ModelManager(config)
cm = CognitiveManager(model_manager)

# Handle request
state = CognitiveState()
request = CognitiveInput(user_message="Merhaba, nasılsın?")
response = cm.handle(state, request)

print(response.text)
print(f"Mode: {response.used_mode}")
```

### Örnek 2: Config ile Kullanım

```python
from cognitive_management import CognitiveManager, CognitiveManagerConfig

# Create config
cfg = CognitiveManagerConfig()
cfg.critic.enabled = True
cfg.critic.strictness = 0.7
cfg.memory.enable_vector_memory = True
cfg.policy.tot_enabled = True

# Initialize with config
cm = CognitiveManager(model_manager, cfg=cfg)

# Use
response = cm.handle(state, request)
```

### Örnek 3: Hot Reload Config

```python
# config.json
{
  "critic": {"enabled": true, "strictness": 0.7},
  "memory": {"enable_vector_memory": true},
  "policy": {"tot_enabled": true}
}

# Python
cm = CognitiveManager(
    model_manager,
    config_path="config.json",
    enable_hot_reload=True  # Auto-reload on file changes
)

# Config file changes are automatically reloaded
```

### Örnek 4: Tool Registration

```python
def search_web(query: str) -> str:
    # Web search implementation
    return f"Search results for: {query}"

cm.register_tool(
    name="search_web",
    func=search_web,
    description="Search the web for information",
    parameters={
        "query": {"type": "string", "description": "Search query", "required": True}
    }
)

# Tool is automatically available for policy routing
response = cm.handle(state, CognitiveInput(
    user_message="Son haberleri ara"
))
```

### Örnek 5: Monitoring Setup

```python
# Health check observer
class HealthObserver:
    def on_event(self, event):
        if event.event_type == "health_check_failed":
            send_alert(f"Health check failed: {event.data}")

observer = HealthObserver()
cm.subscribe_to_events(observer, event_type="health_check_failed")

# Custom alert handler
def critical_alert_handler(alert):
    if alert.level == "critical":
        send_sms("+1234567890", f"CRITICAL: {alert.title}")

cm.register_alert_handler(critical_alert_handler)
```

### Örnek 6: Performance Monitoring

```python
# Get metrics
metrics = cm.get_metrics()
print(f"Success rate: {metrics['global']['success_rate']:.2%}")

# Detect anomalies
anomalies = cm.detect_anomalies()
for anomaly in anomalies:
    if anomaly['severity'] == 'high':
        print(f"High severity anomaly: {anomaly['description']}")

# Predict latency
prediction = cm.predict_latency("latency", horizon_minutes=10)
if prediction and prediction['predicted_value'] > 1.0:
    print("Warning: Latency may exceed 1s in 10 minutes")
```

### Örnek 7: Memory Management

```python
# Add notes
cm.add_memory_note("User prefers detailed explanations")
cm.add_memory_note("User is interested in AI")

# Retrieve context
context = cm.retrieve_context("What did we discuss about AI?", top_k=5)
for item in context:
    print(f"Score: {item['score']:.3f} - {item['content']}")

# Get memory notes
notes = cm.get_memory_notes()
print(notes)
```

### Örnek 8: Async Processing

```python
import asyncio

async def process_requests():
    tasks = [
        cm.handle_async(state1, request1),
        cm.handle_async(state2, request2),
        cm.handle_async(state3, request3),
    ]
    
    responses = await asyncio.gather(*tasks)
    return responses

responses = asyncio.run(process_requests())
```

---

## 🎓 Best Practices

### 1. Configuration Management

- ✅ Config dosyası kullan (hot reload için)
- ✅ Ortam değişkenlerini kullan (sensitive data için)
- ✅ Config validation'ı aktif tut
- ✅ Config listener'ları kullan (dinamik reconfiguration için)

### 2. Memory Management

- ✅ Vector memory'yi aktif et (RAG için)
- ✅ Memory pruning'ı optimize et (context window için)
- ✅ Semantic search kullan (context retrieval için)
- ✅ Memory notes kullan (user preferences için)

### 3. Monitoring

- ✅ Health check'leri kaydet
- ✅ Alert handler'ları kaydet
- ✅ Event subscriber'ları kullan
- ✅ Performance profiling'i aktif et (bottleneck detection için)

### 4. Performance

- ✅ Cache'i aktif et (repeated queries için)
- ✅ Semantic cache kullan (similar queries için)
- ✅ Connection pooling kullan (high throughput için)
- ✅ Batch processing kullan (bulk operations için)

### 5. Error Handling

- ✅ Circuit breaker'ı optimize et
- ✅ Retry logic'i ayarla
- ✅ Timeout'ları set et
- ✅ Error logging'i aktif et

---

## 🔗 İlgili Dokümantasyon

- [API Reference](./api/README.md) - Detaylı API dokümantasyonu
- [Architecture](./architecture/README.md) - Sistem mimarisi
- [Usage Guides](./guides/README.md) - Kullanım örnekleri
- [Development](./development/README.md) - Geliştirme rehberi
- [Main Architecture Documentation](../../ARCHITECTURE.md)
- [Model Management Documentation](../model_management/README.md)
- [Neural Network Documentation](../neural_network/README.md)

---

**Hazırlayan:** AI Assistant (Auto)  
**Versiyon:** 2.0  
**Durum:** ✅ Production-Ready | Enterprise-Grade
