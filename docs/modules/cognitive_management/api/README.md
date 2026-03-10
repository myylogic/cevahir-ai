# Cognitive Management V2 - API Reference

**Versiyon:** 2.0  
**Son Güncelleme:** 2025-01-27

---

## 📋 İÇİNDEKİLER

1. [CognitiveManager API](#cognitivemanager-api)
2. [Configuration Management API](#configuration-management-api)
3. [Monitoring API](#monitoring-api)
4. [Cache Management API](#cache-management-api)
5. [Tracing API](#tracing-api)
6. [Tool Management API](#tool-management-api)

---

## 🎯 CognitiveManager API

### `CognitiveManager`

Ana sınıf - Cognitive Management sisteminin giriş noktası.

#### Constructor

```python
CognitiveManager(
    model_manager: ModelAPI,
    cfg: Optional[CognitiveManagerConfig] = None,
    config_path: Optional[str] = None,
    enable_hot_reload: bool = False
)
```

**Parametreler:**
- `model_manager` (ModelAPI): Model API interface
- `cfg` (Optional[CognitiveManagerConfig]): Konfigürasyon (None = default)
- `config_path` (Optional[str]): Config dosya yolu (JSON/YAML)
- `enable_hot_reload` (bool): Hot reload aktif mi?

**Örnek:**
```python
from cognitive_management import CognitiveManager
from model_management import ModelManager

cm = CognitiveManager(
    model_manager=ModelManager(),
    config_path="config.json",
    enable_hot_reload=True
)
```

#### `handle()`

Senkron request işleme.

```python
def handle(
    self,
    state: CognitiveState,
    request: CognitiveInput,
    *,
    decoding: Optional[DecodingConfig] = None
) -> CognitiveOutput
```

**Parametreler:**
- `state` (CognitiveState): Cognitive state
- `request` (CognitiveInput): Input request
- `decoding` (Optional[DecodingConfig]): Decoding config override

**Returns:**
- `CognitiveOutput`: Response output

**Örnek:**
```python
state = CognitiveState()
request = CognitiveInput(user_message="Merhaba")
response = cm.handle(state, request)
print(response.text)
```

#### `handle_async()`

Asenkron request işleme.

```python
async def handle_async(
    self,
    state: CognitiveState,
    request: CognitiveInput,
    *,
    decoding: Optional[DecodingConfig] = None
) -> CognitiveOutput
```

**Örnek:**
```python
import asyncio

async def main():
    state = CognitiveState()
    request = CognitiveInput(user_message="Merhaba")
    response = await cm.handle_async(state, request)
    print(response.text)

asyncio.run(main())
```

#### `handle_multimodal()`

Multimodal input işleme (text, audio, image).

```python
def handle_multimodal(
    self,
    state: CognitiveState,
    text: str = None,
    audio: bytes = None,
    image: bytes = None,
    *,
    decoding: Optional[DecodingConfig] = None
) -> CognitiveOutput
```

**Örnek:**
```python
response = cm.handle_multimodal(
    state=state,
    text="Bu resimde ne var?",
    image=image_bytes
)
```

---

## ⚙️ Configuration Management API

### `get_config_value()`

Config değeri alma (dot notation).

```python
def get_config_value(self, key: str, default: Any = None) -> Any
```

**Parametreler:**
- `key` (str): Config key (örn: "critic.enabled")
- `default` (Any): Default değer

**Returns:**
- `Any`: Config değeri

**Örnek:**
```python
critic_enabled = cm.get_config_value("critic.enabled", True)
tool_enabled = cm.get_config_value("tools.enable_tools", False)
```

### `set_config_value()`

Config değeri ayarlama (hot reload ile).

```python
def set_config_value(self, key: str, value: Any) -> None
```

**Parametreler:**
- `key` (str): Config key
- `value` (Any): Yeni değer

**Örnek:**
```python
cm.set_config_value("critic.enabled", False)
cm.set_config_value("policy.debate_enabled", True)
```

### `reload_config()`

Config dosyasını yeniden yükle.

```python
def reload_config(self) -> None
```

**Örnek:**
```python
cm.reload_config()
```

### `register_config_listener()`

Config değişikliği listener kaydı.

```python
def register_config_listener(
    self,
    listener: Callable[[ConfigChangeEvent], None]
) -> None
```

**Örnek:**
```python
from cognitive_management.v2.config import ConfigChangeEvent

def on_config_change(event: ConfigChangeEvent):
    print(f"Config '{event.key}' changed: {event.old_value} -> {event.new_value}")

cm.register_config_listener(on_config_change)
```

### `export_config()`

Config'i JSON olarak export et.

```python
def export_config(self, path: Optional[str] = None) -> str
```

**Örnek:**
```python
json_str = cm.export_config("backup_config.json")
```

---

## 📊 Monitoring API

### `get_metrics()`

Performans metrikleri.

```python
def get_metrics(self) -> Dict[str, Any]
```

**Returns:**
```python
{
    "global": {
        "request_count": 100,
        "success_count": 95,
        "error_count": 5,
        "success_rate": 0.95,
        "avg_latency": 0.15,
        "last_request_time": 1234567890.0
    },
    "modes": {
        "direct": {...},
        "think1": {...},
        "debate2": {...}
    }
}
```

### `get_health_status()`

Sistem sağlık durumu.

```python
def get_health_status(self) -> Dict[str, Any]
```

**Returns:**
```python
{
    "status": "healthy",
    "components": {
        "backend": {"status": "healthy", ...},
        "memory_service": {"status": "healthy", ...}
    },
    "circuit_breaker": "closed",
    "metrics": {...}
}
```

### `get_performance_metrics()`

Component/operation performans metrikleri.

```python
def get_performance_metrics(
    self,
    component: Optional[str] = None
) -> Dict[str, Any]
```

**Örnek:**
```python
# All components
all_metrics = cm.get_performance_metrics()

# Specific component
backend_metrics = cm.get_performance_metrics("backend")
```

### `check_component_health()`

Component sağlık kontrolü.

```python
def check_component_health(
    self,
    component_name: str
) -> Optional[Dict[str, Any]]
```

**Örnek:**
```python
health = cm.check_component_health("backend")
```

### `raise_alert()`

Alert oluştur.

```python
def raise_alert(
    self,
    level: str,
    title: str,
    message: str,
    component: str = "",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

**Örnek:**
```python
cm.raise_alert(
    level="warning",
    title="High Latency",
    message="Average latency exceeded threshold",
    component="backend"
)
```

---

## 💾 Cache Management API

### `get_cache_stats()`

Cache istatistikleri.

```python
def get_cache_stats(self) -> Optional[Dict[str, Any]]
```

**Returns:**
```python
{
    "size": 100,
    "hits": 85,
    "misses": 15,
    "hit_rate": 0.85,
    "ttl": 3600.0
}
```

### `invalidate_cache()`

Cache entry'leri invalidate et.

```python
def invalidate_cache(
    self,
    pattern: Optional[str] = None
) -> int
```

**Örnek:**
```python
# Invalidate all
count = cm.invalidate_cache()

# Invalidate specific pattern
count = cm.invalidate_cache("user_*")
```

### `clear_cache()`

Tüm cache'i temizle.

```python
def clear_cache(self) -> None
```

---

## 🔍 Tracing API

### `get_trace()`

Trace ID ile trace getir.

```python
def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]
```

**Örnek:**
```python
trace = cm.get_trace("trace_123")
```

### `get_all_traces()`

Tüm trace'leri getir.

```python
def get_all_traces(
    self,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]
```

**Örnek:**
```python
# Last 10 traces
traces = cm.get_all_traces(limit=10)
```

### `export_trace()`

Trace'i export et.

```python
def export_trace(
    self,
    trace_id: str,
    format: str = "json"
) -> Optional[str]
```

**Örnek:**
```python
json_str = cm.export_trace("trace_123", format="json")
```

---

## 🔧 Tool Management API

### Tool Executor

Tool execution için `ToolExecutorV2` kullanılır.

**Örnek:**
```python
from cognitive_management.v2.components.tool_executor_v2 import ToolExecutorV2
from cognitive_management.config import CognitiveManagerConfig

cfg = CognitiveManagerConfig()
tool_executor = ToolExecutorV2(cfg)

# Register custom tool
def my_tool(param: str) -> str:
    return f"Result: {param}"

tool_executor.register_tool("my_tool", my_tool)

# Execute tool
result = tool_executor.execute("my_tool", {"param": "test"})

# Get metrics
metrics = tool_executor.get_tool_metrics("my_tool")
```

---

## 📝 Data Types

### `CognitiveState`

```python
@dataclass
class CognitiveState:
    history: List[Dict[str, Any]] = field(default_factory=list)
    step: int = 0
    last_mode: Optional[str] = None
    last_entropy: float = 0.0
```

### `CognitiveInput`

```python
@dataclass
class CognitiveInput:
    user_message: str
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### `CognitiveOutput`

```python
@dataclass
class CognitiveOutput:
    text: str
    used_mode: str
    tool_used: Optional[str] = None
    revised_by_critic: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### `DecodingConfig`

```python
@dataclass
class DecodingConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: Optional[int] = None
    repetition_penalty: float = 1.0
```

---

## 🔗 İlgili Dokümantasyon

- [Architecture Documentation](../architecture/README.md)
- [Usage Guides](../guides/README.md)
- [Development Guide](../development/README.md)
- [Main Documentation](../README.md)

---

**Hazırlayan:** AI Assistant (Auto)  
**Versiyon:** 2.0

