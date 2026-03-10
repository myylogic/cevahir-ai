# Cognitive Management V2 - Usage Guides

**Versiyon:** 2.0  
**Son Güncelleme:** 2025-01-27

---

## 📋 İÇİNDEKİLER

1. [Quick Start Guide](#quick-start-guide)
2. [Advanced Usage](#advanced-usage)
3. [Best Practices](#best-practices)
4. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start Guide

### 1. Temel Kurulum

```python
from cognitive_management import CognitiveManager
from model_management import ModelManager
from cognitive_management.types import CognitiveState, CognitiveInput

# Model manager oluştur
model_manager = ModelManager()

# Cognitive manager oluştur
cm = CognitiveManager(model_manager)

# State ve request hazırla
state = CognitiveState()
request = CognitiveInput(user_message="Merhaba, nasılsın?")

# Request işle
response = cm.handle(state, request)

# Sonucu yazdır
print(response.text)
print(f"Mode: {response.used_mode}")
print(f"Tool used: {response.tool_used}")
```

### 2. Config File ile Kullanım

```python
# config.json
{
  "critic": {
    "enabled": true,
    "strictness": 0.7
  },
  "tools": {
    "enable_tools": true,
    "allow": ["calculator", "search"]
  },
  "policy": {
    "debate_enabled": true,
    "entropy_gate_think": 1.5
  }
}

# Python
cm = CognitiveManager(
    model_manager=ModelManager(),
    config_path="config.json",
    enable_hot_reload=True
)
```

### 3. Async Kullanım

```python
import asyncio

async def main():
    cm = CognitiveManager(ModelManager())
    state = CognitiveState()
    request = CognitiveInput(user_message="Merhaba")
    
    response = await cm.handle_async(state, request)
    print(response.text)

asyncio.run(main())
```

---

## 🎯 Advanced Usage

### 1. Custom Tool Registration

```python
from cognitive_management.v2.components.tool_executor_v2 import ToolExecutorV2
from cognitive_management.config import CognitiveManagerConfig

# Config
cfg = CognitiveManagerConfig()
cfg.tools.enable_tools = True
cfg.tools.allow = ("calculator", "search", "my_custom_tool")

# Tool executor
tool_executor = ToolExecutorV2(cfg)

# Custom tool
def weather_tool(city: str) -> str:
    # Weather API call
    return f"Weather in {city}: Sunny, 25°C"

# Register tool
tool_executor.register_tool(
    "weather",
    weather_tool,
    schema={
        "description": "Get weather for a city",
        "parameters": {
            "city": {"type": "string", "description": "City name"}
        }
    }
)

# Use in CognitiveManager
cm = CognitiveManager(ModelManager(), cfg=cfg)
# Tool executor otomatik olarak container'a kaydedilir
```

### 2. Config Change Listeners

```python
from cognitive_management.v2.config import ConfigChangeEvent

def on_config_change(event: ConfigChangeEvent):
    print(f"Config '{event.key}' changed:")
    print(f"  Old: {event.old_value}")
    print(f"  New: {event.new_value}")
    
    # Component'leri güncelle
    if event.key == "critic.enabled":
        # Critic component'i yeniden yapılandır
        pass

cm = CognitiveManager(
    ModelManager(),
    config_path="config.json",
    enable_hot_reload=True
)

cm.register_config_listener(on_config_change)

# Config dosyası değiştiğinde listener otomatik çağrılır
```

### 3. Performance Monitoring

```python
# Metrics al
metrics = cm.get_metrics()
print(f"Success rate: {metrics['global']['success_rate']}")
print(f"Avg latency: {metrics['global']['avg_latency']}s")

# Component health check
health = cm.check_component_health("backend")
print(f"Backend status: {health['status']}")

# Performance metrics
perf = cm.get_performance_metrics("backend")
print(f"Backend latency P95: {perf['latency_p95']}s")
```

### 4. Cache Management

```python
# Cache stats
stats = cm.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']}")

# Invalidate cache
count = cm.invalidate_cache("user_*")
print(f"Invalidated {count} entries")

# Clear all cache
cm.clear_cache()
```

### 5. Distributed Tracing

```python
# Get trace
trace = cm.get_trace("trace_123")
print(f"Trace duration: {trace['duration']}s")

# Get all traces
traces = cm.get_all_traces(limit=10)
for t in traces:
    print(f"Trace {t['trace_id']}: {t['duration']}s")

# Export trace
json_str = cm.export_trace("trace_123", format="json")
```

### 6. Alert Management

```python
# Raise alert
alert = cm.raise_alert(
    level="warning",
    title="High Latency",
    message="Average latency exceeded 1s",
    component="backend",
    metadata={"threshold": 1.0, "current": 1.5}
)

# Get active alerts
alerts = cm.get_active_alerts(level="warning")
for alert in alerts:
    print(f"{alert['title']}: {alert['message']}")

# Alert stats
stats = cm.get_alert_stats()
print(f"Total alerts: {stats['total']}")
```

---

## ✅ Best Practices

### 1. State Management

```python
# State'i conversation boyunca koru
state = CognitiveState()

# Her request için aynı state'i kullan
for message in conversation:
    request = CognitiveInput(user_message=message)
    response = cm.handle(state, request)
    # State otomatik güncellenir
```

### 2. Error Handling

```python
try:
    response = cm.handle(state, request)
except Exception as e:
    # Error handling
    print(f"Error: {e}")
    # Health check
    health = cm.get_health_status()
    if health["status"] != "healthy":
        # System unhealthy, take action
        pass
```

### 3. Config Management

```python
# Production: Config file kullan
cm = CognitiveManager(
    ModelManager(),
    config_path="config/production.json",
    enable_hot_reload=False  # Production'da hot reload kapalı
)

# Development: Hot reload aktif
cm = CognitiveManager(
    ModelManager(),
    config_path="config/development.json",
    enable_hot_reload=True
)
```

### 4. Performance Optimization

```python
# Cache'i aktif et
cfg = CognitiveManagerConfig()
cfg.runtime.enable_caching = True
cfg.runtime.cache_ttl = 3600.0  # 1 hour

cm = CognitiveManager(ModelManager(), cfg=cfg)

# Async kullan (concurrent requests için)
responses = await asyncio.gather(*[
    cm.handle_async(state, request)
    for request in requests
])
```

### 5. Monitoring

```python
# Regular health checks
def health_check_loop():
    while True:
        health = cm.get_health_status()
        if health["status"] != "healthy":
            # Alert
            cm.raise_alert(
                level="error",
                title="System Unhealthy",
                message=f"Status: {health['status']}"
            )
        time.sleep(60)  # Check every minute

# Start health check thread
threading.Thread(target=health_check_loop, daemon=True).start()
```

---

## 🔧 Troubleshooting

### Problem: High Latency

**Çözüm:**
```python
# Cache'i aktif et
cfg.runtime.enable_caching = True
cm = CognitiveManager(ModelManager(), cfg=cfg)

# Metrics kontrol et
metrics = cm.get_metrics()
if metrics["global"]["avg_latency"] > 1.0:
    # Performance optimization
    pass
```

### Problem: Config Not Reloading

**Çözüm:**
```python
# Hot reload aktif mi kontrol et
cm = CognitiveManager(
    ModelManager(),
    config_path="config.json",
    enable_hot_reload=True  # True olmalı
)

# Manuel reload
cm.reload_config()
```

### Problem: Tool Not Executing

**Çözüm:**
```python
# Tools enabled mi kontrol et
cfg = CognitiveManagerConfig()
cfg.tools.enable_tools = True
cfg.tools.allow = ("calculator", "search")

cm = CognitiveManager(ModelManager(), cfg=cfg)

# Tool metrics kontrol et
# (ToolExecutor'a direkt erişim gerekir)
```

### Problem: Memory Issues

**Çözüm:**
```python
# Memory config optimize et
cfg = CognitiveManagerConfig()
cfg.memory.max_history_tokens = 2048  # Düşür
cfg.memory.session_summary_every = 4  # Daha sık özetle

cm = CognitiveManager(ModelManager(), cfg=cfg)
```

### Problem: Circuit Breaker Open

**Çözüm:**
```python
# Health check
health = cm.get_health_status()
if health["circuit_breaker"] == "open":
    # System recovering, wait
    time.sleep(60)
    # Retry
    response = cm.handle(state, request)
```

---

## 🔗 İlgili Dokümantasyon

- [API Reference](../api/README.md)
- [Architecture Documentation](../architecture/README.md)
- [Development Guide](../development/README.md)
- [Main Documentation](../README.md)

---

**Hazırlayan:** AI Assistant (Auto)  
**Versiyon:** 2.0

