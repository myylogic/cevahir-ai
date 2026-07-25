# Cognitive Management — Mimari

**Modül:** `cognitive_management`
**Son Güncelleme:** 2026-03-16

---

## İçindekiler

1. [Genel Mimari Yaklaşım](#genel-mimari-yaklaşım)
2. [İşleme Pipeline'ı](#i̇şleme-pipelineı)
3. [Chain of Responsibility Handlers](#chain-of-responsibility-handlers)
4. [CognitiveOrchestrator](#cognitiveorchestrator)
5. [Middleware Katmanı](#middleware-katmanı)
6. [Event Bus](#event-bus)
7. [Dependency Injection Container](#dependency-injection-container)
8. [Interfaces / Protocol Tanımları](#interfaces--protocol-tanımları)
9. [Asenkron Pipeline](#asenkron-pipeline)
10. [Bileşen Detayları](#bileşen-detayları)

---

## Genel Mimari Yaklaşım

Cognitive Management V2 birden fazla design pattern'i katmanlı biçimde uygular:

```
+-----------------------------------------------------------------+
|                     CognitiveManager                           |
|                  (Facade Pattern - Giris Noktasi)              |
+--------------------------+--------------------------------------+
                           |
                           v
+-----------------------------------------------------------------+
|                  CognitiveOrchestrator                         |
|           (Orchestrator Pattern - Ic Koordinasyon)             |
|                                                                 |
|  +----------------+  +-----------------+  +-----------------+  |
|  |  Middleware    |  |  EventBus       |  |  PerformanceMon |  |
|  |  Chain         |  |  (Pub/Sub)      |  |  itor           |  |
|  +-------+--------+  +-----------------+  +-----------------+  |
|          |                                                      |
|          v                                                      |
|  +-------------------------------------------------------+      |
|  |              ProcessingPipeline                       |      |
|  |         (Chain of Responsibility Pattern)             |      |
|  |                                                       |      |
|  |  Handler 1 --> Handler 2 --> ... --> Handler N        |      |
|  +-------------------------------------------------------+      |
+-----------------------------------------------------------------+
```

### Uygulanan Design Patterns

| Pattern | Sınıf | Amaç |
|---------|-------|------|
| Facade | `CognitiveManager` | Tek giriş noktası, iç karmaşıklığı gizler |
| Orchestrator | `CognitiveOrchestrator` | Bileşenleri koordine eder |
| Chain of Responsibility | `ProcessingPipeline` | Her handler tek sorumluluk |
| Strategy | `PolicyRouterV2` | Runtime'da algoritma seçimi |
| Observer / Pub-Sub | `EventBus` | Gevşek bağlı event yönetimi |
| Dependency Injection | `DependencyContainer` | Bağımlılıkları dışarıdan enjekte eder |
| Adapter | `BackendAdapter` | Model backend'ini soyutlar |
| Repository | `VectorStore` | Bellek kayıt/erişimini soyutlar |

### SOLID Prensipleri

- **S** (Single Responsibility): Her handler tek bir görevi yerine getirir
- **O** (Open/Closed): Yeni handler eklemek için mevcut handler'ları değiştirmek gerekmez
- **L** (Liskov Substitution): `IMemoryService`, `ICritic` vb. Protocol tanımlarına uymak yeterli
- **I** (Interface Segregation): `FullModelBackend`, `MinimalBackend` gibi ayrı Protocol'ler
- **D** (Dependency Inversion): `CognitiveOrchestrator` concrete sınıflara değil Protocol'lere bağlıdır

---

## İşleme Pipeline'ı

### Tam Akış Diyagramı

```
Kullanici Mesaji (user_message: str)
          |
          v
  [ValidationMW]  <- Giris dogrulama middleware
          |
  [MetricsMW]     <- Metrik sayaclari baslat
          |
  [TracingMW]     <- Dagitik izleme span baslat
          |
          v
  =========================================
         ProcessingPipeline
  =========================================

  1. FeatureExtractionHandler
     - QueryType sinifla
     - DomainType belirle
     - Karmasiklik skoru hesapla
     - Model entropi tahmini (0-3 normalize)

  2. MemoryRetrievalHandler
     - Oturum gecmisini al (token siniri ile budanmis)
     - RAG: VektorStore'dan ilgili bellek parcalari cek

  3. ToolHandler
     - Arac gerekli mi? (heuristik kural eslesme)
     - Gerekirse araci calistir (calculator / search / file)

  4. PolicyRoutingHandler
     - Entropi + uzunluk --> mod sec
     - Domain --> sicaklik ayarla
     - DecodingConfig olustur

  5. SelfConsistencyHandler
     - Yalnizca mode == "self_consistency" ise aktif
     - N orneklem uret (varsayilan N=3)
     - Cogunluk / hibrit secim

  6. ContextBuildingHandler
     - Yapılandırılmış prompt inşa et:
       [SYSTEM | MEMORY | COT | USER]

  7. DeliberationHandler
     - CoT / ToT / debate / react adimlarini uret
     - ThoughtCandidate listesi olustur

  8. GenerationHandler
     - Nihai model cagrisi (backend.generate)
     - ReasoningTrace kaydet

  9. CriticHandler
     - Constitutional AI kontrolü
     - Risk / guvenlik kontrolü
     - Gercek dogrulama (Wikipedia vb.)
     - Self-Refine revizyonu

  10. MemoryUpdateHandler
      - Gecmise yaz
      - Her 6 turda: ozet + vektor kayit
  =========================================
          |
          v
  CognitiveOutput (text, mode, metadata, ...)
```

### ProcessingContext

Her handler `ProcessingContext` nesnesini alır ve değiştirir:

```python
@dataclass
class ProcessingContext:
    cognitive_input: CognitiveInput    # Orijinal kullanici girdisi
    cognitive_state: CognitiveState    # Oturum durumu
    policy_output: PolicyOutput | None # Politika karari
    context_messages: list[dict]       # Prompt icin mesaj listesi
    raw_response: str                  # Model ham ciktisi
    final_response: str                # Islenmi son yanit
    tool_result: str | None            # Arac ciktisi
    metadata: dict                     # Handler'lar arasi paylasilan meta
```

---

## Chain of Responsibility Handlers

Her handler `BaseProcessingHandler`'dan türer ve `handle(ctx)` metodunu uygular:

```python
class BaseProcessingHandler:
    def __init__(self):
        self._next: BaseProcessingHandler | None = None

    def set_next(self, handler) -> BaseProcessingHandler:
        self._next = handler
        return handler

    def handle(self, ctx: ProcessingContext) -> ProcessingContext:
        # Alt sinif uygular; sonunda self._next.handle(ctx) cagrilir
        raise NotImplementedError
```

### Handler Açıklamaları

#### 1. FeatureExtractionHandler
- Kullanıcı mesajından `QueryType`, `DomainType`, karmaşıklık skoru çıkarır
- Model backend'inden logit entropisi hesaplar
- `ctx.metadata["features"]` alanını doldurur
- `ReasoningTrace(source="direct")` başlatır

#### 2. MemoryRetrievalHandler
- `MemoryServiceV2`'den oturum geçmişi alır (token sınırına göre budanmış)
- `enable_rag=True` ise: Kullanıcı mesajını embed et → vektör benzerlik sorgusu → ilgili bellek parçaları
- `ctx.metadata["memory_context"]` ve `ctx.metadata["memory_hits"]` alanlarını doldurur

#### 3. ToolHandler
- `ToolPolicyV2` ile araç kullanım kararı: `none` | `maybe` | `must`
- `must` ise `ToolExecutorV2` ile aracı çalıştır
- Sonuç `ctx.tool_result` alanına yazılır

#### 4. PolicyRoutingHandler
- `PolicyRouterV2.route()` çağrısı
- Giriş: entropi, uzunluk, query_type, domain
- Çıkış: `PolicyOutput(mode, decoding, inner_steps)`
- `ctx.policy_output` güncellenir

#### 5. SelfConsistencyHandler
- Yalnızca `mode == "self_consistency"` ise aktif
- `N` kez `backend.generate()` çağrısı (varsayılan N=3)
- `_bigram_overlap()` ile aday benzerliği ölç
- `method = "majority"` → kelime örtüşmesine göre; `"score"` → kalite puanına göre; `"hybrid"` → her ikisi
- `ctx.metadata["self_consistency_result"]` doldurulur

#### 6. ContextBuildingHandler
- Nihai prompt mesaj listesini inşa eder:
  - `[SYSTEM]` → system_prompt
  - `[MEMORY]` → vektör belleği + oturum özeti
  - `[COT]` → iç düşünce adımları (think1/debate2/tot ise)
  - `[USER]` → kullanıcı mesajı
- `ctx.context_messages` alanını doldurur

#### 7. DeliberationHandler
- `DeliberationEngineV2` ile akıl yürütme adımları üretir
- `think1`: Tek CoT adımı — "Adım adım düşünelim: ..."
- `debate2`: İki farklı perspektiften yanıt üret, en iyisini seç
- `tot`: `TreeOfThoughts` bileşeni ile BFS/beam search
- `react`: Düşünce → Eylem → Gözlem döngüsü
- `ctx.metadata["reasoning_steps"]` doldurulur

#### 8. GenerationHandler
- `backend.generate(messages, decoding)` ile nihai yanıt üretir
- `ReasoningTrace` listesine son adım eklenir
- Hata durumunda fallback olarak doğrudan üretim dener

#### 9. CriticHandler

```
Gelen yanitmetni
    |
    v
ConstitutionalCritic.check()
    - Varsayilan 16 ilke (siddet, zararli icerik, yanlilik vb.)
    - Ihlal bulunursa --> CriticFeedback(constitutional=True)
    |
    v
Risk / Guvenlik kontrolu
    - risk_keywords_sensitive mesajda var mi? -> risk_score artar
    - claim_markers yanita var mi? -> fact-check tetiklenir
    |
    v
Task-match skoru
    - Sorgu ile yanit kelime ortusumu
    - Dusuk overlap --> needs_revision = True
    |
    v
FactChecker.verify()  (enable_external_fact_checking=True ise)
    - WikipediaChecker: Iddia Wikipedia'da gecerli mi?
    - LLM fact verifier: Model kendi ciktisini sorgular
    |
    v
Revizyon gerekli mi?
    Evet --> Revizyon prompt olustur --> backend.generate()
             critic_passes += 1 (max_passes=1 varsayilan)
    Hayir --> Yanit onaylanir
```

#### 10. MemoryUpdateHandler
- `MemoryServiceV2.add_turn(role, content)` ile geçmişe yaz
- Her 6 turda: `_generate_session_summary()` → özet metni → vektör store'a kayıt
- `CognitiveState.query_type`, `CognitiveState.domain` alanları güncellenir

---

## CognitiveOrchestrator

`CognitiveOrchestrator` tüm bileşenlerin merkezi koordinatörüdür. Pipeline'ı oluşturur, middleware'leri zincirir, event'leri yayar.

### Yapılandırma

```python
orchestrator = CognitiveOrchestrator(
    backend=backend_adapter,           # FullModelBackend implementasyonu
    policy_router=policy_router,       # IPolicyRouter implementasyonu
    memory_service=memory_service,     # IMemoryService implementasyonu
    critic=critic,                     # ICritic implementasyonu
    deliberation_engine=deliberation,  # IDeliberationEngine (opsiyonel)
    event_bus=event_bus,               # EventBus (opsiyonel)
    middleware=[                       # Middleware zinciri (opsiyonel)
        ValidationMiddleware(),
        MetricsMiddleware(),
        TracingMiddleware(),
    ],
    performance_monitor=perf_mon,      # PerformanceMonitor (opsiyonel)
    tool_executor=tool_executor,       # ToolExecutor (opsiyonel)
)
```

### Senkron İşlem

```python
output = orchestrator.process(cognitive_input, cognitive_state)
```

### Asenkron İşlem

```python
output = await orchestrator.process_async(cognitive_input, cognitive_state)
```

---

## Middleware Katmanı

Middleware'ler istek/yanıt döngüsünde kesişen işlevler (cross-cutting concerns) için kullanılır. Her biri `handle_request()` ve `handle_response()` metotlarını uygular.

### Yerleşik Middleware'ler

**ValidationMiddleware**
- Gelen `CognitiveInput` nesnesini doğrular
- `user_message` boş ise hata fırlatır
- Güvenlik keyword filtreleme (opsiyonel)

**MetricsMiddleware**
- İstek sayacı, hata sayacı, gecikme histogramı
- Prometheus uyumlu metrik formatı
- `ctx.metadata["request_start_time"]` kayıt

**TracingMiddleware**
- Dağıtık izleme için span oluşturur
- `trace_id`, `span_id` üretir
- Handler bazlı süre ölçümü

### Özel Middleware Yazma

```python
from cognitive_management.v2.middleware.base import BaseMiddleware

class MyMiddleware(BaseMiddleware):
    def handle_request(self, ctx: ProcessingContext) -> ProcessingContext:
        ctx.metadata["my_flag"] = True
        return ctx

    def handle_response(self, ctx: ProcessingContext) -> ProcessingContext:
        print(f"Islem tamamlandi: {ctx.final_response[:50]}")
        return ctx
```

---

## Event Bus

`EventBus` bileşenler arası gevşek bağlı iletişim sağlar. Herhangi bir bileşen, diğer bileşenlere doğrudan referans tutmak zorunda kalmadan event yayabilir.

### CognitiveEvent Tipleri

| Event | Tetiklendiği An |
|-------|----------------|
| `REQUEST_STARTED` | İstek pipeline'a girdiğinde |
| `POLICY_SELECTED` | PolicyRouter mod seçtiğinde |
| `DELIBERATION_COMPLETE` | Akıl yürütme adımları tamamlandığında |
| `GENERATION_COMPLETE` | Model çıktı ürettiğinde |
| `CRITIC_TRIGGERED` | Critic değerlendirme başlattığında |
| `REVISION_APPLIED` | Self-Refine revizyonu yapıldığında |
| `MEMORY_UPDATED` | Bellek güncellediğinde |
| `REQUEST_COMPLETE` | Tüm pipeline tamamlandığında |
| `ERROR_OCCURRED` | Herhangi bir hata oluştuğunda |

### Kullanım

```python
# Event'e abone ol
def on_policy_selected(event: CognitiveEvent):
    print(f"Secilen mod: {event.data['mode']}")

event_bus.subscribe("POLICY_SELECTED", on_policy_selected)

# Async handler kaydet
async def on_generation(event: CognitiveEvent):
    await log_to_external_service(event.data)

event_bus.subscribe_async("GENERATION_COMPLETE", on_generation)
```

---

## Dependency Injection Container

`DependencyContainer` bileşenlerin yaşam döngüsünü ve bağımlılıklarını yönetir.

```python
from cognitive_management.v2.container.dependency_container import DependencyContainer

container = DependencyContainer()

# Singleton kayit
container.register_singleton("event_bus", EventBus())
container.register_singleton("memory_service", MemoryServiceV2(config.memory))

# Factory kayit (her resolve'da yeni instance)
container.register_factory("critic", lambda: CriticV2(backend, config.critic))

# Cozumleme
event_bus = container.resolve("event_bus")
```

---

## Interfaces / Protocol Tanımları

### Backend Protocols (`backend_protocols.py`)

```python
class MinimalBackend(Protocol):
    """Sadece generate() gerektiren minimal backend."""
    def generate(self, messages: list[dict], config: DecodingConfig) -> str: ...

class FullModelBackend(Protocol):
    """Tam ozellikli backend - embed + generate + forward."""
    def generate(self, messages: list[dict], config: DecodingConfig) -> str: ...
    def embed(self, text: str) -> list[float]: ...
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def entropy_estimate(self, text: str) -> float: ...
```

### Component Protocols (`component_protocols.py`)

```python
class PolicyRouter(Protocol):
    def route(self, features: dict, state: CognitiveState) -> PolicyOutput: ...

class MemoryService(Protocol):
    def get_history(self, max_tokens: int) -> list[dict]: ...
    def add_turn(self, role: str, content: str) -> None: ...
    def retrieve_relevant(self, query: str, top_k: int) -> list[dict]: ...

class DeliberationEngine(Protocol):
    def deliberate(self, mode: Mode, context: str, config: DecodingConfig) -> list[ThoughtCandidate]: ...

class Critic(Protocol):
    def review(self, text: str, query: str) -> tuple[bool, list[CriticFeedback]]: ...

class ToolExecutor(Protocol):
    def execute(self, tool_name: str, args: dict) -> str: ...
```

---

## Asenkron Pipeline

`AsyncProcessingPipeline` ve `AsyncProcessingHandler` sınıfları, senkron pipeline'ın tam asenkron karşılığıdır.

### Asenkron Handler Yapısı

```python
class BaseAsyncProcessingHandler:
    async def handle(self, ctx: ProcessingContext) -> ProcessingContext:
        # Islemi yap
        result = await some_async_operation()
        ctx.metadata["result"] = result
        # Zinciri devam ettir
        if self._next:
            return await self._next.handle(ctx)
        return ctx
```

### Asenkron Pipeline Oluşturma

```python
pipeline = AsyncProcessingPipeline()
pipeline.add_handler(AsyncFeatureExtractionHandler(backend))
pipeline.add_handler(AsyncMemoryRetrievalHandler(memory_service))
pipeline.add_handler(AsyncGenerationHandler(backend))

output = await pipeline.execute(ctx)
```

---

## Bileşen Detayları

### PolicyRouterV2 — Karar Mantığı

```
entropi < entropy_gate_think (1.5)
    --> mode = "direct"

entropy_gate_think <= entropi < entropy_gate_debate (2.5)
    --> mode = "think1"  (Chain-of-Thought)
    --> inner_steps = 1

entropy_gate_debate <= entropi < entropy_gate_tot (3.0)
    --> mode = "debate2" veya "self_consistency"
    --> inner_steps = self_consistency_n (3)

entropi >= entropy_gate_tot (3.0) VE uzunluk > length_gate_tot (300)
    --> mode = "tot"
    --> inner_steps = tot_max_depth x tot_branching_factor
```

Domain bazlı sıcaklık ayarı:

```python
temperature_map = {
    "math":     0.45,   # config.decoding_bounds.math_temperature
    "code":     0.50,   # config.decoding_bounds.code_temperature
    "creative": 0.85,   # config.decoding_bounds.creative_temperature
    # Diger domain'ler: default_decoding.temperature (0.65)
}
```

### TreeOfThoughts Algoritması

```
Kok dugum: baslangic baglamlari
    |
    +-- Genislet (branching_factor = 3 aday)
    |       |
    |       +-- Degerlendir (scorer: kelime/anlam uyum skoru)
    |       |
    |       +-- Budama (top_k = 5 en iyi tut)
    |
    +-- Derinligi artir (max_depth = 3'e kadar)
    |
    +-- En yuksek skorlu yaprak --> nihai yanit
```

### MemoryServiceV2 — Bellek Katmanları

```
Oturum Bellegi (RAM - kisa donem)
    history: list[dict]
    max_history_tokens: 3072
    Budama: token siniri asilinca eski mesajlari sil

Epizodik Vektor Bellegi (ChromaDB - uzun donem)
    Collection: "cognitive_memory"
    Path: "./memory/episodic_store"
    Embedding: sentence-transformers/all-MiniLM-L6-v2
    top_k: 5, score_threshold: 0.7

Her 6 turda:
    1. Oturum ozeti uret
    2. Ozeti embed et (EmbeddingAdapter)
    3. ChromaDB'ye yaz (metadata + vektor)
    4. history.clear() (eski mesajlar vektor'de artik)
```

### AIOps İzleme Sistemi

```
PerformanceMonitor
    - Request latency (ortalama, p95, p99)
    - Token/saniye throughput
    - Handler bazli sure profili

AnomalyDetector
    - Latency spike tespiti (EWMA tabanli)
    - Error rate artisi
    - Bellek kullanim anomalisi

TrendAnalyzer
    - Zaman serisi trend analizi
    - Mevsimsellik tespiti

PredictiveAnalytics
    - Yaklasan yuk artisi tahmini
    - Onleyici cache isitma onerileri

AlertingManager
    - Threshold bazli uyari uretimi
    - Webhook / callback entegrasyonu

HealthCheck
    - Sistem durumu ozeti
    - Backend baglanti sagligi
    - VectorStore baglanti sagligi
```
