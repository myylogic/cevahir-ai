# Code Reality — Cognitive

> Kanonik Birim #7 · Kaynak: `cognitive_management/` (v2)
> Akıl yürütme / bilişsel katmanın kodundan çıkarılmış mimarisi.
> Bağlam: [master-architecture](../../master-architecture.md) (L5), [search index](../../architecture-search-index.md).
>
> ⚠️ Bu doküman **doğrudan koddan** üretilmiştir. Depodaki eski
> `cognitive_management/docs` içeriğiyle (ör. handler listesi) **çelişebilir**;
> çelişkide bu doküman ve kaynak kod esastır.

---

## 1. Kimlik

| Alan | Değer |
|------|-------|
| **Birim** | Cognitive (akıl yürütme katmanı) |
| **Kaynak dizin** | `cognitive_management/` (üst düzey + `v2/`) |
| **Toplam boyut** | ~20.600 LOC (~3.046 üst + ~17.581 v2; testler hariç) |
| **Facade** | `CognitiveManager` — `cognitive_management/cognitive_manager.py:113` |
| **Orkestratör** | `CognitiveOrchestrator` — `v2/core/orchestrator.py:73` |
| **Giriş sözleşmesi** | `ModelAPI` protokolü (Model/Engine'in `CevahirModelAPI`'si uyar) |
| **Çalışma zamanı** | Online (çıkarım) |
| **Dış bağımlılık** | `torch` (dolaylı), ChromaDB (vektör bellek), sentence-transformers (embedding) |

---

## 2. Sorumluluk

Modelin yalnızca metin üretmesini değil **düşünmesini** sağlamak: sorgu
özniteliklerini çıkarmak, uygun akıl yürütme modunu seçmek (direct / think / debate /
tot / self-consistency), bellekten bağlam getirmek (RAG), araç kullanmak, yanıtı
üretmek, eleştirmen (critic) ile güvenlik/doğruluk denetlemek ve belleği güncellemek.
Ayrıca middleware, event bus, DI container, önbellek ve AIOps izleme sağlar.

**Kapsam dışı:** Model ileri geçişi (Neural Network), tokenizasyon (Tokenizer),
model IO (Model Mgmt). Cognitive, backend'e yalnızca `ModelAPI` protokolü üzerinden
erişir.

---

## 3. Dosya Envanteri

### 3.1 Üst Düzey

| Dosya | LOC | Rol | Anahtar üyeler |
|-------|-----|-----|----------------|
| `cognitive_manager.py` | 2070 | **`CognitiveManager`** (Facade) + `ModelAPI` protokolü | `handle`, `handle_multimodal`, `get_metrics`, `get_health_status`, event/cache/trace API'leri |
| `cognitive_types.py` | 370 | Veri tipleri (`CognitiveInput/Output`, `DecodingConfig`, `PolicyOutput`, `ThoughtCandidate`, ...) | — |
| `config.py` | 469 | `CognitiveManagerConfig` + alt config'ler | — |
| `exceptions.py` | 137 | Bilişsel hata hiyerarşisi | — |

### 3.2 Çekirdek + İşleme (`v2/core`, `v2/processing`)

| Dosya | LOC | Sınıf | Rol |
|-------|-----|-------|-----|
| `core/orchestrator.py` | 768 | **`CognitiveOrchestrator`** | Pipeline + middleware kurar, sync/async/batch `handle` |
| `processing/pipeline.py` | 316 | `ProcessingPipeline`, `BaseProcessingHandler` | Chain of Responsibility altyapısı |
| `processing/handlers.py` | 795 | **8 handler** (aşağıda) | Sync işleme adımları |
| `processing/async_pipeline.py` / `async_handlers.py` | — | Asenkron karşılıklar | Tam async yol |

**Gerçek handler zinciri** (`orchestrator._build_pipeline`, sırayla):
`FeatureExtractionHandler` → `PolicyRoutingHandler` → `DeliberationHandler` →
`ContextBuildingHandler` → `GenerationHandler` → `SelfConsistencyHandler` →
`CriticHandler` → `MemoryUpdateHandler`.

> Bellek getirme `FeatureExtractionHandler`/`ContextBuildingHandler`'a
> (`memory_service` ile), araç kullanımı `ContextBuildingHandler`'a (`tool_policy`
> ile) **gömülüdür** — ayrı handler değildir.

### 3.3 Bileşenler (`v2/components`)

| Dosya | LOC | Sınıf | Rol |
|-------|-----|-------|-----|
| `critic_v2.py` | 775 | `CriticV2(ICritic)` | Ana eleştirmen: risk, task-match, fact-check, self-refine |
| `constitutional_critic.py` | 453 | `ConstitutionalCritic` | İlke tabanlı içerik denetimi |
| `memory_service_v2.py` | 680 | `MemoryServiceV2(IMemoryService)` | Oturum + epizodik vektör bellek |
| `tree_of_thoughts.py` | 582 | `TreeOfThoughts` | ToT BFS/beam akıl yürütme |
| `deliberation_engine_v2.py` | — | `DeliberationEngineV2` | CoT/debate/ToT/react üretimi |
| `policy_router_v2.py` | 349 | `PolicyRouterV2` | Entropi/uzunluk → mod seçimi |
| `tool_executor_v2.py` | 371 | `ToolExecutorV2` | Araç çalıştırma + metrik |
| `tool_policy_v2.py` | — | `ToolPolicyV2` | Araç gerekliliği kararı (none/maybe/must) |
| `embedding_adapter.py` | 306 | `EmbeddingAdapter` | Metin → vektör |
| `rag_enhancer.py` | — | RAG zenginleştirme | Getirme sonrası bağlam |
| `vector_store/` | — | `base.py`, `chroma_vector_store.py`, `memory_vector_store.py` | Vektör deposu (Repository deseni) |

### 3.4 Kesişen Altyapı

| Dizin | Dosyalar | Rol |
|-------|----------|-----|
| `v2/middleware` | `base`, `validation`, `metrics`, `tracing`, `cache`, `error_handler`, `async_middleware` | İstek/yanıt kesişen ilgiler |
| `v2/events` | `event_bus`, `event_handlers` | Observer/Pub-Sub |
| `v2/container` | `dependency_container` | DI container |
| `v2/interfaces` | `backend_protocols`, `component_protocols` | Protocol tanımları (DIP) |
| `v2/adapters` | `backend_adapter` | Backend soyutlama |
| `v2/config` | `config_manager`, `constitutional_principles` | Config + ilke verisi |
| `v2/monitoring` | `performance_monitor`, `anomaly_detector`, `trend_analyzer`, `predictive_analytics`, `alerting`, `health_check` | AIOps |
| `v2/utils` | `cache`, `semantic_cache`, `cache_warming`, `context_pruning`, `claim_extraction`, `connection_pool`, `request_batcher`, `heuristics`, `selectors`, `performance_profiler`, `tracing` | Yardımcılar |

---

## 4. İç Mimari (katmanlı desenler)

```
┌─────────────────────────────────────────────────────────────────────┐
│  CognitiveManager  (Facade)              cognitive_manager.py         │
│  handle(user_message) · metrics · health · events · cache · traces   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│  CognitiveOrchestrator  (Orchestrator)   v2/core/orchestrator.py     │
│                                                                      │
│   ┌───────────────┐   ┌───────────────┐   ┌───────────────────────┐ │
│   │ Middleware    │   │ EventBus      │   │ Monitoring (AIOps)    │ │
│   │ chain         │   │ (Pub/Sub)     │   │ perf/anomaly/alert    │ │
│   └───────┬───────┘   └───────────────┘   └───────────────────────┘ │
│           │  DI: DependencyContainer, interfaces/ (Protocol)         │
│           ▼                                                          │
│   ┌──────────────────────────────────────────────────────────────┐ │
│   │  ProcessingPipeline   (Chain of Responsibility)              │ │
│   │  H1 → H2 → ... → H8  (ProcessingContext taşınır/değiştirilir) │ │
│   └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
             │ backend: ModelAPI protokolü (asla somut model değil)
             ▼
     CevahirModelAPI  [Model/Engine birimi] → NN → Tokenizer
```

**Uygulanan desenler (kodda gözlemlenen):** Facade (`CognitiveManager`),
Orchestrator (`CognitiveOrchestrator`), Chain of Responsibility (`ProcessingPipeline`),
Strategy (`PolicyRouterV2`), Observer (`EventBus`), Dependency Injection
(`DependencyContainer`), Adapter (`BackendAdapter`), Repository (`vector_store`),
Protocol-based DIP (`interfaces/`).

---

## 5. Veri / Kontrol Akışı — İşleme Zinciri

```
CognitiveInput (user_message)
   │  [Middleware: validation → metrics → tracing → cache]
   ▼
1. FeatureExtractionHandler
      · QueryType/Domain/karmaşıklık + entropi (backend.entropy_estimate)
      · (memory_service ile) oturum geçmişi/RAG getirme
2. PolicyRoutingHandler
      · PolicyRouterV2.route(entropi, uzunluk, domain) → mode + DecodingConfig
3. DeliberationHandler
      · mode'a göre CoT(think) / debate / ToT / react adımları
      · TreeOfThoughts / DeliberationEngineV2
4. ContextBuildingHandler
      · [SYSTEM | MEMORY | COT | USER] prompt inşası
      · tool_policy ile araç gerekliyse ToolExecutorV2 çalıştır
5. GenerationHandler
      · backend.generate(context, decoding) → ham yanıt
6. SelfConsistencyHandler   (yalnız mode == self_consistency)
      · N örneklem + çoğunluk/skor/hibrit seçim
7. CriticHandler
      · ConstitutionalCritic (ilkeler) + CriticV2 (risk/task-match/fact-check)
      · gerekirse self-refine revizyon (backend.generate tekrar)
8. MemoryUpdateHandler
      · MemoryServiceV2.add_turn; periyodik özet → vektör store
   │  [Middleware: response tarafı]
   ▼
CognitiveOutput (text, mode, metadata, trace)
```

### 5.1 Politika Kararı (PolicyRouterV2 — Strategy)

Entropi eşiklerine göre mod: düşük → `direct`, orta → `think` (CoT), yüksek →
`debate`/`self_consistency`, çok yüksek + uzun → `tot`. Domain'e göre sıcaklık
ayarı (math/code düşük, creative yüksek). Kesin eşikler `config.py`/`PolicyRouterV2`.

### 5.2 Bellek (MemoryServiceV2 — iki katman)

Kısa dönem: RAM oturum geçmişi (token sınırı ile budama). Uzun dönem: ChromaDB
epizodik vektör bellek (`vector_store/chroma_vector_store.py`), embedding
`EmbeddingAdapter` ile. Periyodik özet → vektör kayıt.

---

## 6. Genişletme Noktaları

| Ne | Nereye | Not |
|----|--------|-----|
| Yeni akıl yürütme modu | `PolicyRouterV2` (mod ekle) + `DeliberationHandler` | `PolicyOutput.mode` sözleşmesi |
| Yeni işleme adımı | `processing/handlers.py` + `orchestrator._build_pipeline` zincire ekle | `BaseProcessingHandler.handle(ctx)` |
| Yeni middleware | `v2/middleware/base.py` (`BaseMiddleware`) | request/response metotları |
| Yeni critic kuralı | `CriticV2` / `ConstitutionalCritic` + `constitutional_principles.py` | `ICritic` protokolü |
| Yeni araç | `ToolExecutorV2` + `ToolPolicyV2` | `ToolExecutor` protokolü |
| Yeni vektör deposu | `vector_store/base.py` uygula | Repository arayüzü |
| Yeni event | `events/event_bus.py` + subscriber | Pub/Sub |
| Backend değişimi | `interfaces/` protokollerini karşıla | Somut modele dokunma |

---

## 7. Bağımlılıklar

**Bağımlı olduğu:** yalnızca `ModelAPI` protokolü (Model/Engine `CevahirModelAPI`'si
sağlar), ChromaDB + sentence-transformers (vektör bellek/embedding).
**Buna bağımlı olanlar:** Model/Engine (`Cevahir._init_cognitive`), dolaylı olarak
API/Chatting.

> **Temiz sınır:** Cognitive, somut sinir ağını **görmez**; yalnızca protokole
> bağlıdır (DIP). Refactor'da backend değişimi bu birimi etkilemez.

---

## 8. Refactor Sinyalleri / Tech-Debt

| Sinyal | Kanıt | Risk | Not |
|--------|-------|------|-----|
| **Dev facade** | `cognitive_manager.py` 2070 LOC | Yüksek | Facade + event/cache/trace API'leri ayrılabilir |
| **Doküman-kod uyumsuzluğu** | Eski doc 10 handler diyor; kodda 8 var (memory/tool gömülü) | Orta | Eski `cognitive_management/docs` bu doküman ile değiştirilmeli |
| **Sync/async ikizliği** | `pipeline`+`handlers` vs `async_pipeline`+`async_handlers`; middleware de çift | Yüksek | İki yolun senkron bakımı; ortak çekirdek |
| **Çoklu ModelAPI protokolü** | `ModelAPI` hem `cognitive_manager.py` hem `critic_v2.py` içinde tanımlı | Orta | Tek yerde (interfaces/) toplanmalı |
| **Cache çeşitliliği** | `middleware/cache`, `utils/cache`, `utils/semantic_cache`, `cache_warming` | Orta | Cache stratejisi birleştirilebilir |
| **Ağır izleme yükü** | 6 monitoring modülü (anomaly/trend/predictive/alerting...) | Düşük-Orta | Kullanım/aktiflik teyit edilmeli |
| **`v2` tek sürüm** | v3 yok ama isim "v2" | Düşük | Sürümleme kararı (v2 kanonik mi?) |

---

## 9. Kod Referansları

| Amaç | Referans |
|------|----------|
| Facade giriş | `cognitive_management/cognitive_manager.py:509` (`handle`) |
| Orkestratör | `cognitive_management/v2/core/orchestrator.py:73` |
| Pipeline kurulumu | `v2/core/orchestrator.py:154` (`_build_pipeline`) |
| Handler zinciri | `v2/processing/handlers.py:151` (ilk handler) |
| Politika seçimi | `v2/components/policy_router_v2.py` (`PolicyRouterV2`) |
| ToT | `v2/components/tree_of_thoughts.py` |
| Critic | `v2/components/critic_v2.py:164` (`CriticV2`) |
| Anayasal critic | `v2/components/constitutional_critic.py` |
| Bellek servisi | `v2/components/memory_service_v2.py:67` |
| Vektör deposu | `v2/components/vector_store/chroma_vector_store.py` |
| Araç çalıştırıcı | `v2/components/tool_executor_v2.py:73` |
| Event bus | `v2/events/event_bus.py` |
| DI container | `v2/container/dependency_container.py` |
| Protokoller | `v2/interfaces/{backend,component}_protocols.py` |

---

*Kaynak: `cognitive_management/` — analiz kodun mevcut halinden çıkarılmıştır.*
