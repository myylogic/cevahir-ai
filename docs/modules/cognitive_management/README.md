# Cognitive Management Modülü

**Versiyon:** V2 (Enterprise)
**Son Güncelleme:** 2026-03-16
**Durum:** Production-Ready

---

## İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Dizin Yapısı](#dizin-yapısı)
3. [Ne İş Yapar?](#ne-i̇ş-yapar)
4. [Temel Veri Tipleri](#temel-veri-tipleri)
5. [Bilişsel Modlar](#bilişsel-modlar)
6. [Hızlı Başlangıç](#hızlı-başlangıç)
7. [Bağımlılıklar](#bağımlılıklar)

---

## Genel Bakış

**Cognitive Management**, Cevahir Sinir Sistemi'nin üst-bilişsel (meta-cognitive) katmanıdır. Modelin ham metin üretiminin ötesine geçerek:

- Sorunun ne tür bir sorgu olduğunu sınıflandırır
- En uygun akıl yürütme stratejisini seçer (doğrudan yanıt, zincir düşünce, ağaç düşünce vb.)
- Gerekirse araçları (hesap makinesi, arama, dosya) kullanır
- Üretilen yanıtı eleştirel gözle değerlendirir ve gerekirse revize eder
- Oturum geçmişini vektör tabanlı bellekte saklar, ilgili geçmiş bağlamı otomatik olarak geri çeker

Sistem, akademik LLM araştırmalarından (CoT, ToT, Self-Consistency, Self-Refine, Constitutional AI) esinlenen bileşenleri endüstriyel enterprise desenleriyle (Dependency Injection, Chain of Responsibility, Event Bus, Middleware) birleştirir.

### Tasarım Felsefesi

```
Kullanıcı Mesajı
       │
       ▼
┌──────────────────────────────────────┐
│          CognitiveManager            │  ← Giriş noktası (facade)
│  ┌────────────────────────────────┐  │
│  │      CognitiveOrchestrator     │  │  ← İç koordinatör
│  │                                │  │
│  │  Middleware → Pipeline         │  │  ← İşleme zinciri
│  │     ↓                          │  │
│  │  PolicyRouter (strateji seç)   │  │
│  │     ↓                          │  │
│  │  DeliberationEngine (düşün)    │  │
│  │     ↓                          │  │
│  │  MemoryService (bellek yönet)  │  │
│  │     ↓                          │  │
│  │  CriticV2 (değerlendir/revize) │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
       │
       ▼
  CognitiveOutput
```

---

## Dizin Yapısı

```
cognitive_management/
│
├── cognitive_manager.py          # Ana giriş noktası — CognitiveManager sınıfı
├── cognitive_types.py            # Tüm veri tipleri (dataclass + Literal)
├── config.py                     # CognitiveManagerConfig ve alt config'ler
├── exceptions.py                 # Exception hiyerarşisi
│
├── utils/
│   ├── logging.py                # Loglama yardımcıları
│   └── timers.py                 # Zamanlama yardımcıları
│
└── v2/                           # Enterprise V2 mimarisi
    │
    ├── adapters/
    │   └── backend_adapter.py    # Model backend'ini soyutlayan adaptör
    │
    ├── components/               # Temel bilişsel bileşenler
    │   ├── critic_v2.py          # CriticV2 — Self-Refine döngüsü
    │   ├── constitutional_critic.py  # Constitutional AI kontrol katmanı
    │   ├── deliberation_engine_v2.py # CoT / ToT / debate üretimi
    │   ├── embedding_adapter.py  # Embedding sağlayıcı adaptörü
    │   ├── memory_service_v2.py  # Oturum belleği + vektör RAG
    │   ├── policy_router_v2.py   # Strateji seçim motorou
    │   ├── rag_enhancer.py       # RAG bağlam zenginleştirici
    │   ├── tool_executor_v2.py   # Araç çalıştırıcı
    │   ├── tool_policy_v2.py     # Araç kullanım politikası
    │   ├── tree_of_thoughts.py   # ToT arama algoritması
    │   │
    │   ├── fact_checkers/        # Dış kaynaklı gerçek doğrulama
    │   │   ├── base.py           # FactChecker arayüzü
    │   │   └── wikipedia_checker.py
    │   │
    │   └── vector_store/         # Vektör veritabanı katmanı
    │       ├── base.py           # VectorStore arayüzü
    │       ├── chroma_vector_store.py   # ChromaDB implementasyonu
    │       └── memory_vector_store.py   # In-memory implementasyonu
    │
    ├── config/
    │   ├── config_manager.py     # Çalışma zamanı config yöneticisi
    │   └── constitutional_principles.py  # Varsayılan anayasal ilkeler
    │
    ├── container/
    │   └── dependency_container.py  # DI Container — bağımlılık enjeksiyonu
    │
    ├── core/
    │   └── orchestrator.py       # CognitiveOrchestrator — merkezi koordinatör
    │
    ├── events/
    │   ├── event_bus.py          # EventBus — asenkron/senkron event yönetimi
    │   └── event_handlers.py     # Yerleşik event dinleyicileri
    │
    ├── interfaces/
    │   ├── backend_protocols.py  # Model backend Protocol tanımları
    │   └── component_protocols.py  # Bileşen Protocol tanımları
    │
    ├── middleware/
    │   ├── metrics.py            # Metrik toplama middleware
    │   ├── tracing.py            # Dağıtık izleme middleware
    │   └── validation.py         # Girdi doğrulama middleware
    │
    ├── monitoring/               # AIOps izleme alt sistemi
    │   ├── alerting.py           # Uyarı yöneticisi
    │   ├── anomaly_detector.py   # Anomali tespiti
    │   ├── health_check.py       # Sağlık kontrol API'si
    │   ├── performance_monitor.py # Performans izleyici
    │   ├── predictive_analytics.py # Tahminsel analitik
    │   └── trend_analyzer.py     # Trend analizi
    │
    ├── processing/
    │   ├── handlers.py           # Chain of Responsibility handler'ları
    │   ├── pipeline.py           # Senkron işleme pipeline'ı
    │   ├── async_handlers.py     # Asenkron handler versiyonları
    │   └── async_pipeline.py     # Asenkron pipeline
    │
    └── utils/
        ├── cache.py              # In-memory LRU cache
        ├── cache_warming.py      # Cache ön ısıtma
        ├── claim_extraction.py   # İddia çıkarımı (fact-checking için)
        ├── connection_pool.py    # Bağlantı havuzu
        ├── context_pruning.py    # Bağlam budama
        ├── heuristics.py         # build_features — sorgu özelliği çıkarımı
        ├── performance_profiler.py # Performans profiler
        ├── request_batcher.py    # Toplu istek işleyici
        ├── selectors.py          # Self-Consistency aday seçici
        ├── semantic_cache.py     # Anlamsal benzerlik cache'i
        └── tracing.py            # İzleme yardımcıları
```

---

## Ne İş Yapar?

### 1. Sorgu Sınıflandırma

Her kullanıcı mesajı işlenmeden önce `FeatureExtractionHandler` tarafından analiz edilir:

- **QueryType**: `factual`, `reasoning`, `creative`, `conversational`, `math`, `code`, `unknown`
- **DomainType**: `math`, `science`, `law`, `medical`, `technology`, `history`, `creative`, `general`
- **Karmaşıklık skoru** (0.0–1.0): Mesaj uzunluğu, kelime çeşitliliği, soru işareti yoğunluğuna göre
- **Entropi tahmini**: Model logit belirsizliği — hangi akıl yürütme moduna geçileceğini belirler

### 2. Strateji Seçimi (PolicyRouter)

Entropi ve uzunluk eşiklerine göre mod seçilir:

| Entropi | Uzunluk | Seçilen Mod | Akademik Kaynak |
|---------|---------|-------------|-----------------|
| < 1.5 | — | `direct` | — |
| 1.5–2.5 | — | `think1` (CoT) | Wei et al. 2022 |
| 2.5–3.0 | > 200 tok. | `debate2` / `self_consistency` | Wang et al. 2022 |
| > 3.0 | > 300 tok. | `tot` | Yao et al. 2023 |

Ayrıca domain'e göre sıcaklık otomatik ayarlanır:
- `math` / `code` → temperature ≈ 0.45–0.50
- `creative` → temperature ≈ 0.85

### 3. Deliberation (Akıl Yürütme)

`DeliberationEngineV2` seçilen moda göre iç düşünce adımları üretir:
- **think1**: Tek CoT adımı — önce düşün, sonra yanıtla
- **debate2**: İki paralel aday üretimi, kazanan seçilir
- **tot**: Ağaç arama — genişletme → değerlendirme → budama
- **self_consistency**: N örneklem (varsayılan 3), çoğunluk/hibrit seçim

### 4. Bellek Yönetimi (MemoryService)

İki katmanlı bellek:
- **Oturum geçmişi** (kısa dönem): Son N mesaj, token sınırına göre budanır
- **Epizodik vektör belleği** (uzun dönem): ChromaDB ile disk'e kalıcı olarak yazılır

Her 6 turda oturum özeti üretilir ve vektör store'a kayıt edilir. Yeni sorgularda cosine benzerliği ile ilgili geçmiş otomatik geri çekilir (RAG).

### 5. Eleştiri ve Revizyon (CriticV2)

Üretilen yanıt şu aşamalardan geçer:

1. **Constitutional AI kontrolü** — Anayasal ilkelere uygunluk
2. **Güvenlik/risk kontrolü** — Hassas alan tespiti
3. **Görev uyumu kontrolü** — Yanıt soruyu yanıtlıyor mu?
4. **İddia yoğunluğu** — Sayısal/istatistiksel iddia varsa dış doğrulama tetiklenir
5. **Dış gerçek doğrulama** — Wikipedia (ve opsiyonel Google/Wolfram)
6. **Self-Refine revizyonu** — Revizyon gerekliyse model yeniden üretim yapar

### 6. Araç Kullanımı

`ToolPolicyV2` heuristik kurallara göre araç kararı verir:

| Tetikleyici | Araç |
|-------------|------|
| "bugün", "güncel", "son haber" | `search` |
| "hesapla", "toplam", "oran", "%" | `calculator` |
| "dosya", "oku", "kaydet" | `file` |

---

## Temel Veri Tipleri

### Giriş / Çıkış Tipleri

```python
@dataclass
class CognitiveInput:
    user_message: str          # Kullanıcı mesajı (zorunlu)
    system_prompt: str | None  # Sistem davranış yönergesi (opsiyonel)
    metadata: dict             # Ek sinyaller, risk bayrakları

@dataclass
class CognitiveOutput:
    text: str                        # Üretilen yanıt
    used_mode: Mode                  # Kullanılan bilişsel mod
    tool_used: str | None            # Kullanılan araç adı
    revised_by_critic: bool          # Critic revizyonu yapıldı mı?
    reasoning_chain: list[ReasoningTrace]  # Akıl yürütme adımları
    critic_passes: int               # Kaç Self-Refine turu yapıldı
    critic_feedback: list[CriticFeedback] | None  # Yapılandırılmış feedback
    memory_hits: int                 # RAG'dan çekilen bellek öğesi sayısı
    latency_ms: float                # Toplam işlem süresi (ms)
    query_type: QueryType | None     # Algılanan sorgu tipi
    domain: DomainType | None        # Algılanan alan
    self_consistency_result: SelfConsistencyResult | None
    context_sources: list[str]       # "vector", "history" vb.
    metadata: dict                   # İzleme verileri
```

### Oturum Durumu

```python
@dataclass
class CognitiveState:
    history: list[dict]            # {"role": ..., "content": ...} listesi
    step: int                      # Kaçıncı bilişsel tur
    last_entropy: float | None     # Son belirsizlik kestirimi
    last_mode: Mode | None         # Son kullanılan mod
    session_id: str                # 8 karakter UUID tabanlı oturum ID
    turn_count: int                # Toplam kullanıcı-asistan tur sayısı
    query_type: QueryType | None
    domain: DomainType | None
    reasoning_traces: list[ReasoningTrace]
    metadata: dict
```

### Akıl Yürütme İzleme

```python
@dataclass
class ReasoningTrace:
    step: int       # Adım numarası
    content: str    # Bu adımın içeriği
    score: float    # Kalite skoru (0.0–1.0)
    source: str     # "cot" | "tot" | "debate" | "direct" | "react"

@dataclass
class CriticFeedback:
    aspect: str           # "coherence", "relevance", "safety" vb.
    score: float          # 0.0–1.0
    message: str          # İnsan-okunabilir mesaj
    needs_revision: bool
    constitutional: bool  # Constitutional AI ihlali mi?

@dataclass
class SelfConsistencyResult:
    candidates: list[str]   # N aday yanıt
    selected: str           # Seçilen en iyi yanıt
    agreement_score: float  # Adaylar arası uyum (0.0–1.0)
    method: str             # "majority" | "score" | "hybrid"

@dataclass
class ThoughtCandidate:
    text: str          # Düşünce metni
    score: float       # Değerlendirme skoru
    depth: int         # ToT ağaç derinliği
    path: list[str]    # Bu noktaya gelen düşünceler zinciri
```

---

## Bilişsel Modlar

| Mod | Açıklama | Akademik Kaynak |
|-----|----------|-----------------|
| `direct` | Tek geçişli hızlı üretim — düşünme adımı yok | — |
| `think1` | Chain-of-Thought — önce iç düşünce, sonra yanıt | Wei et al. 2022 |
| `debate2` | İki aday üretimi, kazanan seçilir | Wang et al. 2022 |
| `tot` | Tree of Thoughts — ağaç tabanlı arama | Yao et al. 2023 |
| `react` | Reason+Act — düşünce ve eylem iç içe | Yao et al. 2022 |
| `self_consistency` | N örneklem, çoğunluk/hibrit oylama | Wang et al. 2022 |

---

## Hızlı Başlangıç

### Temel Kullanım

```python
from cognitive_management.cognitive_manager import CognitiveManager
from cognitive_management.config import CognitiveManagerConfig

# Varsayılan config ile başlat
config = CognitiveManagerConfig()
manager = CognitiveManager(model_manager=model, config=config)

# Yanıt al
output = manager.handle(
    user_message="Türkiye'nin en büyük şehri hangisidir?",
    system_prompt="Sen Cevahir, Türkçe konuşan bir asistansın.",
)

print(output.text)          # → "İstanbul, Türkiye'nin en büyük şehridir."
print(output.used_mode)     # → "direct"
print(output.latency_ms)    # → 124.5
```

### Özel Config ile Kullanım

```python
from cognitive_management.config import (
    CognitiveManagerConfig, PolicyConfig, MemoryConfig
)

config = CognitiveManagerConfig()

# Tree of Thoughts'u daha erken devreye al
config.policy.entropy_gate_tot = 2.0
config.policy.tot_max_depth = 4

# Vektör belleği disk'e kaydet
config.memory.vector_store_path = "./my_episodic_memory"
config.memory.enable_rag = True

# Konfigürasyonu doğrula
config.validate()

manager = CognitiveManager(model_manager=model, config=config)
```

### Ortam Değişkeni ile Yükleme

```bash
export CM_CRITIC_ENABLED=true
export CM_TOOLS_ENABLE=true
export CM_DEBATE_ENABLED=false
export CM_MAX_NEW_TOKENS_LO=32
export CM_MAX_NEW_TOKENS_HI=512
```

```python
config = CognitiveManagerConfig.from_env()
```

### Oturum Durumuna Erişim

```python
# CognitiveManager, state'i dahili olarak tutar
output = manager.handle("Merhaba!")
output2 = manager.handle("Peki ya İzmir?")  # Önceki bağlamı hatırlar

# Oturum geçmişi
state = manager.state
print(state.turn_count)   # → 2
print(state.session_id)   # → "a3f8b21c"
```

---

## Bağımlılıklar

### Zorunlu

| Paket | Kullanım |
|-------|---------|
| `torch` | Model inference backend |
| `dataclasses` | Tip tanımları |

### Opsiyonel

| Paket | Kullanım | Aktif Eden Config |
|-------|---------|-------------------|
| `chromadb` | Epizodik vektör belleği | `memory.vector_store_provider = "chroma"` |
| `sentence-transformers` | Embedding üretimi | `memory.embedding_provider = "sentence-transformers"` |
| `wikipedia` | Dış gerçek doğrulama | `critic.enable_wikipedia = True` |
| `openai` | OpenAI embedding/fact-check | `memory.embedding_provider = "openai"` |

### Kurulum

```bash
# Temel kurulum
pip install chromadb sentence-transformers

# Wikipedia fact-checking için
pip install wikipedia-api

# Tüm opsiyonel özellikler
pip install chromadb sentence-transformers wikipedia-api
```

---

Daha fazla bilgi için:
- [Mimari Detayları →](architecture/README.md)
- [API Referansı →](api/README.md)
- [Kullanım Kılavuzları →](guides/README.md)
