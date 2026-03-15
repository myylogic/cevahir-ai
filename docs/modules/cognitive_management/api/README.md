# Cognitive Management — API Referansı

**Modül:** `cognitive_management`
**Son Güncelleme:** 2026-03-16

---

## İçindekiler

1. [CognitiveManagerConfig](#cognitivemanagerconfig)
2. [Alt Konfigürasyon Sınıfları](#alt-konfigürasyon-sınıfları)
3. [Veri Tipleri](#veri-tipleri)
4. [CognitiveManager API](#cognitivemanager-api)
5. [Exception Hiyerarşisi](#exception-hiyerarşisi)
6. [Ortam Değişkenleri](#ortam-değişkenleri)

---

## CognitiveManagerConfig

Ana konfigürasyon sınıfı. Tüm alt modüllerin ayarlarını bir arada tutar.

```python
from cognitive_management.config import CognitiveManagerConfig

config = CognitiveManagerConfig()
config.validate()   # Geçerlilik kontrolü
d = config.to_dict()  # Serileştirme
```

### Alanlar

| Alan | Tip | Açıklama |
|------|-----|----------|
| `critic` | `CriticConfig` | Eleştiri ve gerçek doğrulama ayarları |
| `memory` | `MemoryConfig` | Bellek ve RAG ayarları |
| `policy` | `PolicyConfig` | Strateji seçim eşikleri |
| `tools` | `ToolsConfig` | Araç kullanım politikası |
| `decoding_bounds` | `DecodingBounds` | Üretim parametre sınırları |
| `safety` | `SafetyConfig` | Güvenlik ve risk ayarları |
| `features` | `FeatureRules` | Heuristik kural sözlükleri |
| `runtime` | `RuntimeToggles` | Çalışma zamanı açma/kapama anahtarları |
| `default_decoding` | `DecodingConfig` | Varsayılan üretim parametreleri |
| `default_system_prompt` | `str` | Varsayılan sistem davranışı metni |

### Metotlar

```python
# Dışarıdan gelen dict ile alanları güncelle (rekürsif merge)
config.override_with({
    "policy": {"entropy_gate_tot": 2.5},
    "critic": {"strictness": 0.8},
})

# Serileştir
d = config.to_dict()

# Doğrula (geçersiz değerlerde ValueError)
config.validate()

# Ortam değişkenlerinden yükle
config = CognitiveManagerConfig.from_env()
config = CognitiveManagerConfig.from_env(env={"CM_CRITIC_ENABLED": "true"})
```

---

## Alt Konfigürasyon Sınıfları

### CriticConfig

```python
@dataclass
class CriticConfig:
    enabled: bool = True
    checks: tuple[str, ...] = ("task_match", "safety", "claim_density")
    strictness: float = 0.5          # 0.0-1.0 (yüksek = daha katı revizyon)
    max_passes: int = 1              # Maksimum Self-Refine turu

    # Dış gerçek doğrulama
    enable_external_fact_checking: bool = True
    fact_checking_providers: tuple = ("wikipedia",)
    enable_wikipedia: bool = True
    wikipedia_max_results: int = 3

    # Google Fact Check (opsiyonel)
    enable_google_fact_check: bool = False
    google_fact_check_api_key: str | None = None

    # Wolfram Alpha (opsiyonel)
    enable_wolfram: bool = False
    wolfram_api_key: str | None = None

    # LLM tabanlı doğrulama
    enable_llm_fact_verification: bool = True

    # Constitutional AI
    enable_constitutional_ai: bool = True
    constitutional_strictness: float = 0.7   # 0.0-1.0
    custom_principles: tuple[str, ...] = ()  # Ek anayasal ilkeler

    # Eşikler
    fact_checking_score_threshold: float = 0.7
    claim_extraction_min_confidence: float = 0.5
```

**checks parametresi için geçerli değerler:**

| Değer | Açıklama |
|-------|----------|
| `"task_match"` | Sorgu ile yanıt uyumunu kontrol eder |
| `"safety"` | Güvenlik keyword kontrolü |
| `"claim_density"` | İddia yoğunluğu varsa dış doğrulama tetikler |

---

### MemoryConfig

```python
@dataclass
class MemoryConfig:
    session_summary_every: int = 6       # Kaç turda bir özet üretilir
    salient_topk: int = 8                # Özetde tutulacak önemli mesaj sayısı
    max_history_tokens: int = 3072       # Oturum geçmişi token sınırı
    enable_session_summary: bool = True
    enable_salient_pruning: bool = True

    # Vektör bellek ve RAG
    enable_vector_memory: bool = True
    embedding_provider: str = "sentence-transformers"  # "sentence-transformers" | "openai" | "none"
    embedding_model: str | None = None   # None = all-MiniLM-L6-v2
    openai_api_key: str | None = None

    vector_store_provider: str = "chroma"  # "chroma" | "pinecone" | "weaviate" | "qdrant" | "milvus" | "memory"
    vector_store_path: str | None = "./memory/episodic_store"
    vector_store_collection_name: str = "cognitive_memory"

    # Pinecone (opsiyonel)
    pinecone_api_key: str | None = None
    pinecone_environment: str | None = None
    pinecone_index_name: str | None = None

    # RAG
    enable_rag: bool = True
    rag_top_k: int = 3                   # Kaç bellek öğesi çekilir
    rag_score_threshold: float = 0.7     # Minimum cosine benzerlik skoru

    # Vektör arama
    vector_search_top_k: int = 5
    hybrid_search_alpha: float = 0.7     # 0.0 = sadece keyword, 1.0 = sadece vektör
```

**vector_store_path için özel değerler:**

| Değer | Davranış |
|-------|----------|
| `"./memory/episodic_store"` | Disk'e kalıcı (oturumlar arası hatırlar) |
| `None` | In-memory (oturum kapanınca kaybolur) |

---

### PolicyConfig

```python
@dataclass
class PolicyConfig:
    debate_enabled: bool = True

    # Entropi eşikleri (0-3 normalize ölçek)
    entropy_gate_think: float = 1.5   # Bu eşiğin üstü CoT
    entropy_gate_debate: float = 2.5  # Bu eşiğin üstü debate/self_consistency
    entropy_gate_tot: float = 3.0     # Bu eşiğin üstü ToT

    # Uzunluk eşikleri (token)
    length_gate_debate: int = 200
    length_gate_tot: int = 300

    allow_inner_steps: bool = True

    # Tree of Thoughts
    tot_enabled: bool = True
    tot_max_depth: int = 3
    tot_branching_factor: int = 3
    tot_top_k: int = 5

    # Self-Consistency (Wang et al. 2022)
    self_consistency_enabled: bool = True
    self_consistency_n: int = 3           # Örneklem sayısı (3-5 önerilir)
    self_consistency_method: str = "hybrid"  # "majority" | "score" | "hybrid"
```

**self_consistency_method seçenekleri:**

| Değer | Açıklama |
|-------|----------|
| `"majority"` | Bigram örtüşmesine göre çoğunluk oylaması |
| `"score"` | Model kalite puanına göre seçim |
| `"hybrid"` | Örtüşme + puan birleşimi (önerilen) |

---

### ToolsConfig

```python
@dataclass
class ToolsConfig:
    allow: tuple[str, ...] = ("calculator", "search", "file")
    policy: str = "heuristic"   # "heuristic" | gelecekte "learned"
    enable_tools: bool = True
```

---

### DecodingBounds

```python
@dataclass
class DecodingBounds:
    max_new_tokens_bounds: tuple[int, int] = (32, 512)
    temperature_bounds: tuple[float, float] = (0.40, 0.90)
    top_p_default: float = 0.90
    repetition_penalty_default: float = 1.15

    # Domain bazlı sıcaklık
    math_temperature: float = 0.45     # Deterministik çıktı
    creative_temperature: float = 0.85 # Çeşitlilik
    code_temperature: float = 0.50     # Tutarlı kod
```

---

### SafetyConfig

```python
@dataclass
class SafetyConfig:
    risk_keywords_sensitive: tuple[str, ...] = ("siyaset", "tıbbi", "hukuki")
    claim_markers: tuple[str, ...] = ("istatistik", "oran", "kanıt", "%")
    raise_on_high_risk: bool = False  # True ise yüksek riskte critic zorunlu
    high_risk_threshold: float = 0.6
```

---

### RuntimeToggles

```python
@dataclass
class RuntimeToggles:
    enable_logging: bool = True
    enable_telemetry: bool = True
    enable_internal_thought_logging: bool = False  # Gizlilik: iç sesler loglanmasın
    fail_fast: bool = False   # True ise hatada hızla istisna at

    # AIOps
    aiops_sensitivity: str = "medium"     # "low" | "medium" | "high"
    enable_auto_anomaly_check: bool = True

    # Performans optimizasyonu
    enable_profiling: bool = False
    enable_semantic_cache: bool = False
    semantic_cache_threshold: float = 0.85
    semantic_cache_max_size: int = 1000
    enable_cache_warming: bool = False

    # Gelişmiş özellikler
    enable_connection_pool: bool = False
    connection_pool_size: int = 10
    enable_batch_processing: bool = False
    batch_size: int = 10
    batch_timeout: float = 1.0
```

---

### DecodingConfig

```python
@dataclass
class DecodingConfig:
    max_new_tokens: int = 256
    min_new_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int | None = 0       # 0 = devre dışı
    repetition_penalty: float = 1.1
```

---

## Veri Tipleri

### Literal Tipler

```python
# Bilişsel mod
Mode = Literal["direct", "think1", "debate2", "tot", "react", "self_consistency"]

# Araç kullanım kararı
ToolDecision = Literal["none", "maybe", "must"]

# Sorgu tipi
QueryType = Literal[
    "factual",        # "Türkiye'nin başkenti nedir?"
    "reasoning",      # "Neden...?", "Nasıl etkiler...?"
    "creative",       # Hikaye, şiir, senaryo
    "conversational", # Selamlama, genel sohbet
    "math",           # Sayısal hesaplama, denklem
    "code",           # Kod yazma, debug
    "unknown",
]

# Alan tipi
DomainType = Literal[
    "math", "science", "law", "medical",
    "technology", "history", "creative", "general",
]
```

### PolicyOutput

```python
@dataclass
class PolicyOutput:
    mode: Mode                    # Seçilen bilişsel mod
    tool: ToolDecision            # Araç kullanım önerisi
    decoding: DecodingConfig      # Üretim parametreleri
    inner_steps: int = 0          # Düşünce adımı sayısı
    query_type: QueryType | None = None
    domain: DomainType | None = None
    complexity: float = 0.0       # 0.0-1.0 karmaşıklık skoru
```

### ThoughtCandidate

```python
@dataclass
class ThoughtCandidate:
    text: str          # Düşünce metni
    score: float       # Değerlendirme skoru (yüksek = daha iyi)
    depth: int = 0     # ToT ağaç derinliği (kök = 0)
    path: list[str]    # Bu noktaya gelen önceki düşünceler
```

---

## CognitiveManager API

`CognitiveManager` modülün dış dünyaya açılan ana arayüzüdür.

### Başlatma

```python
from cognitive_management.cognitive_manager import CognitiveManager
from cognitive_management.config import CognitiveManagerConfig

manager = CognitiveManager(
    model_manager=model,    # ModelManager / CevahirModel instance
    config=CognitiveManagerConfig(),
)
```

### handle()

```python
def handle(
    self,
    user_message: str,
    system_prompt: str | None = None,
    metadata: dict | None = None,
    decoding: DecodingConfig | None = None,
) -> CognitiveOutput:
```

**Parametreler:**

| Parametre | Tip | Açıklama |
|-----------|-----|----------|
| `user_message` | `str` | Kullanıcı mesajı (zorunlu) |
| `system_prompt` | `str \| None` | Sistem davranış yönergesi (None = config.default_system_prompt) |
| `metadata` | `dict \| None` | Ek sinyaller, önceden hesaplanmış query_type vb. |
| `decoding` | `DecodingConfig \| None` | Üretim parametrelerini manuel geçersiz kıl |

**Dönüş Değeri:** `CognitiveOutput`

```python
@dataclass
class CognitiveOutput:
    text: str                              # Üretilen yanıt
    used_mode: Mode                        # Kullanılan bilişsel mod
    tool_used: str | None                  # Kullanılan araç (None = araç yok)
    revised_by_critic: bool                # Critic revizyonu yapıldı mı?
    reasoning_chain: list[ReasoningTrace]  # Akıl yürütme adımları
    critic_passes: int                     # Self-Refine tur sayısı
    critic_feedback: list[CriticFeedback] | None
    memory_hits: int                       # RAG'dan çekilen bellek öğesi sayısı
    latency_ms: float                      # Toplam süre (ms)
    query_type: QueryType | None
    domain: DomainType | None
    self_consistency_result: SelfConsistencyResult | None
    context_sources: list[str]             # "vector", "history", "tool" vb.
    metadata: dict                         # İzleme verileri
```

### Oturum Durumu

```python
# Aktif oturum durumuna eriş
state: CognitiveState = manager.state

# Oturumu sıfırla
manager.reset_session()

# Oturum ID
print(state.session_id)   # "a3f8b21c"
print(state.turn_count)   # 5
```

### reset_session()

```python
def reset_session(self) -> None:
    """Oturum geçmişini ve durumu sıfırla. Vektör belleği korunur."""
```

---

## Exception Hiyerarşisi

```
CognitiveError (base)
    |
    +-- CognitiveConfigError         # Geçersiz config değeri
    |
    +-- CognitiveInputError          # Geçersiz veya boş kullanıcı girdisi
    |
    +-- CognitiveBackendError        # Model backend erişim hatası
    |       |
    |       +-- CognitiveTimeoutError   # Backend zaman aşımı
    |
    +-- CognitiveMemoryError         # Vektör store erişim hatası
    |
    +-- CognitiveToolError           # Araç çalıştırma hatası
    |
    +-- CognitiveCriticError         # Critic değerlendirme hatası
    |
    +-- CognitivePipelineError       # Pipeline işleme hatası
```

### Kullanım

```python
from cognitive_management.exceptions import (
    CognitiveError,
    CognitiveBackendError,
    CognitiveInputError,
)

try:
    output = manager.handle(user_message)
except CognitiveInputError as e:
    print(f"Gecersiz giris: {e}")
except CognitiveBackendError as e:
    print(f"Model hatasi: {e}")
except CognitiveError as e:
    print(f"Genel cognitive hata: {e}")
```

---

## Ortam Değişkenleri

`CognitiveManagerConfig.from_env()` şu ortam değişkenlerini okur:

| Değişken | Tip | Karşılık | Örnek |
|----------|-----|----------|-------|
| `CM_CRITIC_ENABLED` | bool | `critic.enabled` | `true` |
| `CM_TOOLS_ENABLE` | bool | `tools.enable_tools` | `false` |
| `CM_DEBATE_ENABLED` | bool | `policy.debate_enabled` | `true` |
| `CM_MAX_NEW_TOKENS_LO` | int | `decoding_bounds.max_new_tokens_bounds[0]` | `32` |
| `CM_MAX_NEW_TOKENS_HI` | int | `decoding_bounds.max_new_tokens_bounds[1]` | `512` |
| `CM_TEMP_LO` | float | `decoding_bounds.temperature_bounds[0]` | `0.4` |
| `CM_TEMP_HI` | float | `decoding_bounds.temperature_bounds[1]` | `0.9` |

**bool için geçerli değerler:** `1`, `true`, `yes`, `on` → `True`; `0`, `false`, `no`, `off` → `False`

---

## DEFAULT_RULES Sözlüğü

Araç politikası ve mod seçimi için hazır kural seti:

```python
from cognitive_management.config import DEFAULT_RULES

# Yapısı:
DEFAULT_RULES = {
    "tool_policy": {
        "priority": ["search", "calculator", "file"],
        "force_search_if_recent": True,
        "force_calc_if_numeric": True,
    },
    "mode_selection": {
        "prefer_direct_for_short_inputs": True,
        "debate_on_long_or_uncertain": True,
    },
    "critic_policy": {
        "require_on_high_risk": True,
        "light_on_low_risk": True,
        "revise_once_max": True,
    },
}
```

---

## FeatureRules (Heuristik Kural Sözlükleri)

```python
@dataclass
class FeatureRules:
    tool_rules: dict = {
        "needs_recent_info_triggers": ("bugün", "güncel", "son haber", "en yeni"),
        "needs_calc_or_parse_triggers": ("hesapla", "toplam", "adet", "oran"),
        "default_decision": "none",  # "none" | "maybe" | "must"
    }
    risk_rules: dict = {
        "base_risk": 0.0,
        "claim_bonus": 0.5,      # İddia algılanırsa risk_score artışı
        "sensitive_bonus": 0.5,  # Hassas alan algılanırsa risk_score artışı
        "max_risk": 1.0,
    }
```
